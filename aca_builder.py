# aca_builder.py
from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
from datetime import timedelta
import pandas as pd

from debug_logging import get_logger, log_df, log_time, log_call
log = get_logger("builder")

from aca_processing import (
    MONTHS, FULL_MONTHS,
    _collect_employee_ids, month_bounds,
    _any_overlap, _all_month,
    _status_from_demographic, _waiting_in_month,
)

__all__ = ["build_interim","build_final","build_penalty_dashboard"]

AFFORDABILITY_THRESHOLD_DEFAULT = 50.0

def _df_or_empty(df: pd.DataFrame | None) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) and not df.empty else pd.DataFrame()

def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    lower_map = {c.lower(): c for c in df.columns}
    cols_lower = set(lower_map.keys())
    if "eligibleplan" in cols_lower and "plancode" not in cols_lower:
        df["plancode"] = df[lower_map["eligibleplan"]].astype(str).str.strip()
    if "eligibletier" in cols_lower and "eligibilitytier" not in cols_lower:
        df["eligibilitytier"] = df[lower_map["eligibletier"]].astype(str).str.strip()
    if "enrollmenttier" not in cols_lower and "tier" in cols_lower:
        df["enrollmenttier"] = df[lower_map["tier"]].astype(str).str.strip()
    return df

def _latest_emp_cost_for_month(el_df: pd.DataFrame, ms, me) -> Optional[float]:
    if el_df is None or el_df.empty:
        return None
    df = el_df.copy()
    for c in ("eligibilitystartdate", "eligibilityenddate"):
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    mask = (df["eligibilityenddate"].fillna(pd.Timestamp.max).dt.date >= ms) & \
           (df["eligibilitystartdate"].fillna(pd.Timestamp.min).dt.date <= me)
    df = df[mask]
    if df.empty or "eligibilitytier" not in df.columns:
        return None
    tier_u = df["eligibilitytier"].astype(str).str.upper().str.strip()
    df = df[tier_u.eq("EMP")]
    if df.empty:
        return None
    df = df.sort_values("eligibilityenddate", ascending=False)
    v = pd.to_numeric(df.iloc[0].get("plancost"), errors="coerce")
    return float(v) if pd.notna(v) else None

def _offered_allmonth(el_emp: pd.DataFrame, ms, me) -> bool:
    if el_emp.empty or "eligibilitytier" not in el_emp.columns:
        return False
    tiers = el_emp["eligibilitytier"].astype(str).str.upper().str.strip()
    mask = tiers.str.contains("EMP", na=False)
    return _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=mask)

def _tier_offered_any(df: pd.DataFrame, tier_col: str, tokens: Tuple[str, ...],
                      start_col: str, end_col: str, ms, me, *, require_enrolled: bool = False) -> bool:
    if df is None or df.empty:
        return False
    tiers = df[tier_col].astype(str).str.upper().str.strip()
    mask = pd.Series(True, index=df.index)
    if require_enrolled and "isenrolled" in df.columns:
        mask &= df["isenrolled"].astype(bool)
        for col in ("plancode", "planname"):
            if col in df.columns:
                s = df[col].astype(str).str.upper().str.strip()
                mask &= ~s.eq("WAIVE")
    tok_mask = pd.Series(False, index=df.index)
    for t in tokens:
        tok_mask |= tiers.str.contains(t, na=False)
    mask &= tok_mask
    return _all_month(df, start_col, end_col, ms, me, mask=mask)

def _enrolled_full_month_union(en_df: pd.DataFrame, ms, me) -> bool:
    if en_df is None or en_df.empty:
        return False
    mask = pd.Series(True, index=en_df.index)
    if "isenrolled" in en_df.columns:
        mask &= en_df["isenrolled"].astype(bool)
    waive_mask = pd.Series(False, index=en_df.index)
    for col in ("plancode","planname"):
        if col in en_df.columns:
            s = en_df[col].astype(str).str.upper().str.strip()
            waive_mask |= s.eq("WAIVE")
    mask &= ~waive_mask
    df = en_df.loc[mask].copy()
    if df.empty:
        return False
    for c in ("enrollmentstartdate","enrollmentenddate"):
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")

    intervals = []
    for _, r in df.iterrows():
        s = pd.to_datetime(r.get("enrollmentstartdate"), errors="coerce")
        e = pd.to_datetime(r.get("enrollmentenddate"), errors="coerce")
        if pd.isna(s) and pd.isna(e):
            continue
        s = (s or pd.Timestamp.min).date()
        e = (e or pd.Timestamp.max).date()
        intervals.append((s, e))
    if not intervals:
        return False
    intervals.sort()
    merged = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= (cur_e + timedelta(days=1)):
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    for s, e in merged:
        if s <= ms and e >= me:
            return True
    return False

def _month_line14(eligible_mv: bool, offer_ee_allmonth: bool, offer_spouse: bool,
                  offer_dependents: bool, affordable: bool) -> str:
    if not offer_ee_allmonth: return "1H"
    if eligible_mv:
        if offer_spouse and offer_dependents: return "1A" if affordable else "1E"
        if offer_spouse and not offer_dependents: return "1D"
        if offer_dependents and not offer_spouse: return "1C"
        return "1B"
    return "1F"

def _month_line16(*, employed: bool, enrolled_full: bool, waiting: bool, ft: bool,
                  offer_ee_allmonth: bool, affordable: bool) -> str:
    if enrolled_full: return "2C"
    if not employed:  return "2A"
    if waiting:       return "2D"
    if not ft:        return "2B"
    if offer_ee_allmonth and affordable: return "2H"
    return ""

@log_call(log)
def build_interim(
    emp_demo: pd.DataFrame,
    emp_elig: pd.DataFrame,
    emp_enroll: pd.DataFrame,
    dep_enroll: pd.DataFrame,
    year: int,
    *,
    emp_wait: pd.DataFrame | None = None,   # Emp Wait Period
    affordability_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Rules engine. waitingperiod_month comes ONLY from Emp Wait Period."""
    with log_time(log, "build_interim"):
        emp_demo   = _apply_aliases(_df_or_empty(emp_demo))
        emp_elig   = _apply_aliases(_df_or_empty(emp_elig))
        emp_enroll = _apply_aliases(_df_or_empty(emp_enroll))
        dep_enroll = _apply_aliases(_df_or_empty(dep_enroll))
        emp_wait   = _df_or_empty(emp_wait)

        for df, sc, ec in (
            (emp_elig, "eligibilitystartdate", "eligibilityenddate"),
            (emp_enroll, "enrollmentstartdate", "enrollmentenddate"),
            (dep_enroll, "enrollmentstartdate", "enrollmentenddate"),
        ):
            if not df.empty:
                for c in (sc, ec):
                    if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
                        df[c] = pd.to_datetime(df[c], errors="coerce")

        st = _status_from_demographic(emp_demo)
        employee_ids = _collect_employee_ids(emp_demo, st, emp_elig, emp_enroll, dep_enroll, emp_wait)
        rows: List[Dict[str, Any]] = []
        thresh = AFFORDABILITY_THRESHOLD_DEFAULT if affordability_threshold is None else float(affordability_threshold)

        def _is_employed_month(st_emp: pd.DataFrame, ms, me) -> bool:
            if st_emp is None or st_emp.empty:
                return False
            for c in ("statusstartdate", "statusenddate"):
                if c in st_emp.columns and not pd.api.types.is_datetime64_any_dtype(st_emp[c]):
                    st_emp[c] = pd.to_datetime(st_emp[c], errors="coerce")
            overlaps = (st_emp["statusenddate"].fillna(pd.Timestamp.max).dt.date >= ms) & \
                       (st_emp["statusstartdate"].fillna(pd.Timestamp.min).dt.date <= me)
            if not overlaps.any():
                return False
            if "_estatus_norm" in st_emp.columns:
                s = st_emp.loc[overlaps, "_estatus_norm"].astype(str)
                any_term = s.str.contains("TERMINAT", na=False) | s.str.fullmatch("TERM", na=False)
                return not bool(any_term.any())
            return True

        def _is_ft(st_emp: pd.DataFrame, ms, me) -> bool:
            if st_emp.empty or "_role_norm" not in st_emp.columns:
                return False
            if "_estatus_norm" in st_emp.columns:
                overlaps = (st_emp["statusenddate"].fillna(pd.Timestamp.max).dt.date >= ms) & \
                           (st_emp["statusstartdate"].fillna(pd.Timestamp.min).dt.date <= me)
                if overlaps.any():
                    s = st_emp.loc[overlaps, "_estatus_norm"].astype(str)
                    any_term = s.str.contains("TERMINAT", na=False) | s.str.fullmatch("TERM", na=False)
                    if any_term.any():
                        return False
            s = st_emp["_role_norm"].astype(str)
            mask = s.str.contains("FULLTIME", na=False) | s.str.fullmatch("FT", na=False)
            return _all_month(st_emp, "statusstartdate", "statusenddate", ms, me, mask=mask)

        def _is_pt(st_emp: pd.DataFrame, ms, me) -> bool:
            if st_emp.empty or "_role_norm" not in st_emp.columns:
                return False
            if "_estatus_norm" in st_emp.columns:
                overlaps = (st_emp["statusenddate"].fillna(pd.Timestamp.max).dt.date >= ms) & \
                           (st_emp["statusstartdate"].fillna(pd.Timestamp.min).dt.date <= me)
                if overlaps.any():
                    s = st_emp.loc[overlaps, "_estatus_norm"].astype(str)
                    any_term = s.str.contains("TERMINAT", na=False) | s.str.fullmatch("TERM", na=False)
                    if any_term.any():
                        return False
            s = st_emp["_role_norm"].astype(str)
            mask = s.str.contains("PARTTIME", na=False) | s.str.fullmatch("PT", na=False)
            return _all_month(st_emp, "statusstartdate", "statusenddate", ms, me, mask=mask)

        for emp in employee_ids:
            el_emp = emp_elig[emp_elig["employeeid"].astype(str) == str(emp)].copy() if not emp_elig.empty else pd.DataFrame()
            en_emp = emp_enroll[emp_enroll["employeeid"].astype(str) == str(emp)].copy() if not emp_enroll.empty else pd.DataFrame()
            st_emp = st[st["employeeid"].astype(str) == str(emp)].copy() if not st.empty else pd.DataFrame()
            wt_emp = emp_wait[emp_wait["employeeid"].astype(str) == str(emp)].copy() if not emp_wait.empty else pd.DataFrame()

            for m in range(1, 13):
                ms, me = month_bounds(year, m)
                employed = _is_employed_month(st_emp, ms, me)
                ft = _is_ft(st_emp, ms, me)
                parttime = (not ft) and _is_pt(st_emp, ms, me)

                elig_any  = _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me) if not el_emp.empty else False
                elig_full = _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me) if not el_emp.empty else False

                eligible_mv = False
                if not el_emp.empty and "plancode" in el_emp.columns:
                    mask_mv = el_emp["plancode"].astype(str).str.upper().str.strip().eq("PLANA")
                    eligible_mv = _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=mask_mv)

                offer_ee_allmonth = _offered_allmonth(el_emp, ms, me)

                offer_spouse = _tier_offered_any(el_emp, "eligibilitytier", ("EMPFAM","EMPSPOUSE"), "eligibilitystartdate", "eligibilityenddate", ms, me) \
                               or _tier_offered_any(en_emp, "enrollmenttier", ("EMPFAM","EMPSPOUSE"), "enrollmentstartdate", "enrollmentenddate", ms, me, require_enrolled=True)
                offer_dependents = _tier_offered_any(el_emp, "eligibilitytier", ("EMPFAM","EMPCHILD"), "eligibilitystartdate", "eligibilityenddate", ms, me) \
                                   or _tier_offered_any(en_emp, "enrollmenttier", ("EMPFAM","EMPCHILD"), "enrollmentstartdate", "enrollmentenddate", ms, me, require_enrolled=True)

                enrolled_full = _enrolled_full_month_union(en_emp, ms, me)

                emp_cost = _latest_emp_cost_for_month(el_emp, ms, me)
                affordable = (emp_cost is not None) and (emp_cost < (AFFORDABILITY_THRESHOLD_DEFAULT if affordability_threshold is None else float(affordability_threshold)))

                waiting = _waiting_in_month(wt_emp, ms, me)  # ONLY from Emp Wait Period

                l14 = _month_line14(eligible_mv, offer_ee_allmonth, offer_spouse, offer_dependents, affordable)
                if (not bool(elig_any)) and bool(enrolled_full):
                    l14 = "1E"
                l16 = _month_line16(
                    employed=bool(employed),
                    enrolled_full=bool(enrolled_full),
                    waiting=bool(waiting),
                    ft=bool(ft),
                    offer_ee_allmonth=bool(offer_ee_allmonth),
                    affordable=bool(affordable),
                )

                rows.append({
                    "EmployeeID": str(emp),
                    "Year": int(year),
                    "MonthNum": int(m),
                    "Month": MONTHS[m-1],
                    "MonthStart": ms,
                    "MonthEnd": me,
                    "employed": bool(employed),
                    "ft": bool(ft),
                    "parttime": bool(parttime),
                    "eligibleforcoverage": bool(elig_any),
                    "eligible_allmonth": bool(elig_full),
                    "eligible_mv": bool(eligible_mv),
                    "offer_ee_allmonth": bool(offer_ee_allmonth),
                    "enrolled_allmonth": bool(enrolled_full),
                    "offer_spouse": bool(offer_spouse),
                    "offer_dependents": bool(offer_dependents),
                    "spouse_eligible": bool(offer_spouse),
                    "child_eligible": bool(offer_dependents),
                    "spouse_enrolled": False,
                    "child_enrolled": False,
                    "waitingperiod_month": bool(waiting),
                    "affordable_plan": bool(affordable),
                    "line14_final": l14,
                    "line16_final": l16,
                    "line14_all12": "",
                })

        interim = pd.DataFrame.from_records(rows)

        # Code 1G handling
        one_g_emp_ids = []
        if not interim.empty:
            for emp in interim["EmployeeID"].unique():
                it = interim[interim["EmployeeID"] == emp]
                any_ft = bool(it["ft"].any())
                any_enrolled_full = bool(it["enrolled_allmonth"].any())
                if (not any_ft) and any_enrolled_full:
                    one_g_emp_ids.append(emp)
            if one_g_emp_ids:
                interim.loc[interim["EmployeeID"].isin(one_g_emp_ids), "line14_all12"] = "1G"
                interim.loc[interim["EmployeeID"].isin(one_g_emp_ids), "line14_final"] = ""

        order = [
            "EmployeeID","Year","MonthNum","Month","MonthStart","MonthEnd",
            "employed","ft","parttime",
            "eligibleforcoverage","eligible_allmonth","eligible_mv",
            "offer_ee_allmonth","enrolled_allmonth",
            "offer_spouse","offer_dependents",
            "spouse_eligible","child_eligible","spouse_enrolled","child_enrolled",
            "waitingperiod_month","affordable_plan",
            "line14_final","line16_final","line14_all12",
        ]
        cols = [c for c in order if c in interim.columns] + [c for c in interim.columns if c not in order]

        log_df(log, interim, "interim")
        log.info("interim summary", extra={"extra_data": {
            "employees": int(interim["EmployeeID"].nunique()) if not interim.empty else 0,
            "rows": int(interim.shape[0]) if hasattr(interim, "shape") else 0
        }})

        return interim.loc[:, cols]

def build_final(interim: pd.DataFrame) -> pd.DataFrame:
    if interim is None or interim.empty:
        return pd.DataFrame(columns=["EmployeeID", "Month", "Line14_Final", "Line16_Final"])
    df = interim.copy().sort_values(["EmployeeID", "MonthNum"])
    out = []
    for emp, g in df.groupby("EmployeeID", sort=False):
        l14_all12 = (g["line14_all12"].astype(str).str.strip().iloc[0] or "")
        if l14_all12:
            out.append({"EmployeeID": emp, "Month": "All 12 months", "Line14_Final": l14_all12, "Line16_Final": ""})
        else:
            for _, r in g.iterrows():
                out.append({"EmployeeID": emp, "Month": r["Month"], "Line14_Final": r["line14_final"], "Line16_Final": r["line16_final"]})
    return pd.DataFrame.from_records(out)

def build_penalty_dashboard(interim: pd.DataFrame) -> pd.DataFrame:
    if interim is None or interim.empty:
        return pd.DataFrame(columns=["EmployeeID", "Reason"] + FULL_MONTHS)
    def month_reason(row) -> str:
        if not row["offer_ee_allmonth"]:
            if not row["employed"]: return "Not employed"
            if row["waitingperiod_month"]: return "Waiting period"
            if not row["eligibleforcoverage"]: return "Not eligible"
            return "No MEC offer"
        if (not row["enrolled_allmonth"]) and (not row["affordable_plan"]): return "Offered but not affordable (B)"
        return "–"
    df = interim.copy()
    out_rows = []
    for emp, g in df.groupby("EmployeeID", sort=False):
        rec = {"EmployeeID": emp}
        reasons = [month_reason(r) for _, r in g.sort_values("MonthNum").iterrows()]
        for i, m in enumerate(FULL_MONTHS):
            rec[m] = reasons[i] if i < len(reasons) else "–"
        rec["Reason"] = next((x for x in reasons if x != "–"), "–")
        out_rows.append(rec)
    cols = ["EmployeeID", "Reason"] + FULL_MONTHS
    return pd.DataFrame.from_records(out_rows, columns=cols)
