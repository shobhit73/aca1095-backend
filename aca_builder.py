# aca_builder.py
# Build a rich Interim grid (employee x month) + Final + Penalty dashboard

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
from datetime import timedelta

import pandas as pd

from aca_processing import (
    MONTHS,
    FULL_MONTHS,
    _collect_employee_ids, month_bounds,
    _any_overlap, _all_month,
    _status_from_demographic,
)

AFFORDABILITY_THRESHOLD = 50.0  # dollars (employee-only)
PENALTY_A_MONTHLY = 241.67
PENALTY_B_MONTHLY = 362.50

def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()

    # Case-insensitive map
    lower_map = {c.lower(): c for c in df.columns}
    cols_lower = set(lower_map.keys())

    # EligiblePlan -> plancode
    if "eligibleplan" in cols_lower and "plancode" not in cols_lower:
        df["plancode"] = df[lower_map["eligibleplan"]].astype(str).str.strip()

    # EligibleTier -> eligibilitytier
    if "eligibletier" in cols_lower and "eligibilitytier" not in cols_lower:
        df["eligibilitytier"] = df[lower_map["eligibletier"]].astype(str).str.strip()

    # Tier (Enrollment) -> enrollmenttier
    if "enrollmenttier" not in cols_lower and "tier" in cols_lower:
        df["enrollmenttier"] = df[lower_map["tier"]].astype(str).str.strip()

    return df

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _latest_emp_cost_for_month(el_df: pd.DataFrame, ms, me) -> Optional[float]:
    """
    Return the employee-only (EMP) plan cost from Emp Eligibility that overlaps the month,
    taking the row with the latest end-date.
    """
    if el_df is None or el_df.empty:
        return None
    df = el_df.copy()
    for c in ("eligibilitystartdate","eligibilityenddate"):
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # overlapping the month and EMP tier
    mask = (
        df["eligibilityenddate"].fillna(pd.Timestamp.max).dt.date >= ms
    ) & (
        df["eligibilitystartdate"].fillna(pd.Timestamp.min).dt.date <= me
    )
    df = df[mask]
    if df.empty or "eligibilitytier" not in df.columns:
        return None

    tier_u = df["eligibilitytier"].astype(str).str.upper().str.strip()
    df = df[tier_u.eq("EMP")]
    if df.empty:
        return None

    df = df.sort_values("eligibilityenddate", ascending=False)
    v = pd.to_numeric(df.iloc[0]["plancost"], errors="coerce")
    return float(v) if not pd.isna(v) else None

def _offered_allmonth(el_emp: pd.DataFrame, ms, me) -> bool:
    """Employee-level MEC offer for full month (any EMP* tier)."""
    if el_emp.empty or "eligibilitytier" not in el_emp.columns:
        return False
    tiers = el_emp["eligibilitytier"].astype(str).str.upper().str.strip()
    mask = tiers.str.contains("EMP", na=False)
    return _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=mask)

def _tier_offered_any(
    df: pd.DataFrame,
    tier_col: str,
    tokens: Tuple[str, ...],
    start_col: str,
    end_col: str,
    ms, me,
    *,
    require_enrolled: bool = False,
) -> bool:
    """
    True if ANY row for the given tokens overlaps the full month.

    If require_enrolled=True, filters to rows with isenrolled==True (excludes WAIVE plan rows).
    """
    if df is None or df.empty: return False
    tiers = df[tier_col].astype(str).str.upper().str.strip()
    mask = pd.Series(True, index=df.index)

    if require_enrolled and "isenrolled" in df.columns:
        mask &= df["isenrolled"].astype(bool)
        # exclude 'WAIVE' if present in plan code/name
        for col in ("plancode","planname"):
            if col in df.columns:
                s = df[col].astype(str).str.upper().str.strip()
                mask &= ~s.eq("WAIVE")

    tok_mask = pd.Series(False, index=df.index)
    for t in tokens:
        tok_mask |= tiers.str.contains(t, na=False)
    mask &= tok_mask

    return _all_month(df, start_col, end_col, ms, me, mask=mask)

def _enrolled_full_month_union(en_df: pd.DataFrame, ms, me) -> bool:
    """
    TRUE if, after filtering to enrolled (non-WAIVE) rows,
    the UNION of enrollment intervals covers the entire month [ms, me].
    - Honors isenrolled==True if present
    - Excludes WAIVE rows
    - Allows multiple rows stitched together (including adjacent days)
    """
    if en_df is None or en_df.empty:
        return False

    mask = pd.Series(True, index=en_df.index)
    if "isenrolled" in en_df.columns:
        mask &= en_df["isenrolled"].astype(bool)

    waive_mask = pd.Series(False, index=en_df.index)
    for col in ("plancode", "planname"):
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

    # Build minimal interval cover
    intervals = []
    for _, r in df.iterrows():
        s = pd.to_datetime(r.get("enrollmentstartdate"), errors="coerce")
        e = pd.to_datetime(r.get("enrollmentenddate"), errors="coerce")
        if pd.isna(s) and pd.isna(e):
            continue
        s = (s or pd.Timestamp.min).date()
        e = (e or pd.Timestamp.max).date()
        intervals.append((s,e))

    if not intervals:
        return False

    # merge intervals
    intervals.sort()
    merged = []
    cur_s, cur_e = intervals[0]
    for s,e in intervals[1:]:
        if s <= (cur_e + timedelta(days=1)):
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # does merged cover entire [ms,me] ?
    for s,e in merged:
        if s <= ms and e >= me:
            return True
    return False

def _is_employed_month(st_emp: pd.DataFrame, ms, me) -> bool:
    """
    Employed if there exists any status range that overlaps [ms, me]
    AND not terminated within that month.
    """
    if st_emp is None or st_emp.empty:
        return False
    for c in ("statusstartdate","statusenddate"):
        if c in st_emp.columns and not pd.api.types.is_datetime64_any_dtype(st_emp[c]):
            st_emp[c] = pd.to_datetime(st_emp[c], errors="coerce")

    overlaps = (
        st_emp["statusenddate"].fillna(pd.Timestamp.max).dt.date >= ms
    ) & (
        st_emp["statusstartdate"].fillna(pd.Timestamp.min).dt.date <= me
    )
    if not overlaps.any():
        return False

    # any terminating status overlapping month => not employed
    s = st_emp.loc[overlaps, "_estatus_norm"].astype(str) if "_estatus_norm" in st_emp.columns else pd.Series("", index=st_emp.index)
    any_term = s.str.contains("TERMINAT", na=False) | s.str.fullmatch("TERM", na=False)
    return not bool(any_term.any())

def _is_ft(st_emp: pd.DataFrame, ms, me) -> bool:
    """
    Full-time only if:
      - there is NO 'Terminated' status overlapping the month, and
      - Role shows FT (or FULLTIME) covering the ENTIRE month.
    """
    if st_emp.empty or "_role_norm" not in st_emp.columns:
        return False

    if "_estatus_norm" in st_emp.columns:
        overlaps = (
            st_emp["statusenddate"].fillna(pd.Timestamp.max).dt.date >= ms
        ) & (
            st_emp["statusstartdate"].fillna(pd.Timestamp.min).dt.date <= me
        )
        if overlaps.any():
            s = st_emp.loc[overlaps, "_estatus_norm"].astype(str)
            any_term = s.str.contains("TERMINAT", na=False) | s.str.fullmatch("TERM", na=False)
            if any_term.any():
                return False

    s = st_emp["_role_norm"].astype(str)
    mask = s.str.contains("FULLTIME", na=False) | s.str.fullmatch("FT", na=False)
    return _all_month(st_emp, "statusstartdate", "statusenddate", ms, me, mask=mask)

def _is_pt(st_emp: pd.DataFrame, ms, me) -> bool:
    """
    Part-time only if:
      - there is NO 'Terminated' status overlapping the month, and
      - Role shows PT (or PARTTIME) covering the ENTIRE month, and it's not FT.
    """
    if st_emp.empty or "_role_norm" not in st_emp.columns:
        return False

    if "_estatus_norm" in st_emp.columns:
        overlaps = (
            st_emp["statusenddate"].fillna(pd.Timestamp.max).dt.date >= ms
        ) & (
            st_emp["statusstartdate"].fillna(pd.Timestamp.min).dt.date <= me
        )
        if overlaps.any():
            s = st_emp.loc[overlaps, "_estatus_norm"].astype(str)
            any_term = s.str.contains("TERMINAT", na=False) | s.str.fullmatch("TERM", na=False)
            if any_term.any():
                return False

    s = st_emp["_role_norm"].astype(str)
    mask = s.str.contains("PARTTIME", na=False) | s.str.fullmatch("PT", na=False)
    return _all_month(st_emp, "statusstartdate", "statusenddate", ms, me, mask=mask)

# ------------------------------------------------------------
# Line 14/16 (rules)
# ------------------------------------------------------------
def _month_line14(eligible_mv: bool, offer_ee_allmonth: bool, offer_spouse: bool,
                  offer_dependents: bool, affordable: bool) -> str:
    """
    Correct mapping:
      - 1H: No full-month offer to employee
      - 1F: MEC (not MV) full-month offer to employee
      - MV full-month offer to employee:
          • 1A if spouse+dependents also offered AND affordable
          • 1E if spouse+dependents also offered but NOT affordable
          • 1D if spouse only
          • 1C if dependents only
          • 1B if employee only
    """
    if not offer_ee_allmonth:
        return "1H"
    if eligible_mv:
        if offer_spouse and offer_dependents:
            return "1A" if affordable else "1E"
        if offer_spouse and not offer_dependents:
            return "1D"
        if offer_dependents and not offer_spouse:
            return "1C"
        return "1B"  # employee only
    return "1F"

def _month_line16(*, employed: bool, enrolled_full: bool, waiting: bool, ft: bool,
                  offer_ee_allmonth: bool, affordable: bool) -> str:
    """
    Precedence:
      2C (enrolled full month) → 2A (not employed) → 2D (waiting) → 2B (not FT) → 2H (affordable offer) → blank
    """
    if enrolled_full:
        return "2C"
    if not employed:
        return "2A"
    if waiting:
        return "2D"
    if not ft:
        return "2B"
    if offer_ee_allmonth and affordable:
        return "2H"
    return ""

# ------------------------------------------------------------
# Build Interim / Final / Penalty
# ------------------------------------------------------------
def build_interim(
    emp_demo: pd.DataFrame,
    emp_elig: pd.DataFrame,
    emp_enroll: pd.DataFrame,
    dep_enroll: pd.DataFrame,
    year: int,
    **_kwargs,
) -> pd.DataFrame:
    """
    Build the monthly Interim table.

    Key rules:
      - employed False if any 'Terminated' row overlaps the month.
      - ft/pt require role coverage entire month (and not terminated).
      - offer/enrollment derived from eligibility/enrollment tables.
      - enrollment union allows stitched/adjacent intervals across the month.
    """
    emp_demo = _apply_aliases(emp_demo or pd.DataFrame())
    emp_elig = _apply_aliases(emp_elig or pd.DataFrame())
    emp_enroll = _apply_aliases(emp_enroll or pd.DataFrame())
    dep_enroll = _apply_aliases(dep_enroll or pd.DataFrame())

    for df, sc, ec in (
        (emp_elig, "eligibilitystartdate", "eligibilityenddate"),
        (emp_enroll, "enrollmentstartdate", "enrollmentenddate"),
        (dep_enroll, "enrollmentstartdate", "enrollmentenddate"),
    ):
        if not df.empty:
            for c in (sc, ec):
                if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
                    df[c] = pd.to_datetime(df[c], errors="coerce")

    # Status table derived from demographic
    st = _status_from_demographic(emp_demo)

    employee_ids = _collect_employee_ids(emp_demo, st, emp_elig, emp_enroll, dep_enroll)
    rows: List[Dict[str, Any]] = []

    for emp in employee_ids:
        el_emp = emp_elig[emp_elig["employeeid"].astype(str) == str(emp)].copy() if not emp_elig.empty else pd.DataFrame()
        en_emp = emp_enroll[emp_enroll["employeeid"].astype(str) == str(emp)].copy() if not emp_enroll.empty else pd.DataFrame()
        de_emp = dep_enroll[dep_enroll["employeeid"].astype(str) == str(emp)].copy() if not dep_enroll.empty else pd.DataFrame()
        st_emp = st[st["employeeid"].astype(str) == str(emp)].copy() if not st.empty else pd.DataFrame()

        for m in range(1, 13):
            ms, me = month_bounds(year, m)

            # ---- status flags
            employed = _is_employed_month(st_emp, ms, me)
            ft = _is_ft(st_emp, ms, me)
            parttime = (not ft) and _is_pt(st_emp, ms, me)

            # ---- eligibility / offer flags
            elig_any = _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me) if not el_emp.empty else False
            elig_full = _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me) if not el_emp.empty else False

            # MV: PlanA (EMP, EMPFAM, EMPCHILD, EMPSPOUSE) full-month eligibility
            eligible_mv = False
            if not el_emp.empty:
                el_mv_mask = el_emp["plancode"].astype(str).str.upper().str.strip().eq("PLANA") if "plancode" in el_emp.columns else pd.Series(False, index=el_emp.index)
                eligible_mv = _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=el_mv_mask)

            offer_ee_allmonth = _offered_allmonth(el_emp, ms, me)

            # spouse/dependents offered if EITHER eligibility (family tiers) or enrollment (family tiers while enrolled)
            offer_spouse = _tier_offered_any(el_emp, "eligibilitytier", ("EMPFAM","EMPSPOUSE"), "eligibilitystartdate", "eligibilityenddate", ms, me) \
                           or _tier_offered_any(en_emp, "enrollmenttier", ("EMPFAM","EMPSPOUSE"), "enrollmentstartdate", "enrollmentenddate", ms, me, require_enrolled=True)
            offer_dependents = _tier_offered_any(el_emp, "eligibilitytier", ("EMPFAM","EMPCHILD"), "eligibilitystartdate", "eligibilityenddate", ms, me) \
                               or _tier_offered_any(en_emp, "enrollmenttier", ("EMPFAM","EMPCHILD"), "enrollmentstartdate", "enrollmentenddate", ms, me, require_enrolled=True)

            # ---- enrollment union full month
            enrolled_full = _enrolled_full_month_union(en_emp, ms, me)

            # ---- affordability (employee-only cost)
            emp_cost = _latest_emp_cost_for_month(el_emp, ms, me)
            affordable = (emp_cost is not None) and (emp_cost < AFFORDABILITY_THRESHOLD)

            # ---- waiting period heuristic
            waiting = False
            if employed and not elig_any and not el_emp.empty and "eligibilitystartdate" in el_emp.columns:
                future_starts = el_emp["eligibilitystartdate"].dropna()
                waiting = (future_starts.dt.date > me).any()

            # ---- monthly codes
            # If no eligibility data exists for this month but ENROLLED full month, force Line 14 to 1E.
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
                "spouse_enrolled": False,   # optional detail; could be added if needed
                "child_enrolled": False,
                "waitingperiod_month": bool(waiting),
                "affordable_plan": bool(affordable),
                "line14_final": l14,
                "line16_final": l16,
                "line14_all12": "",
            })

    interim = pd.DataFrame.from_records(rows)

    # roll up 1G if never FT but enrolled at least one month
    one_g_emp_ids = []
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
    return interim.loc[:, cols]

def build_final(interim: pd.DataFrame) -> pd.DataFrame:
    if interim is None or interim.empty:
        return pd.DataFrame(columns=["EmployeeID","Month","Line14_Final","Line16_Final"])
    df = interim.copy()
    df = df.sort_values(["EmployeeID","MonthNum"])
    out = []
    for emp, g in df.groupby("EmployeeID", sort=False):
        l14_all12 = (g["line14_all12"].astype(str).str.strip().iloc[0] or "")
        if l14_all12:
            out.append({"EmployeeID": emp, "Month":"All 12 months",
                        "Line14_Final": l14_all12, "Line16_Final": ""})
        else:
            for _, r in g.iterrows():
                out.append({"EmployeeID": emp, "Month": r["Month"],
                            "Line14_Final": r["line14_final"], "Line16_Final": r["line16_final"]})
    return pd.DataFrame.from_records(out)

def build_penalty_dashboard(interim: pd.DataFrame) -> pd.DataFrame:
    if interim is None or interim.empty:
        return pd.DataFrame(columns=["EmployeeID","Reason"] + FULL_MONTHS)

    # Reasoning per month
    def month_reason(row) -> str:
        if not row["offer_ee_allmonth"]:
            # A-penalty exposure if no MEC offer
            if not row["employed"]: return "Not employed"
            if row["waitingperiod_month"]: return "Waiting period"
            if not row["eligibleforcoverage"]: return "Not eligible"
            return "No MEC offer"
        # MEC offered but not enrolled and not affordable → B-penalty exposure
        if (not row["enrolled_allmonth"]) and (not row["affordable_plan"]):
            return "Offered but not affordable (B)"
        return "–"

    df = interim.copy()
    out_rows = []
    for emp, g in df.groupby("EmployeeID", sort=False):
        rec = {"EmployeeID": emp}
        reasons = []
        for _, r in g.sort_values("MonthNum").iterrows():
            reasons.append(month_reason(r))
        for i, m in enumerate(FULL_MONTHS):
            rec[m] = reasons[i] if i < len(reasons) else "–"
        # Dominant reason (first non-–)
        dom = next((x for x in reasons if x != "–"), "–")
        rec["Reason"] = dom
        out_rows.append(rec)

    cols = ["EmployeeID","Reason"] + FULL_MONTHS
    return pd.DataFrame.from_records(out_rows, columns=cols)
