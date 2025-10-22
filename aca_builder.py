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

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
AFFORDABILITY_THRESHOLD = 50.00  # < $50 => affordable (use <= if you want $50 to count)

# Penalty amounts (per month)
PENALTY_A_MONTHLY = 241.67  # 
PENALTY_B_MONTHLY = 386.67  # 

# ------------------------------------------------------------
# Normalizers / helpers for column naming
# ------------------------------------------------------------
def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _ensure_employeeid_str(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    if "employeeid" in df.columns:
        df["employeeid"] = df["employeeid"].astype(str).str.strip()
    return df

def _normalize_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = _lower_cols(df)
    df = _ensure_employeeid_str(df)

    cols_lower = set(df.columns)
    lower_map = {c: c for c in df.columns}

    # Eligibility dates
    if "eligibilitystartdate" not in cols_lower:
        for alt in ("eligiblestartdate", "startdate", "elig start"):
            if alt in cols_lower:
                df["eligibilitystartdate"] = pd.to_datetime(df[lower_map[alt]], errors="coerce")
                break
    else:
        df["eligibilitystartdate"] = pd.to_datetime(df["eligibilitystartdate"], errors="coerce")

    if "eligibilityenddate" not in cols_lower:
        for alt in ("eligibleenddate", "enddate", "elig end"):
            if alt in cols_lower:
                df["eligibilityenddate"] = pd.to_datetime(df[lower_map[alt]], errors="coerce")
                break
    else:
        df["eligibilityenddate"] = pd.to_datetime(df["eligibilityenddate"], errors="coerce")

    # Tier columns
    if "eligibilitytier" not in cols_lower and "eligibletier" in cols_lower:
        df["eligibilitytier"] = df[lower_map["eligibletier"]].astype(str).str.strip()

    # MV / MEC flags
    if "mv" not in cols_lower and "minimumvalue" in cols_lower:
        df["mv"] = df["minimumvalue"].astype(bool)
    if "mec" not in cols_lower and "minimumessentialcoverage" in cols_lower:
        df["mec"] = df["minimumessentialcoverage"].astype(bool)

    # cost columns (employee-only)
    for col in ("employeeonlycost", "employeecost", "emp cost", "contribution"):
        if col in cols_lower:
            df["employeeonlycost"] = pd.to_numeric(df[col], errors="coerce")
            break
    return df

def _normalize_enrollment(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = _lower_cols(df)
    df = _ensure_employeeid_str(df)

    # Dates
    if "enrollmentstartdate" in df.columns:
        df["enrollmentstartdate"] = pd.to_datetime(df["enrollmentstartdate"], errors="coerce")
    elif "startdate" in df.columns:
        df["enrollmentstartdate"] = pd.to_datetime(df["startdate"], errors="coerce")

    if "enrollmentenddate" in df.columns:
        df["enrollmentenddate"] = pd.to_datetime(df["enrollmentenddate"], errors="coerce")
    elif "enddate" in df.columns:
        df["enrollmentenddate"] = pd.to_datetime(df["enddate"], errors="coerce")

    # Tier
    if "enrollmenttier" not in df.columns and "tier" in df.columns:
        df["enrollmenttier"] = df["tier"].astype(str).str.strip()

    # Status / waive
    if "waive" not in df.columns and "status" in df.columns:
        df["waive"] = df["status"].astype(str).str.contains("waiv", case=False, na=False)

    return df

def _normalize_depenroll(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = _lower_cols(df)
    df = _ensure_employeeid_str(df)

    # Dates
    if "eligiblestartdate" in df.columns:
        df["eligiblestartdate"] = pd.to_datetime(df["eligiblestartdate"], errors="coerce")
    if "eligibleenddate" in df.columns:
        df["eligibleenddate"] = pd.to_datetime(df["eligibleenddate"], errors="coerce")

    # EligibleTier -> eligibilitytier
    if "eligibletier" in df.columns and "eligibilitytier" not in df.columns:
        df["eligibilitytier"] = df["eligibletier"].astype(str).str.strip()

    # Tier (Enrollment) -> enrollmenttier
    if "enrollmenttier" not in df.columns and "tier" in df.columns:
        df["enrollmenttier"] = df["tier"].astype(str).str.strip()

    return df


# ------------------------------------------------------------
# Helpers
# -----

def _is_employed_month(st_emp: pd.DataFrame, ms, me) -> bool:
    if st_emp is None or st_emp.empty:
        return False
    # employed if any status overlap with month
    return _any_overlap(st_emp, "statusstartdate", "statusenddate", ms, me)

def _is_ft(st_emp: pd.DataFrame, ms, me) -> bool:
    if st_emp is None or st_emp.empty:
        return False
    # ft if any FT overlap
    ft_rows = st_emp[st_emp["fulltime"].astype(bool)]
    return _any_overlap(ft_rows, "statusstartdate", "statusenddate", ms, me) if not ft_rows.empty else False

def _eligible_any(el_emp: pd.DataFrame, ms, me) -> bool:
    if el_emp is None or el_emp.empty:
        return False
    return _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me)

def _eligible_mv_full_month(el_emp: pd.DataFrame, ms, me) -> bool:
    if el_emp is None or el_emp.empty:
        return False
    mv_rows = el_emp[el_emp.get("mv", False).astype(bool)]
    return _all_month(mv_rows, "eligibilitystartdate", "eligibilityenddate", ms, me) if not mv_rows.empty else False

def _offered_allmonth(el_emp: pd.DataFrame, ms, me) -> bool:
    if el_emp is None or el_emp.empty:
        return False
    # offer if any MEC OR MV
    has_offer = el_emp[(el_emp.get("mec", False).astype(bool)) | (el_emp.get("mv", False).astype(bool))]
    return _all_month(has_offer, "eligibilitystartdate", "eligibilityenddate", ms, me) if not has_offer.empty else False

def _enrolled_full_month_union(en_emp: pd.DataFrame, ms, me) -> bool:
    """
    Full-month enrollment if the UNION of enrollment intervals (excluding waived rows) covers the whole month.
    """
    if en_emp is None or en_emp.empty:
        return False
    keep = en_emp[~en_emp.get("waive", False).astype(bool)]
    return _all_month(keep, "enrollmentstartdate", "enrollmentenddate", ms, me) if not keep.empty else False

def _tier_offered_any(df: pd.DataFrame, tier_col: str, tiers: Tuple[str, ...],
                      start_col: str, end_col: str, ms, me, require_enrolled: bool=False) -> bool:
    if df is None or df.empty or tier_col not in df.columns:
        return False
    tmp = df[df[tier_col].astype(str).str.upper().isin([t.upper() for t in tiers])]
    if tmp.empty:
        return False
    if require_enrolled:
        tmp = tmp[~tmp.get("waive", False).astype(bool)]
        if tmp.empty:
            return False
    return _any_overlap(tmp, start_col, end_col, ms, me)

def _tier_enrolled_full_month(df: pd.DataFrame, tiers: Tuple[str, ...], ms, me) -> bool:
    if df is None or df.empty or "enrollmenttier" not in df.columns:
        return False
    tmp = df[df["enrollmenttier"].astype(str).str.upper().isin([t.upper() for t in tiers])]
    if tmp.empty:
        return False
    tmp = tmp[~tmp.get("waive", False).astype(bool)]
    return _all_month(tmp, "enrollmentstartdate", "enrollmentenddate", ms, me) if not tmp.empty else False

def _latest_emp_cost_for_month(el_emp: pd.DataFrame, ms, me) -> Optional[float]:
    """
    Pick the latest employee-only cost row active within the month.
    """
    if el_emp is None or el_emp.empty:
        return None
    elig = el_emp.copy()
    elig = elig[_any_overlap(elig, "eligibilitystartdate", "eligibilityenddate", ms, me)]
    if elig.empty:
        return None
    elig = elig.sort_values(by=["eligibilitystartdate", "eligibilityenddate"], ascending=[True, True])
    cost = pd.to_numeric(elig.get("employeeonlycost"), errors="coerce")
    if cost is None:
        return None
    try:
        return float(cost.iloc[-1])
    except Exception:
        return None


# ------------------------------------------------------------
# Line 14 / Line 16 mapping
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
        return "1B"
    return "1F"

def _month_line16(
    employed: bool,
    enrolled_full: bool,
    waiting: bool,
    ft: bool,
    offer_ee_allmonth: bool,
    affordable: bool,
) -> str:
    """
    Precedence:
      2C: enrolled full month (even if not employed)
      2A: not employed
      2D: waiting period
      2B: not full-time for the month
      2H: rate-of-pay safe harbor (use affordable flag)
      else: blank
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
# Public: build_interim / build_final / build_penalty_dashboard
# ----------
def build_interim(
    emp_demo: pd.DataFrame,
    emp_status: pd.DataFrame,
    emp_elig: pd.DataFrame,
    emp_enroll: pd.DataFrame,
    dep_enroll: pd.DataFrame,
    year: int,
    **_kwargs,  # ab
) -> pd.DataFrame:

    emp_demo = _lower_cols(emp_demo)
    emp_status = _lower_cols(emp_status)
    emp_elig = _normalize_eligibility(emp_elig)
    emp_enroll = _normalize_enrollment(emp_enroll)
    dep_enroll = _normalize_depenroll(dep_enroll)

    # Normalize dates if caller bypassed prepare_inputs
    for df, sc, ec in [
        (emp_elig, "eligibilitystartdate", "eligibilityenddate"),
        (emp_enroll, "enrollmentstartdate", "enrollmentenddate"),
        (dep_enroll, "eligiblestartdate", "eligibleenddate"),
    ]:
        if not df.empty:
            for c in (sc, ec):
                if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
                    df[c] = pd.to_datetime(df[c], errors="coerce")

    # Status table (fallback to demographic if missing)
    st = emp_status if emp_status is not None else pd.DataFrame()
    if st.empty:
        st = _status_from_demographic(emp_demo)

    # --- optional authoritative Emp Wait Period sheet
    emp_wait = _kwargs.get("emp_wait_period", pd.DataFrame())
    wait_map: Dict[str, Tuple] = {}
    if emp_wait is not None and not emp_wait.empty:
        ew = emp_wait.copy()
        # Normalize columns
        if "effectivedate" in ew.columns and not pd.api.types.is_datetime64_any_dtype(ew["effectivedate"]):
            ew["effectivedate"] = pd.to_datetime(ew["effectivedate"], errors="coerce")
        if "wait period" in ew.columns:
            ew["wait period"] = pd.to_numeric(ew["wait period"], errors="coerce").fillna(0).astype(int)
        for _, r in ew.iterrows():
            eid = str(r.get("employeeid","")).strip()
            eff = r.get("effectivedate")
            days = int(r.get("wait period") or 0)
            if eid and pd.notna(eff) and days > 0:
                ws = (eff.date() - timedelta(days=days))
                we = (eff.date() - timedelta(days=1))
                wait_map[eid] = (ws, we)

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

            # ---- eligibility overlap flags
            elig_any = _eligible_any(el_emp, ms, me)
            elig_full = _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me) if not el_emp.empty else False

            eligible_mv = _eligible_mv_full_month(el_emp, ms, me)
            offer_ee_allmonth = _offered_allmonth(el_emp, ms, me)

            # ---- enrolled full-month (UNION across rows, excluding "Waive")
            if not en_emp.empty:
                enrolled_full = _enrolled_full_month_union(en_emp, ms, me)
            else:
                enrolled_full = False

            # ---- spouse/child eligibility (any overlap in Eligibility)
            spouse_eligible = _tier_offered_any(
                el_emp, "eligibilitytier", ("EMPFAM", "EMPSPOUSE"),
                "eligibilitystartdate", "eligibilityenddate", ms, me,
                require_enrolled=False
            )
            child_eligible = _tier_offered_any(
                el_emp, "eligibilitytier", ("EMPFAM", "EMPCHILD"),
                "eligibilitystartdate", "eligibilityenddate", ms, me,
                require_enrolled=False
            )

            # ---- spouse/child enrolled (FULL MONTH in Enrollment tiers)
            spouse_enrolled = _tier_enrolled_full_month(en_emp, ("EMPFAM", "EMPSPOUSE"), ms, me)
            child_enrolled = _tier_enrolled_full_month(en_emp, ("EMPFAM", "EMPCHILD"), ms, me)

            # ---- offer flags (eligibility OR enrollment)
            offer_spouse = spouse_eligible or spouse_enrolled
            offer_dependents = child_eligible or child_enrolled

            # ---- affordability (employee-only cost)
            emp_cost = _latest_emp_cost_for_month(el_emp, ms, me)
            affordable = (emp_cost is not None) and (emp_cost < AFFORDABILITY_THRESHOLD)

            # ---- waiting period (authoritative sheet -> fallback heuristic)
            waiting = False
            eid = str(emp)
            if eid in wait_map:
                ws, we = wait_map[eid]
                # Only flag waiting if employed and the window overlaps the month
                if employed and not (we < ms or ws > me):
                    waiting = True
            else:
                # Existing heuristic
                if employed and not elig_any and not el_emp.empty and "eligibilitystartdate" in el_emp.columns:
                    future_starts = el_emp["eligibilitystartdate"].dropna()
                    waiting = (future_starts.dt.date > me).any()

            # ---- monthly codes
            # FEEDBACK 2025-10-21:
            # If no eligibility data exists for this month (elig_any=False) but the employee is
            # ENROLLED for the entire month (enrolled_full=True), assume employee-only cost is > $50
            # and force Line 14 to 1E. This mirrors the decision for cases like Emp 1007.
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
                "Year": year,
                "MonthNum": m,
                "MonthName": MONTHS[m-1],
                "MonthStart": ms,
                "MonthEnd": me,
                "employed": bool(employed),
                "ft": bool(ft),
                "elig_any": bool(elig_any),
                "elig_full": bool(elig_full),
                "eligible_mv": bool(eligible_mv),
                "offer_ee_allmonth": bool(offer_ee_allmonth),
                "enrolled_full": bool(enrolled_full),
                "spouse_eligible": bool(spouse_eligible),
                "child_eligible": bool(child_eligible),
                "spouse_enrolled": bool(spouse_enrolled),
                "child_enrolled": bool(child_enrolled),
                "offer_spouse": bool(offer_spouse),
                "offer_dependents": bool(offer_dependents),
                "affordable_plan": bool(affordable),
                "waitingperiod_month": bool(waiting),
                "line14_final": l14,
                "line16_final": l16,
            })

    cols = [
        "EmployeeID","Year","MonthNum","MonthName","MonthStart","MonthEnd",
        "employed","ft","elig_any","elig_full","eligible_mv","offer_ee_allmonth",
        "enrolled_full","spouse_eligible","child_eligible","spouse_enrolled","child_enrolled",
        "offer_spouse","offer_dependents","affordable_plan","waitingperiod_month",
        "line14_final","line16_final"
    ]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def build_final(interim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse interim into Final per employee with per-month Line 14/16 values and flags.
    """
    if interim_df is None or interim_df.empty:
        return pd.DataFrame(columns=["EmployeeID"])

    df = interim_df.copy()

    # Year-level 1G: if no employment in the year, set ALL Line 14 to 1G and blank monthly codes
    # (your existing logic may already handle this upstream; left here as placeholder if needed)

    # Flatten months into columns
    out_rows = []
    for emp, grp in df.groupby("EmployeeID", sort=False):
        grp = grp.sort_values("MonthNum")
        row = {"EmployeeID": emp}
        # Codes
        for i, mon in enumerate(FULL_MONTHS, start=1):
            row[f"L14_{mon}"] = str(grp.iloc[i-1]["line14_final"]) if i-1 < len(grp) else ""
            row[f"L16_{mon}"] = str(grp.iloc[i-1]["line16_final"]) if i-1 < len(grp) else ""
        # Flags
        for i, mon in enumerate(FULL_MONTHS, start=1):
            for col in ("employed","ft","elig_any","elig_full","eligible_mv","offer_ee_allmonth",
                        "enrolled_full","offer_spouse","offer_dependents","affordable_plan","waitingperiod_month"):
                row[f"{col}_{mon}"] = bool(grp.iloc[i-1][col]) if i-1 < len(grp) else False

        out_rows.append(row)

    return pd.DataFrame(out_rows)


# ------------------------------------------------------------
# Penalty dashboard (compact)
# ------------------------------------------------------------
def build_penalty_dashboard(interim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Very compact A/B exposure summary from interim.
    """
    if interim_df is None or interim_df.empty:
        return pd.DataFrame(columns=["EmployeeID", "Reason"] + MONTHS)

    def month_penalty_vals(grp: pd.DataFrame) -> List[str]:
        vals = []
        for i in range(1, 13):
            r = grp[grp["MonthNum"] == i]
            if r.empty:
                vals.append("")
                continue
            l14 = str(r.iloc[0]["line14_final"])
            l16 = str(r.iloc[0]["line16_final"])
            if l14 in ("1H","") or l16 in ("2C","2B","2D","2A","2F","2G","2H"):
                vals.append("")  # no penalty this month
            else:
                vals.append("A/B")
        return vals

    BR = "\n"
    def fmt_month_list(ms: List[int]) -> str:
        names = [FULL_MONTHS[i-1] for i in ms]
        return ", ".join(names)

    rows = []
    for emp, grp in interim_df.groupby("EmployeeID", sort=False):
        month_vals = month_penalty_vals(grp)
        # quick categorization text (optional)
        months_not_emp = [i for i in range(1,13) if bool(grp[grp["MonthNum"]==i].iloc[0]["employed"]) is False]
        months_wait = [i for i in range(1,13) if bool(grp[grp["MonthNum"]==i].iloc[0]["waitingperiod_month"]) is True]
        months_not_elig = [i for i in range(1,13) if bool(grp[grp["MonthNum"]==i].iloc[0]["elig_any"]) is False]
        months_other = [i for i in range(1,13)
                        if (i not in months_not_emp) and (i not in months_wait) and (i not in months_not_elig)]

        reason_b = "Penalty B: Not affordable or no 2-series relief"
        # crude rule-of-thumb: if most months are 1H -> say A
        if month_vals.count("A/B") >= 6:
            sublines = []
            if months_not_emp:
                sublines.append(
                    f"No coverage could be offered in {fmt_month_list(months_not_emp)} because the employee was not employed."
                )
            if months_wait:
                sublines.append(
                    f"Employee was not eligible for coverage in ...ecause they were in their waiting period during those month(s)."
                )
            if months_not_elig:
                sublines.append(
                    f"Employee was not eligible for coverage in {fmt_month_list(months_not_elig)}."
                )
            if months_other:
                sublines.append(
                    f"No minimum essential coverage offer was recorded in {fmt_month_list(months_other)}."
                )

            reason = "Penalty A: No MEC offered"
            if sublines:
                reason = reason + BR + BR.join(sublines)
        else:
            reason = reason_b

        rows.append({
            "EmployeeID": str(emp),
            "Reason": reason,
            **dict(zip(MONTHS, month_vals))
        })

    cols = ["EmployeeID", "Reason"] + MONTHS
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
