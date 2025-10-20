# aca_builder.py
# Builds the Interim, Final, and Penalty Dashboard tables

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

# -------------------------
# Constants / small helpers
# -------------------------

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
AFFORDABLE_THRESHOLD = 50.0  # "affordable" if EMP-only cost is strictly < $50
ENROLLMENT_IMPLIES_OFFER = True  # infer offer (and MV only when NO eligibility exists) from enrollment

def _month_bounds(year: int, m: int) -> Tuple[date, date]:
    start = date(year, m, 1)
    if m == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, m + 1, 1) - timedelta(days=1)
    return start, end

def _any_overlap(df: pd.DataFrame, start_col: str, end_col: str,
                 m_start: date, m_end: date, mask: Optional[pd.Series] = None) -> bool:
    if df is None or df.empty:
        return False
    _m = mask if mask is not None else pd.Series(True, index=df.index)
    s = pd.to_datetime(df.loc[_m, start_col], errors="coerce")
    e = pd.to_datetime(df.loc[_m, end_col], errors="coerce")
    s = s.fillna(pd.Timestamp.min).dt.date
    e = e.fillna(pd.Timestamp.max).dt.date
    return bool(((e >= m_start) & (s <= m_end)).any())

def _all_month(df: pd.DataFrame, start_col: str, end_col: str,
               m_start: date, m_end: date, mask: Optional[pd.Series] = None) -> bool:
    """
    IRS 'full-month' = every day of the month is covered by a single row.
    """
    if df is None or df.empty:
        return False
    _m = mask if mask is not None else pd.Series(True, index=df.index)
    s = pd.to_datetime(df.loc[_m, start_col], errors="coerce")
    e = pd.to_datetime(df.loc[_m, end_col], errors="coerce")
    s = s.fillna(pd.Timestamp.min).dt.date
    e = e.fillna(pd.Timestamp.max).dt.date
    return bool(((s <= m_start) & (e >= m_end)).any())

def _all_month_union(df: pd.DataFrame, start_col: str, end_col: str,
                     m_start: date, m_end: date, mask: Optional[pd.Series] = None) -> bool:
    """
    Treat multiple overlapping/adjacent rows as continuous coverage.
    Used for ENROLLMENT only (do NOT loosen eligibility).
    """
    if df is None or df.empty:
        return False
    _m = mask if mask is not None else pd.Series(True, index=df.index)
    s = pd.to_datetime(df.loc[_m, start_col], errors="coerce").fillna(pd.Timestamp.min).dt.date
    e = pd.to_datetime(df.loc[_m, end_col], errors="coerce").fillna(pd.Timestamp.max).dt.date

    intervals: List[Tuple[date, date]] = []
    for a, b in zip(s, e):
        lo, hi = max(a, m_start), min(b, m_end)
        if hi >= lo:
            intervals.append((lo, hi))
    if not intervals:
        return False
    intervals.sort()
    cur_s, cur_e = intervals[0]
    if cur_s > m_start:
        return False
    for lo, hi in intervals[1:]:
        # allow adjacent
        if lo > cur_e + timedelta(days=1):
            return False
        cur_e = max(cur_e, hi)
        if cur_e >= m_end:
            return True
    return cur_e >= m_end

def _tier_full_month(df: pd.DataFrame, tier_col: str, tokens: Tuple[str, ...],
                     start_col: str, end_col: str, ms: date, me: date, *,
                     require_enrolled: bool = True) -> bool:
    """
    Returns True if FULL-MONTH coverage exists at any tier containing tokens.
    Excludes WAIVE-like values. Uses union logic.
    """
    if df is None or df.empty or tier_col not in df.columns:
        return False
    mask = pd.Series(True, index=df.index)
    if require_enrolled and "isenrolled" in df.columns:
        mask &= df["isenrolled"].astype(bool)

    # Exclude WAIVE / WAIVED / WAIVER / variants
    waive_mask = pd.Series(False, index=df.index)
    for col in ("plancode", "planname"):
        if col in df.columns:
            s = df[col].astype(str).str.upper().str.strip()
            s = s.str.replace(r"[^A-Z]", "", regex=True)
            waive_mask |= s.str.startswith("WAIV")
    mask &= ~waive_mask

    tiers = df[tier_col].astype(str).str.upper().str.strip()
    tok_mask = pd.Series(False, index=df.index)
    for t in tokens:
        tok_mask |= tiers.str.contains(t, na=False)
    mask &= tok_mask

    return _all_month_union(df, start_col, end_col, ms, me, mask=mask)

def _ensure_col(df: pd.DataFrame, target: str, candidates: List[str]) -> pd.DataFrame:
    """
    If target column is missing, attempt to alias from any candidate.
    """
    if df is None or df.empty:
        return df
    if target in df.columns:
        return df
    for c in candidates:
        if c in df.columns:
            df[target] = df[c]
            break
    return df

# -------------------------
# Employment / status logic
# -------------------------

def _is_employed(st_emp: pd.DataFrame, ms: date, me: date) -> bool:
    if st_emp is None or st_emp.empty:
        return False
    return _any_overlap(st_emp, "statusstartdate", "statusenddate", ms, me)

def _is_ft(st_emp: pd.DataFrame, ms: date, me: date) -> bool:
    """
    FT only if (FT present) AND (NOT LOA/LEAVE/TERMINATED) during the overlap.
    """
    if st_emp is None or st_emp.empty:
        return False
    est = st_emp.get("_estatus_norm", pd.Series("", index=st_emp.index))
    rol = st_emp.get("_role_norm", pd.Series("", index=st_emp.index))
    s_ft = (est.str.contains("FULLTIME|^FT$", na=False) |
            rol.str.contains("FULLTIME|^FT$", na=False))
    s_off = (est.str.contains("LOA|LEAVE|TERM|TERMINATED", na=False) |
             rol.str.contains("LOA|LEAVE|TERM|TERMINATED", na=False))
    mask = s_ft & ~s_off
    return _any_overlap(st_emp, "statusstartdate", "statusenddate", ms, me, mask=mask)

def _is_pt(st_emp: pd.DataFrame, ms: date, me: date) -> bool:
    if st_emp is None or st_emp.empty:
        return False
    est = st_emp.get("_estatus_norm", pd.Series("", index=st_emp.index))
    rol = st_emp.get("_role_norm", pd.Series("", index=st_emp.index))
    mask = (est.str.contains("PARTTIME|^PT$", na=False) |
            rol.str.contains("PARTTIME|^PT$", na=False))
    return _any_overlap(st_emp, "statusstartdate", "statusenddate", ms, me, mask=mask)

# -------------------------
# Line codes
# -------------------------

def _month_line14(eligible_mv: bool, offer_ee_allmonth: bool,
                  offer_spouse: bool, offer_dependents: bool,
                  affordable: bool) -> str:
    """
    Returns the 1A/1B/1E/1F/1H code for the month.
    1H: no full-month offer
    If MV offered:
      - spouse & dependents + affordable → 1A
      - spouse & dependents (not affordable) → 1E
      - employee-only → 1B
    MEC but not MV → 1F
    """
    if not offer_ee_allmonth:
        return "1H"

    if eligible_mv:
        both = (offer_spouse and offer_dependents)
        if both and affordable:
            return "1A"
        if both:
            return "1E"
        return "1B"

    return "1F"

def _month_line16(employed: bool, enrolled_full: bool, waiting: bool,
                  ft: bool, offer_ee_allmonth: bool, affordable: bool) -> str:
    """
    Precedence:
      2C (enrolled all month) >
      2A (not employed any day) >
      2D (waiting period) >
      2B (not FT) >
      2H (offered & affordable)
      else blank
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

# -----------------------------------
# Core builder: Interim / Final / PD
# -----------------------------------

def _collect_employee_ids(*dfs: pd.DataFrame) -> List[str]:
    ids = set()
    for df in dfs:
        if df is None or df.empty:
            continue
        if "employeeid" in df.columns:
            vals = df["employeeid"].dropna().astype(str).str.strip()
            ids.update(vals.tolist())
    return sorted(ids)

def build_interim(emp_demo: pd.DataFrame, emp_status: pd.DataFrame,
                  emp_elig: pd.DataFrame, emp_enroll: pd.DataFrame,
                  dep_enroll: pd.DataFrame, *, year: int) -> pd.DataFrame:
    """
    Build the month-level interim table for a given year.
    """

    # Aliases (belt & suspenders — prepare_inputs already normalizes these)
    for df in (emp_elig, emp_enroll):
        if df is None or df.empty:
            continue
        for need, cands in [
            ("plancode",        ["eligibleplan", "plancode", "plan"]),
            ("eligibilitytier", ["eligibletier", "eligible tier", "tier"]),
            ("enrollmenttier",  ["enrollmenttier", "enrollment tier", "tier"]),
        ]:
            _ensure_col(df, need, cands)

    # Make sure status table has normalized helper cols
    if emp_status is not None and not emp_status.empty:
        if "_estatus_norm" not in emp_status.columns and "employmentstatus" in emp_status.columns:
            emp_status["_estatus_norm"] = emp_status["employmentstatus"].astype(str).str.upper().str.replace(r"\s+", "", regex=True)
        if "_role_norm" not in emp_status.columns and "role" in emp_status.columns:
            emp_status["_role_norm"] = emp_status["role"].astype(str).str.upper().str.replace(r"\s+", "", regex=True)

    employee_ids = _collect_employee_ids(emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll)

    rows: List[Dict[str, Any]] = []

    for emp in employee_ids:
        st_emp = emp_status[emp_status["employeeid"].astype(str) == str(emp)] if (emp_status is not None and "employeeid" in emp_status.columns) else pd.DataFrame()
        el_emp = emp_elig[emp_elig["employeeid"].astype(str) == str(emp)] if (emp_elig is not None and "employeeid" in emp_elig.columns) else pd.DataFrame()
        en_emp = emp_enroll[emp_enroll["employeeid"].astype(str) == str(emp)] if (emp_enroll is not None and "employeeid" in emp_enroll.columns) else pd.DataFrame()

        # First eligibility start (for waiting-period logic)
        first_elig_start = None
        if not el_emp.empty and "eligibilitystartdate" in el_emp.columns:
            tmp = pd.to_datetime(el_emp["eligibilitystartdate"], errors="coerce")
            if not tmp.dropna().empty:
                first_elig_start = tmp.min().date()

        for m in range(1, 13):
            ms, me = _month_bounds(year, m)

            employed = _is_employed(st_emp, ms, me)
            ft = _is_ft(st_emp, ms, me)
            parttime = _is_pt(st_emp, ms, me)

            # ELIGIBILITY — any/full-month
            elig_any = False
            elig_full = False
            if not el_emp.empty and {"eligibilitystartdate","eligibilityenddate"} <= set(el_emp.columns):
                elig_any  = _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me)
                elig_full = _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me)

            # ELIGIBLE_MV — strictly from Eligibility (PlanA)
            eligible_mv = False
            if "plancode" in el_emp.columns and not el_emp.empty:
                plan_u = el_emp["plancode"].astype(str).str.upper().str.strip()
                eligible_mv = _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate",
                                           ms, me, mask=plan_u.eq("PLANA"))

            # OFFER to employee (full month) — from Eligibility (tiers with EMP)
            offer_ee_allmonth = False
            if not el_emp.empty and "eligibilitytier" in el_emp.columns:
                tiers = el_emp["eligibilitytier"].astype(str).str.upper().str.strip()
                mask_emp = tiers.str.contains("EMP", na=False)
                offer_ee_allmonth = _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate",
                                               ms, me, mask=mask_emp)

            # ENROLLMENT — full month (union); exclude WAIVE
            enrolled_full = False
            if not en_emp.empty and {"enrollmentstartdate","enrollmentenddate"} <= set(en_emp.columns):
                mask_en = pd.Series(True, index=en_emp.index)
                if "isenrolled" in en_emp.columns:
                    mask_en &= en_emp["isenrolled"].astype(bool)
                # exclude any WAIVE-like values
                waive_mask = pd.Series(False, index=en_emp.index)
                for col in ("plancode", "planname"):
                    if col in en_emp.columns:
                        s = en_emp[col].astype(str).str.upper().str.strip()
                        s = s.str.replace(r"[^A-Z]", "", regex=True)
                        waive_mask |= s.str.startswith("WAIV")
                mask_en &= ~waive_mask
                enrolled_full = _all_month_union(en_emp, "enrollmentstartdate", "enrollmentenddate", ms, me, mask=mask_en)

            # Spouse/Child eligibility (any overlap)
            spouse_eligible = False
            child_eligible  = False
            if not el_emp.empty and "eligibilitytier" in el_emp.columns:
                tiers = el_emp["eligibilitytier"].astype(str).str.upper().str.strip()
                sp_mask = tiers.str.contains("EMPFAM|EMPSPOUSE", na=False)
                ch_mask = tiers.str.contains("EMPFAM|EMPCHILD", na=False)
                spouse_eligible = _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=sp_mask)
                child_eligible  = _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=ch_mask)

            # Spouse/Child ENROLLED (full-month at those tiers)
            spouse_enrolled = _tier_full_month(
                en_emp, "enrollmenttier", ("EMPFAM", "EMPSPOUSE"),
                "enrollmentstartdate", "enrollmentenddate", ms, me, require_enrolled=True
            )
            child_enrolled = _tier_full_month(
                en_emp, "enrollmenttier", ("EMPFAM", "EMPCHILD"),
                "enrollmentstartdate", "enrollmentenddate", ms, me, require_enrolled=True
            )

            # Offer spouse/dependents — eligibility OR enrollment
            offer_spouse      = bool(spouse_eligible or spouse_enrolled)
            offer_dependents  = bool(child_eligible  or child_enrolled)

            # Waiting period: ONLY if employed, no eligibility in this month,
            # and the FIRST eligibility starts AFTER this month.
            waiting = False
            if employed and not elig_any and first_elig_start is not None:
                waiting = (first_elig_start > me)

            # Affordability from Eligibility EMP rows overlapping the month
            affordable = False
            if not el_emp.empty and {"eligibilitystartdate","eligibilityenddate","eligibilitytier"} <= set(el_emp.columns):
                overlap = _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me)
                if overlap:
                    mask = el_emp["eligibilitytier"].astype(str).str.upper().str.contains("EMP", na=False)
                    if "plancost" in el_emp.columns:
                        costs = pd.to_numeric(el_emp.loc[mask, "plancost"], errors="coerce")
                        if not costs.dropna().empty:
                            affordable = bool((costs.dropna() < AFFORDABLE_THRESHOLD).any())

            # Optional inference from ENROLLMENT when there is NO eligibility at all
            if ENROLLMENT_IMPLIES_OFFER and (not offer_ee_allmonth) and enrolled_full and (not elig_any):
                offer_ee_allmonth = True
                # ONLY infer MV from enrollment if NO eligibility exists in the month
                if not eligible_mv and "plancode" in en_emp.columns:
                    plan_u2 = en_emp["plancode"].astype(str).str.upper().str.strip()
                    mv_from_enroll = _any_overlap(en_emp, "enrollmentstartdate","enrollmentenddate",
                                                  ms, me, mask=plan_u2.eq("PLANA"))
                    if mv_from_enroll:
                        eligible_mv = True

            # Line 14 / 16
            l14 = _month_line14(eligible_mv, offer_ee_allmonth, offer_spouse, offer_dependents, affordable)
            l16 = _month_line16(employed, enrolled_full, waiting, ft, offer_ee_allmonth, affordable)

            rows.append({
                "EmployeeID": emp,
                "Year": year,
                "MonthNum": m,
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
                "spouse_eligible": bool(spouse_eligible),
                "child_eligible": bool(child_eligible),
                "spouse_enrolled": bool(spouse_enrolled),
                "child_enrolled": bool(child_enrolled),
                "waitingperiod_month": bool(waiting),
                "affordable_plan": bool(affordable),
                "line14_final": l14,
                "line16_final": l16,
                "line14_all12": ""  # filled later if 1G condition met
            })

    interim = pd.DataFrame.from_records(rows)

    # 1G: Not FT any month in year AND enrolled at least one full month → 1G flag, blank line14_final
    for emp, g in interim.groupby("EmployeeID"):
        any_ft = bool(g["ft"].any())
        any_enroll_full = bool(g["enrolled_allmonth"].any())
        if (not any_ft) and any_enroll_full:
            interim.loc[interim["EmployeeID"] == emp, "line14_all12"] = "1G"
            interim.loc[interim["EmployeeID"] == emp, "line14_final"] = ""

    # Ensure boolean-like columns are True/False
    bool_cols = [
        "employed","ft","parttime","eligibleforcoverage","eligible_allmonth","eligible_mv",
        "offer_ee_allmonth","enrolled_allmonth","offer_spouse","offer_dependents",
        "spouse_eligible","child_eligible","spouse_enrolled","child_enrolled",
        "waitingperiod_month","affordable_plan"
    ]
    for c in bool_cols:
        if c in interim.columns:
            interim[c] = interim[c].astype(bool)

    return interim


def build_final(interim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Final table for PDF filling: one row per employee with months.
    We keep the table that downstream expects: Month, Line14_Final, Line16_Final for each employee.
    """
    final_cols = ["EmployeeID","Month","line14_final","line16_final","line14_all12"]
    out = interim_df.loc[:, final_cols].copy()
    out = out.rename(columns={"line14_final":"Line14_Final","line16_final":"Line16_Final","line14_all12":"Line14_All12"})
    return out


# -------------------------
# Penalty Dashboard (light)
# -------------------------

A_PENALTY_MONTHLY = 241.67  # reference/example amount
B_PENALTY_MONTHLY = 362.50  # reference/example amount

def _month_penalty_reason(row: pd.Series) -> Tuple[str, float]:
    """
    Return (reason_html, amount) for a given row of interim.
    This is a simplified heuristic, aligned with your review text.
    """
    l14 = row.get("line14_final", "")
    l16 = row.get("line16_final", "")
    employed = bool(row.get("employed", False))
    affordable = bool(row.get("affordable_plan", False))
    waiting = bool(row.get("waitingperiod_month", False))

    # Enrollment (2C) takes precedence → no penalty
    if l16 == "2C":
        return ("Enrolled all month; no penalty.", 0.0)

    # No full-month offer (1H) while employed → Penalty A
    if l14 == "1H" and employed:
        if waiting:
            msg = ("Penalty A: No MEC offered<br/>"
                   "Employee was not eligible for coverage yet (waiting period) during this month.")
        else:
            msg = ("Penalty A: No MEC offered<br/>"
                   "The employee was not offered minimum essential coverage (MEC) this month.")
        return (msg, A_PENALTY_MONTHLY)

    # Offer made but unaffordable and not enrolled → Penalty B
    if l14 in {"1E","1B","1F","1A"} and l16 not in {"2C","2A"}:
        if not affordable:
            msg = ("Penalty B: Offered but unaffordable & waived<br/>"
                   "The employee was offered coverage but the lowest-cost EMP-only option was unaffordable and they did not enroll.")
            return (msg, B_PENALTY_MONTHLY)

    # Not employed at all
    if l16 == "2A" or not employed:
        return ("Not employed during the month.", 0.0)

    # default
    return ("No penalty.", 0.0)

def build_penalty_dashboard(interim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-employee dashboard with reason text and monthly penalty amounts.
    """
    records: List[Dict[str, Any]] = []
    for emp, g in interim_df.groupby("EmployeeID"):
        # Monthly columns
        monthly_amts = []
        reasons = []
        for m in range(1, 13):
            r = g[g["MonthNum"] == m].iloc[0] if not g[g["MonthNum"] == m].empty else None
            if r is None:
                reasons.append("-")
                monthly_amts.append("-")
                continue
            reason, amt = _month_penalty_reason(r)
            reasons.append(reason)
            monthly_amts.append(("$" + f"{amt:,.2f}") if amt else "-")

        # Try to produce a dominant reason summary (first non-trivial)
        summary_reason = "-"
        for r in reasons:
            if r and r != "-" and "No penalty" not in r and "Enrolled" not in r:
                summary_reason = r
                break

        rec = {"EmployeeID": emp, "Reason": summary_reason}
        for i, mon in enumerate(MONTHS, start=1):
            rec[mon] = monthly_amts[i-1]
        records.append(rec)

    dash = pd.DataFrame.from_records(records, columns=["EmployeeID","Reason"] + MONTHS)
    return dash
