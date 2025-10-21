# aca_builder.py

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict

import pandas as pd
import numpy as np
from datetime import date, timedelta

# --- Constants / knobs -------------------------------------------------------

AFFORDABILITY_THRESHOLD = 50.00  # Simplified (UAT mode)
PENALTY_A_MONTHLY = 241.67       # Example A penalty per month
PENALTY_B_MONTHLY = 362.50       # Example B penalty per month

USE_HTML_BREAKS = True
BR = "<br/>" if USE_HTML_BREAKS else "\n"

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# --- Small utils -------------------------------------------------------------

def _U(s: str) -> str:
    return "".join(ch for ch in str(s).upper() if ch.isalnum())

def _to_date(x) -> Optional[date]:
    if pd.isna(x): return None
    try:
        d = pd.to_datetime(x, errors="coerce")
        if pd.isna(d): return None
        return d.date()
    except Exception:
        return None

def _month_bounds(y: int, m: int) -> Tuple[date, date]:
    start = date(y, m, 1)
    if m == 12:
        end = date(y, 12, 31)
    else:
        end = date(y, m+1, 1) - timedelta(days=1)
    return start, end

def _any_overlap_frame(df: pd.DataFrame, s_col: str, e_col: str, ms: date, me: date, mask=None) -> bool:
    if df is None or df.empty: return False
    m = mask if mask is not None else pd.Series(True, index=df.index)
    ss = pd.to_datetime(df.loc[m, s_col], errors="coerce")
    ee = pd.to_datetime(df.loc[m, e_col], errors="coerce")
    ss = ss.fillna(pd.Timestamp.min).dt.date
    ee = ee.fillna(pd.Timestamp.max).dt.date
    return bool(((ee >= ms) & (ss <= me)).any())

def _full_month_frame(df: pd.DataFrame, s_col: str, e_col: str, ms: date, me: date, mask=None) -> bool:
    if df is None or df.empty: return False
    m = mask if mask is not None else pd.Series(True, index=df.index)
    ss = pd.to_datetime(df.loc[m, s_col], errors="coerce")
    ee = pd.to_datetime(df.loc[m, e_col], errors="coerce")
    ss = ss.fillna(pd.Timestamp.min).dt.date
    ee = ee.fillna(pd.Timestamp.max).dt.date
    return bool(((ss <= ms) & (ee >= me)).any())

def _interval_intersection(a: Tuple[date,date], b: Tuple[date,date]) -> Optional[Tuple[date,date]]:
    (s1, e1), (s2, e2) = a, b
    s = max(s1, s2)
    e = min(e1, e2)
    if s <= e: return (s, e)
    return None

def _merge_intervals(intervals: List[Tuple[date,date]]) -> List[Tuple[date,date]]:
    if not intervals: return []
    xs = sorted(intervals, key=lambda t: t[0])
    out = [xs[0]]
    for s, e in xs[1:]:
        ls, le = out[-1]
        if s <= (le + timedelta(days=1)):
            out[-1] = (ls, max(le, e))
        else:
            out.append((s, e))
    return out

def _union_covers_month(intervals: List[Tuple[date,date]], ms: date, me: date) -> bool:
    if not intervals: return False
    merged = _merge_intervals(intervals)
    cur = ms
    for s, e in merged:
        if s > cur:
            return False  # gap
        cur = max(cur, e)
        if cur >= me:
            return True
        cur = cur + timedelta(days=1)
    return cur >= me

# --- Normalization helpers ---------------------------------------------------

def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common column variants used by uploaded workbooks.
    """
    if df is None or df.empty: return df
    dd = df.copy()
    # eligibility -> expected names
    if "eligibleplan" in dd.columns and "plancode" not in dd.columns:
        dd["plancode"] = dd["eligibleplan"]
    if "eligibletier" in dd.columns and "eligibilitytier" not in dd.columns:
        dd["eligibilitytier"] = dd["eligibletier"]

    # enrollment tier variants
    if "tier" in dd.columns and "enrollmenttier" not in dd.columns:
        dd["enrollmenttier"] = dd["tier"]
    for c in ("plancode", "eligibilitytier", "enrollmenttier", "planname"):
        if c in dd.columns:
            dd[c] = dd[c].astype(str).str.strip()
    return dd

# --- Monthly flags -----------------------------------------------------------

def _is_employed_month(st_df: pd.DataFrame, emp: str, ms: date, me: date) -> bool:
    """
    Employed = the month is fully covered by a non-TERMINATED status row.
    If ANY overlapping row has EmploymentStatus 'Terminated', employed=False for the month.
    """
    if st_df is None or st_df.empty: return False
    ssub = st_df[st_df["employeeid"].astype(str) == str(emp)].copy()
    if ssub.empty: return False

    # Any termination overlapping month? -> whole month false
    if _any_overlap_frame(
        ssub, "statusstartdate", "statusenddate", ms, me,
        mask=ssub["employmentstatus"].astype(str).str.strip().str.upper().eq("TERMINATED")
    ):
        return False

    # Full-month coverage from any non-terminated row?
    mask_non_term = ~ssub["employmentstatus"].astype(str).str.strip().str.upper().eq("TERMINATED")
    return _full_month_frame(ssub, "statusstartdate", "statusenddate", ms, me, mask=mask_non_term)

def _is_role_full_month(st_df: pd.DataFrame, emp: str, ms: date, me: date, role_token: str) -> bool:
    if st_df is None or st_df.empty: return False
    ssub = st_df[st_df["employeeid"].astype(str) == str(emp)].copy()
    if ssub.empty: return False

    # Termination anywhere in month -> False
    if _any_overlap_frame(
        ssub, "statusstartdate", "statusenddate", ms, me,
        mask=ssub["employmentstatus"].astype(str).str.strip().str.upper().eq("TERMINATED")
    ):
        return False

    role_mask = ssub["role"].astype(str).str.strip().str.upper().eq(role_token)
    return _full_month_frame(ssub, "statusstartdate", "statusenddate", ms, me, mask=role_mask)

def _is_ft(st_df, emp, ms, me) -> bool:
    return _is_role_full_month(st_df, emp, ms, me, "FT")

def _is_pt(st_df, emp, ms, me) -> bool:
    return _is_role_full_month(st_df, emp, ms, me, "PT")

# --- Eligibility / Enrollment month logic -----------------------------------

_ALLOWED_TIERS_FOR_MV = {"EMP", "EMPFAM", "EMPCHILD", "EMPSPOUSE"}

def _eligible_mv_full_month(el_df: pd.DataFrame, emp: str, ms: date, me: date) -> bool:
    """
    Eligibility sheet ONLY. PlanA + allowed tiers. Must cover full month.
    """
    if el_df is None or el_df.empty: return False
    el = _apply_aliases(el_df)
    ss = el[el["employeeid"].astype(str) == str(emp)].copy()
    if ss.empty: return False
    tier = ss["eligibilitytier"].astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    plan = ss["plancode"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    mask = plan.eq("PLANA") & tier.isin(_ALLOWED_TIERS_FOR_MV)
    return _full_month_frame(ss, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=mask)

def _offered_allmonth(el_df: pd.DataFrame, emp: str, ms: date, me: date) -> bool:
    """
    Offer to employee all month — we treat any eligibility row that covers
    the full month (any plan) as an offer of MEC to the employee.
    """
    if el_df is None or el_df.empty: return False
    el = _apply_aliases(el_df)
    ss = el[el["employeeid"].astype(str) == str(emp)].copy()
    if ss.empty: return False
    # Any row that covers full month is considered "offer to EE"
    return _full_month_frame(ss, "eligibilitystartdate", "eligibilityenddate", ms, me)

def _enrolled_full_month_union(en_df: pd.DataFrame, emp: str, ms: date, me: date) -> bool:
    """
    Use union of intervals (excluding Waive) to decide if coverage spans whole month.
    """
    if en_df is None or en_df.empty: return False
    en = _apply_aliases(en_df)
    ss = en[en["employeeid"].astype(str) == str(emp)].copy()
    if ss.empty: return False

    # Exclude Waive
    plan = ss["plancode"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    mask = ~plan.eq("WAIVE")
    if not mask.any(): return False

    intervals: List[Tuple[date,date]] = []
    s = pd.to_datetime(ss.loc[mask, "enrollmentstartdate"], errors="coerce").dt.date
    e = pd.to_datetime(ss.loc[mask, "enrollmentenddate"], errors="coerce").dt.date
    for a, b in zip(s, e):
        if a is None or b is None: continue
        it = _interval_intersection((a, b), (ms, me))
        if it: intervals.append(it)
    return _union_covers_month(intervals, ms, me)

def _tier_enrolled_full_month(en_df: pd.DataFrame, emp: str, ms: date, me: date, allowed_tiers: Iterable[str]) -> bool:
    """
    Full-month enrollment in specific tiers (union across rows), excluding Waive.
    """
    if en_df is None or en_df.empty: return False
    en = _apply_aliases(en_df)
    ss = en[en["employeeid"].astype(str) == str(emp)].copy()
    if ss.empty: return False

    tiers = set(t.upper() for t in allowed_tiers)
    plan = ss["plancode"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    tier = ss["enrollmenttier"].astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    mask = ~plan.eq("WAIVE") & tier.isin(tiers)
    if not mask.any(): return False

    intervals: List[Tuple[date,date]] = []
    s = pd.to_datetime(ss.loc[mask, "enrollmentstartdate"], errors="coerce").dt.date
    e = pd.to_datetime(ss.loc[mask, "enrollmentenddate"], errors="coerce").dt.date
    for a, b in zip(s, e):
        if a is None or b is None: continue
        it = _interval_intersection((a, b), (ms, me))
        if it: intervals.append(it)
    return _union_covers_month(intervals, ms, me)

def _tier_offered_any(el_df: pd.DataFrame, emp: str, ms: date, me: date, allowed_tiers: Iterable[str]) -> bool:
    """
    Any overlap in eligibility for allowed tiers (not necessarily full-month).
    """
    if el_df is None or el_df.empty: return False
    el = _apply_aliases(el_df)
    ss = el[el["employeeid"].astype(str) == str(emp)].copy()
    if ss.empty: return False
    tiers = set(t.upper() for t in allowed_tiers)
    tier = ss["eligibilitytier"].astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    mask = tier.isin(tiers)
    return _any_overlap_frame(ss, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=mask)

def _latest_emp_cost_for_month(el_df: pd.DataFrame, emp: str, ms: date, me: date) -> Optional[float]:
    """
    Best-effort: pick the lowest employee-only cost (PlanA, EMP) overlapping the month.
    """
    if el_df is None or el_df.empty: return None
    el = _apply_aliases(el_df)
    ss = el[el["employeeid"].astype(str) == str(emp)].copy()
    if ss.empty: return None

    plan = ss["plancode"].astype(str).str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    tier = ss["eligibilitytier"].astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    cost = pd.to_numeric(ss.get("plancost"), errors="coerce")
    mask = plan.eq("PLANA") & tier.eq("EMP") & cost.notna()

    if not mask.any(): return None
    s_ok = pd.to_datetime(ss.loc[mask, "eligibilitystartdate"], errors="coerce").dt.date
    e_ok = pd.to_datetime(ss.loc[mask, "eligibilityenddate"], errors="coerce").dt.date
    c_ok = cost.loc[mask]

    vals = []
    for a, b, c in zip(s_ok, e_ok, c_ok):
        if a is None or b is None: continue
        if _interval_intersection((a, b), (ms, me)) is not None:
            vals.append(float(c))
    if not vals: return None
    return float(min(vals))

# --- Wait period -------------------------------------------------------------

def _waiting_period_month(wait_df: pd.DataFrame, emp: str, ms: date, me: date) -> bool:
    """
    Overlap with [effective_date, effective_date + wait_days - 1].
    Any overlap marks the month as 'waiting period'.
    """
    if wait_df is None or wait_df.empty: return False
    ss = wait_df[wait_df["employeeid"].astype(str) == str(emp)].copy()
    if ss.empty: return False
    ed = pd.to_datetime(ss.get("effectivedate"), errors="coerce").dt.date
    days = pd.to_numeric(ss.get("wait_days"), errors="coerce").fillna(0).astype(int)

    for a, d in zip(ed, days):
        if a is None: continue
        b = a + timedelta(days=max(0, int(d)) - 1)
        if _interval_intersection((a, b), (ms, me)) is not None:
            return True
    return False

# --- Main builders -----------------------------------------------------------

def build_interim(
    emp_demo: pd.DataFrame,
    emp_status: pd.DataFrame,
    emp_elig: pd.DataFrame,
    emp_enroll: pd.DataFrame,
    dep_enroll: pd.DataFrame,
    *,
    year: int,
    emp_wait: Optional[pd.DataFrame] = None,
    affordability_threshold: float = AFFORDABILITY_THRESHOLD,
) -> pd.DataFrame:

    # Defensive guards to avoid ambiguous truth errors
    if not isinstance(emp_status, pd.DataFrame): emp_status = pd.DataFrame()
    if not isinstance(emp_elig,   pd.DataFrame): emp_elig   = pd.DataFrame()
    if not isinstance(emp_enroll, pd.DataFrame): emp_enroll = pd.DataFrame()
    if not isinstance(dep_enroll, pd.DataFrame): dep_enroll = pd.DataFrame()
    if emp_wait is None or not isinstance(emp_wait, pd.DataFrame): emp_wait = pd.DataFrame()

    # Collect employees
    ids = set()
    for df in (emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll):
        if df is not None and not df.empty and "employeeid" in df.columns:
            ids.update(map(str, df["employeeid"].dropna().astype(str)))
    emps = sorted(ids)

    rows = []
    for emp in emps:
        for m, mname in enumerate(MONTHS, start=1):
            ms, me = _month_bounds(year, m)

            employed = _is_employed_month(emp_status, emp, ms, me)
            ft       = _is_ft(emp_status, emp, ms, me)
            pt       = _is_pt(emp_status, emp, ms, me)

            # eligibility (any/whole)
            elig_any  = _any_overlap_frame(_apply_aliases(emp_elig), "eligibilitystartdate","eligibilityenddate", ms, me) if not emp_elig.empty else False
            elig_full = _full_month_frame(_apply_aliases(emp_elig), "eligibilitystartdate","eligibilityenddate", ms, me) if not emp_elig.empty else False

            # MV (eligibility sheet only)
            eligible_mv = _eligible_mv_full_month(emp_elig, emp, ms, me)

            # Offer to EE = any eligibility covering full month
            offer_ee_allmonth = _offered_allmonth(emp_elig, emp, ms, me)

            # Enrollment full month (union across rows; exclude Waive)
            enrolled_allmonth = _enrolled_full_month_union(emp_enroll, emp, ms, me)

            # Spouse/child eligibility (any overlap)
            spouse_eligible = _tier_offered_any(emp_elig, emp, ms, me, {"EMPFAM","EMPSPOUSE"})
            child_eligible  = _tier_offered_any(emp_elig, emp, ms, me, {"EMPFAM","EMPCHILD"})

            # Spouse/child enrolled must be full-month
            spouse_enrolled = _tier_enrolled_full_month(emp_enroll, emp, ms, me, {"EMPFAM","EMPSPOUSE"})
            child_enrolled  = _tier_enrolled_full_month(emp_enroll, emp, ms, me, {"EMPFAM","EMPCHILD"})

            # Derived offers
            offer_spouse     = bool(spouse_eligible or spouse_enrolled)
            offer_dependents = bool(child_eligible or child_enrolled)

            # Waiting period month
            waiting = _waiting_period_month(emp_wait, emp, ms, me)

            # Affordability: use lowest employee-only PlanA cost overlapping month
            emp_cost = _latest_emp_cost_for_month(emp_elig, emp, ms, me)
            affordable_plan = (emp_cost is not None and float(emp_cost) <= float(affordability_threshold))

            # Line 14 (simplified)
            if offer_ee_allmonth and eligible_mv:
                line14 = "1E"
            else:
                line14 = "1H"

            # Line 16 precedence
            if not employed:
                line16 = "2A"
            elif enrolled_allmonth:
                line16 = "2C"
            elif waiting and not offer_ee_allmonth:
                line16 = "2D"
            elif not ft:
                line16 = "2B"
            else:
                line16 = ""

            rows.append({
                "EmployeeID": emp,
                "Year": year,
                "MonthNum": m,
                "Month": mname,
                "MonthStart": ms,
                "MonthEnd": me,
                "employed": bool(employed),
                "ft": bool(ft),
                "parttime": bool(pt),
                "eligibleforcoverage": bool(elig_any),
                "eligible_allmonth": bool(elig_full),
                "eligible_mv": bool(eligible_mv),
                "offer_ee_allmonth": bool(offer_ee_allmonth),
                "enrolled_allmonth": bool(enrolled_allmonth),
                "offer_spouse": bool(offer_spouse),
                "offer_dependents": bool(offer_dependents),
                "spouse_eligible": bool(spouse_eligible),
                "child_eligible": bool(child_eligible),
                "spouse_enrolled": bool(spouse_enrolled),
                "child_enrolled": bool(child_enrolled),
                "waitingperiod_month": bool(waiting),
                "affordable_plan": bool(affordable_plan),
                "line14_final": line14,
                "line16_final": line16,
                "line14_all12": ""  # we are not auto-issuing 1G
            })

    out = pd.DataFrame.from_records(rows)
    # Keep column order stable
    col_order = [
        "EmployeeID","Year","MonthNum","Month","MonthStart","MonthEnd",
        "employed","ft","parttime",
        "eligibleforcoverage","eligible_allmonth","eligible_mv",
        "offer_ee_allmonth","enrolled_allmonth",
        "offer_spouse","offer_dependents",
        "spouse_eligible","child_eligible","spouse_enrolled","child_enrolled",
        "waitingperiod_month","affordable_plan",
        "line14_final","line16_final","line14_all12"
    ]
    return out.loc[:, col_order]

# --- Final sheet builder -----------------------------------------------------

def build_final(interim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight 'final' — one row per employee per month with final Line 14/16.
    (Your PDF filler expects a Month column with codes.)
    """
    if interim_df is None or interim_df.empty:
        return pd.DataFrame(columns=["EmployeeID","Month","Line14_Final","Line16_Final"])

    final = interim_df.loc[:, ["EmployeeID","Month","line14_final","line16_final"]].copy()
    final = final.rename(columns={"line14_final":"Line14_Final", "line16_final":"Line16_Final"})
    # Ensure Month is in Jan..Dec order per employee
    final["MonthIdx"] = final["Month"].map({m:i for i,m in enumerate(MONTHS,1)})
    final = final.sort_values(["EmployeeID","MonthIdx"]).drop(columns=["MonthIdx"])
    return final

# --- Penalty dashboard -------------------------------------------------------

def _penalty_reason(line14: str, line16: str, affordable: bool, waiting: bool) -> Tuple[str, Optional[float]]:
    """
    Return (reason_html, penalty_amount) where amount is per-month penalty.
    """
    if line14 == "1H":
        # No offer -> Penalty A
        reason = (
            "Penalty A: No MEC offered" + BR +
            "The employee was not offered minimum essential coverage (MEC) during the month."
        )
        return reason, PENALTY_A_MONTHLY

    # Offer exists, check affordability/waive
    if line14 == "1E":
        if line16 == "2C":
            # enrolled all month, no penalty
            return "Enrolled in coverage for the full month.", 0.0
        if waiting and line16 == "2D":
            return ("In waiting period (no full-month offer).", 0.0)
        if not affordable:
            reason = (
                "Penalty B: Waived unaffordable coverage" + BR +
                "Offered MEC but lowest employee-only option was not affordable (cost exceeded threshold)."
            )
            return reason, PENALTY_B_MONTHLY
        # Affordable but not enrolled and not in waiting -> generally no penalty to ALE for affordability safe harbor
        return "Offered affordable coverage; not enrolled.", 0.0

    # Default
    return "No penalty reason detected.", 0.0

def build_penalty_dashboard(interim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide table: one row per employee, 12 month columns with $ amounts and a Reason text.
    """
    if interim_df is None or interim_df.empty:
        return pd.DataFrame(columns=["EmployeeID","Reason"] + MONTHS)

    rows = []
    for emp, g in interim_df.groupby("EmployeeID", sort=False):
        g = g.sort_values("MonthNum")
        month_amounts = []
        reasons = []
        for _, r in g.iterrows():
            reason, amt = _penalty_reason(
                str(r.get("line14_final","")),
                str(r.get("line16_final","")),
                bool(r.get("affordable_plan", False)),
                bool(r.get("waitingperiod_month", False))
            )
            month_amounts.append(0.0 if amt is None else float(amt))
            reasons.append(reason)

        # Prefer the first non-empty reason that actually has a penalty
        reason_final = ""
        for rr, amt in zip(reasons, month_amounts):
            if amt and rr:
                reason_final = rr
                break
        if not reason_final:
            # fallback, first non-empty reason
            for rr in reasons:
                if rr:
                    reason_final = rr; break

        row = {"EmployeeID": emp, "Reason": reason_final}
        for i, m in enumerate(MONTHS):
            val = month_amounts[i]
            row[m] = ("" if val == 0 else f"${val:,.2f}")
        rows.append(row)

    cols = ["EmployeeID","Reason"] + MONTHS
    return pd.DataFrame(rows, columns=cols)
