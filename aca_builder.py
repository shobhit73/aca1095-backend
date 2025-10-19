# aca_builder.py
from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import numpy as np

from aca_processing import (
    MONTHS,
    FULL_MONTHS,
    _collect_employee_ids, _grid_for_year, month_bounds,
    _any_overlap, _all_month,
    _status_from_demographic,
)

# ----------------------------
# Tunables / constants
# ----------------------------
AFFORDABILITY_THRESHOLD = 50.0  # EMP-only monthly employee contribution ($) considered "affordable"
PLAN_A_NAME = "PLANA"           # normalize for comparisons
WAIVE_TOKENS = {"WAIVE", "WAIVED"}

TIER_EMP_TOKENS = {"EMP"}
TIER_SPOUSE_TOKENS = {"EMPSPOUSE"}
TIER_CHILD_TOKENS = {"EMPCHILD"}
TIER_FAMILY_TOKENS = {"EMPFAM"}  # counts for both spouse & child


def _tok(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def _is_waive(plan: str) -> bool:
    return _tok(plan) in WAIVE_TOKENS


def _is_plan_a(plan: str) -> bool:
    return _tok(plan) == PLAN_A_NAME


def _has_emp_tier(tier: str) -> bool:
    return _tok(tier) in TIER_EMP_TOKENS


def _has_spouse_tier(tier: str) -> bool:
    tt = _tok(tier)
    return (tt in TIER_SPOUSE_TOKENS) or (tt in TIER_FAMILY_TOKENS)


def _has_child_tier(tier: str) -> bool:
    tt = _tok(tier)
    return (tt in TIER_CHILD_TOKENS) or (tt in TIER_FAMILY_TOKENS)


def _month_cover_any(df: pd.DataFrame, start_col: str, end_col: str,
                     ms: pd.Timestamp, me: pd.Timestamp, mask: Optional[pd.Series] = None) -> bool:
    """Any overlap in month."""
    if df is None or df.empty:
        return False
    m = mask if mask is not None else pd.Series(True, index=df.index)
    s = pd.to_datetime(df.loc[m, start_col], errors="coerce")
    e = pd.to_datetime(df.loc[m, end_col], errors="coerce")
    return bool(((e >= ms) & (s <= me)).fillna(False).any())


def _month_cover_all(df: pd.DataFrame, start_col: str, end_col: str,
                     ms: pd.Timestamp, me: pd.Timestamp, mask: Optional[pd.Series] = None) -> bool:
    """Covers whole month."""
    if df is None or df.empty:
        return False
    m = mask if mask is not None else pd.Series(True, index=df.index)
    s = pd.to_datetime(df.loc[m, start_col], errors="coerce")
    e = pd.to_datetime(df.loc[m, end_col], errors="coerce")
    return bool(((s <= ms) & (e >= me)).fillna(False).any())


def _employment_flags_for_month(emp_status: pd.DataFrame,
                                ms: pd.Timestamp, me: pd.Timestamp) -> Tuple[bool, bool, bool]:
    """
    Returns (employed, ft, pt) for month.
    - 'employed' if any status overlaps and not 'Terminated'
    - 'ft' if any overlapping status has role 'FT'
    - 'pt' if any overlapping status has role 'PT'
    """
    if emp_status is None or emp_status.empty:
        return False, False, False
    s = pd.to_datetime(emp_status.get("statusstartdate"), errors="coerce")
    e = pd.to_datetime(emp_status.get("statusenddate"), errors="coerce")
    overlaps = ((e >= ms) & (s <= me)).fillna(False)

    if not overlaps.any():
        return False, False, False

    active_mask = overlaps & (emp_status.get("employmentstatus", "").astype(str).str.upper() != "TERMINATED")
    if not active_mask.any():
        return False, False, False

    roles = emp_status.loc[active_mask, "role"].astype(str).str.upper()
    ft = roles.str.contains(r"\bFT\b", regex=True).any()
    pt = roles.str.contains(r"\bPT\b", regex=True).any()
    return True, bool(ft), bool(pt)


def _eligible_any(emp_elig: pd.DataFrame, ms: pd.Timestamp, me: pd.Timestamp) -> bool:
    return _month_cover_any(emp_elig, "eligibilitystartdate", "eligibilityenddate", ms, me)


def _eligible_all(emp_elig: pd.DataFrame, ms: pd.Timestamp, me: pd.Timestamp) -> bool:
    return _month_cover_all(emp_elig, "eligibilitystartdate", "eligibilityenddate", ms, me)


def _enrolled_all(emp_enroll: pd.DataFrame, ms: pd.Timestamp, me: pd.Timestamp) -> bool:
    if emp_enroll is None or emp_enroll.empty:
        return False
    m = ~emp_enroll["plancode"].astype(str).str.strip().str.upper().isin(WAIVE_TOKENS)
    return _month_cover_all(emp_enroll, "enrollmentstartdate", "enrollmentenddate", ms, me, mask=m)


def _elig_mv_for_month(emp_elig: pd.DataFrame, emp_enroll: pd.DataFrame,
                       ms: pd.Timestamp, me: pd.Timestamp) -> bool:
    """
    Eligible MV for month if:
      - eligibility includes PlanA overlapping the month, OR
      - enrollment includes PlanA overlapping the month (even if eligibility shows PlanB)
    """
    ok_e = False
    if emp_elig is not None and not emp_elig.empty:
        mask_plan_a = emp_elig["plancode"].astype(str).str.upper().eq(PLAN_A_NAME)
        ok_e = _month_cover_any(emp_elig, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=mask_plan_a)

    ok_en = False
    if emp_enroll is not None and not emp_enroll.empty:
        mask_en_plan_a = emp_enroll["plancode"].astype(str).str.upper().eq(PLAN_A_NAME)
        ok_en = _month_cover_any(emp_enroll, "enrollmentstartdate", "enrollmentenddate", ms, me, mask=mask_en_plan_a)

    return bool(ok_e or ok_en)


def _affordable_for_month(emp_elig: pd.DataFrame, ms: pd.Timestamp, me: pd.Timestamp) -> bool:
    """
    Affordability test for month: any PlanA + EMP tier row overlapping the month
    with plancost <= AFFORDABILITY_THRESHOLD.
    """
    if emp_elig is None or emp_elig.empty:
        return False
    df = emp_elig.copy()
    df["_plana_emp"] = (df["plancode"].astype(str).str.upper().eq(PLAN_A_NAME)) & \
                       (df["eligibilitytier"].astype(str).str.upper().isin(TIER_EMP_TOKENS))
    m = df["_plana_emp"]
    if not m.any():
        return False
    if not _month_cover_any(df, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=m):
        return False
    # among overlapping rows, check cost
    s = pd.to_datetime(df.loc[m, "eligibilitystartdate"], errors="coerce")
    e = pd.to_datetime(df.loc[m, "eligibilityenddate"], errors="coerce")
    overlapping = df.loc[m & (e >= ms) & (s <= me)]
    return bool((overlapping["plancost"].astype(float) <= AFFORDABILITY_THRESHOLD).any())


def _spouse_child_flags_from_elig(emp_elig: pd.DataFrame, ms: pd.Timestamp, me: pd.Timestamp) -> Tuple[bool, bool]:
    spouse, child = False, False
    if emp_elig is None or emp_elig.empty:
        return spouse, child
    df = emp_elig.copy()
    # any overlap in month counts
    s = pd.to_datetime(df["eligibilitystartdate"], errors="coerce")
    e = pd.to_datetime(df["eligibilityenddate"], errors="coerce")
    ov = (e >= ms) & (s <= me)
    if ov.any():
        tiers = df.loc[ov, "eligibilitytier"].astype(str).str.upper()
        spouse = tiers.map(_has_spouse_tier).any()
        child = tiers.map(_has_child_tier).any()
    return bool(spouse), bool(child)


def _spouse_child_flags_from_enroll(emp_enroll: pd.DataFrame, ms: pd.Timestamp, me: pd.Timestamp) -> Tuple[bool, bool]:
    spouse, child = False, False
    if emp_enroll is None or emp_enroll.empty:
        return spouse, child
    df = emp_enroll.copy()
    # exclude Waive
    df = df[~df["plancode"].astype(str).str.strip().str.upper().isin(WAIVE_TOKENS)]
    if df.empty:
        return spouse, child
    s = pd.to_datetime(df["enrollmentstartdate"], errors="coerce")
    e = pd.to_datetime(df["enrollmentenddate"], errors="coerce")
    ov = (e >= ms) & (s <= me)
    if ov.any():
        tiers = df.loc[ov, "tier"].astype(str).str.upper() if "tier" in df.columns else df.loc[ov, "enrollmenttier"].astype(str).str.upper()
        spouse = tiers.map(_has_spouse_tier).any()
        child = tiers.map(_has_child_tier).any()
    return bool(spouse), bool(child)


def _enrolled_all_month_nonwaive(emp_enroll: pd.DataFrame, ms: pd.Timestamp, me: pd.Timestamp) -> bool:
    return _enrolled_all(emp_enroll, ms, me)


def _offered_all_month(emp_elig: pd.DataFrame, emp_enroll: pd.DataFrame,
                       ms: pd.Timestamp, me: pd.Timestamp) -> bool:
    """
    We consider 'offer_ee_allmonth' True if either:
      - eligibility covers the whole month (any plan), OR
      - enrollment in a non-waive plan covers the whole month.
    """
    elig_all = _eligible_all(emp_elig, ms, me)
    enr_all  = _enrolled_all_month_nonwaive(emp_enroll, ms, me)
    return bool(elig_all or enr_all)


def _first_ft_start(emp_status: pd.DataFrame) -> Optional[pd.Timestamp]:
    if emp_status is None or emp_status.empty:
        return None
    ft_rows = emp_status[emp_status["role"].astype(str).str.upper().eq("FT")]
    if ft_rows.empty:
        return None
    s = pd.to_datetime(ft_rows["statusstartdate"], errors="coerce")
    if s.isna().all():
        return None
    return s.min()


def _in_wait_period_for_month(emp_status: pd.DataFrame, emp_elig: pd.DataFrame,
                              ms: pd.Timestamp, me: pd.Timestamp) -> bool:
    """
    Simple wait-period marker:
    - If FT employment began recently and the employee is not yet eligible in this month,
      mark as wait period. We use a 3-calendar-month window from first FT start.
    """
    ft_start = _first_ft_start(emp_status)
    if ft_start is None:
        return False
    # If already eligible overlapping this month, not wait-period
    if _eligible_any(emp_elig, ms, me):
        return False
    # 3 calendar months from FT start
    end_wait = (ft_start + pd.DateOffset(months=3)).normalize() - pd.DateOffset(days=1)
    return bool(ms <= end_wait)


def _line_codes_for_month(employed: bool, ft: bool,
                          eligible_mv: bool, offer_ee_allmonth: bool,
                          enrolled_allmonth: bool, affordable: bool,
                          offer_spouse: bool, offer_dep: bool) -> Tuple[str, str]:
    """
    Returns (line14, line16) for a single month.
    Mirrors examples you shared.
    """
    # Not employed → 1H / 2A
    if not employed:
        return "1H", "2A"

    # PT and no offer → 1H / 2D
    if (not ft) and (not offer_ee_allmonth):
        return "1H", "2D"

    # Enrolled all month
    if enrolled_allmonth:
        # Qualifying offer if affordable + spouse + dependents offered
        if affordable and offer_spouse and offer_dep:
            return "1A", "2C"
        # Otherwise standard MV offer
        if eligible_mv or offer_ee_allmonth:
            return "1E", "2C"
        # fallback
        return "1H", "2C"

    # Waived/Not enrolled but offered
    if offer_ee_allmonth:
        if affordable and offer_spouse and offer_dep:
            return "1A", "2H"
        if eligible_mv:
            return "1E", "2H"
        return "1H", "2H"

    # Default: no offer while employed
    return "1H", "2D"


def build_interim(emp_demo: pd.DataFrame,
                  emp_status: pd.DataFrame,
                  emp_elig: pd.DataFrame,
                  emp_enroll: pd.DataFrame,
                  dep_enroll: pd.DataFrame,
                  year: int) -> pd.DataFrame:
    """
    Build the per-employee per-month interim table with all flags and line codes.
    """
    # Fallback status from demographic if separate Emp Status is missing
    if emp_status is None or emp_status.empty:
        emp_status = _status_from_demographic(emp_demo)

    # Normalize some expected columns on enrollment
    enr = emp_enroll.copy()
    if "tier" not in enr.columns and "enrollmenttier" in enr.columns:
        enr["tier"] = enr["enrollmenttier"]
    for c in ("plancode", "tier"):
        if c in enr.columns:
            enr[c] = enr[c].astype(str).str.strip()

    elig = emp_elig.copy()
    for c in ("plancode", "eligibilitytier"):
        if c in elig.columns:
            elig[c] = elig[c].astype(str).str.strip()

    # Universe of employees found anywhere
    employee_ids = _collect_employee_ids(emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll)
    grid = _grid_for_year(employee_ids, year)

    out_rows: List[Dict[str, Any]] = []

    # Pre-slice status/elig/enroll by employee for speed
    status_by_emp = {}
    elig_by_emp = {}
    enr_by_emp = {}

    if not emp_status.empty:
        for eid in employee_ids:
            status_by_emp[eid] = emp_status[emp_status["employeeid"].astype(str) == str(eid)].copy()
    else:
        status_by_emp = {eid: pd.DataFrame() for eid in employee_ids}

    if not elig.empty:
        for eid in employee_ids:
            elig_by_emp[eid] = elig[elig["employeeid"].astype(str) == str(eid)].copy()
    else:
        elig_by_emp = {eid: pd.DataFrame() for eid in employee_ids}

    if not enr.empty:
        for eid in employee_ids:
            enr_by_emp[eid] = enr[enr["employeeid"].astype(str) == str(eid)].copy()
    else:
        enr_by_emp = {eid: pd.DataFrame() for eid in employee_ids}

    # Determine if employee is FT in ANY month of the year (for 1G)
    ft_any_by_emp: Dict[str, bool] = {}
    for eid in employee_ids:
        st = status_by_emp.get(eid, pd.DataFrame())
        ft_any = False
        if not st.empty:
            # If any FT status overlaps any day in the report year
            s = pd.to_datetime(st["statusstartdate"], errors="coerce")
            e = pd.to_datetime(st["statusenddate"], errors="coerce")
            # Overlap with [Jan 1, Dec 31]
            yr_start = pd.Timestamp(year=year, month=1, day=1)
            yr_end = pd.Timestamp(year=year, month=12, day=31)
            ft_any = ((e >= yr_start) & (s <= yr_end) & (st["role"].astype(str).str.upper().eq("FT"))).any()
        ft_any_by_emp[eid] = bool(ft_any)

    # Build rows
    for _, row in grid.iterrows():
        eid = str(row["employeeid"])
        ms = pd.Timestamp(row["monthstart"])
        me = pd.Timestamp(row["monthend"])

        st = status_by_emp.get(eid, pd.DataFrame())
        el = elig_by_emp.get(eid, pd.DataFrame())
        en = enr_by_emp.get(eid, pd.DataFrame())

        employed, ft, pt = _employment_flags_for_month(st, ms, me)

        eligibleforcoverage = _eligible_any(el, ms, me)
        eligible_allmonth = _eligible_all(el, ms, me)
        enrolled_allmonth = _enrolled_all_month_nonwaive(en, ms, me)
        offer_ee_allmonth = _offered_all_month(el, en, ms, me)

        # spouse/child flags (monthly)
        sp_elig, ch_elig = _spouse_child_flags_from_elig(el, ms, me)
        sp_enr, ch_enr = _spouse_child_flags_from_enroll(en, ms, me)
        offer_spouse = bool(sp_elig or sp_enr)
        offer_dependents = bool(ch_elig or ch_enr)

        # eligible_mv (PlanA eligibility or enrollment in month)
        eligible_mv = _elig_mv_for_month(el, en, ms, me)

        # Affordable plan (PlanA + EMP <= threshold) in month
        affordable_plan = _affordable_for_month(el, ms, me)

        # wait-period marker
        waitingperiod_month = _in_wait_period_for_month(st, el, ms, me)

        # Line codes unless 1G special (computed later)
        line14, line16 = _line_codes_for_month(
            employed, ft, eligible_mv, offer_ee_allmonth, enrolled_allmonth, affordable_plan,
            offer_spouse, offer_dependents
        )

        out_rows.append({
            "EmployeeID": eid,
            "Year": row["year"],
            "MonthNum": row["monthnum"],
            "Month": row["month"],
            "MonthStart": ms.date(),
            "MonthEnd": me.date(),
            "employed": employed,
            "ft": ft,
            "parttime": pt,
            "eligibleforcoverage": eligibleforcoverage,
            "eligible_allmonth": eligible_allmonth,
            "eligible_mv": eligible_mv,
            "offer_ee_allmonth": offer_ee_allmonth,
            "enrolled_allmonth": enrolled_allmonth,
            "offer_spouse": offer_spouse,
            "offer_dependents": offer_dependents,
            "spouse_eligible": sp_elig,
            "child_eligible": ch_elig,
            "spouse_enrolled": sp_enr,
            "child_enrolled": ch_enr,
            "waitingperiod_month": waitingperiod_month,
            "affordable_plan": affordable_plan,
            "line14_final": line14,
            "line16_final": line16,
        })

    interim = pd.DataFrame(out_rows)

    # ----------------
    # 1G special rule:
    # If employee is NOT FT in any month of the year and is enrolled at least one full month,
    # then Line 14 is "1G" for all 12 months (we surface it in 'line14_all12')
    # and keep 'line14_final' blank.
    # ----------------
    line14_all12 = []
    for eid in interim["EmployeeID"].unique():
        df_e = interim[interim["EmployeeID"] == eid].sort_values("MonthNum")
        not_ft_any = not ft_any_by_emp.get(eid, False)
        ever_enrolled_full = bool(df_e["enrolled_allmonth"].any())
        flag_1g = not_ft_any and ever_enrolled_full
        line14_all12.extend(["1G" if flag_1g else "" for _ in range(len(df_e))])

        if flag_1g:
            # Blank line14_final for that employee
            interim.loc[df_e.index, "line14_final"] = ""

    interim["line14_all12"] = line14_all12

    return interim


def build_final(interim: pd.DataFrame) -> pd.DataFrame:
    """
    Final table (per Employee × Month) with only the IRS output columns required for PDF filling.
    """
    keep = ["EmployeeID", "Month", "line14_final", "line16_final", "line14_all12"]
    rows = []
    for eid in interim["EmployeeID"].unique():
        sub = interim[interim["EmployeeID"] == eid].sort_values("MonthNum")
        for m in MONTHS:
            r = sub[sub["Month"] == m]
            if r.empty:
                rows.append({"EmployeeID": eid, "Month": m, "Line14_Final": "", "Line16_Final": ""})
            else:
                rows.append({
                    "EmployeeID": eid,
                    "Month": m,
                    "Line14_Final": str(r["line14_final"].iloc[0] or ""),
                    "Line16_Final": str(r["line16_final"].iloc[0] or ""),
                    "line14_all12": str(r["line14_all12"].iloc[0] or ""),
                })
    final = pd.DataFrame(rows)
    return final


# ----------------------------
# Penalty Dashboard
# ----------------------------
PENALTY_A_PER_MONTH = 241.67  # Illustrative
PENALTY_B_PER_MONTH = 362.50  # Illustrative


def _reason_no_mec(df_row: pd.Series) -> str:
    # Detailed reason for "No MEC offered"
    msgs = []
    if df_row.get("waitingperiod_month", False):
        msgs.append("Employee was in a waiting period this month.")
    if not df_row.get("employed", False):
        msgs.append("Employee was not employed during this month.")
    if df_row.get("parttime", False) and not df_row.get("offer_ee_allmonth", False):
        msgs.append("Employee was part-time and no coverage was offered.")
    base = "Penalty A: No MEC offered"
    if msgs:
        base += " <br/> " + " ".join(msgs)
    else:
        base += " <br/> The employee was not offered minimum essential coverage (MEC) in this month."
    return base


def _reason_unaffordable_waive() -> str:
    return ("Penalty B: Waived Unaffordable Coverage <br/> "
            "The employee was offered minimum essential coverage (MEC), but the lowest-cost "
            "employee-only option was not affordable (cost exceeded the threshold). "
            "The employee waived the offer.")


def build_penalty_dashboard(interim: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a wide table (Employee × Months) with amounts and a Reason column.
    A (no MEC), B (unaffordable + waived).
    """
    cols = ["EmployeeID", "Reason"] + MONTHS
    rows: List[Dict[str, Any]] = []

    for eid in sorted(interim["EmployeeID"].unique(), key=lambda x: str(x)):
        sub = interim[interim["EmployeeID"] == eid].sort_values("MonthNum")

        # Track per-month penalties
        amounts = {m: "" for m in MONTHS}
        reason = ""

        for _, r in sub.iterrows():
            m = r["Month"]

            # Penalty A — No MEC offered (employee is employed, and no offer)
            if r["employed"] and (not r["offer_ee_allmonth"]):
                amounts[m] = f"${PENALTY_A_PER_MONTH:,.2f}"
                # Collect reasons (first non-empty reason used)
                if not reason:
                    reason = _reason_no_mec(r)

            # Penalty B — Waived unaffordable (offered but not enrolled, not affordable)
            elif r["offer_ee_allmonth"] and (not r["enrolled_allmonth"]) and (not r["affordable_plan"]):
                amounts[m] = f"${PENALTY_B_PER_MONTH:,.2f}"
                if not reason:
                    reason = _reason_unaffordable_waive()

        if any(v for v in amounts.values()):
            rows.append({"EmployeeID": eid, "Reason": reason, **amounts})

    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(rows, columns=cols)
    return out
