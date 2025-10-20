# aca_builder.py
# Build a rich Interim grid (employee x month) + Final + Penalty dashboard

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple

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
AFFORDABILITY_THRESHOLD = 50.00  # < $50 => affordable   (change to <= if you want 50 to count)

# Penalty amounts (per month)
PENALTY_A_MONTHLY = 241.67  # No MEC offered
PENALTY_B_MONTHLY = 362.50  # Waived unaffordable offer

# Reason cell line breaks:
# False => newline (\n) for Excel (wrap text in cell)
# True  => <br> for HTML rendering (web)
USE_HTML_BREAKS = False
BR = "<br>" if USE_HTML_BREAKS else "\n"


# ------------------------------------------------------------
# Column alias helpers (robust to EligiblePlan/EligibleTier headers)
# ------------------------------------------------------------
def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    cols = set(df.columns)

    # EligiblePlan -> plancode
    if "eligibleplan" in cols and "plancode" not in cols:
        df["plancode"] = df["eligibleplan"].astype(str).str.strip()

    # EligibleTier -> eligibilitytier
    if "eligibletier" in cols and "eligibilitytier" not in cols:
        df["eligibilitytier"] = df["eligibletier"].astype(str).str.strip()

    return df


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _latest_emp_cost_for_month(el_df: pd.DataFrame, ms, me) -> Optional[float]:
    """
    Returns the employee-only (EMP) plan cost from Emp Eligibility that overlaps the month,
    choosing the row with the latest eligibility end date.
    """
    if el_df is None or el_df.empty:
        return None
    need = {"eligibilitystartdate","eligibilityenddate","eligibilitytier"}
    if not need <= set(el_df.columns):
        return None

    df = el_df[
        (el_df["eligibilityenddate"].fillna(pd.Timestamp.max).dt.date >= ms) &
        (el_df["eligibilitystartdate"].fillna(pd.Timestamp.min).dt.date <= me)
    ]
    if df.empty or "plancost" not in df.columns:
        return None

    tier_u = df["eligibilitytier"].astype(str).str.upper().str.strip()
    df = df[tier_u.eq("EMP")]
    if df.empty:
        return None

    df = df.sort_values("eligibilityenddate", ascending=False)
    v = pd.to_numeric(df.iloc[0]["plancost"], errors="coerce")
    return float(v) if not pd.isna(v) else None


def _offered_allmonth(el_emp: pd.DataFrame, ms, me) -> bool:
    """Employee-level MEC offer for full month (any tier that includes employee)."""
    if el_emp.empty or "eligibilitytier" not in el_emp.columns:
        return False
    tiers = el_emp["eligibilitytier"].astype(str).str.upper().str.strip()
    mask = tiers.str.contains("EMP", na=False)  # any EMP* tier implies employee offer
    return _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=mask)


def _tier_offered_any(
    df: pd.DataFrame,
    tier_col: str,
    tokens: Tuple[str, ...],
    start_col: str,
    end_col: str,
    ms, me,
    *,
    require_enrolled: bool = False,   # when True, also require isenrolled True if present AND exclude Waive
) -> bool:
    """
    TRUE if any row's <tier_col> contains any 'tokens' and date overlaps the month.
    String matching is case-insensitive substring match.

    If require_enrolled=True (for Enrollment checks):
      - honors 'isenrolled' if present
      - excludes rows where plancode/planname == 'WAIVE'
    """
    if df.empty or tier_col not in df.columns:
        return False

    # base mask
    mask = pd.Series(True, index=df.index)

    if require_enrolled and "isenrolled" in df.columns:
        mask &= df["isenrolled"].astype(bool)

    # exclude WAIVE rows from enrollment-based checks
    if require_enrolled:
        waive_mask = pd.Series(False, index=df.index)
        for col in ("plancode", "planname"):
            if col in df.columns:
                s = df[col].astype(str).str.upper().str.strip()
                waive_mask |= s.eq("WAIVE")
        mask &= ~waive_mask

    tiers = df[tier_col].astype(str).str.upper().str.strip()
    tok_mask = pd.Series(False, index=df.index)
    for t in tokens:
        tok_mask |= tiers.str.contains(t, na=False)
    mask &= tok_mask

    return _any_overlap(df, start_col, end_col, ms, me, mask=mask)


# ------------------------------------------------------------
# Status helpers (FT/PT/employed)
# ------------------------------------------------------------
def _is_employed_month(st_emp: pd.DataFrame, ms, me) -> bool:
    """
    Employed logic (per your spec):
      - If NO status row overlaps this month → employed = False.
      - If ANY overlapping row has EmploymentStatus = Terminated → employed = False for the whole month.
      - Otherwise → employed = True.

    This ignores FT/PT (handled separately).
    """
    if st_emp.empty:
        return False

    # Filter to rows that overlap the month at all
    overlaps = (
        st_emp["statusenddate"].fillna(pd.Timestamp.max).dt.date >= ms
    ) & (
        st_emp["statusstartdate"].fillna(pd.Timestamp.min).dt.date <= me
    )
    if not overlaps.any():
        return False

    # If we have normalized EmploymentStatus, check for any 'Terminated' among overlapping rows
    if "_estatus_norm" in st_emp.columns:
        s = st_emp.loc[overlaps, "_estatus_norm"].astype(str)
        # match 'TERMINATED' or shorthand 'TERM'
        any_term = s.str.contains("TERMINAT", na=False) | s.str.fullmatch("TERM", na=False)
        if any_term.any():
            return False

    # Otherwise (or if no termination among overlaps), considered employed
    return True


def _is_ft(st_emp: pd.DataFrame, ms, me) -> bool:
    """
    Full-time only if:
      - there is NO 'Terminated' status overlapping the month, and
      - Role shows FT (or FULLTIME) covering the ENTIRE month.
    """
    if st_emp.empty or "_role_norm" not in st_emp.columns:
        return False

    # If any overlapping row is Terminated, FT must be False for the month
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
      - Role shows PT (or PARTTIME) covering the ENTIRE month.
    """
    if st_emp.empty or "_role_norm" not in st_emp.columns:
        return False

    # If any overlapping row is Terminated, PT must be False for the month
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
    Simplified mapping:
      - If no full-month offer to employee → 1H
      - If MV offered (PlanA) full-month:
           if spouse+dependents also offered → 1A if affordable else 1E
           else → 1E
      - If no MV but full-month MEC offer → 1F
    """
    if not offer_ee_allmonth:
        return "1H"
    if eligible_mv:
        if offer_spouse and offer_dependents:
            return "1A" if affordable else "1E"
        return "1E"
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
    Practical precedence:
      2A: not employed any day this month
      2C: enrolled in coverage for the entire month
      2D: limited non-assessment period / waiting
      2B: not full-time for the month
      2H: rate-of-pay safe harbor (use affordable flag)
      else: blank
    """
    if not employed:
        return "2A"
    if enrolled_full:
        return "2C"
    if waiting:
        return "2D"
    if not ft:
        return "2B"
    if offer_ee_allmonth and affordable:
        return "2H"
    return ""


# ------------------------------------------------------------
# Public: build_interim / build_final / build_penalty_dashboard
# ------------------------------------------------------------
def build_interim(
    emp_demo: pd.DataFrame,
    emp_status: pd.DataFrame,
    emp_elig: pd.DataFrame,
    emp_enroll: pd.DataFrame,
    dep_enroll: pd.DataFrame,
    year: int,
    **_kwargs,   # absorbs extra args like pay_deductions
) -> pd.DataFrame:
    """
    Returns a rich employee x month Interim with status/offer/enrollment/affordability flags
    and preliminary Line 14/16.

    NOTE: eligible_mv is TRUE if PlanA is present EITHER in Eligibility OR in Enrollment
    for the month (any overlap in the month).

    Spouse/child logic:
      spouse_eligible  : Eligibility. eligibilitytier has EMPFAM or EMPSPOUSE (any overlap)
      child_eligible   : Eligibility. eligibilitytier has EMPFAM or EMPCHILD  (any overlap)
      spouse_enrolled  : Enrollment.  enrollmenttier has EMPFAM or EMPSPOUSE (any overlap, excluding WAIVE; honors isenrolled)
      child_enrolled   : Enrollment.  enrollmenttier has EMPFAM or EMPCHILD  (any overlap, excluding WAIVE; honors isenrolled)
      offer_spouse     : spouse_eligible OR spouse_enrolled
      offer_dependents : child_eligible  OR child_enrolled
    """

    # Defensive aliasing for alternate headers
    emp_elig  = _apply_aliases(emp_elig)
    emp_enroll= _apply_aliases(emp_enroll)
    dep_enroll= _apply_aliases(dep_enroll)

    # Normalize date columns if caller bypassed prepare_inputs
    for df, sc, ec in [
        (emp_elig,  "eligibilitystartdate","eligibilityenddate"),
        (emp_enroll,"enrollmentstartdate","enrollmentenddate"),
        (dep_enroll,"eligiblestartdate","eligibleenddate"),
    ]:
        if not df.empty:
            for c in (sc, ec):
                if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
                    df[c] = pd.to_datetime(df[c], errors="coerce")

    # Status table (fallback to demographic if Emp Status is missing)
    st = emp_status if emp_status is not None else pd.DataFrame()
    if st.empty:
        st = _status_from_demographic(emp_demo)

    # Build monthly rows
    employee_ids = _collect_employee_ids(emp_demo, st, emp_elig, emp_enroll, dep_enroll)
    rows: List[Dict[str, Any]] = []

    for emp in employee_ids:
        el_emp = emp_elig[emp_elig["employeeid"].astype(str)  == str(emp)].copy() if not emp_elig.empty  else pd.DataFrame()
        en_emp = emp_enroll[emp_enroll["employeeid"].astype(str)== str(emp)].copy() if not emp_enroll.empty else pd.DataFrame()
        de_emp = dep_enroll[dep_enroll["employeeid"].astype(str)== str(emp)].copy() if not dep_enroll.empty else pd.DataFrame()
        st_emp = st[st["employeeid"].astype(str) == str(emp)].copy() if not st.empty else pd.DataFrame()

        for m in range(1, 12 + 1):
            ms, me = month_bounds(year, m)

            # ---- status flags
            employed   = _is_employed_month(st_emp, ms, me)
            ft         = _is_ft(st_emp, ms, me)
            parttime   = (not ft) and _is_pt(st_emp, ms, me)

            # ---- eligibility / offer flags
            elig_any   = _any_overlap(el_emp, "eligibilitystartdate","eligibilityenddate", ms, me) if not el_emp.empty else False
            elig_full  = _all_month(el_emp,  "eligibilitystartdate","eligibilityenddate", ms, me) if not el_emp.empty else False

            # eligible_mv: TRUE if PlanA appears in Eligibility OR Enrollment for the month
            eligible_mv = False
            if "plancode" in el_emp.columns:
                plan_u = el_emp["plancode"].astype(str).str.upper().str.strip()
                eligible_mv |= _any_overlap(
                    el_emp, "eligibilitystartdate","eligibilityenddate", ms, me, mask=plan_u.eq("PLANA")
                )
            if "plancode" in en_emp.columns:
                plan_u2 = en_emp["plancode"].astype(str).str.upper().str.strip()
                eligible_mv |= _any_overlap(
                    en_emp, "enrollmentstartdate","enrollmentenddate", ms, me, mask=plan_u2.eq("PLANA")
                )

            offer_ee_allmonth  = _offered_allmonth(el_emp, ms, me)

            # ---- enrollment flags (for full-month enrollment, excluding "Waive")
            enrolled_full = False
            if not en_emp.empty:
                mask_en = pd.Series(True, index=en_emp.index)
                if "isenrolled" in en_emp.columns:
                    mask_en &= en_emp["isenrolled"].astype(bool)
                waive_mask = pd.Series(False, index=en_emp.index)
                for col in ("plancode", "planname"):
                    if col in en_emp.columns:
                        s = en_emp[col].astype(str).str.upper().str.strip()
                        waive_mask |= s.eq("WAIVE")
                mask_en &= ~waive_mask
                enrolled_full = _all_month(
                    en_emp, "enrollmentstartdate","enrollmentenddate", ms, me, mask=mask_en
                )

            # ---- spouse/child eligibility from Emp Eligibility tiers
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

            # ---- spouse/child enrollment from Emp Enrollment tiers (exclude WAIVE, honor isenrolled)
            spouse_enrolled = _tier_offered_any(
                en_emp, "enrollmenttier", ("EMPFAM", "EMPSPOUSE"),
                "enrollmentstartdate", "enrollmentenddate", ms, me,
                require_enrolled=True
            )
            child_enrolled = _tier_offered_any(
                en_emp, "enrollmenttier", ("EMPFAM", "EMPCHILD"),
                "enrollmentstartdate", "enrollmentenddate", ms, me,
                require_enrolled=True
            )

            # ---- final dependent/spouse offer flags (eligibility OR enrollment)
            offer_spouse     = spouse_eligible or spouse_enrolled
            offer_dependents = child_eligible  or child_enrolled

            # ---- affordability
            emp_cost = _latest_emp_cost_for_month(el_emp, ms, me)
            affordable = (emp_cost is not None) and (emp_cost < AFFORDABILITY_THRESHOLD)
            # To count $50 as affordable, change the comparator above to <=

            # ---- waiting period (simple heuristic)
            waiting = False
            if employed and not elig_any and not el_emp.empty and "eligibilitystartdate" in el_emp.columns:
                future_starts = el_emp["eligibilitystartdate"].dropna()
                waiting = (future_starts.dt.date > me).any()

            # ---- Line 14/16 (per month)
            l14 = _month_line14(eligible_mv, offer_ee_allmonth, offer_spouse, offer_dependents, affordable)
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
                "MonthStart": pd.Timestamp(ms),
                "MonthEnd": pd.Timestamp(me),

                "employed": bool(employed),
                "ft": bool(ft),
                "parttime": bool(parttime),

                "eligibleforcoverage": bool(elig_any),
                "eligible_allmonth": bool(elig_full),
                "eligible_mv": bool(eligible_mv),

                "offer_ee_allmonth": bool(offer_ee_allmonth),
                "enrolled_allmonth": bool(enrolled_full),

                # spouse/child flags per your rules
                "spouse_eligible": bool(spouse_eligible),
                "child_eligible": bool(child_eligible),
                "spouse_enrolled": bool(spouse_enrolled),
                "child_enrolled": bool(child_enrolled),
                "offer_spouse": bool(offer_spouse),
                "offer_dependents": bool(offer_dependents),

                "waitingperiod_month": bool(waiting),
                "affordable_plan": bool(affordable),

                "line14_final": l14,
                "line16_final": l16,
            })

    interim = pd.DataFrame(rows)

    # Year-level '1G' code:
    #  Report 1G only if:
    #    - Employee was NOT full-time in ANY month (never FT all year), AND
    #    - They were enrolled in the employer's plan for AT LEAST ONE FULL MONTH.
    if not interim.empty:
        ft_by_emp = interim.groupby("EmployeeID")["ft"].sum(min_count=0)              # how many months FT
        any_enrolled_full = interim.groupby("EmployeeID")["enrolled_allmonth"].any()  # any full-month enrollment

        def code_1g(emp_id: str) -> str:
            return "1G" if (ft_by_emp.get(emp_id, 0) == 0 and bool(any_enrolled_full.get(emp_id, False))) else ""

        interim["line14_all12"] = interim["EmployeeID"].map(code_1g)

        # If employee is 1G, blank out all monthly Line 14 codes
        if "line14_final" in interim.columns:
            one_g_emp_ids = set(interim.loc[interim["line14_all12"].eq("1G"), "EmployeeID"].unique())
            if one_g_emp_ids:
                interim.loc[interim["EmployeeID"].isin(one_g_emp_ids), "line14_final"] = ""

    # Consistent ordering
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


def build_final(interim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Final sheet expected by the PDF filler:
      EmployeeID, Month, Line14_Final, Line16_Final
    """
    if interim_df is None or interim_df.empty:
        return pd.DataFrame(columns=["EmployeeID","Month","Line14_Final","Line16_Final"])

    df = interim_df.copy()

    # Ensure Month exists
    if "Month" not in df.columns:
        if "MonthNum" in df.columns:
            df["Month"] = df["MonthNum"].map(lambda i: MONTHS[int(i)-1])
        else:
            df["Month"] = pd.Categorical(MONTHS, categories=MONTHS, ordered=True)

    # Use existing monthly codes or derive from flags
    if "line14_final" not in df.columns or "line16_final" not in df.columns:
        l14 = []
        l16 = []
        for _, r in df.iterrows():
            l14.append(_month_line14(
                bool(r.get("eligible_mv")),
                bool(r.get("offer_ee_allmonth")),
                bool(r.get("offer_spouse")),
                bool(r.get("offer_dependents")),
                bool(r.get("affordable_plan")),
            ))
            l16.append(_month_line16(
                employed=bool(r.get("employed")),
                enrolled_full=bool(r.get("enrolled_allmonth")),
                waiting=bool(r.get("waitingperiod_month")),
                ft=bool(r.get("ft")),
                offer_ee_allmonth=bool(r.get("offer_ee_allmonth")),
                affordable=bool(r.get("affordable_plan")),
            ))
        df["line14_final"] = l14
        df["line16_final"] = l16

    out = df.loc[:, ["EmployeeID","Month","line14_final","line16_final"]].copy()
    out = out.rename(columns={"line14_final":"Line14_Final","line16_final":"Line16_Final"})

    # sort for readability
    out["MonthIdx"] = out["Month"].map({m:i for i,m in enumerate(MONTHS)})
    out = out.sort_values(by=["EmployeeID","MonthIdx"]).drop(columns=["MonthIdx"])
    return out


def build_penalty_dashboard(interim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Output shape:
      EmployeeID | Reason | January ... December

    Monthly penalties:
      Penalty A → offer_ee_allmonth == False → $PENALTY_A_MONTHLY
      Penalty B → offer_ee_allmonth == True AND enrolled_allmonth == False AND affordable_plan == False
                   → $PENALTY_B_MONTHLY
      Else      → "-"

    The Reason text explains *why* MEC was not offered in Penalty A months
    (waiting period, not employed, not eligible), listing the specific months.
    """
    if interim_df is None or interim_df.empty:
        return pd.DataFrame(columns=["EmployeeID", "Reason"] + MONTHS)

    df = interim_df.copy()

    # Ensure required flags exist
    for col in ["offer_ee_allmonth", "enrolled_allmonth", "affordable_plan",
                "waitingperiod_month", "employed", "eligibleforcoverage", "MonthNum"]:
        if col not in df.columns:
            df[col] = False if col != "MonthNum" else None

    def money(x: float) -> str:
        return f"${x:,.2f}"

    thr = float(AFFORDABILITY_THRESHOLD)
    threshold_txt = f"${thr:,.0f}" if thr.is_integer() else f"${thr:,.2f}"

    def fmt_month_list(idx_list: List[int]) -> str:
        names = [FULL_MONTHS[i-1] for i in idx_list]
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return f"{', '.join(names[:-1])}, and {names[-1]}"

    reason_b = (
        f"Penalty B: Waived Unaffordable Coverage{BR}"
        "The employee was offered minimum essential coverage (MEC), but the lowest-cost option for "
        f"employee-only coverage was not affordable (>{threshold_txt}). The employee chose to waive "
        "this unaffordable coverage."
    )

    rows = []
    for emp, sub in df.groupby("EmployeeID", dropna=False):
        by_m = {int(r["MonthNum"]): r for _, r in sub.iterrows() if pd.notna(r.get("MonthNum"))}

        any_a = False
        any_b = False
        month_vals: List[str] = []

        months_wait: List[int] = []
        months_not_emp: List[int] = []
        months_not_elig: List[int] = []
        months_other: List[int] = []

        for idx, _ in enumerate(MONTHS, start=1):
            r = by_m.get(idx)
            if r is None:
                month_vals.append("-")
                continue

            offered_full  = bool(r.get("offer_ee_allmonth", False))
            enrolled_full = bool(r.get("enrolled_allmonth", False))
            affordable    = bool(r.get("affordable_plan", False))
            waiting       = bool(r.get("waitingperiod_month", False))
            employed      = bool(r.get("employed", False))
            eligible      = bool(r.get("eligibleforcoverage", False))

            if not offered_full:
                month_vals.append(money(PENALTY_A_MONTHLY))
                any_a = True

                if not employed:
                    months_not_emp.append(idx)
                elif waiting:
                    months_wait.append(idx)
                elif not eligible:
                    months_not_elig.append(idx)
                else:
                    months_other.append(idx)

            elif offered_full and (not enrolled_full) and (not affordable):
                month_vals.append(money(PENALTY_B_MONTHLY))
                any_b = True
            else:
                month_vals.append("-")

        if not any_a and not any_b:
            continue

        if any_a:
            sublines: List[str] = []
            if months_not_emp:
                sublines.append(
                    f"No coverage could be offered in {fmt_month_list(months_not_emp)} because the employee was not employed."
                )
            if months_wait:
                sublines.append(
                    f"Employee was not eligible for coverage in {fmt_month_list(months_wait)} because they were in their waiting period during those month(s)."
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
