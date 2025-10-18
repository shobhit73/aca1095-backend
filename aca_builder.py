# aca_builder.py
# Build a rich Interim grid (employee x month) + Final + Penalty dashboard

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd

from aca_processing import (
    MONTHS,
    _collect_employee_ids, _grid_for_year, month_bounds,
    _any_overlap, _all_month,
    _status_from_demographic,
)

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
AFFORDABILITY_THRESHOLD = 50.00  # < $50 => affordable   (change to <= if you want 50 to count)
EMP_TIERS = ("EMP", "EMPFAM", "EMPSPOUSE")
SPOUSE_TOKENS = ("SPOUSE",)      # detect spouse offer in eligibility tier text
DEPEND_TOKENS = ("FAM", "CHILD", "DEPEND")  # detect dependents offer in tier text


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
# Affordability helper (EMP tier)
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


# ------------------------------------------------------------
# Offer detection from Eligibility tiers
# ------------------------------------------------------------
def _offered_allmonth(el_emp: pd.DataFrame, ms, me) -> bool:
    """Employee-level offer for full month (any tier that includes employee)."""
    if el_emp.empty or "eligibilitytier" not in el_emp.columns:
        return False
    tiers = el_emp["eligibilitytier"].astype(str).str.upper().str.strip()
    mask = tiers.isin(EMP_TIERS) | tiers.str.contains("EMP", na=False)
    return _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=mask)

def _offer_spouse_any(el_emp: pd.DataFrame, ms, me) -> bool:
    if el_emp.empty or "eligibilitytier" not in el_emp.columns:
        return False
    tiers = el_emp["eligibilitytier"].astype(str).str.upper().str.strip()
    mask = pd.Series(False, index=el_emp.index)
    for t in SPOUSE_TOKENS:
        mask |= tiers.str.contains(t, na=False)
    return _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=mask)

def _offer_dependents_any(el_emp: pd.DataFrame, ms, me) -> bool:
    if el_emp.empty or "eligibilitytier" not in el_emp.columns:
        return False
    tiers = el_emp["eligibilitytier"].astype(str).str.upper().str.strip()
    mask = pd.Series(False, index=el_emp.index)
    for t in DEPEND_TOKENS:
        mask |= tiers.str.contains(t, na=False)
    return _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=mask)


# ------------------------------------------------------------
# Dependent eligibility/enrollment helpers
# ------------------------------------------------------------
def _dep_any(df: pd.DataFrame, ms, me, rel_tokens: Tuple[str, ...], flag_col: str) -> bool:
    if df.empty or "dependentrelationship" not in df.columns or flag_col not in df.columns:
        return False
    rel = df["dependentrelationship"].astype(str).str.upper().str.strip()
    mask_rel = pd.Series(False, index=df.index)
    for t in rel_tokens:
        mask_rel |= rel.str.contains(t, na=False)
    mask_final = mask_rel & df[flag_col].astype(bool)
    # choose the right date columns based on flag type
    start_col = "eligiblestartdate" if flag_col == "eligible" else "enrollmentstartdate"
    end_col   = "eligibleenddate"   if flag_col == "eligible" else "enrollmentenddate"
    return _any_overlap(df, start_col, end_col, ms, me, mask=mask_final)

# spouse/child tokens
SPOUSE_REL = ("SPOUSE",)
CHILD_REL  = ("CHILD","SON","DAUGHTER","DEPEND")


# ------------------------------------------------------------
# Status helpers (FT/PT)
# ------------------------------------------------------------
def _has_status_any(st_emp: pd.DataFrame, ms, me) -> bool:
    if st_emp.empty: return False
    return _any_overlap(st_emp, "statusstartdate","statusenddate", ms, me)

def _is_ft(st_emp: pd.DataFrame, ms, me) -> bool:
    if st_emp.empty: return False
    tok_cols = [c for c in ["_estatus_norm","_role_norm"] if c in st_emp.columns]
    if not tok_cols:
        return False
    mask = pd.Series(False, index=st_emp.index)
    for c in tok_cols:
        s = st_emp[c].astype(str)
        mask |= s.str.contains("FULLTIME", na=False) | s.str.fullmatch("FT", na=False)
    return _any_overlap(st_emp, "statusstartdate","statusenddate", ms, me, mask=mask)

def _is_pt(st_emp: pd.DataFrame, ms, me) -> bool:
    if st_emp.empty: return False
    tok_cols = [c for c in ["_estatus_norm","_role_norm"] if c in st_emp.columns]
    if not tok_cols:
        return False
    mask = pd.Series(False, index=st_emp.index)
    for c in tok_cols:
        s = st_emp[c].astype(str)
        mask |= s.str.contains("PARTTIME", na=False) | s.str.fullmatch("PT", na=False)
    return _any_overlap(st_emp, "statusstartdate","statusenddate", ms, me, mask=mask)


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
    st = emp_status
    if st is None or st.empty:
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
            employed   = _has_status_any(st_emp, ms, me)
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
            offer_spouse       = _offer_spouse_any(el_emp, ms, me)
            offer_dependents   = _offer_dependents_any(el_emp, ms, me)

            # ---- enrollment flags
            enrolled_full = False
            if not en_emp.empty:
                mask_en = pd.Series(True, index=en_emp.index)
                if "isenrolled" in en_emp.columns:
                    mask_en = en_emp["isenrolled"].astype(bool)
                enrolled_full = _all_month(en_emp, "enrollmentstartdate","enrollmentenddate", ms, me, mask=mask_en)

            # ---- dependents eligible/enrolled
            spouse_eligible = _dep_any(de_emp, ms, me, SPOUSE_REL, "eligible")
            child_eligible  = _dep_any(de_emp, ms, me, CHILD_REL,  "eligible")
            spouse_enrolled = _dep_any(de_emp, ms, me, SPOUSE_REL, "enrolled")
            child_enrolled  = _dep_any(de_emp, ms, me, CHILD_REL,  "enrolled")

            # ---- affordability
            emp_cost = _latest_emp_cost_for_month(el_emp, ms, me)
            affordable = (emp_cost is not None) and (emp_cost < AFFORDABILITY_THRESHOLD)
            # To count $50 as affordable, change the comparator above to <=

            # ---- waiting period (simple heuristic)
            # Employed this month, not yet eligible this month, and a future eligibility start exists.
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
            })

    interim = pd.DataFrame(rows)

    # Year-level 1G (never FT all year)
    if not interim.empty:
        ft_by_emp = interim.groupby("EmployeeID")["ft"].sum(min_count=0)
        one_g = ft_by_emp == 0
        interim["line14_all12"] = interim["EmployeeID"].map(lambda e: "1G" if bool(one_g.get(e, False)) else "")

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
    Simple rollup for visibility:
      EmployeeID, Months_EligibleMV, Months_Affordable, Months_EligibleMV_NotAffordable
    """
    if interim_df is None or interim_df.empty:
        return pd.DataFrame(columns=[
            "EmployeeID","Months_EligibleMV","Months_Affordable","Months_EligibleMV_NotAffordable"
        ])

    df = interim_df.copy()
    g = df.groupby("EmployeeID", dropna=False)
    res = pd.DataFrame({
        "Months_EligibleMV": g["eligible_mv"].sum(min_count=0),
        "Months_Affordable": g["affordable_plan"].sum(min_count=0),
    }).reset_index()

    df["_mv_not_aff"] = df["eligible_mv"].astype(bool) & (~df["affordable_plan"].astype(bool))
    mna = df.groupby("EmployeeID")["_mv_not_aff"].sum(min_count=0).reset_index(
        name="Months_EligibleMV_NotAffordable"
    )

    out = res.merge(mna, on="EmployeeID", how="left").fillna(0)
    for c in ["Months_EligibleMV","Months_Affordable","Months_EligibleMV_NotAffordable"]:
        out[c] = out[c].astype(int)
    return out
