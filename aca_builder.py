# aca_builder.py
# Build interim grid and compute key monthly flags

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import pandas as pd
from pandas import Timestamp

from aca_processing import (
    _collect_employee_ids, _grid_for_year, month_bounds,
    _any_overlap, _all_month, normalize_columns
)

# ----- Affordability threshold (EMP-only) -----
AFFORDABILITY_THRESHOLD = 50.00  # < $50 => affordable

def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Defensive column aliasing in case inputs didn't come via load_excel/prepare_inputs."""
    if df is None or df.empty:
        return df
    df = df.copy()
    cols = set(df.columns)

    # EligiblePlan -> plancode
    if "eligibleplan" in cols and "plancode" not in cols:
        df["plancode"] = df["eligibleplan"].astype(str).str.strip()

    # EligibleTier -> eligibilitytier  (normalized is 'eligibletier')
    if "eligibletier" in cols and "eligibilitytier" not in cols:
        df["eligibilitytier"] = df["eligibletier"].astype(str).str.strip()

    return df

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

    # overlap with month [ms, me]
    df = el_df[(el_df["eligibilityenddate"].fillna(pd.Timestamp.max).dt.date >= ms) &
               (el_df["eligibilitystartdate"].fillna(pd.Timestamp.min).dt.date <= me)]
    if df.empty or "plancost" not in df.columns:
        return None

    # restrict to EMP (employee-only) tier
    tier_u = df["eligibilitytier"].astype(str).str.strip().str.upper()
    df = df[tier_u.eq("EMP")]
    if df.empty:
        return None

    # pick the row with the latest eligibility end date
    df = df.sort_values("eligibilityenddate", ascending=False)
    v = pd.to_numeric(df.iloc[0]["plancost"], errors="coerce")
    return float(v) if not pd.isna(v) else None

def build_interim(
    emp_demo: pd.DataFrame,
    emp_status: pd.DataFrame,
    emp_elig: pd.DataFrame,
    emp_enroll: pd.DataFrame,
    dep_enroll: pd.DataFrame,
    year: int
) -> pd.DataFrame:
    """
    Returns an employee x month interim table with flags:
      - eligible_mv: eligible for PlanA any time in month
      - affordable_plan: EMP-tier plan cost < AFFORDABILITY_THRESHOLD
    """
    # Defensive aliasing
    emp_elig = _apply_aliases(emp_elig)
    emp_enroll = _apply_aliases(emp_enroll)
    dep_enroll = _apply_aliases(dep_enroll)

    # Normalize some expected columns to datetime just in case
    for df, sc, ec in [
        (emp_elig, "eligibilitystartdate","eligibilityenddate"),
        (emp_enroll,"enrollmentstartdate","enrollmentenddate"),
        (dep_enroll,"eligiblestartdate","eligibleenddate"),
    ]:
        if not df.empty:
            for c in (sc, ec):
                if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
                    df[c] = pd.to_datetime(df[c], errors="coerce")

    # Employee list & monthly grid
    employee_ids = _collect_employee_ids(emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll)
    grid = _grid_for_year(employee_ids, year)

    out_rows = []
    for emp in employee_ids:
        el_emp = emp_elig[emp_elig["employeeid"].astype(str) == str(emp)].copy() if not emp_elig.empty else pd.DataFrame()

        for m in range(1, 12+1):
            ms, me = month_bounds(year, m)

            eligible_mv = False
            affordable = False

            if not el_emp.empty and {"eligibilitystartdate","eligibilityenddate"} <= set(el_emp.columns):

                # eligible_mv → TRUE if eligible for PlanA at any time during the month
                if "plancode" in el_emp.columns:
                    plan_u = el_emp["plancode"].astype(str).str.strip().str.upper()
                    eligible_mv = _any_overlap(
                        el_emp, "eligibilitystartdate","eligibilityenddate", ms, me, mask=plan_u.eq("PLANA")
                    )

                # affordability from EMP-tier cost (<$50 by default)
                emp_cost = _latest_emp_cost_for_month(el_emp, ms, me)
                affordable = (emp_cost is not None) and (emp_cost < AFFORDABILITY_THRESHOLD)
                # If you want $50 to count as affordable, flip to <=:
                # affordable = (emp_cost is not None) and (emp_cost <= AFFORDABILITY_THRESHOLD)

            out_rows.append({
                "employeeid": emp,
                "year": year,
                "monthnum": m,
                "eligible_mv": bool(eligible_mv),
                "affordable_plan": bool(affordable),
            })

    interim = pd.DataFrame(out_rows)
    return interim
