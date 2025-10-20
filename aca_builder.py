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
        st_emp = emp_status[emp_status["employeeid"].astype(str) == str(emp)] if (emp_status is not None and "employeeid" in emp_status.columns) else pd.Data
