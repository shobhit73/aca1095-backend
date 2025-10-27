# aca_processing.py
from __future__ import annotations

import io
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("aca.processing")
if not logger.handlers:
    _h = logging.StreamHandler()
    _f = logging.Formatter("[ACA][processing] %(levelname)s: %(message)s")
    _h.setFormatter(_f)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Constants / helpers
# -----------------------------------------------------------------------------
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _norm_txt(x: str) -> str:
    return re.sub(r"\W+", "", (str(x) or "")).lower()

def _pick_col(df_or_index, candidates: List[str]) -> Optional[str]:
    cols = list(df_or_index) if not hasattr(df_or_index, "columns") else df_or_index.columns
    norm = {_norm_txt(c): c for c in cols}
    for c in candidates:
        k = _norm_txt(c)
        if k in norm:
            return norm[k]
    return None

# -----------------------------------------------------------------------------
# Year helpers (back-compat for main_fastapi)
# -----------------------------------------------------------------------------
def choose_report_year(requested: Optional[int], fallback: Optional[int] = None) -> int:
    """
    Back-compat: normalize the filing/report year.
    - If 'requested' is a truthy int, return it.
    - Else if 'fallback' provided, return fallback.
    - Else return current calendar year.
    """
    try:
        if requested is not None:
            y = int(requested)
            if y > 1900:
                return y
    except Exception:
        pass
    if fallback:
        return int(fallback)
    return int(datetime.utcnow().year)

# Preserve older alias some code used
get_report_year = choose_report_year

# -----------------------------------------------------------------------------
# Excel I/O
# -----------------------------------------------------------------------------
def read_excel_sheets(xlsx_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """
    Read all worksheets into a dict[name] -> DataFrame, trimmed & cleaned.
    """
    logger.info("Reading Excel bytes into sheet dict")
    with io.BytesIO(xlsx_bytes) as bio:
        xl = pd.read_excel(bio, sheet_name=None, dtype=str, engine="openpyxl")
    sheets: Dict[str, pd.DataFrame] = {}
    for name, df in xl.items():
        if df is None:
            continue
        df2 = df.copy()
        df2.columns = [str(c).strip() for c in df2.columns]
        df2 = df2.dropna(how="all")
        sheets[str(name).strip()] = df2
    logger.info("Loaded %d sheet(s): %s", len(sheets), list(sheets.keys()))
    return sheets

# Backward-compatible aliases
extract_sheets = read_excel_sheets
read_excel_to_sheets = read_excel_sheets
load_excel = read_excel_sheets  # legacy alias
load_sheets = read_excel_sheets
get_sheets = read_excel_sheets

# -----------------------------------------------------------------------------
# Employee row (Demographics)
# -----------------------------------------------------------------------------
def find_employee_row(sheets: Dict[str, pd.DataFrame], employee_id: str) -> pd.Series:
    """
    Find an employee's demographic row by EmployeeID.
    """
    emp_id = str(employee_id).strip()
    logger.info("Finding employee row for EmployeeID=%s", emp_id)
    for sh in ["Employee Demographics", "Employees", "Employee Master", "Emp Demo", "Demographics"]:
        df = sheets.get(sh)
        if df is None or df.empty:
            continue
        col_empid = _pick_col(df, ["EmployeeID","EmpID","Employee Id","Employee_Id"])
        if not col_empid:
            continue
        hit = df[df[col_empid].astype(str).str.strip() == emp_id]
        if not hit.empty:
            logger.info("Matched employee on sheet '%s'", sh)
            return hit.iloc[0]
    logger.warning("EmployeeID=%s not found in demographic sheets; returning empty row", emp_id)
    return pd.Series({}, dtype="object")

# Alias
get_employee_row = find_employee_row

# -----------------------------------------------------------------------------
# Emp Wait Period
# -----------------------------------------------------------------------------
def _load_wait_period(sheets: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    wp = sheets.get("Emp Wait Period")
    if wp is None or wp.empty:
        return None
    wp2 = wp.copy()
    id_col  = _pick_col(wp2, ["EmployeeID","EmpID","Employee Id"])
    eff_col = _pick_col(wp2, ["EffectiveDate","Effective Date"])
    waitcol = _pick_col(wp2, ["Wait Period","WaitPeriod","Waiting Period"])
    if not id_col or not waitcol:
        return None
    keep = [c for c in [id_col, eff_col, waitcol] if c]
    wp2 = wp2[keep].copy()
    wp2.rename(columns={
        id_col: "EmployeeID",
        eff_col or "EffectiveDate": "EffectiveDate",
        waitcol: "Wait Period",
    }, inplace=True)
    return wp2

def attach_wait_period(df: pd.DataFrame, sheets: Dict[str, pd.DataFrame], employee_id: Optional[str] = None) -> pd.DataFrame:
    """
    Attach 'Wait Period' (and EffectiveDate if present) into df by EmployeeID.
    """
    wp = _load_wait_period(sheets)
    if wp is None or wp.empty:
        return df

    work = df.copy()
    if "EmployeeID" not in work.columns and employee_id is not None:
        work["EmployeeID"] = str(employee_id).strip()
    if "EmployeeID" not in work.columns:
        return df

    out = work.merge(wp, how="left", on="EmployeeID")
    return out

# -----------------------------------------------------------------------------
# Spouse/Child enrollment (Plan-agnostic)
# -----------------------------------------------------------------------------
_TIER_KW = {
    "spouse": {
        "ee+sp","emp+sp","employeespouse","employee+spouse",
        "family","ee+fam","emp+fam","employee+family",
        "ee+sp+ch","ee+sp+children","emp+sp+ch",
    },
    "child": {
        "ee+ch","emp+ch","employeechild","employee+child",
        "employeechildren","employee+children","ee+children",
        "family","ee+fam","emp+fam","employee+family",
        "ee+sp+ch","ee+sp+children","emp+sp+ch",
    },
}

def _tier_covers(kind: str, tier_text: str) -> bool:
    t = _norm_txt(tier_text)
    for kw in _TIER_KW.get(kind, set()):
        if kw in t:
            return True
    return False

def _row_month_flags(row: pd.Series) -> Dict[str, bool]:
    flags = {m: False for m in MONTHS}
    all12_col = _pick_col(row.index, ["All 12 Months","All12Months","All12"])
    if all12_col and str(row.get(all12_col,"")).strip().lower() in {"1","true","yes","y","x"}:
        return {m: True for m in MONTHS}
    for m in MONTHS:
        c = _pick_col(row.index, [m, m.upper(), m.capitalize()])
        if c and str(row.get(c,"")).strip().lower() in {"1","true","yes","y","x"}:
            flags[m] = True
    return flags

def _full_year(flags: Dict[str, bool]) -> bool:
    return all(flags.get(m, False) for m in MONTHS)

def derive_spouse_child_enrollment_from_emp_row(emp_enroll_row: pd.Series) -> Tuple[bool,bool]:
    """
    Plan-agnostic: spouse/child enrolled if tier implies coverage AND months cover full year.
    """
    tier_col = _pick_col(emp_enroll_row.index, ["Tier","Coverage Tier","Plan Tier","Enrollment Tier","Tier Name"])
    tier = str(emp_enroll_row.get(tier_col,"")) if tier_col else ""
    flags = _row_month_flags(emp_enroll_row)
    full = _full_year(flags) if flags else False
    spouse = _tier_covers("spouse", tier) and full
    child  = _tier_covers("child", tier) and full
    return bool(spouse), bool(child)

def derive_spouse_child_enrollment_from_dependents(dep_df: Optional[pd.DataFrame], employee_id: str) -> Tuple[bool,bool]:
    """
    Optional reinforcement using a 'Dep Enrollment' style sheet:
    If any dependent with Relationship=Spouse (or Child) has full-year months, mark True.
    """
    if dep_df is None or dep_df.empty:
        return False, False
    col_empid = _pick_col(dep_df, ["EmployeeID","EmpID","Employee Id"])
    if not col_empid:
        return False, False
    rel_col = _pick_col(dep_df, ["Relationship","Rel","Relation"])
    spouse_flag, child_flag = False, False
    view = dep_df[dep_df[col_empid].astype(str).str.strip() == str(employee_id).strip()]
    for _, r in view.iterrows():
        rel = _norm_txt(str(r.get(rel_col,""))) if rel_col else ""
        flags = _row_month_flags(r)
        if _full_year(flags):
            if "spouse" in rel:
                spouse_flag = True
            if "child" in rel or "dependent" in rel:
                child_flag = True
        if spouse_flag and child_flag:
            break
    return bool(spouse_flag), bool(child_flag)

# -----------------------------------------------------------------------------
# Monthly final slice (Line14/16 + flags + wait period)
# -----------------------------------------------------------------------------
def build_final_for_employee(sheets: Dict[str, pd.DataFrame], employee_id: str) -> pd.DataFrame:
    """
    Returns a tidy per-month DataFrame for an employee with columns:
      EmployeeID, Month, Line14_Final, Line16_Final, Spouse Enrolled, Child Enrolled, (optionally Wait Period, EffectiveDate)
    """
    emp_id = str(employee_id).strip()
    logger.info("Building final monthly slice for EmployeeID=%s", emp_id)

    # 1) Monthly Line14/16
    base_df = _find_monthly_codes(sheets, emp_id)

    # 2) Spouse/Child Enrolled (plan-agnostic)
    sp_enr, ch_enr = _compute_spouse_child_flags(sheets, emp_id)
    base_df["Spouse Enrolled"] = bool(sp_enr)
    base_df["Child Enrolled"] = bool(ch_enr)

    # 3) Wait Period merge
    base_df = attach_wait_period(base_df, sheets, employee_id=emp_id)

    # 4) Order months
    base_df["Month"] = pd.Categorical(base_df["Month"], categories=MONTHS, ordered=True)
    base_df = base_df.sort_values("Month").reset_index(drop=True)

    logger.info("Final monthly slice ready: %d rows", len(base_df))
    return base_df

# Alias
slice_final_for_employee = build_final_for_employee

def _find_monthly_codes(sheets: Dict[str, pd.DataFrame], employee_id: str) -> pd.DataFrame:
    """
    Look for a per-month table with columns like Month/Line14/Line16 in common sheets.
    If not found, return empty codes for all months.
    """
    candidates = [
        "Final", "Final Output", "ACA Final", "Results",
        "Interim", "Interim Output", "ACA Interim",
    ]
    emp_id = str(employee_id).strip()
    for sh in candidates:
        df = sheets.get(sh)
        if df is None or df.empty:
            continue
        col_empid = _pick_col(df, ["EmployeeID","EmpID","Employee Id"])
        col_month = _pick_col(df, ["Month","Months","Coverage Month","Period"])
        if not col_empid or not col_month:
            continue
        view = df[df[col_empid].astype(str).str.strip() == emp_id].copy()
        if view.empty:
            continue
        col_l14 = _pick_col(view, ["Line14_Final","Line14","Line 14","L14","Line 14 Code"])
        col_l16 = _pick_col(view, ["Line16_Final","Line16","Line 16","L16","Line 16 Code"])

        out = []
        # Case 1: explicit month rows
        if view[col_month].str.strip().str.lower().isin([m.lower() for m in MONTHS]).any():
            for _, r in view.iterrows():
                mon = str(r.get(col_month,"")).strip()
                if _norm_txt(mon) in [_norm_txt(m) for m in MONTHS]:
                    out.append({
                        "EmployeeID": emp_id,
                        "Month": next(M for M in MONTHS if _norm_txt(M) == _norm_txt(mon)),
                        "Line14_Final": str(r.get(col_l14, "") or "") if col_l14 else "",
                        "Line16_Final": str(r.get(col_l16, "") or "") if col_l16 else "",
                    })
        else:
            # Case 2: "All 12 Months" broadcast
            all12_col = _pick_col(view, ["All 12 Months","All12Months","All12"])
            if all12_col:
                r = view.iloc[0]
                v14 = str(r.get(col_l14,"") or "") if col_l14 else ""
                v16 = str(r.get(col_l16,"") or "") if col_l16 else ""
                for m in MONTHS:
                    out.append({
                        "EmployeeID": emp_id,
                        "Month": m,
                        "Line14_Final": v14,
                        "Line16_Final": v16,
                    })
        if out:
            return pd.DataFrame(out, columns=["EmployeeID","Month","Line14_Final","Line16_Final"])

    # Fallback: rows for all months with empty codes
    return pd.DataFrame({
        "EmployeeID": [emp_id]*12,
        "Month": MONTHS,
        "Line14_Final": ["" for _ in MONTHS],
        "Line16_Final": ["" for _ in MONTHS],
    })

def _compute_spouse_child_flags(sheets: Dict[str, pd.DataFrame], employee_id: str) -> Tuple[bool,bool]:
    """
    Plan-agnostic flags:
      True if tier implies coverage AND months cover full year;
      or any dependent (spouse/child) shows full-year coverage.
    """
    emp_id = str(employee_id).strip()

    # Emp Enrollment
    emp_enroll = None
    for sh in ["Emp Enrollment","Employee Enrollment","Employee Coverage","Enrollment"]:
        df = sheets.get(sh)
        if df is None or df.empty:
            continue
        col_empid = _pick_col(df, ["EmployeeID","EmpID","Employee Id"])
        if not col_empid:
            continue
        hit = df[df[col_empid].astype(str).str.strip() == emp_id]
        if not hit.empty:
            emp_enroll = hit.iloc[0]
            break

    spouse, child = False, False
    if emp_enroll is not None:
        spouse, child = derive_spouse_child_enrollment_from_emp_row(emp_enroll)

    # Dep Enrollment reinforcement
    dep_df = None
    for sh in ["Dep Enrollment","Dependents","Dependent Enrollment","Covered Individuals"]:
        if sh in sheets and not sheets[sh].empty:
            dep_df = sheets[sh]
            break
    if dep_df is not None:
        dep_sp, dep_ch = derive_spouse_child_enrollment_from_dependents(dep_df, emp_id)
        spouse = spouse or dep_sp
        child  = child or dep_ch

    return bool(spouse), bool(child)

# -----------------------------------------------------------------------------
# FastAPI convenience
# -----------------------------------------------------------------------------
def prepare_employee_context(
    xlsx_bytes: bytes,
    employee_id: str,
    year: int
) -> Tuple[Dict[str, pd.DataFrame], pd.Series, pd.DataFrame]:
    """
    Reads Excel, finds the employee row, and builds the monthly final slice.
    Returns: (sheets, emp_row, final_df_emp)
    """
    sheets = read_excel_sheets(xlsx_bytes)
    emp_row = find_employee_row(sheets, employee_id)
    final_df_emp = build_final_for_employee(sheets, employee_id)
    return sheets, emp_row, final_df_emp

# -----------------------------------------------------------------------------
# Backward-compat shims (to avoid breaking older imports)
# -----------------------------------------------------------------------------
# Old projects might import these names:
prepare_inputs = prepare_employee_context  # legacy alias
