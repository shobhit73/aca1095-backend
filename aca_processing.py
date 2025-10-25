# aca_processing.py
from __future__ import annotations

import io
from datetime import date, timedelta
from typing import Dict, Tuple, Iterable, List
import pandas as pd

# ----------------- Month labels -----------------
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
FULL_MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ----------------- Utilities -----------------
def _coerce_str(x) -> str:
    try:
        return str(x)
    except Exception:
        return ""

def month_bounds(year: int, month: int) -> Tuple[date, date]:
    import calendar
    last = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last)

# ----------------- Excel I/O -----------------
def load_excel(excel_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """Read supported sheets. Missing sheets return empty DataFrames."""
    x = pd.ExcelFile(io.BytesIO(excel_bytes))

    def pick(name: str) -> str | None:
        for s in x.sheet_names:
            if s.strip().lower() == name.strip().lower():
                return s
        return None

    def read_sheet(name: str) -> pd.DataFrame:
        sname = pick(name)
        if not sname:
            return pd.DataFrame()
        df = pd.read_excel(x, sname)
        if df is None:
            return pd.DataFrame()
        df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]
        return df

    data: Dict[str, pd.DataFrame] = {}
    data["emp_demo"]   = read_sheet("Emp Demographic")
    data["emp_elig"]   = read_sheet("Emp Eligibility")
    data["emp_enroll"] = read_sheet("Emp Enrollment")
    data["dep_enroll"] = read_sheet("Dep Enrollment")

    # NEW: Emp Wait Period (exact name)
    data["emp_wait"]   = read_sheet("Emp Wait Period")  # employeeid, effectivedate, waitperiod
    # Backward-friendly aliasing if needed
    w = data["emp_wait"]
    if not w.empty:
        if "waitperioddays" in w.columns and "waitperiod" not in w.columns:
            w.rename(columns={"waitperioddays": "waitperiod"}, inplace=True)

    return data


def prepare_inputs(data: Dict[str, pd.DataFrame]) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Return normalized frames (NO boolean use of DataFrames)."""
    def _as_df(x) -> pd.DataFrame:
        return x if isinstance(x, pd.DataFrame) else pd.DataFrame()

    emp_demo   = _as_df(data.get("emp_demo")).copy()
    emp_elig   = _as_df(data.get("emp_elig")).copy()
    emp_enroll = _as_df(data.get("emp_enroll")).copy()
    dep_enroll = _as_df(data.get("dep_enroll")).copy()
    emp_wait   = _as_df(data.get("emp_wait")).copy()

    for df in (emp_demo, emp_elig, emp_enroll, dep_enroll, emp_wait):
        if not df.empty and "employeeid" in df.columns:
            df["employeeid"] = df["employeeid"].astype(str)

    if not emp_wait.empty:
        if "effectivedate" in emp_wait.columns:
            emp_wait["effectivedate"] = pd.to_datetime(emp_wait["effectivedate"], errors="coerce")
        if "waitperiod" in emp_wait.columns:
            emp_wait["waitperiod"] = pd.to_numeric(emp_wait["waitperiod"], errors="coerce").fillna(0).astype(int)

    return emp_demo, emp_elig, emp_enroll, dep_enroll, emp_wait


def choose_report_year(emp_elig: pd.DataFrame) -> int:
    import datetime as _dt
    if not emp_elig.empty:
        for c in ("eligibilitystartdate","eligibilityenddate"):
            if c in emp_elig.columns:
                s = pd.to_datetime(emp_elig[c], errors="coerce")
                yr = s.dt.year.dropna()
                if not yr.empty:
                    return int(yr.iloc[0])
    return _dt.date.today().year


# ----------------- Helpers imported by builder -----------------
def _collect_employee_ids(*frames: Iterable[pd.DataFrame]) -> List[str]:
    ids: set[str] = set()
    for df in frames:
        if isinstance(df, pd.DataFrame) and not df.empty and "employeeid" in df.columns:
            ids |= set(df["employeeid"].astype(str).tolist())
    return sorted(ids, key=lambda x: (len(x), x))

def _status_from_demographic(emp_demo: pd.DataFrame) -> pd.DataFrame:
    if emp_demo is None or emp_demo.empty:
        return pd.DataFrame(columns=["employeeid","statusstartdate","statusenddate","_role_norm","_estatus_norm"])
    df = emp_demo.copy()
    for col in ("statusstartdate","statusenddate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "_role_norm" not in df.columns:
        df["_role_norm"] = ""
    if "_estatus_norm" not in df.columns:
        df["_estatus_norm"] = ""
    if "employeeid" in df.columns:
        df["employeeid"] = df["employeeid"].astype(str)
    return df[["employeeid","statusstartdate","statusenddate","_role_norm","_estatus_norm"]]

def _any_overlap(df: pd.DataFrame, start_col: str, end_col: str, ms: date, me: date, *, mask=None) -> bool:
    if df is None or df.empty or start_col not in df.columns or end_col not in df.columns:
        return False
    if mask is None:
        mask = pd.Series(True, index=df.index)
    s = pd.to_datetime(df[start_col], errors="coerce").dt.date.fillna(date(1900,1,1))
    e = pd.to_datetime(df[end_col], errors="coerce").dt.date.fillna(date(9999,12,31))
    return bool(((e >= ms) & (s <= me) & mask).any())

def _all_month(df: pd.DataFrame, start_col: str, end_col: str, ms: date, me: date, *, mask=None) -> bool:
    if df is None or df.empty or start_col not in df.columns or end_col not in df.columns:
        return False
    if mask is None:
        mask = pd.Series(True, index=df.index)
    s = pd.to_datetime(df[start_col], errors="coerce").dt.date.fillna(date(1900,1,1))
    e = pd.to_datetime(df[end_col], errors="coerce").dt.date.fillna(date(9999,12,31))
    return bool((mask & (s <= ms) & (e >= me)).any())

# ---- NEW: compute waiting by window overlap from Emp Wait Period
def _waiting_in_month(wait_df_emp: pd.DataFrame, ms: date, me: date) -> bool:
    if wait_df_emp is None or wait_df_emp.empty:
        return False
    if "effectivedate" not in wait_df_emp.columns or "waitperiod" not in wait_df_emp.columns:
        return False
    s = pd.to_datetime(wait_df_emp["effectivedate"], errors="coerce").dt.date
    d = pd.to_numeric(wait_df_emp["waitperiod"], errors="coerce").fillna(0).astype(int)
    e = s + pd.to_timedelta((d.clip(lower=0) - 1).astype(int), unit="D")
    e = e.where(d > 0, s - timedelta(days=1))  # 0-day windows → non-overlap
    return bool(((e >= ms) & (s <= me)).any())
