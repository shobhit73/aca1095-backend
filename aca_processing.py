# aca_processing.py
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Tuple, Iterable, List
import io
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
    """Read all supported sheets. Missing sheets return empty DataFrames."""
    x = pd.ExcelFile(io.BytesIO(excel_bytes))

    def pick(name: str) -> str | None:
        # case-insensitive match
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
        # normalize columns: lower + strip spaces
        df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]
        return df

    data: Dict[str, pd.DataFrame] = {}
    data["emp_demo"]   = read_sheet("Emp Demographic")      # your existing demographic sheet
    data["emp_elig"]   = read_sheet("Emp Eligibility")      # eligibility
    data["emp_enroll"] = read_sheet("Emp Enrollment")       # employee enrollment
    data["dep_enroll"] = read_sheet("Dep Enrollment")       # dependent enrollment

    # NEW: Wait Period sheet (exact name required by you)
    wait_df = read_sheet("Emp Wait Period")
    # rename expected headers just in case
    ren = {"waitperiod":"waitperiod", "waitperioddays":"waitperiod"}
    for k,v in ren.items():
        if k in wait_df.columns and v not in wait_df.columns:
            wait_df.rename(columns={k:v}, inplace=True)
    data["emp_wait"] = wait_df  # {employeeid, effectivedate, waitperiod}

    return data


def prepare_inputs(data: Dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return normalized frames in a fixed order."""
    emp_demo   = (data.get("emp_demo")   or pd.DataFrame()).copy()
    emp_elig   = (data.get("emp_elig")   or pd.DataFrame()).copy()
    emp_enroll = (data.get("emp_enroll") or pd.DataFrame()).copy()
    dep_enroll = (data.get("dep_enroll") or pd.DataFrame()).copy()
    emp_wait   = (data.get("emp_wait")   or pd.DataFrame()).copy()

    # Ensure employeeid present as string where possible
    for df in (emp_demo, emp_elig, emp_enroll, dep_enroll, emp_wait):
        if not df.empty and "employeeid" in df.columns:
            df["employeeid"] = df["employeeid"].astype(str)

    # Wait sheet: parse date & integer days if present
    if not emp_wait.empty:
        if "effectivedate" in emp_wait.columns:
            emp_wait["effectivedate"] = pd.to_datetime(emp_wait["effectivedate"], errors="coerce")
        if "waitperiod" in emp_wait.columns:
            emp_wait["waitperiod"] = pd.to_numeric(emp_wait["waitperiod"], errors="coerce").fillna(0).astype(int)

    return emp_demo, emp_elig, emp_enroll, dep_enroll, emp_wait


def choose_report_year(emp_elig: pd.DataFrame) -> int:
    """Pick a sensible default filing year if UI doesn't provide one."""
    import datetime as _dt
    if not emp_elig.empty:
        for c in ("eligibilitystartdate","eligibilityenddate"):
            if c in emp_elig.columns:
                s = pd.to_datetime(emp_elig[c], errors="coerce")
                yr = s.dt.year.dropna()
                if not yr.empty:
                    return int(yr.iloc[0])
    return _dt.date.today().year

# ----------------- Status & helper funcs the builder imports -----------------
def _collect_employee_ids(*frames: Iterable[pd.DataFrame]) -> List[str]:
    ids: set[str] = set()
    for df in frames:
        if isinstance(df, pd.DataFrame) and not df.empty and "employeeid" in df.columns:
            ids |= set(df["employeeid"].astype(str).tolist())
    return sorted(ids, key=lambda x: (len(x), x))

def _status_from_demographic(emp_demo: pd.DataFrame) -> pd.DataFrame:
    """Simplified normalizer. Keep as-is to avoid changing FT/PT/Employment logic elsewhere."""
    if emp_demo is None or emp_demo.empty:
        return pd.DataFrame(columns=["employeeid","statusstartdate","statusenddate","_role_norm","_estatus_norm"])
    df = emp_demo.copy()
    # tolerant parsing of dates and role/status text if present
    for col in ("statusstartdate","statusenddate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("role","employeestatus","status","employeeclass"):
        if col in df.columns and "_role_norm" not in df.columns:
            df["_role_norm"] = df[col].astype(str).str.upper().str.strip()
        if col in df.columns and "_estatus_norm" not in df.columns:
            df["_estatus_norm"] = df[col].astype(str).str.upper().str.strip()
    if "_role_norm" not in df.columns:
        df["_role_norm"] = ""
    if "_estatus_norm" not in df.columns:
        df["_estatus_norm"] = ""
    if "employeeid" in df.columns:
        df["employeeid"] = df["employeeid"].astype(str)
    return df[["employeeid","statusstartdate","statusenddate","_role_norm","_estatus_norm"]]

# These two were added earlier and are used by the builder.
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

# ----------------- NEW: Wait Period overlap logic -----------------
def _waiting_in_month(wait_df_emp: pd.DataFrame, ms: date, me: date) -> bool:
    """
    True if ANY wait window overlaps the calendar month [ms, me].
    Each row in 'Emp Wait Period' defines a window:
        start = EffectiveDate
        end   = EffectiveDate + (Wait Period days) - 1
    """
    if wait_df_emp is None or wait_df_emp.empty:
        return False
    if "effectivedate" not in wait_df_emp.columns or "waitperiod" not in wait_df_emp.columns:
        return False

    s = pd.to_datetime(wait_df_emp["effectivedate"], errors="coerce").dt.date
    d = pd.to_numeric(wait_df_emp["waitperiod"], errors="coerce").fillna(0).astype(int)
    e = s + pd.to_timedelta((d.clip(lower=0) - 1).astype(int), unit="D")
    # if waitperiod == 0, treat as 0-day (start=end=start-1 day). We'll consider that as not waiting.
    e = e.where(d > 0, s - timedelta(days=1))

    overlap = (e >= ms) & (s <= me)
    return bool(overlap.any())
