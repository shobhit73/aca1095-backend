# aca_processing.py
from __future__ import annotations

import io
from datetime import date, timedelta
from typing import Dict, Tuple, Iterable, List
import pandas as pd

from debug_logging import get_logger, log_df, log_call
log = get_logger("processing")

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
@log_call(log)
def load_excel(excel_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """
    Read supported sheets (case-insensitive).
    Missing sheets return empty DataFrames.
    """
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

    # Emp Wait Period (exact name)
    data["emp_wait"]   = read_sheet("Emp Wait Period")  # employeeid, effectivedate, waitperiod
    w = data["emp_wait"]
    if not w.empty and "waitperioddays" in w.columns and "waitperiod" not in w.columns:
        w.rename(columns={"waitperioddays": "waitperiod"}, inplace=True)

    log.info("load_excel: sheets detected", extra={"extra_data": {"sheets": list(x.sheet_names)}})
    return data

@log_call(log)
def prepare_inputs(data: Dict[str, pd.DataFrame]) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Normalize frames and return them in a fixed order
    (NO boolean evaluation of DataFrames).
    """
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

    log_df(log, emp_demo, "emp_demo")
    log_df(log, emp_elig, "emp_elig")
    log_df(log, emp_enroll, "emp_enroll")
    log_df(log, dep_enroll, "dep_enroll")
    log_df(log, emp_wait, "emp_wait")
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
    """
    Normalize employment status from Emp Demographic with robust FT/PT + ACTIVE/TERM detection.
    - Creates/aligns: statusstartdate, statusenddate
    - Derives: _role_norm in {"FULLTIME","PARTTIME",""} and _estatus_norm in {"ACTIVE","TERM",""}
    - Scans MANY column names/values and also numeric heuristics (FTE, hours).
    """
    if emp_demo is None or emp_demo.empty:
        return pd.DataFrame(columns=["employeeid","statusstartdate","statusenddate","_role_norm","_estatus_norm"])

    df = emp_demo.copy()

    if "employeeid" in df.columns:
        df["employeeid"] = df["employeeid"].astype(str)

    start_cands = ["statusstartdate","hiredate","startdate","effectivedate","originalhiredate","employmentstartdate"]
    end_cands   = ["statusenddate","termdate","terminationdate","enddate","separationdate","employmentenddate"]

    def _first(cols):
        for c in cols:
            if c in df.columns: return c
        return None

    s_col = _first(start_cands)
    e_col = _first(end_cands)

    if s_col and s_col != "statusstartdate":
        df.rename(columns={s_col: "statusstartdate"}, inplace=True)
    elif "statusstartdate" not in df.columns:
        df["statusstartdate"] = pd.NaT

    if e_col and e_col != "statusenddate":
        df.rename(columns={e_col: "statusenddate"}, inplace=True)
    elif "statusenddate" not in df.columns:
        df["statusenddate"] = pd.NaT

    for c in ("statusstartdate","statusenddate"):
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # ACTIVE / TERM
    status_cands = ["employeestatus","status","employmentstatus","workstatus","jobstatus"]
    present_stat = [c for c in status_cands if c in df.columns]

    def detect_estatus(row) -> str:
        for c in present_stat:
            val = str(row.get(c, "")).upper()
            if any(t in val for t in ("TERM", "TERMINAT", "SEPARAT", "ENDED", "INACTIVE")):
                return "TERM"
            if any(t in val for t in ("ACTIVE", "EMPLOY", "HIRED", "CURRENT")):
                return "ACTIVE"
        return ""

    # FT/PT (names + values + numerics)
    name_ftpt_keywords = ("FT", "FULL", "PART", "PT", "FTE", "HOUR", "STDHOUR", "STANDARDHOUR", "WEEKLYHOUR")
    role_like_cols = [c for c in df.columns if any(k in c.upper() for k in name_ftpt_keywords)]

    role_cands = set(role_like_cols) | set([
        "role","ftpt","ft_pt","employmenttype","employeetype","employeeclass","jobclass",
        "employmentcategory","fulltimeparttime","fte_status","fte","positiontype",
        "classification","class","standardhours","hoursperweek","avgweeklyhours","weeklyhours"
    ])
    role_cands = [c for c in role_cands if c in df.columns]

    FT_STR_TOKENS = ("FULL-TIME","FULLTIME"," FULL"," F/T"," F T"," FT","REGULAR FULL","SALARIED FULL","RFT")
    PT_STR_TOKENS = ("PART-TIME","PARTTIME"," PART"," P/T"," P T"," PT","RPT")

    def is_ft_numeric(val, colname: str) -> bool | None:
        try:
            v = float(val)
        except Exception:
            return None
        cu = colname.upper()
        if "FTE" in cu:
            return True if v >= 0.75 else (False if v > 0 else None)
        if any(k in cu for k in ("HOUR","STDHOUR","WEEKLY")):
            return True if v >= 30 else (False if 0 < v < 30 else None)
        return None

    def detect_role(row) -> str:
        for c in role_cands:
            val = str(row.get(c, "")).upper()
            if any(t in val for t in FT_STR_TOKENS): return "FULLTIME"
            if any(t in val for t in PT_STR_TOKENS): return "PARTTIME"
        for c in role_cands:
            res = is_ft_numeric(row.get(c, None), c)
            if res is True:  return "FULLTIME"
            if res is False: return "PARTTIME"
        for c in df.columns:
            val = str(row.get(c, "")).strip().upper()
            if val == "FT": return "FULLTIME"
            if val == "PT": return "PARTTIME"
        return ""

    df["_role_norm"] = df.apply(detect_role, axis=1)
    df["_estatus_norm"] = df.apply(detect_estatus, axis=1)

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

# ---- Wait Period overlap from Emp Wait Period
def _waiting_in_month(wait_df_emp: pd.DataFrame, ms: date, me: date) -> bool:
    """
    True if ANY wait window overlaps the month [ms, me].
    Each row: start = EffectiveDate, end = start + WaitPeriodDays - 1.
    """
    if wait_df_emp is None or wait_df_emp.empty:
        return False
    if "effectivedate" not in wait_df_emp.columns or "waitperiod" not in wait_df_emp.columns:
        return False
    s = pd.to_datetime(wait_df_emp["effectivedate"], errors="coerce").dt.date
    d = pd.to_numeric(wait_df_emp["waitperiod"], errors="coerce").fillna(0).astype(int)
    e = s + pd.to_timedelta((d.clip(lower=0) - 1).astype(int), unit="D")
    e = e.where(d > 0, s - timedelta(days=1))  # 0-day → non-wait
    return bool(((e >= ms) & (s <= me)).any())

# --- compatibility shim for main_fastapi.py ---
def preprocess_inputs(sheets: dict) -> dict:
    """
    Passthrough normalize hook.
    If you later need column aliasing / 'Emp Wait Period' merging, implement it here.
    For now it just returns the uploaded sheets unchanged.
    """
    return sheets

