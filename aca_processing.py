# aca_processing.py
from __future__ import annotations

import io, re
from datetime import datetime, date, timedelta
from typing import Tuple, List

import numpy as np
import pandas as pd

TRUTHY = {"y","yes","true","t","1",1,True}
FALSY  = {"n","no","false","f","0",0,False,None,np.nan}

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
FULL_MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]

CANON_ALIASES = {
    "employee id": "employeeid",
    "empid": "employeeid",
    "emp id": "employeeid",
    "first name": "firstname",
    "last name": "lastname",
    "address 1": "addressline1",
    "address1": "addressline1",
    "address 2": "addressline2",
    "address2": "addressline2",
    "zip": "zipcode",
    "zip code": "zipcode",
    "employment status": "employmentstatus",
    "status start date": "statusstartdate",
    "status end date": "statusenddate",

    "eligibleplan": "plancode",
    "eligibletier": "eligibilitytier",
    "tier": "enrollmenttier",

    "is eligible for coverage": "iseligibleforcoverage",
    "eligibility start date": "eligibilitystartdate",
    "eligibility end date": "eligibilityenddate",

    "is enrolled": "isenrolled",
    "enrollment start date": "enrollmentstartdate",
    "enrollment end date": "enrollmentenddate",

    "eligible start date": "eligiblestartdate",
    "eligible end date": "eligibleenddate",
}

EXPECTED_SHEETS = {
    "emp demographic": [
        "employeeid","firstname","lastname","ssn","addressline1","addressline2",
        "city","state","zipcode","role","employmentstatus","statusstartdate","statusenddate"
    ],
    "emp eligibility": [
        "employeeid","iseligibleforcoverage","eligibilitystartdate","eligibilityenddate",
        "plancode","eligibilitytier","plancost"
    ],
    "emp enrollment": [
        "employeeid","isenrolled","enrollmentstartdate","enrollmentenddate",
        "plancode","planname","enrollmenttier"
    ],
    "dep enrollment": [
        "employeeid","dependentrelationship","eligible","enrolled",
        "eligiblestartdate","eligibleenddate","enrollmentstartdate","enrollmentenddate",
        "plancode"
    ]
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _normalize_employeeid(x) -> str:
    if x is None or (isinstance(x,float) and np.isnan(x)): return ""
    s = str(x).strip().replace(",", "")
    if s == "" or s.lower() in {"nan","none"}: return ""
    m = re.fullmatch(r"(\d+)\.0+", s)
    if m: return m.group(1)
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s

def to_bool(val) -> bool:
    if isinstance(val, str):
        v = val.strip().lower()
        if v in TRUTHY: return True
        if v in FALSY:  return False
    return bool(val) and val not in FALSY

def parse_date_safe(d, default_end: bool=False):
    if pd.isna(d) or d is None:
        return None
    if isinstance(d, (pd.Timestamp, datetime, date)):
        return d
    s = str(d).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d","%m/%d/%Y","%d/%m/%Y","%d-%b-%Y","%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    try:
        n = float(s)
        base = datetime(1899,12,30)
        return base + timedelta(days=n)
    except Exception:
        return None

def _ensure_employeeid_str(df):
    if df.empty or "employeeid" not in df.columns: return df
    df = df.copy()
    df["employeeid"] = df["employeeid"].map(_normalize_employeeid)
    return df

def _parse_date_cols(df, cols, default_end_cols=()):
    if df.empty: return df
    df = df.copy(); endset = set(default_end_cols)
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: parse_date_safe(x, default_end=c in endset))
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _boolify(df, cols):
    if df.empty: return df
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(to_bool)
    return df

def load_excel(file_bytes: bytes) -> dict:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    out = {}
    for raw in xls.sheet_names:
        df = pd.read_excel(xls, raw)
        df = normalize_columns(df)
        df = df.rename(columns={k:v for k,v in CANON_ALIASES.items() if k in df.columns})
        if "employeeid" in df.columns:
            df["employeeid"] = df["employeeid"].map(_normalize_employeeid)
        out[raw.strip().lower()] = df
    return out

def _pick_sheet(data: dict, key: str) -> pd.DataFrame:
    if key in data: return data[key]
    for k in data:
        if key in k: return data[k]
    return pd.DataFrame()

def month_bounds(year:int, month:int) -> tuple[date, date]:
    ms = date(int(year), int(month), 1)
    me = date(year, 12, 31) if month==12 else (date(year, month+1, 1) - timedelta(days=1))
    return (ms, me)

def _norm_token(x) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())

def _any_overlap(df: pd.DataFrame, start_col: str, end_col: str, ms: date, me: date, *, mask=None) -> bool:
    """
    True if ANY row overlaps the month [ms, me].
    Optional `mask` filters rows first (boolean Series aligned to df.index).
    """
    if df is None or df.empty:
        return False
    if start_col not in df.columns or end_col not in df.columns:
        return False

    if mask is None:
        mask = pd.Series(True, index=df.index)

    s = pd.to_datetime(df[start_col], errors="coerce").dt.date.fillna(date(1900, 1, 1))
    e = pd.to_datetime(df[end_col], errors="coerce").dt.date.fillna(date(9999, 12, 31))

    hits = (e >= ms) & (s <= me) & mask
    return bool(hits.any())


def _all_month(df: pd.DataFrame, start_col: str, end_col: str, ms: date, me: date, *, mask=None) -> bool:
    """
    True if ANY row covers the ENTIRE month [ms, me] (i.e., start <= ms AND end >= me).
    Optional `mask` filters rows first (boolean Series aligned to df.index).
    """
    if df is None or df.empty:
        return False
    if start_col not in df.columns or end_col not in df.columns:
        return False

    if mask is None:
        mask = pd.Series(True, index=df.index)

    s = pd.to_datetime(df[start_col], errors="coerce").dt.date.fillna(date(1900, 1, 1))
    e = pd.to_datetime(df[end_col], errors="coerce").dt.date.fillna(date(9999, 12, 31))

    covers = mask & (s <= ms) & (e >= me)
    return bool(covers.any()))

def _status_from_demographic(emp_demo: pd.DataFrame) -> pd.DataFrame:
    need = {"employeeid","role","employmentstatus","statusstartdate","statusenddate"}
    if emp_demo.empty or not need <= set(emp_demo.columns):
        return pd.DataFrame(columns=list(need))
    st = emp_demo.loc[:, list(need)].copy()
    st["employeeid"] = st["employeeid"].map(_normalize_employeeid)
    st["role"] = st["role"].astype(str).str.strip()
    st["employmentstatus"] = st["employmentstatus"].astype(str).str.strip()
    st["_role_norm"] = st["role"].map(_norm_token)
    st["_estatus_norm"] = st["employmentstatus"].map(_norm_token)
    st = _parse_date_cols(st, ["statusstartdate","statusenddate"], default_end_cols=["statusenddate"])
    return st

def _collect_employee_ids(*dfs):
    ids=set()
    for df in dfs:
        if df is None or df.empty: continue
        if "employeeid" in df.columns:
            ids.update(map(_normalize_employeeid, df["employeeid"].dropna().tolist()))
    return sorted(ids)

def prepare_inputs(data: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cleaned = {}
    for sheet, cols in EXPECTED_SHEETS.items():
        df = _pick_sheet(data, sheet)
        if df.empty:
            cleaned[sheet] = pd.DataFrame(columns=cols); continue

        for misspell, canon in CANON_ALIASES.items():
            if misspell in df.columns and canon not in df.columns:
                df = df.rename(columns={misspell: canon})
        df = _ensure_employeeid_str(df)

        if sheet == "emp eligibility":
            if "eligibilitytier" in df.columns:
                df["eligibilitytier"] = df["eligibilitytier"].astype(str).str.strip()
            for c in ("plancode","planname"):
                if c in df.columns: df[c] = df[c].astype(str).str.strip()
            df = _boolify(df, ["iseligibleforcoverage"])
            if "plancost" in df.columns:
                df["plancost"] = pd.to_numeric(df["plancost"], errors="coerce")
            df = _parse_date_cols(df, ["eligibilitystartdate","eligibilityenddate"],
                                  default_end_cols=["eligibilityenddate"])

        elif sheet == "emp enrollment":
            if "tier" in df.columns and "enrollmenttier" not in df.columns:
                df["enrollmenttier"] = df["tier"]
            if "enrollmenttier" in df.columns:
                df["enrollmenttier"] = df["enrollmenttier"].astype(str).str.strip()
            for c in ("plancode","planname"):
                if c in df.columns: df[c] = df[c].astype(str).str.strip()
            df = _boolify(df, ["isenrolled"])
            df = _parse_date_cols(df, ["enrollmentstartdate","enrollmentenddate"],
                                  default_end_cols=["enrollmentenddate"])

        elif sheet == "dep enrollment":
            if "dependentrelationship" in df.columns:
                df["dependentrelationship"] = df["dependentrelationship"].astype(str).str.strip().str.title()
            df = _boolify(df, ["eligible","enrolled"])
            df = _parse_date_cols(
                df,
                ["eligiblestartdate","eligibleenddate","enrollmentstartdate","enrollmentenddate"],
                default_end_cols=["eligibleenddate","enrollmentenddate"]
            )
            if "plancode" in df.columns:
                df["plancode"] = df["plancode"].astype(str).str.strip()

        cleaned[sheet] = df

    return (cleaned["emp demographic"], cleaned["emp eligibility"],
            cleaned["emp enrollment"], cleaned["dep enrollment"])

def choose_report_year(emp_elig: pd.DataFrame, fallback_to_current=True) -> int:
    if emp_elig.empty or not {"eligibilitystartdate","eligibilityenddate"} <= set(emp_elig.columns):
        return datetime.now().year if fallback_to_current else 2024
    counts={}
    for _, r in emp_elig.iterrows():
        for x in (
            pd.to_datetime(r.get("eligibilitystartdate"), errors="coerce"),
            pd.to_datetime(r.get("eligibilityenddate"), errors="coerce")
        ):
            if pd.notna(x):
                y = int(x.year)
                counts[y] = counts.get(y, 0) + 1
    if counts:
        return max(sorted(counts), key=lambda y: (counts[y], y))
    return datetime.now().year if fallback_to_current else 2024

def _coerce_str(x) -> str:
    return "" if x is None or (isinstance(x,float) and np.isnan(x)) else str(x).strip()
