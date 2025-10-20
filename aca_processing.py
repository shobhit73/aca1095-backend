# aca_processing.py
from __future__ import annotations

import io
from calendar import monthrange
from typing import Dict, List, Tuple

import pandas as pd


# ------------------------------------------------------------
# Constants / small helpers
# ------------------------------------------------------------
MONTHS: List[str] = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def _coerce_str(x) -> str:
    """Safe string for IDs etc. Keeps digits, trims spaces; None -> ''."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # Excel sometimes gives '1001.0' for ids -> make '1001'
    if s.endswith(".0") and s.replace(".0", "").isdigit():
        s = s[:-2]
    return s


def month_bounds(year: int, month: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return (first_day, last_day) for a given year+month."""
    start = pd.Timestamp(year=year, month=month, day=1)
    end = pd.Timestamp(year=year, month=month, day=monthrange(year, month)[1])
    return start, end


def _norm(s: str) -> str:
    """Normalized key for column matching."""
    s = (s or "").strip().lower()
    for ch in [" ", "_", "-", ".", "/"]:
        s = s.replace(ch, "")
    return s


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.tz_localize(None)


def _rename_like(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Rename columns based on loose 'looks like' keys in mapping.

    mapping keys are *normalized* candidates (e.g. 'eligibilitystartdate'),
    values are the final standardized names you want to set.
    """
    out = df.copy()
    current = {_norm(c): c for c in out.columns}
    for want_norm, final_name in mapping.items():
        if want_norm in current:
            out.rename(columns={current[want_norm]: final_name}, inplace=True)
        else:
            # allow prefix matches like 'eligibilitystartda'
            hit = next((c for n, c in current.items() if n.startswith(want_norm)), None)
            if hit:
                out.rename(columns={hit: final_name}, inplace=True)
    return out


# ------------------------------------------------------------
# Load / prepare
# ------------------------------------------------------------
def load_excel(excel_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """Read Excel bytes and return raw sheets (dict)."""
    xls = pd.ExcelFile(io.BytesIO(excel_bytes))

    # Try to find sheets by loose names
    names = { _norm(n): n for n in xls.sheet_names }

    def _get(name_candidates: List[str]) -> pd.DataFrame:
        for cand in name_candidates:
            for k, real in names.items():
                if k == _norm(cand) or k.startswith(_norm(cand)):
                    return xls.parse(real)
        # not found -> empty df
        return pd.DataFrame()

    return {
        "emp_demo": _get(["Emp Demographic", "Demographic", "Employees", "EmpDemo"]),
        "emp_elig": _get(["Emp Eligibility", "Eligibility", "EmpElig"]),
        "emp_enroll": _get(["Emp Enrollment", "Enrollment", "EmpEnroll"]),
        "dep_enroll": _get(["Dep Enrollment", "Dependent Enrollment", "DepEnroll"]),
        "pay_deductions": _get(["Pay Deductions", "Deductions"]),  # optional; may be empty
        "scenarios": _get(["Scenarios", "Settings", "Config"]),     # optional
    }


def prepare_inputs(
    data: Dict[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize the incoming sheets and return the 6 frames expected by the API:
    (emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, pay_deductions)

    Notes
    -----
    * We keep `emp_status` as the same frame as `emp_demo` (it carries status spans).
    * We do not compute any month-by-month logic here; builder handles that.
    * We only clean/standardize columns, IDs, and dates.
    """

    # ---------- Emp Demographic (aka employment status) ----------
    demo = data.get("emp_demo", pd.DataFrame()).copy()
    demo = _rename_like(
        demo,
        {
            "employeeid": "employeeid",
            "employee": "employeeid",
            "statusstartdate": "statusstartdate",
            "statusenddate": "statusenddate",
            "empstatuscode": "empstatuscode",
            "employmentstatus": "employmentstatus",
            "role": "role",
            "payfrequency": "payfrequency",
        },
    )

    # Coerce essential columns
    if "employeeid" not in demo.columns:
        demo["employeeid"] = []
    demo["employeeid"] = demo["employeeid"].map(_coerce_str)

    for c in ["statusstartdate", "statusenddate"]:
        if c in demo.columns:
            demo[c] = _to_date(demo[c])

    # Ensure status columns exist
    for c in ["empstatuscode", "employmentstatus", "role", "payfrequency"]:
        if c not in demo.columns:
            demo[c] = pd.NA

    # ---------- Emp Eligibility ----------
    elig = data.get("emp_elig", pd.DataFrame()).copy()
    elig = _rename_like(
        elig,
        {
            "employeeid": "employeeid",
            "employee": "employeeid",
            "eligibilitystartdate": "eligibilitystartdate",
            "eligibilityenddate": "eligibilityenddate",
            "eligibleplan": "eligibleplan",
            "eligibletier": "eligibletier",
            "plancost": "plancost",
            "planco": "plancost",
        },
    )
    if "employeeid" not in elig.columns:
        elig["employeeid"] = []
    elig["employeeid"] = elig["employeeid"].map(_coerce_str)

    for c in ["eligibilitystartdate", "eligibilityenddate"]:
        if c in elig.columns:
            elig[c] = _to_date(elig[c])

    # normalize strings
    if "eligibleplan" in elig.columns:
        elig["eligibleplan"] = elig["eligibleplan"].fillna("").astype(str).str.strip()
    else:
        elig["eligibleplan"] = ""

    if "eligibletier" in elig.columns:
        elig["eligibletier"] = (
            elig["eligibletier"].fillna("").astype(str).str.strip().str.upper()
        )
    else:
        elig["eligibletier"] = ""

    if "plancost" not in elig.columns:
        elig["plancost"] = pd.NA

    # ---------- Emp Enrollment ----------
    enr = data.get("emp_enroll", pd.DataFrame()).copy()
    enr = _rename_like(
        enr,
        {
            "employeeid": "employeeid",
            "enrollmentstartdate": "enrollmentstartdate",
            "enrollmentenddate": "enrollmentenddate",
            "plancode": "plancode",
            "planname": "planname",
            "tier": "tier",
        },
    )
    if "employeeid" not in enr.columns:
        enr["employeeid"] = []
    enr["employeeid"] = enr["employeeid"].map(_coerce_str)

    for c in ["enrollmentstartdate", "enrollmentenddate"]:
        if c in enr.columns:
            enr[c] = _to_date(enr[c])

    # Normalize text fields
    for c in ["plancode", "planname", "tier"]:
        if c in enr.columns:
            enr[c] = enr[c].fillna("").astype(str).str.strip()
    if "tier" in enr.columns:
        enr["tier"] = enr["tier"].str.upper()

    # ---------- Dep Enrollment (optional) ----------
    dep = data.get("dep_enroll", pd.DataFrame()).copy()
    if not dep.empty:
        dep = _rename_like(
            dep,
            {
                "employeeid": "employeeid",
                "enrollmentstartdate": "enrollmentstartdate",
                "enrollmentenddate": "enrollmentenddate",
                "tier": "tier",
            },
        )
        if "employeeid" not in dep.columns:
            dep["employeeid"] = []
        dep["employeeid"] = dep["employeeid"].map(_coerce_str)
        for c in ["enrollmentstartdate", "enrollmentenddate"]:
            if c in dep.columns:
                dep[c] = _to_date(dep[c])
        if "tier" in dep.columns:
            dep["tier"] = dep["tier"].fillna("").astype(str).str.strip().str.upper()

    # ---------- Pay Deductions (optional) ----------
    pays = data.get("pay_deductions", pd.DataFrame()).copy()
    # Keep as-is; builder currently does not require this. Ensure it exists.
    if pays is None or isinstance(pays, float):
        pays = pd.DataFrame()

    # Return emp_demo AND emp_status (as same frame for compatibility)
    emp_demo = demo.copy()
    emp_status = demo.copy()

    # Make sure essential columns exist downstream
    for df, cols in [
        (emp_demo, ["employeeid", "statusstartdate", "statusenddate", "empstatuscode"]),
        (emp_status, ["employeeid", "statusstartdate", "statusenddate", "empstatuscode"]),
        (elig, ["employeeid", "eligibilitystartdate", "eligibilityenddate", "eligibleplan", "eligibletier", "plancost"]),
        (enr, ["employeeid", "enrollmentstartdate", "enrollmentenddate", "plancode", "planname", "tier"]),
    ]:
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA

    # Clean obvious “Waive” spelling differences / case
    if not enr.empty and "plancode" in enr.columns:
        enr["plancode"] = enr["plancode"].fillna("").astype(str).str.strip()

    return emp_demo, emp_status, elig, enr, dep, pays


def choose_report_year(emp_elig: pd.DataFrame) -> int:
    """Pick a filing/report year from the eligibility sheet.

    Strategy:
    1) Use the most common year among eligibility date ranges.
    2) If empty/ambiguous, fall back to the most common year in any date column present.
    3) If still unknown, use current year.
    """
    candidates: List[int] = []

    def add_years(series: pd.Series):
        if series.empty:
            return
        years = series.dropna().dt.year.tolist()
        candidates.extend([int(y) for y in years if pd.notna(y)])

    for c in ["eligibilitystartdate", "eligibilityenddate"]:
        if c in emp_elig.columns:
            add_years(emp_elig[c])

    # If plancost is actually a per-month amount tagged by year (rare), ignore it.
    if not candidates:
        # as a fallback try any datetime-looking columns
        for c in emp_elig.columns:
            if "date" in c.lower():
                try:
                    s = pd.to_datetime(emp_elig[c], errors="coerce")
                    add_years(s)
                except Exception:
                    pass

    if candidates:
        return pd.Series(candidates).mode().iloc[0]

    return pd.Timestamp.today().year
