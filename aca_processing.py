# aca_processing.py
from __future__ import annotations
import io
import pandas as pd
from typing import Dict, Tuple, List

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _coerce_str(x) -> str:
    try:
        return str(int(x))
    except Exception:
        return str(x)

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "").replace("_", "") for c in df.columns]
    return df

def _to_date(s):
    return pd.to_datetime(s, errors="coerce")

def load_excel(excel_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """Return dict of sheets by semantic names."""
    x = pd.ExcelFile(io.BytesIO(excel_bytes))
    sheets = {name.lower(): _norm_cols(pd.read_excel(x, name)) for name in x.sheet_names}

    # Try to map to expected keys
    def pick(*candidates):
        for c in candidates:
            if c in sheets:
                return sheets[c]
        return pd.DataFrame()

    return {
        "emp_demo": pick("empdemographic", "emp demographic", "employee", "employees", "demographic"),
        "emp_status": pick("empdemographic", "emp demographic", "employee", "employees", "demographic"),
        "emp_elig": pick("empeligibility", "emp eligibility", "eligibility"),
        "emp_enroll": pick("empenrollment", "emp enrollment", "enrollment"),
        "dep_enroll": pick("depenrollment", "dep enrollment", "dependentenrollment"),
    }

def prepare_inputs(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    emp_demo   = data["emp_demo"].copy()
    emp_status = data["emp_status"].copy()
    emp_elig   = data["emp_elig"].copy()
    emp_enroll = data["emp_enroll"].copy()
    dep_enroll = data["dep_enroll"].copy()

    # ---- Demographic / status (we use EmpStatusCode & date range) ----
    # expected: employeeid, statusstartdate, statusenddate, empstatuscode (FT/PT), employmentstatus ("Active")
    rename_demo = {
        "employeeid":"employeeid",
        "statusstartdate":"statusstartdate",
        "statusenddate":"statusenddate",
        "empstatuscode":"empstatuscode",
        "employmentstatus":"employmentstatus",
        "role":"role",
        "payfrequency":"payfrequency",
        "firstname":"firstname",
        "lastname":"lastname",
    }
    emp_demo = emp_demo.rename(columns=rename_demo)
    for c in ["employeeid","empstatuscode","employmentstatus","role","payfrequency"]:
        if c not in emp_demo.columns: emp_demo[c] = ""
    for c in ["statusstartdate","statusenddate"]:
        if c in emp_demo.columns:
            emp_demo[c] = _to_date(emp_demo[c])
        else:
            emp_demo[c] = pd.NaT
    emp_demo["employeeid"] = emp_demo["employeeid"].map(_coerce_str)

    # ---- Eligibility ----
    # expected: employee / employeeid, eligibilitystartdate, eligibilityenddate, eligibleplan, eligibletier, plancost
    ren_elig = {
        "employee":"employeeid",
        "employeeid":"employeeid",
        "eligibilitystartdate":"eligibilitystartdate",
        "eligibilityenddate":"eligibilityenddate",
        "eligibleplan":"eligibleplan",
        "eligibletier":"eligibletier",
        "plancost":"plancost",
        "plancode":"plancode"
    }
    emp_elig = emp_elig.rename(columns=ren_elig)
    for c in ["employeeid","eligibleplan","eligibletier"]:
        if c not in emp_elig.columns: emp_elig[c] = ""
    for c in ["eligibilitystartdate","eligibilityenddate"]:
        if c in emp_elig.columns:
            emp_elig[c] = _to_date(emp_elig[c])
        else:
            emp_elig[c] = pd.NaT
    if "plancost" not in emp_elig.columns:
        emp_elig["plancost"] = pd.NA
    emp_elig["employeeid"] = emp_elig["employeeid"].map(_coerce_str)

    # ---- Enrollment ----
    # expected: employeeid, enrollmentstartdate, enrollmentenddate, plancode, planname, tier
    ren_enr = {
        "employeeid":"employeeid",
        "enrollmentstartdate":"enrollmentstartdate",
        "enrollmentenddate":"enrollmentenddate",
        "plancode":"plancode",
        "planname":"planname",
        "tier":"tier",
    }
    emp_enroll = emp_enroll.rename(columns=ren_enr)
    for c in ["employeeid","plancode","tier"]:
        if c not in emp_enroll.columns: emp_enroll[c] = ""
    for c in ["enrollmentstartdate","enrollmentenddate"]:
        if c in emp_enroll.columns:
            emp_enroll[c] = _to_date(emp_enroll[c])
        else:
            emp_enroll[c] = pd.NaT
    emp_enroll["employeeid"] = emp_enroll["employeeid"].map(_coerce_str)

    # ---- Dep enrollment (optional) ----
    ren_dep = {
        "employeeid":"employeeid",
        "enrollmentstartdate":"enrollmentstartdate",
        "enrollmentenddate":"enrollmentenddate",
        "tier":"tier",
    }
    if not dep_enroll.empty:
        dep_enroll = dep_enroll.rename(columns=ren_dep)
        dep_enroll["employeeid"] = dep_enroll["employeeid"].map(_coerce_str)
        for c in ["enrollmentstartdate","enrollmentenddate"]:
            dep_enroll[c] = _to_date(dep_enroll[c])
    else:
        dep_enroll = pd.DataFrame(columns=["employeeid","enrollmentstartdate","enrollmentenddate","tier"])

    return emp_demo, emp_demo, emp_elig, emp_enroll, dep_enroll

def choose_report_year(emp_elig: pd.DataFrame,
                       emp_enroll: pd.DataFrame | None = None,
                       emp_status: pd.DataFrame | None = None) -> int:
    years: List[int] = []
    for df, cols in [
        (emp_elig, ["eligibilitystartdate","eligibilityenddate"]),
        (emp_enroll or pd.DataFrame(), ["enrollmentstartdate","enrollmentenddate"]),
        (emp_status or pd.DataFrame(), ["statusstartdate","statusenddate"]),
    ]:
        for c in cols:
            if c in df.columns:
                years += list(pd.to_datetime(df[c], errors="coerce").dt.year.dropna().astype(int).unique())
    years = [y for y in years if y > 1900]
    return max(years) if years else pd.Timestamp.today().year
