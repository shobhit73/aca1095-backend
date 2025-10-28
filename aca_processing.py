# aca_processing.py
from __future__ import annotations

from typing import Optional, Tuple, Dict, List, Any
import io
import calendar
import pandas as pd

from debug_logging import get_logger, log_time
import aca_builder as builder
from aca_pdf import save_excel_outputs, fill_pdf_for_employee

log = get_logger("processing")

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
FULL_MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


# -------------------------
# Helpers
# -------------------------

def month_bounds(year: int, m: int):
    last = calendar.monthrange(year, m)[1]
    return pd.Timestamp(year, m, 1).date(), pd.Timestamp(year, m, last).date()

def _find_sheet(xls: pd.ExcelFile, candidates: List[str]) -> Optional[str]:
    names = xls.sheet_names
    lc_map = {n.lower().strip(): n for n in names}
    for c in candidates:
        if c in names:
            return c
        if c.lower().strip() in lc_map:
            return lc_map[c.lower().strip()]
    # contains-like
    for n in names:
        ln = n.lower()
        for c in candidates:
            if c.lower() in ln:
                return n
    return None

def _read_sheet(xls: pd.ExcelFile, candidates: List[str]) -> pd.DataFrame:
    name = _find_sheet(xls, candidates)
    if not name:
        return pd.DataFrame()
    return pd.read_excel(xls, sheet_name=name)

def choose_report_year(emp_elig_df: pd.DataFrame) -> Optional[int]:
    """
    Infer a filing year if the frontend didn't provide one.
    Priority: eligibility start/end -> enrollment start/end -> status dates.
    """
    def _year_from(df: pd.DataFrame, cols: List[str]) -> Optional[int]:
        for c in cols:
            if c in df.columns:
                try:
                    ys = pd.to_datetime(df[c], errors="coerce").dt.year.dropna().astype(int)
                    if not ys.empty:
                        # Pick most frequent to stabilize
                        return int(ys.mode().iloc[0])
                except Exception:
                    pass
        return None

    y = _year_from(emp_elig_df, ["eligibilitystartdate","EligibilityStartDate","EligibilityEndDate","eligibilityenddate"])
    return y

def _normalize_aliases_for_demo(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    # common aliases
    rename = {}
    for c in x.columns:
        lc = str(c).strip().lower()
        if lc in ("employee id","empid","emp id","id"):
            rename[c] = "EmployeeID"
        elif lc in ("first name","firstname","fname"):
            rename[c] = "FirstName"
        elif lc in ("middle name","middlename","mname"):
            rename[c] = "MiddleName"
        elif lc in ("last name","lastname","lname","surname"):
            rename[c] = "LastName"
        elif lc in ("ssn","ssn#","social","socialsecuritynumber"):
            rename[c] = "SSN"
        elif lc in ("address1","address line 1","addr1","address"):
            rename[c] = "Address1"
        elif lc in ("address2","address line 2","addr2"):
            rename[c] = "Address2"
        elif lc in ("city",):
            rename[c] = "City"
        elif lc in ("state","province","region"):
            rename[c] = "State"
        elif lc in ("zip","postal","zipcode"):
            rename[c] = "Zip"
        elif lc in ("statusstartdate","employmentstatusstartdate","estatusstartdate"):
            rename[c] = "StatusStartDate"
        elif lc in ("statusenddate","employmentstatusenddate","estatusenddate"):
            rename[c] = "StatusEndDate"
        elif lc in ("status","employmentstatus","empstatus","estatus"):
            rename[c] = "Status"
        elif lc in ("role","employeerole","emprole"):
            rename[c] = "Role"
    x = x.rename(columns=rename)
    return x


# -------------------------
# Public: Excel -> DataFrames / PDFs
# -------------------------

def load_excel(input_excel_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """
    Load workbook sheets into dataframes with minimal assumptions.
    """
    xls = pd.ExcelFile(io.BytesIO(input_excel_bytes))
    data = {
        "emp_demo":   _read_sheet(xls, ["Emp Demographic","Employee Demographic","Demographic"]),
        "emp_elig":   _read_sheet(xls, ["Emp Eligibility","Eligibility"]),
        "emp_enroll": _read_sheet(xls, ["Emp Enrollment","Enrollment"]),
        "dep_enroll": _read_sheet(xls, ["Dep Enrollment","Dependent Enrollment","Dependents"]),
        "emp_wait":   _read_sheet(xls, ["Emp Wait Period","Wait Period","Waiting Period"]),
    }
    return data


def build_outputs_from_excel(input_excel_bytes: bytes, filing_year: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Returns (interim, final, penalty, year_used)
    """
    with log_time(log, "build_outputs_from_excel"):
        sheets = load_excel(input_excel_bytes)
        emp_demo   = sheets["emp_demo"]
        emp_elig   = sheets["emp_elig"]
        emp_enroll = sheets["emp_enroll"]
        dep_enroll = sheets["dep_enroll"]
        emp_wait   = sheets.get("emp_wait", pd.DataFrame())

        year = int(filing_year) if filing_year else (choose_report_year(emp_elig) or pd.Timestamp.today().year)

        interim = builder.build_interim(emp_demo, emp_elig, emp_enroll, dep_enroll, year, emp_wait=emp_wait)
        final   = builder.build_final(interim)
        penalty = builder.build_penalty_dashboard(interim)

        return interim, final, penalty, year


def process_excel_to_workbook(input_excel_bytes: bytes, filing_year: Optional[int] = None) -> Tuple[bytes, Dict[str, Any]]:
    """
    Build (Interim, Final, Penalty) and return an .xlsx (bytes).
    Accepts both positional and keyword calls to save_excel_outputs().
    """
    interim, final, penalty, year = build_outputs_from_excel(input_excel_bytes, filing_year)
    # Both positional and keyword supported by save_excel_outputs
    xlsx_bytes = save_excel_outputs(interim, final, penalty, f"ACA Outputs ({year})")
    meta = {"year": year, "rows_interim": len(interim), "rows_final": len(final)}
    return xlsx_bytes, meta


# -------------------------
# PDF generation utilities
# -------------------------

def _final_maps_for_employee(final_df: pd.DataFrame, employee_id: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Extract month -> Line14, month -> Line16 for one employee from Final.
    """
    row = final_df[final_df["EmployeeID"].astype(str) == str(employee_id)]
    if row.empty:
        return {}, {}
    r = row.iloc[0]
    l14 = {
        1: r.get("Jan",""), 2: r.get("Feb",""), 3: r.get("Mar",""), 4: r.get("Apr",""),
        5: r.get("May",""), 6: r.get("Jun",""), 7: r.get("Jul",""), 8: r.get("Aug",""),
        9: r.get("Sep",""), 10: r.get("Oct",""), 11: r.get("Nov",""), 12: r.get("Dec",""),
    }
    l16 = {
        1: r.get("Line16_Jan",""), 2: r.get("Line16_Feb",""), 3: r.get("Line16_Mar",""), 4: r.get("Line16_Apr",""),
        5: r.get("Line16_May",""), 6: r.get("Line16_Jun",""), 7: r.get("Line16_Jul",""), 8: r.get("Line16_Aug",""),
        9: r.get("Line16_Sep",""), 10: r.get("Line16_Oct",""), 11: r.get("Line16_Nov",""), 12: r.get("Line16_Dec",""),
    }
    # Coerce to str
    l14 = {k: ("" if pd.isna(v) else str(v)) for k,v in l14.items()}
    l16 = {k: ("" if pd.isna(v) else str(v)) for k,v in l16.items()}
    return l14, l16


def _employee_pi_from_demo(demo_df: pd.DataFrame, employee_id: str) -> Dict[str, str]:
    """
    Get Part I (PI) fields from demographic.
    """
    if demo_df is None or demo_df.empty:
        return {"first_name":"","middle_name":"","last_name":"","ssn":"","address1":"","address2":"","city":"","state":"","zip":""}
    x = _normalize_aliases_for_demo(demo_df)
    row = x[x["EmployeeID"].astype(str) == str(employee_id)]
    if row.empty:
        return {"first_name":"","middle_name":"","last_name":"","ssn":"","address1":"","address2":"","city":"","state":"","zip":""}
    r = row.iloc[0]
    return {
        "first_name": str(r.get("FirstName", "") or ""),
        "middle_name": str(r.get("MiddleName", "") or ""),
        "last_name":  str(r.get("LastName", "") or ""),
        "ssn":        str(r.get("SSN", "") or ""),
        "address1":   str(r.get("Address1", "") or ""),
        "address2":   str(r.get("Address2", "") or ""),
        "city":       str(r.get("City", "") or ""),
        "state":      str(r.get("State", "") or ""),
        "zip":        str(r.get("Zip", "") or ""),
    }


def generate_single_pdf_from_excel(
    *,
    input_excel_bytes: bytes,
    blank_pdf_bytes: bytes,
    employee_id: Optional[str] = None,
    filing_year: Optional[int] = None,
    flatten: bool = True
) -> bytes:
    """
    Build outputs, pick employee (given or first), and render one PDF.
    Accepts both positional and keyword calling styles downstream.
    """
    with log_time(log, "generate_single_pdf_from_excel"):
        interim, final, penalty, year = build_outputs_from_excel(input_excel_bytes, filing_year)

        # Decide employee
        if employee_id is None:
            # Pick first EmployeeID from Final to be consistent
            if final.empty:
                raise ValueError("No employees found to generate PDF.")
            employee_id = str(final["EmployeeID"].iloc[0])

        l14_map, l16_map = _final_maps_for_employee(final, str(employee_id))
        if not l14_map:
            raise ValueError(f"No Final row found for EmployeeID={employee_id}")

        # Part I payload
        demo = load_excel(input_excel_bytes)["emp_demo"]
        employee_pi = _employee_pi_from_demo(demo, str(employee_id))

        # Positional call (backward compatible)
        flat_bytes, _editable = fill_pdf_for_employee(
            blank_pdf_bytes, employee_pi, l14_map, l16_map, None, bool(flatten)
        )
        return flat_bytes


def generate_bulk_pdfs_from_excel(
    *,
    input_excel_bytes: bytes,
    blank_pdf_bytes: bytes,
    filing_year: Optional[int] = None,
    flatten: bool = True
) -> List[Dict[str, Any]]:
    """
    Build outputs for all employees and return a list of PDF blobs with names.
    """
    with log_time(log, "generate_bulk_pdfs_from_excel"):
        interim, final, penalty, year = build_outputs_from_excel(input_excel_bytes, filing_year)
        if final.empty:
            return []

        demo = load_excel(input_excel_bytes)["emp_demo"]
        out: List[Dict[str, Any]] = []
        for _, row in final.iterrows():
            emp = str(row["EmployeeID"])
            l14_map, l16_map = _final_maps_for_employee(final, emp)
            pi = _employee_pi_from_demo(demo, emp)
            pdf_bytes, _ = fill_pdf_for_employee(
                blank_pdf_bytes, pi, l14_map, l16_map, None, bool(flatten)
            )
            # Safe filename
            fname = f"1095C_{emp}.pdf"
            out.append({"employee_id": emp, "filename": fname, "pdf_bytes": pdf_bytes})
        return out
