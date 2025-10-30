# pdf_filler.py
# Fill 1095-C PDFs from an Interim dataframe.
# - Works with builder output (EmployeeID, line14_final, line16_final, MonthNum/Month)
# - Accepts a JSON field-map (recommended) or uses a safe fallback mapping
# - Exposes:
#     * fill_pdf_for_employee(...bytes...) -> bytes
#     * generate_single_pdf(interim_df, year, template_path, fields_json_path, emp_id, demo_df=None, dep_df=None) -> bytes
#     * generate_all_pdfs(interim_df, year, template_path, fields_json_path, demo_df=None, dep_df=None) -> List[Tuple[str, bytes]]
#     * adapt_interim_from_builder(df)  # builder → filler schema adapter

from __future__ import annotations

import io
import json
from typing import Dict, List, Tuple, Optional

import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject, BooleanObject

# -----------------------------
# Month labels & normalization
# -----------------------------
MONTHS_3 = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MONTHS_LONG = ["January","February","March","April","May","June","July","August","September","October","November","December"]
MONTHS_ALT = {"Jun":"June","Jul":"July","Sep":"Sept"}  # many templates use "Sept"

def _norm_month_key(k: str) -> str:
    k = (k or "").strip()
    if k in MONTHS_3:
        return k
    if k.title() in MONTHS_LONG:
        idx = MONTHS_LONG.index(k.title())
        return MONTHS_3[idx]
    if k in MONTHS_ALT:
        return [kk for kk,v in MONTHS_ALT.items() if v == k][0]  # reverse
    # try first 3
    t = k[:3].title()
    return t if t in MONTHS_3 else k

# -----------------------------
# Builder → filler schema adapter
# -----------------------------
def adapt_interim_from_builder(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your builder outputs: EmployeeID, Month/MonthNum, line14_final, line16_final.
    Some older fillers used: Employee_ID, Month ("June"/"Sept"), line_14, line_16.
    This adapter makes the dataframe compatible with both worlds.
    """
    if df is None or df.empty:
        return df
    x = df.copy()
    # Column names
    rename_map = {
        "EmployeeID": "Employee_ID",
        "line14_final": "line_14",
        "line16_final": "line_16",
    }
    for old, new in rename_map.items():
        if old in x.columns and new not in x.columns:
            x = x.rename(columns={old: new})

    # Normalize Month column to 3-letter, then to template’s label if needed later
    if "Month" in x.columns:
        x["Month"] = x["Month"].astype(str).map(_norm_month_key)
    elif "MonthNum" in x.columns:
        x["Month"] = x["MonthNum"].astype(int).map(lambda n: MONTHS_3[n-1] if 1 <= n <= 12 else "")

    return x

# -----------------------------
# Field map (JSON) handling
# -----------------------------
DEFAULT_MAP: Dict[str, object] = {
    # Part I (customize to your template field names via JSON)
    "part1": {
        "employee_name": "EmployeeName",   # fallback field
        "employee_id": "EmployeeID",       # optional
        "ssn": "SSN",                      # optional
    },
    # Month fields: each month has keys L14 and L16 pointing to field names
    "months": {
        m: {"L14": f"L14_{m}", "L16": f"L16_{m}"} for m in MONTHS_3
    },
    # Part III (optional): treated leniently; skipped if fields don’t exist
    "part3": {
        "all_year_checkbox": "PartIII_AllYear",
        "dep_rows": [
            {"name": "Dep1_Name", "ssn": "Dep1_SSN", "months_all_year": "Dep1_AllYear"},
            {"name": "Dep2_Name", "ssn": "Dep2_SSN", "months_all_year": "Dep2_AllYear"},
            {"name": "Dep3_Name", "ssn": "Dep3_SSN", "months_all_year": "Dep3_AllYear"},
        ],
    },
}

def _load_field_map(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return DEFAULT_MAP
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # shallow-merge onto defaults so missing keys don’t crash
            m = json.loads(json.dumps(DEFAULT_MAP))  # deep copy
            for k, v in (data or {}).items():
                m[k] = v
            return m
    except Exception:
        return DEFAULT_MAP

# -----------------------------
# Lower-level PDF helpers
# -----------------------------
def _enable_need_appearances(writer: PdfWriter) -> None:
    # Ensure viewers render appearances
    root = writer._root_object  # type: ignore[attr-defined]
    if "/AcroForm" not in root:
        root.update({NameObject("/AcroForm"): writer._add_object({})})
    acro_form = root["/AcroForm"]
    acro_form.update({NameObject("/NeedAppearances"): BooleanObject(True)})

def _update_fields_on_page(writer: PdfWriter, page, field_values: Dict[str, str]) -> None:
    # PyPDF2 convenience (works on 3.0+). This won’t crash if field names are missing.
    try:
        writer.update_page_form_field_values(page, field_values)
    except Exception:
        pass

def _flatten_annotations(page) -> None:
    # Minimal "flatten": remove annotation actions while preserving printed values.
    # (Visual flattening is viewer-dependent; NeedAppearances helps.)
    try:
        if "/Annots" in page:
            del page["/Annots"]
    except Exception:
        pass

# -----------------------------
# Single-employee filler (bytes API)
# -----------------------------
def fill_pdf_for_employee(
    blank_pdf_bytes=None,
    employee_pi: Dict[str, str] = None,
    line14_by_month: Dict[str, str] = None,
    line16_by_month: Dict[str, str] = None,
    covered_individuals: List[Dict[str, str]] = None,
    flatten: bool = True,
    *,
    field_map_path: Optional[str] = None,
):
    """
    Primary bytes-based API used by the FastAPI routes.
    Accepts kwargs, but will also work if called positionally (older code).
    Returns: bytes of the filled PDF.
    """
    # --- positional fallback support ----------------------------------------
    if not isinstance(blank_pdf_bytes, (bytes, bytearray)) and isinstance(employee_pi, (bytes, bytearray)):
        # called in positional style; re-map
        args = [blank_pdf_bytes, employee_pi, line14_by_month, line16_by_month, covered_individuals, flatten]
        blank_pdf_bytes = args[0]
        employee_pi = args[1]
        line14_by_month = args[2]
        line16_by_month = args[3]
        covered_individuals = args[4] if len(args) >= 5 else None
        flatten = args[5] if len(args) >= 6 else True
    # ------------------------------------------------------------------------

    employee_pi = employee_pi or {}
    line14_by_month = line14_by_month or {}
    line16_by_month = line16_by_month or {}
    covered_individuals = covered_individuals or []

    fmap = _load_field_map(field_map_path)

    # Build field → value dictionary
    fvals: Dict[str, str] = {}

    # Part I
    p1 = fmap.get("part1", {})
    name_field = p1.get("employee_name", "")
    emp_name = employee_pi.get("name") or employee_pi.get("employee_name") or ""
    if name_field:
        fvals[str(name_field)] = str(emp_name)
    id_field = p1.get("employee_id", "")
    if id_field and "employee_id" in employee_pi:
        fvals[str(id_field)] = str(employee_pi["employee_id"])
    ssn_field = p1.get("ssn", "")
    if ssn_field and "ssn" in employee_pi:
        fvals[str(ssn_field)] = str(employee_pi["ssn"])

    # Months (Line 14 / Line 16)
    months_map: Dict[str, Dict[str, str]] = fmap.get("months", {})  # { "Jan": {"L14": "...", "L16": "..."}, ... }
    for mk, pair in months_map.items():
        m3 = _norm_month_key(mk)
        fld14 = str(pair.get("L14", ""))
        fld16 = str(pair.get("L16", ""))
        if fld14:
            v14 = line14_by_month.get(m3) or line14_by_month.get(MONTHS_ALT.get(m3, ""), "")
            fvals[fld14] = str(v14 or "")
        if fld16:
            v16 = line16_by_month.get(m3) or line16_by_month.get(MONTHS_ALT.get(m3, ""), "")
            fvals[fld16] = str(v16 or "")

    # Part III (optional; lenient)
    p3 = fmap.get("part3", {})
    dep_rows = p3.get("dep_rows", [])
    for i, dep in enumerate(covered_individuals[:len(dep_rows)]):
        dr = dep_rows[i]
        if dr.get("name"):
            fvals[str(dr["name"])] = str(dep.get("name", ""))
        if dr.get("ssn") and dep.get("ssn"):
            fvals[str(dr["ssn"])] = str(dep["ssn"])
        if dr.get("months_all_year"):
            # simple: if dependent is covered all 12 months, tick the all-year checkbox (assume "Yes")
            if dep.get("all_year") in (True, "true", "True", "YES", "Yes", "yes", 1, "1"):
                fvals[str(dr["months_all_year"])] = "Yes"

    # Fill PDF
    reader = PdfReader(io.BytesIO(blank_pdf_bytes))
    writer = PdfWriter()

    # ensure form rendering
    _enable_need_appearances(writer)

    # copy pages while updating fields
    for page in reader.pages:
        writer.add_page(page)
        _update_fields_on_page(writer, writer.pages[-1], fvals)
        if flatten:
            _flatten_annotations(writer.pages[-1])

    # carry over AcroForm if present
    try:
        if "/AcroForm" in reader.trailer["/Root"]:
            writer._root_object.update({NameObject("/AcroForm"): reader.trailer["/Root"]["/AcroForm"]})
    except Exception:
        pass

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()

# -----------------------------
# Bulk / Single convenience (file-path API)
# -----------------------------
def _ensure_builder_months(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure we have a 3-letter Month column and 12 rows per Employee_ID if present.
    """
    x = adapt_interim_from_builder(df)
    if "Month" not in x.columns and "MonthNum" in x.columns:
        x["Month"] = x["MonthNum"].astype(int).map(lambda n: MONTHS_3[n-1] if 1 <= n <= 12 else "")
    return x

def _payload_from_rows(g: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    g = g.sort_values("Month").copy()
    employee_pi = {
        "employee_id": str(g["Employee_ID"].iloc[0]) if "Employee_ID" in g.columns else "",
        "name": str(g["Name"].iloc[0]) if "Name" in g.columns else "",
    }
    line14 = {m: "" for m in MONTHS_3}
    line16 = {m: "" for m in MONTHS_3}
    for _, r in g.iterrows():
        m = _norm_month_key(str(r.get("Month", "")))
        if m in MONTHS_3:
            if "line_14" in r:
                line14[m] = str(r["line_14"] or "")
            elif "line14_final" in r:
                line14[m] = str(r["line14_final"] or "")
            if "line_16" in r:
                line16[m] = str(r["line_16"] or "")
            elif "line16_final" in r:
                line16[m] = str(r["line16_final"] or "")
    return employee_pi, line14, line16

def generate_single_pdf(
    interim_df: pd.DataFrame,
    year: int,
    template_path: str,
    fields_json_path: Optional[str],
    emp_id: str,
    demo_df: pd.DataFrame | None = None,
    dep_df: pd.DataFrame | None = None,
) -> bytes:
    """
    Filter the interim DF to a single employee and produce one filled PDF (bytes).
    """
    if interim_df is None or interim_df.empty:
        return b""
    x = _ensure_builder_months(interim_df)
    sub = x[x["Employee_ID"].astype(str) == str(emp_id)]
    if sub.empty:
        return b""

    employee_pi, line14_map, line16_map = _payload_from_rows(sub)

    # read template
    with open(template_path, "rb") as f:
        blank = f.read()

    return fill_pdf_for_employee(
        blank_pdf_bytes=blank,
        employee_pi=employee_pi,
        line14_by_month=line14_map,
        line16_by_month=line16_map,
        covered_individuals=[],           # wire in if you later map Part III
        flatten=True,
        field_map_path=fields_json_path,
    )

def generate_all_pdfs(
    interim_df: pd.DataFrame,
    year: int,
    template_path: str,
    fields_json_path: Optional[str],
    demo_df: pd.DataFrame | None = None,
    dep_df: pd.DataFrame | None = None,
) -> List[Tuple[str, bytes]]:
    """
    Produce PDFs for all employees present in the interim DF.
    Returns list of (employee_id, pdf_bytes).
    """
    results: List[Tuple[str, bytes]] = []
    if interim_df is None or interim_df.empty:
        return results

    x = _ensure_builder_months(interim_df)
    with open(template_path, "rb") as f:
        blank = f.read()

    for emp_id, g in x.groupby(x["Employee_ID"].astype(str), sort=False):
        employee_pi, line14_map, line16_map = _payload_from_rows(g)
        try:
            pdf_bytes = fill_pdf_for_employee(
                blank_pdf_bytes=blank,
                employee_pi=employee_pi,
                line14_by_month=line14_map,
                line16_by_month=line16_map,
                covered_individuals=[],
                flatten=True,
                field_map_path=fields_json_path,
            )
            results.append((str(emp_id), pdf_bytes))
        except Exception:
            # Continue on error for a single employee
            results.append((str(emp_id), b""))
    return results
