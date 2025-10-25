# aca_pdf.py

from __future__ import annotations
import io
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject

from aca_processing import MONTHS, _coerce_str, month_bounds

# ----------------------------- small helpers -----------------------------
def _safe_int(x, default=None):
    try:
        f = float(x)
        if np.isnan(f):  # type: ignore[arg-type]
            return default
        return int(f)
    except Exception:
        return default

def normalize_ssn_digits(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if "X" in s.upper():
        return s  # already masked
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 9:
        return f"{digits[0:3]}-{digits[3:5]}-{digits[5:9]}"
    return s

# ----------------------------- PDF field helpers -----------------------------
def _all_fields(reader: PdfReader) -> Dict[str, dict]:
    fields = reader.get_fields() or {}
    out: Dict[str, dict] = {}
    for k, v in fields.items():
        rec = {kk: v.get(kk) for kk in list(v.keys())}
        out[k] = rec
    return out

def _checkbox_on_name(widget: dict) -> NameObject:
    try:
        ap = widget.get("/AP")
        if ap and "/N" in ap:
            n_dict = ap["/N"]
            for key in n_dict.keys():
                if str(key) != "/Off":
                    return NameObject(str(key))
    except Exception:
        pass
    return NameObject("/Yes")

def _set_checkbox_on(page, field_name: str):
    if "/Annots" not in page:
        return
    for annot in page["/Annots"]:
        try:
            obj = annot.get_object()
        except Exception:
            obj = annot
        if obj.get("/T") == field_name:
            on_name = _checkbox_on_name(obj)
            obj.update({NameObject("/V"): on_name, NameObject("/AS"): on_name})
            return

def _update_text(page, name_to_value: Dict[str, str]):
    try:
        PdfWriter().update_page_form_field_values(page, name_to_value)
    except Exception:
        if "/Annots" in page:
            for annot in page["/Annots"]:
                try:
                    obj = annot.get_object()
                except Exception:
                    obj = annot
                nm = obj.get("/T")
                if nm in name_to_value:
                    obj.update({NameObject("/V"): name_to_value[nm]})

def _set_need_appearances(writer: PdfWriter):
    try:
        root = writer._root_object  # type: ignore[attr-defined]
        if "/AcroForm" in root:
            root["/AcroForm"].update({NameObject("/NeedAppearances"): NameObject("/true")})
    except Exception:
        pass

def _flatten(writer: PdfWriter) -> PdfWriter:
    out = PdfWriter()
    for p in writer.pages:
        page = p
        if "/Annots" in page:
            del page[NameObject("/Annots")]
        out.add_page(page)
    if "/AcroForm" in out._root_object:
        del out._root_object[NameObject("/AcroForm")]
    return out

# ----------------------------- Part III discovery -----------------------------
@dataclass
class Part3RowRefs:
    text_fields: List[str]   # [last, first, mi, ssn, dob] order resolved below
    month_boxes: List[str]   # [All12, Jan..Dec]

def _discover_part3_rows(reader: PdfReader) -> List[Part3RowRefs]:
    fields = _all_fields(reader)
    text_by_row: Dict[str, List[str]] = {}
    box_by_row: Dict[str, List[str]] = {}

    for name, rec in fields.items():
        if name.startswith("f3_"):  # text inputs
            parent = rec.get("/Parent", {})
            row = parent.get("/T") if isinstance(parent, dict) else str(parent.get("/T"))
            row = str(row) if row else "Row?"
            text_by_row.setdefault(row, []).append(name)
        elif name.startswith("c3_"):  # checkboxes
            parent = rec.get("/Parent", {})
            row = parent.get("/T") if isinstance(parent, dict) else str(parent.get("/T"))
            row = str(row) if row else "Row?"
            box_by_row.setdefault(row, []).append(name)

    def _sort_key(n: str) -> int:
        try:
            return int(n.split("_")[1].split("[")[0])
        except Exception:
            return 9999

    rows: List[Part3RowRefs] = []
    for row_name in sorted(text_by_row.keys(), key=lambda r: int(r.replace("Row", "").strip("[]") or "999")):
        texts = sorted(text_by_row.get(row_name, []), key=_sort_key)
        boxes = sorted(box_by_row.get(row_name, []), key=_sort_key)
        rows.append(Part3RowRefs(text_fields=texts, month_boxes=boxes))
    return rows

# ----------------------------- Month coverage math -----------------------------
def _months_from_periods(periods: List[Tuple[date, date]], year: int) -> Tuple[bool, List[bool]]:
    jan1, dec31 = date(year, 1, 1), date(year, 12, 31)
    covered = [False] * 12
    for (s, e) in periods:
        s = max(s, jan1)
        e = min(e, dec31)
        if e < jan1 or s > dec31:
            continue
        for m in range(1, 13):
            ms, me = month_bounds(year, m)
            if not (e < ms or s > me):
                covered[m - 1] = True
    all12 = all(covered)
    return all12, covered

# ----------------------------- helpers for Part I & II -----------------------------
def _f1_block(start: int, count: int) -> List[str]:
    return [f"f1_{i}[0]" for i in range(start, start + count)]

# ----------------------------- Public API -----------------------------
def fill_pdf_for_employee(
    pdf_bytes: bytes,
    emp_row: pd.Series,
    final_df_emp: pd.DataFrame,
    year_used: int,
    emp_enroll_emp: Optional[pd.DataFrame] = None,
    dep_enroll_emp: Optional[pd.DataFrame] = None,
):
    """
    Fill 1095-C PDF (Parts I, II, and PART III) using field names discovered from your PDF.
    Returns: (editable_name, editable_bytes, flattened_name, flattened_bytes)
    """

    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    for p in reader.pages:
        writer.add_page(p)

    # ---------------- Part I (employee info) ----------------
    first = _coerce_str(emp_row.get("firstname"))
    mi    = _coerce_str(emp_row.get("middleinitial"))
    last  = _coerce_str(emp_row.get("lastname"))
    ssn   = normalize_ssn_digits(_coerce_str(emp_row.get("ssn")))
    addr1 = _coerce_str(emp_row.get("addressline1"))
    addr2 = _coerce_str(emp_row.get("addressline2"))
    city  = _coerce_str(emp_row.get("city"))
    state = _coerce_str(emp_row.get("state"))
    zipcode = _coerce_str(emp_row.get("zipcode"))

    # From your field dump: f1_1..f1_8 live in EmployeeName[0] (Part I)  ⟶ :contentReference[oaicite:1]{index=1}
    part1_values = {
        "f1_1[0]": last,                         # Last name
        "f1_2[0]": first,                        # First name
        "f1_3[0]": mi,                           # MI
        "f1_4[0]": ssn,                          # SSN
        "f1_5[0]": addr1,                        # Address
        "f1_6[0]": addr2,                        # Address 2
        "f1_7[0]": city,                         # City
        "f1_8[0]": f"{state} {zipcode}".strip(), # State + ZIP
    }
    _update_text(writer.pages[0], part1_values)

    # ---------------- Part II (Line 14 + Line 16) ----------------
    # Your dump shows:
    # - f1_17[0] in Row1 (treat as "All 12" for Line 14)
    # - f1_18[0]..f1_29[0] are Jan..Dec (Line 14 by month)
    # - f1_44[0]..f1_55[0] are Jan..Dec (Line 16 by month)       ⟶ :contentReference[oaicite:2]{index=2}
    l14_all12_name = "f1_17[0]"
    l14_month_fields = _f1_block(18, 12)   # f1_18..f1_29
    l16_month_fields = _f1_block(44, 12)   # f1_44..f1_55

    m_to_l14: Dict[str, str] = {}
    m_to_l16: Dict[str, str] = {}
    for _, r in final_df_emp.iterrows():
        m = _coerce_str(r.get("Month"))[:3]
        if not m:
            continue
        m_to_l14[m] = _coerce_str(r.get("Line14_Final"))
        m_to_l16[m] = _coerce_str(r.get("Line16_Final"))

    # Line 14: If all months are identical & non-empty, fill "All 12" cell; else fill per month
    l14_vals = [m_to_l14.get(m, "") for m in MONTHS]
    if l14_vals and all(v == l14_vals[0] and v for v in l14_vals):
        _update_text(writer.pages[0], {l14_all12_name: l14_vals[0]})
    else:
        _update_text(writer.pages[0], {fld: val for fld, val in zip(l14_month_fields, l14_vals)})

    # Line 16 (monthly)
    l16_vals = [m_to_l16.get(m, "") for m in MONTHS]
    _update_text(writer.pages[0], {fld: val for fld, val in zip(l16_month_fields, l16_vals)})

    # ---------------- Part III (Covered Individuals) ----------------
    covered_rows: List[Tuple[str, str, str, str, Tuple[bool, List[bool]]]] = []

    # Employee enrollment (from tables if present) else derive from L16==2C
    emp_months_enrolled = [False] * 12
    if emp_enroll_emp is not None and not emp_enroll_emp.empty:
        periods: List[Tuple[date, date]] = []
        for _, rr in emp_enroll_emp.iterrows():
            plan_code = _coerce_str(rr.get("plancode") or rr.get("PlanCode"))
            if plan_code and plan_code.strip().lower() == "waive":
                continue
            s = rr.get("enrollmentstartdate") or rr.get("EnrollmentStartDate")
            e = rr.get("enrollmentenddate") or rr.get("EnrollmentEndDate")
            if pd.isna(s) or pd.isna(e):
                continue
            periods.append((pd.to_datetime(s).date(), pd.to_datetime(e).date()))
        if periods:
            all12, emp_months_enrolled = _months_from_periods(periods, year_used)
            covered_rows.append((first, mi, last, ssn, (all12, emp_months_enrolled)))
    else:
        emp_months_enrolled = [(_coerce_str(m_to_l16.get(m, "")) == "2C") for m in MONTHS]
        if any(emp_months_enrolled):
            all12 = all(emp_months_enrolled)
            covered_rows.append((first, mi, last, ssn, (all12, emp_months_enrolled)))

    # Dependents (skip WAIVE)
    if dep_enroll_emp is not None and not dep_enroll_emp.empty:
        for _, rr in dep_enroll_emp.iterrows():
            plan_code = _coerce_str(rr.get("plancode") or rr.get("PlanCode"))
            if plan_code and plan_code.strip().lower() == "waive":
                continue
            dep_first = _coerce_str(rr.get("depfirstname") or rr.get("DepFirstName"))
            dep_mi    = _coerce_str(rr.get("depmidname") or rr.get("DepMidName"))
            dep_last  = _coerce_str(rr.get("deplastname") or rr.get("DepLastName"))
            dep_ssn   = ""  # not in your inputs
            s = rr.get("enrollmentstartdate") or rr.get("EnrollmentStartDate")
            e = rr.get("enrollmentenddate") or rr.get("EnrollmentEndDate")
            if pd.isna(s) or pd.isna(e):
                continue
            all12, months = _months_from_periods([(pd.to_datetime(s).date(), pd.to_datetime(e).date())], year_used)
            if any(months):
                covered_rows.append((dep_first, dep_mi, dep_last, dep_ssn, (all12, months)))

    # Discover Part III rows and fill them
    p3_rows = _discover_part3_rows(reader)
    for idx, person in enumerate(covered_rows[: len(p3_rows)]):
        first_n, mi_n, last_n, ssn_n, (all12, mlist) = person
        rowref = p3_rows[idx]

        texts_sorted = rowref.text_fields
        assign = {}
        if len(texts_sorted) >= 5:
            # heuristic: smaller numeric ids align as L, F, MI, SSN, DOB
            assign[texts_sorted[0]] = last_n
            assign[texts_sorted[1]] = first_n
            assign[texts_sorted[2]] = mi_n
            assign[texts_sorted[3]] = ssn_n
            # texts_sorted[4] presumed DOB -> left blank
        _update_text(writer.pages[-1], assign)

        # Month boxes: [All12, Jan..Dec]
        if rowref.month_boxes:
            if all12:
                _set_checkbox_on(writer.pages[-1], rowref.month_boxes[0])
            else:
                for m_idx, on in enumerate(mlist):
                    if on and m_idx + 1 < len(rowref.month_boxes):
                        _set_checkbox_on(writer.pages[-1], rowref.month_boxes[m_idx + 1])

    _set_need_appearances(writer)

    editable = io.BytesIO()
    writer.write(editable); editable.seek(0)

    flattened_writer = _flatten(writer)
    flattened = io.BytesIO()
    flattened_writer.write(flattened); flattened.seek(0)

    first_last = f"{first}_{last}".strip().replace(" ", "_") or (_coerce_str(emp_row.get("employeeid")) or "employee")
    editable_name   = f"1095c_filled_fields_{first_last}_{year_used}.pdf"
    flattened_name  = f"1095c_filled_flattened_{first_last}_{year_used}.pdf"
    return editable_name, editable, flattened_name, flattened

# ----------------------------- Excel writer -----------------------------
def save_excel_outputs(
    interim: pd.DataFrame,
    final: pd.DataFrame,
    year: int,
    penalty_dashboard: Optional[pd.DataFrame] = None,
) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as xw:
        final.to_excel(xw, index=False, sheet_name=f"Final {year}")
        interim.to_excel(xw, index=False, sheet_name=f"Interim {year}")
        if penalty_dashboard is not None and not penalty_dashboard.empty:
            penalty_dashboard.to_excel(xw, index=False, sheet_name=f"Penalty Dashboard {year}")
    buf.seek(0)
    return buf.getvalue()
