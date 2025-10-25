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

import re

def _to_int_safe(val, default=9999):
    """Extract first integer from any string like 'f1_10', 'Row[12]', 'f1_[10]'.
    Returns default if none found."""
    try:
        if isinstance(val, (int, float)):
            return int(val)
        m = re.search(r"\d+", str(val))
        return int(m.group(0)) if m else default
    except Exception:
        return default

def _safe_int(x, default=None):
    try:
        f = float(x)
        if np.isnan(f):  # type: ignore[arg-type]
            return default
        return int(f)
    except Exception:
        return default


def normalize_ssn_digits(s: str) -> str:
    """Keep digits only; if already masked like 'XXX-XX-1234' leave as is."""
    s = (s or "").strip()
    if not s:
        return ""
    if "X" in s.upper():
        return s  # already masked
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 9:
        return f"{digits[0:3]}-{digits[3:5]}-{digits[5:9]}"
    return s


def _all_fields(reader: PdfReader) -> Dict[str, dict]:
    """Return a mapping of field name -> field dict (PyPDF2 get_fields-compatible)."""
    fields: Dict[str, dict] = {}
    try:
        raw = reader.get_fields()
        if isinstance(raw, dict):
            for k, v in raw.items():
                # Normalize to dict (sometimes indirect objects live here)
                try:
                    fields[str(k)] = dict(v)
                except Exception:
                    try:
                        fields[str(k)] = v.get_object()  # type: ignore[attr-defined]
                    except Exception:
                        fields[str(k)] = {"name": str(k)}
    except Exception:
        pass
    return fields


def _find_on_value(annot_obj) -> NameObject:
    """
    Determine the 'on' appearance name for a checkbox.
    Typically '/Yes' or '/1'. If not found, fall back to '/Yes'.
    """
    try:
        ap = annot_obj.get("/AP")
        if ap and "/N" in ap:
            n = ap["/N"]
            # n could be a dict of appearances. Choose the first non-Off key.
            keys = list(n.keys()) if isinstance(n, dict) else []
            for k in keys:
                if str(k) not in ("/Off", "Off", "None"):
                    return NameObject(k)
    except Exception:
        pass
    return NameObject("/Yes")


def _update_text(writer: PdfWriter, page, name_to_value: Dict[str, str]):
    """
    Update text fields on a given page using PdfWriter.update_page_form_field_values.
    Also sets the /V directly on the widget if necessary.
    """
    try:
        writer.update_page_form_field_values(page, name_to_value)
    except Exception:
        pass

    # Hard-set /V in annotations when present
    try:
        if "/Annots" in page:
            for annot in page["/Annots"]:
                try:
                    obj = annot.get_object()
                except Exception:
                    obj = annot
                nm = obj.get("/T")
                if nm in name_to_value:
                    obj.update({NameObject("/V"): name_to_value[nm]})
    except Exception:
        pass


def _set_checkbox_on(writer: PdfWriter, page, field_name: str):
    """Turn on a single checkbox by updating both /V and /AS."""
    try:
        if "/Annots" not in page:
            return
        for annot in page["/Annots"]:
            try:
                obj = annot.get_object()
            except Exception:
                obj = annot
            nm = obj.get("/T")
            if str(nm) == field_name:
                on_name = _find_on_value(obj)
                obj.update({NameObject("/V"): on_name})
                obj.update({NameObject("/AS"): on_name})
    except Exception:
        pass


def _set_need_appearances(writer: PdfWriter):
    try:
        root = writer._root_object  # type: ignore[attr-defined]
        if "/AcroForm" in root:
            acro = root["/AcroForm"]
            acro.update({NameObject("/NeedAppearances"): NameObject("/true")})
        else:
            # Create minimal AcroForm if missing
            from PyPDF2.generic import DictionaryObject
            root.update({
                NameObject("/AcroForm"): DictionaryObject({
                    NameObject("/NeedAppearances"): NameObject("/true")
                })
            })
    except Exception:
        pass


def _flatten(writer: PdfWriter) -> bytes:
    """
    'Flatten' by writing out a copy without AcroForm/Annots so values render everywhere.
    """
    try:
        root = writer._root_object  # type: ignore[attr-defined]
        if "/AcroForm" in root:
            del root["/AcroForm"]
        # Also strip page-level annotations
        for i in range(len(writer.pages)):
            page = writer.pages[i]
            if "/Annots" in page:
                del page["/Annots"]
    except Exception:
        pass

    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.getvalue()


# ----------------------------- Part III discovery -----------------------------
@dataclass
class Part3RowRefs:
    # text fields (5): Last, First, MI, SSN, DOB
    text_fields: List[str]
    # checkboxes (13): [all12, Jan..Dec]
    month_boxes: List[str]


def _discover_part3_rows(reader: PdfReader) -> List[Part3RowRefs]:
    """
    Inspect the PDF's AcroForm and return row references for Part III (Covered Individuals).
    Relies on /Parent '/T' like 'Row1', 'Row2', ... (present in the IRS form).
    """
    fields = _all_fields(reader)
    text_by_row: Dict[str, List[str]] = {}
    box_by_row: Dict[str, List[str]] = {}
    for name, rec in fields.items():
        if name.startswith("f3_"):  # text inputs
            parent = rec.get("/Parent", {})
            try:
                row = parent.get("/T") if isinstance(parent, dict) else str(parent.get("/T"))
            except Exception:
                row = None
            row = str(row) if row else "Row?"
            text_by_row.setdefault(row, []).append(name)
        elif name.startswith("c3_"):  # checkboxes
            parent = rec.get("/Parent", {})
            try:
                row = parent.get("/T") if isinstance(parent, dict) else str(parent.get("/T"))
            except Exception:
                row = None
            row = str(row) if row else "Row?"
            box_by_row.setdefault(row, []).append(name)

    def _sort_key(n: str) -> int:
        try:
            return _to_int_safe(n)
        except Exception:
            return 9999

    rows: List[Part3RowRefs] = []
    for row_name in sorted(text_by_row.keys(), key=lambda r: _to_int_safe(r)):
        texts = sorted(text_by_row.get(row_name, []), key=_sort_key)
        boxes = sorted(box_by_row.get(row_name, []), key=_sort_key)
        rows.append(Part3RowRefs(text_fields=texts, month_boxes=boxes))
    return rows


# ----------------------------- Month coverage math -----------------------------
def _months_from_periods(periods: List[Tuple[date, date]], year: int) -> Tuple[bool, List[bool]]:
    """
    Given coverage periods [(start,end), ...] in calendar dates, return:
      all12: bool  — covered in all 12 months
      covered: [Jan..Dec] booleans
    """
    covered = [False] * 12
    for s, e in periods:
        # clip to year
        ys, ye = s.year, e.year
        for m in range(1, 13):
            ms, me = month_bounds(year, m)
            # overlap if max(start) <= min(end)
            ss = max(s, ms)
            ee = min(e, me)
            if ss <= ee:
                covered[m - 1] = True
    all12 = all(covered)
    return all12, covered


# ----------------------------- Part I & II mapping helpers -----------------------------
def _f1_field_names(reader: PdfReader, start_num: int, count: int) -> List[str]:
    """Utility: build f1_<n>[0] names like f1_18.. given a starting numeric id and count."""
    return [f"f1_{i}[0]" for i in range(start_num, start_num + count)]


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
    Fill 1095-C PDF (Parts I, II, and PART III)

    Arguments:
      pdf_bytes    : the blank 1095-C PDF (bytes)
      emp_row      : Series for the target employee (demographic columns)
      final_df_emp : 12-row DataFrame for that employee (Month, Line14_Final, Line16_Final)
      year_used    : filing year (int)

    Returns: (editable_name, editable_bytes, flattened_name, flattened_bytes)
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()

    # Copy pages
    for p in reader.pages:
        writer.add_page(p)

    # ---------------- Part I (employee info) ----------------
    first = _coerce_str(emp_row.get("firstname"))
    mi = _coerce_str(emp_row.get("middleinitial"))
    last = _coerce_str(emp_row.get("lastname"))
    ssn = normalize_ssn_digits(_coerce_str(emp_row.get("ssn")))
    addr1 = _coerce_str(emp_row.get("addressline1"))
    addr2 = _coerce_str(emp_row.get("addressline2"))
    city = _coerce_str(emp_row.get("city"))
    state = _coerce_str(emp_row.get("state"))
    zipcode = _coerce_str(emp_row.get("zipcode"))

    page0 = writer.pages[0]
    # typical IRS 1095-C has Part I/II as f1_* text fields
    # employee name block (f1_18..f1_25) and monthly codes lines (14 & 16)
    # Adjust these starts only if your template is different.
    EMPLOYEE_FIELDS = _f1_field_names(reader, 18, 8)  # First, MI, Last, SSN, Addr1, Addr2, City, State
    ZIP_FIELD = "f1_26[0]"

    tx = {}
    if len(EMPLOYEE_FIELDS) >= 8:
        tx[EMPLOYEE_FIELDS[0]] = first
        tx[EMPLOYEE_FIELDS[1]] = mi
        tx[EMPLOYEE_FIELDS[2]] = last
        tx[EMPLOYEE_FIELDS[3]] = ssn
        tx[EMPLOYEE_FIELDS[4]] = addr1
        tx[EMPLOYEE_FIELDS[5]] = addr2
        tx[EMPLOYEE_FIELDS[6]] = city
        tx[EMPLOYEE_FIELDS[7]] = state
    tx[ZIP_FIELD] = zipcode

    _update_text(writer, page0, tx)

    # ---------------- Part II (Line 14, Line 16) ----------------
    # If all months share the same code, set All-12 field; else set individual months
    # These ranges match the IRS 1095-C form revision your project targets.
    LINE14_FIELDS = _f1_field_names(reader, 44, 13)  # [All12, Jan..Dec]
    LINE16_FIELDS = _f1_field_names(reader, 57, 13)  # [All12, Jan..Dec]

    # Prepare monthly values
    m2 = final_df_emp.copy()
    m2["Month"] = m2["Month"].astype(str).str.strip()
    # Normalize month order to MONTHS
    month_to_code14: Dict[str, str] = {row["Month"]: _coerce_str(row.get("Line14_Final")) for _, row in m2.iterrows()}
    month_to_code16: Dict[str, str] = {row["Month"]: _coerce_str(row.get("Line16_Final")) for _, row in m2.iterrows()}

    # Determine if all 12 months share the same value
    vals14 = [month_to_code14.get(m, "") for m in MONTHS]
    vals16 = [month_to_code16.get(m, "") for m in MONTHS]
    all14_same = len(set(vals14)) == 1 and (vals14[0] or "") != ""
    all16_same = len(set(vals16)) == 1 and (vals16[0] or "") != ""

    # write Line 14
    if LINE14_FIELDS:
        if all14_same:
            _update_text(writer, page0, {LINE14_FIELDS[0]: vals14[0]})
        else:
            for i, m in enumerate(MONTHS, start=1):
                _update_text(writer, page0, {LINE14_FIELDS[i]: month_to_code14.get(m, "")})

    # write Line 16
    if LINE16_FIELDS:
        if all16_same:
            _update_text(writer, page0, {LINE16_FIELDS[0]: vals16[0]})
        else:
            for i, m in enumerate(MONTHS, start=1):
                _update_text(writer, page0, {LINE16_FIELDS[i]: month_to_code16.get(m, "")})

    # ---------------- Part III (Covered Individuals) ----------------
    # If you have enrollment periods, you can drive the month checkboxes per person.
    rows = _discover_part3_rows(reader)
    # A simple single-person fill (employee only) using 'all 12' if Line14 implies MEC for all months.
    # Extend as needed to add dependents (rows[1:], etc.).
    if rows:
        # Expect: 5 text fields (Last, First, MI, SSN, DOB) then 13 month boxes
        r0 = rows[0]
        if len(r0.text_fields) >= 5:
            last_first_mi_ssn_dob = {
                r0.text_fields[0]: last,
                r0.text_fields[1]: first,
                r0.text_fields[2]: mi,
                r0.text_fields[3]: ssn,
                r0.text_fields[4]: "",  # DOB left blank unless provided in sheet
            }
            _update_text(writer, page0, last_first_mi_ssn_dob)

        # MEC months (checkboxes). Here we set All-12 if Line14 is uniform & not blank.
        if r0.month_boxes:
            mec_all12 = all14_same and (vals14[0] or "") != ""
            if mec_all12:
                _set_checkbox_on(writer, page0, r0.month_boxes[0])  # 'All 12'
            else:
                # If you have actual coverage periods, replace with real logic from enrollments
                # Example placeholder: mark months that have a non-blank Line14 code
                for i, m in enumerate(MONTHS, start=1):
                    if month_to_code14.get(m, ""):
                        _set_checkbox_on(writer, page0, r0.month_boxes[i])

    # finalize
    _set_need_appearances(writer)

    # Editable copy (keeps form)
    editable = io.BytesIO()
    writer.write(editable)
    editable.seek(0)

    # Flattened copy (no AcroForm/Annots)
    flat_bytes = _flatten(writer)
    flattened = io.BytesIO(flat_bytes)
    flattened.seek(0)

    empid = _coerce_str(emp_row.get("employeeid"))
    return (
        f"1095c_editable_{empid}.pdf",
        editable,
        f"1095c_{empid}.pdf",
        flattened,
    )


# ----------------------------- Excel writer -----------------------------
def save_excel_outputs(
    interim: pd.DataFrame,
    final: pd.DataFrame,
    year: int,
    penalty_dashboard: Optional[pd.DataFrame] = None,
) -> bytes:
    buf = io.BytesIO()
    # Use openpyxl to avoid "no visible sheet" edge case; ensure at least 1 visible sheet.
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        wrote_any = False
        if final is not None and not final.empty:
            final.to_excel(xw, index=False, sheet_name=f"Final {year}")
            wrote_any = True
        if interim is not None and not interim.empty:
            interim.to_excel(xw, index=False, sheet_name=f"Interim {year}")
            wrote_any = True
        if penalty_dashboard is not None and not penalty_dashboard.empty:
            penalty_dashboard.to_excel(xw, index=False, sheet_name=f"Penalty Dashboard {year}")
            wrote_any = True
        if not wrote_any:
            pd.DataFrame({"Info": [f"No output for year {year}"]}).to_excel(
                xw, index=False, sheet_name="Info"
            )
    buf.seek(0)
    return buf.getvalue()
