# aca_pdf.py

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import (
    NameObject,
    BooleanObject,
    TextStringObject,
    DictionaryObject,
)

from aca_processing import MONTHS, _coerce_str, month_bounds


# =============== utilities ===============

def _to_int_safe(val, default=9999):
    """Extract first integer from strings like 'f1_10', 'Row[12]', 'f1_[10]'.
    Returns default if none found."""
    try:
        if isinstance(val, (int, float)):
            return int(val)
        m = re.search(r"\d+", str(val))
        return int(m.group(0)) if m else default
    except Exception:
        return default


def normalize_ssn_digits(s: str) -> str:
    """Keep digits only; if already masked like 'XXX-XX-1234' leave as-is."""
    s = (s or "").strip()
    if not s:
        return ""
    if "X" in s.upper():
        return s
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


# =============== low-level PDF writes (across *all* pages) ===============

def _find_on_value(annot_obj) -> NameObject:
    """The 'on' appearance name for a checkbox."""
    try:
        ap = annot_obj.get("/AP")
        if ap and "/N" in ap:
            n = ap["/N"]
            keys = list(n.keys()) if isinstance(n, dict) else []
            for k in keys:
                if str(k) not in ("/Off", "Off", "None"):
                    return NameObject(k)
    except Exception:
        pass
    return NameObject("/Yes")


def _update_text_any_page(writer: PdfWriter, name_to_value: Dict[str, str]):
    """
    Update text fields by name across *all* pages.
    Uses both update_page_form_field_values and direct widget /V set (TextStringObject).
    """
    for page in writer.pages:
        try:
            writer.update_page_form_field_values(page, name_to_value)
        except Exception:
            pass
        try:
            if "/Annots" in page:
                for annot in page["/Annots"]:
                    try:
                        obj = annot.get_object()
                    except Exception:
                        obj = annot
                    nm = obj.get("/T")
                    if nm in name_to_value:
                        val = TextStringObject(str(name_to_value[nm]))
                        obj.update({NameObject("/V"): val})
        except Exception:
            pass


def _set_checkbox_on_any(writer: PdfWriter, field_name: str):
    """Turn on a checkbox by name across *all* pages."""
    for page in writer.pages:
        try:
            if "/Annots" not in page:
                continue
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
    """Ensure NeedAppearances is a proper PDF boolean (not a raw string)."""
    try:
        root = writer._root_object  # type: ignore[attr-defined]
        if "/AcroForm" in root:
            acro = root["/AcroForm"]
            acro.update({NameObject("/NeedAppearances"): BooleanObject(True)})
        else:
            root.update({
                NameObject("/AcroForm"): DictionaryObject({
                    NameObject("/NeedAppearances"): BooleanObject(True)
                })
            })
    except Exception:
        pass


def _flatten(writer: PdfWriter) -> bytes:
    """
    Write a 'flattened' copy that still preserves field values.
    IMPORTANT: Do **not** delete /AcroForm or /Annots here, otherwise values vanish.
    Rely on NeedAppearances so viewers render the text.
    """
    _set_need_appearances(writer)
    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.getvalue()


# =============== Part III discovery ===============

@dataclass
class Part3RowRefs:
    text_fields: List[str]   # 5 fields: Last, First, MI, SSN, DOB
    month_boxes: List[str]   # 13 boxes: All12, Jan..Dec


def _discover_part3_rows(reader: PdfReader) -> List[Part3RowRefs]:
    """
    Group Part III fields by parents 'Row1[0]'..'RowN[0]'.
    Within each row:
      - text fields are 'f3_*' (5 per row)
      - month checkboxes are 'c3_*' (13 per row)
    """
    fields = _all_fields(reader)

    by_parent: Dict[str, List[str]] = {}
    for name, meta in fields.items():
        parent = meta.get("parent_T")
        if parent:
            by_parent.setdefault(parent, []).append(name)

    rows: List[Part3RowRefs] = []
    for parent in sorted([p for p in by_parent if p and p.startswith("Row")], key=_to_int_safe):
        names = by_parent[parent]
        texts = sorted([n for n in names if n.startswith("f3_")], key=_to_int_safe)
        boxes = sorted([n for n in names if n.startswith("c3_")], key=_to_int_safe)
        if texts or boxes:
            rows.append(Part3RowRefs(text_fields=texts, month_boxes=boxes))
    return rows


# =============== Public API ===============

def fill_pdf_for_employee(
    pdf_bytes: bytes,
    emp_row: pd.Series,
    final_df_emp: pd.DataFrame,
    year_used: int,
    emp_enroll_emp: Optional[pd.DataFrame] = None,
    dep_enroll_emp: Optional[pd.DataFrame] = None,
):
    """
    Fill 1095-C PDF (Parts I, II, and PART III).
    Returns: (editable_name, editable_bytes, flattened_name, flattened_bytes)
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    for p in reader.pages:
        writer.add_page(p)

    fields_meta = _all_fields(reader)

    # ---------- Part I: Employee info ----------
    first = _coerce_str(emp_row.get("firstname"))
    mi    = _coerce_str(emp_row.get("middleinitial"))
    last  = _coerce_str(emp_row.get("lastname"))
    full_name = " ".join([x for x in [first, mi, last] if x]).strip()

    ssn   = normalize_ssn_digits(_coerce_str(emp_row.get("ssn")))
    addr1 = _coerce_str(emp_row.get("addressline1"))
    addr2 = _coerce_str(emp_row.get("addressline2"))
    city  = _coerce_str(emp_row.get("city"))
    state = _coerce_str(emp_row.get("state"))
    zipcode = _coerce_str(emp_row.get("zipcode"))

    # EmployeeName group exposes 8 fields: f1_1..f1_8 (Name, SSN, Addr, City, State, ZIP, etc.)
    employee_fields = sorted(
        [k for k, v in fields_meta.items()
         if k.startswith("f1_") and v.get("parent_T") == "EmployeeName[0]"],
        key=_to_int_safe,
    )
    named: Dict[str, str] = {}
    if employee_fields:
        # 0: Name
        named[employee_fields[0]] = full_name
        # 1: SSN
        if len(employee_fields) > 1:
            named[employee_fields[1]] = ssn
        # 2: Address
        if len(employee_fields) > 2:
            addr = addr1 if not addr2 else f"{addr1} {addr2}"
            named[employee_fields[2]] = addr
        # 3..5: City / State / ZIP (best-effort)
        if len(employee_fields) > 3:
            named[employee_fields[3]] = city
        if len(employee_fields) > 4:
            named[employee_fields[4]] = state
        if len(employee_fields) > 5:
            named[employee_fields[5]] = zipcode

    _update_text_any_page(writer, named)

    # ---------- Part II: Line 14 & Line 16 ----------
    # Your current PDF revision doesn't expose explicit L14/L16 text fields in the dump
    # (the grid is printed). We skip writing them unless fields are present.

    month_to_code14 = {row["Month"]: _coerce_str(row.get("Line14_Final")) for _, row in final_df_emp.iterrows()}
    vals14 = [month_to_code14.get(m, "") for m in MONTHS]
    all14_same = len(set(vals14)) == 1 and (vals14[0] or "") != ""

    # ---------- Part III: Covered individuals ----------
    rows = _discover_part3_rows(reader)
    if rows:
        # Employee as first row (common)
        r0 = rows[0]
        if len(r0.text_fields) >= 5:
            last_first_mi_ssn_dob = {
                r0.text_fields[0]: last,        # Last
                r0.text_fields[1]: first,       # First
                r0.text_fields[2]: mi,          # MI
                r0.text_fields[3]: ssn,         # SSN
                r0.text_fields[4]: "",          # DOB if you have it
            }
            _update_text_any_page(writer, last_first_mi_ssn_dob)

        if r0.month_boxes:
            if all14_same and (vals14[0] or "") != "":
                _set_checkbox_on_any(writer, r0.month_boxes[0])  # All 12
            else:
                for i, m in enumerate(MONTHS, start=1):
                    if month_to_code14.get(m, ""):
                        _set_checkbox_on_any(writer, r0.month_boxes[i])

    # ---------- finalize ----------
    _set_need_appearances(writer)

    # Editable (form intact)
    editable = io.BytesIO()
    writer.write(editable)
    editable.seek(0)

    # “Flattened” (non-destructive so values remain visible)
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


# =============== Excel outputs (unchanged) ===============

def save_excel_outputs(
    interim: pd.DataFrame,
    final: pd.DataFrame,
    year: int,
    penalty_dashboard: Optional[pd.DataFrame] = None,
) -> bytes:
    buf = io.BytesIO()
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
