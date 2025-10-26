# aca_pdf.py

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import (
    NameObject,
    BooleanObject,
    TextStringObject,
    DictionaryObject,
    IndirectObject,
)

from aca_processing import MONTHS, _coerce_str


# ----------------- small helpers -----------------

def _to_int(s: str, default=10_000):
    m = re.search(r"\d+", str(s or ""))
    return int(m.group(0)) if m else default


def _normalize_ssn(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if "X" in s.upper():
        return s
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 9:
        return f"{digits[0:3]}-{digits[3:5]}-{digits[5:9]}"
    return s


def _resolve(obj):
    """Return the underlying object for IndirectObject, else obj itself."""
    try:
        if isinstance(obj, IndirectObject):
            return obj.get_object()
    except Exception:
        pass
    return obj


# ---------- low-level PDF ops (works for XFA/AcroForm) ----------

def _iter_annots(reader: PdfReader):
    """Yield (page, annot_obj_resolved) pairs for all widget annotations."""
    for page in reader.pages:
        page = _resolve(page)
        if "/Annots" not in page:
            continue
        for a in page["/Annots"]:
            yield page, _resolve(a)


def _set_text_by_name(reader: PdfReader, field_name: str, value: str):
    """Set /V for a text field by its /T across the whole doc."""
    val = TextStringObject(value if value is not None else "")
    for _, annot in _iter_annots(reader):
        if _resolve(annot.get("/T")) == field_name:
            annot.update({NameObject("/V"): val})


def _set_checkbox_on(reader: PdfReader, field_name: str):
    """Turn a checkbox 'on' (set /V and /AS to its 'On' appearance)."""
    for _, annot in _iter_annots(reader):
        if _resolve(annot.get("/T")) != field_name:
            continue
        ap = _resolve(annot.get("/AP"))
        on_name = NameObject("/Yes")
        if ap and "/N" in ap:
            n = _resolve(ap["/N"])
            if isinstance(n, dict):
                for k in n.keys():
                    if str(k) not in ("Off", "/Off"):
                        on_name = NameObject(k)
                        break
        annot.update({NameObject("/V"): on_name})
        annot.update({NameObject("/AS"): on_name})


def _build_writer_with_acroform(reader: PdfReader) -> PdfWriter:
    """Create a writer, copy pages, import the reader's AcroForm (resolved),
    remove XFA, set NeedAppearances=True."""
    writer = PdfWriter()
    for p in reader.pages:
        writer.add_page(p)

    root = _resolve(reader.trailer.get("/Root"))
    acro = _resolve(root.get("/AcroForm")) if root else None
    if not isinstance(acro, dict):
        acro = DictionaryObject()

    # remove XFA if present and set NeedAppearances
    if "/XFA" in acro:
        try:
            del acro["/XFA"]
        except Exception:
            pass
    acro.update({NameObject("/NeedAppearances"): BooleanObject(True)})

    # import AcroForm dict into writer
    acro_ref = writer._add_object(acro)  # type: ignore[attr-defined]
    writer._root_object.update({NameObject("/AcroForm"): acro_ref})
    return writer


def _write_reader(reader: PdfReader) -> bytes:
    """Write the modified reader into a new PDF, keeping AcroForm."""
    writer = _build_writer_with_acroform(reader)
    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.getvalue()


# ---------- discovery for your template ----------

@dataclass
class Part3Row:
    text_fields: List[str]   # Last, First, MI, SSN, DOB (5 items)
    month_boxes: List[str]   # 13 checkboxes: All12, Jan..Dec


def _discover_part3(reader: PdfReader) -> List[Part3Row]:
    """Group Part III by parent Row1[0]..RowN[0], using f3_* (text) and c3_* (checkbox)."""
    fields = reader.get_fields() or {}
    by_parent: Dict[str, List[str]] = {}
    for name, meta in fields.items():
        parent = _resolve(meta.get("/Parent"))
        pT = _resolve(parent.get("/T")) if isinstance(parent, dict) else None
        if pT:
            by_parent.setdefault(pT, []).append(name)

    rows: List[Part3Row] = []
    for parent in sorted([p for p in by_parent if p.startswith("Row")], key=_to_int):
        names = by_parent[parent]
        texts = sorted([n for n in names if n.startswith("f3_")], key=_to_int)
        boxes = sorted([n for n in names if n.startswith("c3_")], key=_to_int)
        if texts or boxes:
            rows.append(Part3Row(text_fields=texts, month_boxes=boxes))
    return rows


# ---------- public API (unchanged signature) ----------

def fill_pdf_for_employee(
    pdf_bytes: bytes,
    emp_row: pd.Series,
    final_df_emp: pd.DataFrame,
    year_used: int,
    emp_enroll_emp: Optional[pd.DataFrame] = None,
    dep_enroll_emp: Optional[pd.DataFrame] = None,
):
    """
    Fill your 1095-C form (Parts I, III; Part II grid on this template is printed).
    Returns: (editable_name, editable_bytes, flattened_name, flattened_bytes)
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))

    # -------- Part I: employee info (EmployeeName[0] has f1_1..f1_8) --------
    first = _coerce_str(emp_row.get("firstname"))
    mi    = _coerce_str(emp_row.get("middleinitial"))
    last  = _coerce_str(emp_row.get("lastname"))
    full  = " ".join(x for x in [first, mi, last] if x).strip()

    ssn   = _normalize_ssn(_coerce_str(emp_row.get("ssn")))
    addr1 = _coerce_str(emp_row.get("addressline1"))
    addr2 = _coerce_str(emp_row.get("addressline2"))
    addr  = addr1 if not addr2 else f"{addr1} {addr2}"
    city  = _coerce_str(emp_row.get("city"))
    state = _coerce_str(emp_row.get("state"))
    zipc  = _coerce_str(emp_row.get("zipcode"))

    # your form uses these names under EmployeeName[0]
    _set_text_by_name(reader, "f1_1[0]", full)      # name
    _set_text_by_name(reader, "f1_2[0]", ssn)       # SSN
    _set_text_by_name(reader, "f1_3[0]", addr)      # address
    _set_text_by_name(reader, "f1_4[0]", city)      # city
    _set_text_by_name(reader, "f1_5[0]", state)     # state
    _set_text_by_name(reader, "f1_6[0]", zipc)      # ZIP

    # -------- Part II (grid is printed in this IRS template) --------
    month_to_l14 = {r["Month"]: _coerce_str(r.get("Line14_Final")) for _, r in final_df_emp.iterrows()}
    vals14 = [month_to_l14.get(m, "") for m in MONTHS]
    all12 = len(set(vals14)) == 1 and (vals14[0] or "") != ""

    # -------- Part III --------
    rows = _discover_part3(reader)
    if rows:
        r0 = rows[0]  # first row = employee
        if len(r0.text_fields) >= 4:
            _set_text_by_name(reader, r0.text_fields[0], last)
            _set_text_by_name(reader, r0.text_fields[1], first)
            _set_text_by_name(reader, r0.text_fields[2], mi)
            _set_text_by_name(reader, r0.text_fields[3], ssn)
        if r0.month_boxes:
            if all12:
                _set_checkbox_on(reader, r0.month_boxes[0])  # All 12
            else:
                for i, m in enumerate(MONTHS, start=1):
                    if month_to_l14.get(m, ""):
                        _set_checkbox_on(reader, r0.month_boxes[i])

    # ---------- outputs ----------
    editable_bytes = _write_reader(reader)
    flattened_bytes = editable_bytes  # non-destructive; values visible

    empid = _coerce_str(emp_row.get("employeeid"))
    return (
        f"1095c_editable_{empid}.pdf",
        io.BytesIO(editable_bytes),
        f"1095c_{empid}.pdf",
        io.BytesIO(flattened_bytes),
    )


# ---------------- Excel output (unchanged) ----------------

def save_excel_outputs(
    interim: pd.DataFrame,
    final: pd.DataFrame,
    year: int,
    penalty_dashboard: Optional[pd.DataFrame] = None,
) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        wrote = False
        if final is not None and not final.empty:
            final.to_excel(xw, index=False, sheet_name=f"Final {year}"); wrote = True
        if interim is not None and not interim.empty:
            interim.to_excel(xw, index=False, sheet_name=f"Interim {year}"); wrote = True
        if penalty_dashboard is not None and not penalty_dashboard.empty:
            penalty_dashboard.to_excel(xw, index=False, sheet_name=f"Penalty Dashboard {year}"); wrote = True
        if not wrote:
            pd.DataFrame({"Info": [f"No output for year {year}"]}).to_excel(
                xw, index=False, sheet_name="Info"
            )
    buf.seek(0)
    return buf.getvalue()
