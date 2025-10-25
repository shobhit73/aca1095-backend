# aca_pdf.py

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject, BooleanObject, TextStringObject, DictionaryObject

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


# ---------- low-level PDF ops (works for XFA forms) ----------

def _iter_annots(reader: PdfReader):
    """Yield (page, annot_obj) pairs for all widget annotations."""
    for page in reader.pages:
        if "/Annots" not in page:
            continue
        for a in page["/Annots"]:
            try:
                yield page, a.get_object()
            except Exception:
                yield page, a


def _set_text(reader: PdfReader, field_name: str, value: str):
    """Set /V for a text field by its /T (name) across the whole doc."""
    val = TextStringObject(value if value is not None else "")
    for _, annot in _iter_annots(reader):
        if annot.get("/T") == field_name:
            annot.update({NameObject("/V"): val})


def _set_checkbox(reader: PdfReader, field_name: str):
    """Turn a checkbox 'on' (set /V and /AS to its 'On' appearance)."""
    for _, annot in _iter_annots(reader):
        if annot.get("/T") != field_name:
            continue
        on = None
        ap = annot.get("/AP")
        if ap and "/N" in ap and isinstance(ap["/N"], dict):
            for k in ap["/N"].keys():
                if str(k) not in ("Off", "/Off"):
                    on = NameObject(k)
                    break
        if on is None:
            on = NameObject("/Yes")
        annot.update({NameObject("/V"): on})
        annot.update({NameObject("/AS"): on})


def _remove_xfa_and_set_need_appearances(reader: PdfReader):
    """Remove XFA so viewers render AcroForm, and set NeedAppearances=True."""
    root = reader.trailer["/Root"]
    acro = root.get("/AcroForm")
    if acro is None:
        acro = DictionaryObject()
        root.update({NameObject("/AcroForm"): acro})
    if "/XFA" in acro:
        del acro["/XFA"]
    acro.update({NameObject("/NeedAppearances"): BooleanObject(True)})


def _write_reader(reader: PdfReader) -> bytes:
    """Write the (possibly modified) reader to bytes, keeping the AcroForm."""
    _remove_xfa_and_set_need_appearances(reader)
    writer = PdfWriter()
    for p in reader.pages:
        writer.add_page(p)
    # carry over the (now XFA-free) AcroForm
    writer._root_object.update({NameObject("/AcroForm"): reader.trailer["/Root"]["/AcroForm"]})
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
    """Your form groups Part III by Row1[0]..RowN[0], with f3_* (text) and c3_* (checkboxes)."""
    fields = reader.get_fields() or {}
    by_parent: Dict[str, List[str]] = {}
    for name, meta in fields.items():
        parent = meta.get("/Parent")
        pT = parent.get("/T") if parent else None
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


# ---------- public API ----------

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

    # these field names exist in your file under EmployeeName[0]
    _set_text(reader, "f1_1[0]", full)      # name
    _set_text(reader, "f1_2[0]", ssn)       # SSN
    _set_text(reader, "f1_3[0]", addr)      # address
    _set_text(reader, "f1_4[0]", city)      # city
    _set_text(reader, "f1_5[0]", state)     # state
    _set_text(reader, "f1_6[0]", zipc)      # ZIP

    # -------- Part II (grid is printed lines in this IRS template) --------
    # We can't write the 14/16 codes unless the template exposes fields for them.
    # If you supply exact field names for L14/L16 cells, I'll wire them in.

    # For Part III coverage, we'll mark "All 12" if Line 14 is same across months and non-empty;
    # otherwise check months that have a Line14 code.
    month_to_l14 = {r["Month"]: _coerce_str(r.get("Line14_Final")) for _, r in final_df_emp.iterrows()}
    vals14 = [month_to_l14.get(m, "") for m in MONTHS]
    all12 = len(set(vals14)) == 1 and (vals14[0] or "") != ""

    rows = _discover_part3(reader)
    if rows:
        r0 = rows[0]  # first row = employee
        # your Row1 text fields include five f3_* (Last, First, MI, SSN, DOB). Fill what we have.
        if len(r0.text_fields) >= 4:
            # best-effort mapping by slot order
            _set_text(reader, r0.text_fields[0], last)
            _set_text(reader, r0.text_fields[1], first)
            _set_text(reader, r0.text_fields[2], mi)
            _set_text(reader, r0.text_fields[3], ssn)
            # DOB not available in your dataset now; leave blank if a 5th slot exists.

        if r0.month_boxes:
            if all12:
                # first checkbox in the row is the "All 12 months" box in this template
                _set_checkbox(reader, r0.month_boxes[0])
            else:
                # check Jan..Dec selectively (boxes 1..12 after "All 12")
                for i, m in enumerate(MONTHS, start=1):
                    if month_to_l14.get(m, ""):
                        _set_checkbox(reader, r0.month_boxes[i])

    # ---------- produce outputs ----------
    # "Editable": keep the form as-is (values already set on widgets)
    editable_bytes = _write_reader(reader)

    # "Flattened": for now, same as editable (non-destructive; values visible).
    # If you *must* burn in text (true raster flatten), we can add a renderer step later.
    flattened_bytes = editable_bytes

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
