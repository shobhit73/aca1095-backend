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

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from aca_processing import MONTHS, _coerce_str


# =========================
#  CONFIG: Part II overlay
# =========================
# Coordinates are PDF points (1/72"). Origin is bottom-left of the page.
# These defaults target the 2024 IRS 1095-C (Cat. No. 60705M).
# If codes land slightly off, tweak only these values.

# Table origin at the *baseline* of the “All 12 Months” column in Line 14 row.
TABLE_X0 = 110.0        # left edge of the All-12 column block (approx)
TABLE_Y_L14 = 405.0     # baseline for Line 14 text
TABLE_Y_L16 = 353.0     # baseline for Line 16 text

DX_ALL12_TO_JAN = 42.0  # distance from All-12 to Jan column center
DX_MONTH = 40.0         # horizontal step between month columns

# Font for overlay
FONT_NAME = "Helvetica"
FONT_SIZE = 9.5

# Fine-tune nudges (quick calibration without moving the base coords)
X_NUDGE = 1.5         # +right / -left
Y_NUDGE_L14 = -7.0    # +up / -down
Y_NUDGE_L16 = -7.0

# Debug crosshairs at each draw point (turn on only while calibrating)
DEBUG_OVERLAY = False


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
    """Return underlying object for IndirectObject, else obj itself."""
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


# ---------- Part II overlay drawing ----------

def _make_partii_overlay_page(width: float, height: float,
                              codes14: Dict[str, str],
                              codes16: Dict[str, str]) -> bytes:
    """
    Create a transparent PDF page that prints Line 14 / Line 16 codes
    at the configured coordinates (with nudges).
    """
    buf = io.BytesIO()
    can = canvas.Canvas(buf, pagesize=(width, height))
    can.setFont(FONT_NAME, FONT_SIZE)

    def draw_centered(x, y, txt):
        if not (txt or "").strip():
            return
        can.drawCentredString(x, y, str(txt).strip())
        if DEBUG_OVERLAY:
            can.setLineWidth(0.25)
            can.line(x - 2, y, x + 2, y)
            can.line(x, y - 2, x, y + 2)

    # ---- Line 14 ----
    all14 = ""
    if len(set(v for v in codes14.values() if v)) == 1:
        v = next((vv for vv in codes14.values() if vv), "")
        if v:
            all14 = v

    x_all = TABLE_X0 + X_NUDGE
    y14 = TABLE_Y_L14 + Y_NUDGE_L14
    if all14:
        draw_centered(x_all, y14, all14)
    else:
        for i, m in enumerate(MONTHS, start=1):
            v = codes14.get(m, "")
            if not v:
                continue
            x = TABLE_X0 + X_NUDGE + DX_ALL12_TO_JAN + (i - 1) * DX_MONTH
            draw_centered(x, y14, v)

    # ---- Line 16 ----
    all16 = ""
    if len(set(v for v in codes16.values() if v)) == 1:
        vv = next((vv for vv in codes16.values() if vv), "")
        if vv:
            all16 = vv

    y16 = TABLE_Y_L16 + Y_NUDGE_L16
    if all16:
        draw_centered(x_all, y16, all16)
    else:
        for i, m in enumerate(MONTHS, start=1):
            v = codes16.get(m, "")
            if not v:
                continue
            x = TABLE_X0 + X_NUDGE + DX_ALL12_TO_JAN + (i - 1) * DX_MONTH
            draw_centered(x, y16, v)

    can.save()
    buf.seek(0)
    return buf.getvalue()


def overlay_line14_line16(reader: PdfReader,
                          codes14: Dict[str, str],
                          codes16: Dict[str, str]):
    """
    Merge an overlay with Line 14/16 codes onto page 1 (index 0).
    """
    p0 = _resolve(reader.pages[0])
    media = _resolve(p0.get("/MediaBox"))
    if media and len(media) == 4:
        width = float(media[2])
        height = float(media[3])
    else:
        width, height = letter  # fallback

    overlay_bytes = _make_partii_overlay_page(width, height, codes14, codes16)
    overlay_reader = PdfReader(io.BytesIO(overlay_bytes))
    overlay_page = overlay_reader.pages[0]

    # Merge overlay onto page 1
    p0.merge_page(overlay_page)


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
    Fill your 1095-C form (Parts I, II overlay for 14/16, PART III).
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

    # -------- Part II (Line 14 & 16) overlay from final_df_emp --------
    # Expect columns like 'Month', 'Line14_Final', 'Line16_Final'
    codes14 = {row["Month"]: _coerce_str(row.get("Line14_Final")) for _, row in final_df_emp.iterrows()}
    codes16 = {row["Month"]: _coerce_str(row.get("Line16_Final")) for _, row in final_df_emp.iterrows()}
    for m in MONTHS:  # normalize missing months
        codes14.setdefault(m, "")
        codes16.setdefault(m, "")
    overlay_line14_line16(reader, codes14, codes16)

    # -------- Part III --------
    rows = _discover_part3(reader)
    if rows:
        r0 = rows[0]  # first row = employee
        if len(r0.text_fields) >= 4:
            _set_text_by_name(reader, r0.text_fields[0], last)
            _set_text_by_name(reader, r0.text_fields[1], first)
            _set_text_by_name(reader, r0.text_fields[2], mi)
            _set_text_by_name(reader, r0.text_fields[3], ssn)

        # Check All-12 or monthly boxes using presence of Line14 code as proxy for MEC
        same14 = len(set(v for v in codes14.values() if v)) == 1 and next((v for v in codes14.values() if v), "") != ""
        if r0.month_boxes:
            if same14:
                _set_checkbox_on(reader, r0.month_boxes[0])  # All 12 months
            else:
                for i, m in enumerate(MONTHS, start=1):
                    if codes14.get(m, ""):
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
