# aca_pdf.py — simple AcroForm filling for 1095-C
from __future__ import annotations

import io
import logging
import re
from typing import Dict, Optional, List

import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import (
    NameObject,
    BooleanObject,
    TextStringObject,
    DictionaryObject,
    IndirectObject,
)

logger = logging.getLogger("pdf")

# Try to use shared helpers if available, else provide minimal fallbacks
try:
    from aca_processing import MONTHS as _MONTHS, _coerce_str as _COERCE
except Exception:  # fallback to safe defaults
    _MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sept","Oct","Nov","Dec"]
    def _COERCE(x): 
        return "" if (x is None) else str(x).strip()

MONTHS = _MONTHS

# ---- Explicit field maps per your spec ----
LINE14_FIELDS: Dict[str, str] = {
    "Jan": "f1_18[0]", "Feb": "f1_19[0]", "Mar": "f1_20[0]", "Apr": "f1_21[0]",
    "May": "f1_22[0]", "Jun": "f1_23[0]", "Jul": "f1_24[0]", "Aug": "f1_25[0]",
    "Sept": "f1_26[0]", "Oct": "f1_27[0]", "Nov": "f1_28[0]", "Dec": "f1_29[0]",
}

LINE16_FIELDS: Dict[str, Optional[str]] = {
    "Jan": "f1_44[0]", "Feb": "f1_45[0]", "Mar": "f1_46[0]", "Apr": "f1_47[0]",
    "May": "f1_48[0]", "Jun": "f1_49[0]", "Jul": "f1_50[0]", "Aug": "f1_51[0]",
    # Sep–Dec intentionally blank:
    "Sept": None, "Oct": None, "Nov": None, "Dec": None,
}

# Part I (per your latest mapping)
FIELD_FIRST = "f1_1[0]"
FIELD_MIDDLE = "f1_2[0]"
FIELD_LAST = "f1_3[0]"
FIELD_SSN_LAST4 = "f1_4[0]"  # your form expects last-4 here

# ---------------- utilities ----------------

def _resolve(obj):
    """Return underlying object if IndirectObject, else obj."""
    try:
        if isinstance(obj, IndirectObject):
            return obj.get_object()
    except Exception:
        pass
    return obj

def _set_text_by_name(reader: PdfReader, field_name: str, value: Optional[str]):
    """Set /V for a named text field across the whole document (no error if missing)."""
    if not field_name:
        return
    val = TextStringObject("" if value is None else str(value))
    for page in reader.pages:
        p = _resolve(page)
        anns = _resolve(p.get("/Annots"))
        if not anns:
            continue
        for a in anns:
            annot = _resolve(a)
            if _resolve(annot.get("/T")) == field_name:
                annot.update({NameObject("/V"): val})

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first existing column matching any candidate (case/space-insensitive)."""
    norm = {re.sub(r"\W+", "", c).lower(): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"\W+", "", cand).lower()
        if key in norm:
            return norm[key]
    return None

def _norm_month(m: str) -> str:
    s = (m or "").strip()
    if not s:
        return ""
    s = s.title().replace("Sept.", "Sept").replace("Sep", "Sept")
    if s in {"All 12 Months", "All", "All12"}:
        return "All"
    # normalize full month names to our 3/4-letter keys
    full2short = {
        "January":"Jan","February":"Feb","March":"Mar","April":"Apr","May":"May",
        "June":"Jun","July":"Jul","August":"Aug","September":"Sept",
        "October":"Oct","November":"Nov","December":"Dec",
    }
    return full2short.get(s, s)

def _extract_last4_ssn(raw: str) -> str:
    if not raw:
        return ""
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) >= 4:
        return digits[-4:]
    # also accept masked strings like XXX-XX-1234 -> keep last 4
    m = re.search(r"(\d{4})\s*$", raw)
    return m.group(1) if m else raw[-4:]

def _copy_acroform_and_pages(reader: PdfReader) -> PdfWriter:
    """
    Create a PdfWriter, copy pages, and clone AcroForm safely.
    """
    writer = PdfWriter()

    # 1) copy pages (use original PageObject; do not resolve to raw dict)
    for p in reader.pages:
        writer.add_page(p)

    # 2) copy AcroForm (resolved) + set NeedAppearances, remove XFA if present
    root = _resolve(reader.trailer.get("/Root"))
    acro = _resolve(root.get("/AcroForm")) if isinstance(root, dict) else None
    if isinstance(acro, dict):
        acro_copy = DictionaryObject()
        for k, v in acro.items():
            if str(k) == "/XFA":
                continue
            acro_copy[NameObject(k)] = v  # safe to reference, then add_object below
        acro_copy.update({NameObject("/NeedAppearances"): BooleanObject(True)})
        acro_ref = writer._add_object(acro_copy)
        writer._root_object.update({NameObject("/AcroForm"): acro_ref})
    return writer

# ---------------- core API ----------------

def fill_pdf_for_employee(
    pdf_bytes: bytes,
    emp_row: pd.Series,
    final_df_emp: pd.DataFrame,
    year_used: int,
    emp_enroll_emp: Optional[pd.DataFrame] = None,
    dep_enroll_emp: Optional[pd.DataFrame] = None,
):
    """
    Fill 1095-C using explicit AcroForm field names you provided.

    Returns:
      (editable_name, editable_io, flat_name, flat_io)
    """
    if not isinstance(pdf_bytes, (bytes, bytearray)):
        raise ValueError("pdf_bytes must be raw bytes")

    reader = PdfReader(io.BytesIO(pdf_bytes))

    # ---- Part I: First / Middle / Last / SSN last-4 ----
    first = _COERCE(emp_row.get("firstname"))
    middle = _COERCE(emp_row.get("middleinitial"))
    last = _COERCE(emp_row.get("lastname"))
    ssn_raw = _COERCE(emp_row.get("ssn"))
    ssn_last4 = _extract_last4_ssn(ssn_raw)

    _set_text_by_name(reader, FIELD_FIRST, first)
    _set_text_by_name(reader, FIELD_MIDDLE, middle)
    _set_text_by_name(reader, FIELD_LAST, last)
    _set_text_by_name(reader, FIELD_SSN_LAST4, ssn_last4)

    # ---- Part II: Line 14 & Line 16 from final_df_emp ----
    if final_df_emp is None or final_df_emp.empty:
        logger.warning("final_df_emp is empty — Line 14/16 will remain blank")
    else:
        col_month = _pick_col(final_df_emp, ["Month", "Months", "Coverage Month", "Period"])
        col_l14   = _pick_col(final_df_emp, ["Line14_Final","Line 14","Line14","L14","Line 14 Code"])
        col_l16   = _pick_col(final_df_emp, ["Line16_Final","Line 16","Line16","L16","Line 16 Code"])

        if not col_month:
            logger.warning("No Month/Months column found; skipping Line 14/16 population")
        else:
            # build month->code maps (with All 12 Months backfill)
            codes14: Dict[str, str] = {}
            codes16: Dict[str, str] = {}
            all14 = None
            all16 = None

            for _, r in final_df_emp.iterrows():
                m = _norm_month(_COERCE(r.get(col_month)))
                v14 = _COERCE(r.get(col_l14)) if col_l14 else ""
                v16 = _COERCE(r.get(col_l16)) if col_l16 else ""

                if m == "All":
                    if v14: all14 = v14
                    if v16: all16 = v16
                    continue
                if m in MONTHS:
                    if v14: codes14[m] = v14
                    if v16: codes16[m] = v16

            # backfill All-12 if specific month missing
            if all14:
                for m in MONTHS:
                    codes14.setdefault(m, all14)
            if all16:
                for m in MONTHS:
                    codes16.setdefault(m, all16)

            # write Line 14 into its explicit fields
            for m in MONTHS:
                target = LINE14_FIELDS.get(m)
                if not target:
                    continue
                val = codes14.get(m, "")
                _set_text_by_name(reader, target, val)

            # write Line 16 into Jan–Aug fields only (Sep–Dec intentionally blank)
            for m in MONTHS:
                target = LINE16_FIELDS.get(m)
                if not target:
                    continue  # None means leave blank
                val = codes16.get(m, "")
                _set_text_by_name(reader, target, val)

    # ---- write out with AcroForm preserved ----
    writer = _copy_acroform_and_pages(reader)
    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    pdf_bytes_out = out.getvalue()

    empid = _COERCE(emp_row.get("employeeid")) or "employee"
    editable_name = f"1095c_editable_{empid}.pdf"
    flat_name = f"1095c_{empid}.pdf"

    # We return both as BytesIO for compatibility with your existing FastAPI handler
    return (
        editable_name, io.BytesIO(pdf_bytes_out),
        flat_name, io.BytesIO(pdf_bytes_out),
    )

# ---------------- Excel bundling (unchanged API) ----------------

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
            pd.DataFrame({"Info":[f"No output for year {year}"]}).to_excel(
                xw, index=False, sheet_name="Info"
            )
    buf.seek(0)
    return buf.getvalue()
