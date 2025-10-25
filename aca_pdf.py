# aca_pdf.py  (robust field matching, keeps your original behavior)

from __future__ import annotations

import io, re
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject

from aca_processing import MONTHS, _coerce_str, month_bounds


# ---------------- small helpers ----------------
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
        return s
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 9:
        return f"{digits[0:3]}-{digits[3:5]}-{digits[5:9]}"
    return s


# ---------------- PDF field helpers ----------------
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


# ---------- robust f1_xx discovery ----------
_F1_RE = re.compile(r"f1[_\.]?(0*)(\d+)\[0\]$")  # matches ...f1_1[0], ...f1_017[0], ...Page1.f1_17[0]

def _index_f1_fields(reader: PdfReader) -> Dict[int, str]:
    """
    Build a map {number: full_field_name} for any field whose name ends with f1_<num>[0],
    regardless of prefixes/zero-padding.
    """
    out: Dict[int, str] = {}
    fields = reader.get_fields() or {}
    for full_name in fields.keys():
        m = _F1_RE.search(full_name)
        if m:
            num = int(m.group(2))
            out[num] = full_name
    return out


# ---------------- Part III discovery ----------------
@dataclass
class Part3RowRefs:
    text_fields: List[str]   # (at least 4–5)
    month_boxes: List[str]   # [All12, Jan..Dec]


def _discover_part3_rows(reader: PdfReader) -> List[Part3RowRefs]:
    fields = _all_fields(reader)
    text_by_row: Dict[str, List[str]] = {}
    box_by_row: Dict[str, List[str]] = {}

    for name, rec in fields.items():
        # accept f3_/c3_ anywhere in the name (not only startswith)
        if "f3_" in name:
            parent = rec.get("/Parent", {})
            row = parent.get("/T") if isinstance(parent, dict) else str(parent.get("/T"))
            row = str(row) if row else "Row?"
            text_by_row.setdefault(row, []).append(name)
        elif "c3_" in name:
            parent = rec.get("/Parent", {})
            row = parent.get("/T") if isinstance(parent, dict) else str(parent.get("/T"))
            row = str(row) if row else "Row?"
            box_by_row.setdefault(row, []).append(name)

    def _field_sort_key(n: str) -> int:
        try:
            return int(re.search(r"_(\d+)\[", n).group(1))  # type: ignore[union-attr]
        except Exception:
            return 9999

    def _row_sort_key(row_label: str) -> int:
        if not row_label:
            return 999
        if row_label.startswith("Row"):
            s = row_label.replace("Row", "")
            try:
                return int(s.strip("[]") or "999")
            except Exception:
                pass
        m = re.search(r"(\d+)", str(row_label))
        return int(m.group(1)) if m else 999

    rows: List[Part3RowRefs] = []
    for row_name in sorted(text_by_row.keys(), key=_row_sort_key):
        texts = sorted(text_by_row.get(row_name, []), key=_field_sort_key)
        boxes = sorted(box_by_row.get(row_name, []), key=_field_sort_key)
        rows.append(Part3RowRefs(text_fields=texts, month_boxes=boxes))
    return rows


# ---------------- Month coverage math ----------------
def _months_from_periods(periods: List[Tuple[date, date]], year: int) -> Tuple[bool, List[bool]]:
    jan1, dec31 = date(year, 1, 1), date(year, 12, 31)
    covered = [False] * 12
    for (s, e) in periods:
        s = max(s, jan1); e = min(e, dec31)
        if e < jan1 or s > dec31:
            continue
        for m in range(1, 13):
            ms, me = month_bounds(year, m)
            if not (e < ms or s > me):
                covered[m - 1] = True
    return all(covered), covered


# ---------------- Public API ----------------
def fill_pdf_for_employee(
    pdf_bytes: bytes,
    emp_row: pd.Series,
    final_df_emp: pd.DataFrame,
    year_used: int,
    emp_enroll_emp: Optional[pd.DataFrame] = None,
    dep_enroll_emp: Optional[pd.DataFrame] = None,
):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    for p in reader.pages:
        writer.add_page(p)

    # Part I — robust field name lookup
    first = _coerce_str(emp_row.get("firstname"))
    mi    = _coerce_str(emp_row.get("middleinitial"))
    last  = _coerce_str(emp_row.get("lastname"))
    ssn   = normalize_ssn_digits(_coerce_str(emp_row.get("ssn")))
    addr1 = _coerce_str(emp_row.get("addressline1"))
    addr2 = _coerce_str(emp_row.get("addressline2"))
    city  = _coerce_str(emp_row.get("city"))
    state = _coerce_str(emp_row.get("state"))
    zipcode = _coerce_str(emp_row.get("zipcode"))

    f1_index = _index_f1_fields(reader)

    def _name_for(num: int) -> Optional[str]:
        # Exact number (handles zero-padding/prefixing)
        return f1_index.get(num)

    part1_map = {}
    if _name_for(1):  part1_map[_name_for(1)]  = last
    if _name_for(2):  part1_map[_name_for(2)]  = first
    if _name_for(3):  part1_map[_name_for(3)]  = mi
    if _name_for(4):  part1_map[_name_for(4)]  = ssn
    if _name_for(5):  part1_map[_name_for(5)]  = addr1
    if _name_for(6):  part1_map[_name_for(6)]  = addr2
    if _name_for(7):  part1_map[_name_for(7)]  = city
    if _name_for(8):  part1_map[_name_for(8)]  = f"{state} {zipcode}".strip()
    if part1_map:
        _update_text(writer.pages[0], part1_map)

    # Part II — Line 14 and Line 16 (robust)
    l14_all12_name = _name_for(17)  # All 12
    l14_month_fields = [ _name_for(n) for n in range(18, 30) ]  # 18..29 (12)
    l16_month_fields = [ _name_for(n) for n in range(44, 56) ]  # 44..55 (12)

    m_to_l14: Dict[str, str] = {}
    m_to_l16: Dict[str, str] = {}
    for _, r in final_df_emp.iterrows():
        m = _coerce_str(r.get("Month"))[:3]
        if not m: continue
        m_to_l14[m] = _coerce_str(r.get("Line14_Final"))
        m_to_l16[m] = _coerce_str(r.get("Line16_Final"))

    # Line 14
    l14_vals = [m_to_l14.get(m, "") for m in MONTHS]
    if l14_all12_name and l14_vals and all(v == l14_vals[0] and v for v in l14_vals):
        _update_text(writer.pages[0], {l14_all12_name: l14_vals[0]})
    else:
        updates = {fld: val for fld, val in zip(l14_month_fields, l14_vals) if fld}
        if updates:
            _update_text(writer.pages[0], updates)

    # Line 16
    l16_vals = [m_to_l16.get(m, "") for m in MONTHS]
    updates = {fld: val for fld, val in zip(l16_month_fields, l16_vals) if fld}
    if updates:
        _update_text(writer.pages[0], updates)

    # Part III — Covered Individuals (same logic; robust row discovery)
    covered_rows: List[Tuple[str, str, str, str, Tuple[bool, List[bool]]]] = []

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

    # Dependents
    if dep_enroll_emp is not None and not dep_enroll_emp.empty:
        for _, rr in dep_enroll_emp.iterrows():
            plan_code = _coerce_str(rr.get("plancode") or rr.get("PlanCode"))
            if plan_code and plan_code.strip().lower() == "waive":
                continue
            dep_first = _coerce_str(rr.get("depfirstname") or rr.get("DepFirstName"))
            dep_mi = _coerce_str(rr.get("depmidname") or rr.get("DepMidName"))
            dep_last = _coerce_str(rr.get("deplastname") or rr.get("DepLastName"))
            dep_ssn = ""
            s = rr.get("enrollmentstartdate") or rr.get("EnrollmentStartDate")
            e = rr.get("enrollmentenddate") or rr.get("EnrollmentEndDate")
            if pd.isna(s) or pd.isna(e): continue
            all12, months = _months_from_periods([(pd.to_datetime(s).date(), pd.to_datetime(e).date())], year_used)
            if any(months):
                covered_rows.append((dep_first, dep_mi, dep_last, dep_ssn, (all12, months)))

    p3_rows = _discover_part3_rows(reader)

    for idx, person in enumerate(covered_rows[: len(p3_rows)]):
        first_n, mi_n, last_n, ssn_n, (all12, mlist) = person
        rowref = p3_rows[idx]
        texts_sorted = rowref.text_fields
        assign = {}
        if len(texts_sorted) >= 4:
            assign[texts_sorted[0]] = last_n
            assign[texts_sorted[1]] = first_n
            assign[texts_sorted[2]] = mi_n
            assign[texts_sorted[3]] = ssn_n
        _update_text(writer.pages[-1], assign)

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
    editable_name = f"1095c_filled_fields_{first_last}_{year_used}.pdf"
    flattened_name = f"1095c_filled_flattened_{first_last}_{year_used}.pdf"
    return editable_name, editable, flattened_name, flattened
