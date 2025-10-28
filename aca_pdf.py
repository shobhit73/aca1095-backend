# aca_pdf.py
from __future__ import annotations

import io
import re
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject, BooleanObject

# ===========================
# Constants / field mappings
# ===========================

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# Line 14 text fields (per month)
LINE14_FIELDS = {
    "Jan": "f1_18[0]",
    "Feb": "f1_19[0]",
    "Mar": "f1_20[0]",
    "Apr": "f1_21[0]",
    "May": "f1_22[0]",
    "Jun": "f1_23[0]",
    "Jul": "f1_24[0]",
    "Aug": "f1_25[0]",
    "Sep": "f1_26[0]",
    "Oct": "f1_27[0]",
    "Nov": "f1_28[0]",
    "Dec": "f1_29[0]",
}

# Line 16 text fields (per month) — (Sep–Dec can stay blank if you wish)
LINE16_FIELDS = {
    "Jan": "f1_44[0]",
    "Feb": "f1_45[0]",
    "Mar": "f1_46[0]",
    "Apr": "f1_47[0]",
    "May": "f1_48[0]",
    "Jun": "f1_49[0]",
    "Jul": "f1_50[0]",
    "Aug": "f1_51[0]",
    # "Sep": ...,
    # "Oct": ...,
    # "Nov": ...,
    # "Dec": ...,
}

# Part I: name/SSN fields
FIELD_FIRST  = "f1_1[0]"
FIELD_MIDDLE = "f1_2[0]"
FIELD_LAST   = "f1_3[0]"
FIELD_SSN    = "f1_4[0]"

# Part I: address fields (Line 3–6)
FIELD_STREET      = "f1_5[0]"  # Street address (incl. apt)
FIELD_CITY        = "f1_6[0]"  # City or town
FIELD_STATE       = "f1_7[0]"  # State or province
FIELD_COUNTRY_ZIP = "f1_8[0]"  # Country and ZIP (or foreign postal)

# ------------------------------
# Part III (Covered Individuals)
# ------------------------------
# NOTE: These IDs are from your mapping dump. Adjust here if your template differs.
PART3_MAP: Dict[int, Dict[str, Any]] = {
    1: {'name': 'f3_76[0]',  'ssn': 'f3_77[0]', 'dob': 'f3_77[0]', 'all12': 'c3_55[0]',
        'months': {'Jan': 'c3_56[0]','Feb': 'c3_57[0]','Mar': 'c3_58[0]','Apr': 'c3_59[0]',
                   'May': 'c3_60[0]','Jun': 'c3_61[0]','Jul': 'c3_62[0]','Aug': 'c3_63[0]',
                   'Sep': 'c3_64[0]','Oct': 'c3_65[0]','Nov': 'c3_66[0]','Dec': 'c3_67[0]'}},
    2: {'name': 'f3_89[0]',  'ssn': 'f3_90[0]', 'dob': 'f3_90[0]', 'all12': 'c3_68[0]',
        'months': {'Jan': 'c3_69[0]','Feb': 'c3_70[0]','Mar': 'c3_71[0]','Apr': 'c3_72[0]',
                   'May': 'c3_73[0]','Jun': 'c3_74[0]','Jul': 'c3_75[0]','Aug': 'c3_86[0]',
                   'Sep': 'c3_87[0]','Oct': 'c3_88[0]','Nov': 'c3_101[0]','Dec': 'c3_102[0]'}},
    3: {'name': 'f3_92[0]',  'ssn': 'f3_93[0]', 'dob': 'f3_93[0]', 'all12': 'c3_81[0]',
        'months': {'Jan': 'c3_82[0]','Feb': 'c3_83[0]','Mar': 'c3_84[0]','Apr': 'c3_85[0]',
                   'May': 'c3_96[0]','Jun': 'c3_97[0]','Jul': 'c3_98[0]','Aug': 'c3_99[0]',
                   'Sep': 'c3_100[0]','Oct': 'c3_113[0]','Nov': 'c3_114[0]','Dec': 'c3_115[0]'}},
    4: {'name': 'f3_95[0]',  'ssn': 'f3_96[0]', 'dob': 'f3_96[0]', 'all12': 'c3_94[0]',
        'months': {'Jan': 'c3_95[0]','Feb': 'c3_108[0]','Mar': 'c3_109[0]','Apr': 'c3_110[0]',
                   'May': 'c3_111[0]','Jun': 'c3_112[0]','Jul': 'c3_125[0]','Aug': 'c3_126[0]',
                   'Sep': 'c3_127[0]','Oct': 'c3_128[0]','Nov': 'c3_129[0]','Dec': 'c3_130[0]'}},
    5: {'name': 'f3_98[0]',  'ssn': 'f3_99[0]', 'dob': 'f3_99[0]', 'all12': 'c3_107[0]',
        'months': {'Jan': 'c3_120[0]','Feb': 'c3_121[0]','Mar': 'c3_122[0]','Apr': 'c3_123[0]',
                   'May': 'c3_124[0]','Jun': 'c3_137[0]','Jul': 'c3_138[0]','Aug': 'c3_139[0]',
                   'Sep': 'c3_140[0]','Oct': 'c3_141[0]','Nov': 'c3_142[0]','Dec': 'c3_143[0]'}},
    6: {'name': 'f3_102[0]','ssn': 'f3_103[0]','dob': 'f3_104[0]','all12': 'c3_116[0]',
        'months': {'Jan': 'c3_117[0]','Feb': 'c3_118[0]','Mar': 'c3_119[0]','Apr': 'c3_131[0]',
                   'May': 'c3_132[0]','Jun': 'c3_133[0]','Jul': 'c3_134[0]','Aug': 'c3_135[0]',
                   'Sep': 'c3_136[0]','Oct': 'c3_147[0]','Nov': 'c3_148[0]','Dec': 'c3_149[0]'}},
    7: {'name': 'f3_105[0]','ssn': 'f3_106[0]','dob': 'f3_106[0]','all12': 'c3_150[0]',
        'months': {'Jan': 'c3_151[0]','Feb': 'c3_152[0]','Mar': 'c3_153[0]','Apr': 'c3_154[0]',
                   'May': 'c3_155[0]','Jun': 'c3_156[0]','Jul': 'c3_157[0]','Aug': 'c3_158[0]',
                   'Sep': 'c3_159[0]','Oct': 'c3_160[0]','Nov': 'c3_161[0]','Dec': 'c3_162[0]'}},
    8: {'name': 'f3_109[0]','ssn': 'f3_110[0]','dob': 'f3_110[0]','all12': 'c3_163[0]',
        'months': {'Jan': 'c3_164[0]','Feb': 'c3_165[0]','Mar': 'c3_166[0]','Apr': 'c3_167[0]',
                   'May': 'c3_168[0]','Jun': 'c3_169[0]','Jul': 'c3_170[0]','Aug': 'c3_171[0]',
                   'Sep': 'c3_172[0]','Oct': 'c3_173[0]','Nov': 'c3_174[0]','Dec': 'c3_175[0]'}},
    9: {'name': 'f3_118[0]','ssn': 'f3_119[0]','dob': 'f3_119[0]','all12': 'c3_138[0]',
        'months': {'Jan': 'c3_139[0]','Feb': 'c3_140[0]','Mar': 'c3_141[0]','Apr': 'c3_142[0]',
                   'May': 'c3_143[0]','Jun': 'c3_144[0]','Jul': 'c3_145[0]','Aug': 'c3_146[0]',
                   'Sep': 'c3_148[0]','Oct': 'c3_149[0]','Nov': 'c3_150[0]','Dec': 'c3_151[0]'}},
}

# ======================
# Utility helpers
# ======================

def _norm(s: str) -> str:
    return re.sub(r"\W+", "", (s or "")).lower()

def _pick(df_or_index, candidates: List[str]) -> Optional[str]:
    cols = list(df_or_index) if not hasattr(df_or_index, "columns") else df_or_index.columns
    norm = {_norm(c): c for c in cols}
    for c in candidates:
        k = _norm(c)
        if k in norm:
            return norm[k]
    return None

def _split_name(first: str, middle: str, last: str, full: str) -> Tuple[str,str,str]:
    first = (first or "").strip()
    middle = (middle or "").strip()
    last = (last or "").strip()
    full = (full or "").strip()
    if first or last:
        return first, middle, last
    if full:
        parts = [p for p in re.split(r"\s+", full) if p]
        if len(parts) == 1:
            return parts[0], "", ""
        if len(parts) == 2:
            return parts[0], "", parts[1]
        return parts[0], " ".join(parts[1:-1]), parts[-1]
    return "", "", ""

def _extract_name_from_row(row: pd.Series) -> Tuple[str,str,str]:
    fn = row.get(_pick(row.index, ["FirstName","First","Given"]))
    mi = row.get(_pick(row.index, ["Middle","MiddleInitial","MI"]))
    ln = row.get(_pick(row.index, ["LastName","Last","Surname"]))
    full = row.get(_pick(row.index, ["Name","FullName","Employee Name"]))
    return _split_name(str(fn or ""), str(mi or ""), str(ln or ""), str(full or ""))

def _extract_ssn(row: pd.Series) -> str:
    ssn_col = _pick(row.index, ["SSN","TIN","SSN/TIN"])
    if not ssn_col:
        return ""
    return str(row.get(ssn_col) or "").strip()

def _extract_address(row: pd.Series) -> Dict[str,str]:
    street = str(row.get(_pick(row.index, ["Address","Address1","Street","Street Address"])) or "").strip()
    apt    = str(row.get(_pick(row.index, ["Address2","Apt","Apartment"])) or "").strip()
    if apt and apt.lower() not in {"nan", "none"}:
        street_out = f"{street} {apt}".strip()
    else:
        street_out = street

    city   = str(row.get(_pick(row.index, ["City","City/Town"])) or "").strip()
    state  = str(row.get(_pick(row.index, ["State","Province","State/Province"])) or "").strip()
    zipc   = str(row.get(_pick(row.index, ["ZIP","Zip","Postal","PostalCode","ZIP Code"])) or "").strip()
    cntry  = str(row.get(_pick(row.index, ["Country","Nation"])) or "").strip()

    cz = " ".join([x for x in [cntry, zipc] if x]).strip()
    return {
        "street": street_out,
        "city": city,
        "state": state,
        "country_zip": cz,
    }

def _extract_line_codes(final_df_emp: Optional[pd.DataFrame]) -> Tuple[Dict[str,str], Dict[str,str]]:
    l14: Dict[str,str] = {m: "" for m in MONTHS}
    l16: Dict[str,str] = {m: "" for m in MONTHS}
    if final_df_emp is None or final_df_emp.empty:
        return l14, l16

    col_month = _pick(final_df_emp, ["Month","Months","Coverage Month","Period"])
    col_l14 = _pick(final_df_emp, ["Line14_Final","Line14","Line 14","L14","Line 14 Code"])
    col_l16 = _pick(final_df_emp, ["Line16_Final","Line16","Line 16","L16","Line 16 Code"])
    if not col_month:
        return l14, l16

    for _, r in final_df_emp.iterrows():
        month_val = str(r.get(col_month) or "").strip()
        v14 = str(r.get(col_l14) or "").strip() if col_l14 else ""
        v16 = str(r.get(col_l16) or "").strip() if col_l16 else ""

        if _norm(month_val) in {"all12months", "all12"}:
            for m in MONTHS:
                if v14: l14[m] = v14
                if v16: l16[m] = v16
            continue

        for m in MONTHS:
            if _norm(month_val) == _norm(m):
                if v14: l14[m] = v14
                if v16: l16[m] = v16
                break

    return l14, l16

def _extract_part3_rows_from_excel(
    sheets: Optional[Dict[str, pd.DataFrame]],
    employee_id: str,
    emp_fullname: str
) -> List[Dict[str, Any]]:
    """
    Row 1 = employee; rows 2..9 = dependents.
    """
    def _month_flags(row: pd.Series) -> Dict[str, bool]:
        out = {m: False for m in MONTHS}
        all12_col = _pick(row.index, ["All 12 Months","All12Months","All12"])
        all12 = False
        if all12_col:
            v = row.get(all12_col)
            all12 = str(v).strip().lower() in {"1","true","yes","y","x"}
        for m in MONTHS:
            col = _pick(row.index, [m, m.upper(), m.capitalize()])
            if col and pd.notna(row.get(col)):
                out[m] = str(row.get(col)).strip().lower() in {"1","true","yes","y","x"}
        if all12:
            out = {m: True for m in MONTHS}
        return out

    def _name_from_row(row: pd.Series) -> str:
        nm = str(row.get(_pick(row.index, ["Name","FullName"])) or "").strip()
        if nm:
            return nm
        fn, mi, ln = _extract_name_from_row(row)
        return " ".join([x for x in [fn, mi, ln] if x])

    def _ids_from_row(row: pd.Series) -> Dict[str,str]:
        ssn_col = _pick(row.index, ["SSN","TIN","SSN/TIN"])
        dob_col = _pick(row.index, ["DOB","Date of Birth","BirthDate","Birth"])
        return {
            "ssn": str(row.get(ssn_col) or "").strip() if ssn_col else "",
            "dob": str(row.get(dob_col) or "").strip() if dob_col else "",
        }

    rows: List[Dict[str, Any]] = []
    if not sheets:
        return rows

    # Employee (row 1)
    emp = None
    for sh in ["Emp Enrollment","Employee Enrollment","Employee Coverage","Enrollment"]:
        df = sheets.get(sh)
        if df is None or df.empty:
            continue
        col_empid = _pick(df, ["EmployeeID","EmpID","Employee Id"])
        if not col_empid:
            continue
        sel = df[df[col_empid].astype(str).str.strip() == str(employee_id).strip()]
        if not sel.empty:
            emp = sel.iloc[0]
            break

    if emp is not None:
        rows.append({
            "name": _name_from_row(emp) or emp_fullname,
            "ssn": _ids_from_row(emp)["ssn"],
            "dob": _ids_from_row(emp)["dob"],
            "months": _month_flags(emp),
        })
    else:
        rows.append({"name": emp_fullname, "ssn": "", "dob": "", "months": {m: False for m in MONTHS}})

    # Dependents (rows 2..9)
    dep_df = None
    for sh in ["Dep Enrollment","Dependents","Dependent Enrollment","Covered Individuals"]:
        df = sheets.get(sh)
        if df is None or df.empty:
            continue
        col_empid = _pick(df, ["EmployeeID","EmpID","Employee Id"])
        if not col_empid:
            continue
        view = df[df[col_empid].astype(str).str.strip() == str(employee_id).strip()]
        if not view.empty:
            dep_df = view
            break

    if dep_df is not None and not dep_df.empty:
        for _, drow in dep_df.iterrows():
            nm = _name_from_row(drow)
            if not nm:
                continue
            ids = _ids_from_row(drow)
            rows.append({"name": nm, "ssn": ids["ssn"], "dob": ids["dob"], "months": _month_flags(drow)})

    return rows[:9]

def _part3_rows_to_field_values(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for idx, data in enumerate(rows, start=1):
        if idx not in PART3_MAP:
            break
        m = PART3_MAP[idx]
        # Name
        fields[m["name"]] = data.get("name","")

        # SSN or DOB
        ssn = (data.get("ssn") or "").strip()
        dob = (data.get("dob") or "").strip()
        if ssn:
            fields[m["ssn"]] = ssn
        elif dob:
            fields[m["dob"]] = dob

        months = data.get("months") or {}
        # Prefer "All 12" if every month checked and the template has that field
        if months and all(months.get(mn, False) for mn in MONTHS) and m.get("all12"):
            fields[m["all12"]] = "/Yes"
        for mn, fname in m["months"].items():
            if months.get(mn, False):
                fields[fname] = "/Yes"
    return fields

def _fill_acroform(pdf_bytes: bytes, field_values: Dict[str, Any]) -> bytes:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)

    # Ensure appearances are regenerated so text actually shows
    if "/AcroForm" in writer._root_object:
        acro = writer._root_object["/AcroForm"]
        acro.update({NameObject("/NeedAppearances"): BooleanObject(True)})

    for page in writer.pages:
        writer.update_page_form_field_values(page, field_values)

    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)
    return buf.getvalue()

# ===========================
# Public: main fill function
# ===========================

def fill_pdf_for_employee(
    pdf_bytes: bytes,
    emp_row: pd.Series,
    final_df_emp: Optional[pd.DataFrame],
    year_used: int,
    sheets: Optional[Dict[str, pd.DataFrame]] = None,
):
    """
    Returns:
      (editable_name: str, editable_bytes: BytesIO,
       flat_name: str, flat_bytes: BytesIO)
    """
    # Part I — names & SSN
    first, middle, last = _extract_name_from_row(emp_row)
    ssn = _extract_ssn(emp_row)
    full_name = " ".join([x for x in [first, middle, last] if x])

    # Part I — address
    addr = _extract_address(emp_row)

    field_values: Dict[str, Any] = {
        FIELD_FIRST: first,
        FIELD_MIDDLE: middle,
        FIELD_LAST: last,
        FIELD_SSN: ssn,
        FIELD_STREET: addr["street"],
        FIELD_CITY: addr["city"],
        FIELD_STATE: addr["state"],
        FIELD_COUNTRY_ZIP: addr["country_zip"],
    }

    # Line 14 + 16 monthly codes
    l14, l16 = _extract_line_codes(final_df_emp)
    for m in MONTHS:
        f = LINE14_FIELDS.get(m)
        if f:
            field_values[f] = l14.get(m, "")
    for m in LINE16_FIELDS.keys():
        f = LINE16_FIELDS.get(m)
        if f:
            field_values[f] = l16.get(m, "")

    # Part III covered individuals
    emp_id_col = _pick(emp_row.index, ["EmployeeID","EmpID","Employee Id"]) or ""
    employee_id = str(emp_row.get(emp_id_col) or "").strip()
    part3_rows = _extract_part3_rows_from_excel(sheets, employee_id, full_name)
    if part3_rows:
        field_values.update(_part3_rows_to_field_values(part3_rows))

    # Write filled AcroForm (editable)
    out_bytes = _fill_acroform(pdf_bytes, field_values)
    editable_name = f"1095c_{employee_id}.pdf" if employee_id else "1095c.pdf"
    flat_name = editable_name  # same bytes; NeedAppearances set

    return (
        editable_name,
        io.BytesIO(out_bytes),
        flat_name,
        io.BytesIO(out_bytes),
    )

# ==========================================================
# Backward-compatible Excel helper expected by main_fastapi
# ==========================================================
import io as _io
import pandas as _pd

def save_excel_outputs(outputs) -> bytes:
    """
    Backward-compatible shim.

    Accepts either:
      - a dict[str, pandas.DataFrame] of sheets to write, OR
      - a BytesIO / bytes that already contains an .xlsx payload.

    Returns: raw bytes of the Excel workbook.
    """
    # If someone already passed a BytesIO/bytes, just return the bytes
    if isinstance(outputs, (bytes, _io.BytesIO)):
        return outputs if isinstance(outputs, bytes) else outputs.getvalue()

    buf = _io.BytesIO()
    with _pd.ExcelWriter(buf, engine="openpyxl") as xw:
        wrote_any = False
        if isinstance(outputs, dict):
            for name, df in outputs.items():
                if df is None:
                    continue
                sheet = str(name or "Sheet1")[:31]
                if isinstance(df, _pd.DataFrame) and not df.empty:
                    df.to_excel(xw, sheet_name=sheet, index=False)
                else:
                    _pd.DataFrame({"Info": ["(empty)"]}).to_excel(xw, sheet_name=sheet, index=False)
                wrote_any = True
        if not wrote_any:
            _pd.DataFrame({"Info": ["No data"]}).to_excel(xw, sheet_name="Output", index=False)
    buf.seek(0)
    return buf.getvalue()
