# pdf_filler.py
# Create one 1095-C PDF per employee from the interim DataFrame.
# Part I is optional (needs demographics sheet); Part III is optional (needs dependent sheet mapping in JSON).

from __future__ import annotations
import json, io, os
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject, BooleanObject

MONTHS = ["Jan","Feb","Mar","Apr","May","June","July","Aug","Sept","Oct","Nov","Dec"]

def month_to_index(m: str) -> int:
    try:
        return MONTHS.index(str(m))
    except Exception:
        return 0

def enable_need_appearances(reader: PdfReader, writer: PdfWriter):
    try:
        root = reader.trailer.get("/Root")
        if hasattr(root,"get_object"): root = root.get_object()
        acro = root.get("/AcroForm") if root else None
        if hasattr(acro,"get_object"): acro = acro.get_object()
        if acro is None: return
        acro.update({NameObject("/NeedAppearances"): BooleanObject(True)})
        writer._root_object.update({NameObject("/AcroForm"): acro})
    except Exception:
        pass

def set_form_text(writer: PdfWriter, field_name: Optional[str], value: str):
    if not field_name: return
    try:
        writer.update_page_form_field_values(writer.pages[0], {field_name: value})
    except Exception:
        try:
            writer.update_page_form_field_values(writer.pages[0], {field_name.strip(): value})
        except Exception:
            pass

def load_field_map(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if "part1" not in m: m["part1"]={}
    if "part2" not in m: m["part2"]={}
    if "part3" not in m: m["part3"]={"rows":[]}
    for which in ["line14","line16"]:
        if which not in m:
            raise ValueError(f"Field map missing '{which}'")
    return m

def derive_part1_values(emp_id: int, demo_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    out = {"employee_first":None,"employee_middle":None,"employee_last":None,"employee_ssn":None,
           "employee_addr1":None,"employee_city":None,"employee_state":None,"employee_zip":None,"employee_country":None,
           "employer_name":None,"employer_ein":None,"employer_addr1":None,"employer_city":None,"employer_state":None,
           "employer_zip":None,"employer_country":None,"employer_phone":None}
    if demo_df is None or "EmployeeID" not in demo_df.columns: return out
    g = demo_df.loc[demo_df["EmployeeID"]==emp_id]
    if g.empty: return out
    def first(col): 
        s = g.get(col)
        if s is None: return None
        s = s.dropna().astype(str).str.strip()
        return s.iloc[0] if len(s) else None
    mapping = {
        "FirstName":"employee_first","MiddleInitial":"employee_middle","LastName":"employee_last",
        "SSN":"employee_ssn","AddressLine1":"employee_addr1","City":"employee_city","State":"employee_state",
        "ZipCode":"employee_zip","Country":"employee_country",
        "EmployerName":"employer_name","EIN":"employer_ein","EmployerAddress":"employer_addr1","EmployerCity":"employer_city",
        "EmployerState":"employer_state","EmployerZipCode":"employer_zip","EmployerCountry":"employer_country",
        "ContactTelephone":"employer_phone"
    }
    for src,tgt in mapping.items():
        out[tgt] = first(src)
    if out["employee_ssn"]:
        out["employee_ssn"] = str(out["employee_ssn"]).replace("-","").replace(" ","")
    for z in ("employee_zip","employer_zip"):
        if out[z] is not None:
            try: out[z] = str(out[z]).split(".")[0]
            except: pass
    return out

def all_12_same(d: Dict[str, Optional[str]]) -> Optional[str]:
    vals = [d.get(m) for m in MONTHS]
    vals = [v for v in vals if v and str(v).strip()]
    return list(set(vals))[0] if len(vals)==12 and len(set(vals))==1 else None

def fill_line_codes(writer: PdfWriter, fields: Dict[str, Any], codes: Dict[str, Optional[str]], which: str):
    sec = fields.get(which, {}) or {}
    same = all_12_same(codes)
    if same and "all" in sec:
        set_form_text(writer, sec["all"], same)
        for m in MONTHS:
            if m in sec: set_form_text(writer, sec[m], "")
    else:
        for m in MONTHS:
            if m in sec:
                set_form_text(writer, sec[m], codes.get(m) or "")

def build_line_dict(block: pd.DataFrame, col: str) -> Dict[str, Optional[str]]:
    out = {m: None for m in MONTHS}
    for _, r in block.iterrows():
        mo = str(r["Month"])
        if mo in out: out[mo] = r.get(col)
    return out

def build_part3_rows(emp_id: int, year: int, block: pd.DataFrame, dep_df: Optional[pd.DataFrame]) -> List[Dict[str,Any]]:
    persons = []
    # Employee row: covered if employee_enrolled is truthy
    cov = {m: False for m in MONTHS}
    if "employee_enrolled" in block.columns:
        for _, r in block.iterrows():
            mo = str(r["Month"])
            if mo in cov:
                cov[mo] = str(r.get("employee_enrolled")).strip().lower() in {"yes","true","1"}
    persons.append({"name": block.get("Name", pd.Series([str(emp_id)])).iloc[0], "ssn": None, "dob": None, "covered": cov})

    # Dependents from Dep Enrollment (optional)
    if dep_df is not None and not dep_df.empty and "EmployeeID" in dep_df.columns:
        use = dep_df.copy()
        for c in ["EnrollmentStartDate","EnrollmentEndDate"]:
            if c in use.columns: use[c] = pd.to_datetime(use[c], errors="coerce")
        use["EmployeeID"] = pd.to_numeric(use["EmployeeID"], errors="coerce").astype("Int64")
        rows = use.loc[use["EmployeeID"]==emp_id]
        for _, d in rows.iterrows():
            name = ("{} {}".format(str(d.get("DepFirstName") or "").strip(), str(d.get("DepLastName") or "").strip())).strip() or "Dependent"
            covered = {m: False for m in MONTHS}
            s = d.get("EnrollmentStartDate"); e = d.get("EnrollmentEndDate")
            if pd.notna(s) and pd.notna(e):
                for i,mn in enumerate(MONTHS, start=1):
                    ms = pd.Timestamp(year=year, month=i, day=1)
                    me = ms + pd.offsets.MonthEnd(1)
                    if (s <= me) and (e >= ms):
                        covered[mn] = True
            persons.append({"name": name, "ssn": None, "dob": None, "covered": covered})
    return persons

def fill_part1(writer: PdfWriter, fields: Dict[str,Any], vals: Dict[str,Any]):
    p1 = fields.get("part1", {}) or {}
    for k,v in vals.items():
        if k in p1 and v is not None:
            set_form_text(writer, p1[k], str(v))

def fill_part3(writer: PdfWriter, fields: Dict[str,Any], persons: List[Dict[str,Any]]):
    p3 = fields.get("part3", {}) or {}
    rowmaps = p3.get("rows", [])
    if not rowmaps: return
    for idx, person in enumerate(persons[:len(rowmaps)]):
        m = rowmaps[idx]
        if m.get("name") and person.get("name"): set_form_text(writer, m["name"], str(person["name"]))
        if m.get("ssn")  and person.get("ssn"):  set_form_text(writer, m["ssn"],  str(person["ssn"]))
        if m.get("dob")  and person.get("dob"):  set_form_text(writer, m["dob"],  str(person["dob"]))
        all_field = m.get("all")
        months_map = m.get("months", {}) or {}
        # all-12 optimization
        if all(person["covered"].get(mm, False) for mm in MONTHS) and all_field:
            set_form_text(writer, all_field, "X")
            for mm, ff in months_map.items():
                set_form_text(writer, ff, "")
        else:
            for mm, ff in months_map.items():
                set_form_text(writer, ff, "X" if person["covered"].get(mm, False) else "")

def generate_all_pdfs(
    interim_df: pd.DataFrame,
    year: int,
    template_path: str,
    fields_json_path: str,
    demo_df: Optional[pd.DataFrame] = None,
    dep_df: Optional[pd.DataFrame] = None
) -> List[Tuple[str, bytes]]:
    """
    Returns [(filename, pdf_bytes), ...]
    """
    fields = load_field_map(fields_json_path)
    out: List[Tuple[str, bytes]] = []

    for emp_id in sorted(interim_df["Employee_ID"].dropna().unique().tolist()):
        block = interim_df.loc[interim_df["Employee_ID"] == emp_id].copy()
        if block.empty: 
            continue
        block = block.sort_values(key=lambda s: s.map(month_to_index) if s.name=="Month" else s)

        # Build Line14/Line16 dicts
        l14 = build_line_dict(block, "line_14")
        l16 = build_line_dict(block, "line_16")

        # PDF
        reader = PdfReader(template_path)
        writer = PdfWriter()
        for p in reader.pages: writer.add_page(p)
        enable_need_appearances(reader, writer)

        # Part I (optional)
        p1_vals = derive_part1_values(int(emp_id), demo_df)
        fill_part1(writer, fields, p1_vals)

        # Part II
        fill_line_codes(writer, fields, l14, "line14")
        fill_line_codes(writer, fields, l16, "line16")

        # Part III (optional)
        persons = build_part3_rows(int(emp_id), year, block, dep_df)
        fill_part3(writer, fields, persons)

        # Save to bytes
        buf = io.BytesIO()
        writer.write(buf)
        out.append((f"1095C_{int(emp_id)}.pdf", buf.getvalue()))

    return out
