# aca_pdf.py
import io
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject, BooleanObject, DictionaryObject
from reportlab.pdfgen import canvas

from aca_processing import MONTHS, _coerce_str

# --------- PDF field names (2024) ----------
F_PART1 = ["f1_1[0]","f1_2[0]","f1_3[0]","f1_4[0]","f1_5[0]","f1_6[0]","f1_7[0]","f1_8[0]"]
F_L14   = ["f1_17[0]","f1_18[0]","f1_19[0]","f1_20[0]","f1_21[0]","f1_22[0]","f1_23[0]",
           "f1_24[0]","f1_25[0]","f1_26[0]","f1_27[0]","f1_28[0]","f1_29[0]"]
F_L16   = ["f1_43[0]","f1_44[0]","f1_45[0]","f1_46[0]","f1_47[0]","f1_48[0]","f1_49[0]",
           "f1_50[0]","f1_51[0]","f1_52[0]","f1_53[0]","f1_54[0]"]

def normalize_ssn_digits(ssn: str) -> str:
    d = "".join(ch for ch in str(ssn) if str(ch).isdigit())
    return f"{d[0:3]}-{d[3:5]}-{d[5:9]}" if len(d)>=9 else d

def set_need_appearances(writer: PdfWriter):
    root = writer._root_object
    if "/AcroForm" not in root:
        root.update({NameObject("/AcroForm"): DictionaryObject()})
    root["/AcroForm"].update({NameObject("/NeedAppearances"): BooleanObject(True)})

def find_rects(reader: PdfReader, target_names, page_index=0):
    rects = {}
    pg = reader.pages[page_index]
    annots = pg.get("/Annots")
    if not annots: return rects
    try:
        arr = annots.get_object()
    except Exception:
        arr = annots
    for a in arr:
        obj = a.get_object()
        if obj.get("/Subtype") != "/Widget": continue
        nm = obj.get("/T")
        ft = obj.get("/FT")
        if ft != "/Tx" or nm not in target_names: continue
        r = obj.get("/Rect")
        if r and len(r) == 4:
            rects[nm] = tuple(float(r[i]) for i in range(4))
    return rects

def build_overlay(page_w, page_h, rects_and_values, font="Helvetica", font_size=10.5, inset=2.0):
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=(page_w, page_h))
    c.setFont(font, font_size)
    for rect, val in rects_and_values:
        if not val: continue
        x0,y0,x1,y1 = rect
        text_x = x0 + inset
        text_y = y1 - font_size - inset
        if text_y < y0 + inset: text_y = y0 + inset
        c.drawString(text_x, text_y, val)
    c.save()
    packet.seek(0)
    return PdfReader(packet)

def flatten_pdf(reader: PdfReader):
    out = PdfWriter()
    for i, page in enumerate(reader.pages):
        annots = page.get("/Annots")
        if annots:
            try: arr = annots.get_object()
            except Exception: arr = annots
            keep=[]
            for a in arr:
                try:
                    if a.get_object().get("/Subtype") != "/Widget": keep.append(a)
                except Exception:
                    keep.append(a)
            if keep:
                page[NameObject("/Annots")] = keep
            else:
                if "/Annots" in page: del page[NameObject("/Annots")]
        out.add_page(page)
    if "/AcroForm" in out._root_object:
        del out._root_object[NameObject("/AcroForm")]
    return out

def fill_pdf_for_employee(pdf_bytes: bytes,
                          emp_row: pd.Series,
                          final_df_emp: pd.DataFrame,
                          year_used: int,
                          interim_df_emp: pd.DataFrame | None = None):
    """Returns: (editable_name, editable_bytes, flattened_name, flattened_bytes)."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    page0 = reader.pages[0]
    W = float(page0.mediabox.width); H = float(page0.mediabox.height)

    # Part I
    first  = _coerce_str(emp_row.get("firstname"))
    mi     = ""
    last   = _coerce_str(emp_row.get("lastname"))
    ssn    = normalize_ssn_digits(_coerce_str(emp_row.get("ssn")))
    addr1  = _coerce_str(emp_row.get("addressline1"))
    addr2  = _coerce_str(emp_row.get("addressline2"))
    city   = _coerce_str(emp_row.get("city"))
    state  = _coerce_str(emp_row.get("state"))
    zipcode= _coerce_str(emp_row.get("zipcode"))
    street = addr1 if not addr2 else f"{addr1} {addr2}"

    part1_map = {"f1_1[0]": first, "f1_2[0]": mi, "f1_3[0]": last, "f1_4[0]": ssn,
                 "f1_5[0]": street, "f1_6[0]": city, "f1_7[0]": state, "f1_8[0]": zipcode}

    # Part II — detect year-level 1G
    line14_all12 = ""
    if interim_df_emp is not None and "line14_all12" in interim_df_emp.columns:
        if (interim_df_emp["line14_all12"] == "1G").any():
            line14_all12 = "1G"

    l14_by_m = {row["Month"]: _coerce_str(row["Line14_Final"]) for _,row in final_df_emp.iterrows()}
    l16_by_m = {row["Month"]: _coerce_str(row["Line16_Final"]) for _,row in final_df_emp.iterrows()}

    def all12_value(d):
        vals = [d.get(m, "") for m in MONTHS]
        uniq = {v for v in vals if v}
        return list(uniq)[0] if len(uniq)==1 else ""

    if line14_all12 == "1G":
        l14_values = ["1G"] + ["" for _ in MONTHS]
        l16_values = [""   ] + ["" for _ in MONTHS]
    else:
        l14_values = [all12_value(l14_by_m)] + [l14_by_m.get(m,"") for m in MONTHS]
        l16_values = [all12_value(l16_by_m)] + [l16_by_m.get(m,"") for m in MONTHS]

    mapping = {}
    for name,val in zip(F_PART1, [first, mi, last, ssn, street, city, state, zipcode]): mapping[name]=val
    for name,val in zip(F_L14, l14_values): mapping[name]=val
    for name,val in zip(F_L16, l16_values): mapping[name]=val

    writer_edit = PdfWriter()
    for i in range(len(reader.pages)): writer_edit.add_page(reader.pages[i])
    for i in range(len(writer_edit.pages)):
        try: writer_edit.update_page_form_field_values(writer_edit.pages[i], mapping)
        except Exception: pass
    root = writer_edit._root_object
    if "/AcroForm" not in root:
        root.update({NameObject("/AcroForm"): DictionaryObject()})
    root["/AcroForm"].update({NameObject("/NeedAppearances"): BooleanObject(True)})

    rects = find_rects(reader, list(mapping.keys()), page_index=0)
    overlay_pairs = [(rects[nm], mapping[nm]) for nm in mapping if nm in rects and mapping[nm]]
    if overlay_pairs:
        overlay_pdf = build_overlay(W, H, overlay_pairs)
        writer_edit.pages[0].merge_page(overlay_pdf.pages[0])

    first_last = f"{first}_{last}".strip().replace(" ","_") or (_coerce_str(emp_row.get("employeeid")) or "employee")
    editable_name = f"1095c_filled_fields_{first_last}_{year_used}.pdf"
    editable_bytes = io.BytesIO(); writer_edit.write(editable_bytes); editable_bytes.seek(0)

    reader_after = PdfReader(io.BytesIO(editable_bytes.getvalue()))
    writer_flat = flatten_pdf(reader_after)
    flattened_name = f"1095c_filled_flattened_{first_last}_{year_used}.pdf"
    flattened_bytes = io.BytesIO(); writer_flat.write(flattened_bytes); flattened_bytes.seek(0)
    return editable_name, editable_bytes, flattened_name, flattened_bytes

# ---------- Excel writer (stable) ----------
def save_excel_outputs(interim: pd.DataFrame, final: pd.DataFrame, year:int, penalty_dashboard: pd.DataFrame | None = None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as xw:
        final.to_excel(xw, index=False, sheet_name=f"Final {year}")
        interim.to_excel(xw, index=False, sheet_name=f"Interim {year}")
        if penalty_dashboard is not None and not penalty_dashboard.empty:
            penalty_dashboard.to_excel(xw, index=False, sheet_name=f"Penalty Dashboard {year}")
    buf.seek(0)
    return buf.getvalue()
