# aca_core.py
# Core logic for ACA-1095 processing (Excel → interim/final) and PDF filling.
# No Streamlit code in this file.

import io, re
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject, BooleanObject, DictionaryObject
from reportlab.pdfgen import canvas

# =========================
# Constants & helpers
# =========================
TRUTHY = {"y","yes","true","t","1",1,True}
FALSY  = {"n","no","false","f","0",0,False,None,np.nan}
ACTIVE_STATUS = {"FT","FULL-TIME","FULL TIME","PT","PART-TIME","PART TIME","ACTIVE"}

FULLTIME_ROLES  = {"FT","FULL-TIME","FULL TIME"}
FULLTIME_STATUS = {"CATEGORY2"}  # per your rule: Category2 full-month counts as FT
PARTTIME_ROLES  = {"PT","PART-TIME","PART TIME"}  # informational; FT takes precedence

AFFORDABILITY_THRESHOLD = 50.00  # placeholder affordability toggle 1A vs 1E

EXPECTED_SHEETS = {
    "emp demographic": ["employeeid","firstname","lastname","ssn","addressline1","addressline2","city","state","zipcode"],
    "emp status": ["employeeid","employmentstatus","role","statusstartdate","statusenddate"],
    "emp eligibility": ["employeeid","iseligibleforcoverage","minimumvaluecoverage","eligibilitystartdate","eligibilityenddate"],
    "emp enrollment": ["employeeid","isenrolled","enrollmentstartdate","enrollmentenddate"],
    "dep enrollment": ["employeeid","dependentrelationship","eligible","enrolled","eligiblestartdate","eligibleenddate"],
    "pay deductions": ["employeeid","amount","startdate","enddate"]
}
CANON_ALIASES = {
    "mimimumvaluecoverage": "minimumvaluecoverage",
    "minimimvaluecoverage": "minimumvaluecoverage",
    "zip": "zipcode", "zip code": "zipcode",
    "ssn (digits only)": "ssn",
}

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip().str.replace(r"\s+", " ", regex=True).str.lower())
    return df

def _coerce_str(x) -> str:
    if pd.isna(x): return ""
    return str(x).strip()

def _normalize_employeeid(x) -> str:
    """Unify EmployeeID across sheets: '1001', '1001.0', '1001.00', '1,001' → '1001'.
       Keeps leading zeros (e.g., '00123' stays '00123')."""
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    s = str(x).strip().replace(",", "")
    if s == "" or s.lower() in {"nan","none"}: return ""
    # if numeric float-like and integer-valued → int string
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            # preserve leading zeros only if original looked like a zero-padded string (not numeric)
            if re.fullmatch(r"\d+\.0*", s):
                return str(int(f))
    except Exception:
        pass
    # strip trailing .0..0 patterns
    m = re.fullmatch(r"(\d+)\.0+", s)
    if m: return m.group(1)
    return s

def to_bool(val) -> bool:
    if isinstance(val, str):
        v = val.strip().lower()
        if v in TRUTHY: return True
        if v in FALSY:  return False
    return bool(val) and val not in FALSY

def _last_day_of_month(y: int, m: int) -> date:
    return date(y,12,31) if m==12 else (date(y, m+1, 1) - timedelta(days=1))

def parse_date_safe(d, default_end: bool=False):
    if pd.isna(d): return None
    if isinstance(d, (datetime, np.datetime64)):
        dt = pd.to_datetime(d, errors="coerce");  return None if pd.isna(dt) else dt.date()
    s = str(d).strip()
    if not s: return None
    try:
        if len(s)==4 and s.isdigit():
            y = int(s); return date(y,12,31) if default_end else date(y,1,1)
        if len(s)==7 and s[4]=="-":
            y,m = map(int, s.split("-"));  return _last_day_of_month(y,m) if default_end else date(y,m,1)
    except:  # noqa: E722
        pass
    dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        try:
            y,m = map(int, s.split("-")[:2])
            return _last_day_of_month(y,m) if default_end else date(y,m,1)
        except:  # noqa: E722
            return None
    return dt.date()

def month_bounds(year:int, month:int):
    return date(year, month, 1), _last_day_of_month(year, month)

def _any_overlap(df, start_col, end_col, m_start, m_end, mask=None) -> bool:
    if df.empty: return False
    _m = mask if mask is not None else pd.Series(True, index=df.index)
    s = df.loc[_m, start_col].fillna(pd.Timestamp.min).dt.date
    e = df.loc[_m, end_col].fillna(pd.Timestamp.max).dt.date
    return bool(((e >= m_start) & (s <= m_end)).any())

def _all_month(df, start_col, end_col, m_start, m_end, mask=None) -> bool:
    if df.empty: return False
    _m = mask if mask is not None else pd.Series(True, index=df.index)
    s = df.loc[_m, start_col].fillna(pd.Timestamp.min).dt.date
    e = df.loc[_m, end_col].fillna(pd.Timestamp.max).dt.date
    return bool(((s <= m_start) & (e >= m_end)).any())

# =========================
# Excel ingestion & transforms
# =========================
def load_excel(file_bytes: bytes) -> dict:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    out = {}
    for raw in xls.sheet_names:
        df = pd.read_excel(xls, raw)
        df = normalize_columns(df)
        df = df.rename(columns={k:v for k,v in CANON_ALIASES.items() if k in df.columns})
        # normalize EmployeeID if present
        if "employeeid" in df.columns:
            df["employeeid"] = df["employeeid"].map(_normalize_employeeid)
        out[raw.strip().lower()] = df
    return out

def _pick_sheet(data: dict, key: str) -> pd.DataFrame:
    if key in data: return data[key]
    for k in data:
        if key in k: return data[k]
    return pd.DataFrame()

def _ensure_employeeid_str(df):
    if df.empty or "employeeid" not in df.columns: return df
    df = df.copy()
    df["employeeid"] = df["employeeid"].map(_normalize_employeeid)
    return df

def _parse_date_cols(df, cols, default_end_cols=()):
    if df.empty: return df
    df = df.copy(); endset = set(default_end_cols)
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: parse_date_safe(x, default_end=c in endset))
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _boolify(df, cols):
    if df.empty: return df
    df = df.copy()
    for c in cols:
        if c in df.columns: df[c] = df[c].apply(to_bool)
    return df

def prepare_inputs(data: dict):
    cleaned = {}
    for sheet, cols in EXPECTED_SHEETS.items():
        df = _pick_sheet(data, sheet)
        if df.empty:
            cleaned[sheet] = pd.DataFrame(columns=cols); continue
        for misspell, canon in CANON_ALIASES.items():
            if misspell in df.columns and canon not in df.columns:
                df = df.rename(columns={misspell: canon})
        df = _ensure_employeeid_str(df)
        if sheet == "emp status":
            if "employmentstatus" in df.columns:
                df["employmentstatus"] = df["employmentstatus"].astype(str).str.strip().str.upper()
            if "role" in df.columns:
                df["role"] = df["role"].astype(str).str.strip().str.upper()
            df = _parse_date_cols(df, ["statusstartdate","statusenddate"], default_end_cols=["statusenddate"])
        elif sheet == "emp eligibility":
            df = _boolify(df, ["iseligibleforcoverage","minimumvaluecoverage"])
            df = _parse_date_cols(df, ["eligibilitystartdate","eligibilityenddate"], default_end_cols=["eligibilityenddate"])
        elif sheet == "emp enrollment":
            df = _boolify(df, ["isenrolled"])
            df = _parse_date_cols(df, ["enrollmentstartdate","enrollmentenddate"], default_end_cols=["enrollmentenddate"])
        elif sheet == "dep enrollment":
            if "dependentrelationship" in df.columns:
                df["dependentrelationship"] = df["dependentrelationship"].astype(str).str.strip().str.title()
            df = _boolify(df, ["eligible","enrolled"])
            df = _parse_date_cols(df, ["eligiblestartdate","eligibleenddate"], default_end_cols=["eligibleenddate"])
        elif sheet == "pay deductions":
            df = _parse_date_cols(df, ["startdate","enddate"], default_end_cols=["enddate"])
        cleaned[sheet] = df
    return (cleaned["emp demographic"], cleaned["emp status"], cleaned["emp eligibility"],
            cleaned["emp enrollment"], cleaned["dep enrollment"], cleaned["pay deductions"])

def choose_report_year(emp_elig: pd.DataFrame, fallback_to_current=True) -> int:
    if emp_elig.empty or not {"eligibilitystartdate","eligibilityenddate"} <= set(emp_elig.columns):
        return datetime.now().year if fallback_to_current else 2024
    counts={}
    for _,r in emp_elig.iterrows():
        s = pd.to_datetime(r.get("eligibilitystartdate"), errors="coerce")
        e = pd.to_datetime(r.get("eligibilityenddate"), errors="coerce")
        if pd.isna(s) and pd.isna(e): continue
        s = s or pd.Timestamp.min; e = e or pd.Timestamp.max
        for y in range(s.year, e.year+1): counts[y]=counts.get(y,0)+1
    return max(sorted(counts), key=lambda y:(counts[y], y)) if counts else (datetime.now().year if fallback_to_current else 2024)

def _collect_employee_ids(*dfs):
    ids=set()
    for df in dfs:
        if df is None or df.empty: continue
        if "employeeid" in df.columns:
            ids.update(map(_normalize_employeeid, df["employeeid"].dropna().tolist()))
    return sorted(ids)

def _grid_for_year(employee_ids, year:int) -> pd.DataFrame:
    recs=[]
    for emp in employee_ids:
        for m in range(1,13):
            ms,me = month_bounds(year,m)
            recs.append({"employeeid":emp,"year":year,"monthnum":m,"month":ms.strftime("%b"),
                         "monthstart":ms,"monthend":me})
    g = pd.DataFrame.from_records(recs)
    g["monthstart"]=pd.to_datetime(g["monthstart"]); g["monthend"]=pd.to_datetime(g["monthend"])
    return g

# =========================
# Pay deduction picker (for Line 14 1A/1E affordability)
# =========================
def _pick_monthly_deduction(pay_df: pd.DataFrame, emp: str, ms: date, me: date) -> float | None:
    if pay_df is None or pay_df.empty:
        return None
    df = pay_df[pay_df["employeeid"] == emp].copy()
    if df.empty or "startdate" not in df.columns or "enddate" not in df.columns:
        return None
    ov = df[(df["enddate"].fillna(pd.Timestamp.max).dt.date >= ms) & (df["startdate"].fillna(pd.Timestamp.min).dt.date <= me)]
    if ov.empty:
        return None
    ov = ov.sort_values("startdate", ascending=False)
    amt_col = "amount" if "amount" in ov.columns else None
    if not amt_col:
        for c in ov.columns:
            if c in {"employeecontribution","contribution","emplcost","cost"}:
                amt_col = c; break
    if not amt_col:
        return None
    try:
        val = pd.to_numeric(ov.iloc[0][amt_col], errors="coerce")
        return float(val) if not pd.isna(val) else None
    except Exception:
        return None

# =========================
# Core: Interim / Final
# =========================
def build_interim(emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, year=None, pay_deductions=None) -> pd.DataFrame:
    if year is None: year = choose_report_year(emp_elig)
    employee_ids = _collect_employee_ids(emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll)
    grid = _grid_for_year(employee_ids, year)

    # de-dup demographic rows early and ensure normalized IDs
    demo = pd.DataFrame(columns=["employeeid","firstname","lastname"])
    if not emp_demo.empty:
        tmp = emp_demo.copy()
        if "employeeid" in tmp.columns:
            tmp["employeeid"] = tmp["employeeid"].map(_normalize_employeeid)
        demo = tmp[["employeeid","firstname","lastname"]].drop_duplicates("employeeid", keep="first")

    out = grid.merge(demo, on="employeeid", how="left")

    stt, elg, enr, dep = emp_status.copy(), emp_elig.copy(), emp_enroll.copy(), dep_enroll.copy()
    pay = pay_deductions.copy() if pay_deductions is not None else pd.DataFrame()

    for df in (stt,elg,enr,dep,pay):
        if (df is not None) and (not df.empty):
            if "employeeid" in df.columns:
                df["employeeid"] = df["employeeid"].map(_normalize_employeeid)
            for c in df.columns:
                if c.endswith("date") and not np.issubdtype(df[c].dtype, np.datetime64):
                    df[c] = pd.to_datetime(df[c], errors="coerce")

    flags=[]
    for _,row in out.iterrows():
        emp = row["employeeid"]; ms=row["monthstart"].date(); me=row["monthend"].date()
        st_emp = stt[stt["employeeid"]==emp] if not stt.empty else stt
        el_emp = elg[elg["employeeid"]==emp] if not elg.empty else elg
        en_emp = enr[enr["employeeid"]==emp] if not enr.empty else enr
        de_emp = dep[dep["employeeid"]==emp] if not dep.empty else dep

        # Employment flags
        employed=False
        if not st_emp.empty and {"employmentstatus","statusstartdate","statusenddate"} <= set(st_emp.columns):
            employed = _any_overlap(st_emp, "statusstartdate","statusenddate", ms,me, mask=st_emp["employmentstatus"].isin(ACTIVE_STATUS))

        # FT / PT determination
        ft_full_by_role = ft_full_by_cat2 = False
        if not st_emp.empty and {"role","statusstartdate","statusenddate"} <= set(st_emp.columns):
            ft_full_by_role = _all_month(st_emp, "statusstartdate","statusenddate", ms,me, mask=st_emp["role"].isin(FULLTIME_ROLES))
        if not st_emp.empty and {"employmentstatus","statusstartdate","statusenddate"} <= set(st_emp.columns):
            ft_full_by_cat2 = _all_month(st_emp, "statusstartdate","statusenddate", ms,me, mask=st_emp["employmentstatus"].isin(FULLTIME_STATUS))
        ft_full_month = bool(ft_full_by_role or ft_full_by_cat2)

        pt_full_by_role = False
        if not st_emp.empty and {"role","statusstartdate","statusenddate"} <= set(st_emp.columns):
            pt_full_by_role = _all_month(st_emp, "statusstartdate","statusenddate", ms,me, mask=st_emp["role"].isin(PARTTIME_ROLES))
        pt_full_month = bool(pt_full_by_role and not ft_full_month)

        # Eligibility flags (employee)
        eligible_any=False; eligible_allmonth=False; eligible_mv_full=False
        if not el_emp.empty and {"eligibilitystartdate","eligibilityenddate"} <= set(el_emp.columns):
            eligible_any = _any_overlap(el_emp, "eligibilitystartdate","eligibilityenddate", ms,me)
            eligible_allmonth = _all_month(el_emp, "eligibilitystartdate","eligibilityenddate", ms,me)
            if "minimumvaluecoverage" in el_emp.columns:
                mv_mask = el_emp["minimumvaluecoverage"].fillna(False).astype(bool)
                eligible_mv_full = _all_month(el_emp, "eligibilitystartdate","eligibilityenddate", ms,me, mask=mv_mask)

        # Enrollment flags
        enrolled_any=False; enrolled_allmonth=False
        if not en_emp.empty and {"enrollmentstartdate","enrollmentenddate"} <= set(en_emp.columns):
            en_mask = en_emp["isenrolled"].fillna(True) if "isenrolled" in en_emp.columns else pd.Series(True,index=en_emp.index)
            enrolled_any = _any_overlap(en_emp, "enrollmentstartdate","enrollmentenddate", ms,me, mask=en_mask)
            enrolled_allmonth = _all_month(en_emp, "enrollmentstartdate","enrollmentenddate", ms,me, mask=en_mask)

        # Dependent offers: full-month
        offer_spouse_full=False; offer_child_full=False
        if not de_emp.empty and {"dependentrelationship","eligiblestartdate","eligibleenddate"} <= set(de_emp.columns):
            offer_spouse_full = _all_month(de_emp, "eligiblestartdate","eligibleenddate", ms,me, mask=de_emp["dependentrelationship"].eq("Spouse"))
            offer_child_full  = _all_month(de_emp, "eligiblestartdate","eligibleenddate", ms,me, mask=de_emp["dependentrelationship"].eq("Child"))

        waitingperiod_month = bool(employed and ft_full_month and not eligible_any)
        monthly_cost = _pick_monthly_deduction(pay, emp, ms, me)

        # ----- LINE 14 -----
        offer_ee_full = bool(eligible_allmonth)
        if offer_ee_full:
            if eligible_mv_full:
                if ft_full_month and offer_spouse_full and offer_child_full and (monthly_cost is not None) and (monthly_cost <= AFFORDABILITY_THRESHOLD):
                    l14 = "1A"
                elif ft_full_month and offer_spouse_full and offer_child_full and (monthly_cost is not None) and (monthly_cost > AFFORDABILITY_THRESHOLD):
                    l14 = "1E"
                else:
                    if (not offer_spouse_full) and (not offer_child_full):
                        l14 = "1B"
                    elif offer_child_full and (not offer_spouse_full):
                        l14 = "1C"
                    elif offer_spouse_full and (not offer_child_full):
                        l14 = "1D"
                    else:
                        l14 = "1E"
            else:
                l14 = "1F"
        else:
            if (not ft_full_month) and enrolled_any:
                l14 = "1G"
            else:
                l14 = "1H"

        # ----- LINE 16 -----
        if l14 == "1A":
            l16 = ""
        elif not employed:
            l16 = "2A"
        elif enrolled_allmonth:
            l16 = "2C"
        elif employed and (not offer_ee_full):
            l16 = "2D"
        elif not ft_full_month:
            l16 = "2B"
        else:
            l16 = ""

        flags.append({
            "employed": employed,
            "ft": ft_full_month,
            "parttime": pt_full_month,   # informational
            "eligibleforcoverage": eligible_any,
            "eligible_allmonth": eligible_allmonth,
            "eligible_mv": eligible_mv_full,
            "offer_ee_allmonth": offer_ee_full,
            "enrolled_allmonth": enrolled_allmonth,
            "offer_spouse": offer_spouse_full,
            "offer_dependents": offer_child_full,
            "waitingperiod_month": waitingperiod_month,
            "line14_final": l14,
            "line16_final": l16,
        })

    interim = pd.concat([out.reset_index(drop=True), pd.DataFrame(flags)], axis=1)
    base_cols = ["employeeid","firstname","lastname","year","monthnum","month","monthstart","monthend"]
    flag_cols = [
        "employed","ft","parttime",
        "eligibleforcoverage","eligible_allmonth","eligible_mv","offer_ee_allmonth",
        "enrolled_allmonth","offer_spouse","offer_dependents","waitingperiod_month",
        "line14_final","line16_final"
    ]
    keep = [c for c in base_cols if c in interim.columns] + [c for c in flag_cols if c in interim.columns]
    interim = interim[keep]

    # Safety: ensure 1 row per (EmployeeID, Year, Month)
    interim = interim.drop_duplicates(subset=["employeeid","year","monthnum"]).sort_values(
        ["employeeid","year","monthnum"]
    ).reset_index(drop=True)

    return interim

def build_final(interim: pd.DataFrame) -> pd.DataFrame:
    df = interim.copy()
    out = df.loc[:, ["employeeid","month","line14_final","line16_final"]].rename(columns={
        "employeeid":"EmployeeID","month":"Month","line14_final":"Line14_Final","line16_final":"Line16_Final"
    })
    if "monthnum" in df.columns:
        out = out.join(df["monthnum"]).sort_values(["EmployeeID","monthnum"]).drop(columns=["monthnum"])
    else:
        order = {m:i for i,m in enumerate(MONTHS, start=1)}
        out["_ord"]=out["Month"].map(order); out=out.sort_values(["EmployeeID","_ord"]).drop(columns=["_ord"])
    return out.reset_index(drop=True)

# =========================
# PDF helpers (Part I + Part II)
# =========================
def normalize_ssn_digits(ssn: str) -> str:
    d = "".join(ch for ch in str(ssn) if str(ch).isdigit())
    return f"{d[0:3]}-{d[3:5]}-{d[5:9]}" if len(d)>=9 else d

# 2024 PDF field names (page 1)
F_PART1 = ["f1_1[0]","f1_2[0]","f1_3[0]","f1_4[0]","f1_5[0]","f1_6[0]","f1_7[0]","f1_8[0]"]
F_L14   = ["f1_17[0]","f1_18[0]","f1_19[0]","f1_20[0]","f1_21[0]","f1_22[0]","f1_23[0]",
           "f1_24[0]","f1_25[0]","f1_26[0]","f1_27[0]","f1_28[0]","f1_29[0]"]
F_L16   = ["f1_43[0]","f1_44[0]","f1_45[0]","f1_46[0]","f1_47[0]","f1_48[0]","f1_49[0]",
           "f1_50[0]","f1_51[0]","f1_52[0]","f1_53[0]","f1_54[0]"]

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
        if obj.get("/Subtype") != "/Widget": 
            continue
        nm = obj.get("/T")
        ft = obj.get("/FT")
        if ft != "/Tx" or nm not in target_names: 
            continue
        r = obj.get("/Rect")
        if r and len(r) == 4:
            rects[nm] = tuple(float(r[i]) for i in range(4))
    return rects

def build_overlay(page_w, page_h, rects_and_values, font="Helvetica", font_size=10.5, inset=2.0):
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=(page_w, page_h))
    c.setFont(font, font_size)
    for rect, val in rects_and_values:
        if not val: 
            continue
        x0,y0,x1,y1 = rect
        text_x = x0 + inset
        text_y = y1 - font_size - inset
        if text_y < y0 + inset: 
            text_y = y0 + inset
        c.drawString(text_x, text_y, val)
    c.save()
    packet.seek(0)
    return PdfReader(packet)

def flatten_pdf(reader: PdfReader):
    out = PdfWriter()
    for i, page in enumerate(reader.pages):
        annots = page.get("/Annots")
        if annots:
            try:
                arr = annots.get_object()
            except Exception:
                arr = annots
            keep=[]
            for a in arr:
                try:
                    if a.get_object().get("/Subtype") != "/Widget":
                        keep.append(a)
                except Exception:
                    keep.append(a)
            if keep:
                page[NameObject("/Annots")] = keep
            else:
                if "/Annots" in page:
                    del page[NameObject("/Annots")]
        out.add_page(page)
    if "/AcroForm" in out._root_object:
        del out._root_object[NameObject("/AcroForm")]
    return out

def fill_pdf_for_employee(pdf_bytes: bytes, emp_row: pd.Series, final_df_emp: pd.DataFrame, year_used: int):
    """Returns: (editable_name, editable_bytes, flattened_name, flattened_bytes)"""
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

    # Part II (L14/L16)
    l14_by_m = {row["Month"]: _coerce_str(row["Line14_Final"]) for _,row in final_df_emp.iterrows()}
    l16_by_m = {row["Month"]: _coerce_str(row["Line16_Final"]) for _,row in final_df_emp.iterrows()}

    def all12_value(d):
        vals = [d.get(m, "") for m in MONTHS]
        uniq = {v for v in vals if v}
        return list(uniq)[0] if len(uniq)==1 else ""

    l14_values = [all12_value(l14_by_m)] + [l14_by_m.get(m,"") for m in MONTHS]
    l16_values = [all12_value(l16_by_m)] + [l16_by_m.get(m,"") for m in MONTHS]

    part2_map = {}
    for name,val in zip(F_L14, l14_values): part2_map[name]=val
    for name,val in zip(F_L16, l16_values): part2_map[name]=val

    mapping = {**part1_map, **part2_map}

    writer_edit = PdfWriter()
    for i in range(len(reader.pages)):
        writer_edit.add_page(reader.pages[i])
    for i in range(len(writer_edit.pages)):
        try:
            writer_edit.update_page_form_field_values(writer_edit.pages[i], mapping)
        except Exception:
            pass
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

def save_excel_outputs(interim: pd.DataFrame, final: pd.DataFrame, year:int) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as xw:
        final.to_excel(xw, index=False, sheet_name=f"Final {year}")
        interim.to_excel(xw, index=False, sheet_name=f"Interim {year}")
    buf.seek(0)
    return buf.getvalue()
