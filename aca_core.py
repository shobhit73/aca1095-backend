# aca_core.py
# Core logic for ACA-1095 processing (Excel → interim/final), PDF filling, and Penalty Dashboard.
# No web framework code in this file.

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

# Canonical token sets (after normalization)
FT_TOKENS = {"FT","FULLTIME","FTE","CATEGORY2","CAT2"}
PT_TOKENS = {"PT","PARTTIME","PTE"}
# Treat ACTIVE + LOA as 'employed' for overlap detection; role FT/PT also implies 'employed'
EMPLOYED_TOKENS = {"ACTIVE","LOA"} | FT_TOKENS | PT_TOKENS

AFFORDABILITY_THRESHOLD = 50.00  # for 1A vs 1E threshold

EXPECTED_SHEETS = {
    "emp demographic": ["employeeid","firstname","lastname","ssn","addressline1","addressline2","city","state","zipcode","role","employmentstatus","statusstartdate","statusenddate"],
    "emp status": ["employeeid","employmentstatus","role","statusstartdate","statusenddate"],  # optional (derived from demo if missing)
    "emp eligibility": ["employeeid","iseligibleforcoverage","minimumvaluecoverage","eligibilitystartdate","eligibilityenddate"],
    "emp enrollment": ["employeeid","isenrolled","enrollmentstartdate","enrollmentenddate"],
    "dep enrollment": ["employeeid","dependentrelationship","eligible","enrolled","eligiblestartdate","eligibleenddate","enrollmentstartdate","enrollmentenddate","plancode"],
    "pay deductions": ["employeeid","amount","startdate","enddate"]
}
CANON_ALIASES = {
    "mimimumvaluecoverage": "minimumvaluecoverage",
    "minimimvaluecoverage": "minimumvaluecoverage",
    "zip": "zipcode", "zip code": "zipcode",
    "ssn (digits only)": "ssn",
}

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
FULL_MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]
MONTHNUM_TO_FULL = {i+1: m for i,m in enumerate(FULL_MONTHS)}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip().str.replace(r"\s+", " ", regex=True).str.lower())
    return df

def _coerce_str(x) -> str:
    if pd.isna(x): return ""
    return str(x).strip()

def _norm_token(x) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())

def _normalize_employeeid(x) -> str:
    """Unify EmployeeID: '1001', '1001.0', '1,001' → '1001'."""
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    s = str(x).strip().replace(",", "")
    if s == "" or s.lower() in {"nan","none"}: return ""
    m = re.fullmatch(r"(\d+)\.0+", s)
    if m: return m.group(1)
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
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
    except:
        pass
    dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        try:
            y,m = map(int, s.split("-")[:2])
            return _last_day_of_month(y,m) if default_end else date(y,m,1)
        except:
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
            # optional sheet; we’ll derive from demo if absent
            if "employmentstatus" in df.columns:
                df["employmentstatus"] = df["employmentstatus"].astype(str).str.strip()
            if "role" in df.columns:
                df["role"] = df["role"].astype(str).str.strip()
            if "employmentstatus" in df.columns:
                df["_estatus_norm"] = df["employmentstatus"].map(_norm_token)
            if "role" in df.columns:
                df["_role_norm"] = df["role"].map(_norm_token)
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
            df = _parse_date_cols(
                df,
                ["eligiblestartdate","eligibleenddate","enrollmentstartdate","enrollmentenddate"],
                default_end_cols=["eligibleenddate","enrollmentenddate"]
            )
            if "plancode" in df.columns:
                df["plancode"] = df["plancode"].astype(str).str.strip()
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
# Build a status table from Emp Demographic (Role/EmploymentStatus are dated here)
# =========================
def _status_from_demographic(emp_demo: pd.DataFrame) -> pd.DataFrame:
    need = {"employeeid","role","employmentstatus","statusstartdate","statusenddate"}
    if emp_demo.empty or not need <= set(emp_demo.columns):
        return pd.DataFrame(columns=list(need))
    st = emp_demo.loc[:, list(need)].copy()
    st["employeeid"] = st["employeeid"].map(_normalize_employeeid)
    st["role"] = st["role"].astype(str).str.strip()
    st["employmentstatus"] = st["employmentstatus"].astype(str).str.strip()
    st["_role_norm"] = st["role"].map(_norm_token)
    st["_estatus_norm"] = st["employmentstatus"].map(_norm_token)
    st = _parse_date_cols(st, ["statusstartdate","statusenddate"], default_end_cols=["statusenddate"])
    return st

# =========================
# Core: Interim / Final
# =========================
def build_interim(emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, year=None, pay_deductions=None) -> pd.DataFrame:
    if year is None: year = choose_report_year(emp_elig)
    employee_ids = _collect_employee_ids(emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll)
    grid = _grid_for_year(employee_ids, year)

    # Demographic minimal fields (names)
    demo_names = pd.DataFrame(columns=["employeeid","firstname","lastname"])
    if not emp_demo.empty:
        tmp = emp_demo.copy()
        if "employeeid" in tmp.columns:
            tmp["employeeid"] = tmp["employeeid"].map(_normalize_employeeid)
        for col in ["firstname","lastname"]:
            if col not in tmp.columns: tmp[col] = ""
        demo_names = tmp[["employeeid","firstname","lastname"]].drop_duplicates("employeeid", keep="first")

    out = grid.merge(demo_names, on="employeeid", how="left")

    # Unified status table: prefer dedicated 'emp status'; else derive from demographic
    stt = emp_status.copy()
    if (stt is None) or stt.empty or not {"statusstartdate","statusenddate"} <= set(stt.columns):
        stt = _status_from_demographic(emp_demo)
    else:
        if "employeeid" in stt.columns:
            stt["employeeid"] = stt["employeeid"].map(_normalize_employeeid)
        if "employmentstatus" in stt.columns:
            stt["_estatus_norm"] = stt["employmentstatus"].astype(str).map(_norm_token)
        if "role" in stt.columns:
            stt["_role_norm"] = stt["role"].astype(str).map(_norm_token)
        stt = _parse_date_cols(stt, ["statusstartdate","statusenddate"], default_end_cols=["statusenddate"])

    elg, enr, dep = emp_elig.copy(), emp_enroll.copy(), dep_enroll.copy()
    pay = pay_deductions.copy() if pay_deductions is not None else pd.DataFrame()

    for df in (elg,enr,dep,pay):
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

        # Ensure normalized tokens for this employee status slice
        if (not st_emp.empty):
            if "_estatus_norm" not in st_emp.columns and "employmentstatus" in st_emp.columns:
                st_emp = st_emp.copy(); st_emp["_estatus_norm"] = st_emp["employmentstatus"].map(_norm_token)
            if "_role_norm" not in st_emp.columns and "role" in st_emp.columns:
                st_emp = st_emp.copy(); st_emp["_role_norm"] = st_emp["role"].map(_norm_token)

        # ----- EMPLOYED -----
        employed=False
        if not st_emp.empty and {"statusstartdate","statusenddate"} <= set(st_emp.columns):
            active_mask = pd.Series(False, index=st_emp.index)
            if "_estatus_norm" in st_emp.columns:
                active_mask = active_mask | st_emp["_estatus_norm"].isin(EMPLOYED_TOKENS)
            if "_role_norm" in st_emp.columns:
                active_mask = active_mask | st_emp["_role_norm"].isin(FT_TOKENS | PT_TOKENS)
            employed = _any_overlap(st_emp, "statusstartdate","statusenddate", ms,me, mask=active_mask)

        # ----- FT/PT FULL-MONTH (from Role/EmploymentStatus)
        ft_full_month = False
        pt_full_month = False
        if not st_emp.empty and {"statusstartdate","statusenddate"} <= set(st_emp.columns):
            ft_mask = pd.Series(False, index=st_emp.index)
            pt_mask = pd.Series(False, index=st_emp.index)
            if "_role_norm" in st_emp.columns:
                ft_mask = ft_mask | st_emp["_role_norm"].isin(FT_TOKENS)
                pt_mask = pt_mask | st_emp["_role_norm"].isin(PT_TOKENS)
            if "_estatus_norm" in st_emp.columns:
                ft_mask = ft_mask | st_emp["_estatus_norm"].isin(FT_TOKENS)
                pt_mask = pt_mask | st_emp["_estatus_norm"].isin(PT_TOKENS)
            ft_full_month = _all_month(st_emp, "statusstartdate","statusenddate", ms,me, mask=ft_mask)
            pt_full_month = (not ft_full_month) and _all_month(st_emp, "statusstartdate","statusenddate", ms,me, mask=pt_mask)

        # ----- ELIGIBILITY (employee)
        eligible_any=False; eligible_allmonth=False; eligible_mv_full=False
        if not el_emp.empty and {"eligibilitystartdate","eligibilityenddate"} <= set(el_emp.columns):
            eligible_any = _any_overlap(el_emp, "eligibilitystartdate","eligibilityenddate", ms,me)
            eligible_allmonth = _all_month(el_emp, "eligibilitystartdate","eligibilityenddate", ms,me)
            if "minimumvaluecoverage" in el_emp.columns:
                mv_mask = el_emp["minimumvaluecoverage"].fillna(False).astype(bool)
                eligible_mv_full = _all_month(el_emp, "eligibilitystartdate","eligibilityenddate", ms,me, mask=mv_mask)

        # ----- ENROLLMENT (employee)
        enrolled_any=False; enrolled_allmonth=False
        if not en_emp.empty and {"enrollmentstartdate","enrollmentenddate"} <= set(en_emp.columns):
            en_mask = en_emp["isenrolled"].fillna(True) if "isenrolled" in en_emp.columns else pd.Series(True,index=en_emp.index)
            enrolled_any = _any_overlap(en_emp, "enrollmentstartdate","enrollmentenddate", ms,me, mask=en_mask)
            enrolled_allmonth = _all_month(en_emp, "enrollmentstartdate","enrollmentenddate", ms,me, mask=en_mask)

        # ----- DEPENDENTS (FULL-MONTH OFFERS for Line 14) -----
        offer_spouse_full=False; offer_child_full=False
        if not de_emp.empty and {"dependentrelationship","eligiblestartdate","eligibleenddate"} <= set(de_emp.columns):
            offer_spouse_full = _all_month(de_emp, "eligiblestartdate","eligibleenddate", ms,me, mask=de_emp["dependentrelationship"].eq("Spouse"))
            offer_child_full  = _all_month(de_emp, "eligiblestartdate","eligibleenddate", ms,me, mask=de_emp["dependentrelationship"].eq("Child"))

        # ----- NEW: DEPENDENTS (ANY-OVERLAP) → analytics columns you asked -----
        spouse_eligible=False; child_eligible=False
        spouse_enrolled=False; child_enrolled=False
        if not de_emp.empty and "dependentrelationship" in de_emp.columns:
            rel_l = de_emp["dependentrelationship"].astype(str).str.lower()
            dep_es = "eligiblestartdate" if "eligiblestartdate" in de_emp.columns else None
            dep_ee = "eligibleenddate"   if "eligibleenddate"   in de_emp.columns else None
            dep_ns = "enrollmentstartdate" if "enrollmentstartdate" in de_emp.columns else None
            dep_ne = "enrollmentenddate"   if "enrollmentenddate"   in de_emp.columns else None
            # Eligibility (prefer explicit eligibility window; else fall back to enrollment window as proxy)
            if dep_es and dep_ee:
                sp_elig_rows = de_emp[(rel_l.str.contains("spouse", na=False)) & (de_emp[dep_es] <= pd.Timestamp(me)) & (de_emp[dep_ee] >= pd.Timestamp(ms))]
                ch_elig_rows = de_emp[(rel_l.str.contains("child",  na=False)) & (de_emp[dep_es] <= pd.Timestamp(me)) & (de_emp[dep_ee] >= pd.Timestamp(ms))]
                spouse_eligible = not sp_elig_rows.empty
                child_eligible  = not ch_elig_rows.empty
            elif dep_ns and dep_ne:
                sp_elig_rows = de_emp[(rel_l.str.contains("spouse", na=False)) & (de_emp[dep_ns] <= pd.Timestamp(me)) & (de_emp[dep_ne] >= pd.Timestamp(ms))]
                ch_elig_rows = de_emp[(rel_l.str.contains("child",  na=False)) & (de_emp[dep_ns] <= pd.Timestamp(me)) & (de_emp[dep_ne] >= pd.Timestamp(ms))]
                spouse_eligible = not sp_elig_rows.empty
                child_eligible  = not ch_elig_rows.empty
            else:
                spouse_eligible = bool((rel_l.str.contains("spouse", na=False)).any())
                child_eligible  = bool((rel_l.str.contains("child",  na=False)).any())
            # Enrollment (any overlap) and NOT a waiver
            if dep_ns and dep_ne:
                enr_sp = de_emp[(rel_l.str.contains("spouse", na=False)) & (de_emp[dep_ns] <= pd.Timestamp(me)) & (de_emp[dep_ne] >= pd.Timestamp(ms))]
                enr_ch = de_emp[(rel_l.str.contains("child",  na=False)) & (de_emp[dep_ns] <= pd.Timestamp(me)) & (de_emp[dep_ne] >= pd.Timestamp(ms))]
                if "plancode" in de_emp.columns:
                    enr_sp = enr_sp[~enr_sp["plancode"].astype(str).str.strip().str.lower().eq("waive")]
                    enr_ch = enr_ch[~enr_ch["plancode"].astype(str).str.strip().str.lower().eq("waive")]
                spouse_enrolled = not enr_sp.empty
                child_enrolled  = not enr_ch.empty

        # Waiting period proxy
        waitingperiod_month = bool(employed and ft_full_month and not eligible_any)

        # Employee monthly contribution (for 1A vs 1E)
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
            "parttime": pt_full_month,
            "eligibleforcoverage": eligible_any,
            "eligible_allmonth": eligible_allmonth,
            "eligible_mv": eligible_mv_full,
            "offer_ee_allmonth": offer_ee_full,
            "enrolled_allmonth": enrolled_allmonth,
            "offer_spouse": offer_spouse_full,     # full-month offer
            "offer_dependents": offer_child_full,  # full-month offer
            # NEW analytics columns (any-overlap):
            "spouse_eligible": spouse_eligible,
            "child_eligible":  child_eligible,
            "spouse_enrolled": spouse_enrolled,
            "child_enrolled":  child_enrolled,
            "waitingperiod_month": waitingperiod_month,
            "line14_final": l14,
            "line16_final": l16,
        })

    interim = pd.concat([out.reset_index(drop=True), pd.DataFrame(flags)], axis=1)
    base_cols = ["employeeid","firstname","lastname","year","monthnum","month","monthstart","monthend"]
    flag_cols = [
        "employed","ft","parttime",
        "eligibleforcoverage","eligible_allmonth","eligible_mv","offer_ee_allmonth",
        "enrolled_allmonth","offer_spouse","offer_dependents",
        "spouse_eligible","child_eligible","spouse_enrolled","child_enrolled",
        "waitingperiod_month","line14_final","line16_final"
    ]
    keep = [c for c in base_cols if c in interim.columns] + [c for c in flag_cols if c in interim.columns]
    interim = interim[keep]

    # Safety: ensure exactly one row per EmployeeID × Year × Month
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
# Penalty Dashboard
# =========================
PENALTY_A = 241.67  # “No MEC offered” (1H)
PENALTY_B = 362.50  # “Waived unaffordable coverage” (1E & not enrolled)

_PENALTY_TEXT_A = (
    "Penalty A: No MEC offered <br/> "
    "The employee was not offered minimum essential coverage (MEC) during the months in which the penalty was incurred."
)
_PENALTY_TEXT_B = (
    "Penalty B: Waived Unaffordable Coverage <br/> "
    "The employee was offered minimum essential coverage (MEC), but the lowest-cost option for employee-only coverage "
    "was not affordable, meaning it cost more than the $50 threshold. The employee chose to waive this unaffordable coverage."
)

def _money(x: float | None) -> str:
    return "-" if (x is None or x == 0) else f"${x:,.2f}"

def build_penalty_dashboard(interim: pd.DataFrame,
                            penalty_a: float = PENALTY_A,
                            penalty_b: float = PENALTY_B) -> pd.DataFrame:
    """
    Returns a wide table: EmployeeID, Reason, January..December
    Rules:
      - Penalty A: Line14 == '1H'
      - Penalty B: Line14 == '1E' AND NOT enrolled_allmonth (waived unaffordable)
      - Adds wait-period explanation for A months where waitingperiod_month == True
    """
    if interim.empty:
        return pd.DataFrame(columns=["EmployeeID","Reason"] + FULL_MONTHS)

    df = interim.copy()
    df["EmployeeID"] = df["employeeid"]
    df["MonthFull"] = df["monthnum"].map(MONTHNUM_TO_FULL)

    cond_A = df["line14_final"].eq("1H")
    cond_B = df["line14_final"].eq("1E") & (~df["enrolled_allmonth"].fillna(False))

    # choose penalty per row (B precedence)
    df["_pen_amt"] = 0.0
    df.loc[cond_A, "_pen_amt"] = penalty_a
    df.loc[cond_B, "_pen_amt"] = penalty_b

    df["_pen_type"] = ""
    df.loc[cond_A, "_pen_type"] = "A"
    df.loc[cond_B, "_pen_type"] = "B"  # overrides A when both match

    rows=[]
    for emp, g in df.groupby("EmployeeID", sort=True):
        months_map = {m:"-" for m in FULL_MONTHS}
        for _, r in g.iterrows():
            months_map[r["MonthFull"]] = _money(float(r["_pen_amt"])) if r["_pen_amt"] else "-"
        has_B = (g["_pen_type"]=="B").any()
        has_A = (g["_pen_type"]=="A").any()
        reason = _PENALTY_TEXT_B if has_B else (_PENALTY_TEXT_A if has_A else "")
        if has_A:
            wait_months = g.loc[g["waitingperiod_month"] & (g["_pen_type"]=="A"), "MonthFull"].tolist()
            if wait_months:
                wait_list = ", ".join(wait_months)
                reason += f"<br/><br/>Employee was not eligible for coverage in {wait_list} because they were in their wait period during those month(s)."
        row = {"EmployeeID": emp, "Reason": reason}
        row.update({m: months_map[m] for m in FULL_MONTHS})
        rows.append(row)

    out = pd.DataFrame(rows, columns=["EmployeeID","Reason"] + FULL_MONTHS)
    return out

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

# =========================
# Excel save
# =========================
def save_excel_outputs(interim: pd.DataFrame, final: pd.DataFrame, year:int, penalty_dashboard: pd.DataFrame | None = None) -> bytes:
    """
    Writes 'Final {year}', 'Interim {year}', and (if provided) 'Penalty Dashboard {year}'.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as xw:
        final.to_excel(xw, index=False, sheet_name=f"Final {year}")
        interim.to_excel(xw, index=False, sheet_name=f"Interim {year}")
        if penalty_dashboard is not None and not penalty_dashboard.empty:
            penalty_dashboard.to_excel(xw, index=False, sheet_name=f"Penalty Dashboard {year}")
    buf.seek(0)
    return buf.getvalue()
