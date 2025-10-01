# aca_core.py
# Core logic for ACA-1095 processing:
#  - Read Excel bytes -> DataFrames
#  - Normalize / prepare inputs
#  - Build Interim (monthly grid)
#  - Build Final (per-employee Jan..Dec)
#  - Write Excel outputs (Final + Interim)
#  - Fill a single employee PDF (editable+flattened) using PyPDF2 + ReportLab

from __future__ import annotations

import io
from dataclasses import dataclass
import calendar
from datetime import date
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# =========================
# Constants & helpers
# =========================

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_TO_NUM = {m: i+1 for i, m in enumerate(MONTHS)}

TRUTHY = {"y","yes","true","t","1",1,True}
FALSY  = {"n","no","false","f","0",0,False,None,np.nan}

# IRS 1095 Line14 canonical set (simplified; extend as needed)
VALID_L14 = {"1A","1B","1C","1D","1E","1F","1H"}
VALID_L16 = {"2A","2B","2C","2D","2E","2F","2G","2H"}

# Column alias maps (lowercased)
ALIASES = {
    # demographic
    "employeeid": {"employee id","empid","id","employee_id","employee-id"},
    "firstname": {"first","first_name","first name","givenname"},
    "lastname": {"last","last_name","last name","surname","familyname"},
    "ssn": {"ssn","social","socialsecuritynumber","social security number"},
    "addressline1": {"address1","address line 1","address line1","addr1","addressline1"},
    "addressline2": {"address2","address line 2","address line2","addr2","addressline2"},
    "city": {"city","town"},
    "state": {"state","statecode","province"},
    "zipcode": {"zip","zip code","postalcode","postcode","zipcode"},

    # status
    "employmentstatus": {"employmentstatus","empstatus","status"},
    "role": {"role","jobtype","job type"},
    "statusstartdate": {"statusstartdate","start","startdate","status start","status start date"},
    "statusenddate": {"statusenddate","end","enddate","status end","status end date"},

    # eligibility
    "iseligibleforcoverage": {"iseligibleforcoverage","eligible","eligibility"},
    "minimumvaluecoverage": {"minimumvaluecoverage","mv","min value","minimum value"},
    "eligibilitystartdate": {"eligibilitystartdate","eligiblestartdate","elig start","elig startdate"},
    "eligibilityenddate": {"eligibilityenddate","eligibleenddate","elig end","elig enddate"},

    # enrollment
    "isenrolled": {"isenrolled","enrolled"},
    "enrollmentstartdate": {"enrollmentstartdate","enrollstartdate","enroll start"},
    "enrollmentenddate": {"enrollmentenddate","enrollenddate","enroll end"},

    # dependents
    "dependentrelationship": {"dependentrelationship","relationship"},
    "eligible": {"eligible"},
    "enrolled": {"enrolled"},
    "eligiblestartdate": {"eligiblestartdate"},
    "eligibleenddate": {"eligibleenddate"},
}

# =========================
# Basic utilities
# =========================

def _lower(s: str) -> str:
    return str(s).strip().lower()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols = {c: _lower(c) for c in df.columns}
    df = df.rename(columns=cols).copy()
    # apply aliasing
    used = set()
    for canon, candidates in ALIASES.items():
        for c in list(df.columns):
            if c == canon:
                used.add(canon)
                break
            if c in candidates and canon not in df.columns:
                df = df.rename(columns={c: canon})
                used.add(canon)
                break
    return df

def parse_date_safe(v) -> Optional[date]:
    if pd.isna(v) or v in ("", None):
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%Y":
                return date(dt.year, 1, 1)
            if fmt == "%Y-%m":
                return date(dt.year, dt.month, 1)
            return dt.date()
        except Exception:
            continue
    # Excel float date?
    try:
        return pd.to_datetime(v).date()
    except Exception:
        return None



def month_bounds(year: int, month_num: int):
    """Return (first_day, last_day) as datetime.date objects for the given month."""
    start = date(year, month_num, 1)
    last_day = calendar.monthrange(year, month_num)[1]
    end = date(year, month_num, last_day)
    return start, end


def to_bool(x) -> bool:
    if isinstance(x, str):
        x = x.strip().lower()
    return x in TRUTHY

def _overlaps(a_start: Optional[date], a_end: Optional[date], b_start: date, b_end: date) -> bool:
    if a_start is None and a_end is None:
        return True
    if a_start is None:
        return b_start <= a_end
    if a_end is None:
        return a_start <= b_end
    return not (a_end < b_start or a_start > b_end)

def _any_overlap(df: pd.DataFrame, start_col: str, end_col: str, ms: date, me: date, mask: Optional[pd.Series]=None) -> bool:
    if df is None or df.empty:
        return False
    s = df[start_col].apply(parse_date_safe)
    e = df[end_col].apply(parse_date_safe)
    ok = s.combine(e, lambda a,b: _overlaps(a,b,ms,me))
    if mask is not None:
        ok = ok & mask
    return bool(ok.any())

# =========================
# FT/PT/Active normalization helpers (fix for your screenshot)
# =========================

def _norm_txt_ser(s: pd.Series, default_val: str = "") -> pd.Series:
    """Uppercase, remove spaces/dashes, coerce to string, fill empties."""
    if s is None:
        return pd.Series(default_val)
    return (
        s.astype(str)
         .str.upper()
         .str.replace(r"[\s\-]+", "", regex=True)
         .fillna(default_val)
    )

# Recognized tokens
FT_TOKENS = {"FT", "FULLTIME", "FTE", "CATEGORY2"}   # Category2 => full-time (your rule)
PT_TOKENS = {"PT", "PARTTIME"}
ACTIVE_TOKENS = {"ACTIVE"} | FT_TOKENS | PT_TOKENS

def _status_masks(df: pd.DataFrame):
    """
    Build boolean masks over the 'emp status' rows indicating Active/FT/PT.
    Works whether the info is in 'role' or 'employmentstatus' (or both).
    Returns: (is_active_mask, is_ft_mask, is_pt_mask)
    """
    if df is None or df.empty:
        idx = pd.RangeIndex(0)
        return (
            pd.Series(False, index=idx),
            pd.Series(False, index=idx),
            pd.Series(False, index=idx),
        )

    roleN  = _norm_txt_ser(df.get("role", pd.Series("", index=df.index)))
    eStatN = _norm_txt_ser(df.get("employmentstatus", pd.Series("", index=df.index)))

    is_ft = roleN.isin(FT_TOKENS) | eStatN.isin(FT_TOKENS)
    is_pt = roleN.isin(PT_TOKENS) | eStatN.isin(PT_TOKENS)
    is_active = eStatN.isin(ACTIVE_TOKENS) | is_ft | is_pt

    # If neither column exists, treat rows as potentially active (date overlap still gates them)
    if ("role" not in df.columns) and ("employmentstatus" not in df.columns):
        is_active = pd.Series(True, index=df.index)

    return is_active, is_ft, is_pt

# =========================
# Loading & preparation
# =========================

def load_excel(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """Read Excel bytes to raw lowercased, normalized DataFrames keyed by sheet name."""
    with io.BytesIO(file_bytes) as bio:
        xls = pd.ExcelFile(bio)
        out = {}
        for name in xls.sheet_names:
            df = xls.parse(name)
            out[_lower(name)] = normalize_columns(df)
        return out

def _pick(data: Dict[str, pd.DataFrame], keys: Iterable[str]) -> Optional[pd.DataFrame]:
    for k in keys:
        if k in data and isinstance(data[k], pd.DataFrame):
            return data[k]
    # allow contains
    for k in data:
        if any(kk in k for kk in keys):
            return data[k]
    return pd.DataFrame()

def prepare_inputs(data: Dict[str, pd.DataFrame]):
    """Return canonical tables (demographic, status, eligibility, enrollment, dep_enrollment, pay_deductions)."""
    emp_demo   = _pick(data, ("emp demographic","demographic","employee demographic","emp_demo","empdemographic"))
    emp_status = _pick(data, ("emp status","status","employment status","empstatus"))
    emp_elig   = _pick(data, ("emp eligibility","eligibility","empeligibility"))
    emp_enroll = _pick(data, ("emp enrollment","enrollment","empenrollment"))
    dep_enroll = _pick(data, ("dep enrollment","dependents","dependent enrollment","depenrollment"))
    pay_ded    = _pick(data, ("pay deductions","deductions","paydeductions"))

    # Ensure key columns exist even if empty
    for df, must in (
        (emp_demo,   ["employeeid","firstname","lastname","ssn","addressline1","addressline2","city","state","zipcode"]),
        (emp_status, ["employeeid","employmentstatus","role","statusstartdate","statusenddate"]),
        (emp_elig,   ["employeeid","iseligibleforcoverage","minimumvaluecoverage","eligibilitystartdate","eligibilityenddate"]),
        (emp_enroll, ["employeeid","isenrolled","enrollmentstartdate","enrollmentenddate"]),
        (dep_enroll, ["employeeid","dependentrelationship","eligible","enrolled","eligiblestartdate","eligibleenddate"]),
    ):
        for c in must:
            if c not in df.columns:
                df[c] = np.nan

    # Make sure employeeid is string-compatible for joins/filters
    for df in (emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, pay_ded):
        if "employeeid" in df.columns:
            df["employeeid"] = df["employeeid"].astype(str)

    # Normalize dates to date objects (not absolutely required, guarded later)
    def _norm_dates(df, cols):
        for c in cols:
            if c in df.columns:
                df[c] = df[c].apply(parse_date_safe)

    _norm_dates(emp_status, ["statusstartdate","statusenddate"])
    _norm_dates(emp_elig,   ["eligibilitystartdate","eligibilityenddate"])
    _norm_dates(emp_enroll, ["enrollmentstartdate","enrollmentenddate"])
    _norm_dates(dep_enroll, ["eligiblestartdate","eligibleenddate"])

    return emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, pay_ded

# =========================
# Interim & Final builders
# =========================

def _infer_report_year(emp_status: pd.DataFrame, emp_elig: pd.DataFrame, emp_enroll: pd.DataFrame) -> int:
    years: List[int] = []
    for df, sc, ec in (
        (emp_status, "statusstartdate", "statusenddate"),
        (emp_elig, "eligibilitystartdate","eligibilityenddate"),
        (emp_enroll, "enrollmentstartdate","enrollmentenddate"),
    ):
        if sc in df.columns:
            years += [d.year for d in df[sc].dropna()]
        if ec in df.columns:
            years += [d.year for d in df[ec].dropna()]
    if not years:
        return datetime.utcnow().year
    # pick the most common or latest
    try:
        return int(pd.Series(years).mode().iloc[0])
    except Exception:
        return int(max(years))

def build_interim(
    emp_demo: pd.DataFrame,
    emp_status: pd.DataFrame,
    emp_elig: pd.DataFrame,
    emp_enroll: pd.DataFrame,
    dep_enroll: pd.DataFrame,
    year: Optional[int] = None,
) -> pd.DataFrame:
    """Return monthly rows per employee with basic flags and simplified L14/L16/L15."""
    y = int(year) if year else _infer_report_year(emp_status, emp_elig, emp_enroll)

    # Determine employee list from demographics (fallback to status)
    if emp_demo is None or emp_demo.empty:
        base_ids = sorted(set(emp_status["employeeid"].dropna().astype(str)))
        demo_cols = ["employeeid","firstname","lastname","ssn","addressline1","addressline2","city","state","zipcode"]
        emp_demo = pd.DataFrame({c: [] for c in demo_cols})
        if base_ids:
            emp_demo = pd.DataFrame({"employeeid": base_ids})
    else:
        emp_demo = emp_demo.copy()

    rows = []
    for _, drow in emp_demo.iterrows():
        emp = str(drow.get("employeeid", ""))
        if not emp:
            continue
        # slices
        st_emp = emp_status[emp_status["employeeid"].astype(str) == emp] if not emp_status.empty else pd.DataFrame()
        el_emp = emp_elig[emp_elig["employeeid"].astype(str) == emp] if not emp_elig.empty else pd.DataFrame()
        en_emp = emp_enroll[emp_enroll["employeeid"].astype(str) == emp] if not emp_enroll.empty else pd.DataFrame()

        for m_idx, m in enumerate(MONTHS, start=1):
            ms, me = month_bounds(y, m_idx)

            # Employment / FT / PT using robust token recognition
            employed = ft = parttime = False
            if not st_emp.empty and {"statusstartdate","statusenddate"} <= set(st_emp.columns):
                active_mask, ft_mask, pt_mask = _status_masks(st_emp)
                employed = _any_overlap(st_emp, "statusstartdate","statusenddate", ms, me, mask=active_mask)
                ft = _any_overlap(st_emp, "statusstartdate","statusenddate", ms, me, mask=ft_mask)
                parttime = _any_overlap(st_emp, "statusstartdate","statusenddate", ms, me, mask=pt_mask) and not ft

            # Eligibility flags
            eligible = False
            eligible_allmonth = False
            eligible_mv = False
            if not el_emp.empty and {"eligibilitystartdate","eligibilityenddate"}.issubset(el_emp.columns):
                eligible = _any_overlap(el_emp, "eligibilitystartdate","eligibilityenddate", ms, me)
                # minimum value coverage ever?
                if "minimumvaluecoverage" in el_emp.columns:
                    mv = el_emp["minimumvaluecoverage"].apply(to_bool)
                    eligible_mv = bool(mv.any() and eligible)
                # all month overlap (crude): require start <= ms and end >= me on any row
                s = el_emp["eligibilitystartdate"].apply(parse_date_safe)
                e = el_emp["eligibilityenddate"].apply(parse_date_safe)
                allm = ((s.isna() | (s <= ms)) & (e.isna() | (e >= me))).any()
                eligible_allmonth = bool(allm and eligible)

            # Enrollment flags
            enrolled_allmonth = False
            if not en_emp.empty and {"enrollmentstartdate","enrollmentenddate"}.issubset(en_emp.columns):
                s = en_emp["enrollmentstartdate"].apply(parse_date_safe)
                e = en_emp["enrollmentenddate"].apply(parse_date_safe)
                allm = ((s.isna() | (s <= ms)) & (e.isna() | (e >= me))).any()
                if "isenrolled" in en_emp.columns:
                    any_en = en_emp["isenrolled"].apply(to_bool).any()
                    enrolled_allmonth = bool(allm and any_en)
                else:
                    enrolled_allmonth = bool(allm)

            # Simplified offer flags (placeholders; adapt to your exact logic)
            offer_ee_allmonth = eligible_allmonth  # often 1E/1C logic derives from eligibility + dependents
            offer_spouse = False
            offer_dependents = False

            # Waiting period (placeholder): eligible but not all-month => waiting
            waitingperiod_month = bool(eligible and not eligible_allmonth)

            # Line 14/16 simplified mapping (extend/replace with your strict rules)
            if not eligible:
                line14 = "1H"  # no offer
                line16 = "2A" if not employed else ("2B" if not ft else "")
            else:
                # offered to EE; dependents/spouse not granular here
                line14 = "1E" if eligible_mv else "1C"
                if enrolled_allmonth:
                    line16 = "2C"
                elif waitingperiod_month:
                    line16 = "2D"
                elif not ft:
                    line16 = "2B"
                else:
                    line16 = ""

            # Line 15 (employee required contribution) — if you want to compute from pay_deductions,
            # inject that logic here (this core version leaves it empty)
            line15_amount = np.nan

            rows.append({
                "employeeid": emp,
                "firstname": drow.get("firstname",""),
                "lastname": drow.get("lastname",""),
                "year": y,
                "monthnum": m_idx,
                "month": f"{m_idx} {m}",
                "monthstart": ms,
                "monthend": me,

                # flags
                "employed": employed,
                "ft": ft,
                "parttime": parttime,

                "eligibleforcoverage": eligible,
                "eligible_allmonth": eligible_allmonth,
                "eligible_mv": eligible_mv,

                "offer_ee_allmonth": offer_ee_allmonth,
                "offer_spouse": offer_spouse,
                "offer_dependents": offer_dependents,

                "enrolled_allmonth": enrolled_allmonth,
                "waitingperiod_month": waitingperiod_month,

                # IRS lines (per month)
                "line14_final": line14,
                "line15_amount": line15_amount,
                "line16_final": line16,
            })

    interim = pd.DataFrame(rows)

    # Order columns
    base_cols = [
        "employeeid","firstname","lastname","year","monthnum","month","monthstart","monthend",
        "employed","ft","parttime",
        "eligibleforcoverage","eligible_allmonth","eligible_mv",
        "offer_ee_allmonth","enrolled_allmonth","offer_spouse","offer_dependents","waitingperiod_month",
        "line14_final","line15_amount","line16_final",
    ]
    # If any optional columns are missing (when emp_demo lacked names), ensure they exist
    for c in base_cols:
        if c not in interim.columns:
            interim[c] = np.nan
    interim = interim[base_cols].sort_values(["employeeid","year","monthnum"], kind="mergesort").reset_index(drop=True)
    return interim

def _pivot_lines(interim: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """Pivot per-month values into Jan..Dec columns named f'{prefix}_Jan'.."""
    tmp = interim[["employeeid","firstname","lastname","year","monthnum",col]].copy()
    tmp["mon"] = tmp["monthnum"].map(lambda i: MONTHS[i-1])
    pvt = tmp.pivot_table(index=["employeeid","firstname","lastname","year"], columns="mon", values=col, aggfunc="first")
    pvt = pvt.reindex(columns=MONTHS)  # ensure Jan..Dec order
    pvt.columns = [f"{prefix}_{m}" for m in pvt.columns]
    pvt = pvt.reset_index()
    return pvt

def build_final(interim: pd.DataFrame) -> pd.DataFrame:
    """Compress Interim monthly rows into single row per employee with Jan..Dec columns for Lines."""
    if interim is None or interim.empty:
        return pd.DataFrame(columns=["employeeid","firstname","lastname","year"] +
                            [f"Line14_{m}" for m in MONTHS] +
                            [f"Line15_{m}" for m in MONTHS] +
                            [f"Line16_{m}" for m in MONTHS])

    p14 = _pivot_lines(interim, "line14_final", "Line14")
    p15 = _pivot_lines(interim, "line15_amount", "Line15")
    p16 = _pivot_lines(interim, "line16_final", "Line16")

    final = p14.merge(p15, on=["employeeid","firstname","lastname","year"], how="outer") \
               .merge(p16, on=["employeeid","firstname","lastname","year"], how="outer")

    # Order columns nicely
    ordered = ["employeeid","firstname","lastname","year"]
    ordered += [f"Line14_{m}" for m in MONTHS]
    ordered += [f"Line15_{m}" for m in MONTHS]
    ordered += [f"Line16_{m}" for m in MONTHS]
    final = final.reindex(columns=ordered)
    return final

# =========================
# Writers
# =========================

def save_excel_outputs(interim: pd.DataFrame, final: pd.DataFrame, year: int) -> bytes:
    """Return bytes of an Excel workbook with Final & Interim sheets."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as xw:
        final.to_excel(xw, index=False, sheet_name=f"Final {year}")
        interim.to_excel(xw, index=False, sheet_name=f"Interim {year}")
    buf.seek(0)
    return buf.getvalue()

# =========================
# PDF filling (simple, field-agnostic burner)
# =========================

def _overlay_text(page_width, page_height, draw_ops: List[Tuple[float,float,str]]) -> bytes:
    """Create a PDF overlay with (x, y, text) draw operations from bottom-left origin."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_width, page_height))
    for x, y, txt in draw_ops:
        c.drawString(x, y, txt)
    c.save()
    buf.seek(0)
    return buf.getvalue()

def _flatten_pdf(reader: PdfReader, writer: PdfWriter) -> bytes:
    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.getvalue()

def fill_pdf_for_employee(
    pdf_bytes: bytes,
    emp_row: pd.Series,
    final_df_emp: pd.DataFrame,
    year_used: int,
) -> Tuple[str, io.BytesIO, str, io.BytesIO]:
    """
    Returns:
      (editable_filename, editable_bytes, flattened_filename, flattened_bytes)
    This simplified burner writes a few Part I fields and Jan..Dec Lines as plain text overlay.
    You can replace coordinates with your exact 1095-C form mapping.
    """
    # Read template
    src = PdfReader(io.BytesIO(pdf_bytes))
    writer_edit = PdfWriter()
    writer_flat = PdfWriter()

    # Basic page metrics
    page0 = src.pages[0]
    mediabox = page0.mediabox
    width = float(mediabox.width)
    height = float(mediabox.height)

    # Build text to draw (placeholders/coordinates need your real mapping)
    first = str(emp_row.get("firstname","")).strip()
    last  = str(emp_row.get("lastname","")).strip()
    ssn   = str(emp_row.get("ssn","")).strip()
    addr1 = str(emp_row.get("addressline1","")).strip()
    addr2 = str(emp_row.get("addressline2","")).strip()
    city  = str(emp_row.get("city","")).strip()
    state = str(emp_row.get("state","")).strip()
    zipc  = str(emp_row.get("zipcode","")).strip()

    # Simple positions (from bottom-left) – you should calibrate on your actual PDF
    ops = [
        (72, height-90, f"{last}, {first}"),
        (72, height-110, f"SSN: {ssn}"),
        (72, height-130, addr1),
        (72, height-150, addr2),
        (72, height-170, f"{city}, {state} {zipc}"),
        (width-200, height-90, f"Tax Year: {year_used}"),
    ]

    # Add monthly Line14/15/16 (first row only from final_df_emp)
    if not final_df_emp.empty:
        row = final_df_emp.iloc[0]
        base_y = height - 220
        dy = 12
        for i, m in enumerate(MONTHS):
            l14 = str(row.get(f"Line14_{m}", "") or "")
            l15 = row.get(f"Line15_{m}", "")
            l16 = str(row.get(f"Line16_{m}", "") or "")
            ops.append((72, base_y - i*dy, f"{m}: L14={l14}  L15={'' if pd.isna(l15) else l15}  L16={l16}"))

    # Build overlay and merge onto page 1
    overlay_bytes = _overlay_text(width, height, ops)
    overlay_reader = PdfReader(io.BytesIO(overlay_bytes))
    merged = page0
    merged.merge_page(overlay_reader.pages[0])
    writer_edit.add_page(merged)

    # Append remaining pages as-is (if any)
    for i in range(1, len(src.pages)):
        writer_edit.add_page(src.pages[i])

    # Editable stream
    editable_bytes = io.BytesIO()
    writer_edit.write(editable_bytes)
    editable_bytes.seek(0)

    # Flattened (by writing again, effectively baking the overlay)
    flat_reader = PdfReader(io.BytesIO(editable_bytes.getvalue()))
    for p in flat_reader.pages:
        writer_flat.add_page(p)
    flattened_bytes = io.BytesIO()
    writer_flat.write(flattened_bytes)
    flattened_bytes.seek(0)

    emp_name = f"{last}_{first}".strip("_") or "employee"
    editable_name = f"1095c_{emp_name}_{year_used}_editable.pdf"
    flat_name     = f"1095c_{emp_name}_{year_used}_flat.pdf"
    return editable_name, editable_bytes, flat_name, flattened_bytes
