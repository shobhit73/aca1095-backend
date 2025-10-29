# aca_builder.py
from __future__ import annotations

import io
import calendar
from datetime import date, timedelta
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd

# -----------------------
# Robust logging imports (graceful fallbacks)
# -----------------------
try:
    from debug_logging import get_logger  # type: ignore
except Exception:  # pragma: no cover
    def get_logger(name: str):
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

try:
    from debug_logging import log_time  # type: ignore
except Exception:  # pragma: no cover
    from contextlib import contextmanager
    @contextmanager
    def log_time(_logger, _msg: str):
        yield

try:
    from debug_logging import log_df  # type: ignore
except Exception:  # pragma: no cover
    def log_df(_logger, _name: str, _df: pd.DataFrame):
        pass

log = get_logger("aca_builder")

AFFORDABILITY_THRESHOLD_DEFAULT = 50.0  # UAT default
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# -----------------------
# Helpers
# -----------------------
def month_bounds(year: int, m: int) -> tuple[date, date]:
    last = calendar.monthrange(year, m)[1]
    return date(year, m, 1), date(year, m, last)

def _df_or_empty(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if isinstance(df, pd.DataFrame) and (col in df.columns):
        s = df[col]
        return s if isinstance(s, pd.Series) else pd.Series([s] * len(df), index=df.index)
    return pd.Series(index=(df.index if isinstance(df, pd.DataFrame) else None), dtype="object")

def _series_str_upper_strip(s: pd.Series) -> pd.Series:
    if not isinstance(s, pd.Series):
        s = pd.Series([], dtype="object")
    try:
        return s.astype(str).str.upper().str.strip()
    except Exception:
        return s.map(lambda x: ("" if pd.isna(x) else str(x).upper().strip()))

ALIASES = {
    "employeeid": {"employee id", "empid", "emp id", "id"},
    "name": {"employee name", "empname", "name"},
    "employmentstatus": {"status", "empstatus", "estatus", "employment status"},
    "role": {"employeerole", "emp role"},
    "statusstartdate": {"employmentstatusstartdate", "estatusstartdate", "status start date"},
    "statusenddate": {"employmentstatusenddate", "estatusenddate", "status end date"},

    "eligibilitystartdate": {"eligstartdate", "eligibility start date", "elig start"},
    "eligibilityenddate": {"eligenddate", "eligibility end date", "elig end"},
    "eligibilitytier": {"elig_tier", "tier", "elig tier"},
    "plancode": {"plan", "plan code"},
    "planname": {"plan name"},
    "plancost": {"employee cost", "cost", "employee share"},

    "enrollmentstartdate": {"enrollstartdate", "enrollment start", "enroll start"},
    "enrollmentenddate": {"enrollenddate", "enrollment end", "enroll end"},
    "tier": {"enrollmenttier", "enrl_tier"},
    "isenrolled": {"enrolled", "is_enrolled"},
}

def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    x = df.copy()
    cols = {c: c for c in x.columns}
    for canon, alts in ALIASES.items():
        for c in list(cols):
            lc = str(c).lower().strip()
            if lc == canon:
                cols[c] = canon
            elif lc in {a.lower() for a in alts}:
                cols[c] = canon
    x = x.rename(columns=cols)

    for c in ("employeeid", "plancode", "planname", "eligibilitytier", "tier", "employmentstatus", "role", "name"):
        if c in x.columns:
            s = _safe_series(x, c)
            x[c] = _series_str_upper_strip(s).str.replace(r"\s+", " ", regex=True)

    for c in (
        "statusstartdate","statusenddate",
        "eligibilitystartdate","eligibilityenddate",
        "enrollmentstartdate","enrollmentenddate",
    ):
        if c in x.columns and not pd.api.types.is_datetime64_any_dtype(x[c]):
            x[c] = pd.to_datetime(x[c], errors="coerce")
    return x

# ---- Interval merging (adjacent = continuous)
def _norm_d(d) -> Optional[date]:
    return (d.date() if pd.notna(d) else None)

def _merge_intervals_touching(intervals: List[Tuple[Optional[date], Optional[date]]]) -> List[Tuple[date, date]]:
    if not intervals:
        return []
    norm: List[Tuple[date, date]] = []
    for s, e in intervals:
        s = s or date.min
        e = e or date.max
        if s > e:
            s, e = e, s
        norm.append((s, e))
    norm.sort()
    merged = [norm[0]]
    for s, e in norm[1:]:
        ls, le = merged[-1]
        if s <= (le + timedelta(days=1)):  # touching counts as continuous
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged

def _covers_full_union(df: pd.DataFrame, start_col: str, end_col: str, ms: date, me: date) -> bool:
    if df is None or df.empty:
        return False
    intervals = [(_norm_d(df.at[i, start_col]), _norm_d(df.at[i, end_col])) for i in df.index]
    merged = _merge_intervals_touching(intervals)
    cur = ms
    for s, e in merged:
        if s > cur:
            return False
        if e >= cur:
            cur = max(cur, min(e, me))
        if cur >= me:
            return True
    return cur >= me

# ---- Overlap / full-month tests with optional masks
def _any_overlap(df: pd.DataFrame, start_col: str, end_col: str, ms: date, me: date, *, mask: Optional[pd.Series]=None) -> bool:
    if df is None or df.empty:
        return False
    s = pd.to_datetime(_safe_series(df, start_col), errors="coerce").fillna(pd.Timestamp.min)
    e = pd.to_datetime(_safe_series(df, end_col), errors="coerce").fillna(pd.Timestamp.max)
    ov = e.dt.date.ge(ms) & s.dt.date.le(me)
    if mask is not None:
        ov = ov & mask.reindex_like(ov).fillna(False)
    return bool(ov.any())

def _all_month(df: pd.DataFrame, start_col: str, end_col: str, ms: date, me: date, *, mask: Optional[pd.Series]=None) -> bool:
    if df is None or df.empty:
        return False
    s = pd.to_datetime(_safe_series(df, start_col), errors="coerce").fillna(pd.Timestamp.min)
    e = pd.to_datetime(_safe_series(df, end_col), errors="coerce").fillna(pd.Timestamp.max)
    full = s.dt.date.le(ms) & e.dt.date.ge(me)
    if mask is not None:
        full = full & mask.reindex_like(full).fillna(False)
    return bool(full.any())

# ---- Tier helpers
TIER_ALIASES = {
    "EE": {"EMP", "EE", "EMPLOYEE"},
    "EMPSPOUSE": {"ES", "EMP+SPOUSE", "EMP_SPOUSE", "EMPSPOUSE"},
    "EMPCHILD": {"EC", "EMP+CHILD", "EMP_CHILD", "EMPCHILD", "EMP+CHILDREN"},
    "EMPFAM": {"EF", "EMP+FAM", "EMP_FAMILY", "FAMILY", "EMPFAM"},
}

def _tier_mask(df: pd.DataFrame, tier_col: str, tiers: Tuple[str, ...]) -> pd.Series:
    if df is None or df.empty or tier_col not in df.columns:
        return pd.Series(False, index=(df.index if isinstance(df, pd.DataFrame) else None))
    tnorm = _series_str_upper_strip(_safe_series(df, tier_col)).str.replace(r"[^A-Z]", "", regex=True)
    tier_set = set()
    for t in tiers:
        u = t.upper()
        tier_set.add(u)
        tier_set |= {a.upper() for a in TIER_ALIASES.get(u, set())}
    return tnorm.isin(tier_set)

def _latest_emp_cost_for_month(el_df: pd.DataFrame, ms: date, me: date) -> Optional[float]:
    if el_df is None or el_df.empty:
        return None
    df = el_df.copy()
    for c in ("eligibilitystartdate", "eligibilityenddate"):
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    mask = df["eligibilityenddate"].fillna(pd.Timestamp.max).dt.date.ge(ms) & \
           df["eligibilitystartdate"].fillna(pd.Timestamp.min).dt.date.le(me)
    df = df[mask]
    if df.empty or ("plancost" not in df.columns):
        return None
    if "eligibilitytier" in df.columns:
        tnorm = _series_str_upper_strip(_safe_series(df, "eligibilitytier")).str.replace(r"[^A-Z]", "", regex=True)
        emp_aliases = (TIER_ALIASES.get("EE", set()) | {"EE"})
        pref = df[tnorm.isin(emp_aliases)]
        if not pref.empty:
            df = pref
    if "eligibilitystartdate" in df.columns:
        df = df.sort_values("eligibilitystartdate", kind="stable")
    try:
        return float(df["plancost"].iloc[-1])
    except Exception:
        return None

# ---- Line code helpers
def _month_line14(eligible_mv: bool, offer_ee_allmonth: bool, offer_spouse: bool, offer_dependents: bool, affordable: bool) -> str:
    if not offer_ee_allmonth:
        return "1H"
    if not eligible_mv:
        return "1F"
    if offer_spouse and offer_dependents:
        return "1A" if affordable else "1E"
    return "1E"

def _month_line16(*, employed: bool, enrolled_full: bool, waiting: bool, ft: bool, offer_ee_allmonth: bool, affordable: bool) -> str:
    if enrolled_full:
        return "2C"
    if not employed:
        return "2A"
    if waiting:
        return "2D"
    if not ft:
        return "2B"
    if offer_ee_allmonth and affordable:
        return "2H"
    return ""

# =======================
# PUBLIC API (compat with main_fastapi)
# =======================
def load_input_workbook(excel_bytes_or_filelike: bytes | io.BytesIO | str) -> dict[str, pd.DataFrame]:
    """Load Excel into normalized dataframes keyed by canonical names."""
    if isinstance(excel_bytes_or_filelike, (bytes, bytearray)):
        buf = io.BytesIO(excel_bytes_or_filelike)
    else:
        buf = excel_bytes_or_filelike
    xls = pd.ExcelFile(buf)
    sheets: dict[str, pd.DataFrame] = {}

    def pick(name_variants: list[str]) -> Optional[str]:
        for s in xls.sheet_names:
            for v in name_variants:
                if s.lower().strip() == v.lower().strip():
                    return s
        for s in xls.sheet_names:
            for v in name_variants:
                if v.lower().strip() in s.lower().strip():
                    return s
        return None

    mapping = {
        "Emp Demographic": ["Emp Demographic","Employee Demographic","Demographic","Emp_Demographic"],
        "Emp Eligibility": ["Emp Eligibility","Eligibility","Employee Eligibility","Emp_Eligibility"],
        "Emp Enrollment": ["Emp Enrollment","Enrollment","Employee Enrollment","Emp_Enrollment"],
        "Dependent Enrollment": ["Dependent Enrollment","Dependents","Dep Enrollment","Dependent_Enrollment"],
        "Waiting Period": ["Waiting Period","Waiting","Wait","Waiting_Period"],
    }

    for key, variants in mapping.items():
        nm = pick(variants)
        sheets[key] = pd.read_excel(xls, sheet_name=nm) if nm else pd.DataFrame()

    return sheets

def build_interim_df(
    year: int,
    sheets: dict[str, pd.DataFrame] | bytes | bytearray | io.BytesIO | str,
    affordability_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Backward-compatible: accepts either a sheets dict OR raw Excel (bytes/filelike/path).
    """
    # Optional visibility for debugging types:
    try:
        log.info(f"build_interim_df: received type={type(sheets).__name__}")
    except Exception:
        pass

    if not isinstance(sheets, dict):
        sheets = load_input_workbook(sheets)

    demo = _apply_aliases(_df_or_empty(sheets.get("Emp Demographic")))
    elig = _apply_aliases(_df_or_empty(sheets.get("Emp Eligibility")))
    enr  = _apply_aliases(_df_or_empty(sheets.get("Emp Enrollment")))
    denr = _apply_aliases(_df_or_empty(sheets.get("Dependent Enrollment")))
    wait = _apply_aliases(_df_or_empty(sheets.get("Waiting Period")))

    return build_interim(
        emp_demo=demo,
        emp_elig=elig,
        emp_enroll=enr,
        dep_enroll=denr,
        year=year,
        emp_wait=wait,
        affordability_threshold=affordability_threshold,
    )

# =======================
# CORE BUILDERS
# =======================
def build_interim(
    emp_demo: pd.DataFrame,
    emp_elig: pd.DataFrame,
    emp_enroll: pd.DataFrame,
    dep_enroll: pd.DataFrame,
    year: int,
    *,
    emp_wait: pd.DataFrame | None = None,
    affordability_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Implements:
      - is_employed_for_full_month: union of all non-termination status intervals covers the month (adjacent intervals merge). LOA counts as employed.
      - ft: a FULL-TIME role interval covers the whole month.
      - parttime: no FT full-month, and a PART-TIME role interval covers the whole month.
      - spouse_enrolled / child_enrolled: True only if EMPFAM non-WAIVE ENROLLMENT covers the month.
      - include employees present only in Eligibility/Enrollment with employment flags = False all months.
    """
    with log_time(log, "build_interim"):
        emp_demo   = _apply_aliases(_df_or_empty(emp_demo))
        emp_elig   = _apply_aliases(_df_or_empty(emp_elig))
        emp_enroll = _apply_aliases(_df_or_empty(emp_enroll))
        dep_enroll = _apply_aliases(_df_or_empty(dep_enroll))
        emp_wait   = _apply_aliases(_df_or_empty(emp_wait))

        # All known EmployeeIDs so we never drop Elig/Enroll-only folks
        ids: set[str] = set()
        for df in (emp_demo, emp_elig, emp_enroll, dep_enroll, emp_wait):
            if not df.empty and "employeeid" in df.columns:
                ids |= set(df["employeeid"].astype(str).tolist())
        employee_ids = sorted(i for i in ids if i)

        # name map (if present)
        name_map: Dict[str, str] = {}
        if "name" in emp_demo.columns:
            emp_demo["name"] = _safe_series(emp_demo, "name").fillna("")
            name_map = {str(r["employeeid"]): str(r["name"]) for _, r in emp_demo.iterrows() if pd.notna(r.get("employeeid"))}

        thresh = AFFORDABILITY_THRESHOLD_DEFAULT if affordability_threshold is None else float(affordability_threshold)
        rows: List[Dict[str, Any]] = []

        # Pre-clean enrollment rows that are non-WAIVE (for spouse/child flags)
        en_non_waive = emp_enroll.copy()
        if not en_non_waive.empty:
            code = _series_str_upper_strip(_safe_series(en_non_waive, "plancode"))
            name = _series_str_upper_strip(_safe_series(en_non_waive, "planname"))
            not_waive_mask = ~(code.eq("WAIVE") | name.eq("WAIVE"))
            en_non_waive = en_non_waive[not_waive_mask]

        def live_rows(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame()
            s = _series_str_upper_strip(_safe_series(df, "employmentstatus"))
            mask_term = s.str.contains("TERMINAT", na=False) | s.eq("TERM")
            return df[~mask_term].copy()  # LOA remains employed

        for emp in employee_ids:
            st_emp = live_rows(emp_demo[emp_demo["employeeid"].astype(str) == emp]) if not emp_demo.empty else pd.DataFrame()
            el_emp = emp_elig[emp_elig["employeeid"].astype(str) == emp] if not emp_elig.empty else pd.DataFrame()
            en_emp = emp_enroll[emp_enroll["employeeid"].astype(str) == emp] if not emp_enroll.empty else pd.DataFrame()
            en_emp_nonwaive = en_non_waive[en_non_waive["employeeid"].astype(str) == emp] if not en_non_waive.empty else pd.DataFrame()
            wt_emp = emp_wait[emp_wait["employeeid"].astype(str) == emp] if not emp_wait.empty else pd.DataFrame()

            for m in range(1, 13):
                ms, me = month_bounds(year, m)

                # Employed/FT/PT
                employed_full = _covers_full_union(st_emp, "statusstartdate", "statusenddate", ms, me) if not st_emp.empty else False

                ft_rows = pd.DataFrame()
                if not st_emp.empty and "role" in st_emp.columns:
                    r = _series_str_upper_strip(_safe_series(st_emp, "role"))
                    ft_mask = r.str.contains("FULLTIME", na=False) | r.eq("FT")
                    ft_rows = st_emp[ft_mask]
                ft_full = _covers_full_union(ft_rows, "statusstartdate", "statusenddate", ms, me) if not ft_rows.empty else False

                pt_rows = pd.DataFrame()
                if not st_emp.empty and "role" in st_emp.columns:
                    r = _series_str_upper_strip(_safe_series(st_emp, "role"))
                    # FIXED: use str.contains (not str_contains)
                    pt_mask = r.str.contains("PARTTIME", na=False) | r.eq("PT")
                    pt_rows = st_emp[pt_mask]
                parttime_full = (not ft_full) and (_covers_full_union(pt_rows, "statusstartdate", "statusenddate", ms, me) if not pt_rows.empty else False)

                # Eligibility / Offer presence
                elig_any  = _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me) if not el_emp.empty else False
                elig_full = _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me) if not el_emp.empty else False

                # MV heuristic: PLANCODE == PLANA full-month ⇒ MV
                eligible_mv = False
                if not el_emp.empty and "plancode" in el_emp.columns:
                    plan_mask = _series_str_upper_strip(_safe_series(el_emp, "plancode")).eq("PLANA")
                    eligible_mv = _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=plan_mask)

                offer_ee_allmonth = _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me)

                offer_spouse = _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me,
                                            mask=_tier_mask(el_emp, "eligibilitytier", ("EMPSPOUSE",))) or \
                               _any_overlap(en_emp, "enrollmentstartdate", "enrollmentenddate", ms, me,
                                            mask=_tier_mask(en_emp, "tier", ("EMPSPOUSE",)))

                offer_dependents = _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me,
                                                mask=_tier_mask(el_emp, "eligibilitytier", ("EMPCHILD","EMPFAM"))) or \
                                   _any_overlap(en_emp, "enrollmentstartdate", "enrollmentenddate", ms, me,
                                                mask=_tier_mask(en_emp, "tier", ("EMPCHILD","EMPFAM")))

                enrolled_full = _all_month(en_emp, "enrollmentstartdate", "enrollmentenddate", ms, me) if not en_emp.empty else False

                # EMPFAM non-WAIVE full-month ⇒ spouse_enrolled & child_enrolled
                spouse_enrolled = False
                child_enrolled = False
                if not en_emp_nonwaive.empty and "tier" in en_emp_nonwaive.columns:
                    mask_empfam = _tier_mask(en_emp_nonwaive, "tier", ("EMPFAM",))
                    sub = en_emp_nonwaive.loc[mask_empfam].copy()
                    if not sub.empty:
                        if _all_month(sub, "enrollmentstartdate", "enrollmentenddate", ms, me):
                            spouse_enrolled = True
                            child_enrolled = True

                # Cost / affordability
                emp_cost = _latest_emp_cost_for_month(el_emp, ms, me)
                affordable = (emp_cost is not None) and (emp_cost <= (AFFORDABILITY_THRESHOLD_DEFAULT if affordability_threshold is None else float(affordability_threshold)))

                # Waiting period (optional)
                waiting = False
                if wt_emp is not None and not wt_emp.empty:
                    waiting = _any_overlap(wt_emp, "statusstartdate", "statusenddate", ms, me)

                l14 = _month_line14(eligible_mv, offer_ee_allmonth, offer_spouse, offer_dependents, affordable)
                if (not bool(elig_any)) and bool(enrolled_full):
                    l14 = "1E"
                l16 = _month_line16(
                    employed=bool(employed_full),
                    enrolled_full=bool(enrolled_full),
                    waiting=bool(waiting),
                    ft=bool(ft_full),
                    offer_ee_allmonth=bool(offer_ee_allmonth),
                    affordable=bool(affordable),
                )

                rows.append({
                    "EmployeeID": emp,
                    "Name": name_map.get(emp, ""),
                    "Year": int(year),
                    "MonthNum": int(m),
                    "Month": MONTHS[m-1],
                    "MonthStart": ms,
                    "MonthEnd": me,

                    "is_employed_for_full_month": bool(employed_full),
                    "ft": bool(ft_full),
                    "parttime": bool(parttime_full),

                    "eligibleforcoverage": bool(elig_any),
                    "eligible_allmonth": bool(elig_full),
                    "eligible_mv": bool(eligible_mv),
                    "offer_ee_allmonth": bool(offer_ee_allmonth),
                    "enrolled_allmonth": bool(enrolled_full),
                    "offer_spouse": bool(offer_spouse),
                    "offer_dependents": bool(offer_dependents),

                    "spouse_enrolled": bool(spouse_enrolled),
                    "child_enrolled": bool(child_enrolled),

                    "waitingperiod_month": bool(waiting),
                    "affordable_plan": bool(affordable),

                    "line14_final": l14,
                    "line16_final": l16,
                    "line14_all12": "",
                })

        interim = pd.DataFrame.from_records(rows)

        # Year-level 1G
        if not interim.empty:
            one_g_emp_ids = []
            for emp in interim["EmployeeID"].unique().tolist():
                g = interim[interim["EmployeeID"] == emp].sort_values("MonthNum", kind="stable")
                was_ft_any = bool(g["ft"].any())
                enrolled_any_month = bool(g["enrolled_allmonth"].any())
                if (not was_ft_any) and enrolled_any_month:
                    one_g_emp_ids.append(emp)
            if one_g_emp_ids:
                for emp in one_g_emp_ids:
                    mask = interim["EmployeeID"] == emp
                    interim.loc[mask, "line14_final"] = ""
                    idx = interim[mask].index.min()
                    interim.loc[idx, "line14_all12"] = "1G"

        log_df(log, "interim", interim)
        return interim

def build_final(interim: pd.DataFrame) -> pd.DataFrame:
    with log_time(log, "build_final"):
        cols = [
            "EmployeeID", "Year",
            "Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec",
            "Line16_Jan","Line16_Feb","Line16_Mar","Line16_Apr","Line16_May","Line16_Jun",
            "Line16_Jul","Line16_Aug","Line16_Sep","Line16_Oct","Line16_Nov","Line16_Dec",
            "Line14_All12"
        ]
        if interim is None or interim.empty:
            return pd.DataFrame(columns=cols)

        df = interim.copy()
        for i, m in enumerate(MONTHS, start=1):
            df[m] = df["line14_final"].where(df["MonthNum"] == i, "")
            df[f"Line16_{m}"] = df["line16_final"].where(df["MonthNum"] == i, "")

        final_rows = []
        for emp, g in df.groupby("EmployeeID", sort=False):
            rec: Dict[str, Any] = {"EmployeeID": emp, "Year": int(g["Year"].iloc[0])}
            for i, m in enumerate(MONTHS, start=1):
                sel = g["MonthNum"] == i
                rec[m] = g.loc[sel, "line14_final"].iloc[0] if sel.any() else ""
                rec[f"Line16_{m}"] = g.loc[sel, "line16_final"].iloc[0] if sel.any() else ""
            rec["Line14_All12"] = "1G" if (g["line14_all12"] == "1G").any() else ""
            final_rows.append(rec)

        out = pd.DataFrame.from_records(final_rows, columns=cols)
        log_df(log, "final", out)
        return out

def build_penalty_dashboard(interim: pd.DataFrame) -> pd.DataFrame:
    with log_time(log, "build_penalty_dashboard"):
        if interim is None or interim.empty:
            return pd.DataFrame(columns=["EmployeeID", "Reason"] + MONTHS)

        def month_reason(r: pd.Series) -> str:
            if not r.get("eligibleforcoverage", False):
                return "Not eligible"
            if not r.get("offer_ee_allmonth", False):
                return "No full-month offer"
            if not r.get("eligible_mv", False):
                return "Offer not MV (1F)"
            if not r.get("affordable_plan", False):
                return "Offered not affordable (B)"
            if not r.get("enrolled_allmonth", False):
                return "Offered not enrolled"
            return "–"

        out_rows: List[Dict[str, Any]] = []
        for emp, g in interim.groupby("EmployeeID", sort=False):
            rec = {"EmployeeID": emp}
            months_sorted = g.sort_values("MonthNum", kind="stable")
            reasons = [month_reason(r) for _, r in months_sorted.iterrows()]
            for i, m in enumerate(MONTHS):
                rec[m] = reasons[i] if i < len(reasons) else "–"
            rec["Reason"] = next((x for x in reasons if x != "–"), "–"
            )
            out_rows.append(rec)

        cols = ["EmployeeID", "Reason"] + MONTHS
        out = pd.DataFrame.from_records(out_rows, columns=cols)
        log_df(log, "penalty_dashboard", out)
        return out
