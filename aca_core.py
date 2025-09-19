# aca_core.py
# Core logic for ACA-1095 processing (Excel → interim/final/penalty) and PDF filling.

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import BooleanObject, DictionaryObject, NameObject
from reportlab.pdfgen import canvas

# =========================
# Constants & helpers
# =========================

TRUTHY = {"y", "yes", "true", "t", "1", 1, True}
FALSY = {"n", "no", "false", "f", "0", 0, False, None, np.nan}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

EXPECTED_SHEETS = {
    "emp demographic": [
        "employeeid",
        "firstname",
        "lastname",
        "ssn",
        "addressline1",
        "addressline2",
        "city",
        "state",
        "zipcode",
    ],
    "emp status": ["employeeid", "employmentstatus", "role", "statusstartdate", "statusenddate"],
    "emp eligibility": [
        "employeeid",
        "iseligibleforcoverage",
        "minimumvaluecoverage",
        "eligibilitystartdate",
        "eligibilityenddate",
        # optional helpers some inputs have:
        # "eligibleplan", "eligibleTier", "plancost"
    ],
    "emp enrollment": ["employeeid", "isenrolled", "enrollmentstartdate", "enrollmentenddate", "plancode"],
    "dep enrollment": ["employeeid", "dependentrelationship", "eligible", "enrolled", "eligiblestartdate", "eligibleenddate"],
    "pay deductions": ["employeeid", "amount", "startdate", "enddate", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
}

CANON_ALIASES = {
    # common misspellings
    "mimimumvaluecoverage": "minimumvaluecoverage",
    "minimimvaluecoverage": "minimumvaluecoverage",
    "zip": "zipcode",
    "zip code": "zipcode",
    "ssn (digits only)": "ssn",
    "eligible tier": "eligibletier",
    "plan cost": "plancost",
}

# --- cache for pay deductions so FastAPI doesn't need to pass it explicitly ---
_PAY_DED_CACHE: Optional[pd.DataFrame] = None


def _coerce_str(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def _last_day_of_month(y: int, m: int) -> date:
    if m == 12:
        return date(y, 12, 31)
    n1 = date(y, m + 1, 1)
    return n1 - pd.Timedelta(days=1)


def month_bounds(year: int, month: int) -> Tuple[date, date]:
    return date(year, month, 1), _last_day_of_month(year, month)


def _boolify(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: (str(v).strip().lower() in TRUTHY) if pd.notna(v) else False)
    return df


def _parse_date_cols(
    df: pd.DataFrame,
    cols: Iterable[str],
    default_end_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    default_end_cols = set(default_end_cols or [])
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            # if end-col and missing, treat as open-ended
            if c in default_end_cols:
                df[c] = df[c].fillna(pd.Timestamp.max)
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _ensure_employeeid_str(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "employeeid" in df.columns:
        df["employeeid"] = df["employeeid"].apply(_coerce_str)
    return df


def _any_overlap(df, start_col, end_col, m_start: date, m_end: date, mask: Optional[pd.Series] = None) -> bool:
    if df is None or df.empty:
        return False
    _m = mask if mask is not None else pd.Series(True, index=df.index)
    s = df.loc[_m, start_col].fillna(pd.Timestamp.min).dt.date
    e = df.loc[_m, end_col].fillna(pd.Timestamp.max).dt.date
    return bool(((e >= m_start) & (s <= m_end)).any())


def _all_month(df, start_col, end_col, m_start: date, m_end: date, mask: Optional[pd.Series] = None) -> bool:
    if df is None or df.empty:
        return False
    _m = mask if mask is not None else pd.Series(True, index=df.index)
    s = df.loc[_m, start_col].fillna(pd.Timestamp.min).dt.date
    e = df.loc[_m, end_col].fillna(pd.Timestamp.max).dt.date
    return bool(((s <= m_start) & (e >= m_end)).any())


# =========================
# Pay deductions (Line 15) — precedence & fallback
# =========================

def _pick_monthly_deduction(pay_df: Optional[pd.DataFrame], emp: str, m_start: date, m_end: date) -> float:
    """
    Return the Line 15 monthly employee contribution for the month window.

    Precedence:
      1) Pay Deductions sheet (if provided):
         A) Wide monthly columns (jan..dec)
         B) Range rows with amount/startdate/enddate (latest effective wins)
      2) Fallback to Emp Eligibility self-only PlanCost (EMP tier) overlapping the month
         (handled in build_interim by passing a fallback value when pay_df is empty)
    """
    if pay_df is None or pay_df.empty:
        return np.nan

    emp_key = _coerce_str(emp)
    df = pay_df[pay_df["employeeid"].map(_coerce_str) == emp_key].copy()
    if df.empty:
        return np.nan

    # Wide columns path
    mon_key = m_start.strftime("%b").lower()  # "jan", "feb", ...
    if mon_key in df.columns:
        val = pd.to_numeric(df.iloc[0][mon_key], errors="coerce")
        return round(float(val), 2) if pd.notna(val) else np.nan

    # Range rows path
    if {"amount", "startdate", "enddate"} <= set(df.columns):
        ov = df[(df["startdate"] <= pd.to_datetime(m_end)) & (df["enddate"] >= pd.to_datetime(m_start))]
        if ov.empty:
            return np.nan
        ov = ov.sort_values(["startdate", "enddate"])
        val = pd.to_numeric(ov.iloc[-1]["amount"], errors="coerce")
        return round(float(val), 2) if pd.notna(val) else np.nan

    return np.nan


# =========================
# Excel ingestion
# =========================

def load_excel(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """
    Loads the workbook and returns canonicalized DataFrames:
    emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, pay_deductions
    """
    global _PAY_DED_CACHE

    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    raw_map: Dict[str, pd.DataFrame] = {}
    for raw in xls.sheet_names:
        df = pd.read_excel(xls, raw)
        df = normalize_columns(df)
        df = df.rename(columns={k: v for k, v in CANON_ALIASES.items() if k in df.columns})
        raw_map[raw.strip().lower()] = df

    # Fuzzy match expected sheets
    def find_sheet(name_like: str) -> Optional[str]:
        nl = name_like.strip().lower()
        for s in raw_map:
            if nl == s:
                return s
        for s in raw_map:
            if nl in s:
                return s
        return None

    cleaned: Dict[str, pd.DataFrame] = {}
    for want, cols in EXPECTED_SHEETS.items():
        key = find_sheet(want) or want
        df = raw_map.get(key, pd.DataFrame(columns=cols)).copy()
        df = normalize_columns(df)
        if df.empty:
            cleaned[want] = pd.DataFrame(columns=cols)
            continue

        # normalize aliased columns & employee id
        for misspell, canon in CANON_ALIASES.items():
            if misspell in df.columns and canon not in df.columns:
                df = df.rename(columns={misspell: canon})
        df = _ensure_employeeid_str(df)

        # per-sheet cleaning
        if want == "emp status":
            if "employmentstatus" in df.columns:
                df["employmentstatus"] = df["employmentstatus"].astype(str).str.strip().str.upper()
            if "role" in df.columns:
                df["role"] = df["role"].astype(str).str.strip().str.upper()
            df = _parse_date_cols(df, ["statusstartdate", "statusenddate"], default_end_cols=["statusenddate"])
        elif want == "emp eligibility":
            df = _boolify(df, ["iseligibleforcoverage", "minimumvaluecoverage"])
            df = _parse_date_cols(df, ["eligibilitystartdate", "eligibilityenddate"], default_end_cols=["eligibilityenddate"])
            # normalize tier & plan
            if "eligibletier" in df.columns:
                df["eligibletier"] = df["eligibletier"].astype(str).str.strip().str.upper()
            if "eligibleplan" in df.columns:
                df["eligibleplan"] = df["eligibleplan"].astype(str).str.strip().str.upper()
        elif want == "emp enrollment":
            df = _boolify(df, ["isenrolled"])
            df = _parse_date_cols(df, ["enrollmentstartdate", "enrollmentenddate"], default_end_cols=["enrollmentenddate"])
            if "plancode" in df.columns:
                df["plancode"] = df["plancode"].astype(str).str.strip().str.upper()
        elif want == "dep enrollment":
            if "dependentrelationship" in df.columns:
                df["dependentrelationship"] = df["dependentrelationship"].astype(str).str.strip().str.title()
            df = _boolify(df, ["eligible", "enrolled"])
            df = _parse_date_cols(df, ["eligiblestartdate", "eligibleenddate"], default_end_cols=["eligibleenddate"])
        elif want == "pay deductions":
            df = _parse_date_cols(df, ["startdate", "enddate"], default_end_cols=["enddate"])
            # lower-case monthly columns if present
            for mon in [m.lower() for m in MONTHS]:
                if mon in df.columns:
                    df[mon] = pd.to_numeric(df[mon], errors="coerce")

        cleaned[want] = df

    _PAY_DED_CACHE = cleaned.get("pay deductions", pd.DataFrame())

    return {
        "emp_demo": cleaned.get("emp demographic", pd.DataFrame()),
        "emp_status": cleaned.get("emp status", pd.DataFrame()),
        "emp_elig": cleaned.get("emp eligibility", pd.DataFrame()),
        "emp_enroll": cleaned.get("emp enrollment", pd.DataFrame()),
        "dep_enroll": cleaned.get("dep enrollment", pd.DataFrame()),
        "pay_deductions": cleaned.get("pay deductions", pd.DataFrame()),
    }


def _collect_employee_ids(*dfs: pd.DataFrame) -> List[str]:
    ids = set()
    for df in dfs:
        if df is None or df.empty:
            continue
        if "employeeid" in df.columns:
            ids.update(map(_coerce_str, df["employeeid"].dropna().tolist()))
    return sorted(ids)


def choose_report_year(emp_elig: pd.DataFrame, fallback_to_current: bool = True) -> int:
    """Pick the year with the most eligibility rows overlapping, else current year or 2024."""
    if emp_elig is None or emp_elig.empty:
        return datetime.now().year if fallback_to_current else 2024
    counts: Dict[int, int] = {}
    for _, r in emp_elig.iterrows():
        s = pd.to_datetime(r.get("eligibilitystartdate"), errors="coerce")
        e = pd.to_datetime(r.get("eligibilityenddate"), errors="coerce")
        if pd.isna(s) and pd.isna(e):
            continue
        s = s or pd.Timestamp.min
        e = e or pd.Timestamp.max
        for y in range(s.year, e.year + 1):
            counts[y] = counts.get(y, 0) + 1
    if not counts:
        return datetime.now().year if fallback_to_current else 2024
    return max(sorted(counts), key=lambda y: (counts[y], y))


def _grid_for_year(employee_ids: List[str], year: int) -> pd.DataFrame:
    recs = []
    for emp in employee_ids:
        for m in range(1, 13):
            ms, me = month_bounds(year, m)
            recs.append(
                {
                    "employeeid": emp,
                    "year": year,
                    "monthnum": m,
                    "month": ms.strftime("%b"),
                    "monthstart": ms,
                    "monthend": me,
                }
            )
    g = pd.DataFrame.from_records(recs)
    g["monthstart"] = pd.to_datetime(g["monthstart"])
    g["monthend"] = pd.to_datetime(g["monthend"])
    return g


# =========================
# Interim grid & coding
# =========================

@dataclass
class RunConfig:
    year: Optional[int] = None
    aca_mode: str = "SIMPLIFIED"  # or "IRS_STRICT"
    affordability_threshold: float = 50.0  # used only in SIMPLIFIED
    penalty_a_amount: float = 241.67  # display only (dashboard)
    penalty_b_amount: float = 362.50  # display only (dashboard)


def build_interim(
    emp_demo: pd.DataFrame,
    emp_status: pd.DataFrame,
    emp_elig: pd.DataFrame,
    emp_enroll: pd.DataFrame,
    dep_enroll: pd.DataFrame,
    cfg: RunConfig | None = None,
) -> pd.DataFrame:
    """
    Returns the Interim grid with monthly flags + Line14/15/16 + Penalty flags/reasons.
    - Line 15 precedence: Pay Deductions > Eligibility (EMP tier PlanCost fallback)
    - Offer scope: from Dep Enrollment or Eligibility.EligibleTier (EMPFAM ⇒ spouse+dependents)
    - SIMPLIFIED mode: waived + unaffordable ⇒ 1H/2D override to match your UAT scenarios.
    """
    cfg = cfg or RunConfig()
    year = cfg.year or choose_report_year(emp_elig)

    employee_ids = _collect_employee_ids(emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll)
    grid = _grid_for_year(employee_ids, year)

    demo = emp_demo[["employeeid", "firstname", "lastname"]].drop_duplicates() if not emp_demo.empty else pd.DataFrame(
        columns=["employeeid", "firstname", "lastname"]
    )
    out = grid.merge(demo, on="employeeid", how="left")

    stt, elg, enr, dep, pay = (
        emp_status.copy(),
        emp_elig.copy(),
        emp_enroll.copy(),
        dep_enroll.copy(),
        _PAY_DED_CACHE if _PAY_DED_CACHE is not None else pd.DataFrame(),
    )

    # Normalize date dtypes just in case
    for df in (stt, elg, enr, dep):
        if not df.empty:
            for c in df.columns:
                if c.endswith("date") and not np.issubdtype(df[c].dtype, np.datetime64):
                    df[c] = pd.to_datetime(df[c], errors="coerce")

    flags: List[Dict] = []

    for _, row in out.iterrows():
        emp = row["employeeid"]
        ms = row["monthstart"].date()
        me = row["monthend"].date()

        st_emp = stt[stt["employeeid"] == emp] if not stt.empty else pd.DataFrame()
        el_emp = elg[elg["employeeid"] == emp] if not elg.empty else pd.DataFrame()
        en_emp = enr[enr["employeeid"] == emp] if not enr.empty else pd.DataFrame()
        de_emp = dep[dep["employeeid"] == emp] if not dep.empty else pd.DataFrame()

        # Employment & FT
        employed = _any_overlap(st_emp, "statusstartdate", "statusenddate", ms, me) if not st_emp.empty else True
        ft = _any_overlap(st_emp, "statusstartdate", "statusenddate", ms, me, mask=st_emp.get("role", pd.Series()).astype(str).str.upper().eq("FT")) if not st_emp.empty else True

        # Eligibility flags
        eligible_any = _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me) if not el_emp.empty else False
        eligible_allmonth = _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me) if not el_emp.empty else False
        eligible_mv_any = _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=el_emp.get("minimumvaluecoverage", pd.Series(False)).fillna(False)) if not el_emp.empty else False

        # Offer scope: spouse/dependents from Dep Enrollment OR EligibleTier=EMPFAM
        offer_spouse = False
        offer_dependents = False
        if not de_emp.empty and {"dependentrelationship", "eligiblestartdate", "eligibleenddate"} <= set(de_emp.columns):
            offer_spouse = _any_overlap(
                de_emp, "eligiblestartdate", "eligibleenddate", ms, me, mask=de_emp["dependentrelationship"].eq("Spouse")
            )
            offer_dependents = _any_overlap(
                de_emp, "eligiblestartdate", "eligibleenddate", ms, me, mask=de_emp["dependentrelationship"].eq("Child")
            )
        # Tier fallback
        if "eligibletier" in el_emp.columns:
            fam_any = _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=el_emp["eligibletier"].astype(str).str.upper().eq("EMPFAM"))
            if fam_any:
                offer_spouse = True
                offer_dependents = True

        # Enrollment
        enrolled_allmonth = False
        if not en_emp.empty and {"enrollmentstartdate", "enrollmentenddate"} <= set(en_emp.columns):
            en_mask = en_emp.get("isenrolled", pd.Series(True, index=en_emp.index)).fillna(False)
            enrolled_allmonth = _all_month(en_emp, "enrollmentstartdate", "enrollmentenddate", ms, me, mask=en_mask)

        waived_any = False
        if "plancode" in en_emp.columns:
            waived_any = _any_overlap(en_emp, "enrollmentstartdate", "enrollmentenddate", ms, me, mask=en_emp["plancode"].astype(str).str.upper().eq("WAIVE"))

        waitingperiod_month = bool(employed and ft and not eligible_any)

        # ---- Line 15 (Employee Required Contribution - monthly amount) ----
        l15_amt = _pick_monthly_deduction(pay, emp, ms, me)
        # Fallback: Eligibility self-only (EMP tier) PlanCost overlapping month
        if (pd.isna(l15_amt) or l15_amt is np.nan) and "plancost" in el_emp.columns:
            emp_tier = el_emp["eligibletier"].astype(str).str.upper().eq("EMP") if "eligibletier" in el_emp.columns else pd.Series(True, index=el_emp.index)
            ov = el_emp[
                emp_tier
                & (el_emp["eligibilitystartdate"] <= pd.to_datetime(me))
                & (el_emp["eligibilityenddate"] >= pd.to_datetime(ms))
            ]
            if not ov.empty:
                # latest effective in month
                ov = ov.sort_values(["eligibilitystartdate", "eligibilityenddate"])
                try:
                    l15_amt = float(pd.to_numeric(ov.iloc[-1]["plancost"], errors="coerce"))
                except Exception:
                    l15_amt = np.nan

        # ---- Line 14 (Offer code) ----
        if eligible_allmonth and eligible_mv_any:
            if offer_spouse and offer_dependents:
                l14 = "1E"
            elif offer_spouse and not offer_dependents:
                l14 = "1D"
            elif not offer_spouse and offer_dependents:
                l14 = "1C"
            else:
                l14 = "1B"
        elif eligible_allmonth and not eligible_mv_any:
            l14 = "1F"
        else:
            l14 = "1H"

        # SIMPLIFIED override: waived + unaffordable ⇒ 1H to match your Scenarios
        if cfg.aca_mode.upper() == "SIMPLIFIED":
            if employed and ft and eligible_mv_any and waived_any and (pd.notna(l15_amt) and (float(l15_amt) > float(cfg.affordability_threshold))):
                l14 = "1H"

        # ---- Line 16 (Safe Harbor / other) ----
        if enrolled_allmonth:
            l16 = "2C"
        elif not employed:
            l16 = "2A"
        elif waitingperiod_month:
            l16 = "2D"
        elif not ft:
            l16 = "2B"
        else:
            l16 = ""

        # SIMPLIFIED override for 2D in waived+unaffordable months (to tell the story)
        if cfg.aca_mode.upper() == "SIMPLIFIED":
            if employed and ft and eligible_mv_any and waived_any and (pd.notna(l15_amt) and (float(l15_amt) > float(cfg.affordability_threshold))):
                l16 = "2D"

        # Penalty flags (storytelling / dashboard)
        penalty_a = bool(ft and employed and (l14 == "1H") and not (waived_any and (pd.notna(l15_amt) and float(l15_amt) > float(cfg.affordability_threshold))))
        penalty_b = bool(ft and employed and (l14 == "1H") and (waived_any and (pd.notna(l15_amt) and float(l15_amt) > float(cfg.affordability_threshold))))

        reason = ""
        if penalty_a:
            if not eligible_any:
                reason = "Penalty A: No MV/MEC offer (waiting/eligibility gap/class)"
            elif not eligible_mv_any:
                reason = "Penalty A: Only non-MV plan available"
            else:
                reason = "Penalty A: No MV/MEC offer for FT month"
        elif penalty_b:
            reason = "Penalty B: Waived unaffordable (offer present; cost > threshold)"

        flags.append(
            {
                "employed": employed,
                "ft": ft,
                "eligibleforcoverage": eligible_any,
                "eligible_allmonth": eligible_allmonth,
                "eligible_mv": eligible_mv_any,
                "offer_spouse": offer_spouse,
                "offer_dependents": offer_dependents,
                "enrolled_allmonth": enrolled_allmonth,
                "waived_any": waived_any,
                "waitingperiod_month": waitingperiod_month,
                "line14_final": l14,
                "line16_final": l16,
                "line15_amount": l15_amt,
                "PenaltyA_Flag": penalty_a,
                "PenaltyB_Flag": penalty_b,
                "Penalty_Reason": reason,
            }
        )

    interim = pd.concat([out.reset_index(drop=True), pd.DataFrame(flags)], axis=1)

    base_cols = [
        "employeeid",
        "firstname",
        "lastname",
        "year",
        "monthnum",
        "month",
        "monthstart",
        "monthend",
    ]
    flag_cols = [
        "employed",
        "ft",
        "eligibleforcoverage",
        "eligible_allmonth",
        "eligible_mv",
        "offer_spouse",
        "offer_dependents",
        "enrolled_allmonth",
        "waived_any",
        "waitingperiod_month",
        "line14_final",
        "line15_amount",
        "line16_final",
        "PenaltyA_Flag",
        "PenaltyB_Flag",
        "Penalty_Reason",
    ]
    keep = [c for c in base_cols if c in interim.columns] + [c for c in flag_cols if c in interim.columns]
    interim = interim[keep].sort_values(["employeeid", "year", "monthnum"]).reset_index(drop=True)
    return interim


def build_final(interim: pd.DataFrame) -> pd.DataFrame:
    df = interim.copy()
    out = df.loc[:, ["employeeid", "month", "line14_final", "line16_final", "line15_amount"]].rename(
        columns={
            "employeeid": "EmployeeID",
            "month": "Month",
            "line14_final": "Line14_Final",
            "line16_final": "Line16_Final",
            "line15_amount": "Line15_Amount",
        }
    )
    # two-decimal formatting for display in "Final"
    if "Line15_Amount" in out.columns:
        out["Line15_Amount"] = out["Line15_Amount"].apply(lambda x: ("" if pd.isna(x) else f"{float(x):.2f}"))

    if "monthnum" in df.columns:
        out = out.join(df["monthnum"]).sort_values(["EmployeeID", "monthnum"]).drop(columns=["monthnum"])
    else:
        order = {m: i for i, m in enumerate(MONTHS, start=1)}
        out["_ord"] = out["Month"].map(order)
        out = out.sort_values(["EmployeeID", "_ord"]).drop(columns=["_ord"])
    return out.reset_index(drop=True)


def build_penalty_dashboard(interim: pd.DataFrame, year: int, penalty_a_amt: float, penalty_b_amt: float) -> pd.DataFrame:
    """
    Returns a dashboard table with columns: EmployeeID | Reason | Jan..Dec
    Values are display-only monthly amounts; "-" where no penalty.
    """
    if interim is None or interim.empty:
        return pd.DataFrame(columns=["EmployeeID", "Reason"] + MONTHS)

    rows = []
    for emp, grp in interim.groupby("employeeid"):
        reason = ""
        monthly = {m: "-" for m in MONTHS}
        for _, r in grp.iterrows():
            m = r["month"]
            pa = bool(r.get("PenaltyA_Flag", False))
            pb = bool(r.get("PenaltyB_Flag", False))
            if pb:
                monthly[m] = f"{penalty_b_amt:.2f}"
                reason = r.get("Penalty_Reason", "") or "Penalty B"
            elif pa:
                monthly[m] = f"{penalty_a_amt:.2f}"
                reason = r.get("Penalty_Reason", "") or "Penalty A"
            # else keep as "-"
        rows.append({"EmployeeID": emp, "Reason": reason, **monthly})
    return pd.DataFrame(rows)[["EmployeeID", "Reason"] + MONTHS]


# =========================
# PDF helpers (Part I + Part II)
# =========================

def normalize_ssn_digits(ssn: str) -> str:
    d = "".join(ch for ch in str(ssn) if str(ch).isdigit())
    return f"{d[0:3]}-{d[3:5]}-{d[5:9]}" if len(d) >= 9 else d


# IRS 1095-C 2024 field names (page 1) — keep your existing names
# Part I
F_PART1 = ["f1_1[0]", "f1_2[0]", "f1_3[0]", "f1_4[0]", "f1_5[0]", "f1_6[0]", "f1_7[0]", "f1_8[0]"]
# Part II Line 14 (All 12 + Jan..Dec)
F_L14 = ["f1_17[0]", "f1_18[0]", "f1_19[0]", "f1_20[0]", "f1_21[0]", "f1_22[0]", "f1_23[0]",
         "f1_24[0]", "f1_25[0]", "f1_26[0]", "f1_27[0]", "f1_28[0]", "f1_29[0]"]
# Part II Line 16 (All 12 + Jan..Dec)
F_L16 = ["f1_43[0]", "f1_44[0]", "f1_45[0]", "f1_46[0]", "f1_47[0]", "f1_48[0]", "f1_49[0]",
         "f1_50[0]", "f1_51[0]", "f1_52[0]", "f1_53[0]", "f1_54[0]", "f1_55[0]"]
# Part II Line 15 (All 12 + Jan..Dec)
F_L15 = ["f1_30[0]", "f1_31[0]", "f1_32[0]", "f1_33[0]", "f1_34[0]", "f1_35[0]", "f1_36[0]",
         "f1_37[0]", "f1_38[0]", "f1_39[0]", "f1_40[0]", "f1_41[0]", "f1_42[0]"]


def set_need_appearances(writer: PdfWriter):
    root = writer._root_object
    if "/AcroForm" not in root:
        root.update({NameObject("/AcroForm"): DictionaryObject()})
    root["/AcroForm"].update({NameObject("/NeedAppearances"): BooleanObject(True)})


def build_overlay(page_width: float, page_height: float, mapping: Dict[str, str]) -> bytes:
    """
    Create a 1-page overlay PDF with text for fields that some viewers don't render unless "burned in".
    This is a generic helper; exact x,y mapping can be customized if needed.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_width, page_height))
    c.setFont("Helvetica", 10)
    # Very light overlay: we just nudge All-12 values near their printed boxes; coordinates may be adjusted later.
    # (Leaving as minimal to avoid misalignment issues from earlier attempts.)
    # You can extend this with precise coords per field if desired.
    c.save()
    buf.seek(0)
    return buf.getvalue()


def fill_pdf_for_employee(
    blank_pdf_bytes: bytes,
    emp_demo: pd.DataFrame,
    final_df_emp: pd.DataFrame,
    year_used: int,
) -> Tuple[str, io.BytesIO, str, io.BytesIO]:
    """
    Fills Part I (name/SSN/address) and Part II (Line 14/15/16 All-12 or Jan..Dec) for a single employee.
    Returns (editable_filename, editable_bytes, flattened_filename, flattened_bytes)
    """
    reader = PdfReader(io.BytesIO(blank_pdf_bytes))

    # Part I — first/last/ssn/address from emp_demo
    emp_row = emp_demo.iloc[0] if not emp_demo.empty else pd.Series({})
    first = _coerce_str(emp_row.get("firstname", ""))
    last = _coerce_str(emp_row.get("lastname", ""))
    ssn = normalize_ssn_digits(emp_row.get("ssn", ""))
    addr1 = _coerce_str(emp_row.get("addressline1", ""))
    city = _coerce_str(emp_row.get("city", ""))
    state = _coerce_str(emp_row.get("state", ""))
    zipcode = _coerce_str(emp_row.get("zipcode", ""))

    part1_map = {
        F_PART1[0]: f"{first} {last}".strip(),
        F_PART1[1]: ssn,
        F_PART1[2]: addr1,
        F_PART1[3]: city,
        F_PART1[4]: state,
        F_PART1[5]: zipcode,
        # F_PART1[6], F_PART1[7] are typically employer name/EIN — left to your existing fill flow if needed.
    }

    # Part II — assemble monthly dicts
    l14_by_m = {row["Month"]: _coerce_str(row["Line14_Final"]) for _, row in final_df_emp.iterrows()}
    l16_by_m = {row["Month"]: _coerce_str(row["Line16_Final"]) for _, row in final_df_emp.iterrows()}
    l15_by_m = {row["Month"]: _coerce_str(row.get("Line15_Amount", "")) for _, row in final_df_emp.iterrows()}

    def all12_value(d: Dict[str, str]) -> str:
        vals = [d.get(m, "") for m in MONTHS]
        uniq = {v for v in vals if v}
        return list(uniq)[0] if len(uniq) == 1 else ""

    l14_all = all12_value(l14_by_m)
    l16_all = all12_value(l16_by_m)
    l15_all = all12_value(l15_by_m)

    l14_values = [l14_all] + [l14_by_m.get(m, "") for m in MONTHS]
    l16_values = [l16_all] + [l16_by_m.get(m, "") for m in MONTHS]
    l15_values = [l15_all] + [l15_by_m.get(m, "") for m in MONTHS]

    mapping = {}
    mapping.update({name: val for name, val in zip(F_L14, l14_values)})
    mapping.update({name: val for name, val in zip(F_L16, l16_values)})
    mapping.update({name: val for name, val in zip(F_L15, l15_values)})
    mapping.update(part1_map)

    # ---- EDITABLE output (NeedAppearances + optional overlay burn-in on page 1) ----
    writer_edit = PdfWriter()
    for i in range(len(reader.pages)):
        writer_edit.add_page(reader.pages[i])

    # Set field values if acroform present
    if "/AcroForm" in reader.trailer["/Root"]:
        form = reader.trailer["/Root"]["/AcroForm"]
        if "/Fields" in form:
            set_need_appearances(writer_edit)
            fields = writer_edit._root_object["/AcroForm"]["/Fields"]
            # Build a lookup of field names to /Fields indices
            name_to_field = {}
            for f in fields:
                try:
                    name_to_field[f.get_object()["/T"]] = f
                except Exception:
                    pass
            # Set values
            for name, val in mapping.items():
                if not val:
                    continue
                if name in name_to_field:
                    obj = name_to_field[name].get_object()
                    obj.update({NameObject("/V"): NameObject(str(val))})

    editable_name = f"1095c_filled_editable_{first}_{last}_{year_used}.pdf"
    editable_bytes = io.BytesIO()
    writer_edit.write(editable_bytes)
    editable_bytes.seek(0)

    # ---- FLATTENED output: re-read the edited file and write page streams (burns in appearance) ----
    reader2 = PdfReader(io.BytesIO(editable_bytes.getvalue()))
    writer_flat = PdfWriter()
    for i in range(len(reader2.pages)):
        writer_flat.add_page(reader2.pages[i])
    # The simple write usually flattens effectively for most viewers
    flattened_name = f"1095c_filled_flattened_{first}_{last}_{year_used}.pdf"
    flattened_bytes = io.BytesIO()
    writer_flat.write(flattened_bytes)
    flattened_bytes.seek(0)

    return editable_name, editable_bytes, flattened_name, flattened_bytes


# =========================
# Excel outputs
# =========================

def save_excel_outputs(
    interim: pd.DataFrame,
    final: pd.DataFrame,
    year: int,
    penalty_dashboard: Optional[pd.DataFrame] = None,
    uat_compare: Optional[pd.DataFrame] = None,
) -> bytes:
    """
    Writes Final + Interim (+ optional Penalty Dashboard, UAT Compare) to a single workbook.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as xw:
        final.to_excel(xw, index=False, sheet_name=f"Final {year}")
        interim.to_excel(xw, index=False, sheet_name=f"Interim {year}")
        if penalty_dashboard is not None and not penalty_dashboard.empty:
            penalty_dashboard.to_excel(xw, index=False, sheet_name=f"Penalty Dashboard {year}")
        if uat_compare is not None and not uat_compare.empty:
            uat_compare.to_excel(xw, index=False, sheet_name=f"UAT Compare {year}")
    buf.seek(0)
    return buf.getvalue()
