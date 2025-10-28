# aca_builder.py
from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

from debug_logging import get_logger, log_df, log_time
from aca_processing import (
    MONTHS, FULL_MONTHS,
    _collect_employee_ids, month_bounds,
    _any_overlap, _all_month,
    _status_from_demographic, _waiting_in_month,
)

log = get_logger("builder")

AFFORDABILITY_THRESHOLD_DEFAULT = 50.0  # UAT/test threshold for affordability

# -----------------------
# Utilities / Aliases
# -----------------------

ALIASES = {
    "employeeid": {"employee id", "empid", "emp id", "id"},
    "firstname": {"first name", "fname", "givenname"},
    "middlename": {"middle name", "mname"},
    "lastname": {"last name", "lname", "surname", "familyname"},
    "ssn": {"social", "ssn#", "socialsecuritynumber"},
    "address1": {"address", "addr1", "address line 1"},
    "address2": {"addr2", "address line 2"},
    "city": set(),
    "state": {"province", "region"},
    "zip": {"postal", "zipcode"},
    "status": {"employmentstatus", "empstatus", "estatus"},
    "statusstartdate": {"employmentstatusstartdate", "estatusstartdate"},
    "statusenddate": {"employmentstatusenddate", "estatusenddate"},

    "eligibilitystartdate": {"eligstartdate", "eligibility start date"},
    "eligibilityenddate": {"eligenddate", "eligibility end date"},
    "eligibilitytier": {"elig_tier", "tier"},
    "plancode": {"plan", "plan code"},
    "planname": {"plan name"},
    "plancost": {"employee cost", "cost", "employee share"},

    "enrollmentstartdate": {"enrollstartdate", "enrollment start"},
    "enrollmentenddate": {"enrollenddate", "enrollment end"},
    "tier": {"enrollmenttier", "enrl_tier"},
    "isenrolled": {"enrolled", "is_enrolled"},

    "dependentrelationship": {"relationship", "dep_relationship"},
    "waitperiod": {"waitperioddays", "wait period days"},
}

TIER_ALIASES = {
    "EE": {"EMP", "EE", "EMPLOYEE"},
    "EMPSPOUSE": {"ES", "EMP+SPOUSE", "EMP_SPOUSE", "EMPSPOUSE"},
    "EMPCHILD": {"EC", "EMP+CHILD", "EMP_CHILD", "EMPCHILD", "EMP+CHILDREN"},
    "EMPFAM": {"EF", "EMP+FAM", "EMP_FAMILY", "FAMILY", "EMPFAM"},
}

# ---- Safe access helpers (prevent '.astype' on scalars) -----------------

def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] as Series or an empty Series aligned to df.index if missing/invalid."""
    if isinstance(df, pd.DataFrame) and (col in df.columns):
        s = df[col]
        if isinstance(s, pd.Series):
            return s
        return pd.Series([s] * len(df), index=df.index)
    return pd.Series(index=(df.index if isinstance(df, pd.DataFrame) else None), dtype="object")

def _series_str_upper_strip(s: pd.Series) -> pd.Series:
    """Coerce to string, uppercase, strip; safe on empty/any dtype."""
    if not isinstance(s, pd.Series):
        s = pd.Series([], dtype="object")
    try:
        return s.astype(str).str.upper().str.strip()
    except Exception:
        return s.map(lambda x: ("" if pd.isna(x) else str(x).upper().strip()))

def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    x = df.copy()
    # column renames
    cols = {c: c for c in x.columns}
    for canon, alts in ALIASES.items():
        for c in list(cols):
            lc = str(c).lower().strip()
            if lc == canon:
                cols[c] = canon
            elif lc in {a.lower() for a in alts}:
                cols[c] = canon
    x = x.rename(columns=cols)
    # normalize common string columns (safely)
    for c in ("employeeid", "plancode", "planname", "eligibilitytier", "tier"):
        if c in x.columns:
            s = _safe_series(x, c)
            x[c] = _series_str_upper_strip(s).str.replace(r"\s+", " ", regex=True)
    return x

def _df_or_empty(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

# ---- Month/offer/enrollment helpers ------------------------------------

def _offered_allmonth(el_df: pd.DataFrame, ms, me) -> bool:
    """Any eligibility row (any tier) covering the full month → offer_ee_allmonth."""
    if el_df is None or el_df.empty:
        return False
    return _all_month(el_df, "eligibilitystartdate", "eligibilityenddate", ms, me)

def _tier_offered_any(
    df: pd.DataFrame,
    tier_col: str,
    tiers: Tuple[str, ...],
    start_col: str,
    end_col: str,
    ms,
    me,
    *,
    require_enrolled: bool = False
) -> bool:
    if df is None or df.empty or tier_col not in df.columns:
        return False

    tnorm = _series_str_upper_strip(_safe_series(df, tier_col)).str.replace(r"[^A-Z]", "", regex=True)

    # Expand aliases to a comparable tier set
    tier_set = set()
    for t in tiers:
        t_up = t.upper()
        tier_set.add(t_up)
        if t_up in TIER_ALIASES:
            tier_set |= {a.upper() for a in TIER_ALIASES[t_up]}

    mask = tnorm.isin(tier_set)
    if require_enrolled and "isenrolled" in df.columns:
        mask &= _safe_series(df, "isenrolled").astype(bool).fillna(False)

    return _any_overlap(df, start_col, end_col, ms, me, mask=mask)

def _enrolled_full_month_union(en_df: pd.DataFrame, ms, me) -> bool:
    """True if any non-WAIVE enrollment interval fully covers [ms, me]."""
    if en_df is None or en_df.empty:
        return False

    df = en_df.copy()

    # filter to enrolled==True if present
    mask = pd.Series(True, index=df.index)
    if "isenrolled" in df.columns:
        mask &= _safe_series(df, "isenrolled").astype(bool).fillna(False)

    # exclude WAIVE if present in code/name
    waive_mask = pd.Series(False, index=df.index)
    for col in ("plancode", "planname"):
        if col in df.columns:
            s = _series_str_upper_strip(_safe_series(df, col))
            waive_mask |= s.eq("WAIVE")

    df = df.loc[mask & ~waive_mask]
    if df.empty:
        return False

    # normalize dates
    for c in ("enrollmentstartdate", "enrollmentenddate"):
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")

    ms_d, me_d = ms, me
    for _, r in df.iterrows():
        s = r.get("enrollmentstartdate")
        e = r.get("enrollmentenddate")
        s = s if pd.notna(s) else pd.Timestamp.min
        e = e if pd.notna(e) else pd.Timestamp.max
        try:
            s_d = s.date()
            e_d = e.date()
        except Exception:
            continue
        if s_d <= ms_d and e_d >= me_d:
            return True
    return False

def _latest_emp_cost_for_month(el_df: pd.DataFrame, ms, me) -> Optional[float]:
    """Return the latest EMP-tier plancost applicable in [ms, me]; fallback to any tier if EMP not present."""
    if el_df is None or el_df.empty:
        return None

    df = el_df.copy()
    # normalize dates
    for c in ("eligibilitystartdate", "eligibilityenddate"):
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # rows that touch the month
    end_series   = _safe_series(df, "eligibilityenddate").fillna(pd.Timestamp.max)
    start_series = _safe_series(df, "eligibilitystartdate").fillna(pd.Timestamp.min)
    if end_series.empty or start_series.empty:
        return None
    mask = (end_series.dt.date >= ms) & (start_series.dt.date <= me)
    df = df[mask]
    if df.empty or ("plancost" not in df.columns):
        return None

    # Prefer EMP tier when we can
    if "eligibilitytier" in df.columns:
        tnorm = _series_str_upper_strip(_safe_series(df, "eligibilitytier")).str.replace(r"[^A-Z]", "", regex=True)
        emp_aliases = (TIER_ALIASES.get("EE", set()) | {"EE"})
        emp_pref_mask = tnorm.isin(emp_aliases)
        df_emp = df[emp_pref_mask] if emp_pref_mask.any() else df
    else:
        df_emp = df

    sort_col = "eligibilitystartdate" if "eligibilitystartdate" in df_emp.columns else None
    if sort_col:
        df_emp = df_emp.sort_values(by=sort_col, kind="stable")

    try:
        return float(df_emp["plancost"].iloc[-1])
    except Exception:
        return None

def _month_line14(
    eligible_mv: bool,
    offer_ee_allmonth: bool,
    offer_spouse: bool,
    offer_dependents: bool,
    affordable: bool
) -> str:
    """
    Simplified (UAT) monthly Line 14:
      - No full-month offer to EE → 1H
      - Offer not MV → 1F
      - MV + spouse + dependents → 1A if affordable else 1E
      - Otherwise 1E
    """
    if not offer_ee_allmonth:
        return "1H"
    if not eligible_mv:
        return "1F"
    if offer_spouse and offer_dependents:
        return "1A" if affordable else "1E"
    return "1E"

def _month_line16(
    *,
    employed: bool,
    enrolled_full: bool,
    waiting: bool,
    ft: bool,
    offer_ee_allmonth: bool,
    affordable: bool
) -> str:
    """
    Simplified (UAT) Line 16:
      2C: enrolled full month
      2A: not employed any day
      2D: waiting period
      2B: not FT any day
      2H: affordable safe harbor (offered all month, not enrolled)
      else: blank
    """
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

# -----------------------
# Public builders
# -----------------------

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
    Build the Interim sheet (12 rows per employee).
    - spouse_enrolled / child_enrolled are True only when an EMPFAM enrollment
      covers the full month (non-WAIVE).
    """
    with log_time(log, "build_interim"):
        emp_demo   = _apply_aliases(_df_or_empty(emp_demo))
        emp_elig   = _apply_aliases(_df_or_empty(emp_elig))
        emp_enroll = _apply_aliases(_df_or_empty(emp_enroll))
        dep_enroll = _apply_aliases(_df_or_empty(dep_enroll))
        emp_wait   = _df_or_empty(emp_wait)

        # normalize critical date columns
        for df, sc, ec in (
            (emp_elig, "eligibilitystartdate", "eligibilityenddate"),
            (emp_enroll, "enrollmentstartdate", "enrollmentenddate"),
            (dep_enroll, "enrollmentstartdate", "enrollmentenddate"),
        ):
            if not df.empty:
                for c in (sc, ec):
                    if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
                        df[c] = pd.to_datetime(df[c], errors="coerce")

        st = _status_from_demographic(emp_demo)
        employee_ids = _collect_employee_ids(emp_demo, st, emp_elig, emp_enroll, dep_enroll, emp_wait)
        rows: List[Dict[str, Any]] = []
        thresh = AFFORDABILITY_THRESHOLD_DEFAULT if affordability_threshold is None else float(affordability_threshold)

        def _is_employed_month(st_emp: pd.DataFrame, ms, me) -> bool:
            if st_emp is None or st_emp.empty:
                return False
            for c in ("statusstartdate", "statusenddate"):
                if c in st_emp.columns and not pd.api.types.is_datetime64_any_dtype(st_emp[c]):
                    st_emp[c] = pd.to_datetime(st_emp[c], errors="coerce")
            overlaps = (_safe_series(st_emp, "statusenddate").fillna(pd.Timestamp.max).dt.date >= ms) & \
                       (_safe_series(st_emp, "statusstartdate").fillna(pd.Timestamp.min).dt.date <= me)
            if not overlaps.any():
                return False
            if "_estatus_norm" in st_emp.columns:
                s = _series_str_upper_strip(_safe_series(st_emp, "_estatus_norm"))
                any_term = s.str.contains("TERMINAT", na=False) | s.str.fullmatch("TERM", na=False)
                return not bool(any_term.any())
            return True

        def _is_ft(st_emp: pd.DataFrame, ms, me) -> bool:
            if st_emp.empty or "_role_norm" not in st_emp.columns:
                return False
            if "_estatus_norm" in st_emp.columns:
                overlaps = (_safe_series(st_emp, "statusenddate").fillna(pd.Timestamp.max).dt.date >= ms) & \
                           (_safe_series(st_emp, "statusstartdate").fillna(pd.Timestamp.min).dt.date <= me)
                if overlaps.any():
                    s = _series_str_upper_strip(_safe_series(st_emp, "_estatus_norm"))
                    any_term = s.str.contains("TERMINAT", na=False) | s.str.fullmatch("TERM", na=False)
                    if any_term.any():
                        return False
            s = _series_str_upper_strip(_safe_series(st_emp, "_role_norm"))
            mask = s.str.contains("FULLTIME", na=False) | s.str.fullmatch("FT", na=False)
            return _all_month(st_emp, "statusstartdate", "statusenddate", ms, me, mask=mask)

        def _is_pt(st_emp: pd.DataFrame, ms, me) -> bool:
            if st_emp.empty or "_role_norm" not in st_emp.columns:
                return False
            if "_estatus_norm" in st_emp.columns:
                overlaps = (_safe_series(st_emp, "statusenddate").fillna(pd.Timestamp.max).dt.date >= ms) & \
                           (_safe_series(st_emp, "statusstartdate").fillna(pd.Timestamp.min).dt.date <= me)
                if overlaps.any():
                    s = _series_str_upper_strip(_safe_series(st_emp, "_estatus_norm"))
                    any_term = s.str.contains("TERMINAT", na=False) | s.str.fullmatch("TERM", na=False)
                    if any_term.any():
                        return False
            s = _series_str_upper_strip(_safe_series(st_emp, "_role_norm"))
            mask = s.str.contains("PARTTIME", na=False) | s.str.fullmatch("PT", na=False)
            return _all_month(st_emp, "statusstartdate", "statusenddate", ms, me, mask=mask)

        # Precompute a non-WAIVE mask in enrollment for EMPFAM full-month checks
        en_non_waive = emp_enroll.copy()
        if not en_non_waive.empty:
            code = _series_str_upper_strip(_safe_series(en_non_waive, "plancode"))
            name = _series_str_upper_strip(_safe_series(en_non_waive, "planname"))
            not_waive_mask = ~(code.eq("WAIVE") | name.eq("WAIVE"))
            en_non_waive = en_non_waive[not_waive_mask]

        for emp in employee_ids:
            el_emp = emp_elig[emp_elig["employeeid"].astype(str) == str(emp)].copy() if not emp_elig.empty else pd.DataFrame()
            en_emp = emp_enroll[emp_enroll["employeeid"].astype(str) == str(emp)].copy() if not emp_enroll.empty else pd.DataFrame()
            en_emp_nonwaive = en_non_waive[en_non_waive["employeeid"].astype(str) == str(emp)].copy() if not en_non_waive.empty else pd.DataFrame()
            st_emp = st[st["employeeid"].astype(str) == str(emp)].copy() if not st.empty else pd.DataFrame()
            wt_emp = emp_wait[emp_wait["employeeid"].astype(str) == str(emp)].copy() if not emp_wait.empty else pd.DataFrame()

            for m in range(1, 12 + 1):
                ms, me = month_bounds(year, m)

                employed = _is_employed_month(st_emp, ms, me)
                ft = _is_ft(st_emp, ms, me)
                parttime = (not ft) and _is_pt(st_emp, ms, me)

                elig_any  = _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me) if not el_emp.empty else False
                elig_full = _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me) if not el_emp.empty else False

                # Minimal MV inference (UAT): PlanCode == 'PLANA' → MV
                eligible_mv = False
                if not el_emp.empty and "plancode" in el_emp.columns:
                    mask_mv = _series_str_upper_strip(_safe_series(el_emp, "plancode")).eq("PLANA")
                    eligible_mv = _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me, mask=mask_mv)

                offer_ee_allmonth = _offered_allmonth(el_emp, ms, me)

                offer_spouse = _tier_offered_any(
                    el_emp, "eligibilitytier", ("EMPSPOUSE",), "eligibilitystartdate", "eligibilityenddate", ms, me
                ) or _tier_offered_any(
                    en_emp, "tier", ("EMPSPOUSE",), "enrollmentstartdate", "enrollmentenddate", ms, me, require_enrolled=True
                )

                offer_dependents = _tier_offered_any(
                    el_emp, "eligibilitytier", ("EMPCHILD", "EMPFAM"), "eligibilitystartdate", "eligibilityenddate", ms, me
                ) or _tier_offered_any(
                    en_emp, "tier", ("EMPCHILD", "EMPFAM"), "enrollmentstartdate", "enrollmentenddate", ms, me, require_enrolled=True
                )

                enrolled_full = _enrolled_full_month_union(en_emp, ms, me)

                emp_cost = _latest_emp_cost_for_month(el_emp, ms, me)
                affordable = (emp_cost is not None) and (emp_cost <= thresh)

                waiting = _waiting_in_month(wt_emp, ms, me)  # ONLY from Emp Wait Period

                l14 = _month_line14(eligible_mv, offer_ee_allmonth, offer_spouse, offer_dependents, affordable)
                if (not bool(elig_any)) and bool(enrolled_full):
                    # Not eligible but enrolled full → treat as 1E offer
                    l14 = "1E"

                l16 = _month_line16(
                    employed=bool(employed),
                    enrolled_full=bool(enrolled_full),
                    waiting=bool(waiting),
                    ft=bool(ft),
                    offer_ee_allmonth=bool(offer_ee_allmonth),
                    affordable=bool(affordable),
                )

                # === EMPFAM full-month → spouse_enrolled & child_enrolled ===
                spouse_enrolled = False
                child_enrolled = False
                if not en_emp_nonwaive.empty and "tier" in en_emp_nonwaive.columns:
                    tnorm = _series_str_upper_strip(_safe_series(en_emp_nonwaive, "tier"))
                    mask_empfam = tnorm.eq("EMPFAM")
                    if mask_empfam.any():
                        sub = en_emp_nonwaive.loc[mask_empfam].copy()
                        # ensure dates are datetimes
                        for c in ("enrollmentstartdate", "enrollmentenddate"):
                            if c in sub.columns and not pd.api.types.is_datetime64_any_dtype(sub[c]):
                                sub[c] = pd.to_datetime(sub[c], errors="coerce")
                        # full-month coverage check
                        full_mask = (
                            _safe_series(sub, "enrollmentstartdate").fillna(pd.Timestamp.min).dt.date.le(ms) &
                            _safe_series(sub, "enrollmentenddate").fillna(pd.Timestamp.max).dt.date.ge(me)
                        )
                        if bool(full_mask.any()):
                            spouse_enrolled = True
                            child_enrolled  = True
                # =================================================================

                rows.append({
                    "EmployeeID": str(emp),
                    "Year": int(year),
                    "MonthNum": int(m),
                    "Month": MONTHS[m-1],
                    "MonthStart": ms,
                    "MonthEnd": me,
                    "employed": bool(employed),
                    "ft": bool(ft),
                    "parttime": bool(parttime),
                    "eligibleforcoverage": bool(elig_any),
                    "eligible_allmonth": bool(elig_full),
                    "eligible_mv": bool(eligible_mv),
                    "offer_ee_allmonth": bool(offer_ee_allmonth),
                    "enrolled_allmonth": bool(enrolled_full),
                    "offer_spouse": bool(offer_spouse),
                    "offer_dependents": bool(offer_dependents),
                    "spouse_eligible": bool(offer_spouse),
                    "child_eligible": bool(offer_dependents),
                    "spouse_enrolled": bool(spouse_enrolled),
                    "child_enrolled": bool(child_enrolled),
                    "waitingperiod_month": bool(waiting),
                    "affordable_plan": bool(affordable),
                    "line14_final": l14,
                    "line16_final": l16,
                    "line14_all12": "",
                })

        interim = pd.DataFrame.from_records(rows)

        # Year-level 1G: never FT in the year, but enrolled for at least one full month
        one_g_emp_ids = []
        if not interim.empty:
            for emp in interim["EmployeeID"].unique().tolist():
                g = interim[interim["EmployeeID"] == emp].sort_values("MonthNum")
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
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            "Line16_Jan", "Line16_Feb", "Line16_Mar", "Line16_Apr", "Line16_May", "Line16_Jun",
            "Line16_Jul", "Line16_Aug", "Line16_Sep", "Line16_Oct", "Line16_Nov", "Line16_Dec",
            "Line14_All12"
        ]
        if interim is None or interim.empty:
            return pd.DataFrame(columns=cols)

        df = interim.copy()

        # Line 14 by month
        df["Jan"] = df["line14_final"].where(df["Month"] == "Jan")
        df["Feb"] = df["line14_final"].where(df["Month"] == "Feb")
        df["Mar"] = df["line14_final"].where(df["Month"] == "Mar")
        df["Apr"] = df["line14_final"].where(df["Month"] == "Apr")
        df["May"] = df["line14_final"].where(df["Month"] == "May")
        df["Jun"] = df["line14_final"].where(df["Month"] == "Jun")
        df["Jul"] = df["line14_final"].where(df["Month"] == "Jul")
        df["Aug"] = df["line14_final"].where(df["Month"] == "Aug")
        df["Sep"] = df["line14_final"].where(df["Month"] == "Sep")
        df["Oct"] = df["line14_final"].where(df["Month"] == "Oct")
        df["Nov"] = df["line14_final"].where(df["Month"] == "Nov")
        df["Dec"] = df["line14_final"].where(df["Month"] == "Dec")

        # Line 16 by month
        df["Line16_Jan"] = df["line16_final"].where(df["Month"] == "Jan")
        df["Line16_Feb"] = df["line16_final"].where(df["Month"] == "Feb")
        df["Line16_Mar"] = df["line16_final"].where(df["Month"] == "Mar")
        df["Line16_Apr"] = df["line16_final"].where(df["Month"] == "Apr")
        df["Line16_May"] = df["line16_final"].where(df["Month"] == "May")
        df["Line16_Jun"] = df["line16_final"].where(df["Month"] == "Jun")
        df["Line16_Jul"] = df["line16_final"].where(df["Month"] == "Jul")
        df["Line16_Aug"] = df["line16_final"].where(df["Month"] == "Aug")
        df["Line16_Sep"] = df["line16_final"].where(df["Month"] == "Sep")
        df["Line16_Oct"] = df["line16_final"].where(df["Month"] == "Oct")
        df["Line16_Nov"] = df["line16_final"].where(df["Month"] == "Nov")
        df["Line16_Dec"] = df["line16_final"].where(df["Month"] == "Dec")

        final_rows = []
        for emp, g in df.groupby("EmployeeID", sort=False):
            rec = {"EmployeeID": emp, "Year": int(g["Year"].iloc[0])}
            # Month columns (Line 14)
            for m_full in FULL_MONTHS:
                vals = g.loc[g["Month"] == m_full[:3], "line14_final"].dropna()
                rec[m_full] = vals.iloc[0] if not vals.empty else ""
            # Line16 columns
            for m_full in FULL_MONTHS:
                vals = g.loc[g["Month"] == m_full[:3], "line16_final"].dropna()
                rec[f"Line16_{m_full}"] = vals.iloc[0] if not vals.empty else ""
            # 1G handling
            rec["Line14_All12"] = g.loc[g["line14_all12"] == "1G", "line14_all12"].iloc[0] if (g["line14_all12"] == "1G").any() else ""
            final_rows.append(rec)

        out = pd.DataFrame.from_records(final_rows, columns=cols)
        log_df(log, "final", out)
        return out


def build_penalty_dashboard(interim: pd.DataFrame) -> pd.DataFrame:
    with log_time(log, "build_penalty_dashboard"):
        if interim is None or interim.empty:
            return pd.DataFrame(columns=["EmployeeID", "Reason"] + FULL_MONTHS)

        def month_reason(row: pd.Series) -> str:
            if not row.get("eligibleforcoverage", False):
                return "Not eligible"
            if not row.get("offer_ee_allmonth", False):
                return "No full-month offer"
            if not row.get("eligible_mv", False):
                return "Offer not MV (1F)"
            if not row.get("affordable_plan", False):
                return "Offered but not affordable (B)"
            if not row.get("enrolled_allmonth", False):
                return "Offered not enrolled"
            return "–"

        df = interim.copy()
        out_rows = []
        for emp, g in df.groupby("EmployeeID", sort=False):
            rec = {"EmployeeID": emp}
            reasons = [month_reason(r) for _, r in g.sort_values("MonthNum").iterrows()]
            for i, m in enumerate(FULL_MONTHS):
                rec[m] = reasons[i] if i < len(reasons) else "–"
            rec["Reason"] = next((x for x in reasons if x != "–"), "–")
            out_rows.append(rec)

        cols = ["EmployeeID", "Reason"] + FULL_MONTHS
        return pd.DataFrame.from_records(out_rows, columns=cols)
