# aca_builder.py
from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
from datetime import timedelta
import pandas as pd

from debug_logging import get_logger, log_df, log_time, log_call
log = get_logger("builder")

from aca_processing import (
    MONTHS, FULL_MONTHS,
    _collect_employee_ids, month_bounds,
    _any_overlap, _all_month,
    _status_from_demographic, _waiting_in_month,
)

AFFORDABILITY_THRESHOLD_DEFAULT = 50.0  # used in Simplified (UAT) mode

# -----------------------
# Utilities / Aliases
# -----------------------

ALIASES = {
    # common column aliasing across varied customer spreadsheets
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


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    cols = {c: c for c in x.columns}
    for canon, alts in ALIASES.items():
        for c in list(cols):
            if c.lower().strip() == canon:
                cols[c] = canon
            elif c.lower().strip() in {a.lower() for a in alts}:
                cols[c] = canon
    x = x.rename(columns=cols)
    # normalize common string columns
    for c in ("employeeid", "plancode", "planname", "eligibilitytier", "tier"):
        if c in x.columns:
            x[c] = x[c].astype(str).str.strip()
    return x


def _df_or_empty(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _offered_allmonth(el_df: pd.DataFrame, ms, me) -> bool:
    """Any eligibility row (any tier) covering the full month → offer_ee_allmonth."""
    if el_df is None or el_df.empty:
        return False
    return _all_month(el_df, "eligibilitystartdate", "eligibilityenddate", ms, me)


def _tier_offered_any(df: pd.DataFrame, tier_col: str, tiers: Tuple[str, ...], start_col: str, end_col: str, ms, me, require_enrolled: bool = False) -> bool:
    if df is None or df.empty or tier_col not in df.columns:
        return False
    tnorm = df[tier_col].astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    # Expand aliases
    tier_set = set()
    for t in tiers:
        t_up = t.upper()
        tier_set.add(t_up)
        for canon, alts in TIER_ALIASES.items():
            if t_up == canon:
                tier_set |= {a.upper() for a in alts}
    mask = tnorm.isin(tier_set)
    if require_enrolled and "isenrolled" in df.columns:
        mask &= df["isenrolled"].astype(bool)
    return _any_overlap(df, start_col, end_col, ms, me, mask=mask)


def _enrolled_full_month(el_df: pd.DataFrame, ms, me) -> bool:
    if el_df is None or el_df.empty:
        return False
    mask = pd.Series(True, index=el_df.index)
    if "isenrolled" in el_df.columns:
        mask &= el_df["isenrolled"].astype(bool)
    return _all_month(el_df, "enrollmentstartdate", "enrollmentenddate", ms, me, mask=mask)


def _enrolled_full_month_union(en_df: pd.DataFrame, ms, me) -> bool:
    """Union of any enrollment rows (excluding WAIVE) that cover the full month."""
    if en_df is None or en_df.empty:
        return False
    mask = pd.Series(True, index=en_df.index)
    if "isenrolled" in en_df.columns:
        mask &= en_df["isenrolled"].astype(bool)
    waive_mask = pd.Series(False, index=en_df.index)
    for col in ("plancode", "planname"):
        if col in en_df.columns:
            s = en_df[col].astype(str).str.upper().str.strip()
            waive_mask |= s.eq("WAIVE")
    mask &= ~waive_mask

    df = en_df.loc[mask].copy()
    if df.empty:
        return False
    for c in ("enrollmentstartdate", "enrollmentenddate"):
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")

    intervals = []
    for _, r in df.iterrows():
        s = pd.to_datetime(r.get("enrollmentstartdate"), errors="coerce")
        e = pd.to_datetime(r.get("enrollmentenddate"), errors="coerce")
        if pd.isna(s) and pd.isna(e):
            continue
        s = (s or pd.Timestamp.min).date()
        e = (e or pd.Timestamp.max).date()
        intervals.append((s, e))
    if not intervals:
        return False
    # Need at least one interval that fully covers the month
    ms_d, me_d = ms, me
    for s, e in intervals:
        if s <= ms_d and e >= me_d:
            return True
    return False


def _latest_emp_cost_for_month(el_df: pd.DataFrame, ms, me) -> Optional[float]:
    if el_df is None or el_df.empty:
        return None
    df = el_df.copy()
    for c in ("eligibilitystartdate", "eligibilityenddate"):
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    mask = (df["eligibilityenddate"].fillna(pd.Timestamp.max).dt.date >= ms) & \
           (df["eligibilitystartdate"].fillna(pd.Timestamp.min).dt.date <= me)
    df = df[mask]
    if df.empty or "plancost" not in df.columns:
        return None
    # Prefer EMP tier if present; else take last row by startdate
    tnorm = df.get("eligibilitytier", "").astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    emp_pref = tnorm.isin(TIER_ALIASES["EE"] | {"EE"})
    df_emp = df[emp_pref] if emp_pref.any() else df
    df_emp = df_emp.sort_values(by="eligibilitystartdate", kind="stable")
    return float(df_emp["plancost"].iloc[-1]) if not df_emp.empty else None


def _month_line14(eligible_mv: bool, offer_ee_allmonth: bool, offer_spouse: bool, offer_dependents: bool, affordable: bool) -> str:
    """
    Simplified (UAT) version of monthly Line 14 selection:
    - If no full-month offer to EE → 1H
    - If there is an offer, but not MV → 1F
    - If MV and offered to spouse+dependents and affordable → 1A (else 1E)
    - Else 1E for employee-only (or spouse/child but not both)
    """
    if not offer_ee_allmonth:
        return "1H"
    if not eligible_mv:
        return "1F"
    if offer_spouse and offer_dependents:
        return "1A" if affordable else "1E"
    return "1E"


def _month_line16(*, employed: bool, enrolled_full: bool, waiting: bool, ft: bool, offer_ee_allmonth: bool, affordable: bool) -> str:
    """
    Simplified (UAT) Line 16:
    2C if enrolled for full month
    2A if not employed any day of the month
    2D if waiting period
    2B if not FT any day (or partial-month FT gaps)
    2H if affordable safe harbor (when offered all month but not enrolled)
    else blank
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
    emp_wait: pd.DataFrame | None = None,   # Emp Wait Period
    affordability_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Rules engine. waitingperiod_month comes ONLY from Emp Wait Period."""
    with log_time(log, "build_interim"):
        emp_demo   = _apply_aliases(_df_or_empty(emp_demo))
        emp_elig   = _apply_aliases(_df_or_empty(emp_elig))
        emp_enroll = _apply_aliases(_df_or_empty(emp_enroll))
        dep_enroll = _apply_aliases(_df_or_empty(dep_enroll))
        emp_wait   = _df_or_empty(emp_wait)

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
            overlaps = (st_emp["statusenddate"].fillna(pd.Timestamp.max).dt.date >= ms) & \
                       (st_emp["statusstartdate"].fillna(pd.Timestamp.min).dt.date <= me)
            if not overlaps.any():
                return False
            if "_estatus_norm" in st_emp.columns:
                s = st_emp.loc[overlaps, "_estatus_norm"].astype(str)
                any_term = s.str.contains("TERMINAT", na=False) | s.str.fullmatch("TERM", na=False)
                return not bool(any_term.any())
            return True

        def _is_ft(st_emp: pd.DataFrame, ms, me) -> bool:
            if st_emp.empty or "_role_norm" not in st_emp.columns:
                return False
            if "_estatus_norm" in st_emp.columns:
                overlaps = (st_emp["statusenddate"].fillna(pd.Timestamp.max).dt.date >= ms) & \
                           (st_emp["statusstartdate"].fillna(pd.Timestamp.min).dt.date <= me)
                if overlaps.any():
                    s = st_emp.loc[overlaps, "_estatus_norm"].astype(str)
                    any_term = s.str.contains("TERMINAT", na=False) | s.str.fullmatch("TERM", na=False)
                    if any_term.any():
                        return False
            s = st_emp["_role_norm"].astype(str)
            mask = s.str.contains("FULLTIME", na=False) | s.str.fullmatch("FT", na=False)
            return _all_month(st_emp, "statusstartdate", "statusenddate", ms, me, mask=mask)

        def _is_pt(st_emp: pd.DataFrame, ms, me) -> bool:
            if st_emp.empty or "_role_norm" not in st_emp.columns:
                return False
            if "_estatus_norm" in st_emp.columns:
                overlaps = (st_emp["statusenddate"].fillna(pd.Timestamp.max).dt.date >= ms) & \
                           (st_emp["statusstartdate"].fillna(pd.Timestamp.min).dt.date <= me)
                if overlaps.any():
                    s = st_emp.loc[overlaps, "_estatus_norm"].astype(str)
                    any_term = s.str.contains("TERMINAT", na=False) | s.str.fullmatch("TERM", na=False)
                    if any_term.any():
                        return False
            s = st_emp["_role_norm"].astype(str)
            mask = s.str.contains("PARTTIME", na=False) | s.str.fullmatch("PT", na=False)
            return _all_month(st_emp, "statusstartdate", "statusenddate", ms, me, mask=mask)

        for emp in employee_ids:
            el_emp = emp_elig[emp_elig["employeeid"].astype(str) == str(emp)].copy() if not emp_elig.empty else pd.DataFrame()
            en_emp = emp_enroll[emp_enroll["employeeid"].astype(str) == str(emp)].copy() if not emp_enroll.empty else pd.DataFrame()
            st_emp = st[st["employeeid"].astype(str) == str(emp)].copy() if not st.empty else pd.DataFrame()
            wt_emp = emp_wait[emp_wait["employeeid"].astype(str) == str(emp)].copy() if not emp_wait.empty else pd.DataFrame()

            for m in range(1, 13):
                ms, me = month_bounds(year, m)
                employed = _is_employed_month(st_emp, ms, me)
                ft = _is_ft(st_emp, ms, me)
                parttime = (not ft) and _is_pt(st_emp, ms, me)

                elig_any  = _any_overlap(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me) if not el_emp.empty else False
                elig_full = _all_month(el_emp, "eligibilitystartdate", "eligibilityenddate", ms, me) if not el_emp.empty else False

                eligible_mv = False
                if not el_emp.empty and "plancode" in el_emp.columns:
                    mask_mv = el_emp["plancode"].astype(str).str.upper().str.strip().eq("PLANA")
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
                affordable = (emp_cost is not None) and (emp_cost <= (AFFORDABILITY_THRESHOLD_DEFAULT if affordability_threshold is None else float(affordability_threshold)))

                waiting = _waiting_in_month(wt_emp, ms, me)  # ONLY from Emp Wait Period

                l14 = _month_line14(eligible_mv, offer_ee_allmonth, offer_spouse, offer_dependents, affordable)
                if (not bool(elig_any)) and bool(enrolled_full):
                    # If not eligible but enrolled full, treat as 1E (offered to EE at minimum)
                    l14 = "1E"
                l16 = _month_line16(
                    employed=bool(employed),
                    enrolled_full=bool(enrolled_full),
                    waiting=bool(waiting),
                    ft=bool(ft),
                    offer_ee_allmonth=bool(offer_ee_allmonth),
                    affordable=bool(affordable),
                )

                # --- BEGIN: spouse/child enrollment flags from EMPFAM full-month ---
                spouse_enrolled = False
                child_enrolled = False
                try:
                    en_emp_check = en_emp if 'en_emp' in locals() else None
                except Exception:
                    en_emp_check = None
                if en_emp_check is not None and not en_emp_check.empty:
                    df_empfam = en_emp_check.copy()
                    if 'tier' in df_empfam.columns:
                        # normalize tier to detect EMPFAM
                        tnorm = df_empfam['tier'].astype(str).str.strip().str.upper()
                        mask_empfam = tnorm.eq('EMPFAM')
                        if mask_empfam.any():
                            # robust dates
                            for c in ('enrollmentstartdate', 'enrollmentenddate'):
                                if c in df_empfam.columns and not pd.api.types.is_datetime64_any_dtype(df_empfam[c]):
                                    df_empfam[c] = pd.to_datetime(df_empfam[c], errors='coerce')
                            sub = df_empfam.loc[mask_empfam]
                            # Full-month coverage: start <= month_start AND (end is NaT or end >= month_end)
                            empfam_full = ((sub['enrollmentstartdate'].fillna(pd.Timestamp.min).dt.date <= ms) &
                                           (sub['enrollmentenddate'].fillna(pd.Timestamp.max).dt.date >= me)).any()
                            if bool(empfam_full):
                                spouse_enrolled = True
                                child_enrolled = True
                # --- END: spouse/child enrollment flags ---

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

        # Code 1G handling
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
                # collapse row-level marker
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
            # Month columns
            for m in FULL_MONTHS:
                rec[m] = g.loc[g["Month"] == m[:3], "line14_final"].dropna().iloc[0] if not g.loc[g["Month"] == m[:3]].empty else ""
            # Line16 month columns
            for m in FULL_MONTHS:
                col = f"Line16_{m}"
                rec[col] = g.loc[g["Month"] == m[:3], "line16_final"].dropna().iloc[0] if not g.loc[g["Month"] == m[:3]].empty else ""
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
                # Not enrolled, offered, MV, but not affordable
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
