# aca_builder.py
# Build a rich Interim grid (employee x month) and a Final (employee-month Line14/Line16) table.
# Also compute a Penalty Dashboard summary.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# -------------------------------
# Config
# -------------------------------

AFFORDABILITY_DEFAULT = 50.0  # employee-only cost/month threshold
PENALTY_A_MONTHLY = 241.67
PENALTY_B_MONTHLY = 362.50

MONTHS = [
    ("Jan", 1), ("Feb", 2), ("Mar", 3), ("Apr", 4), ("May", 5), ("Jun", 6),
    ("Jul", 7), ("Aug", 8), ("Sep", 9), ("Oct", 10), ("Nov", 11), ("Dec", 12),
]

ALLOWED_MV_PLANS = {"PLANA"}  # normalized check


# -------------------------------
# Helpers
# -------------------------------

def _norm(s: Optional[str]) -> str:
    if pd.isna(s):
        return ""
    return str(s).strip().upper()


def _in_any(v: Optional[str], options: Iterable[str]) -> bool:
    if v is None:
        return False
    vv = _norm(v)
    return any(vv == _norm(o) for o in options)


def _overlap_full_month(start: pd.Timestamp, end: pd.Timestamp,
                        ms: pd.Timestamp, me: pd.Timestamp) -> bool:
    """True if [start, end] fully covers [ms, me]."""
    if pd.isna(start) or pd.isna(end):
        return False
    return (start <= ms) and (end >= me)


def _overlap_any(start: pd.Timestamp, end: pd.Timestamp,
                 ms: pd.Timestamp, me: pd.Timestamp) -> bool:
    """True if [start, end] overlaps [ms, me] by any day."""
    if pd.isna(start) or pd.isna(end):
        return False
    return not (end < ms or start > me)


def _any_overlap(df: pd.DataFrame, start_col: str, end_col: str,
                 ms: pd.Timestamp, me: pd.Timestamp) -> bool:
    if df.empty:
        return False
    for _, r in df.iterrows():
        if _overlap_any(r[start_col], r[end_col], ms, me):
            return True
    return False


def _all_month(df: pd.DataFrame, start_col: str, end_col: str,
               ms: pd.Timestamp, me: pd.Timestamp) -> bool:
    if df.empty:
        return False
    for _, r in df.iterrows():
        if _overlap_full_month(r[start_col], r[end_col], ms, me):
            return True
    return False


def _union_intervals_full_month(df: pd.DataFrame, start_col: str, end_col: str,
                                ms: pd.Timestamp, me: pd.Timestamp) -> bool:
    """
    Consider multiple intervals; does their union cover the entire [ms, me] month?
    Only intervals overlapping the month contribute.
    """
    if df.empty:
        return False

    segs: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for _, r in df.iterrows():
        s = r[start_col]; e = r[end_col]
        if pd.isna(s) or pd.isna(e):
            continue
        if not _overlap_any(s, e, ms, me):
            continue
        segs.append((max(s, ms), min(e, me)))

    if not segs:
        return False

    segs.sort(key=lambda x: x[0])
    merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur_s, cur_e = segs[0]
    for s, e in segs[1:]:
        if s <= (cur_e + pd.Timedelta(days=1)):
            if e > cur_e:
                cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    coverage = pd.Timedelta(0)
    need = (me - ms) + pd.Timedelta(days=1)
    for s, e in merged:
        coverage += (e - s) + pd.Timedelta(days=1)
        if coverage >= need:
            return True
    return False


def _value_latest_overlap(df: pd.DataFrame, start_col: str, end_col: str,
                          ms: pd.Timestamp, me: pd.Timestamp, value_col: str) -> Optional[float]:
    """From rows overlapping [ms, me], pick the latest start date and return value_col as float."""
    if df.empty:
        return None

    overl = df[~(df[end_col] < ms) & ~(df[start_col] > me)].copy()
    if overl.empty:
        return None

    if start_col not in overl.columns:
        return None

    overl = overl.sort_values(by=start_col, ascending=False)
    for _, r in overl.iterrows():
        v = r.get(value_col, None)
        if pd.isna(v):
            continue
        try:
            return float(v)
        except Exception:
            continue
    return None


@dataclass
class MonthWindow:
    ym: Tuple[int, int]  # (year, month)
    start: pd.Timestamp
    end: pd.Timestamp


# -------------------------------
# Core build
# -------------------------------

def build_interim(
    year: int,
    demo: pd.DataFrame,
    status: pd.DataFrame,
    elig: pd.DataFrame,
    enroll_emp: pd.DataFrame,
    enroll_dep: pd.DataFrame,
    affordability_threshold: float = AFFORDABILITY_DEFAULT,
) -> pd.DataFrame:
    """Return a rich employee-month grid with flags used to derive Final and PDF."""

    def month_windows(y: int) -> List[MonthWindow]:
        out = []
        for _, m in MONTHS:
            ms = pd.Timestamp(year=y, month=m, day=1)
            me = (pd.Timestamp(year=y + (m == 12), month=(m % 12) + 1, day=1)
                  - pd.Timedelta(days=1)) if m != 12 else pd.Timestamp(year=y, month=12, day=31)
            out.append(MonthWindow((y, m), ms, me))
        return out

    # Normalize date columns
    for df in (demo, status, elig, enroll_emp, enroll_dep):
        if not df.empty:
            for c in df.columns:
                if "date" in c.lower():
                    df[c] = pd.to_datetime(df[c], errors="coerce")

    rows: List[Dict] = []

    # Group by employee
    demo_by_emp = dict(tuple(demo.groupby("employeeid"))) if not demo.empty else {}
    status_by_emp = dict(tuple(status.groupby("employeeid"))) if not status.empty else {}
    elig_by_emp = dict(tuple(elig.groupby("employeeid"))) if not elig.empty else {}
    en_by_emp = dict(tuple(enroll_emp.groupby("employeeid"))) if not enroll_emp.empty else {}
    dep_by_emp = dict(tuple(enroll_dep.groupby("employeeid"))) if not enroll_dep.empty else {}

    for emp_id, demo_rows in demo_by_emp.items():
        en_emp = en_by_emp.get(emp_id, pd.DataFrame())
        en_dep = dep_by_emp.get(emp_id, pd.DataFrame())
        st_emp = status_by_emp.get(emp_id, pd.DataFrame())
        el_emp = elig_by_emp.get(emp_id, pd.DataFrame())

        # Demographic
        first = demo_rows.iloc[0].get("firstname", "")
        middle = demo_rows.iloc[0].get("middlename", "")
        last = demo_rows.iloc[0].get("lastname", "")
        ssn = demo_rows.iloc[0].get("ssn", "")
        addr1 = demo_rows.iloc[0].get("addressline1", "")
        addr2 = demo_rows.iloc[0].get("addressline2", "")
        city = demo_rows.iloc[0].get("city", "")
        state = demo_rows.iloc[0].get("state", "")
        zipcode = demo_rows.iloc[0].get("zipcode", "")

        for win in month_windows(year):
            y, m = win.ym
            ms, me = win.start, win.end

            # Employment presence
            employed_any = _any_overlap(st_emp, "statusstartdate", "statusenddate", ms, me)

            terminated_this_month = False
            if not st_emp.empty:
                for _, r in st_emp.iterrows():
                    role = _norm(r.get("employmentstatus", r.get("status", "")))
                    if role in ("TERM", "TERMINATED"):
                        if _overlap_any(r.get("statusstartdate"), r.get("statusenddate"), ms, me):
                            terminated_this_month = True
                            break

            employed = employed_any and not terminated_this_month

            # FT/PT full-month
            ft_all = False; pt_all = False
            if not st_emp.empty:
                ft_rows = st_emp[
                    st_emp.apply(lambda r: _norm(r.get("role", r.get("employmenttype", ""))) in ("FT", "FULLTIME"), axis=1)
                ]
                pt_rows = st_emp[
                    st_emp.apply(lambda r: _norm(r.get("role", r.get("employmenttype", ""))) in ("PT", "PARTTIME"), axis=1)
                ]
                ft_all = _all_month(ft_rows, "statusstartdate", "statusenddate", ms, me)
                pt_all = _all_month(pt_rows, "statusstartdate", "statusenddate", ms, me)

            # Eligibility rows
            elig_emp = el_emp.copy()
            elig_emp["tier"] = elig_emp.get("eligibilitytier", "").map(_norm)
            elig_emp["plancode_norm"] = elig_emp.get("plancode", "").map(_norm)

            ee_elig_rows = elig_emp[elig_emp["tier"].isin(["EMP", "EMPONLY", "EMPLOYEE"])].copy()
            spouse_elig_rows = elig_emp[elig_emp["tier"].isin(
                ["EMPSPOUSE", "SPOUSE", "EMP+SPOUSE", "EMP_SPOUSE", "EMP-SPOUSE", "EMPFAM"]
            )].copy()
            child_elig_rows = elig_emp[elig_emp["tier"].isin(
                ["EMPCHILD", "CHILD", "EMP+CHILD", "EMP_CHILD", "EMP-CHILD", "EMPFAM"]
            )].copy()

            # MV offer (PlanA full-month in EMP tiers)
            mv_rows = ee_elig_rows[ee_elig_rows["plancode_norm"].isin(ALLOWED_MV_PLANS)].copy()
            mv_offer_full = _all_month(mv_rows, "eligibilitystartdate", "eligibilityenddate", ms, me)

            # Employee offer (EMP tier full-month)
            ee_offer_full = _all_month(ee_elig_rows, "eligibilitystartdate", "eligibilityenddate", ms, me)

            # Enrollment prep
            en_emp_work = en_emp.copy()
            en_emp_work["tier"] = en_emp_work.get("enrollmenttier", "").map(_norm)

            def _tier_enrolled_full_month(df: pd.DataFrame, tier_options: Tuple[str, ...],
                                          ms: pd.Timestamp, me: pd.Timestamp) -> bool:
                if df.empty:
                    return False
                work = df[df["tier"].isin([_norm(t) for t in tier_options])].copy()
                # Exclude WAIVE rows
                work = work[~work.get("plancode", "").map(_norm).eq("WAIVE")]
                if work.empty:
                    return False
                if "isenrolled" in work.columns:
                    work = work[work["isenrolled"].fillna(True) == True]
                    if work.empty:
                        return False
                return _union_intervals_full_month(work, "enrollmentstartdate", "enrollmentenddate", ms, me)

            enrolled_full = _tier_enrolled_full_month(
                en_emp_work, ("EMP", "EMPFAM", "EMPSPOUSE", "EMPCHILD"), ms, me
            )

            # Spouse/child eligibility
            spouse_eligible = _all_month(spouse_elig_rows, "eligibilitystartdate", "eligibilityenddate", ms, me)
            child_eligible  = _all_month(child_elig_rows,  "eligibilitystartdate", "eligibilityenddate", ms, me)

            # Spouse/child enrolled (by tiers)
            spouse_enrolled = _tier_enrolled_full_month(en_emp, ("EMPFAM", "EMPSPOUSE"), ms, me)
            child_enrolled  = _tier_enrolled_full_month(en_emp, ("EMPFAM", "EMPCHILD"),  ms, me)

            # === Your requested override: EMPFAM full-month → both spouse & child enrolled ===
            if _tier_enrolled_full_month(en_emp, ("EMPFAM",), ms, me):
                spouse_enrolled = True
                child_enrolled = True
            # ==============================================================================

            # Offers (elig OR enroll)
            offer_spouse = spouse_eligible or spouse_enrolled
            offer_dependents = child_eligible or child_enrolled

            # Affordability (employee-only cost)
            aff_cost = _value_latest_overlap(
                ee_elig_rows, "eligibilitystartdate", "eligibilityenddate", ms, me, "plancost"
            )
            affordable = False
            if aff_cost is not None:
                affordable = (float(aff_cost) < float(affordability_threshold))

            # Waiting period
            waiting_period_month = False
            if employed:
                have_elig_now = _any_overlap(ee_elig_rows, "eligibilitystartdate", "eligibilityenddate", ms, me)
                if not have_elig_now and not ee_elig_rows.empty:
                    min_future = ee_elig_rows["eligibilitystartdate"].min()
                    if pd.notna(min_future) and min_future > me:
                        waiting_period_month = True

            # Line 14
            line14 = ""
            have_elig_now_any_tier = _any_overlap(elig_emp, "eligibilitystartdate", "eligibilityenddate", ms, me)
            if enrolled_full and not have_elig_now_any_tier:
                line14 = "1E"
            else:
                if not ee_offer_full and not mv_offer_full:
                    line14 = "1H"
                else:
                    if mv_offer_full:
                        if offer_spouse and offer_dependents:
                            line14 = "1A" if affordable else "1E"
                        elif offer_spouse and not offer_dependents:
                            line14 = "1D"
                        elif (not offer_spouse) and offer_dependents:
                            line14 = "1C"
                        else:
                            line14 = "1B"
                    else:
                        line14 = "1F"

            # Line 16 precedence
            line16 = ""
            if enrolled_full:
                line16 = "2C"
            elif not employed:
                line16 = "2A"
            elif waiting_period_month:
                line16 = "2D"
            elif not ft_all:
                line16 = "2B"
            elif affordable and (line14 in ("1A", "1B", "1C", "1D", "1E")):
                line16 = "2H"
            else:
                line16 = ""

            rows.append(dict(
                employeeid=emp_id, year=y, month=m,
                employed=bool(employed), ft_all=bool(ft_all), pt_all=bool(pt_all),

                offer_employee_full=bool(ee_offer_full),
                offer_mv_full=bool(mv_offer_full),

                spouse_eligible=bool(spouse_eligible),
                child_eligible=bool(child_eligible),
                spouse_enrolled=bool(spouse_enrolled),
                child_enrolled=bool(child_enrolled),

                offer_spouse=bool(offer_spouse),
                offer_dependents=bool(offer_dependents),

                enrolled_full=bool(enrolled_full),
                affordability_cost=aff_cost if aff_cost is not None else None,
                affordable=bool(affordable),
                waiting_period_month=bool(waiting_period_month),

                line14=line14, line16=line16,

                firstname=first, middlename=middle, lastname=last, ssn=ssn,
                addressline1=addr1, addressline2=addr2, city=city, state=state, zipcode=zipcode,
            ))

    interim = pd.DataFrame(rows)

    # Employee-level 1G
    g_rows = []
    for emp_id, sub in interim.groupby("employeeid"):
        never_ft = not sub["ft_all"].any()
        enrolled_any = sub["enrolled_full"].any()
        line14_all12 = "1G" if (never_ft and enrolled_any) else ""
        g_rows.append({"employeeid": emp_id, "line14_all12": line14_all12})
    gdf = pd.DataFrame(g_rows)

    interim = interim.merge(gdf, on="employeeid", how="left")
    mask_1g = interim["line14_all12"].eq("1G")
    interim.loc[mask_1g, "line14"] = ""

    return interim


def build_final(interim: pd.DataFrame, year: int) -> pd.DataFrame:
    """Final table: EmployeeID, Month, Line14_Final, Line16_Final (+ All12 columns)"""
    rows: List[Dict] = []
    for _, r in interim.iterrows():
        rows.append(dict(
            employeeid=r["employeeid"], month=r["month"],
            line14=r.get("line14", ""), line16=r.get("line16", "")
        ))
    final = pd.DataFrame(rows)

    # All-12 identical derivation
    def _all12_identical(df: pd.DataFrame, col: str) -> str:
        vals = [str(v or "") for v in df[col].tolist()]
        uniq = set(vals)
        return list(uniq)[0] if len(uniq) == 1 else ""

    all12_rows = []
    for emp_id, sub in final.groupby("employeeid"):
        all12_l14 = _all12_identical(sub, "line14")
        all12_l16 = _all12_identical(sub, "line16")
        all12_rows.append({"employeeid": emp_id, "all12_line14": all12_l14, "all12_line16": all12_l16})
    all12_df = pd.DataFrame(all12_rows)

    agg = interim.groupby("employeeid").agg(line14_all12=("line14_all12", "first")).reset_index()
    final = final.merge(agg, on="employeeid", how="left").merge(all12_df, on="employeeid", how="left")

    final = final[(final["month"] >= 1) & (final["month"] <= 12)].copy()
    final["year"] = year
    return final


def build_penalty_dashboard(interim: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Penalty A if no MEC offer; Penalty B if offered but not enrolled and unaffordable; else '-'
    """
    rows: List[Dict] = []
    for emp_id, sub in interim.groupby("employeeid"):
        totals = {m: "-" for m in range(1, 13)}
        reasons = {m: "" for m in range(1, 13)}
        for _, r in sub.iterrows():
            m = int(r["month"])
            offer = bool(r["offer_employee_full"])
            enrolled = bool(r["enrolled_full"])
            affordable = bool(r["affordable"])
            employed = bool(r["employed"])
            waiting = bool(r["waiting_period_month"])

            if not offer:
                if employed and not waiting:
                    totals[m] = PENALTY_A_MONTHLY; reasons[m] = "No MEC offer"
                elif waiting:
                    totals[m] = "-"; reasons[m] = "Waiting period"
                else:
                    totals[m] = "-"; reasons[m] = "Not employed"
            else:
                if enrolled:
                    totals[m] = "-"; reasons[m] = "Enrolled"
                else:
                    if affordable:
                        totals[m] = "-"; reasons[m] = "Affordable offer declined"
                    else:
                        totals[m] = PENALTY_B_MONTHLY; reasons[m] = "Unaffordable + not enrolled"

        rows.append(dict(
            employeeid=emp_id,
            **{f"m{m:02d}": totals[m] for m in range(1, 13)},
            **{f"reason_{m:02d}": reasons[m] for m in range(1, 13)},
            year=year,
        ))
    return pd.DataFrame(rows)
