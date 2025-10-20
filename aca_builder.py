# aca_builder.py
from __future__ import annotations
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Tuple, Dict, List

import numpy as np
import pandas as pd

# -------------------- shared constants --------------------
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
A_PENALTY = 241.67  # illustrative monthly A penalty
B_PENALTY = 362.50  # illustrative monthly B penalty

# Recognizers for employment status & roles (normalized uppercase, non-alnum stripped)
FT_TOKENS = {"FT","FULLTIME","FTE","CATEGORY2","CAT2"}
PT_TOKENS = {"PT","PARTTIME","PTE"}

# *** KEY FIX ***: include "A" (your file uses A for Active)
EMPLOYED_TOKENS = {"ACTIVE", "A", "LOA"} | FT_TOKENS | PT_TOKENS

MV_PLANS = {"PLANA"}  # MV is PlanA by your spec
WAIVE_TOKENS = {"WAIVE", "WAIVED", "DECLINE"}

SPOUSE_TIERS = {"EMPFAM", "EMPSPOUSE"}
CHILD_TIERS  = {"EMPFAM", "EMPCHILD"}
EE_TIER      = "EMP"

# -------------------- helpers --------------------
def _norm_token(x) -> str:
    s = "" if x is None or (isinstance(x,float) and np.isnan(x)) else str(x)
    return "".join(ch for ch in s.upper().strip() if ch.isalnum())

def _last_day_of_month(d: date) -> date:
    if d.month == 12:
        return date(d.year, 12, 31)
    return date(d.year, d.month + 1, 1) - timedelta(days=1)

def month_bounds(year: int, month: int) -> Tuple[date, date]:
    ms = date(year, month, 1)
    me = _last_day_of_month(ms)
    return ms, me

def _any_overlap(df: pd.DataFrame, s_col: str, e_col: str, ms: date, me: date, mask=None) -> bool:
    if df.empty:
        return False
    m = mask if mask is not None else pd.Series(True, index=df.index)
    s = pd.to_datetime(df.loc[m, s_col], errors="coerce")
    e = pd.to_datetime(df.loc[m, e_col], errors="coerce")
    s = s.fillna(pd.Timestamp.min).dt.date
    e = e.fillna(pd.Timestamp.max).dt.date
    return bool(((e >= ms) & (s <= me)).any())

def _covers_full_month(df: pd.DataFrame, s_col: str, e_col: str, ms: date, me: date, mask=None) -> bool:
    if df.empty:
        return False
    m = mask if mask is not None else pd.Series(True, index=df.index)
    s = pd.to_datetime(df.loc[m, s_col], errors="coerce")
    e = pd.to_datetime(df.loc[m, e_col], errors="coerce")
    s = s.fillna(pd.Timestamp.min).dt.date
    e = e.fillna(pd.Timestamp.max).dt.date
    return bool(((s <= ms) & (e >= me)).any())

def _ids_from(*dfs: pd.DataFrame) -> List[str]:
    out = set()
    for d in dfs:
        if d is not None and not d.empty and "employeeid" in d.columns:
            out.update(map(lambda z: str(z).strip(), d["employeeid"].dropna().astype(str)))
    return sorted(out)

def _safe_min(x: Iterable[float]) -> float | None:
    vals = [v for v in x if pd.notna(v)]
    return min(vals) if vals else None

# -------------------- core build --------------------
def build_interim(
    emp_demo: pd.DataFrame,
    emp_status: pd.DataFrame,
    emp_elig: pd.DataFrame,
    emp_enroll: pd.DataFrame,
    dep_enroll: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """
    Produces the monthly 'interim' grid with all derived booleans and Line14/16.
    Expects cleaned dataframes from prepare_inputs().
    """

    year = int(year)
    affordability_threshold = float(os.getenv("AFFORDABILITY_THRESHOLD", "50"))

    # Normalize a few columns used below just in case
    for df in (emp_status, emp_elig, emp_enroll):
        if df is None or df.empty:
            continue
        for c in df.columns:
            if "tier" in c:
                df[c] = df[c].astype(str).str.strip()

    # Tokens for status/role
    if emp_status is not None and not emp_status.empty:
        if "_estatus_norm" not in emp_status.columns and "employmentstatus" in emp_status.columns:
            emp_status["_estatus_norm"] = emp_status["employmentstatus"].map(_norm_token)
        if "_role_norm" not in emp_status.columns and "role" in emp_status.columns:
            emp_status["_role_norm"] = emp_status["role"].map(_norm_token)

    # Enrollment: treat WAIVE as not enrolled
    def _not_waive(plancode: str) -> bool:
        return _norm_token(plancode) not in WAIVE_TOKENS

    # Build monthly grid
    employees = _ids_from(emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll)
    rows = []
    for eid in employees:
        for m in range(1, 13):
            ms, me = month_bounds(year, m)

            # restrict frames for eid
            st = emp_status[(emp_status["employeeid"].astype(str) == eid)] if emp_status is not None else pd.DataFrame()
            eg = emp_elig[(emp_elig["employeeid"].astype(str) == eid)] if emp_elig is not None else pd.DataFrame()
            en = emp_enroll[(emp_enroll["employeeid"].astype(str) == eid)] if emp_enroll is not None else pd.DataFrame()

            # EMPLOYMENT / FT / PT
            employed = _any_overlap(st, "statusstartdate", "statusenddate", ms, me,
                                    mask=st["_estatus_norm"].isin(list(EMPLOYED_TOKENS)) if not st.empty and "_estatus_norm" in st.columns else None)
            ft = _any_overlap(st, "statusstartdate", "statusenddate", ms, me,
                              mask=st["_role_norm"].isin(list(FT_TOKENS)) if not st.empty and "_role_norm" in st.columns else None)
            parttime = _any_overlap(st, "statusstartdate", "statusenddate", ms, me,
                                    mask=st["_role_norm"].isin(list(PT_TOKENS)) if not st.empty and "_role_norm" in st.columns else None)

            # ELIGIBILITY
            eligibleforcoverage = _any_overlap(eg, "eligibilitystartdate", "eligibilityenddate", ms, me)
            eligible_allmonth   = _covers_full_month(eg, "eligibilitystartdate", "eligibilityenddate", ms, me)

            # MV if eligible or enrolled with PlanA during the month
            mv_by_elig = _any_overlap(eg, "eligibilitystartdate", "eligibilityenddate", ms, me,
                                      mask=eg["plancode"].str.upper().eq("PLANA") if not eg.empty and "plancode" in eg.columns else None)
            mv_by_enrl = _any_overlap(en, "enrollmentstartdate", "enrollmentenddate", ms, me,
                                      mask=en["plancode"].str.upper().eq("PLANA") if not en.empty and "plancode" in en.columns else None)
            eligible_mv = bool(mv_by_elig or mv_by_enrl)

            # Offered to EE all month? (any tier covering full month)
            offer_ee_allmonth = _covers_full_month(eg, "eligibilitystartdate", "eligibilityenddate", ms, me)

            # Enrolled all month (non-Waive)
            enrolled_allmonth = False
            if not en.empty:
                mask_non_waive = en["plancode"].astype(str).map(_not_waive)
                enrolled_allmonth = _covers_full_month(en, "enrollmentstartdate", "enrollmentenddate", ms, me, mask=mask_non_waive)

            # Spouse / Child eligibility & enrollment
            spouse_eligible = _any_overlap(eg, "eligibilitystartdate", "eligibilityenddate", ms, me,
                                           mask=eg["eligibilitytier"].astype(str).str.upper().isin(SPOUSE_TIERS) if "eligibilitytier" in eg.columns else None)
            child_eligible = _any_overlap(eg, "eligibilitystartdate", "eligibilityenddate", ms, me,
                                          mask=eg["eligibilitytier"].astype(str).str.upper().isin(CHILD_TIERS) if "eligibilitytier" in eg.columns else None)

            spouse_enrolled = _any_overlap(en, "enrollmentstartdate", "enrollmentenddate", ms, me,
                                           mask=en["tier"].astype(str).str.upper().isin(SPOUSE_TIERS) if "tier" in en.columns else None)
            child_enrolled  = _any_overlap(en, "enrollmentstartdate", "enrollmentenddate", ms, me,
                                           mask=en["tier"].astype(str).str.upper().isin(CHILD_TIERS) if "tier" in en.columns else None)

            offer_spouse     = bool(spouse_eligible or spouse_enrolled)
            offer_dependents = bool(child_eligible  or child_enrolled)

            # Waiting period heuristic: employed but not eligible yet and first elig date is in future
            first_elig = None
            if not eg.empty:
                fe = pd.to_datetime(eg["eligibilitystartdate"], errors="coerce").dropna()
                if not fe.empty:
                    first_elig = fe.min().date()
            waitingperiod_month = bool(employed and not eligibleforcoverage and first_elig and me < first_elig)

            # Affordability (EMP tier cost during the month)
            emp_tier_mask = None
            if not eg.empty and "eligibilitytier" in eg.columns:
                emp_tier_mask = eg["eligibilitytier"].astype(str).str.upper().eq(EE_TIER)
            cost_rows = eg[emp_tier_mask] if emp_tier_mask is not None else pd.DataFrame()
            affordable_plan = False
            if not cost_rows.empty and "plancost" in cost_rows.columns:
                # among rows overlapping the month, is any EMP cost <= threshold?
                m = _any_overlap(cost_rows, "eligibilitystartdate", "eligibilityenddate", ms, me)
                if m:
                    # choose min cost among overlapping rows
                    cc = []
                    for _, r in cost_rows.iterrows():
                        s = pd.to_datetime(r["eligibilitystartdate"], errors="coerce")
                        e = pd.to_datetime(r["eligibilityenddate"], errors="coerce")
                        s = (s if pd.notna(s) else pd.Timestamp.min).date()
                        e = (e if pd.notna(e) else pd.Timestamp.max).date()
                        if e >= ms and s <= me:
                            cc.append(r.get("plancost"))
                    cmin = _safe_min(cc)
                    affordable_plan = (cmin is not None) and (float(cmin) <= affordability_threshold)

            rows.append(dict(
                EmployeeID=eid, Year=year, MonthNum=m, Month=MONTHS[m-1],
                MonthStart=pd.Timestamp(ms), MonthEnd=pd.Timestamp(me),
                employed=bool(employed), ft=bool(ft), parttime=bool(parttime),
                eligibleforcoverage=bool(eligibleforcoverage),
                eligible_allmonth=bool(eligible_allmonth),
                eligible_mv=bool(eligible_mv),
                offer_ee_allmonth=bool(offer_ee_allmonth),
                enrolled_allmonth=bool(enrolled_allmonth),
                offer_spouse=bool(offer_spouse),
                offer_dependents=bool(offer_dependents),
                spouse_eligible=bool(spouse_eligible),
                child_eligible=bool(child_eligible),
                spouse_enrolled=bool(spouse_enrolled),
                child_enrolled=bool(child_enrolled),
                waitingperiod_month=bool(waitingperiod_month),
                affordable_plan=bool(affordable_plan),
            ))

    out = pd.DataFrame(rows)

    # -------------------- Line 14 & Line 16 --------------------
    def _line14(r) -> str:
        if not r["employed"]:
            return "1G"  # handled later for all-12 override
        if r["eligible_mv"] and r["offer_ee_allmonth"]:
            # Qualifying offer if affordable and dependents/spouse offered
            if r["affordable_plan"] and r["offer_spouse"] and r["offer_dependents"]:
                return "1A"
            # Offered MV to EE (+ maybe family)
            if r["offer_spouse"] or r["offer_dependents"]:
                return "1E"
            return "1B"
        # No offer / not eligible
        return "1H"

    def _line16(r) -> str:
        if not r["employed"]:
            return "2A"
        if r["enrolled_allmonth"]:
            return "2C"
        if r["waitingperiod_month"]:
            return "2D"
        # Fallback for employed but not enrolled & not in waiting period
        return "2H"

    out["line14_final"] = out.apply(_line14, axis=1)
    out["line16_final"] = out.apply(_line16, axis=1)

    # If all 12 months line14 are 1G, blank out line14_final for that employee (your rule)
    def _all_12_are_1g(g: pd.DataFrame) -> bool:
        return (g["line14_final"] == "1G").all()

    out["line14_all12"] = ""
    for eid, g in out.groupby("EmployeeID", sort=False):
        if _all_12_are_1g(g):
            out.loc[g.index, "line14_all12"] = "1G"
            out.loc[g.index, "line14_final"] = ""  # blank when all 12 are 1G

    # Column ordering (stable)
    cols = [
        "EmployeeID","Year","MonthNum","Month","MonthStart","MonthEnd",
        "employed","ft","parttime",
        "eligibleforcoverage","eligible_allmonth","eligible_mv",
        "offer_ee_allmonth","enrolled_allmonth",
        "offer_spouse","offer_dependents",
        "spouse_eligible","child_eligible","spouse_enrolled","child_enrolled",
        "waitingperiod_month","affordable_plan",
        "line14_final","line16_final","line14_all12"
    ]
    return out.loc[:, cols]

# -------------------- Final table (by employee) --------------------
def build_final(interim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a 12-row (months) × 3-column (Month, Line14_Final, Line16_Final)
    grid per employee for the PDF fill.
    """
    pieces = []
    for eid, g in interim_df.groupby("EmployeeID", sort=False):
        g = g.sort_values("MonthNum")
        pieces.append(pd.DataFrame({
            "EmployeeID": eid,
            "Month": g["Month"].tolist(),
            "Line14_Final": g["line14_final"].tolist(),
            "Line16_Final": g["line16_final"].tolist(),
        }))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["EmployeeID","Month","Line14_Final","Line16_Final"])

# -------------------- Penalty dashboard --------------------
def build_penalty_dashboard(interim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize potential penalties with friendly 'Reason' text and monthly dollars.
    - Penalty A when no MEC offered to FT employee (Line14 '1H' while employed & FT)
    - Penalty B when MV offered but unaffordable and employee waived (not enrolled)
    Adds extra sentence when month(s) are marked waiting period.
    """
    df = interim_df.copy()
    df["is_A"] = (df["line14_final"] == "1H") & df["employed"]  # rough: no offer while employed
    df["is_B"] = (df["eligible_mv"] & df["offer_ee_allmonth"] & ~df["affordable_plan"] & ~df["enrolled_allmonth"])

    # monthly dollars
    for m in range(1, 13):
        col = MONTHS[m-1]
        df[col] = 0.0
        mask_m = df["MonthNum"] == m
        df.loc[mask_m & df["is_A"], col] = A_PENALTY
        df.loc[mask_m & df["is_B"], col] = B_PENALTY

    # build reason
    reasons = {}
    for eid, g in df.groupby("EmployeeID", sort=False):
        months_A = g.loc[g["is_A"], "Month"].tolist()
        months_B = g.loc[g["is_B"], "Month"].tolist()
        wp_months = g.loc[g["waitingperiod_month"], "Month"].tolist()

        lines = []
        if months_B:
            lines.append(
                "Penalty B: Waived Unaffordable Coverage <br/> "
                "The employee was offered minimum essential coverage (MEC) with minimum value, "
                "but the lowest-cost employee-only option was not affordable, and the employee waived."
            )
        if months_A:
            base = (
                "Penalty A: No MEC offered <br/> "
                "The employee was not offered minimum essential coverage (MEC) during the months in which the penalty was incurred."
            )
            if wp_months:
                wp_text = ", ".join(wp_months)
                base += f" Employee was not eligible for coverage in {wp_text} because they were in their waiting period during those months."
            lines.append(base)

        reasons[eid] = " ".join(lines) if lines else ""

    out_rows = []
    for eid, g in df.groupby("EmployeeID", sort=False):
        row = {"EmployeeID": eid, "Reason": reasons.get(eid, "")}
        for m, mon in enumerate(MONTHS, start=1):
            row[mon] = float(g.loc[g["MonthNum"] == m, mon].sum()) if not g.empty else 0.0
            if row[mon] == 0:
                row[mon] = "-"  # display dash for zero
            else:
                row[mon] = f"${row[mon]:.2f}"
        out_rows.append(row)

    cols = ["EmployeeID","Reason"] + MONTHS
    return pd.DataFrame(out_rows, columns=cols)
