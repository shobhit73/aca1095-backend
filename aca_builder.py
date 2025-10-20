# aca_builder.py
from __future__ import annotations
import pandas as pd
from typing import List, Tuple, Dict
from datetime import date
from calendar import monthrange

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------- helpers ----------
def month_edges(year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, int, str]]:
    out = []
    for m in range(1, 13):
        start = pd.Timestamp(year=year, month=m, day=1)
        end   = pd.Timestamp(year=year, month=m, day=monthrange(year, m)[1])
        out.append((start, end, m, MONTHS[m-1]))
    return out

def _covered_full_month(ranges: List[Tuple[pd.Timestamp, pd.Timestamp]],
                        mstart: pd.Timestamp, mend: pd.Timestamp) -> bool:
    """True if union of ranges covers every day in [mstart, mend]."""
    if not ranges:
        return False
    # clip to month, merge
    segs = []
    for s, e in ranges:
        if pd.isna(s) or pd.isna(e): 
            continue
        if e < mstart or s > mend:
            continue
        segs.append((max(s, mstart), min(e, mend)))
    if not segs:
        return False
    segs.sort()
    cur_s, cur_e = segs[0]
    for s, e in segs[1:]:
        if s <= (cur_e + pd.Timedelta(days=1)):
            cur_e = max(cur_e, e)
        else:
            # gap
            return False
    return cur_s <= mstart and cur_e >= mend

def _overlaps_any(ranges: List[Tuple[pd.Timestamp, pd.Timestamp]],
                  mstart: pd.Timestamp, mend: pd.Timestamp) -> bool:
    for s, e in ranges:
        if pd.isna(s) or pd.isna(e): 
            continue
        if not (e < mstart or s > mend):
            return True
    return False

def _has_future_elig(elig_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]],
                     m_end: pd.Timestamp) -> bool:
    for s, _ in elig_ranges:
        if pd.isna(s): 
            continue
        if s > m_end:
            return True
    return False

def _tier_has_spouse(tier: str) -> bool:
    t = (tier or "").upper()
    return "EMPFAM" in t or "EMPSPOUSE" in t

def _tier_has_child(tier: str) -> bool:
    t = (tier or "").upper()
    return "EMPFAM" in t or "EMPCHILD" in t

# ---------- main ----------
def build_interim(emp_demo: pd.DataFrame,
                  emp_status: pd.DataFrame,
                  emp_elig: pd.DataFrame,
                  emp_enroll: pd.DataFrame,
                  dep_enroll: pd.DataFrame,
                  year: int,
                  affordability_threshold: float = 50.0,
                  mode: str = "UAT") -> pd.DataFrame:

    # employees universe
    ids = set(map(str, pd.concat([
        emp_demo["employeeid"],
        emp_elig["employeeid"],
        emp_enroll["employeeid"],
        dep_enroll["employeeid"]
    ], ignore_index=True).dropna().astype(str).unique()))
    months = month_edges(year)

    # Pre-slice ranges per employee
    demo_by_id: Dict[str, pd.DataFrame] = {k: v for k, v in emp_demo.groupby(emp_demo["employeeid"].astype(str))}
    elig_by_id: Dict[str, pd.DataFrame] = {k: v for k, v in emp_elig.groupby(emp_elig["employeeid"].astype(str))}
    enr_by_id:  Dict[str, pd.DataFrame] = {k: v for k, v in emp_enroll.groupby(emp_enroll["employeeid"].astype(str))}
    dep_by_id:  Dict[str, pd.DataFrame] = {k: v for k, v in dep_enroll.groupby(dep_enroll["employeeid"].astype(str))}

    rows = []

    for emp_id in sorted(ids, key=lambda x: (len(x), x)):
        demo = demo_by_id.get(emp_id, pd.DataFrame())
        elig = elig_by_id.get(emp_id, pd.DataFrame())
        enr  = enr_by_id.get(emp_id, pd.DataFrame())
        denr = dep_by_id.get(emp_id, pd.DataFrame())

        # ranges
        ft_ranges = []
        pt_ranges = []
        employed_ranges = []
        if not demo.empty:
            for _, r in demo.iterrows():
                s = r.get("statusstartdate")
                e = r.get("statusenddate")
                code = str(r.get("empstatuscode") or "").upper()
                emp_status_label = str(r.get("employmentstatus") or "").upper()
                if pd.isna(s) or pd.isna(e):
                    continue
                if emp_status_label in ("A", "ACTIVE", ""):
                    employed_ranges.append((s, e))
                if code == "FT":
                    ft_ranges.append((s, e))
                if code == "PT":
                    pt_ranges.append((s, e))

        elig_ranges = []
        eligA_ranges = []
        spouse_elig_ranges = []
        child_elig_ranges  = []
        elig_cost_rows = []
        if not elig.empty:
            for _, r in elig.iterrows():
                s = r.get("eligibilitystartdate")
                e = r.get("eligibilityenddate")
                plan = str(r.get("eligibleplan") or "").upper()
                tier = str(r.get("eligibletier") or "").upper()
                cost = r.get("plancost")
                if pd.isna(s) or pd.isna(e):
                    continue
                elig_ranges.append((s, e))
                if plan == "PLANA":
                    eligA_ranges.append((s, e))
                if _tier_has_spouse(tier):
                    spouse_elig_ranges.append((s, e))
                if _tier_has_child(tier):
                    child_elig_ranges.append((s, e))
                if pd.notna(cost):
                    elig_cost_rows.append((s, e, float(cost)))

        enr_ranges = []
        enrA_ranges = []
        spouse_enr_ranges = []
        child_enr_ranges  = []
        if not enr.empty:
            for _, r in enr.iterrows():
                s = r.get("enrollmentstartdate")
                e = r.get("enrollmentenddate")
                plan = str(r.get("plancode") or "").upper()
                tier = str(r.get("tier") or "").upper()
                if pd.isna(s) or pd.isna(e):
                    continue
                # Waive is NOT enrollment
                if plan != "WAIVE":
                    enr_ranges.append((s, e))
                    if plan == "PLANA":
                        enrA_ranges.append((s, e))
                    if _tier_has_spouse(tier):
                        spouse_enr_ranges.append((s, e))
                    if _tier_has_child(tier):
                        child_enr_ranges.append((s, e))

        # build month rows
        for mstart, mend, mnum, mname in months:
            employed = _overlaps_any(employed_ranges, mstart, mend)
            ft       = _covered_full_month(ft_ranges, mstart, mend)
            parttime = _covered_full_month(pt_ranges, mstart, mend)

            eligibleforcoverage = _overlaps_any(elig_ranges, mstart, mend)
            eligible_allmonth   = _covered_full_month(elig_ranges, mstart, mend)

            # MV: eligible to PlanA OR enrolled in PlanA during the month
            eligible_mv = _overlaps_any(eligA_ranges, mstart, mend) or _overlaps_any(enrA_ranges, mstart, mend)

            # employee offered for whole month if eligible all month OR enrolled all month
            enrolled_allmonth = _covered_full_month(enr_ranges, mstart, mend)
            offer_ee_allmonth = bool(eligible_allmonth or enrolled_allmonth)

            spouse_eligible   = _overlaps_any(spouse_elig_ranges, mstart, mend)
            child_eligible    = _overlaps_any(child_elig_ranges,  mstart, mend)
            spouse_enrolled   = _overlaps_any(spouse_enr_ranges,  mstart, mend)
            child_enrolled    = _overlaps_any(child_enr_ranges,   mstart, mend)

            offer_spouse      = bool(spouse_eligible or spouse_enrolled)
            offer_dependents  = bool(child_eligible  or child_enrolled)

            # waiting period month: employed & not eligible yet and a future eligibility exists
            waitingperiod_month = bool(employed and not eligibleforcoverage and _has_future_elig(elig_ranges, mend))

            # affordable plan (UAT threshold style)
            affordable_plan = False
            if elig_cost_rows:
                # any eligible period in the month with cost <= threshold
                for s, e, c in elig_cost_rows:
                    if not (e < mstart or s > mend):
                        if c <= float(affordability_threshold):
                            affordable_plan = True
                            break

            # ----- Line 14 -----
            line14 = ""
            # default rule
            if not offer_ee_allmonth:
                line14 = "1H"                       # no MEC offer
            elif eligible_mv and offer_spouse and offer_dependents and affordable_plan:
                line14 = "1A"                       # qualifying offer
            elif eligible_mv and offer_spouse and offer_dependents:
                line14 = "1E"                       # MV offered to EE+spouse+dep
            elif eligible_mv and offer_ee_allmonth:
                line14 = "1B"                       # MV to EE only
            elif offer_ee_allmonth:
                line14 = "1B"                       # MEC (non-MV) to EE

            # ----- Line 16 -----
            line16 = ""
            if not employed:
                line16 = "2A"
            elif enrolled_allmonth:
                line16 = "2C"
            elif waitingperiod_month:
                line16 = "2D"

            rows.append({
                "EmployeeID": emp_id,
                "Year": year,
                "MonthNum": mnum,
                "Month": mname,
                "MonthStart": mstart.normalize(),
                "MonthEnd": mend.normalize(),
                "employed": bool(employed),
                "ft": bool(ft),
                "parttime": bool(parttime),
                "eligibleforcoverage": bool(eligibleforcoverage),
                "eligible_allmonth": bool(eligible_allmonth),
                "eligible_mv": bool(eligible_mv),
                "offer_ee_allmonth": bool(offer_ee_allmonth),
                "enrolled_allmonth": bool(enrolled_allmonth),
                "offer_spouse": bool(offer_spouse),
                "offer_dependents": bool(offer_dependents),
                "spouse_eligible": bool(spouse_eligible),
                "child_eligible": bool(child_eligible),
                "spouse_enrolled": bool(spouse_enrolled),
                "child_enrolled": bool(child_enrolled),
                "waitingperiod_month": bool(waitingperiod_month),
                "affordable_plan": bool(affordable_plan),
                "line14_final": line14,
                "line16_final": line16,
                "line14_all12": ""  # fill later for 1G special case
            })

    out = pd.DataFrame(rows)

    # ---- Special: 1G (all 12 months not FT + enrolled at least 1 month) ----
    # If TRUE, we *leave line14_final blank* and set line14_all12 = '1G' per your requirement.
    by_emp = out.groupby("EmployeeID", as_index=False)
    ft_any = by_emp["ft"].transform("sum")
    enr_any = by_emp["enrolled_allmonth"].transform("sum") > 0
    is_1g_employee = (ft_any == 0) & (enr_any)
    out.loc[is_1g_employee, "line14_all12"] = "1G"
    out.loc[is_1g_employee, "line14_final"] = ""

    # ---- Type safety (prevents Excel column drift) ----
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype(int)
    out["MonthNum"] = pd.to_numeric(out["MonthNum"], errors="coerce").astype(int)
    out["MonthStart"] = pd.to_datetime(out["MonthStart"], errors="coerce")
    out["MonthEnd"]   = pd.to_datetime(out["MonthEnd"], errors="coerce")
    out["Month"] = out["MonthNum"].map(lambda m: MONTHS[m-1])

    # explicit column order for saving
    cols = [
        "EmployeeID","Year","MonthNum","Month","MonthStart","MonthEnd",
        "employed","ft","parttime",
        "eligibleforcoverage","eligible_allmonth","eligible_mv",
        "offer_ee_allmonth","enrolled_allmonth",
        "offer_spouse","offer_dependents",
        "spouse_eligible","child_eligible","spouse_enrolled","child_enrolled",
        "waitingperiod_month","affordable_plan",
        "line14_final","line16_final","line14_all12",
    ]
    return out.loc[:, cols]

def build_final(interim_df: pd.DataFrame) -> pd.DataFrame:
    """Long, per-employee 12 rows with Line14/Line16 used by PDF fill."""
    need = interim_df.loc[:, ["EmployeeID","Month","line14_final","line16_final","line14_all12"]].copy()
    need = need.rename(columns={"line14_final":"Line14_Final","line16_final":"Line16_Final"})
    # If employee is 1G-all-year, Line14_Final stays blank; the PDF layer can use line14_all12
    need["EmployeeID"] = need["EmployeeID"].astype(str)
    # keep month order
    need["Month"] = pd.Categorical(need["Month"], categories=MONTHS, ordered=True)
    need = need.sort_values(["EmployeeID","Month"])
    return need[["EmployeeID","Month","Line14_Final","Line16_Final","line14_all12"]]

def build_penalty_dashboard(interim_df: pd.DataFrame,
                            affordability_threshold: float = 50.0) -> pd.DataFrame:
    """Very simple penalty sketch using your wording + amounts."""
    MONTH_AMOUNTS = {
        "A": round(2900/12, 2),   # example annualized split (example numbers)
        "B": round(4350/12, 2),
    }
    # Reasons
    reason_A = ("Penalty A: No MEC offered <br/> "
                "The employee was not offered minimum essential coverage (MEC) "
                "during the months in which the penalty was incurred.")
    reason_B_unaff_waive = ("Penalty B: Waived Unaffordable Coverage <br/> "
                            "The employee was offered minimum essential coverage (MEC), "
                            "but the lowest-cost option for employee-only coverage was not affordable "
                            f"(>{affordability_threshold}). The employee chose to waive this unaffordable coverage.")
    # determine by month
    rows = []
    for emp_id, sub in interim_df.groupby("EmployeeID"):
        sub = sub.sort_values("MonthNum")
        months = {r["Month"]: 0.0 for _, r in sub.iterrows()}
        reason = ""
        # A penalty if line14 == 1H and employed (no offer)
        for _, r in sub.iterrows():
            m = r["Month"]
            if r["line14_final"] == "1H" and r["employed"]:
                months[m] = MONTH_AMOUNTS["A"]
                reason = reason_A
            # B penalty (your example): waive unaffordable — we infer: offered (not 1H), not enrolled, not affordable
            elif (r["line14_final"] in ("1B","1E") and not r["enrolled_allmonth"] and not r["affordable_plan"]):
                months[m] = MONTH_AMOUNTS["B"]
                reason = reason_B_unaff_waive

        if any(v > 0 for v in months.values()):
            rows.append({
                "EmployeeID": emp_id,
                "Reason": reason,
                **months
            })
    if not rows:
        # empty dashboard with headers
        return pd.DataFrame(columns=["EmployeeID","Reason", *MONTHS])
    return pd.DataFrame(rows)[["EmployeeID","Reason", *MONTHS]]
