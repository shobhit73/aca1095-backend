# aca_builder.py
# Build the full interim table (all employees, all months), infer flags, and compute Line 14/16.

from __future__ import annotations
import io
from typing import Dict, List, Optional, Tuple
import pandas as pd

# ---------- Month helpers ----------
MONTHS = ["Jan","Feb","Mar","Apr","May","June","July","Aug","Sept","Oct","Nov","Dec"]
MONTH_ALIASES = {
    "Jan":"Jan","January":"Jan",
    "Feb":"Feb","February":"Feb",
    "Mar":"Mar","March":"Mar",
    "Apr":"Apr","April":"Apr",
    "May":"May",
    "Jun":"June","June":"June",
    "Jul":"July","July":"July",
    "Aug":"Aug","August":"Aug",
    "Sep":"Sept","Sept":"Sept","September":"Sept",
    "Oct":"Oct","October":"Oct",
    "Nov":"Nov","November":"Nov",
    "Dec":"Dec","December":"Dec",
}

def to_canon_month(m: str) -> str:
    return MONTH_ALIASES.get(str(m).strip(), str(m).strip())

def months_in_range(start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
    if pd.isna(start) or pd.isna(end):
        return []
    rng = pd.date_range(start, end, freq="MS")
    return [MONTHS[d.month-1] for d in rng]

# ---------- Core loader ----------
def load_input_workbook(excel_bytes: bytes) -> Dict[str, pd.DataFrame]:
    bio = io.BytesIO(excel_bytes)
    # Known sheets (best-effort; missing ones become empty)
    sheets = {}
    with pd.ExcelFile(bio) as xf:
        for name in xf.sheet_names:
            try:
                df = xf.parse(name)
                df.columns = df.columns.str.strip().str.replace(" ", "_")
                sheets[name] = df
            except Exception:
                pass
    return sheets

# ---------- Interim builder ----------
def build_interim_df(year: int, excel_bytes: bytes) -> pd.DataFrame:
    """
    Returns a row-per-employee-per-month DataFrame for the given reporting year.
    Required sheets (if present): Emp_Demographic, Emp_Eligibility, Emp_Enrollment, Emp_Wait_Period (optional)
    Column names are normalized (spaces -> _).
    """
    sheets = load_input_workbook(excel_bytes)

    demo = sheets.get("Emp Demographic") or sheets.get("Emp_Demographic") or pd.DataFrame()
    elig = sheets.get("Emp Eligibility") or sheets.get("Emp_Eligibility") or pd.DataFrame()
    enrl = sheets.get("Emp Enrollment") or sheets.get("Emp_Enrollment") or pd.DataFrame()
    wait = sheets.get("Emp Wait Period") or sheets.get("Emp_Wait_Period") or pd.DataFrame()

    # Normalize
    for df in [demo, elig, enrl, wait]:
        if not df.empty:
            df.columns = df.columns.str.strip().str.replace(" ", "_")

    # Types
    if not demo.empty and "EmployeeID" in demo.columns:
        demo["EmployeeID"] = pd.to_numeric(demo["EmployeeID"], errors="coerce").astype("Int64")
    if not elig.empty and "EmployeeID" in elig.columns:
        elig["EmployeeID"] = pd.to_numeric(elig["EmployeeID"], errors="coerce").astype("Int64")
    if not enrl.empty and "EmployeeID" in enrl.columns:
        enrl["EmployeeID"] = pd.to_numeric(enrl["EmployeeID"], errors="coerce").astype("Int64")
    if not wait.empty and "EmployeeID" in wait.columns:
        wait["EmployeeID"] = pd.to_numeric(wait["EmployeeID"], errors="coerce").astype("Int64")

    # Dates
    for c in ["StatusStartDate", "StatusEndDate"]:
        if c in demo.columns:
            demo[c] = pd.to_datetime(demo[c], errors="coerce")
    for pair in [("EligibilityStartDate","EligibilityEndDate"), ("EnrollmentStartDate","EnrollmentEndDate")]:
        for c in pair:
            if c in elig.columns:
                elig[c] = pd.to_datetime(elig[c], errors="coerce")
            if c in enrl.columns:
                enrl[c] = pd.to_datetime(enrl[c], errors="coerce")
            if c in wait.columns:
                wait[c] = pd.to_datetime(wait[c], errors="coerce")

    # Exclude WAIVE in eligibility/enrollment
    if "EligiblePlan" in elig.columns:
        elig = elig[~elig["EligiblePlan"].astype(str).str.contains("Waive", case=False, na=False)]
    if "PlanCode" in enrl.columns:
        enrl = enrl[~enrl["PlanCode"].astype(str).str.contains("Waive", case=False, na=False)]

    # Build base grid: each employee x each month of the year
    months = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="MS")
    base_records = []
    for _, emp in demo.iterrows():
        emp_id = emp.get("EmployeeID")
        if pd.isna(emp_id):
            continue
        name = (str(emp.get("FirstName") or "") + " " +
                str(emp.get("MiddleInitial") or "") + " " +
                str(emp.get("LastName") or "")).strip()
        role = str(emp.get("Role") or "").strip().upper()
        sstart = emp.get("StatusStartDate")
        send   = emp.get("StatusEndDate")

        for m in months:
            mend = (m + pd.offsets.MonthEnd(1))
            employed_full = (pd.notna(sstart) and pd.notna(send) and sstart <= m and send >= mend)
            base_records.append({
                "Employee_ID": int(emp_id),
                "Name": name if name else str(int(emp_id)),
                "Year": year,
                "Month": MONTHS[m.month-1],
                "Is_Employed_full_month": "Yes" if employed_full else "No",
                "Is_full_time_full_month": "Yes" if (employed_full and role == "FT") else "No",
                "Is_Part_time_full_month": "Yes" if (employed_full and role == "PT") else "No",
            })

    base = pd.DataFrame(base_records)
    if base.empty:
        return base

    # Expand eligibility to month-level
    elig_monthly = pd.DataFrame()
    if not elig.empty:
        rows = []
        for _, r in elig.iterrows():
            for m in months_in_range(r.get("EligibilityStartDate"), r.get("EligibilityEndDate")):
                rows.append({
                    "EmployeeID": r.get("EmployeeID"),
                    "Month": m,
                    "EligiblePlan": r.get("EligiblePlan"),
                    "EligibleTier": r.get("EligibleTier"),
                    "PlanCost": r.get("PlanCost")
                })
        elig_monthly = pd.DataFrame(rows).drop_duplicates()

    # Expand enrollment to month-level
    enrl_monthly = pd.DataFrame()
    if not enrl.empty:
        rows = []
        for _, r in enrl.iterrows():
            for m in months_in_range(r.get("EnrollmentStartDate"), r.get("EnrollmentEndDate")):
                rows.append({
                    "EmployeeID": r.get("EmployeeID"),
                    "Month": m,
                    "PlanCode": r.get("PlanCode"),
                    "Tier": r.get("Tier")
                })
        enrl_monthly = pd.DataFrame(rows).drop_duplicates()

    # Aggregate by (EmployeeID, Month)
    if not elig_monthly.empty:
        agg_elig = (
            elig_monthly
            .groupby(["EmployeeID","Month"])
            .agg({
                "EligiblePlan": lambda x: set([str(v) for v in x.dropna()]),
                "EligibleTier": lambda x: set([str(v) for v in x.dropna()]),
                "PlanCost": lambda x: pd.to_numeric(pd.Series(list(x.dropna())), errors="coerce").min() if len(x.dropna()) else None
            })
            .reset_index()
        )
    else:
        agg_elig = pd.DataFrame(columns=["EmployeeID","Month","EligiblePlan","EligibleTier","PlanCost"])

    if not enrl_monthly.empty:
        agg_enrl = (
            enrl_monthly
            .groupby(["EmployeeID","Month"])
            .agg({
                "PlanCode": lambda x: set([str(v) for v in x.dropna()]),
                "Tier": lambda x: set([str(v) for v in x.dropna()])
            })
            .reset_index()
        )
    else:
        agg_enrl = pd.DataFrame(columns=["EmployeeID","Month","PlanCode","Tier"])

    # Merge into base
    final = base.merge(
        agg_elig, left_on=["Employee_ID","Month"], right_on=["EmployeeID","Month"], how="left"
    ).merge(
        agg_enrl, left_on=["Employee_ID","Month"], right_on=["EmployeeID","Month"], how="left"
    )
    final = final.drop(columns=["EmployeeID_x","EmployeeID_y"], errors="ignore")

    # Wait period flags per month (true if any wait row overlaps the month)
    final["is_waiting_period_true_full_month"] = "No"
    if not wait.empty:
        # Normalize
        for col in ["EligibilityStartDate","EligibilityEndDate"]:
            if col in wait.columns:
                wait[col] = pd.to_datetime(wait[col], errors="coerce")
        def month_overlap(emp_id: int, mon: str) -> bool:
            idx = MONTHS.index(mon) + 1
            ms = pd.Timestamp(year=year, month=idx, day=1)
            me = (ms + pd.offsets.MonthEnd(1))
            w = wait.loc[wait["EmployeeID"]==emp_id]
            for _, rr in w.iterrows():
                s = rr.get("EligibilityStartDate"); e = rr.get("EligibilityEndDate")
                if pd.isna(s) or pd.isna(e): 
                    continue
                if (s <= me) and (e >= ms):
                    return True
            return False
        final["is_waiting_period_true_full_month"] = final.apply(
            lambda r: "Yes" if month_overlap(r["Employee_ID"], r["Month"]) else "No", axis=1
        )

    # Plan cost > 50 (EMP-only affordability check)
    def plan_cost_gt_50(r) -> str:
        try:
            c = r.get("PlanCost")
            if pd.isna(c):
                return "No"
            return "Yes" if float(c) > 50 else "No"
        except Exception:
            return "No"
    final["plan_cost_greater_than_50"] = final.apply(plan_cost_gt_50, axis=1)

    # Eligibility flags (full-month; any tier signal present)
    def infer_flags(row):
        tiers_e = row["EligibleTier"] if isinstance(row.get("EligibleTier"), set) else set()
        plans_e = row["EligiblePlan"] if isinstance(row.get("EligiblePlan"), set) else set()
        tiers_n = row["Tier"] if isinstance(row.get("Tier"), set) else set()

        emp_tiers = {"EMP","EMPFAM","EMPSPOUSE"}
        family_tiers = {"EMPFAM","EMPSPOUSE"}
        child_tier = "EMPFAM"

        employee_eligible = bool(tiers_e.intersection(emp_tiers))
        spouse_eligible   = bool(tiers_e.intersection(family_tiers))
        child_eligible    = child_tier in tiers_e
        employee_enrolled = bool(tiers_n.intersection(emp_tiers))
        spouse_enrolled   = bool(tiers_n.intersection(family_tiers))
        child_enrolled    = child_tier in tiers_n

        eligible_planA_emp_any = ("PlanA" in plans_e) and employee_eligible

        return pd.Series([
            "Yes" if eligible_planA_emp_any else "No",
            "Yes" if employee_eligible else "No",
            "Yes" if spouse_eligible else "No",
            "Yes" if child_eligible else "No",
            "Yes" if employee_enrolled else "No",
            "Yes" if spouse_enrolled else "No",
            "Yes" if child_enrolled else "No",
        ])

    final[[
        "employee_eligible_for_planA_full_month",
        "employee_eligible",
        "spouse_eligible",
        "child_eligible",
        "employee_enrolled",
        "spouse_enrolled",
        "child_enrolled"
    ]] = final.apply(infer_flags, axis=1)

    # ------------- Line 14 / Line 16 logic -------------
    def compute_line_codes(r):
        ft = (r["Is_full_time_full_month"] == "Yes")
        emp_elg = (r["employee_eligible"] == "Yes")
        sp_elg  = (r["spouse_eligible"] == "Yes")
        ch_elg  = (r["child_eligible"] == "Yes")
        emp_enr = (r["employee_enrolled"] == "Yes")
        waitp   = (r["is_waiting_period_true_full_month"] == "Yes")
        cost_gt = (r["plan_cost_greater_than_50"] == "Yes")
        planA_emp = (r["employee_eligible_for_planA_full_month"] == "Yes")

        # ---- Line 14 ----
        l14 = ""
        if ft and planA_emp and sp_elg and ch_elg and not cost_gt:
            l14 = "1A"
        elif emp_elg and not sp_elg and not ch_elg:
            l14 = "1B"
        elif emp_elg and ch_elg and not sp_elg:
            l14 = "1C"
        elif emp_elg and sp_elg and not ch_elg:
            l14 = "1D"
        elif ft and planA_emp and sp_elg and ch_elg and cost_gt:
            l14 = "1E"
        elif emp_elg and not planA_emp:  # eligible for a plan but NOT Plan A
            l14 = "1F"
        elif (not ft) and emp_enr:
            l14 = "1G"
        else:
            # not eligible for A/B OR only eligible for non A/B plan
            l14 = "1H"

        # ---- Line 16 ----
        l16 = ""
        if r["Is_Employed_full_month"] == "No":
            l16 = "2A"
        elif not ft:
            l16 = "2B"
        elif emp_enr:
            l16 = "2C"        # precedence
        elif waitp:
            l16 = "2D"
        elif not cost_gt:
            l16 = "2F"
        else:
            l16 = "2H"

        return pd.Series([l14, l16])

    final[["line_14","line_16"]] = final.apply(compute_line_codes, axis=1)

    # Order columns nicely
    pref = [
        "Employee_ID","Name","Year","Month",
        "Is_Employed_full_month","Is_full_time_full_month","Is_Part_time_full_month",
        "employee_eligible_for_planA_full_month",
        "employee_eligible","spouse_eligible","child_eligible",
        "employee_enrolled","spouse_enrolled","child_enrolled",
        "is_waiting_period_true_full_month","plan_cost_greater_than_50",
        "line_14","line_16"
    ]
    rest = [c for c in final.columns if c not in pref and c not in {"EligiblePlan","EligibleTier","PlanCode","Tier","PlanCost"}]
    final = final[pref + rest]

    return final
