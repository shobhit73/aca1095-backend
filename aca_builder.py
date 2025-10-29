# aca_builder.py
# Build the full interim table (all employees, all months), infer flags, and compute Line 14/16.

from __future__ import annotations
import io
from typing import Dict, List, Optional
import pandas as pd

# ---------- helpers to avoid ambiguous truth ----------
def is_df_present(df) -> bool:
    return isinstance(df, pd.DataFrame) and (not df.empty)

MONTHS = ["Jan","Feb","Mar","Apr","May","June","July","Aug","Sept","Oct","Nov","Dec"]
ALIASES = {
    "January":"Jan","Jan":"Jan",
    "February":"Feb","Feb":"Feb",
    "March":"Mar","Mar":"Mar",
    "April":"Apr","Apr":"Apr",
    "May":"May",
    "June":"June","Jun":"June",
    "July":"July","Jul":"July",
    "August":"Aug","Aug":"Aug",
    "September":"Sept","Sep":"Sept","Sept":"Sept",
    "October":"Oct","Oct":"Oct",
    "November":"Nov","Nov":"Nov",
    "December":"Dec","Dec":"Dec",
}

def months_in_range(start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
    if pd.isna(start) or pd.isna(end):
        return []
    rng = pd.date_range(start, end, freq="MS")
    return [MONTHS[d.month-1] for d in rng]

def load_input_workbook(excel_bytes: bytes) -> Dict[str, pd.DataFrame]:
    bio = io.BytesIO(excel_bytes)
    out: Dict[str, pd.DataFrame] = {}
    with pd.ExcelFile(bio) as xf:
        for s in xf.sheet_names:
            try:
                df = xf.parse(s)
                if isinstance(df, pd.DataFrame):
                    df.columns = df.columns.str.strip().str.replace(" ", "_")
                out[s] = df
            except Exception:
                pass
    return out

def build_interim_df(year: int, excel_bytes: bytes) -> pd.DataFrame:
    sheets = load_input_workbook(excel_bytes)

    demo = sheets.get("Emp Demographic") or sheets.get("Emp_Demographic") or pd.DataFrame()
    elig = sheets.get("Emp Eligibility") or sheets.get("Emp_Eligibility") or pd.DataFrame()
    enrl = sheets.get("Emp Enrollment") or sheets.get("Emp_Enrollment") or pd.DataFrame()
    wait = sheets.get("Emp Wait Period") or sheets.get("Emp_Wait_Period") or pd.DataFrame()

    for df in [demo, elig, enrl, wait]:
        if is_df_present(df):
            df.columns = df.columns.str.strip().str.replace(" ", "_")

    if is_df_present(demo) and "EmployeeID" in demo.columns:
        demo["EmployeeID"] = pd.to_numeric(demo["EmployeeID"], errors="coerce").astype("Int64")
    if is_df_present(elig) and "EmployeeID" in elig.columns:
        elig["EmployeeID"] = pd.to_numeric(elig["EmployeeID"], errors="coerce").astype("Int64")
    if is_df_present(enrl) and "EmployeeID" in enrl.columns:
        enrl["EmployeeID"] = pd.to_numeric(enrl["EmployeeID"], errors="coerce").astype("Int64")
    if is_df_present(wait) and "EmployeeID" in wait.columns:
        wait["EmployeeID"] = pd.to_numeric(wait["EmployeeID"], errors="coerce").astype("Int64")

    for c in ["StatusStartDate","StatusEndDate"]:
        if is_df_present(demo) and c in demo.columns:
            demo[c] = pd.to_datetime(demo[c], errors="coerce")
    for pair in [("EligibilityStartDate","EligibilityEndDate"), ("EnrollmentStartDate","EnrollmentEndDate")]:
        for c in pair:
            if is_df_present(elig) and c in elig.columns:
                elig[c] = pd.to_datetime(elig[c], errors="coerce")
            if is_df_present(enrl) and c in enrl.columns:
                enrl[c] = pd.to_datetime(enrl[c], errors="coerce")
            if is_df_present(wait) and c in wait.columns:
                wait[c] = pd.to_datetime(wait[c], errors="coerce")

    if is_df_present(elig) and "EligiblePlan" in elig.columns:
        elig = elig[~elig["EligiblePlan"].astype(str).str.contains("Waive", case=False, na=False)]
    if is_df_present(enrl) and "PlanCode" in enrl.columns:
        enrl = enrl[~enrl["PlanCode"].astype(str).str.contains("Waive", case=False, na=False)]

    months = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="MS")

    # base grid
    base_rows = []
    if is_df_present(demo):
        for _, emp in demo.iterrows():
            emp_id = emp.get("EmployeeID")
            if pd.isna(emp_id):
                continue
            name = (str(emp.get("FirstName") or "") + " " +
                    str(emp.get("MiddleInitial") or "") + " " +
                    str(emp.get("LastName") or "")).strip()
            role = str(emp.get("Role") or "").upper().strip()
            sstart = emp.get("StatusStartDate")
            send   = emp.get("StatusEndDate")
            for m in months:
                mend = m + pd.offsets.MonthEnd(1)
                employed_full = (pd.notna(sstart) and pd.notna(send) and sstart <= m and send >= mend)
                base_rows.append({
                    "Employee_ID": int(emp_id),
                    "Name": name if name else str(int(emp_id)),
                    "Year": year,
                    "Month": MONTHS[m.month-1],
                    "Is_Employed_full_month": "Yes" if employed_full else "No",
                    "Is_full_time_full_month": "Yes" if (employed_full and role=="FT") else "No",
                    "Is_Part_time_full_month": "Yes" if (employed_full and role=="PT") else "No",
                })
    base = pd.DataFrame(base_rows)
    if base.empty:
        return base

    # month-level eligibility
    elig_monthly = pd.DataFrame()
    if is_df_present(elig):
        rows = []
        for _, r in elig.iterrows():
            for mon in months_in_range(r.get("EligibilityStartDate"), r.get("EligibilityEndDate")):
                rows.append({
                    "EmployeeID": r.get("EmployeeID"),
                    "Month": mon,
                    "EligiblePlan": r.get("EligiblePlan"),
                    "EligibleTier": r.get("EligibleTier"),
                    "PlanCost": r.get("PlanCost"),
                })
        if rows:
            elig_monthly = pd.DataFrame(rows).drop_duplicates()

    # month-level enrollment
    enrl_monthly = pd.DataFrame()
    if is_df_present(enrl):
        rows = []
        for _, r in enrl.iterrows():
            for mon in months_in_range(r.get("EnrollmentStartDate"), r.get("EnrollmentEndDate")):
                rows.append({
                    "EmployeeID": r.get("EmployeeID"),
                    "Month": mon,
                    "PlanCode": r.get("PlanCode"),
                    "Tier": r.get("Tier"),
                })
        if rows:
            enrl_monthly = pd.DataFrame(rows).drop_duplicates()

    # aggregate
    if is_df_present(elig_monthly):
        agg_elig = (
            elig_monthly.groupby(["EmployeeID","Month"])
            .agg({
                "EligiblePlan": lambda x: set([str(v) for v in x.dropna()]),
                "EligibleTier": lambda x: set([str(v) for v in x.dropna()]),
                "PlanCost": lambda x: pd.to_numeric(pd.Series(list(x.dropna())), errors="coerce").min()
                          if len(x.dropna()) else None
            }).reset_index()
        )
    else:
        agg_elig = pd.DataFrame(columns=["EmployeeID","Month","EligiblePlan","EligibleTier","PlanCost"])

    if is_df_present(enrl_monthly):
        agg_enrl = (
            enrl_monthly.groupby(["EmployeeID","Month"])
            .agg({
                "PlanCode": lambda x: set([str(v) for v in x.dropna()]),
                "Tier":     lambda x: set([str(v) for v in x.dropna()]),
            }).reset_index()
        )
    else:
        agg_enrl = pd.DataFrame(columns=["EmployeeID","Month","PlanCode","Tier"])

    final = base.merge(
        agg_elig, left_on=["Employee_ID","Month"], right_on=["EmployeeID","Month"], how="left"
    ).merge(
        agg_enrl, left_on=["Employee_ID","Month"], right_on=["EmployeeID","Month"], how="left"
    )
    final = final.drop(columns=["EmployeeID_x","EmployeeID_y"], errors="ignore")

    # wait period per month
    final["is_waiting_period_true_full_month"] = "No"
    if is_df_present(wait):
        for c in ["EligibilityStartDate","EligibilityEndDate"]:
            if c in wait.columns:
                wait[c] = pd.to_datetime(wait[c], errors="coerce")

        def month_overlap(emp_id: int, mon: str) -> bool:
            try:
                idx = MONTHS.index(mon) + 1
            except Exception:
                return False
            ms = pd.Timestamp(year=year, month=idx, day=1)
            me = ms + pd.offsets.MonthEnd(1)
            w = wait.loc[wait["EmployeeID"]==emp_id] if "EmployeeID" in wait.columns else pd.DataFrame()
            if w.empty:
                return False
            for _, rr in w.iterrows():
                s = rr.get("EligibilityStartDate"); e = rr.get("EligibilityEndDate")
                if pd.notna(s) and pd.notna(e) and (s <= me) and (e >= ms):
                    return True
            return False

        final["is_waiting_period_true_full_month"] = final.apply(
            lambda r: "Yes" if month_overlap(r["Employee_ID"], r["Month"]) else "No", axis=1
        )

    # plan cost > 50 (EMP-only)
    def plan_cost_gt_50(r) -> str:
        c = r.get("PlanCost", None)
        try:
            if pd.isna(c):
                return "No"
            return "Yes" if float(c) > 50 else "No"
        except Exception:
            return "No"
    final["plan_cost_greater_than_50"] = final.apply(plan_cost_gt_50, axis=1)

    # eligibility/enrollment flags
    def infer_flags(row):
        tiers_e = row["EligibleTier"] if isinstance(row.get("EligibleTier"), set) else set()
        plans_e = row["EligiblePlan"] if isinstance(row.get("EligiblePlan"), set) else set()
        tiers_n = row["Tier"]        if isinstance(row.get("Tier"), set)        else set()

        emp_tiers    = {"EMP","EMPFAM","EMPSPOUSE"}
        family_tiers = {"EMPFAM","EMPSPOUSE"}

        employee_eligible = len(tiers_e.intersection(emp_tiers)) > 0
        spouse_eligible   = len(tiers_e.intersection(family_tiers)) > 0
        child_eligible    = "EMPFAM" in tiers_e

        employee_enrolled = len(tiers_n.intersection(emp_tiers)) > 0
        spouse_enrolled   = len(tiers_n.intersection(family_tiers)) > 0
        child_enrolled    = "EMPFAM" in tiers_n

        planA_emp_any = ("PlanA" in plans_e) and employee_eligible

        return pd.Series([
            "Yes" if planA_emp_any else "No",
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

    # line 14/16
    def compute_line_codes(r):
        ft      = (r.get("Is_full_time_full_month") == "Yes")
        emp_elg = (r.get("employee_eligible") == "Yes")
        sp_elg  = (r.get("spouse_eligible") == "Yes")
        ch_elg  = (r.get("child_eligible") == "Yes")
        emp_enr = (r.get("employee_enrolled") == "Yes")
        wait_p  = (r.get("is_waiting_period_true_full_month") == "Yes")
        cost_gt = (r.get("plan_cost_greater_than_50") == "Yes")
        planA_emp = (r.get("employee_eligible_for_planA_full_month") == "Yes")

        # Line 14
        if ft and planA_emp and sp_elg and ch_elg and not cost_gt:
            l14 = "1A"
        elif emp_elg and (not sp_elg) and (not ch_elg):
            l14 = "1B"
        elif emp_elg and ch_elg and (not sp_elg):
            l14 = "1C"
        elif emp_elg and sp_elg and (not ch_elg):
            l14 = "1D"
        elif ft and planA_emp and sp_elg and ch_elg and cost_gt:
            l14 = "1E"
        elif emp_elg and (not planA_emp):
            l14 = "1F"
        elif (not ft) and emp_enr:
            l14 = "1G"
        else:
            l14 = "1H"

        # Line 16
        if r.get("Is_Employed_full_month") == "No":
            l16 = "2A"
        elif not ft:
            l16 = "2B"
        elif emp_enr:
            l16 = "2C"
        elif wait_p:
            l16 = "2D"
        elif not cost_gt:
            l16 = "2F"
        else:
            l16 = "2H"

        return pd.Series([l14, l16])

    final[["line_14","line_16"]] = final.apply(compute_line_codes, axis=1)

    pref = [
        "Employee_ID","Name","Year","Month",
        "Is_Employed_full_month","Is_full_time_full_month","Is_Part_time_full_month",
        "employee_eligible_for_planA_full_month",
        "employee_eligible","spouse_eligible","child_eligible",
        "employee_enrolled","spouse_enrolled","child_enrolled",
        "is_waiting_period_true_full_month","plan_cost_greater_than_50",
        "line_14","line_16"
    ]
    extra = [c for c in final.columns if c not in pref and c not in {"EligiblePlan","EligibleTier","PlanCode","Tier","PlanCost"}]
    return final[pref + extra]
