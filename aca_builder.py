# aca_builder.py
from datetime import datetime
import numpy as np
import pandas as pd

from aca_processing import (
    _int_year, _safe_int, _norm_token, _collect_employee_ids, _grid_for_year,
    _status_from_demographic, _any_overlap, _all_month, month_bounds,
    MONTHS, FULL_MONTHS, MONTHNUM_TO_FULL
)

# Role tokens & “employed” semantics for overlap detection
FT_TOKENS = {"FT","FULLTIME","FTE","CATEGORY2","CAT2"}
PT_TOKENS = {"PT","PARTTIME","PTE"}
EMPLOYED_TOKENS = {"ACTIVE","LOA"} | FT_TOKENS | PT_TOKENS

# Affordability threshold (EMP-only) for 1A vs 1E
AFFORDABILITY_THRESHOLD = 50.00

def _latest_emp_cost_for_month(el_df: pd.DataFrame, ms, me) -> float | None:
    if el_df is None or el_df.empty: return None
    need = {"eligibilitystartdate","eligibilityenddate","eligibilitytier"}
    if not need <= set(el_df.columns): return None
    df = el_df[(el_df["eligibilityenddate"].fillna(pd.Timestamp.max).dt.date >= ms) &
               (el_df["eligibilitystartdate"].fillna(pd.Timestamp.min).dt.date <= me)]
    if df.empty or "plancost" not in df.columns: return None
    tier_u = df["eligibilitytier"].astype(str).str.strip().str.upper()
    df = df[tier_u.eq("EMP")]
    if df.empty: return None
    df = df.sort_values("eligibilityenddate", ascending=False)
    v = pd.to_numeric(df.iloc[0]["plancost"], errors="coerce")
    return float(v) if not pd.isna(v) else None

def build_interim(emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, year=None, pay_deductions=None) -> pd.DataFrame:
    if year is None:
        from aca_processing import choose_report_year
        year = choose_report_year(emp_elig)
    year = _int_year(year, datetime.now().year)

    employee_ids = _collect_employee_ids(emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll)
    grid = _grid_for_year(employee_ids, year)

    # Names
    demo_names = pd.DataFrame(columns=["employeeid","firstname","lastname"])
    if not emp_demo.empty:
        tmp = emp_demo.copy()
        if "employeeid" in tmp.columns:
            from aca_processing import _normalize_employeeid
            tmp["employeeid"] = tmp["employeeid"].map(_normalize_employeeid)
        for col in ["firstname","lastname"]:
            if col not in tmp.columns: tmp[col] = ""
        demo_names = tmp[["employeeid","firstname","lastname"]].drop_duplicates("employeeid", keep="first")

    out = grid.merge(demo_names, on="employeeid", how="left")

    # Unified status table
    stt = emp_status.copy()
    if (stt is None) or stt.empty or not {"statusstartdate","statusenddate"} <= set(stt.columns):
        stt = _status_from_demographic(emp_demo)
    else:
        from aca_processing import _normalize_employeeid
        if "employeeid" in stt.columns:
            stt["employeeid"] = stt["employeeid"].map(_normalize_employeeid)
        if "employmentstatus" in stt.columns:
            stt["_estatus_norm"] = stt["employmentstatus"].astype(str).map(_norm_token)
        if "role" in stt.columns:
            stt["_role_norm"] = stt["role"].astype(str).map(_norm_token)
        from aca_processing import _parse_date_cols
        stt = _parse_date_cols(stt, ["statusstartdate","statusenddate"], default_end_cols=["statusenddate"])

    elg, enr, dep = emp_elig.copy(), emp_enroll.copy(), dep_enroll.copy()

    for df in (elg,enr,dep):
        if (df is not None) and (not df.empty):
            from aca_processing import _normalize_employeeid
            if "employeeid" in df.columns:
                df["employeeid"] = df["employeeid"].map(_normalize_employeeid)
            for c in df.columns:
                if c.endswith("date") and not np.issubdtype(df[c].dtype, np.datetime64):
                    df[c] = pd.to_datetime(df[c], errors="coerce")

    flags=[]
    for _,row in out.iterrows():
        emp = row["employeeid"]; ms=row["monthstart"].date(); me=row["monthend"].date()
        st_emp = stt[stt["employeeid"]==emp] if not stt.empty else stt
        el_emp = elg[elg["employeeid"]==emp] if not elg.empty else elg
        en_emp = enr[enr["employeeid"]==emp] if not enr.empty else enr
        de_emp = dep[dep["employeeid"]==emp] if not dep.empty else dep

        # EMPLOYED
        employed=False
        if not st_emp.empty and {"statusstartdate","statusenddate"} <= set(st_emp.columns):
            active_mask = pd.Series(False, index=st_emp.index)
            if "_estatus_norm" in st_emp.columns:
                active_mask = active_mask | st_emp["_estatus_norm"].isin(EMPLOYED_TOKENS)
            if "_role_norm" in st_emp.columns:
                active_mask = active_mask | st_emp["_role_norm"].isin(FT_TOKENS | PT_TOKENS)
            employed = _any_overlap(st_emp, "statusstartdate","statusenddate", ms,me, mask=active_mask)

        # FT/PT full-month
        ft_full_month = pt_full_month = False
        if not st_emp.empty and {"statusstartdate","statusenddate"} <= set(st_emp.columns):
            ft_mask = pd.Series(False, index=st_emp.index)
            pt_mask = pd.Series(False, index=st_emp.index)
            if "_role_norm" in st_emp.columns:
                ft_mask = ft_mask | st_emp["_role_norm"].isin(FT_TOKENS)
                pt_mask = pt_mask | st_emp["_role_norm"].isin(PT_TOKENS)
            if "_estatus_norm" in st_emp.columns:
                ft_mask = ft_mask | st_emp["_estatus_norm"].isin(FT_TOKENS)
                pt_mask = pt_mask | st_emp["_estatus_norm"].isin(PT_TOKENS)
            ft_full_month = _all_month(st_emp, "statusstartdate","statusenddate", ms,me, mask=ft_mask)
            pt_full_month = (not ft_full_month) and _all_month(st_emp, "statusstartdate","statusenddate", ms,me, mask=pt_mask)

        # Eligibility (PlanA => MV; tiers => dependents eligibility)
        eligible_any=False; eligible_allmonth=False; eligible_mv=False
        spouse_elig_any=False; child_elig_any=False
        if not el_emp.empty and {"eligibilitystartdate","eligibilityenddate"} <= set(el_emp.columns):
            eligible_any = _any_overlap(el_emp, "eligibilitystartdate","eligibilityenddate", ms,me)
            eligible_allmonth = _all_month(el_emp, "eligibilitystartdate","eligibilityenddate", ms,me)
            if "plancode" in el_emp.columns:
                plan_u = el_emp["plancode"].astype(str).str.strip().str.upper()
                eligible_mv = _any_overlap(el_emp, "eligibilitystartdate","eligibilityenddate", ms,me, mask=plan_u.eq("PLANA"))
            if "eligibilitytier" in el_emp.columns:
                tier_u = el_emp["eligibilitytier"].astype(str).str.strip().str.upper()
                spouse_elig_any = _any_overlap(el_emp, "eligibilitystartdate","eligibilityenddate", ms,me, mask=tier_u.eq("EMPFAM"))
                child_elig_any  = _any_overlap(el_emp, "eligibilitystartdate","eligibilityenddate", ms,me, mask=tier_u.isin(["EMPFAM","EMPCHILD"]))

        # Enrollment (employee & dependents)
        enrolled_allmonth=False
        spouse_enrolled=False; child_enrolled=False
        if not en_emp.empty and {"enrollmentstartdate","enrollmentenddate"} <= set(en_emp.columns):
            en_mask = en_emp["isenrolled"].fillna(True) if "isenrolled" in en_emp.columns else pd.Series(True,index=en_emp.index)
            enrolled_allmonth = _all_month(en_emp, "enrollmentstartdate","enrollmentenddate", ms,me, mask=en_mask)
            t = (en_emp["enrollmenttier"] if "enrollmenttier" in en_emp.columns else en_emp.get("plancode", pd.Series(index=en_emp.index, dtype=str))).astype(str).str.strip().str.upper()
            spouse_enrolled = _any_overlap(en_emp, "enrollmentstartdate","enrollmentenddate", ms,me, mask=t.eq("EMPFAM"))
            child_enrolled  = _any_overlap(en_emp, "enrollmentstartdate","enrollmentenddate", ms,me, mask=t.isin(["EMPFAM","EMPCHILD"]))

        if not de_emp.empty and {"enrollmentstartdate","enrollmentenddate","dependentrelationship"} <= set(de_emp.columns):
            rel = de_emp["dependentrelationship"].astype(str).str.lower()
            sp_rows = de_emp[rel.str.startswith("sp")]
            ch_rows = de_emp[rel.str.startswith("ch")]
            if "plancode" in de_emp.columns:
                sp_rows = sp_rows[~sp_rows["plancode"].astype(str).str.strip().str.lower().eq("waive")]
                ch_rows = ch_rows[~ch_rows["plancode"].astype(str).str.strip().str.lower().eq("waive")]
            if not sp_rows.empty:
                spouse_enrolled = spouse_enrolled or _any_overlap(sp_rows, "enrollmentstartdate","enrollmentenddate", ms,me)
            if not ch_rows.empty:
                child_enrolled  = child_enrolled  or _any_overlap(ch_rows, "enrollmentstartdate","enrollmentenddate", ms,me)

        offer_spouse = bool(spouse_elig_any or spouse_enrolled)
        offer_dependents = bool(child_elig_any or child_enrolled)
        offer_ee_allmonth = bool(eligible_allmonth or enrolled_allmonth)

        emp_cost = _latest_emp_cost_for_month(el_emp, ms, me)
        affordable = (emp_cost is not None) and (emp_cost < AFFORDABILITY_THRESHOLD)

        waitingperiod_month = bool(employed and ft_full_month and not eligible_any)

        # Line 14
        if offer_ee_allmonth:
            if eligible_mv:
                if offer_spouse and offer_dependents:
                    l14 = "1A" if affordable else "1E"
                elif offer_spouse and not offer_dependents:
                    l14 = "1D"
                elif (not offer_spouse) and offer_dependents:
                    l14 = "1C"
                else:
                    l14 = "1B"
            else:
                l14 = "1F"
        else:
            l14 = "1H"

        # Line 16
        if l14 == "1A":
            l16 = ""
        elif enrolled_allmonth:
            l16 = "2C"
        elif not offer_ee_allmonth:
            l16 = "2B"
        else:
            l16 = ""

        flags.append({
            "employed": employed,
            "ft": ft_full_month,
            "parttime": pt_full_month,
            "eligibleforcoverage": eligible_any,
            "eligible_allmonth": eligible_allmonth,
            "eligible_mv": eligible_mv,
            "offer_ee_allmonth": offer_ee_allmonth,
            "enrolled_allmonth": enrolled_allmonth,
            "offer_spouse": offer_spouse,
            "offer_dependents": offer_dependents,
            "spouse_eligible": spouse_elig_any,
            "child_eligible":  child_elig_any,
            "spouse_enrolled": spouse_enrolled,
            "child_enrolled":  child_enrolled,
            "waitingperiod_month": waitingperiod_month,
            "line14_final": l14,
            "line16_final": l16,
        })

    interim = pd.concat([out.reset_index(drop=True), pd.DataFrame(flags)], axis=1)

    base_cols = ["employeeid","firstname","lastname","year","monthnum","month","monthstart","monthend"]
    flag_cols = [
        "employed","ft","parttime",
        "eligibleforcoverage","eligible_allmonth","eligible_mv","offer_ee_allmonth",
        "enrolled_allmonth","offer_spouse","offer_dependents",
        "spouse_eligible","child_eligible","spouse_enrolled","child_enrolled",
        "waitingperiod_month","line14_final","line16_final"
    ]
    keep = [c for c in base_cols if c in interim.columns] + [c for c in flag_cols if c in interim.columns]
    interim = interim[keep].drop_duplicates(subset=["employeeid","year","monthnum"]).sort_values(
        ["employeeid","year","monthnum"]
    ).reset_index(drop=True)

    # Year-level 1G (never FT)
    rows=[]
    for emp_id, g in interim.groupby("employeeid", sort=False):
        g = g.copy()
        never_ft = not bool((g["ft"] == True).any())
        g["line14_all12"] = "1G" if never_ft else ""
        if never_ft:
            g["line14_final"] = ""
            g["line16_final"] = ""
        rows.append(g)
    return pd.concat(rows, ignore_index=True)

def build_final(interim: pd.DataFrame) -> pd.DataFrame:
    df = interim.copy()
    if "monthnum" in df.columns:
        df["monthnum"] = df["monthnum"].apply(lambda x: _safe_int(x, x))
    out = df.loc[:, ["employeeid","month","line14_final","line16_final"]].rename(columns={
        "employeeid":"EmployeeID","month":"Month","line14_final":"Line14_Final","line16_final":"Line16_Final"
    })
    if "monthnum" in df.columns:
        out = out.join(df["monthnum"]).sort_values(["EmployeeID","monthnum"]).drop(columns=["monthnum"])
    else:
        order = {m:i for i,m in enumerate(MONTHS, start=1)}
        out["_ord"]=out["Month"].map(order); out=out.sort_values(["EmployeeID","_ord"]).drop(columns=["_ord"])
    return out.reset_index(drop=True)

# ---------- Penalty Dashboard ----------
PENALTY_A = 241.67
PENALTY_B = 362.50

_PENALTY_TEXT_A = ("Penalty A: No MEC offered <br/> "
                   "The employee was not offered minimum essential coverage (MEC) during the months in which the penalty was incurred.")
_PENALTY_TEXT_B = ("Penalty B: Waived Unaffordable Coverage <br/> "
                   "The employee was offered minimum essential coverage (MEC), but the lowest-cost option for employee-only coverage "
                   "was not affordable, meaning it cost more than the $50 threshold. The employee chose to waive this unaffordable coverage.")

def _money(x: float | None) -> str:
    return "-" if (x is None or x == 0) else f"${x:,.2f}"

def build_penalty_dashboard(interim: pd.DataFrame,
                            penalty_a: float = PENALTY_A,
                            penalty_b: float = PENALTY_B) -> pd.DataFrame:
    if interim.empty:
        return pd.DataFrame(columns=["EmployeeID","Reason"] + FULL_MONTHS)

    df = interim.copy()
    df["EmployeeID"] = df["employeeid"]
    if "monthnum" in df.columns:
        df["monthnum"] = df["monthnum"].apply(lambda x: _safe_int(x, x))
    df["MonthFull"] = df["monthnum"].map(MONTHNUM_TO_FULL)

    cond_A = df["line14_final"].eq("1H")
    cond_B = df["line14_final"].eq("1E") & (~df["enrolled_allmonth"].fillna(False))

    df["_pen_amt"] = 0.0
    df.loc[cond_A, "_pen_amt"] = penalty_a
    df.loc[cond_B, "_pen_amt"] = penalty_b

    df["_pen_type"] = ""
    df.loc[cond_A, "_pen_type"] = "A"
    df.loc[cond_B, "_pen_type"] = "B"

    rows=[]
    for emp, g in df.groupby("EmployeeID", sort=True):
        months_map = {m:"-" for m in FULL_MONTHS}
        for _, r in g.iterrows():
            months_map[r["MonthFull"]] = _money(float(r["_pen_amt"])) if r["_pen_amt"] else "-"
        has_B = (g["_pen_type"]=="B").any()
        has_A = (g["_pen_type"]=="A").any()
        reason = _PENALTY_TEXT_B if has_B else (_PENALTY_TEXT_A if has_A else "")
        if has_A:
            wait_months = g.loc[g["waitingperiod_month"] & (g["_pen_type"]=="A"), "MonthFull"].tolist()
            if wait_months:
                reason += f"<br/><br/>Employee was not eligible for coverage in {', '.join(wait_months)} because they were in their wait period."
        row = {"EmployeeID": emp, "Reason": reason}
        row.update({m: months_map[m] for m in FULL_MONTHS})
        rows.append(row)

    return pd.DataFrame(rows, columns=["EmployeeID","Reason"] + FULL_MONTHS)
