# aca_builder.py
# Build Interim / Final / Penalty tables.
# NOTE: Only change vs. prior behavior is how waitingperiod_month is computed when an
#       "Emp Wait Period" sheet is provided. All other logic remains the same.

from __future__ import annotations
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from aca_processing import (
    _collect_employee_ids, _grid_for_year, month_bounds,
    _any_overlap, _all_month, _norm_token, _parse_date_cols, _normalize_employeeid
)

# ----- Tokens / constants (unchanged) -----
FT_TOKENS = {"FT","FULLTIME","FTE","CATEGORY2","CAT2"}
PT_TOKENS = {"PT","PARTTIME","PTE"}
EMPLOYED_TOKENS = {"ACTIVE","LOA"} | FT_TOKENS | PT_TOKENS

AFFORDABILITY_THRESHOLD = 50.00  # strict: < 50.00 is affordable (unchanged)

# =========================
# Internal helpers
# =========================

def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Defensive aliasing in case inputs didn't come via standardized loader."""
    if df is None or df.empty:
        return df
    out = df.copy()
    cols = set(out.columns)
    # EligiblePlan -> plancode
    if "eligibleplan" in {c.lower() for c in cols} and "plancode" not in {c.lower() for c in cols}:
        # find original-case column
        src = next(c for c in out.columns if c.lower()=="eligibleplan")
        out["plancode"] = out[src].astype(str).str.strip()
    # EligibleTier -> eligibilitytier
    if "eligibletier" in {c.lower() for c in cols} and "eligibilitytier" not in {c.lower() for c in cols}:
        src = next(c for c in out.columns if c.lower()=="eligibletier")
        out["eligibilitytier"] = out[src].astype(str).str.strip()
    return out

def _latest_emp_cost_for_month(el_df: pd.DataFrame, ms, me) -> Optional[float]:
    """
    Return employee-only (EMP) plan cost for rows overlapping the month,
    choosing the row with the latest eligibility end date. (Unchanged)
    """
    if el_df is None or el_df.empty:
        return None
    need = {"eligibilitystartdate","eligibilityenddate","eligibilitytier"}
    if not need <= set(map(str.lower, el_df.columns.str.lower())):
        # try case-insensitive check by remapping
        need = {"eligibilitystartdate","eligibilityenddate","eligibilitytier"}
        if not need <= set(el_df.columns):
            return None

    df = el_df.copy()
    # ensure datetime
    for c in ("eligibilitystartdate","eligibilityenddate"):
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # overlap [ms, me]
    df = df[(df["eligibilityenddate"].fillna(pd.Timestamp.max).dt.date >= ms) &
            (df["eligibilitystartdate"].fillna(pd.Timestamp.min).dt.date <= me)]
    if df.empty or "plancost" not in df.columns:
        return None

    # EMP tier only
    tier_u = df["eligibilitytier"].astype(str).str.strip().str.upper()
    df = df[tier_u.eq("EMP")]
    if df.empty:
        return None

    # latest end date
    df = df.sort_values("eligibilityenddate", ascending=False)
    v = pd.to_numeric(df.iloc[0]["plancost"], errors="coerce")
    return float(v) if not pd.isna(v) else None

def _wait_full_month_for(emp_wait_df: pd.DataFrame, emp_id, ms, me) -> bool:
    """
    Month is in waiting period if the Emp Wait Period window fully covers [ms..me].
    - Pick the row with latest EffectiveDate <= month-end for that employee.
    - Window = [EffectiveDate, EffectiveDate + (WaitPeriodDays - 1)].
    Returns False if sheet missing or no matching in-force row.
    """
    if emp_wait_df is None or emp_wait_df.empty:
        return False

    df = emp_wait_df.copy()

    # Resolve column variants:
    # Expect: EmployeeID, EffectiveDate, Wait Period
    name_map = {c.strip().lower().replace(" ", ""): c for c in df.columns}
    e_col = name_map.get("employeeid")
    d_col = name_map.get("effectivedate") or name_map.get("effective_date") or name_map.get("effective")
    w_col = (name_map.get("waitperiod") or name_map.get("wait_period") or
             name_map.get("waitperioddays") or name_map.get("waitperiodinadays"))
    if not w_col:
        # accept literal "wait period"
        for c in df.columns:
            if str(c).strip().lower() == "wait period":
                w_col = c
                break
    if not (e_col and d_col and w_col):
        return False

    # Filter to this employee
    sub = df[df[e_col].astype(str).str.strip() == str(emp_id)].copy()
    if sub.empty:
        return False

    sub[d_col] = pd.to_datetime(sub[d_col], errors="coerce")
    sub[w_col] = pd.to_numeric(sub[w_col], errors="coerce").fillna(0).clip(lower=0)

    # Only consider rows in force by month end
    sub = sub[sub[d_col] <= pd.Timestamp(me)].copy()
    if sub.empty:
        return False

    # Use most recent EffectiveDate
    sub = sub.sort_values(d_col, ascending=False)
    eff = sub.iloc[0][d_col]
    days = int(sub.iloc[0][w_col])

    if days <= 0 or pd.isna(eff):
        return False

    start = eff.date()
    end = (eff + pd.Timedelta(days=days-1)).date()
    return (start <= ms) and (end >= me)

# =========================
# Public builders
# =========================

def build_interim(
    emp_demo: pd.DataFrame,
    emp_status: pd.DataFrame,
    emp_elig: pd.DataFrame,
    emp_enroll: pd.DataFrame,
    dep_enroll: pd.DataFrame,
    year: int,
    pay_deductions: Optional[pd.DataFrame] = None,
    emp_wait_period: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the Interim table for employee x month.
    IMPORTANT: The only new behavior is that waitingperiod_month uses the Emp Wait Period sheet
    when provided; otherwise the previous fallback is used. All other columns/rules unchanged.
    """

    # Defensive aliasing for key inputs (no rule changes)
    emp_elig = _apply_aliases(emp_elig)
    emp_enroll = _apply_aliases(emp_enroll)
    dep_enroll = _apply_aliases(dep_enroll)

    # Normalize datetimes
    for df, sc, ec in [
        (emp_status, "statusstartdate", "statusenddate"),
        (emp_elig,  "eligibilitystartdate", "eligibilityenddate"),
        (emp_enroll,"enrollmentstartdate", "enrollmentenddate"),
        (dep_enroll,"eligiblestartdate",   "eligibleenddate"),
        (dep_enroll,"enrollmentstartdate", "enrollmentenddate"),
    ]:
        if df is not None and not df.empty:
            for c in (sc, ec):
                if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
                    df[c] = pd.to_datetime(df[c], errors="coerce")

    # Employees (unchanged: we DO NOT add employees solely from Emp Wait Period)
    employee_ids = _collect_employee_ids(emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll)
    grid = _grid_for_year(employee_ids, year)

    # Names
    names = pd.DataFrame(columns=["employeeid","firstname","lastname"])
    if not emp_demo.empty:
        tmp = emp_demo.copy()
        if "employeeid" in tmp.columns:
            tmp["employeeid"] = tmp["employeeid"].map(_normalize_employeeid)
        fn = tmp.get("firstname", pd.Series(index=tmp.index, dtype=str)).astype(str)
        ln = tmp.get("lastname",  pd.Series(index=tmp.index, dtype=str)).astype(str)
        names = pd.DataFrame({"employeeid": tmp["employeeid"], "firstname": fn, "lastname": ln})

    base = grid.merge(names, how="left", on="employeeid")

    # Status source (if missing, upstream code typically derives it; we just use what we get)
    stt = emp_status.copy() if emp_status is not None else pd.DataFrame()
    if not stt.empty:
        if "employeeid" in stt.columns:
            stt["employeeid"] = stt["employeeid"].map(_normalize_employeeid)
        if "employmentstatus" in stt.columns:
            stt["_estatus_norm"] = stt["employmentstatus"].astype(str).map(_norm_token)
        if "role" in stt.columns:
            stt["_role_norm"] = stt["role"].astype(str).map(_norm_token)
        stt = _parse_date_cols(stt, ["statusstartdate","statusenddate"], default_end_cols=["statusenddate"])

    # Normalize IDs in the others
    for df in (emp_elig, emp_enroll, dep_enroll):
        if df is not None and not df.empty and "employeeid" in df.columns:
            df["employeeid"] = df["employeeid"].map(_normalize_employeeid)

    rows = []
    for _, r in base.iterrows():
        emp = r["employeeid"]; ms = r["monthstart"].date(); me = r["monthend"].date()

        st_emp = stt[stt["employeeid"]==emp] if not stt.empty else stt
        el_emp = emp_elig[emp_elig["employeeid"]==emp] if not emp_elig.empty else emp_elig
        en_emp = emp_enroll[emp_enroll["employeeid"]==emp] if not emp_enroll.empty else emp_enroll
        de_emp = dep_enroll[dep_enroll["employeeid"]==emp] if not dep_enroll.empty else dep_enroll

        # ----- Employed / FT / PT (unchanged) -----
        employed=False; ft_full_month=False; pt_full_month=False
        if not st_emp.empty and {"statusstartdate","statusenddate"} <= set(st_emp.columns):
            active_mask = pd.Series(False, index=st_emp.index)
            if "_estatus_norm" in st_emp.columns:
                active_mask = active_mask | st_emp["_estatus_norm"].isin(EMPLOYED_TOKENS)
            employed = _any_overlap(st_emp, "statusstartdate","statusenddate", ms, me, mask=active_mask)

            ft_mask = pd.Series(False, index=st_emp.index)
            pt_mask = pd.Series(False, index=st_emp.index)
            if "_role_norm" in st_emp.columns:
                ft_mask = ft_mask | st_emp["_role_norm"].isin(FT_TOKENS)
                pt_mask = pt_mask | st_emp["_role_norm"].isin(PT_TOKENS)
            if "_estatus_norm" in st_emp.columns:
                ft_mask = ft_mask | st_emp["_estatus_norm"].isin(FT_TOKENS)
                pt_mask = pt_mask | st_emp["_estatus_norm"].isin(PT_TOKENS)
            ft_full_month = _all_month(st_emp, "statusstartdate","statusenddate", ms, me, mask=ft_mask)
            pt_full_month = (not ft_full_month) and _all_month(st_emp, "statusstartdate","statusenddate", ms, me, mask=pt_mask)

        # ----- Eligibility & dependent eligibility (unchanged) -----
        eligible_any=False; eligible_allmonth=False; eligible_mv=False
        spouse_elig_any=False; child_elig_any=False
        if not (el_emp is None or el_emp.empty) and {"eligibilitystartdate","eligibilityenddate"} <= set(el_emp.columns):
            eligible_any = _any_overlap(el_emp, "eligibilitystartdate","eligibilityenddate", ms, me)
            eligible_allmonth = _all_month(el_emp, "eligibilitystartdate","eligibilityenddate", ms, me)
            # MV (your existing rule: PlanA + allowed tiers must cover the full month)
            if "plancode" in el_emp.columns and "eligibilitytier" in el_emp.columns:
                plan_u = el_emp["plancode"].astype(str).str.strip().str.upper()
                tier_u = el_emp["eligibilitytier"].astype(str).str.strip().str.upper()
                mask = plan_u.eq("PLANA") & tier_u.isin({"EMP","EMPFAM","EMPCHILD","EMPSPOUSE"})
                eligible_mv = _all_month(el_emp, "eligibilitystartdate","eligibilityenddate", ms, me, mask=mask)
            # Dependent eligibility
            if "eligibilitytier" in el_emp.columns:
                t = el_emp["eligibilitytier"].astype(str).str.strip().str.upper()
                spouse_elig_any = _any_overlap(el_emp, "eligibilitystartdate","eligibilityenddate", ms, me, mask=t.eq("EMPFAM"))
                child_elig_any  = _any_overlap(el_emp, "eligibilitystartdate","eligibilityenddate", ms, me, mask=t.isin(["EMPFAM","EMPCHILD"]))

        # ----- Enrollment (unchanged) -----
        enrolled_allmonth=False
        spouse_enrolled=False; child_enrolled=False
        if not (en_emp is None or en_emp.empty) and {"enrollmentstartdate","enrollmentenddate"} <= set(en_emp.columns):
            en_mask = en_emp["isenrolled"].fillna(True) if "isenrolled" in en_emp.columns else pd.Series(True, index=en_emp.index)
            enrolled_allmonth = _all_month(en_emp, "enrollmentstartdate","enrollmentenddate", ms, me, mask=en_mask)
            t = (en_emp["enrollmenttier"] if "enrollmenttier" in en_emp.columns else en_emp.get("plancode", pd.Series(index=en_emp.index, dtype=str))).astype(str).str.strip().str.upper()
            spouse_enrolled = _any_overlap(en_emp, "enrollmentstartdate","enrollmentenddate", ms, me, mask=t.eq("EMPFAM"))
            child_enrolled  = _any_overlap(en_emp, "enrollmentstartdate","enrollmentenddate", ms, me, mask=t.isin(["EMPFAM","EMPCHILD"]))

        # Dep Enrollment may also prove dependent enrollment (ignoring waivers)
        if not (de_emp is None or de_emp.empty) and {"enrollmentstartdate","enrollmentenddate","dependentrelationship"} <= set(de_emp.columns):
            rel = de_emp["dependentrelationship"].astype(str).str.lower()
            sp_rows = de_emp[rel.str.startswith("sp")]
            ch_rows = de_emp[rel.str.startswith("ch")]
            if "plancode" in de_emp.columns:
                sp_rows = sp_rows[~de_emp["plancode"].astype(str).str.strip().str.lower().eq("waive")]
                ch_rows = ch_rows[~de_emp["plancode"].astype(str).str.strip().str.lower().eq("waive")]
            if not sp_rows.empty:
                spouse_enrolled = spouse_enrolled or _any_overlap(sp_rows, "enrollmentstartdate","enrollmentenddate", ms, me)
            if not ch_rows.empty:
                child_enrolled  = child_enrolled  or _any_overlap(ch_rows, "enrollmentstartdate","enrollmentenddate", ms, me)

        # ----- Offer flags (unchanged) -----
        offer_spouse = bool(spouse_elig_any or spouse_enrolled)
        offer_dependents = bool(child_elig_any or child_enrolled)
        offer_ee_allmonth = bool(eligible_allmonth or enrolled_allmonth)

        # ----- Affordability from EMP-tier cost (unchanged) -----
        emp_cost = _latest_emp_cost_for_month(el_emp, ms, me)
        affordable = (emp_cost is not None) and (emp_cost < AFFORDABILITY_THRESHOLD)

        # ----- Waiting period (ONLY place changed) -----
        # If Emp Wait Period provided, use full-month rule; else fallback to prior proxy.
        wp_full = _wait_full_month_for(emp_wait_period, emp, ms, me) if (emp_wait_period is not None and not emp_wait_period.empty) else None
        if wp_full is not None:
            waitingperiod_month = bool(wp_full)
        else:
            waitingperiod_month = bool(employed and ft_full_month and not eligible_any)

        # ----- Line 14 (unchanged) -----
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
            l14 = ""

        # ----- Line 16 (unchanged) -----
        if enrolled_allmonth:
            l16 = "2C"
        elif not offer_ee_allmonth:
            l16 = "2B" if employed else ""
        elif employed and not affordable:
            l16 = "2H"  # placeholder if no safe harbor satisfied
        else:
            l16 = ""

        rows.append({
            "employeeid": emp,
            "firstname": r.get("firstname",""),
            "lastname": r.get("lastname",""),
            "year": r["year"],
            "monthnum": r["monthnum"],
            "month": r["month"],
            "monthstart": r["monthstart"],
            "monthend": r["monthend"],
            # unchanged fields
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
            # only changed source:
            "waitingperiod_month": waitingperiod_month,
            # unchanged affordability & codes:
            "affordable_plan": affordable,
            "line14_final": l14,
            "line16_final": l16,
        })

    interim = pd.DataFrame(rows).sort_values(["employeeid","year","monthnum"]).reset_index(drop=True)

    # ----- Year-level 1G rule (unchanged) -----
    out_rows = []
    for emp_id, g in interim.groupby("employeeid", sort=False):
        g = g.copy()
        never_ft = not bool((g["ft"] == True).any())
        g["line14_all12"] = "1G" if never_ft else ""
        if never_ft:
            g["line14_final"] = ""
        out_rows.append(g)
    interim = pd.concat(out_rows, ignore_index=True) if out_rows else interim

    return interim


def build_final(interim: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse interim to Final (one row per employee) for 1095-C Part II.
    Unchanged behavior.
    """
    if interim is None or interim.empty:
        return pd.DataFrame(columns=[
            "EmployeeID","FirstName","LastName","Line14_All12",
            *[f"Line14_{m}" for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]],
            *[f"Line16_{m}" for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]],
        ])

    df = interim.copy()
    df["EmployeeID"] = df["employeeid"]
    df["FirstName"] = df.get("firstname","")
    df["LastName"]  = df.get("lastname","")

    # Fill month code columns directly from interim
    final = (df
        .pivot_table(index=["EmployeeID","FirstName","LastName"],
                     columns="monthnum",
                     values=["line14_final","line16_final"],
                     aggfunc="first")
        .reset_index())
    # Flatten columns
    final.columns = ["EmployeeID","FirstName","LastName"] + [
        f"{lvl0}_{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1]}"
        for (lvl0, m) in final.columns.tolist()[3:]
    ]

    # Add All-12 shortcut when exactly one code is used across all 12 months
    l14_cols = [c for c in final.columns if c.startswith("line14_final_")]
    final["Line14_All12"] = ""
    # Normalize names to your expected export headers
    rename_map = {f"line14_final_{m}": f"Line14_{m}" for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]}
    rename_map.update({f"line16_final_{m}": f"Line16_{m}" for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]})
    final = final.rename(columns=rename_map)

    # If all 12 Line14_* are identical and non-empty, set Line14_All12 to that code
    l14_list = [f"Line14_{m}" for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]]
    same = final[l14_list].apply(lambda s: len(set([x for x in s if isinstance(x,str) and x]))==1, axis=1)
    code = final[l14_list].apply(lambda s: next((x for x in s if isinstance(x,str) and x), ""), axis=1)
    final.loc[same & code.ne(""), "Line14_All12"] = code

    # Reorder columns
    final = final[["EmployeeID","FirstName","LastName","Line14_All12",
                   *[f"Line14_{m}" for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]],
                   *[f"Line16_{m}" for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]]]]
    return final


def build_penalty_dashboard(interim: pd.DataFrame) -> pd.DataFrame:
    """
    Simple monthly penalty view based on interim flags. Unchanged behavior.
    """
    if interim is None or interim.empty:
        return pd.DataFrame(columns=["EmployeeID","EmployeeName","MonthFull","Reason","Amount"])

    df = interim.copy()
    df["EmployeeID"] = df["employeeid"]
    df["EmployeeName"] = (df.get("firstname","").fillna("").astype(str).str.strip() + " " +
                          df.get("lastname","").fillna("").astype(str).str.strip()).str.strip()

    # Rough penalty types (placeholder logic, unchanged)
    cond_A = (df["line14_final"]=="") & (df["line16_final"]!="2C")
    cond_B = (df["line14_final"].isin(["1E","1B","1C","1D"])) & (df["affordable_plan"]==False) & (df["line16_final"]!="2C")

    df["_pen_type"] = ""
    df.loc[cond_A, "_pen_type"] = "A"
    df.loc[cond_B, "_pen_type"] = "B"

    # Summarize by employee
    rows = []
    for emp, g in df.groupby("EmployeeID", sort=True):
        months = g.sort_values("monthnum")["month"].tolist()
        months_str = ", ".join(months) if months else ""
        reason = "B: Offer made but unaffordable (employee share too high)" if (g["_pen_type"]=="B").any() else (
                 "A: No offer of MEC for one or more months" if (g["_pen_type"]=="A").any() else "")
        rows.append({
            "EmployeeID": emp,
            "EmployeeName": g["EmployeeName"].iloc[0],
            "MonthFull": months_str,
            "Reason": reason,
            "Amount": "-"  # keep placeholder/unchanged
        })
    return pd.DataFrame(rows)
