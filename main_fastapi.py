# main_fastapi.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io, zipfile, json, os, importlib, types
import pandas as pd

from aca_processing import (
    load_excel, prepare_inputs, choose_report_year, MONTHS, _coerce_str
)

# ---- Safe import for builder functions (avoids startup crash on missing symbols) ----
def _load_builder():
    m = importlib.import_module("aca_builder")
    # required
    if not hasattr(m, "build_interim"):
        raise ImportError("aca_builder.build_interim is missing")

    # optional (provide safe fallbacks so the API still boots)
    if not hasattr(m, "build_final"):
        def _final_stub(interim_df: pd.DataFrame) -> pd.DataFrame:
            # Minimal shape expected by PDF filler
            if interim_df is None or interim_df.empty:
                return pd.DataFrame(columns=["EmployeeID","Month","Line14_Final","Line16_Final"])
            df = interim_df.rename(columns={"employeeid":"EmployeeID","month":"Month","monthnum":"MonthNum"}).copy()
            if "Month" not in df.columns and "MonthNum" in df.columns:
                months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                df["Month"] = df["MonthNum"].map(lambda i: months[int(i)-1])
            out = df[["EmployeeID","Month"]].copy()
            out["Line14_Final"] = ""
            out["Line16_Final"] = ""
            return out
        m.build_final = _final_stub

    if not hasattr(m, "build_penalty_dashboard"):
        def _pen_stub(interim_df: pd.DataFrame) -> pd.DataFrame:
            if interim_df is None or interim_df.empty:
                return pd.DataFrame(columns=[
                    "EmployeeID","Months_EligibleMV","Months_Affordable","Months_EligibleMV_NotAffordable"
                ])
            d = interim_df.rename(columns={"employeeid":"EmployeeID"}).copy()
            g = d.groupby("EmployeeID")
            res = pd.DataFrame({
                "Months_EligibleMV": g["eligible_mv"].sum(min_count=0),
                "Months_Affordable": g["affordable_plan"].sum(min_count=0),
            }).reset_index()
            d["_mv_not_aff"] = d["eligible_mv"].astype(bool) & (~d["affordable_plan"].astype(bool))
            mna = d.groupby("EmployeeID")["_mv_not_aff"].sum(min_count=0).reset_index(name="Months_EligibleMV_NotAffordable")
            out = res.merge(mna, on="EmployeeID", how="left").fillna(0)
            for c in ["Months_EligibleMV","Months_Affordable","Months_EligibleMV_NotAffordable"]:
                out[c] = out[c].astype(int)
            return out
        m.build_penalty_dashboard = _pen_stub

    return m.build_interim, m.build_final, m.build_penalty_dashboard

build_interim, build_final, build_penalty_dashboard = _load_builder()
# -----------------------------------------------------------------------------

from aca_pdf import (  # keep this import after the robust loader to simplify startup
    save_excel_outputs, fill_pdf_for_employee
)

app = FastAPI(title="ACA-1095 Builder API", version="1.0.0")

# CORS (tighten allow_origins for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST","GET","OPTIONS"],
    allow_headers=["*"],
)

API_KEYS = set(filter(None, os.getenv("API_KEYS", "supersecret-key-123").split(",")))

async def require_api_key(request: Request):
    key = request.headers.get("x-api-key")
    if key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

@app.get("/health")
async def health():
    # Expose which builder funcs are present — handy for Render logs
    try:
        m = importlib.import_module("aca_builder")
        return {
            "ok": True,
            "builder_has": {
                "build_interim": hasattr(m, "build_interim"),
                "build_final": hasattr(m, "build_final"),
                "build_penalty_dashboard": hasattr(m, "build_penalty_dashboard"),
            }
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# -------- Excel -> Interim/Final (+Penalty Dashboard) --------
@app.post("/process/excel", dependencies=[Depends(require_api_key)])
async def process_excel(excel: UploadFile = File(...)):
    if not excel.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Upload a .xlsx file")
    excel_bytes = await excel.read()
    try:
        data = load_excel(excel_bytes)
        emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, pay_deductions = prepare_inputs(data)

        year_used = choose_report_year(emp_elig)

        interim_df = build_interim(
            emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, year=year_used
        )
        final_df   = build_final(interim_df)
        penalty_df = build_penalty_dashboard(interim_df)

        out_bytes = save_excel_outputs(
            interim_df, final_df, year_used, penalty_dashboard=penalty_df
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to process Excel: {e}")

    fname = f"final_interim_penalty_{year_used}.xlsx"
    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
    return StreamingResponse(
        io.BytesIO(out_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers
    )

# -------- Single PDF fill --------
@app.post("/generate/single", dependencies=[Depends(require_api_key)])
async def generate_single(
    excel: UploadFile = File(...),
    pdf: UploadFile = File(...),
    employee_id: str = Form(None),
    flattened_only: str = Form("true")
):
    if not excel.filename.lower().endswith(".xlsx") or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload .xlsx and .pdf")

    excel_bytes = await excel.read()
    pdf_bytes = await pdf.read()

    try:
        data = load_excel(excel_bytes)
        emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, pay_deductions = prepare_inputs(data)
        if emp_demo.empty:
            raise HTTPException(status_code=422, detail="No employees in Emp Demographic")

        if not employee_id:
            employee_id = _coerce_str(emp_demo["employeeid"].iloc[0])

        row = emp_demo[emp_demo["employeeid"].astype(str)==str(employee_id)]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"EmployeeID {employee_id} not found")

        year_used = choose_report_year(emp_elig)

        interim_df = build_interim(
            emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, year=year_used
        )
        final_df = build_final(interim_df)

        emp_final = final_df[final_df["EmployeeID"].astype(str)==str(employee_id)].copy()
        if emp_final.empty:
            emp_final = pd.DataFrame({
                "Month": MONTHS,
                "Line14_Final": ["" for _ in MONTHS],
                "Line16_Final": ["" for _ in MONTHS]
            })

        editable_name, editable_bytes, flat_name, flat_bytes = fill_pdf_for_employee(
            pdf_bytes, row.iloc[0], emp_final, year_used
        )

        if flattened_only.lower() in {"true","1","yes","y"}:
            headers = {"Content-Disposition": f'attachment; filename="{flat_name}"'}
            return StreamingResponse(io.BytesIO(flat_bytes.getvalue()), media_type="application/pdf", headers=headers)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(editable_name, editable_bytes.getvalue())
            z.writestr(flat_name, flat_bytes.getvalue())
        zip_buf.seek(0)
        headers = {"Content-Disposition": f'attachment; filename="1095c_{employee_id}_{year_used}.zip"'}
        return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF generation failed: {e}")

# -------- Bulk PDF fill --------
@app.post("/generate/bulk", dependencies=[Depends(require_api_key)])
async def generate_bulk(
    excel: UploadFile = File(...),
    pdf: UploadFile = File(...),
    employee_ids: str = Form(None)  # JSON array or None=ALL
):
    if not excel.filename.lower().endswith(".xlsx") or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload .xlsx and .pdf")

    excel_bytes = await excel.read()
    pdf_bytes = await pdf.read()

    try:
        data = load_excel(excel_bytes)
        emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, pay_deductions = prepare_inputs(data)
        if emp_demo.empty:
            raise HTTPException(status_code=422, detail="No employees in Emp Demographic")

        year_used = choose_report_year(emp_elig)
        all_ids = list(map(str, emp_demo["employeeid"].astype(str).unique()))
        if employee_ids:
            import json as _json
            ids = list(map(str, _json.loads(employee_ids)))
        else:
            ids = all_ids

        interim_df = build_interim(
            emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, year=year_used
        )
        final_df = build_final(interim_df)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for eid in ids:
                row = emp_demo[emp_demo["employeeid"].astype(str)==eid]
                if row.empty:
                    continue
                emp_final = final_df[final_df["EmployeeID"].astype(str)==eid].copy()
                if emp_final.empty:
                    emp_final = pd.DataFrame({
                        "Month": MONTHS,
                        "Line14_Final": ["" for _ in MONTHS],
                        "Line16_Final": ["" for _ in MONTHS]
                    })
                _, _, flat_name, flat_bytes = fill_pdf_for_employee(pdf_bytes, row.iloc[0], emp_final, year_used)
                z.writestr(flat_name, flat_bytes.getvalue())
        zip_buf.seek(0)
        headers = {"Content-Disposition": f'attachment; filename="1095c_bulk_{year_used}.zip"'}
        return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Bulk generation failed: {e}")
