# main_fastapi.py
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io, zipfile, json, os
import pandas as pd

from aca_processing import (
    load_excel, prepare_inputs, choose_report_year, MONTHS, _coerce_str
)
from aca_builder import (
    build_interim, build_final, build_penalty_dashboard
)
from aca_pdf import (
    fill_pdf_for_employee, save_excel_outputs
)

# ------------------ App & CORS ------------------
app = FastAPI(title="ACA 1095 Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ API Keys ------------------
def _get_api_key(request: Request):
    keys = os.getenv("API_KEYS", "supersecret-key-123").split(",")
    keys = [k.strip() for k in keys if k.strip()]
    incoming = request.headers.get("x-api-key", "")
    if incoming not in keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return incoming

@app.get("/health")
def health():
    return {"ok": True}

# ------------------ Excel → Final/Interim/Penalty (xlsx) ------------------
@app.post("/process/excel")
async def process_excel(
    excel: UploadFile = File(...),
    api_key: str = Depends(_get_api_key),
):
    if not excel.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Upload a .xlsx file")
    excel_bytes = await excel.read()
    try:
        data = load_excel(excel_bytes)
        emp_demo, emp_elig, emp_enroll, dep_enroll = prepare_inputs(data)

        year_used = choose_report_year(emp_elig)

        interim_df = build_interim(
            emp_demo, emp_elig, emp_enroll, dep_enroll, year=year_used
        )
        final_df   = build_final(interim_df)
        penalty_df = build_penalty_dashboard(interim_df)

        # Write 3 sheets: Final, Interim, Penalty Dashboard
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

# ------------------ Single 1095-C (PDF) ------------------
@app.post("/generate/single")
async def generate_single(
    excel: UploadFile = File(...),
    pdf:   UploadFile = File(...),
    employee_id: str | None = Form(None),
    flattened_only: str = Form("true"),
    api_key: str = Depends(_get_api_key),
):
    if not excel.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Upload a .xlsx file")
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a 1095-C base PDF")

    excel_bytes = await excel.read()
    pdf_bytes   = io.BytesIO(await pdf.read())

    try:
        data = load_excel(excel_bytes)
        emp_demo, emp_elig, emp_enroll, dep_enroll = prepare_inputs(data)
        if emp_demo.empty:
            raise HTTPException(status_code=422, detail="No employees in Emp Demographic")

        # Default EmployeeID if not provided
        if not employee_id:
            employee_id = _coerce_str(emp_demo["employeeid"].iloc[0])

        row = emp_demo[emp_demo["employeeid"].astype(str)==str(employee_id)]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"EmployeeID {employee_id} not found")

        year_used = choose_report_year(emp_elig)

        interim_df = build_interim(
            emp_demo, emp_elig, emp_enroll, dep_enroll, year=year_used
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
        headers = {"Content-Disposition": f'attachment; filename="1095c_{employee_id}.zip"'}
        return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF generation failed: {e}")

# ------------------ Bulk 1095-Cs (ZIP of flattened PDFs) ------------------
@app.post("/generate/bulk")
async def generate_bulk(
    excel: UploadFile = File(...),
    pdf:   UploadFile = File(...),
    employee_ids: str | None = Form(None),  # JSON list of ids or None for all
    api_key: str = Depends(_get_api_key),
):
    if not excel.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Upload a .xlsx file")
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a 1095-C base PDF")

    excel_bytes = await excel.read()
    pdf_bytes   = io.BytesIO(await pdf.read())

    try:
        data = load_excel(excel_bytes)
        emp_demo, emp_elig, emp_enroll, dep_enroll = prepare_inputs(data)
        if emp_demo.empty:
            raise HTTPException(status_code=422, detail="No employees in Emp Demographic")

        year_used = choose_report_year(emp_elig)
        all_ids = list(map(str, emp_demo["employeeid"].astype(str).unique()))
        if employee_ids:
            ids = list(map(str, json.loads(employee_ids)))
        else:
            ids = all_ids

        interim_df = build_interim(
            emp_demo, emp_elig, emp_enroll, dep_enroll, year=year_used
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
