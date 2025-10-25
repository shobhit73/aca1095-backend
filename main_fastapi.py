# main_fastapi.py

from __future__ import annotations

import io
import os
import zipfile
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse

import pandas as pd

# ---- project imports ----
from aca_processing import (
    load_excel,          # must return dict of dataframes, incl. 'year_used'
    _coerce_str,
)
from aca_builder import (
    build_interim,
    build_final,
    build_penalty_dashboard,
)
from aca_pdf import (
    fill_pdf_for_employee,
    save_excel_outputs,
    list_pdf_fields,     # debug helper
)

# ----------------------------------------------------------------------
# App + CORS
# ----------------------------------------------------------------------
app = FastAPI(title="ACA 1095 Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

FASTAPI_API_KEY = os.getenv("FASTAPI_API_KEY")  # optional
def _check_key(x_api_key: Optional[str]):
    if FASTAPI_API_KEY and (x_api_key or "") != FASTAPI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
async def _read_bytes(file: UploadFile) -> bytes:
    if file is None:
        raise HTTPException(status_code=422, detail="Missing file")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail=f"Empty upload: {file.filename}")
    return data

def _stream_bytes(b: bytes, filename: str, content_type: str) -> StreamingResponse:
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(io.BytesIO(b), headers=headers, media_type=content_type)


# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------------------------------------------------
# DEBUG: list PDF fields (diagnose blank PDFs / field-name mismatches)
# Use from Swagger UI at http://127.0.0.1:8000/docs
# ----------------------------------------------------------------------
@app.post("/debug/pdf_fields")
async def debug_pdf_fields(
    pdf: UploadFile = File(...),
):
    pdf_bytes = await _read_bytes(pdf)
    fields = list_pdf_fields(pdf_bytes)
    return JSONResponse(fields)


# ----------------------------------------------------------------------
# Core: Process Excel only (returns Excel workbook of outputs)
# ----------------------------------------------------------------------
@app.post("/process/excel")
async def process_excel(
    excel: UploadFile = File(...),
    filing_year: int = Form(...),
    affordability_threshold: float = Form(...),
    include_penalty_dashboard: Optional[bool] = Form(False),
    x_api_key: Optional[str] = Header(None),
):
    _check_key(x_api_key)

    # 1) Load excel
    excel_bytes = await _read_bytes(excel)
    try:
        data = load_excel(
            io.BytesIO(excel_bytes),
            filing_year=filing_year,
            affordability_threshold=affordability_threshold,
        )
    except Exception as e:
        return PlainTextResponse(f"Failed to load Excel: {e}", status_code=422)

    # 2) Build dataframes
    try:
        interim = build_interim(data)
        final = build_final(interim, data)
        penalty = build_penalty_dashboard(final, data) if include_penalty_dashboard else None
    except Exception as e:
        return PlainTextResponse(f"Failed during build: {e}", status_code=422)

    # 3) Write outputs to xlsx
    try:
        out_bytes = save_excel_outputs(
            interim=interim,
            final=final,
            year=data.get("year_used", filing_year),
            penalty_dashboard=penalty,
        )
    except Exception as e:
        return PlainTextResponse(f"Failed to write Excel: {e}", status_code=500)

    return _stream_bytes(out_bytes, f"aca_outputs_{filing_year}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ----------------------------------------------------------------------
# Core: Generate single employee PDF (returns one flattened PDF)
# Form fields expected:
#   - excel: xlsx input
#   - pdf: blank 1095-C template
#   - filing_year, affordability_threshold
#   - employee_id (optional, if blank uses the first employee)
#   - include_penalty_dashboard (ignored here)
# ----------------------------------------------------------------------
@app.post("/generate/single")
async def generate_single(
    excel: UploadFile = File(...),
    pdf: UploadFile = File(...),
    filing_year: int = Form(...),
    affordability_threshold: float = Form(...),
    employee_id: Optional[str] = Form(None),
    x_api_key: Optional[str] = Header(None),
):
    _check_key(x_api_key)

    excel_bytes = await _read_bytes(excel)
    pdf_bytes = await _read_bytes(pdf)

    # load & build
    try:
        data = load_excel(io.BytesIO(excel_bytes), filing_year=filing_year, affordability_threshold=affordability_threshold)
        interim = build_interim(data)
        final = build_final(interim, data)
        year_used = int(data.get("year_used", filing_year))
    except Exception as e:
        return PlainTextResponse(f"Failed to prepare data: {e}", status_code=422)

    # choose employee row
    try:
        # Expect final to have 'EmployeeID' and 'Month'
        emp_ids = final["EmployeeID"].dropna().astype(str).unique().tolist()
        if not emp_ids:
            return PlainTextResponse("No employees found in 'Final' sheet.", status_code=422)

        target_emp = _coerce_str(employee_id) if employee_id else _coerce_str(emp_ids[0])

        emp_final = final[final["EmployeeID"].astype(str) == target_emp]
        if emp_final.empty:
            return PlainTextResponse(f"EmployeeID {target_emp} not found in Final.", status_code=422)

        # demographic row for Part I fields: take first
        emp_demo = interim[interim["EmployeeID"].astype(str) == target_emp].head(1).squeeze()
        if isinstance(emp_demo, pd.DataFrame):  # just in case
            emp_demo = emp_demo.iloc[0]
    except Exception as e:
        return PlainTextResponse(f"Failed selecting employee: {e}", status_code=422)

    # optional coverage slices (if these exist in data)
    emp_enroll = data.get("emp_enroll")
    if isinstance(emp_enroll, pd.DataFrame):
        emp_enroll_emp = emp_enroll[emp_enroll["EmployeeID"].astype(str) == target_emp]
    else:
        emp_enroll_emp = None

    dep_enroll = data.get("dep_enroll")
    if isinstance(dep_enroll, pd.DataFrame):
        dep_enroll_emp = dep_enroll[dep_enroll["EmployeeID"].astype(str) == target_emp]
    else:
        dep_enroll_emp = None

    # fill the PDF
    try:
        editable_name, _editable, flat_name, flat_bytes_io = fill_pdf_for_employee(
            pdf_bytes=pdf_bytes,
            emp_row=emp_demo,
            final_df_emp=emp_final,
            year_used=year_used,
            emp_enroll_emp=emp_enroll_emp,
            dep_enroll_emp=dep_enroll_emp,
        )
        flat_bytes = flat_bytes_io.getvalue()
    except Exception as e:
        return PlainTextResponse(f"PDF generation failed: {e}", status_code=422)

    return _stream_bytes(flat_bytes, flat_name, "application/pdf")


# ----------------------------------------------------------------------
# Core: Bulk generate PDFs (returns ZIP of flattened PDFs)
# ----------------------------------------------------------------------
@app.post("/generate/bulk")
async def generate_bulk(
    excel: UploadFile = File(...),
    pdf: UploadFile = File(...),
    filing_year: int = Form(...),
    affordability_threshold: float = Form(...),
    x_api_key: Optional[str] = Header(None),
):
    _check_key(x_api_key)

    excel_bytes = await _read_bytes(excel)
    pdf_bytes = await _read_bytes(pdf)

    # Prepare dataframes
    try:
        data = load_excel(io.BytesIO(excel_bytes), filing_year=filing_year, affordability_threshold=affordability_threshold)
        interim = build_interim(data)
        final = build_final(interim, data)
        year_used = int(data.get("year_used", filing_year))
    except Exception as e:
        return PlainTextResponse(f"Failed to prepare data: {e}", status_code=422)

    emp_enroll = data.get("emp_enroll")
    dep_enroll = data.get("dep_enroll")

    # Build zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for emp in final["EmployeeID"].dropna().astype(str).unique():
            try:
                emp_final = final[final["EmployeeID"].astype(str) == emp]
                demo_row = interim[interim["EmployeeID"].astype(str) == emp].head(1).squeeze()
                if isinstance(demo_row, pd.DataFrame):
                    demo_row = demo_row.iloc[0]

                emp_enroll_emp = None
                dep_enroll_emp = None
                if isinstance(emp_enroll, pd.DataFrame):
                    emp_enroll_emp = emp_enroll[emp_enroll["EmployeeID"].astype(str) == emp]
                if isinstance(dep_enroll, pd.DataFrame):
                    dep_enroll_emp = dep_enroll[dep_enroll["EmployeeID"].astype(str) == emp]

                _, _, flat_name, flat_io = fill_pdf_for_employee(
                    pdf_bytes=pdf_bytes,
                    emp_row=demo_row,
                    final_df_emp=emp_final,
                    year_used=year_used,
                    emp_enroll_emp=emp_enroll_emp,
                    dep_enroll_emp=dep_enroll_emp,
                )
                zf.writestr(flat_name, flat_io.getvalue())
            except Exception as e:
                zf.writestr(f"error_{emp}.txt", f"Failed to build PDF for EmployeeID={emp}: {e}")

    buf.seek(0)
    return _stream_bytes(buf.getvalue(), f"aca_pdfs_{filing_year}.zip", "application/zip")
