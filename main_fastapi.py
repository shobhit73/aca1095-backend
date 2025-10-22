# main_fastapi.py
# FastAPI surface for ACA processing.
# NOTE: Only change vs earlier versions is reading the optional
# "Emp Wait Period" sheet and passing it into build_interim().

import io
import os
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Query
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware

import pandas as pd

from aca_processing import (
    load_excel,
    prepare_inputs,
    choose_report_year,
)
from aca_builder import (
    build_interim,
    build_final,
    build_penalty_dashboard,
)
from aca_pdf import (
    generate_single_pdf,     # your existing single-PDF function
    generate_bulk_pdfs,      # your existing bulk ZIP function
)

# ------------------------------
# App + simple API key guard
# ------------------------------
API_KEY = os.getenv("API_KEY", "")  # if you require one; empty means "off"

app = FastAPI(title="ACA Processor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _check_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ------------------------------
# Small helper: read Emp Wait Period sheet if present
# ------------------------------
def _read_emp_wait_period(file_bytes: bytes) -> pd.DataFrame:
    """
    Tolerant reader for the 'Emp Wait Period' sheet.
    Accepts exact name or variants containing both 'wait' and 'period'.
    Returns empty DataFrame if not found; builder will fallback to old behavior.
    """
    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception:
        return pd.DataFrame()

    wp_sheet = None
    for n in xls.sheet_names:
        name = n.strip().lower()
        if name == "emp wait period" or ("wait" in name and "period" in name):
            wp_sheet = n
            break

    if not wp_sheet:
        return pd.DataFrame()

    try:
        df = pd.read_excel(xls, wp_sheet)
        return df
    except Exception:
        return pd.DataFrame()

# ------------------------------
# /process/excel
# ------------------------------
@app.post("/process/excel")
async def process_excel(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_key(x_api_key)

    try:
        file_bytes = await file.read()
        # Load & normalize (same as before)
        raw = load_excel(file_bytes)
        emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, pay_deductions = prepare_inputs(raw)

        # Report year (same logic)
        year = choose_report_year(emp_elig)

        # NEW: read Emp Wait Period (optional)
        emp_wait_df = _read_emp_wait_period(file_bytes)

        # Build sheets (only change is passing emp_wait_period=emp_wait_df)
        interim = build_interim(
            emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, year,
            pay_deductions=pay_deductions,
            emp_wait_period=emp_wait_df
        )
        final = build_final(interim)
        penalty = build_penalty_dashboard(interim)

        # Return combined workbook as stream
        out_bytes = io.BytesIO()
        with pd.ExcelWriter(out_bytes, engine="xlsxwriter") as writer:
            final.to_excel(writer, sheet_name="Final", index=False)
            interim.to_excel(writer, sheet_name="Interim", index=False)
            penalty.to_excel(writer, sheet_name="Penalty Dashboard", index=False)
        out_bytes.seek(0)

        filename = f"final_interim_penalty_{year}.xlsx"
        return StreamingResponse(
            out_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

# ------------------------------
# /generate/single  (returns one filled 1095-C)
# ------------------------------
@app.post("/generate/single")
async def generate_single(
    file: UploadFile = File(...),
    employee_id: str = Query(..., alias="employeeId"),
    flatten_pdf: bool = Query(False, alias="flattenPdf"),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_key(x_api_key)

    try:
        file_bytes = await file.read()
        raw = load_excel(file_bytes)
        emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, pay_deductions = prepare_inputs(raw)
        year = choose_report_year(emp_elig)

        # Optional Emp Wait Period usage here as well, so the Interim used for PDF matches /process
        emp_wait_df = _read_emp_wait_period(file_bytes)

        interim = build_interim(
            emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, year,
            pay_deductions=pay_deductions,
            emp_wait_period=emp_wait_df
        )
        final = build_final(interim)

        # Your existing PDF function — keep the same signature you already had
        pdf_bytes = generate_single_pdf(final, interim, employee_id, year, flatten_pdf)

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="1095C_{employee_id}_{year}.pdf"'}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation error: {e}")

# ------------------------------
# /generate/bulk  (returns ZIP of PDFs)
# ------------------------------
@app.post("/generate/bulk")
async def generate_bulk(
    file: UploadFile = File(...),
    employee_ids: Optional[List[str]] = Query(default=None, alias="employeeIds"),
    flatten_pdf: bool = Query(False, alias="flattenPdf"),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_key(x_api_key)

    try:
        file_bytes = await file.read()
        raw = load_excel(file_bytes)
        emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, pay_deductions = prepare_inputs(raw)
        year = choose_report_year(emp_elig)

        emp_wait_df = _read_emp_wait_period(file_bytes)

        interim = build_interim(
            emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, year,
            pay_deductions=pay_deductions,
            emp_wait_period=emp_wait_df
        )
        final = build_final(interim)

        # Your existing ZIP function — keep your internal behavior the same
        zip_bytes = generate_bulk_pdfs(final, interim, employee_ids, year, flatten_pdf)

        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="1095C_PDFs.zip"'}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk PDF generation error: {e}")

# Health check (optional)
@app.get("/healthz")
def health():
    return JSONResponse({"ok": True})
