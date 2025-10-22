# main_fastapi.py
# FastAPI for ACA 1095 processing — Render ready

import io
import os
import importlib
from typing import Optional, List

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Query
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware

from aca_processing import load_excel, prepare_inputs, choose_report_year
from aca_builder import build_interim, build_final, build_penalty_dashboard

# -----------------------------
# App setup
# -----------------------------
API_KEY = os.getenv("API_KEY", "")
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

# -----------------------------
# Read Emp Wait Period
# -----------------------------
def _read_emp_wait_period(file_bytes: bytes) -> pd.DataFrame:
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
        return pd.read_excel(xls, wp_sheet)
    except Exception:
        return pd.DataFrame()

# -----------------------------
# PDF function lookup
# -----------------------------
def _pdf_funcs():
    try:
        mod = importlib.import_module("aca_pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not import aca_pdf: {e}")

    singles = ["generate_single_pdf", "generate_single"]
    bulks = ["generate_bulk_pdfs", "generate_bulk"]

    fn_single = next((getattr(mod, n, None) for n in singles if callable(getattr(mod, n, None))), None)
    fn_bulk = next((getattr(mod, n, None) for n in bulks if callable(getattr(mod, n, None))), None)

    if not fn_single or not fn_bulk:
        raise HTTPException(status_code=500, detail="Missing PDF functions in aca_pdf.")
    return fn_single, fn_bulk

# -----------------------------
# /process/excel
# -----------------------------
@app.post("/process/excel")
async def process_excel(file: UploadFile = File(...), x_api_key: Optional[str] = Header(default=None)):
    _check_key(x_api_key)
    try:
        file_bytes = await file.read()
        raw = load_excel(file_bytes)
        emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, pay_deductions = prepare_inputs(raw)
        year = choose_report_year(emp_elig)

        emp_wait_df = _read_emp_wait_period(file_bytes)

        interim = build_interim(
            emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, year,
            pay_deductions=pay_deductions, emp_wait_period=emp_wait_df
        )
        final = build_final(interim)
        penalty = build_penalty_dashboard(interim)

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as w:
            final.to_excel(w, sheet_name="Final", index=False)
            interim.to_excel(w, sheet_name="Interim", index=False)
            penalty.to_excel(w, sheet_name="Penalty Dashboard", index=False)
        out.seek(0)
        return StreamingResponse(
            out,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="final_interim_penalty_{year}.xlsx"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# /generate/single
# -----------------------------
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
        emp_wait_df = _read_emp_wait_period(file_bytes)

        interim = build_interim(
            emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, year,
            pay_deductions=pay_deductions, emp_wait_period=emp_wait_df
        )
        final = build_final(interim)

        fn_single, _ = _pdf_funcs()
        pdf_bytes = fn_single(final, interim, employee_id, year, flatten_pdf)

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="1095C_{employee_id}_{year}.pdf"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# /generate/bulk
# -----------------------------
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
            pay_deductions=pay_deductions, emp_wait_period=emp_wait_df
        )
        final = build_final(interim)

        _, fn_bulk = _pdf_funcs()
        zip_bytes = fn_bulk(final, interim, employee_ids, year, flatten_pdf)

        return StreamingResponse(
            io.BytesIO(zip_bytes),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="1095C_PDFs.zip"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# /healthz
# -----------------------------
@app.get("/healthz")
def health():
    return JSONResponse({"ok": True})
