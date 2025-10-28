# main_fastapi.py
# FastAPI service for ACA 1095: Excel -> (Final, Interim, Penalty Dashboard) and PDF generation.

from __future__ import annotations

import io
import zipfile
import logging
from typing import Optional, List, Dict, Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic_settings import BaseSettings

from debug_logging import get_logger, log_time
import aca_processing as proc

log = get_logger("api")
logging.basicConfig(level=logging.INFO)


# =========================
# Settings
# =========================

class Settings(BaseSettings):
    API_KEY: Optional[str] = None

    class Config:
        env_prefix = "FASTAPI_"


def get_settings():
    return Settings()


def _check_api_key(request_key: Optional[str], settings: Settings):
    if settings.API_KEY:
        if not request_key or request_key != settings.API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")


# =========================
# App
# =========================

app = FastAPI(title="ACA 1095 Service")


# =========================
# Endpoints
# =========================

@app.post("/process/excel")
async def process_excel(
    file: UploadFile = File(..., description="Input Excel"),
    year: Optional[int] = Form(None, description="Filing year (optional)"),
    x_api_key: Optional[str] = Header(None, convert_underscores=False),
    settings: Settings = Depends(get_settings),
):
    """
    Accepts an Excel, builds Interim, Final, and Penalty Dashboard, and returns a workbook.
    - If 'year' is provided by the frontend, we use it.
    - Otherwise, we infer via aca_processing.choose_report_year(...).
    """
    _check_api_key(x_api_key, settings)

    with log_time(log, "/process/excel"):
        try:
            data_bytes = await file.read()
            out_bytes, meta = proc.process_excel_to_workbook(
                input_excel_bytes=data_bytes,
                filing_year=year  # may be None; processor will infer
            )
            headers = {
                "Content-Disposition": f'attachment; filename="ACA_Outputs.xlsx"'
            }
            return StreamingResponse(
                io.BytesIO(out_bytes),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers=headers,
            )
        except Exception as e:
            log.exception("Failed to process Excel")
            return JSONResponse(status_code=400, content={"error": f"Failed to process Excel: {e}"})


@app.post("/generate/single")
async def generate_single(
    file: UploadFile = File(..., description="Input Excel"),
    pdf_template: UploadFile = File(..., description="Blank 1095-C PDF template"),
    employee_id: Optional[str] = Form(None),
    year: Optional[int] = Form(None),
    flatten: Optional[bool] = Form(True),
    x_api_key: Optional[str] = Header(None, convert_underscores=False),
    settings: Settings = Depends(get_settings),
):
    """
    Generates a single employee 1095-C PDF.
    - If employee_id is omitted, the first employee encountered is used.
    - Year can be provided from frontend; otherwise inferred.
    """
    _check_api_key(x_api_key, settings)

    with log_time(log, "/generate/single"):
        try:
            excel_bytes = await file.read()
            pdf_bytes = await pdf_template.read()
            pdf_out = proc.generate_single_pdf_from_excel(
                input_excel_bytes=excel_bytes,
                blank_pdf_bytes=pdf_bytes,
                employee_id=employee_id,
                filing_year=year,
                flatten=bool(flatten),
            )
            headers = {
                "Content-Disposition": f'attachment; filename="1095C_single.pdf"'
            }
            return StreamingResponse(
                io.BytesIO(pdf_out),
                media_type="application/pdf",
                headers=headers
            )
        except Exception as e:
            log.exception("PDF generation failed")
            return JSONResponse(status_code=400, content={"error": f"PDF generation failed: {e}"})


@app.post("/generate/bulk")
async def generate_bulk(
    file: UploadFile = File(..., description="Input Excel"),
    pdf_template: UploadFile = File(..., description="Blank 1095-C PDF template"),
    year: Optional[int] = Form(None),
    flatten: Optional[bool] = Form(True),
    x_api_key: Optional[str] = Header(None, convert_underscores=False),
    settings: Settings = Depends(get_settings),
):
    """
    Generates a ZIP of 1095-C PDFs for all employees discovered in the workbook.
    """
    _check_api_key(x_api_key, settings)

    with log_time(log, "/generate/bulk"):
        try:
            excel_bytes = await file.read()
            pdf_bytes = await pdf_template.read()

            files: List[Dict[str, Any]] = proc.generate_bulk_pdfs_from_excel(
                input_excel_bytes=excel_bytes,
                blank_pdf_bytes=pdf_bytes,
                filing_year=year,
                flatten=bool(flatten),
            )

            mem = io.BytesIO()
            with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for item in files:
                    # item: {"employee_id": "...", "filename": "...", "pdf_bytes": b"..."}
                    zf.writestr(item["filename"], item["pdf_bytes"])
            mem.seek(0)

            headers = {
                "Content-Disposition": f'attachment; filename="1095C_bulk.zip"'
            }
            return StreamingResponse(mem, media_type="application/zip", headers=headers)
        except Exception as e:
            log.exception("Bulk PDF generation failed")
            return JSONResponse(status_code=400, content={"error": f"Bulk PDF generation failed: {e}"})


@app.get("/health")
async def health():
    return {"status": "ok"}
