# main_fastapi.py
# FastAPI entrypoint: /pipeline => returns a ZIP with interim_full.xlsx + pdfs/*

from __future__ import annotations
import io, os, zipfile
from typing import Optional, List, Tuple
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from debug_logging import get_logger
from aca_builder import build_interim_df, load_input_workbook
from pdf_filler import generate_all_pdfs

log = get_logger("aca1095-backend")

API_KEY = os.getenv("FASTAPI_API_KEY", "")
PDF_TEMPLATE_PATH = os.getenv("PDF_TEMPLATE_PATH", "/opt/app/f1095c.pdf")
FIELDS_JSON_PATH  = os.getenv("FIELDS_JSON_PATH",  "/opt/app/pdf_acro_fields_details.json")

app = FastAPI(title="ACA 1095 Pipeline", version="1.0.0")

def _require_api_key(x_api_key: Optional[str]):
    # explicit comparison; never evaluate DataFrame/Series
    if API_KEY and (x_api_key != API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/pipeline")
async def pipeline(
    year: int = Form(...),
    input_excel: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None)
):
    _require_api_key(x_api_key)
    try:
        excel_bytes = await input_excel.read()
        log.info(f"Pipeline start | year={year} | file={input_excel.filename} | bytes={len(excel_bytes)}")

        # 1) Build interim DF (all employees, all months)
        interim_df = build_interim_df(year, excel_bytes)
        if interim_df is None or interim_df.empty:
            raise ValueError("No employees or no data after processing.")

        # 2) Load demographics / dependents for PDF Part I/III (optional)
        sheets = load_input_workbook(excel_bytes)
        demo = sheets.get("Emp Demographic") or sheets.get("Emp_Demographic")
        dep  = sheets.get("Dep Enrollment") or sheets.get("Dep_Enrollment")
        if isinstance(demo, pd.DataFrame) and (not demo.empty) and ("EmployeeID" in demo.columns):
            demo["EmployeeID"] = pd.to_numeric(demo["EmployeeID"], errors="coerce").astype("Int64")

        # 3) Generate PDFs
        if not os.path.exists(PDF_TEMPLATE_PATH):
            raise FileNotFoundError(f"PDF template not found at: {PDF_TEMPLATE_PATH}")
        if not os.path.exists(FIELDS_JSON_PATH):
            raise FileNotFoundError(f"Fields JSON not found at: {FIELDS_JSON_PATH}")

        pdf_files: List[Tuple[str, bytes]] = generate_all
