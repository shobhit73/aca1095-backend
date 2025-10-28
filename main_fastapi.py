# main_fastapi.py
# FastAPI service for ACA 1095: Excel -> (Final, Interim, Penalty Dashboard) and PDF generation.

from __future__ import annotations

import io
import os
import zipfile
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, Iterable

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("aca1095")

# -------------------------
# Dependency: API key check
# -------------------------
def get_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """
    If FASTAPI_API_KEY env var is set, enforce header x-api-key to match it.
    If FASTAPI_API_KEY is not set, allow all requests (useful for local dev).
    """
    expected = os.getenv("FASTAPI_API_KEY")
    if expected:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="Missing x-api-key header")
        if x_api_key != expected:
            raise HTTPException(status_code=403, detail="Invalid API key")

# -------------------------
# Try imports from your codebase
# -------------------------
# These aliases let you keep your internal module structure while exposing a stable service layer here.
_load_excel = _build_interim = _build_final = _build_penalty = _save_excel_outputs = _fill_pdf_for_employee = None

import_error_msgs = []

try:
    # Sheet loader / parsing
    from aca_processing import load_excel as _load_excel  # parses the uploaded workbook into in-memory sheets
except Exception as e:
    import_error_msgs.append(f"aca_processing.load_excel: {e}")

try:
    # Interim/Final/Penalty builders
    from aca_builder import (
        build_interim as _build_interim,
        build_final as _build_final,
        build_penalty_dashboard as _build_penalty,
    )
except Exception as e:
    import_error_msgs.append(f"aca_builder.(build_*): {e}")

try:
    # Excel writer + PDF filler
    from aca_pdf import (
        save_excel_outputs as _save_excel_outputs,      # expects a dict[str, DataFrame] -> bytes(xlsx)
        fill_pdf_for_employee as _fill_pdf_for_employee # fills a single employee's 1095-C, returns tuples of (names, bytes)
    )
except Exception as e:
    import_error_msgs.append(f"aca_pdf.(save_excel_outputs/fill_pdf_for_employee): {e}")

if any(fn is None for fn in [_load_excel, _build_interim, _build_final, _build_penalty, _save_excel_outputs, _fill_pdf_for_employee]):
    # Fail early with a clear message listing which imports failed.
    msg = "Startup import errors:\n" + "\n".join(f"- {m}" for m in import_error_msgs)
    log.error(msg)

# -------------------------
# App
# -------------------------
app = FastAPI(
    title="ACA 1095-C Generator API",
    version="1.0.0",
    description="Processes ACA workbooks to produce Interim/Final sheets and filled 1095-C PDFs.",
)

# (Optional) Open up for your frontend domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Helpers
# -------------------------
def _require_ready():
    """Ensure all required functions imported correctly."""
    missing = []
    if _load_excel is None: missing.append("load_excel")
    if _build_interim is None: missing.append("build_interim")
    if _build_final is None: missing.append("build_final")
    if _build_penalty is None: missing.append("build_penalty_dashboard")
    if _save_excel_outputs is None: missing.append("save_excel_outputs")
    if _fill_pdf_for_employee is None: missing.append("fill_pdf_for_employee")
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Server not ready; missing functions: {', '.join(missing)}. Check module imports/logs."
        )

def _bytes_response(content: bytes, media_type: str, filename: str) -> StreamingResponse:
    return StreamingResponse(
        io.BytesIO(content),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

def _zip_bytes(pairs: Iterable[Tuple[str, bytes]]) -> bytes:
    """Create a zip archive from iterable of (arcname, file_bytes)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in pairs:
            zf.writestr(name, content)
    return buf.getvalue()

# -------------------------
# Routes
# -------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}

@app.post("/process/excel")
async def process_excel(
    file: UploadFile = File(..., description="Input ACA workbook (xlsx)"),
    year: Optional[int] = Form(None, description="Tax year override; defaults to current year"),
    _: None = Depends(get_api_key),
):
    """
    Accepts a single ACA input workbook.
    Returns a workbook with sheets: Interim, Final, [Penalty Dashboard].
    """
    _require_ready()
    try:
        excel_bytes = await file.read()
        data = _load_excel(excel_bytes)            # dict of parsed sheets (or an object your builder expects)
        year_used = int(year) if year else datetime.now().year

        # Build DataFrames
        interim_df = _build_interim(data, year_used)
        final_df   = _build_final(interim_df, year_used)
        penalty_df = _build_penalty(final_df)     # may be None/empty

        # IMPORTANT: save_excel_outputs expects ONE dict argument, not kwargs.
        outputs = {"Interim": interim_df, "Final": final_df}
        try:
            if penalty_df is not None and hasattr(penalty_df, "empty") and not penalty_df.empty:
                outputs["Penalty Dashboard"] = penalty_df
        except Exception:
            # In case penalty_df is a plain list or None, ignore silently.
            pass

        out_bytes = _save_excel_outputs(outputs)  # -> bytes (xlsx)
        filename = f"ACA-Outputs-{year_used}.xlsx"
        return _bytes_response(out_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename)

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to process Excel")
        raise HTTPException(status_code=400, detail=f"Failed to process Excel: {e}")

@app.post("/generate/single")
async def generate_single_pdf(
    file: UploadFile = File(..., description="Input ACA workbook (xlsx)"),
    pdf: UploadFile  = File(..., description="Blank 1095-C PDF template"),
    year: Optional[int] = Form(None, description="Tax year override; defaults to current year"),
    employee_id: Optional[str] = Form(None, description="Specific Employee ID; if omitted, first eligible row is used"),
    _: None = Depends(get_api_key),
):
    """
    Produces a single filled 1095-C PDF for one employee (flat PDF).
    Returns: application/pdf
    """
    _require_ready()
    try:
        excel_bytes = await file.read()
        pdf_bytes   = await pdf.read()
        data = _load_excel(excel_bytes)
        year_used = int(year) if year else datetime.now().year

        # Build Interim/Final to get employee rows
        interim_df = _build_interim(data, year_used)
        final_df   = _build_final(interim_df, year_used)

        # Filter for the requested employee if provided
        emp_final = final_df
        if employee_id:
            emp_final = final_df[final_df["EmployeeID"].astype(str) == str(employee_id)]
            if emp_final.empty:
                raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found in Final output")

        # Pick one row
        row = emp_final.iloc[0:1]  # keep as DataFrame to preserve columns/index
        # Call your PDF filler. Pass sheets=data so Part III can be derived.
        editable_name, editable_bytes, flat_name, flat_bytes = _fill_pdf_for_employee(
            pdf_bytes, row.iloc[0], emp_final, year_used, sheets=data
        )

        # Return the flat PDF (commonly used). If you prefer editable, swap names/bytes here.
        return _bytes_response(flat_bytes, "application/pdf", flat_name)

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to generate single PDF")
        raise HTTPException(status_code=400, detail=f"Failed to generate single PDF: {e}")

@app.post("/generate/bulk")
async def generate_bulk_pdfs(
    file: UploadFile = File(..., description="Input ACA workbook (xlsx)"),
    pdf: UploadFile  = File(..., description="Blank 1095-C PDF template"),
    year: Optional[int] = Form(None, description="Tax year override; defaults to current year"),
    _: None = Depends(get_api_key),
):
    """
    Produces a ZIP of filled 1095-C PDFs (flat) for all employees in Final.
    Returns: application/zip
    """
    _require_ready()
    try:
        excel_bytes = await file.read()
        pdf_bytes   = await pdf.read()
        data = _load_excel(excel_bytes)
        year_used = int(year) if year else datetime.now().year

        interim_df = _build_interim(data, year_used)
        final_df   = _build_final(interim_df, year_used)

        out_pairs = []  # (arcname, bytes)
        for _, row in final_df.iterrows():
            try:
                _, _, flat_name, flat_bytes = _fill_pdf_for_employee(
                    pdf_bytes, row, final_df, year_used, sheets=data
                )
                out_pairs.append((flat_name, flat_bytes))
            except Exception as inner_e:
                # Skip a bad row but include a text note for visibility
                err_note = f"ERROR_{row.get('EmployeeID', 'unknown')}.txt"
                out_pairs.append((err_note, str(inner_e).encode("utf-8")))

        zip_bytes = _zip_bytes(out_pairs)
        filename = f"ACA-1095C-Bulk-{year_used}.zip"
        return _bytes_response(zip_bytes, "application/zip", filename)

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to generate bulk PDFs")
        raise HTTPException(status_code=400, detail=f"Failed to generate bulk PDFs: {e}")
