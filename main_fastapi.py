# main_fastapi.py
# FastAPI service for ACA 1095-C: Excel -> (Interim, Final, Penalty Dashboard) and filled PDFs.

from __future__ import annotations

import io
import os
import zipfile
from datetime import datetime
from typing import Optional, Dict, Any, Iterable, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# -------------------------
# Logging (use your helper if present)
# -------------------------
try:
    from debug_logging import get_logger, log_time, log_df  # optional utility in your repo
    log = get_logger("aca1095")
except Exception:  # fallback
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("aca1095")
    def log_time(_log, label):
        from contextlib import contextmanager
        @contextmanager
        def _cm():
            _log.info(f"{label} - start")
            try:
                yield
            finally:
                _log.info(f"{label} - end")
        return _cm()
    def log_df(_log, df, name):
        try:
            _log.info(f"{name}: shape={getattr(df,'shape',None)} cols={list(getattr(df,'columns',[]))[:8]}")
        except Exception:
            _log.info(f"{name}: (unprintable)")

# -------------------------
# Your modules
# -------------------------
from aca_processing import load_excel
from aca_builder import build_interim, build_final, build_penalty_dashboard
from aca_pdf import save_excel_outputs, fill_pdf_for_employee

# -------------------------
# Helpers
# -------------------------
def _to_bytes(obj):
    """Accept bytes/bytearray/BytesIO/file-like and return raw bytes."""
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if hasattr(obj, "getvalue"):   # BytesIO-like
        return obj.getvalue()
    if hasattr(obj, "read"):       # file-like
        return obj.read()
    raise TypeError(f"Expected bytes-like or buffer, got {type(obj).__name__}")

def _bytes_response(content: bytes, media_type: str, filename: str) -> StreamingResponse:
    return StreamingResponse(
        io.BytesIO(content),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

def _zip_bytes(pairs: Iterable[Tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, b in pairs:
            zf.writestr(name, b)
    return buf.getvalue()

# -------------------------
# Auth (API key via header)
# -------------------------
def get_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    expected = os.getenv("FASTAPI_API_KEY")
    if expected:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="Missing x-api-key")
        if x_api_key != expected:
            raise HTTPException(status_code=403, detail="Invalid x-api-key")

# -------------------------
# App + CORS
# -------------------------
app = FastAPI(title="ACA 1095-C Generator API", version="1.0.0", docs_url="/docs", redoc_url="/redoc")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}

@app.post("/process/excel")
async def process_excel(
    file: UploadFile = File(..., description="Input ACA workbook (.xlsx)"),
    year: Optional[int] = Form(None),
    affordability_threshold: Optional[float] = Form(None),
    _: None = Depends(get_api_key),
):
    """
    Accepts a single ACA input workbook.
    Returns an XLSX with sheets: Interim, Final, [Penalty Dashboard].
    """
    try:
        excel_bytes = await file.read()                        # bytes
        data = load_excel(excel_bytes)                         # dict of DataFrames
        year_used = int(year) if year else datetime.now().year

        # Unpack sheets for builder signature:
        emp_demo   = data.get("emp_demo")
        emp_elig   = data.get("emp_elig")
        emp_enroll = data.get("emp_enroll")
        dep_enroll = data.get("dep_enroll")
        emp_wait   = data.get("emp_wait")

        with log_time(log, "build_interim"):
            interim_df = build_interim(
                emp_demo, emp_elig, emp_enroll, dep_enroll,
                year_used,
                emp_wait=emp_wait,
                affordability_threshold=affordability_threshold,
            )
        log_df(log, interim_df, "Interim")

        final_df = build_final(interim_df)
        log_df(log, final_df, "Final")

        # Penalty dashboard is based on INTERIM
        penalty_df = build_penalty_dashboard(interim_df)
        log_df(log, penalty_df, "Penalty")

        # save_excel_outputs expects ONE dict argument
        outputs = {"Interim": interim_df, "Final": final_df}
        try:
            if penalty_df is not None and hasattr(penalty_df, "empty") and not penalty_df.empty:
                outputs["Penalty Dashboard"] = penalty_df
        except Exception:
            pass

        out_bytes = save_excel_outputs(outputs)  # -> bytes (xlsx)
        return _bytes_response(
            out_bytes,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            f"ACA-Outputs-{year_used}.xlsx",
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to process Excel")
        raise HTTPException(status_code=400, detail=f"Failed to process Excel: {e}")

@app.post("/generate/single")
async def generate_single_pdf(
    file: UploadFile = File(..., description="Input ACA workbook (.xlsx)"),
    pdf: UploadFile  = File(..., description="Blank 1095-C PDF template"),
    year: Optional[int] = Form(None),
    employee_id: Optional[str] = Form(None),
    _: None = Depends(get_api_key),
):
    """
    Returns a single filled (flat) 1095-C PDF for one employee.
    """
    try:
        excel_bytes = await file.read()
        pdf_bytes   = await pdf.read()
        data = load_excel(excel_bytes)
        year_used = int(year) if year else datetime.now().year

        # Unpack + build
        emp_demo   = data.get("emp_demo")
        emp_elig   = data.get("emp_elig")
        emp_enroll = data.get("emp_enroll")
        dep_enroll = data.get("dep_enroll")
        emp_wait   = data.get("emp_wait")

        interim_df = build_interim(emp_demo, emp_elig, emp_enroll, dep_enroll, year_used, emp_wait=emp_wait)
        final_df   = build_final(interim_df)

        # Filter to one employee if provided
        df = final_df
        if employee_id:
            df = final_df[final_df["EmployeeID"].astype(str) == str(employee_id)]
            if df.empty:
                raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found in Final")

        row = df.iloc[0]  # pd.Series

        # Fill PDF; ensure we return BYTES
        editable_name, editable_buf, flat_name, flat_buf = fill_pdf_for_employee(
            pdf_bytes, row, df, year_used, sheets=data
        )
        flat_bytes = _to_bytes(flat_buf)

        return _bytes_response(flat_bytes, "application/pdf", flat_name)

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to generate single PDF")
        raise HTTPException(status_code=400, detail=f"Failed to generate single PDF: {e}")

@app.post("/generate/bulk")
async def generate_bulk_pdfs(
    file: UploadFile = File(..., description="Input ACA workbook (.xlsx)"),
    pdf: UploadFile  = File(..., description="Blank 1095-C PDF template"),
    year: Optional[int] = Form(None),
    _: None = Depends(get_api_key),
):
    """
    Returns a ZIP of filled (flat) 1095-C PDFs for all employees in Final.
    """
    try:
        excel_bytes = await file.read()
        pdf_bytes   = await pdf.read()
        data = load_excel(excel_bytes)
        year_used = int(year) if year else datetime.now().year

        # Unpack + build
        emp_demo   = data.get("emp_demo")
        emp_elig   = data.get("emp_elig")
        emp_enroll = data.get("emp_enroll")
        dep_enroll = data.get("dep_enroll")
        emp_wait   = data.get("emp_wait")

        interim_df = build_interim(emp_demo, emp_elig, emp_enroll, dep_enroll, year_used, emp_wait=emp_wait)
        final_df   = build_final(interim_df)

        out_pairs: list[tuple[str, bytes]] = []
        for _, row in final_df.iterrows():
            try:
                _, _, flat_name, flat_buf = fill_pdf_for_employee(
                    pdf_bytes, row, final_df, year_used, sheets=data
                )
                out_pairs.append((flat_name, _to_bytes(flat_buf)))
            except Exception as inner:
                emp = row.get("EmployeeID", "unknown")
                out_pairs.append((f"ERROR_{emp}.txt", str(inner).encode("utf-8")))

        zip_bytes = _zip_bytes(out_pairs)
        return _bytes_response(zip_bytes, "application/zip", f"ACA-1095C-Bulk-{year_used}.zip")

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to generate bulk PDFs")
        raise HTTPException(status_code=400, detail=f"Failed to generate bulk PDFs: {e}")
