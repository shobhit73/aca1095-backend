# main_fastapi.py
from __future__ import annotations

import io
import os
import json
import uuid
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from debug_logging import (
    setup_logging, get_logger, set_request_context, clear_request_context,
    log_df, log_time
)

from aca_processing import (
    load_excel,
    prepare_inputs,        # (emp_demo, emp_elig, emp_enroll, dep_enroll, emp_wait)
    choose_report_year,
    MONTHS,
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
)

# ---------- App & Logging ----------
app = FastAPI(title="ACA 1095 Service", version="1.4.0")
setup_logging()                      # reads LOG_LEVEL / LOG_JSON / LOG_FILE
log = get_logger("api")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request ID middleware ----------
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        route = f"{request.method} {request.url.path}"
        set_request_context(rid, route)
        log.info("REQUEST START", extra={"extra_data": {"path": request.url.path}})
        try:
            resp = await call_next(request)
            return resp
        finally:
            log.info("REQUEST END")
            clear_request_context()

app.add_middleware(RequestIDMiddleware)

# ---------- API key guard ----------
def _get_api_key(request: Request):
    keys = os.getenv("API_KEYS", "supersecret-key-123").split(",")
    keys = [k.strip() for k in keys if k.strip()]
    incoming = request.headers.get("x-api-key", "")
    if incoming not in keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return incoming

# ---------- Health ----------
@app.get("/health")
def health():
    return {"ok": True}

# ---------- Excel → xlsx (Final/Interim/Penalty) ----------
@app.post("/process/excel")
async def process_excel(
    file: UploadFile = File(...),
    year: Optional[int] = Form(None),
    x_api_key: str = Header(None)
):
    try:
        excel_bytes = await file.read()
        data = load_excel(excel_bytes)                 # returns parsed sheets
        year_used = year or datetime.now().year

        interim_df = build_interim(data, year_used)    # your existing builder
        final_df = build_final(interim_df, year_used)  # your existing final
        penalty_df = build_penalty_dashboard(final_df) # may be None/empty

        outputs = {"Interim": interim_df, "Final": final_df}
        if penalty_df is not None and not penalty_df.empty:
            outputs["Penalty Dashboard"] = penalty_df

        out_bytes = save_excel_outputs(outputs)        # <-- single dict arg

        return StreamingResponse(
            io.BytesIO(out_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="ACA-Outputs-{year_used}.xlsx"'},
        )
    except Exception as e:
        log.exception("Failed to process Excel")
        raise HTTPException(status_code=400, detail=f"Failed to process Excel: {e}")


    fname = f"final_interim_penalty_{year_used}.xlsx"
    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
    return StreamingResponse(
        io.BytesIO(out_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers
    )

# ---------- Single 1095-C (PDF) ----------
@app.post("/generate/single")
async def generate_single(
    excel: UploadFile = File(...),
    pdf:   UploadFile = File(...),
    employee_id: str | None = Form(None),
    filing_year: Optional[int] = Form(None),
    affordability_threshold: Optional[float] = Form(None),
    flattened_only: str = Form("true"),
    api_key: str = Depends(_get_api_key),
):
    if not excel.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Upload a .xlsx file")
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a 1095-C base PDF")

    excel_bytes = await excel.read()
    pdf_bytes   = await pdf.read()  # IMPORTANT: pass raw bytes to fill_pdf_for_employee
    log.info("generate_single: files received", extra={"extra_data": {"xlsx": excel.filename, "pdf": pdf.filename}})

    try:
        data = load_excel(excel_bytes)
        emp_demo, emp_elig, emp_enroll, dep_enroll, emp_wait = prepare_inputs(data)

        if emp_demo.empty:
            raise HTTPException(status_code=422, detail="No employees in Emp Demographic")

        if not employee_id:
            employee_id = _coerce_str(emp_demo["employeeid"].iloc[0])

        row = emp_demo[emp_demo["employeeid"].astype(str)==str(employee_id)]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"EmployeeID {employee_id} not found")

        year_used = int(filing_year) if filing_year else choose_report_year(emp_elig)

        with log_time(log, "build_interim"):
            interim_df = build_interim(
                emp_demo, emp_elig, emp_enroll, dep_enroll,
                year=year_used,
                emp_wait=emp_wait,
                affordability_threshold=affordability_threshold,
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

        import zipfile
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
        import traceback, sys
        log.exception("generate_single failed")
        print("=== /generate/single FAILED ===", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise HTTPException(status_code=422, detail=f"PDF generation failed: {e}")

# ---------- Bulk 1095-Cs (ZIP) ----------
@app.post("/generate/bulk")
async def generate_bulk(
    excel: UploadFile = File(...),
    pdf:   UploadFile = File(...),
    employee_ids: str | None = Form(None),  # JSON array or None for all
    filing_year: Optional[int] = Form(None),
    affordability_threshold: Optional[float] = Form(None),
    api_key: str = Depends(_get_api_key),
):
    if not excel.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Upload a .xlsx file")
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a 1095-C base PDF")

    excel_bytes = await excel.read()
    pdf_bytes   = await pdf.read()  # IMPORTANT: raw bytes (reused for each employee)
    log.info("generate_bulk: files received", extra={"extra_data": {"xlsx": excel.filename, "pdf": pdf.filename}})

    try:
        data = load_excel(excel_bytes)
        emp_demo, emp_elig, emp_enroll, dep_enroll, emp_wait = prepare_inputs(data)
        if emp_demo.empty:
            raise HTTPException(status_code=422, detail="No employees in Emp Demographic")

        year_used = int(filing_year) if filing_year else choose_report_year(emp_elig)
        all_ids = list(map(str, emp_demo["employeeid"].astype(str).unique()))
        ids = list(map(str, json.loads(employee_ids))) if employee_ids else all_ids

        with log_time(log, "build_interim"):
            interim_df = build_interim(
                emp_demo, emp_elig, emp_enroll, dep_enroll,
                year=year_used,
                emp_wait=emp_wait,
                affordability_threshold=affordability_threshold,
            )
        final_df = build_final(interim_df)

        import zipfile
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
                _, _, flat_name, flat_bytes = fill_pdf_for_employee(
                    pdf_bytes, row.iloc[0], emp_final, year_used
                )
                z.writestr(flat_name, flat_bytes.getvalue())
        zip_buf.seek(0)
        headers = {"Content-Disposition": f'attachment; filename="1095c_bulk_{year_used}.zip"'}
        return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        import traceback, sys
        log.exception("generate_bulk failed")
        print("=== /generate/bulk FAILED ===", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise HTTPException(status_code=422, detail=f"Bulk generation failed: {e}")
