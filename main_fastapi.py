# main_fastapi.py
from __future__ import annotations

import io, zipfile, traceback, logging
from typing import Optional, Iterable

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

import pandas as pd

from aca_builder import (
    load_input_workbook,
    build_interim_df,       # back-compat: accepts sheets-dict OR raw bytes
)
from aca_processing import (
    parse_bool, parse_int, parse_float,
    read_interim_xlsx, build_interim_from_excel_bytes,
    employee_ids_from_interim, build_pdf_payload_from_interim_row,
    safe_fill_pdf_for_employee,                       # robust wrapper to your PDF filler
)

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aca_api")

app = FastAPI(title="ACA 1095 Builder API (3-step)")

# ---------- Error handlers ----------
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    log.error("422 validation: %s", exc.errors())
    return JSONResponse(status_code=422, content={
        "error": "Validation failed",
        "where": "request-validation",
        "detail": exc.errors(),
    })

@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    log.exception("UNHANDLED:")
    return JSONResponse(
        status_code=500,
        content={
            "error": f"{type(exc).__name__}: {exc}",
            "where": "global-exception",
            "trace": traceback.format_exc(),
        },
    )

# =========================================================
# 1) Generate INTERIM ONLY
# =========================================================
@app.post("/interim")
async def interim_endpoint(
    filing_year: str = Form(...),
    excel: UploadFile = File(...),
    affordability_threshold: Optional[str] = Form(None),
    return_json: Optional[str] = Form("false"),
):
    """
    Returns Interim.xlsx (or JSON preview if return_json=true)
    """
    year = parse_int(filing_year, 2025)
    thr = parse_float(affordability_threshold, None)
    want_json = parse_bool(return_json, False)

    excel_bytes = await excel.read()
    if not excel_bytes:
        raise HTTPException(status_code=400, detail="Empty Excel upload")

    try:
        interim = build_interim_from_excel_bytes(year, excel_bytes, thr)
    except Exception as e:
        log.exception("build_interim failed")
        return JSONResponse(status_code=400, content={
            "error": f"Interim build failed: {type(e).__name__}: {e}",
            "where": "build_interim_from_excel_bytes",
            "trace": traceback.format_exc(),
        })

    if want_json:
        return JSONResponse({"year": year, "interim_head": interim.head(50).to_dict(orient="records")})

    # stream an Excel file with only Interim
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        interim.to_excel(xw, index=False, sheet_name="Interim")
    buf.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="Interim_{year}.xlsx"'}
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers=headers)

# =========================================================
# 2) Fill ONE PDF (by EmployeeID)
# =========================================================
@app.post("/pdf/single")
async def pdf_single_endpoint(
    employee_id: str = Form(...),
    blank_pdf: UploadFile = File(...),

    # Provide either interim.xlsx OR (excel + filing_year [+threshold])
    interim: UploadFile | None = File(None),
    excel: UploadFile | None = File(None),
    filing_year: Optional[str] = Form(None),
    affordability_threshold: Optional[str] = Form(None),

    flatten: Optional[str] = Form("true"),
):
    """
    Returns a single filled PDF for the given EmployeeID.

    Preferred: upload 'interim' (generated in Step 1).
    Alternate: upload 'excel' + 'filing_year' (we'll build interim on the fly).
    """
    if not employee_id.strip():
        raise HTTPException(status_code=400, detail="employee_id is required")

    flat = parse_bool(flatten, True)

    # Resolve Interim rows
    interim_df: pd.DataFrame
    if interim is not None:
        interim_bytes = await interim.read()
        if not interim_bytes:
            raise HTTPException(status_code=400, detail="Uploaded interim file is empty")
        try:
            interim_df = read_interim_xlsx(interim_bytes)
        except Exception as e:
            log.exception("read_interim_xlsx failed")
            return JSONResponse(status_code=400, content={
                "error": f"Interim read failed: {type(e).__name__}: {e}",
                "where": "read_interim_xlsx",
                "trace": traceback.format_exc(),
            })
    else:
        if excel is None or filing_year is None:
            raise HTTPException(status_code=400, detail="Either upload 'interim', or provide 'excel' + 'filing_year'")
        year = parse_int(filing_year, 2025)
        thr = parse_float(affordability_threshold, None)
        excel_bytes = await excel.read()
        if not excel_bytes:
            raise HTTPException(status_code=400, detail="Empty Excel upload")
        try:
            interim_df = build_interim_from_excel_bytes(year, excel_bytes, thr)
        except Exception as e:
            log.exception("build_interim_from_excel_bytes failed")
            return JSONResponse(status_code=400, content={
                "error": f"Interim build failed: {type(e).__name__}: {e}",
                "where": "build_interim_from_excel_bytes",
                "trace": traceback.format_exc(),
            })

    # Pick employee row(s)
    sub = interim_df[interim_df["EmployeeID"].astype(str) == str(employee_id).strip()]
    if sub.empty:
        return JSONResponse(status_code=404, content={"error": f"EmployeeID {employee_id} not found in Interim"})

    # Build payload (employee PI, L14/L16 codes, covered individuals placeholder)
    try:
        payload = build_pdf_payload_from_interim_row(sub)
    except Exception as e:
        log.exception("build_pdf_payload_from_interim_row failed")
        return JSONResponse(status_code=400, content={
            "error": f"Could not prepare PDF payload: {type(e).__name__}: {e}",
            "where": "build_pdf_payload_from_interim_row",
            "trace": traceback.format_exc(),
        })

    # Load blank PDF
    blank_pdf_bytes = await blank_pdf.read()
    if not blank_pdf_bytes:
        raise HTTPException(status_code=400, detail="blank_pdf is empty")

    # Fill using tolerant wrapper (works with your existing aca_pdf.fill_pdf_for_employee signature variants)
    try:
        filled_bytes = safe_fill_pdf_for_employee(
            blank_pdf_bytes=blank_pdf_bytes,
            employee_pi=payload["employee_pi"],
            line14_by_month=payload["line14_by_month"],
            line16_by_month=payload["line16_by_month"],
            covered_individuals=payload.get("covered_individuals", []),
            flatten=flat,
        )
    except Exception as e:
        log.exception("safe_fill_pdf_for_employee failed")
        return JSONResponse(status_code=400, content={
            "error": f"PDF generation failed: {type(e).__name__}: {e}",
            "where": "safe_fill_pdf_for_employee",
            "trace": traceback.format_exc(),
        })

    # Stream the single PDF
    headers = {"Content-Disposition": f'attachment; filename="1095C_{employee_id}.pdf"'}
    return StreamingResponse(io.BytesIO(filled_bytes), media_type="application/pdf", headers=headers)

# =========================================================
# 3) BULK PDFs
# =========================================================
@app.post("/pdf/bulk")
async def pdf_bulk_endpoint(
    blank_pdf: UploadFile = File(...),

    # Provide either interim.xlsx OR (excel + filing_year [+threshold])
    interim: UploadFile | None = File(None),
    excel: UploadFile | None = File(None),
    filing_year: Optional[str] = Form(None),
    affordability_threshold: Optional[str] = Form(None),

    # optional filters: comma-separated list of EmployeeIDs
    include_ids: Optional[str] = Form(None),

    flatten: Optional[str] = Form("true"),
):
    """
    Returns a ZIP containing PDFs.
    Preferred: upload 'interim' (generated in Step 1).
    Alternate: upload 'excel' + 'filing_year' (we'll build interim on the fly).
    """
    flat = parse_bool(flatten, True)

    # Resolve Interim DF
    if interim is not None:
        interim_bytes = await interim.read()
        if not interim_bytes:
            raise HTTPException(status_code=400, detail="Uploaded interim file is empty")
        try:
            interim_df = read_interim_xlsx(interim_bytes)
        except Exception as e:
            log.exception("read_interim_xlsx failed")
            return JSONResponse(status_code=400, content={
                "error": f"Interim read failed: {type(e).__name__}: {e}",
                "where": "read_interim_xlsx",
                "trace": traceback.format_exc(),
            })
    else:
        if excel is None or filing_year is None:
            raise HTTPException(status_code=400, detail="Either upload 'interim', or provide 'excel' + 'filing_year'")
        year = parse_int(filing_year, 2025)
        thr = parse_float(affordability_threshold, None)
        excel_bytes = await excel.read()
        if not excel_bytes:
            raise HTTPException(status_code=400, detail="Empty Excel upload")
        try:
            interim_df = build_interim_from_excel_bytes(year, excel_bytes, thr)
        except Exception as e:
            log.exception("build_interim_from_excel_bytes failed")
            return JSONResponse(status_code=400, content={
                "error": f"Interim build failed: {type(e).__name__}: {e}",
                "where": "build_interim_from_excel_bytes",
                "trace": traceback.format_exc(),
            })

    # IDs to include
    ids: Iterable[str] = employee_ids_from_interim(interim_df)
    if include_ids:
        wanted = {s.strip() for s in include_ids.split(",") if s.strip()}
        ids = [i for i in ids if i in wanted]
        if not ids:
            return JSONResponse(status_code=404, content={"error": "No matching EmployeeIDs to generate"})

    # Load blank PDF
    blank_pdf_bytes = await blank_pdf.read()
    if not blank_pdf_bytes:
        raise HTTPException(status_code=400, detail="blank_pdf is empty")

    # Build ZIP
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for emp_id in ids:
            sub = interim_df[interim_df["EmployeeID"].astype(str) == str(emp_id)]
            if sub.empty:
                continue
            try:
                payload = build_pdf_payload_from_interim_row(sub)
                pdf_bytes = safe_fill_pdf_for_employee(
                    blank_pdf_bytes=blank_pdf_bytes,
                    employee_pi=payload["employee_pi"],
                    line14_by_month=payload["line14_by_month"],
                    line16_by_month=payload["line16_by_month"],
                    covered_individuals=payload.get("covered_individuals", []),
                    flatten=flat,
                )
                zf.writestr(f"1095C_{emp_id}.pdf", pdf_bytes)
            except Exception as e:
                # write a small text error file instead of crashing whole batch
                zf.writestr(f"ERROR_{emp_id}.txt", f"Failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")

    zip_buf.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="1095C_bulk.zip"'}
    return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)
