# main_fastapi.py
from __future__ import annotations

import io
import json
import os
import zipfile
from typing import Optional, List

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from aca_processing import (
    load_excel,
    prepare_inputs,
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
    save_excel_outputs,
    fill_pdf_for_employee,
)

# ----------------------------
# App & Config
# ----------------------------

def _get_api_keys() -> List[str]:
    raw = os.getenv("API_KEYS", "supersecret-key-123")
    return [k.strip() for k in raw.split(",") if k.strip()]

def require_api_key(request: Request):
    key = request.headers.get("x-api-key")
    if key not in _get_api_keys():
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

def _cors_origins() -> List[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "*")
    if raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]

app = FastAPI(title="ACA 1095-C Builder API", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Routes
# ----------------------------

@app.get("/health")
def health(_: bool = Depends(require_api_key)):
    return {"ok": True}

@app.post("/process/excel")
async def process_excel(
    excel: UploadFile = File(...),
    filing_year: Optional[int] = Form(None),
    affordability_threshold: Optional[float] = Form(None),
    include_penalty_dashboard: Optional[bool] = Form(True),
    _: bool = Depends(require_api_key),
):
    """
    In: Excel (.xlsx)
    Out: Excel with 2-3 sheets: Interim, Final, (optional) Penalty Dashboard
    """
    if not excel.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Upload a .xlsx file")
    excel_bytes = await excel.read()

    try:
        data = load_excel(excel_bytes)
        emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, _ = prepare_inputs(data)

        year_used = filing_year or choose_report_year(emp_elig)
        aff_thr = affordability_threshold if affordability_threshold is not None else None

        # --- KEYWORD-ONLY calls ---
        interim_df = build_interim(
            year=year_used,
            demo=emp_demo,
            status=emp_status,
            elig=emp_elig,
            enroll_emp=emp_enroll,
            enroll_dep=dep_enroll,
            affordability_threshold=aff_thr if aff_thr is not None else 50.0,
        )
        final_df = build_final(
            interim=interim_df,
            year=year_used,
        )
        penalty_df = None
        if include_penalty_dashboard:
            penalty_df = build_penalty_dashboard(
                interim=interim_df,
                year=year_used,
            )

        xlsx_buf = save_excel_outputs(
            interim=interim_df,
            final=final_df,
            year=year_used,
            penalty_dashboard=penalty_df,
        )
        headers = {
            "Content-Disposition": f'attachment; filename="final_interim_penalty_{year_used}.xlsx"'
        }
        xlsx_buf.seek(0)
        return StreamingResponse(
            xlsx_buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"process_excel failed: {e}")

@app.post("/generate/single")
async def generate_single(
    excel: UploadFile = File(...),
    pdf: UploadFile = File(...),
    employee_id: Optional[str] = Form(None),
    filing_year: Optional[int] = Form(None),
    affordability_threshold: Optional[float] = Form(None),
    flattened_only: Optional[bool] = Form(True),
    _: bool = Depends(require_api_key),
):
    """
    In: Excel, base IRS 1095-C PDF, (optional) employee_id
    Out: a flattened PDF (or zip with editable+flattened when flattened_only=false)
    """
    if not excel.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Upload a .xlsx file")
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a base 1095-C .pdf file")

    excel_bytes = await excel.read()
    pdf_bytes = await pdf.read()

    try:
        data = load_excel(excel_bytes)
        emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, _ = prepare_inputs(data)

        year_used = filing_year or choose_report_year(emp_elig)
        aff_thr = affordability_threshold if affordability_threshold is not None else None

        interim_df = build_interim(
            year=year_used,
            demo=emp_demo,
            status=emp_status,
            elig=emp_elig,
            enroll_emp=emp_enroll,
            enroll_dep=dep_enroll,
            affordability_threshold=aff_thr if aff_thr is not None else 50.0,
        )
        final_df = build_final(
            interim=interim_df,
            year=year_used,
        )

        # pick employee
        all_ids = [str(x) for x in sorted(set(interim_df["employeeid"].astype(str)))]
        target_id = _coerce_str(employee_id) if employee_id is not None else (all_ids[0] if all_ids else None)
        if not target_id:
            raise HTTPException(status_code=400, detail="No employee rows found.")

        emp_row = emp_demo[emp_demo["employeeid"].astype(str) == target_id]
        if emp_row.empty:
            raise HTTPException(status_code=400, detail=f"Employee {target_id} not found")

        emp_final = final_df[final_df["employeeid"].astype(str) == target_id].copy()
        if emp_final.empty:
            # build an empty shell for safety
            emp_final = pd.DataFrame({
                "employeeid": [target_id] * len(MONTHS),
                "month": list(range(1, 13)),
                "line14": ["" for _ in MONTHS],
                "line16": ["" for _ in MONTHS],
                "year": [year_used] * len(MONTHS),
            })

        editable_name, editable_bytes, flat_name, flat_bytes = fill_pdf_for_employee(
            pdf_bytes=pdf_bytes,
            emp_demo_row=emp_row.iloc[0],
            emp_final_df=emp_final,
            year=year_used,
        )

        if flattened_only:
            headers = {"Content-Disposition": f'attachment; filename="{flat_name}"'}
            flat_bytes.seek(0)
            return StreamingResponse(flat_bytes, media_type="application/pdf", headers=headers)
        else:
            # return both editable & flattened
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
                editable_bytes.seek(0)
                flat_bytes.seek(0)
                z.writestr(editable_name, editable_bytes.read())
                z.writestr(flat_name, flat_bytes.read())
            zip_buf.seek(0)
            headers = {"Content-Disposition": f'attachment; filename="1095c_single_{target_id}_{year_used}.zip"'}
            return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Single generation failed: {e}")

@app.post("/generate/bulk")
async def generate_bulk(
    excel: UploadFile = File(...),
    pdf: UploadFile = File(...),
    employee_ids: Optional[str] = Form(None),   # JSON array string or None
    filing_year: Optional[int] = Form(None),
    affordability_threshold: Optional[float] = Form(None),
    _: bool = Depends(require_api_key),
):
    """
    In: Excel, base IRS 1095-C PDF, optional employee_ids JSON array
    Out: ZIP of flattened PDFs
    """
    if not excel.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Upload a .xlsx file")
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a base 1095-C .pdf file")

    excel_bytes = await excel.read()
    pdf_bytes = await pdf.read()

    try:
        data = load_excel(excel_bytes)
        emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, _ = prepare_inputs(data)

        year_used = filing_year or choose_report_year(emp_elig)
        aff_thr = affordability_threshold if affordability_threshold is not None else None

        interim_df = build_interim(
            year=year_used,
            demo=emp_demo,
            status=emp_status,
            elig=emp_elig,
            enroll_emp=emp_enroll,
            enroll_dep=dep_enroll,
            affordability_threshold=aff_thr if aff_thr is not None else 50.0,
        )
        final_df = build_final(
            interim=interim_df,
            year=year_used,
        )

        if employee_ids:
            try:
                sel = json.loads(employee_ids)
                if not isinstance(sel, list):
                    raise ValueError
                wanted = {str(_coerce_str(x)) for x in sel}
            except Exception:
                raise HTTPException(status_code=400, detail="employee_ids must be a JSON array")
        else:
            wanted = {str(x) for x in sorted(set(final_df["employeeid"].astype(str)))}

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
            for emp_id in sorted(wanted):
                row = emp_demo[emp_demo["employeeid"].astype(str) == emp_id]
                if row.empty:
                    continue
                emp_final = final_df[final_df["employeeid"].astype(str) == emp_id].copy()
                if emp_final.empty:
                    emp_final = pd.DataFrame({
                        "employeeid": [emp_id] * len(MONTHS),
                        "month": list(range(1, 13)),
                        "line14": ["" for _ in MONTHS],
                        "line16": ["" for _ in MONTHS],
                        "year": [year_used] * len(MONTHS),
                    })

                _, _, flat_name, flat_bytes = fill_pdf_for_employee(
                    pdf_bytes=pdf_bytes,
                    emp_demo_row=row.iloc[0],
                    emp_final_df=emp_final,
                    year=year_used,
                )
                flat_bytes.seek(0)
                z.writestr(flat_name, flat_bytes.read())

        zip_buf.seek(0)
        headers = {"Content-Disposition": f'attachment; filename="1095c_bulk_{year_used}.zip"'}
        return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Bulk generation failed: {e}")
