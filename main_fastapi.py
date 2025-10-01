# main_fastapi.py
# FastAPI service for ACA 1095: Excel -> (Final, Interim, Penalty Dashboard) and PDF generation.

from __future__ import annotations

import io
import zipfile
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic_settings import BaseSettings

# NOTE: Do NOT import RunConfig from aca_core (it no longer exists there).
from aca_core import (
    load_excel,
    build_interim,
    build_final,
    build_penalty_dashboard,
    save_excel_outputs,
    fill_pdf_for_employee,
)

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aca1095")


# =========================
# Local RunConfig (compat shim)
# =========================
@dataclass
class RunConfig:
    year: Optional[int]
    aca_mode: str
    affordability_threshold: float
    penalty_a_amount: float
    penalty_b_amount: float


# =========================
# Settings & App
# =========================

class Settings(BaseSettings):
    FASTAPI_API_KEY: str = ""                  # required in prod; empty disables auth
    ACA_MODE: str = "SIMPLIFIED"               # SIMPLIFIED | IRS_STRICT
    FILING_YEAR: Optional[int] = None          # if None, auto-pick from data
    AFFORDABILITY_THRESHOLD: float = 50.00     # used only in SIMPLIFIED
    PENALTY_A_AMOUNT: float = 241.67           # display-only in dashboard
    PENALTY_B_AMOUNT: float = 362.50           # display-only in dashboard
    APP_VERSION: str = "1.1.0"

settings = Settings()
app = FastAPI(title="ACA 1095 Backend", version=settings.APP_VERSION)


def _require_api_key(x_api_key: Optional[str] = Header(None)):
    if settings.FASTAPI_API_KEY and (x_api_key != settings.FASTAPI_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


def _parse_bool(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _resolve_run_config(
    aca_mode: Optional[str],
    filing_year: Optional[str],
    affordability_threshold: Optional[str],
) -> RunConfig:
    mode = (aca_mode or settings.ACA_MODE or "SIMPLIFIED").upper()
    try:
        year = int(filing_year) if filing_year not in (None, "", "null") else settings.FILING_YEAR
    except Exception:
        year = settings.FILING_YEAR
    try:
        thr = float(affordability_threshold) if affordability_threshold not in (None, "", "null") else settings.AFFORDABILITY_THRESHOLD
    except Exception:
        thr = settings.AFFORDABILITY_THRESHOLD

    return RunConfig(
        year=year,
        aca_mode=mode,
        affordability_threshold=thr,
        penalty_a_amount=float(settings.PENALTY_A_AMOUNT),
        penalty_b_amount=float(settings.PENALTY_B_AMOUNT),
    )


def _err_response(ctx: str, e: Exception, status: int = 500):
    # Log full traceback server-side and return a compact error + first 4k chars of trace to caller
    log.exception("%s failed", ctx)
    return JSONResponse(
        status_code=status,
        content={
            "ok": False,
            "error": str(e) or "<no message>",
            "trace": traceback.format_exc()[:4000],
            "where": ctx,
        },
    )


# =========================
# Routes
# =========================

@app.get("/health", dependencies=[Depends(_require_api_key)])
def health():
    return {
        "ok": True,
        "version": settings.APP_VERSION,
        "mode": settings.ACA_MODE.upper(),
        "default_year": settings.FILING_YEAR,
        "threshold": f"{settings.AFFORDABILITY_THRESHOLD:.2f}",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/final_xlsx", dependencies=[Depends(_require_api_key)])
@app.post("/final_and_interim", dependencies=[Depends(_require_api_key)])
async def final_and_interim(
    excel: UploadFile = File(..., description="Input workbook (.xlsx)"),
    # Optional run-time controls from the UI:
    aca_mode: Optional[str] = Form(None),
    filing_year: Optional[str] = Form(None),
    affordability_threshold: Optional[str] = Form(None),
    include_penalty_dashboard: Optional[str] = Form("true"),
):
    try:
        excel_bytes = await excel.read()
        frames = load_excel(excel_bytes)

        cfg = _resolve_run_config(aca_mode, filing_year, affordability_threshold)

        # Build Interim (includes Line14/15/16 + Penalty flags/reasons)
        interim = build_interim(
            frames["emp_demo"],
            frames["emp_status"],
            frames["emp_elig"],
            frames["emp_enroll"],
            frames["dep_enroll"],
            cfg=cfg,
        )

        # Decide year used (cfg may have None -> build_interim auto-picked)
        year_used = (
            int(interim["year"].iloc[0])
            if ("year" in interim.columns and not interim.empty)
            else (cfg.year or settings.FILING_YEAR or datetime.utcnow().year)
        )

        # Final table
        final = build_final(interim)

        # Optional Penalty Dashboard
        penalty_df = None
        if _parse_bool(include_penalty_dashboard):
            penalty_df = build_penalty_dashboard(
                interim,
                year=year_used,
                penalty_a_amt=cfg.penalty_a_amount,
                penalty_b_amt=cfg.penalty_b_amount,
            )

        # Save workbook
        xlsx_bytes = save_excel_outputs(
            interim=interim,
            final=final,
            year=year_used,
            penalty_dashboard=penalty_df,
        )

        filename = (
            f"Final_Interim_{year_used}.xlsx"
            if penalty_df is None
            else f"Final_Interim_Penalty_{year_used}.xlsx"
        )
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Backend-Mode": cfg.aca_mode,
            "X-Backend-Year": str(year_used),
        }
        return StreamingResponse(
            io.BytesIO(xlsx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers,
        )
    except HTTPException:
        raise
    except Exception as e:
        return _err_response("final_and_interim", e)


@app.post("/generate/single", dependencies=[Depends(_require_api_key)])
async def generate_single_pdf(
    excel: UploadFile = File(..., description="Input workbook (.xlsx)"),
    pdf: UploadFile = File(..., description="Blank 1095-C PDF form"),
    employee_id: Optional[str] = Form(None),
    flattened_only: Optional[str] = Form("true"),
    # Optional run-time controls
    aca_mode: Optional[str] = Form(None),
    filing_year: Optional[str] = Form(None),
    affordability_threshold: Optional[str] = Form(None),
):
    try:
        excel_bytes = await excel.read()
        pdf_bytes = await pdf.read()

        frames = load_excel(excel_bytes)
        cfg = _resolve_run_config(aca_mode, filing_year, affordability_threshold)

        interim = build_interim(
            frames["emp_demo"],
            frames["emp_status"],
            frames["emp_elig"],
            frames["emp_enroll"],
            frames["dep_enroll"],
            cfg=cfg,
        )
        year_used = (
            int(interim["year"].iloc[0])
            if ("year" in interim.columns and not interim.empty)
            else (cfg.year or settings.FILING_YEAR or datetime.utcnow().year)
        )

        # Pick employee
        emp_ids = sorted(set(interim["employeeid"].astype(str))) if not interim.empty else []
        target_emp = employee_id or (emp_ids[0] if emp_ids else None)
        if not target_emp:
            raise HTTPException(status_code=400, detail="No employees found in the input workbook.")

        # Filter rows for the employee
        interim_emp = interim[interim["employeeid"].astype(str) == str(target_emp)]
        if interim_emp.empty:
            raise HTTPException(status_code=404, detail=f"EmployeeID {target_emp} not found.")

        final_emp = build_final(interim_emp)

        # Emp demographic slice (Part I)
        demo_emp = frames["emp_demo"]
        if not demo_emp.empty:
            demo_emp = demo_emp[demo_emp["employeeid"].astype(str) == str(target_emp)]

        editable_name, editable_bytes, flat_name, flat_bytes = fill_pdf_for_employee(
            blank_pdf_bytes=pdf_bytes,
            emp_demo=demo_emp,
            final_df_emp=final_emp,
            year_used=year_used,
        )

        if _parse_bool(flattened_only):
            filename = flat_name
            stream = flat_bytes
        else:
            filename = editable_name
            stream = editable_bytes

        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Backend-Mode": cfg.aca_mode,
            "X-Backend-Year": str(year_used),
        }
        return StreamingResponse(stream, media_type="application/pdf", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        return _err_response("generate_single", e)


@app.post("/generate/zip", dependencies=[Depends(_require_api_key)])
async def generate_zip_pdfs(
    excel: UploadFile = File(..., description="Input workbook (.xlsx)"),
    pdf: UploadFile = File(..., description="Blank 1095-C PDF form"),
    # Optional run-time controls
    aca_mode: Optional[str] = Form(None),
    filing_year: Optional[str] = Form(None),
    affordability_threshold: Optional[str] = Form(None),
    flattened_only: Optional[str] = Form("true"),
):
    try:
        excel_bytes = await excel.read()
        pdf_bytes = await pdf.read()

        frames = load_excel(excel_bytes)
        cfg = _resolve_run_config(aca_mode, filing_year, affordability_threshold)

        interim = build_interim(
            frames["emp_demo"],
            frames["emp_status"],
            frames["emp_elig"],
            frames["emp_enroll"],
            frames["dep_enroll"],
            cfg=cfg,
        )
        year_used = (
            int(interim["year"].iloc[0])
            if ("year" in interim.columns and not interim.empty)
            else (cfg.year or settings.FILING_YEAR or datetime.utcnow().year)
        )

        # Group by employee, generate PDFs, zip them
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for emp in sorted(set(interim["employeeid"].astype(str))):
                interim_emp = interim[interim["employeeid"].astype(str) == emp]
                final_emp = build_final(interim_emp)

                demo_emp = frames["emp_demo"]
                if not demo_emp.empty:
                    demo_emp = demo_emp[demo_emp["employeeid"].astype(str) == emp]

                editable_name, editable_bytes, flat_name, flat_bytes = fill_pdf_for_employee(
                    blank_pdf_bytes=pdf_bytes,
                    emp_demo=demo_emp,
                    final_df_emp=final_emp,
                    year_used=year_used,
                )

                if _parse_bool(flattened_only):
                    zf.writestr(flat_name, flat_bytes.getvalue())
                else:
                    zf.writestr(editable_name, editable_bytes.getvalue())

        mem.seek(0)
        headers = {
            "Content-Disposition": f'attachment; filename="1095c_all_{year_used}.zip"',
            "X-Backend-Mode": cfg.aca_mode,
            "X-Backend-Year": str(year_used),
        }
        return StreamingResponse(mem, media_type="application/zip", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        return _err_response("generate_zip", e)


# Convenience alias for your previous naming (if any)
@app.post("/process/excel", dependencies=[Depends(_require_api_key)])
async def process_excel_alias(
    excel: UploadFile = File(...),
    aca_mode: Optional[str] = Form(None),
    filing_year: Optional[str] = Form(None),
    affordability_threshold: Optional[str] = Form(None),
    include_penalty_dashboard: Optional[str] = Form("true"),
):
    # Forward to /final_and_interim to keep backward compatibility with your Vercel proxy
    return await final_and_interim(
        excel=excel,
        aca_mode=aca_mode,
        filing_year=filing_year,
        affordability_threshold=affordability_threshold,
        include_penalty_dashboard=include_penalty_dashboard,
    )
