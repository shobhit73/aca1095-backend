# main_fastapi.py
# FastAPI service for ACA 1095: Excel -> (Final, Interim, Penalty Dashboard) and PDF generation.
from __future__ import annotations

import io
import os
import zipfile
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ---- modular stack (NOTE: we only import prepare_inputs)
from aca_processing import prepare_inputs
from aca_builder import build_interim, build_final, build_penalty_dashboard
from aca_pdf import fill_pdf_for_employee

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aca1095")

# ---------- config ----------
API_KEY_ENV = "API_KEY"
DEFAULT_OUTPUT_FILENAME = "ACA1095_Output.xlsx"


def require_api_key(x_api_key: Optional[str]) -> None:
    """Simple API key check via header x-api-key."""
    expected = os.getenv(API_KEY_ENV, "").strip()
    if expected:
        if not x_api_key or x_api_key.strip() != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")
    # if no API key set in env, allow all (dev mode)


# ---------- year detector (local helper to avoid importing it) ----------
DATE_COLS_BY_SHEET = {
    "emp_status": ["statusstartdate", "statusenddate", "hiredate", "terminationdate"],
    "emp_elig": ["eligibilitystartdate", "eligibilityenddate"],
    "emp_enroll": ["enrollmentstartdate", "enrollmentenddate"],
}

def _collect_years(df: pd.DataFrame, cols: List[str]) -> List[int]:
    years: List[int] = []
    if df is None or df.empty:
        return years
    for c in cols:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            if hasattr(s, "dt"):
                ys = s.dt.year.dropna().astype(int).tolist()
                years.extend(ys)
    return years

def detect_year_from_inputs(
    emp_status: pd.DataFrame,
    emp_elig: pd.DataFrame,
    emp_enroll: pd.DataFrame,
    fallback: Optional[int] = None,
) -> int:
    """
    Heuristic:
      1) Gather years from known date columns across Status, Eligibility, Enrollment.
      2) Prefer the mode (most frequent year) between now-1 and now+1.
      3) Else fallback to the max seen, else current year or provided fallback.
    """
    now_year = datetime.utcnow().year
    candidates: List[int] = []
    candidates += _collect_years(emp_status, DATE_COLS_BY_SHEET["emp_status"])
    candidates += _collect_years(emp_elig, DATE_COLS_BY_SHEET["emp_elig"])
    candidates += _collect_years(emp_enroll, DATE_COLS_BY_SHEET["emp_enroll"])

    if candidates:
        # constrain to a sensible window (last year..next year) but keep a backup
        window = [y for y in candidates if now_year - 1 <= y <= now_year + 1]
        pool = window or candidates
        if pool:
            # mode with tie -> max
            s = pd.Series(pool)
            try:
                mode_vals = s.mode()
                if len(mode_vals) > 0:
                    return int(mode_vals.iloc[-1])
            except Exception:
                pass
            try:
                return int(max(pool))
            except Exception:
                pass

    return int(fallback or now_year)


# ---------- app ----------
app = FastAPI(title="ACA 1095-C Builder API (modular stack)", version="1.0.1")

# CORS (open-by-default; restrict if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Routes
# =========================

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "aca1095-modular", "time": datetime.utcnow().isoformat() + "Z"}


@app.post("/process/excel")
async def process_excel(
    file: UploadFile = File(..., description="Input Excel workbook"),
    year: Optional[int] = Form(None, description="Override tax year (e.g., 2025)"),
    x_api_key: Optional[str] = Header(None),
):
    """
    Parse Excel -> build Interim, Final, Penalty Dashboard -> return a single Excel (xlsx).
    Accepts Emp Wait Period sheet (EmployeeID, EffectiveDate, Wait Period).
    """
    require_api_key(x_api_key)

    try:
        raw = await file.read()
        data = io.BytesIO(raw)

        # Prepare inputs (7 values incl. Emp Wait Period)
        (
            emp_demo,
            emp_status,
            emp_elig,
            emp_enroll,
            dep_enroll,
            pay_deductions,
            emp_wait,
        ) = prepare_inputs(data)

        # Determine year if not provided
        year_used = year or detect_year_from_inputs(emp_status, emp_elig, emp_enroll)

        # Build interim (thread Emp Wait Period in)
        interim_df = build_interim(
            emp_demo=emp_demo,
            emp_status=emp_status,
            emp_elig=emp_elig,
            emp_enroll=emp_enroll,
            dep_enroll=dep_enroll,
            year=year_used,
            emp_wait_period=emp_wait,  # authoritative waiting window
        )

        # Final + Penalty
        final_df = build_final(interim_df)
        penalty_df = build_penalty_dashboard(interim_df)

        # Write to an xlsx in-memory
        out_buf = io.BytesIO()
        with pd.ExcelWriter(out_buf, engine="xlsxwriter") as writer:
            final_df.to_excel(writer, index=False, sheet_name="Final")
            interim_df.to_excel(writer, index=False, sheet_name="Interim")
            penalty_df.to_excel(writer, index=False, sheet_name="PenaltyDashboard")
        out_buf.seek(0)

        headers = {
            "Content-Disposition": f'attachment; filename="{DEFAULT_OUTPUT_FILENAME}"'
        }
        return StreamingResponse(out_buf, headers=headers, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to process Excel")
        return JSONResponse(status_code=400, content={"error": f"Failed to process Excel: {e}"})


@app.post("/generate/single")
async def generate_single_pdf(
    excel_file: UploadFile = File(..., description="Input Excel workbook"),
    irs_pdf_template: UploadFile = File(..., description="Blank IRS 1095-C PDF (2024/2025 form)"),
    employee_id: Optional[str] = Form(None, description="Target EmployeeID; if None, first employee"),
    year: Optional[int] = Form(None, description="Override tax year (e.g., 2025)"),
    x_api_key: Optional[str] = Header(None),
):
    """
    Given Excel + blank IRS PDF, fill a single 1095-C PDF for one employee.
    Uses Interim/Final from modular builder. Supports Part III (covered individuals).
    """
    require_api_key(x_api_key)

    try:
        # Read inputs
        excel_bytes = await excel_file.read()
        pdf_bytes = await irs_pdf_template.read()
        excel_io = io.BytesIO(excel_bytes)

        # Prepare 7 inputs (with Emp Wait Period)
        (
            emp_demo,
            emp_status,
            emp_elig,
            emp_enroll,
            dep_enroll,
            pay_deductions,
            emp_wait,
        ) = prepare_inputs(excel_io)

        # Determine year
        year_used = year or detect_year_from_inputs(emp_status, emp_elig, emp_enroll)

        # Build interim/final
        interim_df = build_interim(
            emp_demo=emp_demo,
            emp_status=emp_status,
            emp_elig=emp_elig,
            emp_enroll=emp_enroll,
            dep_enroll=dep_enroll,
            year=year_used,
            emp_wait_period=emp_wait,
        )
        final_df = build_final(interim_df)

        # Decide employee (default to first in Final)
        target_emp = (employee_id or
                      (str(final_df.iloc[0]["EmployeeID"]) if not final_df.empty else None))
        if not target_emp:
            raise HTTPException(status_code=400, detail="No employees found in input.")

        # Fill PDF for that employee (aca_pdf handles Part III & month boxes)
        out_pdf = fill_pdf_for_employee(
            final_df=final_df,
            interim_df=interim_df,
            emp_demo=emp_demo,
            target_employee_id=str(target_emp),
            blank_pdf_bytes=pdf_bytes,
            year=year_used,
        )

        headers = {
            "Content-Disposition": f'attachment; filename="1095C_{target_emp}.pdf"'
        }
        return StreamingResponse(io.BytesIO(out_pdf), headers=headers, media_type="application/pdf")

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to generate single PDF")
        return JSONResponse(status_code=400, content={"error": f"Failed to generate PDF: {e}"})


@app.post("/generate/zip")
async def generate_zip_pdfs(
    excel_file: UploadFile = File(..., description="Input Excel workbook"),
    irs_pdf_template: UploadFile = File(..., description="Blank IRS 1095-C PDF"),
    year: Optional[int] = Form(None, description="Override tax year (e.g., 2025)"),
    x_api_key: Optional[str] = Header(None),
):
    """
    Bulk: for every employee in Final, produce a filled PDF and return a ZIP.
    """
    require_api_key(x_api_key)

    try:
        excel_bytes = await excel_file.read()
        pdf_bytes = await irs_pdf_template.read()
        excel_io = io.BytesIO(excel_bytes)

        # Prepare inputs (7 values)
        (
            emp_demo,
            emp_status,
            emp_elig,
            emp_enroll,
            dep_enroll,
            pay_deductions,
            emp_wait,
        ) = prepare_inputs(excel_io)

        # Determine year
        year_used = year or detect_year_from_inputs(emp_status, emp_elig, emp_enroll)

        # Build interim/final once
        interim_df = build_interim(
            emp_demo=emp_demo,
            emp_status=emp_status,
            emp_elig=emp_elig,
            emp_enroll=emp_enroll,
            dep_enroll=dep_enroll,
            year=year_used,
            emp_wait_period=emp_wait,
        )
        final_df = build_final(interim_df)

        if final_df.empty:
            raise HTTPException(status_code=400, detail="No employees found to generate PDFs.")

        # Build PDFs into a ZIP stream
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for _, row in final_df.iterrows():
                eid = str(row["EmployeeID"])
                try:
                    pdf_bytes_filled = fill_pdf_for_employee(
                        final_df=final_df,
                        interim_df=interim_df,
                        emp_demo=emp_demo,
                        target_employee_id=eid,
                        blank_pdf_bytes=pdf_bytes,
                        year=year_used,
                    )
                    zf.writestr(f"1095C_{eid}.pdf", pdf_bytes_filled)
                except Exception as per_emp_err:
                    zf.writestr(f"ERROR_{eid}.txt", f"Failed to build PDF for {eid}: {per_emp_err}")

        zip_buf.seek(0)
        headers = {"Content-Disposition": 'attachment; filename="1095C_PDFs.zip"'}
        return StreamingResponse(zip_buf, headers=headers, media_type="application/zip")

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Failed to generate ZIP")
        return JSONResponse(status_code=400, content={"error": f"Failed to generate ZIP: {e}"})


# =========================
# Render notes
# =========================
# Start command:
#   uvicorn main_fastapi:app --host 0.0.0.0 --port 10000
# Env:
#   API_KEY=<your key>  (optional; if blank, all calls allowed)
#   CORS_ALLOW_ORIGINS=https://your-frontend.vercel.app
