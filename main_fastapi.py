# main_fastapi.py
from __future__ import annotations

import io
import traceback
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

import pandas as pd

from aca_processing import (
    read_input_excel,          # your helper: returns dict of sheets as DataFrames
    preprocess_inputs,         # applies aliases, normalization, waits, etc.
)
from aca_builder import (
    build_interim,             # builds interim grid
    build_final,               # builds final (lines 14/16) from interim
    build_penalty_dashboard,   # optional
)
from aca_pdf import (
    fill_pdf_for_employee,
    save_excel_outputs,
    list_pdf_fields,           # debug helper
)


app = FastAPI(title="ACA 1095-C Backend", version="1.0.0")

# CORS (adjust to your frontend origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


# -------------------------- helpers --------------------------

def _get_bool(form, *names: str, default: bool = False) -> bool:
    for n in names:
        if n in form:
            v = str(form[n]).strip().lower()
            return v in {"1", "true", "t", "yes", "y", "on"}
    return default

def _get_int(form, *names: str, default: Optional[int] = None) -> Optional[int]:
    for n in names:
        if n in form:
            try:
                return int(str(form[n]).strip())
            except Exception:
                return default
    return default

def _get_float(form, *names: str, default: Optional[float] = None) -> Optional[float]:
    for n in names:
        if n in form:
            try:
                return float(str(form[n]).strip())
            except Exception:
                return default
    return default

def _get_str(form, *names: str, default: Optional[str] = None) -> Optional[str]:
    for n in names:
        if n in form:
            v = str(form[n]).strip()
            return v if v != "" else default
    return default


def _xlsx_from_upload(file: UploadFile) -> dict[str, pd.DataFrame]:
    """Read an uploaded Excel file into {sheet_name: DataFrame} (lowercased column names)."""
    with io.BytesIO(file.file.read()) as buf:
        buf.seek(0)
        xl = pd.ExcelFile(buf)
        sheets: dict[str, pd.DataFrame] = {}
        for s in xl.sheet_names:
            df = xl.parse(s)
            # normalize header case
            df.columns = [str(c).strip().lower() for c in df.columns]
            sheets[s.strip()] = df
        return sheets


def _first_visible_sheet_required(xw: pd.ExcelWriter):
    """No-op placeholder; kept in case we later enforce 'at least one visible sheet' logic."""
    return


def _build_everything(
    sheets: dict[str, pd.DataFrame],
    year: int,
    threshold: float,
    include_penalty: bool,
):
    """
    Orchestrates processing -> interim -> final (+ optional penalty).
    """
    # user-specific normalization (aliases, waits, etc.)
    inputs = preprocess_inputs(sheets)

    # pipeline
    interim = build_interim(inputs, year=year, affordability_threshold=threshold)
    final = build_final(interim, year=year)

    penalty = None
    if include_penalty:
        try:
            penalty = build_penalty_dashboard(final)
        except Exception:
            penalty = None

    return interim, final, penalty


# -------------------------- API routes --------------------------

@app.post("/debug/pdf_fields")
async def debug_pdf_fields(pdf: UploadFile = File(...)):
    """Upload a blank 1095-C PDF; returns a dict of its AcroForm fields."""
    try:
        pdf_bytes = await pdf.read()
        fields = list_pdf_fields(pdf_bytes)
        return JSONResponse(fields)
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


@app.post("/generate/single")
async def generate_single(
    # files
    excel: UploadFile = File(..., description="Input XLSX"),
    pdf: UploadFile | None = File(None, description="Blank 1095-C PDF"),
    # optional filter
    employee_id: Optional[str] = Form(None, alias="employeeId"),
    # options (accept multiple aliases so 422 never happens due to key mismatch)
    filing_year: Optional[int] = Form(None, alias="filingYear"),
    year: Optional[int] = Form(None),
    affordability_threshold: Optional[float] = Form(None, alias="affordabilityThreshold"),
    threshold: Optional[float] = Form(None),
    include_penalty_dashboard: Optional[str] = Form(None, alias="includePenaltyDashboard"),
    includePenalty: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),  # "single" | "bulk" | "process" (ignored here but harmless)
):
    try:
        # Parse options defensively
        used_year = filing_year or year
        if used_year is None:
            return JSONResponse({"error": "Missing filing year"}, status_code=422)
        used_threshold = affordability_threshold if affordability_threshold is not None else threshold
        if used_threshold is None:
            used_threshold = 50.0  # safe default for UAT

        include_penalty = _get_bool(
            {
                "includePenaltyDashboard": include_penalty_dashboard,
                "includePenalty": includePenalty,
            },
            "includePenaltyDashboard", "includePenalty",
            default=False,
        )

        # Read Excel
        sheets = _xlsx_from_upload(excel)
        if not sheets:
            return JSONResponse({"error": "Excel file has no readable sheets"}, status_code=422)

        # Build data
        interim, final, penalty = _build_everything(
            sheets=sheets, year=int(used_year), threshold=float(used_threshold), include_penalty=include_penalty
        )

        # If no employee_id provided, return Excel bundle only
        if not employee_id:
            out_bytes = save_excel_outputs(
                interim=interim, final=final, year=int(used_year), penalty_dashboard=penalty
            )
            filename = f"ACA_outputs_{used_year}.xlsx"
            return Response(
                content=out_bytes,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        # Generate one employee’s PDFs
        # Find the row to print from final
        emp_id_str = str(employee_id).strip()
        final_emp = final[final["EmployeeID"].astype(str) == emp_id_str]
        if final_emp.empty:
            return JSONResponse({"error": f"EmployeeID {emp_id_str} not found in Final grid"}, status_code=404)

        # Need blank form to fill
        if pdf is None:
            return JSONResponse({"error": "Blank 1095-C PDF is required when EmployeeID is provided"}, status_code=422)
        pdf_bytes = await pdf.read()

        # Optional enrollment sheets for Part III
        emp_enroll = sheets.get("Emp Enrollment") or sheets.get("emp enrollment")
        dep_enroll = sheets.get("Dep Enrollment") or sheets.get("dep enrollment")

        # A single employee row from a demographic sheet (for Part I), fallback to final row
        demo = sheets.get("EmployeeID") or sheets.get("employeeid") or sheets.get("demographics") or sheets.get("Demographics")
        if demo is not None and not demo.empty:
            dem_row = demo[demo["employeeid"].astype(str) == emp_id_str]
            emp_row = dem_row.iloc[0] if not dem_row.empty else final_emp.iloc[0]
        else:
            emp_row = final_emp.iloc[0]

        editable_name, editable, flattened_name, flattened = fill_pdf_for_employee(
            pdf_bytes=pdf_bytes,
            emp_row=emp_row,
            final_df_emp=final_emp,
            year_used=int(used_year),
            emp_enroll_emp=emp_enroll,
            dep_enroll_emp=dep_enroll,
        )

        # Return the flattened by default (same as editable if you used the safe version)
        return StreamingResponse(
            flattened,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{flattened_name}"'},
        )

    except Exception as e:
        return JSONResponse(
            {"error": str(e), "trace": traceback.format_exc()},
            status_code=500,
        )


@app.post("/generate/bulk")
async def generate_bulk(
    excel: UploadFile = File(...),
    pdf: UploadFile = File(...),
    filing_year: Optional[int] = Form(None, alias="filingYear"),
    year: Optional[int] = Form(None),
    affordability_threshold: Optional[float] = Form(None, alias="affordabilityThreshold"),
    threshold: Optional[float] = Form(None),
    include_penalty_dashboard: Optional[str] = Form(None, alias="includePenaltyDashboard"),
    includePenalty: Optional[str] = Form(None),
):
    """
    Stub for bulk ZIP (left minimal; returns Excel outputs for now).
    """
    try:
        used_year = filing_year or year
        if used_year is None:
            return JSONResponse({"error": "Missing filing year"}, status_code=422)
        used_threshold = affordability_threshold if affordability_threshold is not None else threshold
        if used_threshold is None:
            used_threshold = 50.0

        include_penalty = _get_bool(
            {"includePenaltyDashboard": include_penalty_dashboard, "includePenalty": includePenalty},
            "includePenaltyDashboard", "includePenalty", default=False
        )

        sheets = _xlsx_from_upload(excel)
        interim, final, penalty = _build_everything(
            sheets=sheets, year=int(used_year), threshold=float(used_threshold), include_penalty=include_penalty
        )
        out_bytes = save_excel_outputs(interim=interim, final=final, year=int(used_year), penalty_dashboard=penalty)

        filename = f"ACA_outputs_{used_year}.xlsx"
        return Response(
            content=out_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except Exception as e:
        return JSONResponse(
            {"error": str(e), "trace": traceback.format_exc()},
            status_code=500,
        )


@app.post("/process/excel")
async def process_excel(
    excel: UploadFile = File(...),
    filing_year: Optional[int] = Form(None, alias="filingYear"),
    year: Optional[int] = Form(None),
    affordability_threshold: Optional[float] = Form(None, alias="affordabilityThreshold"),
    threshold: Optional[float] = Form(None),
    include_penalty_dashboard: Optional[str] = Form(None, alias="includePenaltyDashboard"),
    includePenalty: Optional[str] = Form(None),
):
    """
    Returns an Excel workbook (Interim + Final [+ optional Penalty Dashboard]).
    """
    try:
        used_year = filing_year or year
        if used_year is None:
            return JSONResponse({"error": "Missing filing year"}, status_code=422)
        used_threshold = affordability_threshold if affordability_threshold is not None else threshold
        if used_threshold is None:
            used_threshold = 50.0

        include_penalty = _get_bool(
            {"includePenaltyDashboard": include_penalty_dashboard, "includePenalty": includePenalty},
            "includePenaltyDashboard", "includePenalty", default=False
        )

        sheets = _xlsx_from_upload(excel)
        interim, final, penalty = _build_everything(
            sheets=sheets, year=int(used_year), threshold=float(used_threshold), include_penalty=include_penalty
        )
        out_bytes = save_excel_outputs(interim=interim, final=final, year=int(used_year), penalty_dashboard=penalty)
        filename = f"ACA_outputs_{used_year}.xlsx"
        return Response(
            content=out_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        return JSONResponse(
            {"error": str(e), "trace": traceback.format_exc()},
            status_code=500,
        )
