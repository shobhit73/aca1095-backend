# main_fastapi.py  — ACA 1095-C backend (key-agnostic + correct build_interim args)
from __future__ import annotations

import io
import traceback
from typing import Optional, Any, Mapping

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

# Your project modules
from aca_builder import build_interim, build_final, build_penalty_dashboard
from aca_pdf import fill_pdf_for_employee, save_excel_outputs, list_pdf_fields

app = FastAPI(title="ACA 1095-C Backend", version="1.0.2")

# ───────────────── CORS (tighten in prod) ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────── health ─────────────────────────
@app.get("/health")
def health():
    return {"ok": True}

# ───────────────────────── helpers ─────────────────────────
def _to_plain_dict(m: Mapping[str, Any]) -> dict[str, Any]:
    # keep original keys, we only do contains() checks
    return {str(k): m[k] for k in m}

def _try_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def _try_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None

def _truthy(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

def _detect_year(form: Mapping[str, Any], explicit: Optional[int] = None) -> Optional[int]:
    """
    Prefer explicit year (filingYear/year). Otherwise scan any key containing 'year'.
    """
    if explicit is not None:
        return explicit
    for k, v in form.items():
        if "year" in str(k).lower():
            y = _try_int(v)
            if y:
                return y
    return None

def _detect_threshold(form: Mapping[str, Any], explicit: Optional[float] = None) -> Optional[float]:
    if explicit is not None:
        return explicit
    # common names first
    for key in ("affordabilityThreshold", "affordability_threshold", "threshold", "aff_threshold"):
        if key in form:
            t = _try_float(form[key])
            if t is not None:
                return t
    # fallback: any key containing 'threshold'
    for k, v in form.items():
        if "threshold" in str(k).lower():
            t = _try_float(v)
            if t is not None:
                return t
    return None

def _xlsx_from_upload(file: UploadFile) -> dict[str, pd.DataFrame]:
    """
    Read an uploaded Excel file into {sheet_name: DataFrame} using original sheet
    names (trimmed), and lower-cased column headers.
    """
    with io.BytesIO(file.file.read()) as buf:
        buf.seek(0)
        xl = pd.ExcelFile(buf)
        sheets: dict[str, pd.DataFrame] = {}
        for s in xl.sheet_names:
            df = xl.parse(s)
            df.columns = [str(c).strip().lower() for c in df.columns]
            sheets[s.strip()] = df
        return sheets

def _get_sheet(sheets: dict[str, pd.DataFrame], *candidates: str) -> pd.DataFrame:
    """
    Return the first matching sheet from `sheets` by trying several candidate names.
    Case-insensitive, ignores extra spaces. Returns empty DataFrame if not found.
    """
    # normalize keys once
    norm = {k.strip().lower(): v for k, v in sheets.items()}
    for cand in candidates:
        key = cand.strip().lower()
        if key in norm:
            return norm[key]
    return pd.DataFrame()

def _build_everything(
    sheets: dict[str, pd.DataFrame],
    year: int,
    threshold: float,
    include_penalty: bool,
):
    """
    Orchestrate: extract inputs -> build interim -> build final -> (optional) penalty.
    IMPORTANT: build_interim in your code expects (emp_elig, emp_enroll, dep_enroll, ...).
    """

    # Tolerant sheet extraction for your pipeline
    emp_elig  = _get_sheet(
        sheets,
        "emp eligibility", "employee eligibility", "eligibility", "emp_eligibility"
    )
    emp_enroll = _get_sheet(
        sheets,
        "emp enrollment", "employee enrollment", "emp_enrollment", "employee enrolment"
    )
    dep_enroll = _get_sheet(
        sheets,
        "dep enrollment", "dependent enrollment", "dep_enrollment", "dependent enrolment"
    )

    # If your builder internally reads "Emp Wait Period", you’re done.
    # If it expects it explicitly, add it and pass to build_interim accordingly:
    # emp_wait = _get_sheet(sheets, "emp wait period", "wait period")

    # Pass the three frames exactly as your builder expects
    interim = build_interim(
        emp_elig,
        emp_enroll,
        dep_enroll,
        year=year,
        affordability_threshold=threshold,
    )

    final = build_final(interim, year=year)

    penalty = None
    if include_penalty:
        try:
            penalty = build_penalty_dashboard(final)
        except Exception:
            penalty = None

    return interim, final, penalty

# ─────────────────────── debug endpoint ───────────────────────
@app.post("/debug/pdf_fields")
async def debug_pdf_fields(pdf: UploadFile = File(...)):
    """Upload a blank 1095-C PDF; returns a dict of its AcroForm fields (for mapping)."""
    try:
        pdf_bytes = await pdf.read()
        fields = list_pdf_fields(pdf_bytes)
        return JSONResponse(fields)
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)

# ─────────────────────── main endpoints ───────────────────────
@app.post("/generate/single")
async def generate_single(
    request: Request,
    excel: UploadFile = File(..., description="Input XLSX"),
    pdf: UploadFile | None = File(None, description="Blank 1095-C PDF"),
    # convenient aliases (still read full form for robustness)
    employee_id: Optional[str] = Form(None, alias="employeeId"),
    filing_year: Optional[int] = Form(None, alias="filingYear"),
    year: Optional[int] = Form(None),
    affordability_threshold: Optional[float] = Form(None, alias="affordabilityThreshold"),
    threshold: Optional[float] = Form(None),
    include_penalty_dashboard: Optional[str] = Form(None, alias="includePenaltyDashboard"),
    includePenalty: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
):
    try:
        # Read full form to be key-agnostic
        raw_form = _to_plain_dict(await request.form())

        # Year: prefer explicit, else detect
        used_year = _detect_year(raw_form, filing_year or year)
        if used_year is None:
            return JSONResponse({"error": "Missing filing year"}, status_code=422)

        # Threshold: explicit → detect → default
        used_threshold = affordability_threshold if affordability_threshold is not None else threshold
        if used_threshold is None:
            used_threshold = _detect_threshold(raw_form, None)
        if used_threshold is None:
            used_threshold = 50.0  # safe default for UAT

        include_penalty = _truthy(
            raw_form.get("includePenaltyDashboard", include_penalty_dashboard)
            or raw_form.get("includePenalty", includePenalty),
            default=False
        )

        # Read Excel
        sheets = _xlsx_from_upload(excel)
        if not sheets:
            return JSONResponse({"error": "Excel file has no readable sheets"}, status_code=422)

        # Build data
        interim, final, penalty = _build_everything(
            sheets=sheets, year=int(used_year), threshold=float(used_threshold), include_penalty=include_penalty
        )

        # If no EmployeeID → return Excel bundle
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

        # Generate a single employee’s PDF
        emp_id_str = str(employee_id).strip()
        final_emp = final[final["EmployeeID"].astype(str) == emp_id_str]
        if final_emp.empty:
            return JSONResponse({"error": f"EmployeeID {emp_id_str} not found in Final grid"}, status_code=404)

        if pdf is None:
            return JSONResponse({"error": "Blank 1095-C PDF is required when EmployeeID is provided"}, status_code=422)
        pdf_bytes = await pdf.read()

        emp_enroll = sheets.get("Emp Enrollment") or sheets.get("emp enrollment")
        dep_enroll = sheets.get("Dep Enrollment") or sheets.get("dep enrollment")

        # Demographic/ID sheet to populate Part I (fallback to final row if missing)
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

        return StreamingResponse(
            flattened,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{flattened_name}"'},
        )
    except Exception as e:
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)

@app.post("/generate/bulk")
async def generate_bulk(
    request: Request,
    excel: UploadFile = File(...),
    pdf: UploadFile = File(...),
    filing_year: Optional[int] = Form(None, alias="filingYear"),
    year: Optional[int] = Form(None),
    affordability_threshold: Optional[float] = Form(None, alias="affordabilityThreshold"),
    threshold: Optional[float] = Form(None),
    include_penalty_dashboard: Optional[str] = Form(None, alias="includePenaltyDashboard"),
    includePenalty: Optional[str] = Form(None),
):
    """Simple bulk stub: returns processed Excel bundle."""
    try:
        raw_form = _to_plain_dict(await request.form())
        used_year = _detect_year(raw_form, filing_year or year)
        if used_year is None:
            return JSONResponse({"error": "Missing filing year"}, status_code=422)

        used_threshold = affordability_threshold if affordability_threshold is not None else threshold
        if used_threshold is None:
            used_threshold = _detect_threshold(raw_form, None)
        if used_threshold is None:
            used_threshold = 50.0

        include_penalty = _truthy(
            raw_form.get("includePenaltyDashboard", include_penalty_dashboard)
            or raw_form.get("includePenalty", includePenalty),
            default=False
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
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)

@app.post("/process/excel")
async def process_excel(
    request: Request,
    excel: UploadFile = File(...),
    filing_year: Optional[int] = Form(None, alias="filingYear"),
    year: Optional[int] = Form(None),
    affordability_threshold: Optional[float] = Form(None, alias="affordabilityThreshold"),
    threshold: Optional[float] = Form(None),
    include_penalty_dashboard: Optional[str] = Form(None, alias="includePenaltyDashboard"),
    includePenalty: Optional[str] = Form(None),
):
    """Returns an Excel workbook (Interim + Final [+ optional Penalty])."""
    try:
        raw_form = _to_plain_dict(await request.form())
        used_year = _detect_year(raw_form, filing_year or year)
        if used_year is None:
            return JSONResponse({"error": "Missing filing year"}, status_code=422)

        used_threshold = affordability_threshold if affordability_threshold is not None else threshold
        if used_threshold is None:
            used_threshold = _detect_threshold(raw_form, None)
        if used_threshold is None:
            used_threshold = 50.0

        include_penalty = _truthy(
            raw_form.get("includePenaltyDashboard", include_penalty_dashboard)
            or raw_form.get("includePenalty", includePenalty),
            default=False
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
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)
