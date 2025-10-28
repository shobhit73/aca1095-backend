# main_fastapi.py
# FastAPI service for ACA 1095-C: Excel -> (Final, Interim, Penalty) and filled PDFs.

from __future__ import annotations

import io
import os
import zipfile
from datetime import datetime
from typing import Optional, Dict, Any, Iterable, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# =============================================================================
# Logging (use your helper if present; otherwise fallback)
# =============================================================================
try:
    from debug_logging import get_logger, log_time, log_df  # optional helper in your repo
    log = get_logger("aca1095")
except Exception:  # fallback
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log = logging.getLogger("aca1095")

    from contextlib import contextmanager
    @contextmanager
    def log_time(_log, label):
        _log.info(f"{label} - start")
        try:
            yield
        finally:
            _log.info(f"{label} - end")

    def log_df(_log, df, name):
        try:
            _log.info(f"{name}: shape={getattr(df,'shape',None)} cols={list(getattr(df,'columns',[]))[:12]}")
        except Exception:
            _log.info(f"{name}: (unprintable)")

# =============================================================================
# Project modules
# =============================================================================
from aca_processing import load_excel
from aca_builder import build_interim, build_final, build_penalty_dashboard
from aca_pdf import save_excel_outputs, fill_pdf_for_employee


# =============================================================================
# Small helpers
# =============================================================================
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


# =============================================================================
# Sheet formatting helpers (final/interim/penalty) + enforced order
# =============================================================================
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _ensure_str(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def _yes_no(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map({True:"Yes", False:"No"}).fillna("")
    return df

def prepare_final_for_export(final_df, year:int):
    """Reorder/normalize Final sheet. Keeps extra columns at the end."""
    if final_df is None or getattr(final_df, "empty", True):
        return final_df

    df = final_df.copy()

    # Compose EmployeeName if not present
    if "EmployeeName" not in df.columns and {"FirstName","LastName"}.issubset(df.columns):
        df["EmployeeName"] = (df["FirstName"].fillna("") + " " + df["LastName"].fillna("")).str.strip()

    base_cols = [
        "EmployeeID", "EmployeeName", "SSN", "TIN", "Company", "FEIN",
        "OfferTier", "LowestCostMonthlyPrem"
    ]

    # Line14/15/16 (prefer TitleCase, fallback to lowercase variants)
    l14 = [f"Line14_{m}" for m in MONTHS if f"Line14_{m}" in df.columns]
    l15 = [f"Line15_{m}" for m in MONTHS if f"Line15_{m}" in df.columns]
    l16 = [f"Line16_{m}" for m in MONTHS if f"Line16_{m}" in df.columns]
    if not l14:
        l14 = [f"line14_{m.lower()}" for m in MONTHS if f"line14_{m.lower()}" in df.columns]
    if not l15:
        l15 = [f"line15_{m.lower()}" for m in MONTHS if f"line15_{m.lower()}" in df.columns]
    if not l16:
        l16 = [f"line16_{m.lower()}" for m in MONTHS if f"line16_{m.lower()}" in df.columns]

    all12 = [c for c in ["Line14_All12","Line15_All12","Line16_All12",
                         "line14_all12","line15_all12","line16_all12"] if c in df.columns]

    ordered = [c for c in base_cols if c in df.columns] + l14 + l15 + l16 + all12
    extras  = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]

    df = _ensure_str(df, ["EmployeeID","SSN","TIN","FEIN"])

    # Money-ish cells (Line15_*, lowest cost)
    money_cols = [c for c in df.columns if c.startswith("Line15_") or c.lower().startswith("lowestcost")]
    for c in money_cols:
        if c in df.columns:
            try:
                df[c] = df[c].apply(lambda x: (None if x in (None, "", "NaN") else float(x)))
            except Exception:
                pass

    return df

def prepare_interim_dashboard(interim_df):
    """Human-friendly Interim: stable order, Yes/No flags, ISO dates."""
    if interim_df is None or getattr(interim_df, "empty", True):
        return interim_df

    df = interim_df.copy()

    rename_map = {
        "employed":"Employed", "ft":"FullTime", "parttime":"PartTime",
        "eligibleforcoverage":"EligibleForCoverage",
        "eligible_allmonth":"EligibleAllMonth",
        "eligible_mv":"EligibleMV",
        "offer_ee_allmonth":"OfferEEAllMonth",
        "enrolled_allmonth":"EnrolledAllMonth",
        "offer_spouse":"OfferSpouse",
        "offer_dependents":"OfferDependents",
        "spouse_eligible":"SpouseEligible",
        "child_eligible":"ChildEligible",
        "spouse_enrolled":"SpouseEnrolled",
        "child_enrolled":"ChildEnrolled",
        "waitingperiod_month":"WaitingPeriodMonth",
        "affordable_plan":"AffordablePlan",
        "line14_final":"Line14",
        "line16_final":"Line16",
        "line14_all12":"Line14_All12",
    }
    df.rename(columns=rename_map, inplace=True)

    wanted = [
        "EmployeeID","Year","MonthNum","Month","MonthStart","MonthEnd",
        "Employed","FullTime","PartTime",
        "EligibleForCoverage","EligibleAllMonth","EligibleMV",
        "OfferEEAllMonth","EnrolledAllMonth",
        "OfferSpouse","OfferDependents",
        "SpouseEligible","ChildEligible",
        "SpouseEnrolled","ChildEnrolled",
        "WaitingPeriodMonth","AffordablePlan",
        "Line14","Line16","Line14_All12",
    ]
    extras = [c for c in df.columns if c not in wanted]
    df = df[[c for c in wanted if c in df.columns] + extras]

    df = _ensure_str(df, ["EmployeeID"])
    for c in ("MonthStart","MonthEnd"):
        if c in df.columns:
            try:
                df[c] = df[c].astype("datetime64[ns]").dt.strftime("%Y-%m-%d")
            except Exception:
                pass

    boolish = [
        "Employed","FullTime","PartTime","EligibleForCoverage","EligibleAllMonth","EligibleMV",
        "OfferEEAllMonth","EnrolledAllMonth","OfferSpouse","OfferDependents",
        "SpouseEligible","ChildEligible","SpouseEnrolled","ChildEnrolled","AffordablePlan"
    ]
    df = _yes_no(df, [c for c in boolish if c in df.columns])

    return df

def prepare_penalty_dashboard(penalty_df):
    if penalty_df is None or getattr(penalty_df, "empty", True):
        return penalty_df
    df = penalty_df.copy()
    if "EmployeeID" in df.columns:
        df["EmployeeID"] = df["EmployeeID"].astype(str)
    return df


# =============================================================================
# Auth (x-api-key)
# =============================================================================
def get_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    expected = os.getenv("FASTAPI_API_KEY")
    if expected:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="Missing x-api-key")
        if x_api_key != expected:
            raise HTTPException(status_code=403, detail="Invalid x-api-key")


# =============================================================================
# App + CORS
# =============================================================================
app = FastAPI(title="ACA 1095-C Generator API", version="1.0.0", docs_url="/docs", redoc_url="/redoc")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Routes
# =============================================================================
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
    Accepts one ACA workbook and returns XLSX with sheets:
      1) Final {YEAR}
      2) Interim Dashboard
      3) Penalty Dashboard (only if non-empty)
    """
    try:
        excel_bytes = await file.read()             # bytes
        data = load_excel(excel_bytes)              # dict of DataFrames
        year_used = int(year) if year else datetime.now().year

        # Unpack per builder signature
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

        penalty_df = build_penalty_dashboard(interim_df)  # penalty works off Interim
        log_df(log, penalty_df, "Penalty")

        # ---- Format + enforce sheet order ----
        final_pretty   = prepare_final_for_export(final_df, year_used)
        interim_pretty = prepare_interim_dashboard(interim_df)
        penalty_pretty = prepare_penalty_dashboard(penalty_df)

        outputs: Dict[str, Any] = {}
        outputs[f"Final {year_used}"] = final_pretty
        outputs["Interim Dashboard"]  = interim_pretty
        if penalty_pretty is not None and hasattr(penalty_pretty, "empty") and not penalty_pretty.empty:
            outputs["Penalty Dashboard"] = penalty_pretty

        out_bytes = save_excel_outputs(outputs)  # single dict arg -> bytes(xlsx)

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

        # Build interim/final
        emp_demo   = data.get("emp_demo")
        emp_elig   = data.get("emp_elig")
        emp_enroll = data.get("emp_enroll")
        dep_enroll = data.get("dep_enroll")
        emp_wait   = data.get("emp_wait")

        interim_df = build_interim(emp_demo, emp_elig, emp_enroll, dep_enroll, year_used, emp_wait=emp_wait)
        final_df   = build_final(interim_df)

        # Pick the employee
        df = final_df
        if employee_id:
            df = final_df[final_df["EmployeeID"].astype(str) == str(employee_id)]
            if df.empty:
                raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found in Final")

        row = df.iloc[0]  # pd.Series

        # Fill PDF; normalize any buffers to bytes
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

        # Build interim/final
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
