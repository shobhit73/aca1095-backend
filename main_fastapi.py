# main_fastapi.py
from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Optional, Dict, Any, Iterable, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import zipfile

from debug_logging import get_logger, log_time, log_df

log = get_logger("aca1095")

# ---- imports from your modules ----
from aca_processing import load_excel
from aca_builder import build_interim, build_final, build_penalty_dashboard
from aca_pdf import save_excel_outputs, fill_pdf_for_employee


# -------------------------
# API key dependency
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
app = FastAPI(title="ACA 1095-C Generator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Small helpers
# -------------------------
def _bytes_response(content: bytes, media_type: str, filename: str) -> StreamingResponse:
    return StreamingResponse(
        io.BytesIO(content),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

def _zip_bytes(pairs: Iterable[Tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, b in pairs:
            z.writestr(name, b)
    return buf.getvalue()


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}


@app.post("/process/excel")
async def process_excel(
    file: UploadFile = File(...),
    year: Optional[int] = Form(None),
    affordability_threshold: Optional[float] = Form(None),
    _: None = Depends(get_api_key),
):
    """
    Returns an XLSX with: Interim, Final, [Penalty Dashboard]
    """
    try:
        excel_bytes = await file.read()
        data = load_excel(excel_bytes)                     # dict of DataFrames

        # unpack required sheets
        emp_demo   = data.get("emp_demo")
        emp_elig   = data.get("emp_elig")
        emp_enroll = data.get("emp_enroll")
        dep_enroll = data.get("dep_enroll")
        emp_wait   = data.get("emp_wait")
        year_used  = int(year) if year else datetime.now().year

        with log_time(log, "build_interim"):
            interim_df = build_interim(
                emp_demo, emp_elig, emp_enroll, dep_enroll,
                year=year_used,
                emp_wait=emp_wait,
                affordability_threshold=affordability_threshold,
            )
        log_df(log, interim_df, "interim")

        final_df = build_final(interim_df)
        log_df(log, final_df, "final")

        # Penalty dashboard expects INTERIM (not final)
        penalty_df = build_penalty_dashboard(interim_df)
        log_df(log, penalty_df, "penalty")

        # Write workbook: pass ONE dict of sheets
        outputs = {"Interim": interim_df, "Final": final_df}
        if penalty_df is not None and not penalty_df.empty:
            outputs["Penalty Dashboard"] = penalty_df

        out_bytes = save_excel_outputs(outputs)
        return _bytes_response(
            out_bytes,
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            f"ACA-Outputs-{year_used}.xlsx",
        )

    except Exception as e:
        log.exception("Failed to process Excel")
        raise HTTPException(status_code=400, detail=f"Failed to process Excel: {e}")


@app.post("/generate/single")
async def generate_single_pdf(
    file: UploadFile = File(...),
    pdf: UploadFile  = File(...),
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

        # unpack + build interim/final
        emp_demo   = data.get("emp_demo")
        emp_elig   = data.get("emp_elig")
        emp_enroll = data.get("emp_enroll")
        dep_enroll = data.get("dep_enroll")
        emp_wait   = data.get("emp_wait")

        interim_df = build_interim(emp_demo, emp_elig, emp_enroll, dep_enroll, year_used, emp_wait=emp_wait)
        final_df   = build_final(interim_df)

        # filter one employee if provided
        df = final_df
        if employee_id:
            df = final_df[final_df["EmployeeID"].astype(str) == str(employee_id)]
            if df.empty:
                raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")

        row = df.iloc[0]  # pd.Series

        # pass sheets=data so Part III coverage can be computed
        editable_name, editable_bytes, flat_name, flat_bytes = fill_pdf_for_employee(
            pdf_bytes, row, df, year_used, sheets=data
        )

        return _bytes_response(flat_bytes, "application/pdf", flat_name)

    except Exception as e:
        log.exception("Failed to generate single PDF")
        raise HTTPException(status_code=400, detail=f"Failed to generate single PDF: {e}")


@app.post("/generate/bulk")
async def generate_bulk_pdfs(
    file: UploadFile = File(...),
    pdf: UploadFile  = File(...),
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

        # unpack + build
        emp_demo   = data.get("emp_demo")
        emp_elig   = data.get("emp_elig")
        emp_enroll = data.get("emp_enroll")
        dep_enroll = data.get("dep_enroll")
        emp_wait   = data.get("emp_wait")

        interim_df = build_interim(emp_demo, emp_elig, emp_enroll, dep_enroll, year_used, emp_wait=emp_wait)
        final_df   = build_final(interim_df)

        out: list[tuple[str, bytes]] = []
        for _, row in final_df.iterrows():
            try:
                _, _, flat_name, flat_bytes = fill_pdf_for_employee(
                    pdf_bytes, row, final_df, year_used, sheets=data
                )
                out.append((flat_name, flat_bytes))
            except Exception as inner:
                emp = row.get("EmployeeID", "unknown")
                out.append((f"ERROR_{emp}.txt", str(inner).encode("utf-8")))

        zip_bytes = _zip_bytes(out)
        return _bytes_response(zip_bytes, "application/zip", f"ACA-1095C-Bulk-{year_used}.zip")

    except Exception as e:
        log.exception("Failed to generate bulk PDFs")
        raise HTTPException(status_code=400, detail=f"Failed to generate bulk PDFs: {e}")
