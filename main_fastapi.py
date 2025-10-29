# main_fastapi.py
# FastAPI service for ACA 1095 builder:
# - Accepts Excel (.xlsx) + options
# - Builds Interim, Final, and optional Penalty Dashboard
# - Streams ZIP or returns JSON (for debugging)

from __future__ import annotations

import io
import zipfile
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

import pandas as pd

from aca_builder import (
    load_input_workbook,
    build_interim_df,
    build_final,
    build_penalty_dashboard,
)

app = FastAPI(title="ACA 1095 Builder API")

# -----------------------------
# Helpers
# -----------------------------
def _to_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on")

def _to_int(v: Optional[str], default: int) -> int:
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def _to_float(v: Optional[str], default: Optional[float]) -> Optional[float]:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default

# -----------------------------
# Single pipeline endpoint used by the Vercel UI:
# -----------------------------
@app.post("/pipeline")
async def pipeline(
    # Mode flags from UI; we accept any string and branch
    mode: str = Form("final_interim"),
    filing_year: str = Form(...),
    affordability_threshold: Optional[str] = Form(None),
    include_penalty_dashboard: Optional[str] = Form("true"),

    # Files
    excel: UploadFile = File(...),

    # Optional: blank PDF and targeted employee ID (for single-PDF flows)
    blank_pdf: UploadFile | None = File(None),
    employee_id: Optional[str] = Form(None),

    # Debug: return json instead of zip
    return_json: Optional[str] = Form("false"),
):
    """
    Expected from UI:
      - filing_year (string, e.g., "2025")
      - affordability_threshold (string/number) [optional]
      - include_penalty_dashboard (bool-ish string)
      - excel: UploadFile (.xlsx)
      - mode: "final_interim" | "zip" | "json"
      - return_json: "true" to return JSON instead of ZIP (debug)
    """

    try:
        year = _to_int(filing_year, default=2025)
        thr = _to_float(affordability_threshold, None)
        want_penalty = _to_bool(include_penalty_dashboard, True)
        want_json = _to_bool(return_json, False) or (mode.lower() == "json")

        # 1) Read Excel bytes
        excel_bytes = await excel.read()
        if not excel_bytes:
            raise HTTPException(status_code=400, detail="Empty Excel upload")

        # 2) Convert bytes -> sheets dict (THIS FIXES the 'bytes'.get error)
        sheets = load_input_workbook(excel_bytes)

        # 3) Build Interim
        interim = build_interim_df(year=year, sheets=sheets, affordability_threshold=thr)

        # 4) Build Final + (optional) Penalty
        final = build_final(interim)
        penalty = build_penalty_dashboard(interim) if want_penalty else pd.DataFrame()

        if want_json:
            # Return a JSON preview (first few rows) for debugging
            payload = {
                "year": year,
                "interim_head": interim.head(25).to_dict(orient="records"),
                "final_head": final.head(25).to_dict(orient="records"),
                "penalty_head": ([] if penalty.empty else penalty.head(25).to_dict(orient="records")),
            }
            return JSONResponse(payload)

        # 5) Otherwise, stream a ZIP with all sheets
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Interim.xlsx
            ibuf = io.BytesIO()
            with pd.ExcelWriter(ibuf, engine="xlsxwriter") as writer:
                interim.to_excel(writer, index=False, sheet_name="Interim")
            zf.writestr("Interim.xlsx", ibuf.getvalue())

            # Final.xlsx
            fbuf = io.BytesIO()
            with pd.ExcelWriter(fbuf, engine="xlsxwriter") as writer:
                final.to_excel(writer, index=False, sheet_name="Final")
            zf.writestr("Final.xlsx", fbuf.getvalue())

            # Penalty.xlsx (optional)
            if want_penalty and not penalty.empty:
                pbuf = io.BytesIO()
                with pd.ExcelWriter(pbuf, engine="xlsxwriter") as writer:
                    penalty.to_excel(writer, index=False, sheet_name="Penalty")
                zf.writestr("Penalty.xlsx", pbuf.getvalue())

        zip_buf.seek(0)
        headers = {"Content-Disposition": f'attachment; filename="aca_outputs_{year}.zip"'}
        return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        # Surface a clear message to the UI
        return JSONResponse(
            status_code=400,
            content={"error": f"Pipeline failed: {type(e).__name__}: {e}"},
        )
