# main_fastapi.py
from __future__ import annotations

import io, zipfile, traceback, logging
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd

from aca_builder import (
    load_input_workbook,
    build_interim_df,
    build_final,
    build_penalty_dashboard,
)

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aca_api")

app = FastAPI(title="ACA 1095 Builder API")

# ---------- GLOBAL ERROR HANDLER ----------
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

# ---------- helpers ----------
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

# ---------- pipeline ----------
@app.post("/pipeline")
async def pipeline(
    mode: str = Form("final_interim"),                     # "json" to force JSON preview
    filing_year: str = Form(...),
    affordability_threshold: Optional[str] = Form(None),
    include_penalty_dashboard: Optional[str] = Form("true"),

    excel: UploadFile = File(...),                         # the Excel file
    blank_pdf: UploadFile | None = File(None),
    employee_id: Optional[str] = Form(None),

    return_json: Optional[str] = Form("false"),            # or pass mode=json
):
    """
    Vercel UI calls this. Returns ZIP by default; JSON if mode=json or return_json=true.
    On errors: returns JSON with {error, where, trace}.
    """
    year = _to_int(filing_year, default=2025)
    thr = _to_float(affordability_threshold, None)
    want_penalty = _to_bool(include_penalty_dashboard, True)
    want_json = _to_bool(return_json, False) or (str(mode).lower() == "json")

    try:
        # 1) Read Excel bytes
        excel_bytes = await excel.read()
        size = len(excel_bytes or b"")
        log.info(f"/pipeline: year={year}, thr={thr}, penalty={want_penalty}, json={want_json}, excel_bytes={size} bytes")
        if not excel_bytes:
            raise HTTPException(status_code=400, detail="Empty Excel upload")

        # 2) Convert to sheets (NOTE: aca_builder can also accept bytes, but we normalize here)
        try:
            sheets = load_input_workbook(excel_bytes)
        except Exception as e:
            log.exception("load_input_workbook failed")
            return JSONResponse(status_code=400, content={
                "error": f"Excel load failed: {type(e).__name__}: {e}",
                "where": "load_input_workbook",
                "trace": traceback.format_exc(),
            })

        # 3) Build Interim
        try:
            interim = build_interim_df(year=year, sheets=sheets, affordability_threshold=thr)
        except Exception as e:
            log.exception("build_interim_df failed")
            return JSONResponse(status_code=400, content={
                "error": f"Interim build failed: {type(e).__name__}: {e}",
                "where": "build_interim_df",
                "trace": traceback.format_exc(),
            })

        # 4) Build Final + (optional) Penalty
        try:
            final = build_final(interim)
        except Exception as e:
            log.exception("build_final failed")
            return JSONResponse(status_code=400, content={
                "error": f"Final build failed: {type(e).__name__}: {e}",
                "where": "build_final",
                "trace": traceback.format_exc(),
            })

        penalty = pd.DataFrame()
        if want_penalty:
            try:
                penalty = build_penalty_dashboard(interim)
            except Exception as e:
                log.exception("build_penalty_dashboard failed")
                return JSONResponse(status_code=400, content={
                    "error": f"Penalty build failed: {type(e).__name__}: {e}",
                    "where": "build_penalty_dashboard",
                    "trace": traceback.format_exc(),
                })

        if want_json:
            # JSON preview for debugging in UI
            payload = {
                "year": year,
                "interim_head": interim.head(25).to_dict(orient="records"),
                "final_head": final.head(25).to_dict(orient="records"),
                "penalty_head": ([] if penalty.empty else penalty.head(25).to_dict(orient="records")),
            }
            return JSONResponse(payload)

        # 5) Stream ZIP
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

            # Penalty.xlsx
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
        log.exception("pipeline crashed")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Pipeline crashed: {type(e).__name__}: {e}",
                "where": "pipeline-outer",
                "trace": traceback.format_exc(),
            },
        )
