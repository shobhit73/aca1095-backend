# main_fastapi.py
# FastAPI service for ACA 1095: Excel -> (Final, Interim, optional Penalty Dashboard) and PDF generation.

from __future__ import annotations

import io
import zipfile
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic_settings import BaseSettings

# Import ONLY the symbols that exist in your current aca_core.py
from aca_core import (
    load_excel,          # (file_bytes: bytes) -> dict of raw sheets
    prepare_inputs,      # (data: dict) -> (emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, pay_deductions)
    build_interim,       # (emp_demo, emp_status, emp_elig, emp_enroll, dep_enroll, year=None) -> DataFrame
    build_final,         # (interim) -> DataFrame
    save_excel_outputs,  # (interim, final, year) -> bytes    # 2-sheet writer (Final/Interim)
    fill_pdf_for_employee,  # (pdf_bytes, emp_row: Series, final_df_emp: DataFrame, year_used: int)
)

# ---------- logging ----------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aca1095")


# =========================
# Local RunConfig (lightweight shim)
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
    ACA_MODE: str = "SIMPLIFIED"               # placeholder; current aca_core doesn't branch on mode
    FILING_YEAR: Optional[int] = None          # if None, aca_core.build_interim will auto-pick from data
    AFFORDABILITY_THRESHOLD: float = 50.00     # not used by current aca_core, kept for forward-compat
    PENALTY_A_AMOUNT: float = 241.67           # display-only in dashboard
    PENALTY_B_AMOUNT: float = 362.50           # display-only in dashboard
    APP_VERSION: str = "1.1.1"

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
# Simple Penalty Dashboard (local)
# =========================
def build_penalty_dashboard_local(interim: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight, non-authoritative dashboard so the API can return a 3-sheet workbook.
    You can replace this later with a richer aca_core implementation.
    """
    if interim.empty:
        return pd.DataFrame(columns=[
            "EmployeeID", "Months_No_Offer_1H", "Months_Enrolled_2C",
            "Months_Waiting_2D", "Months_NotFT_2B", "Months_NotEmployed_2A",
            "Months_Blank", "Potential_Risk_Flag", "Notes"
        ])

    df = interim.copy()
    df["EmployeeID"] = df["employeeid"].astype(str)

    def agg(grp: pd.DataFrame):
        l14 = grp["line14_final"].astype(str).str.upper().fillna("")
        l16 = grp["line16_final"].astype(str).str.upper().fillna("")
        no_offer = (l14 == "1H").sum()
        e2c = (l16 == "2C").sum()
        e2d = (l16 == "2D").sum()
        e2b = (l16 == "2B").sum()
        e2a = (l16 == "2A").sum()
        blanks = ((l16 == "") | (l16.isna())).sum()
        # naive "risk": months with 1H offer AND not covered by 2A/2B/2D that month
        # (we don't compute per-month pairing here; this is a coarse head-up flag)
        potential = "Yes" if no_offer > 0 and (e2a + e2b + e2d) == 0 else "Review"
        return pd.Series({
            "Months_No_Offer_1H": int(no_offer),
            "Months_Enrolled_2C": int(e2c),
            "Months_Waiting_2D": int(e2d),
            "Months_NotFT_2B": int(e2b),
            "Months_NotEmployed_2A": int(e2a),
            "Months_Blank": int(blanks),
            "Potential_Risk_Flag": potential,
            "Notes": "",
        })

    out = df.groupby("EmployeeID", as_index=False).apply(agg, include_groups=False)
    # Show names if present
    if {"firstname", "lastname"}.issubset(df.columns):
        names = df[["EmployeeID", "firstname", "lastname"]].drop_duplicates()
        out = names.merge(out, on="EmployeeID", how="right")
    return out.reset_index(drop=True)


def write_three_sheet_workbook(interim: pd.DataFrame, final: pd.DataFrame, penalty: pd.DataFrame, year: int) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as xw:
        final.to_excel(xw, index=False, sheet_name=f"Final {year}")
        interim.to_excel(xw, index=False, sheet_name=f"Interim {year}")
        penalty.to_excel(xw, index=False, sheet_name=f"Penalty Dashboard {year}")
    buf.seek(0)
    ret
