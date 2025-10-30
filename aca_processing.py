# aca_processing.py
# Helpers and orchestration for the 3-step flow:
# - parse helpers
# - build interim from Excel bytes
# - read Interim.xlsx back
# - prepare per-employee PDF payload
# - safe wrapper that calls your existing pdf_filler.py (any of several signatures)

from __future__ import annotations

import io
import logging
import traceback
from typing import Optional, Dict, Any, Iterable, List

import pandas as pd

from aca_builder import build_interim_df, load_input_workbook

# IMPORTANT: this matches your repo structure (pdf_filler.py)
import pdf_filler as _pdf

log = logging.getLogger("aca_processing")

# -----------------------------
# Simple parsers
# -----------------------------
def parse_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")

def parse_int(v: Optional[str], default: int) -> int:
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def parse_float(v: Optional[str], default: Optional[float]) -> Optional[float]:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default

# -----------------------------
# Interim build/read
# -----------------------------
def build_interim_from_excel_bytes(year: int, excel_bytes: bytes, affordability_threshold: Optional[float]) -> pd.DataFrame:
    """
    Normalizes Excel bytes -> sheets -> interim via builder (builder also accepts bytes directly,
    but we normalize to keep logs predictable and errors clearer).
    """
    sheets = load_input_workbook(excel_bytes)
    return build_interim_df(year=year, sheets=sheets, affordability_threshold=affordability_threshold)

def read_interim_xlsx(xlsx_bytes: bytes) -> pd.DataFrame:
    """
    Load an Interim workbook back to DataFrame.
    Accepts a few likely sheet names; falls back to first sheet.
    """
    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    name = None
    for s in xls.sheet_names:
        if s.lower().strip() in {"interim", "interim sheet", "interim_table"}:
            name = s
            break
    if name is None:
        name = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=name)

    # Normalize key columns if needed
    if "Employee_ID" in df.columns and "EmployeeID" not in df.columns:
        df = df.rename(columns={"Employee_ID": "EmployeeID"})

    if "MonthNum" not in df.columns and "Month" in df.columns:
        m = df["Month"].astype(str).str[:3].str.title().map(
            {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
        )
        df["MonthNum"] = m

    return df

def employee_ids_from_interim(interim: pd.DataFrame) -> List[str]:
    if interim is None or interim.empty or "EmployeeID" not in interim.columns:
        return []
    return sorted(interim["EmployeeID"].astype(str).unique().tolist())

# -----------------------------
# PDF payload builders
# -----------------------------
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def build_pdf_payload_from_interim_row(interim_rows: pd.DataFrame) -> Dict[str, Any]:
    """
    Takes one employee's 12 rows and returns inputs for the PDF filler.
    Expected columns in interim: EmployeeID, Name, MonthNum, line14_final, line16_final (per our builder).
    """
    g = interim_rows.sort_values("MonthNum", kind="stable")
    emp_id = str(g["EmployeeID"].iloc[0])
    name = str(g["Name"].iloc[0]) if "Name" in g.columns else ""

    line14_by_month = {
        MONTHS[i-1]: (g.loc[g["MonthNum"] == i, "line14_final"].iloc[0] if (g["MonthNum"] == i).any() else "")
        for i in range(1, 13)
    }
    line16_by_month = {
        MONTHS[i-1]: (g.loc[g["MonthNum"] == i, "line16_final"].iloc[0] if (g["MonthNum"] == i).any() else "")
        for i in range(1, 13)
    }

    employee_pi = {
        "employee_id": emp_id,
        "name": name,
        # Add other PI fields here if your pdf_filler uses them (SSN, address, etc.)
    }

    payload: Dict[str, Any] = {
        "employee_pi": employee_pi,
        "line14_by_month": line14_by_month,
        "line16_by_month": line16_by_month,
        # If you’re wiring Part III (covered individuals), populate here:
        "covered_individuals": [],
    }
    return payload

# -----------------------------
# Robust wrapper for your pdf_filler
# -----------------------------
def safe_fill_pdf_for_employee(
    *,
    blank_pdf_bytes: bytes,
    employee_pi: Dict[str, Any],
    line14_by_month: Dict[str, str],
    line16_by_month: Dict[str, str],
    covered_individuals: Iterable[Dict[str, Any]] | None = None,
    flatten: bool = True,
) -> bytes:
    """
    Calls your pdf_filler with maximum compatibility.
    Tries:
      1) pdf_filler.fill_pdf_for_employee(...kwargs)
      2) pdf_filler.fill_pdf_for_employee(*args)         # older positional style
      3) pdf_filler.generate_single_pdf(...kwargs)       # fallback name
      4) pdf_filler.generate_pdf_for_employee(...kwargs) # another common variant
    Returns bytes of the filled PDF or raises on failure.
    """
    covered_individuals = list(covered_individuals or [])

    # Candidate function names (order matters)
    candidates = [
        "fill_pdf_for_employee",
        "generate_single_pdf",
        "generate_pdf_for_employee",
    ]

    last_exc: Optional[Exception] = None

    for fname in candidates:
        func = getattr(_pdf, fname, None)
        if func is None:
            continue

        # Try kwargs first
        try:
            out = func(
                blank_pdf_bytes=blank_pdf_bytes,
                employee_pi=employee_pi,
                line14_by_month=line14_by_month,
                line16_by_month=line16_by_month,
                covered_individuals=covered_individuals,
                flatten=flatten,
            )
            if isinstance(out, (bytes, bytearray)):
                return bytes(out)
        except Exception as e:
            last_exc = e
            log.debug(f"{fname}(kwargs) failed; trying positional if supported", exc_info=True)

        # Try positional (some implementations unpack *args internally)
        try:
            out = func(
                blank_pdf_bytes,
                employee_pi,
                line14_by_month,
                line16_by_month,
                covered_individuals,
                flatten,
            )
            if isinstance(out, (bytes, bytearray)):
                return bytes(out)
        except Exception as e:
            last_exc = e
            log.debug(f"{fname}(positional) failed", exc_info=True)

    # If we get here, every attempt failed
    if last_exc:
        log.error("PDF filler failed: %s", last_exc)
        raise last_exc
    raise RuntimeError("No compatible PDF filler function found in pdf_filler.py")
