# aca_pdf.py
from __future__ import annotations

import io
from typing import Tuple

import pandas as pd
from PyPDF2 import PdfReader, PdfWriter


def fill_pdf_for_employee(
    base_pdf_bytes_io: io.BytesIO,
    emp_row: pd.Series,
    emp_final_df: pd.DataFrame,
    filing_year: int,
) -> Tuple[str, io.BytesIO, str, io.BytesIO]:
    """
    Minimal placeholder filler:
    - Returns a copy of the base PDF as "editable"
    - Returns a flattened copy (no form fields) as "flat"
    """
    base_pdf_bytes_io.seek(0)
    reader = PdfReader(base_pdf_bytes_io)

    # Editable copy
    writer_edit = PdfWriter()
    for p in reader.pages:
        writer_edit.add_page(p)
    editable_buf = io.BytesIO()
    writer_edit.write(editable_buf)
    editable_buf.seek(0)

    # Flattened copy
    reader2 = PdfReader(editable_buf)
    writer_flat = PdfWriter()
    for p in reader2.pages:
        writer_flat.add_page(p)
    flat_buf = io.BytesIO()
    writer_flat.write(flat_buf)
    flat_buf.seek(0)

    emp_id = str(emp_row.get("employeeid", "employee")).strip() or "employee"
    editable_name = f"1095C_{emp_id}_{filing_year}_editable.pdf"
    flat_name = f"1095C_{emp_id}_{filing_year}.pdf"
    return editable_name, editable_buf, flat_name, flat_buf


def _safe_df(df: pd.DataFrame | None, *, name: str) -> pd.DataFrame:
    """
    Ensure we always have something to write so openpyxl has
    at least one visible sheet. If df is None or has 0 rows AND 0 columns,
    produce a tiny placeholder DataFrame.
    """
    if isinstance(df, pd.DataFrame):
        # If it has any columns, to_excel will create a sheet even with 0 rows.
        if df.shape[1] > 0:
            return df
        # If it has 0 cols but some rows (unlikely), still write a placeholder.
    # Placeholder sheet
    return pd.DataFrame(
        {"Info": [f"No data available for '{name}' at the time of export."]}
    )


def save_excel_outputs(
    interim: pd.DataFrame,
    final: pd.DataFrame,
    year: int,
    *,
    penalty_dashboard: pd.DataFrame | None = None,
) -> bytes:
    """
    Write Final/Interim/(optional) Penalty sheets safely.
    Guarantees at least one visible sheet so openpyxl never raises:
    'IndexError: At least one sheet must be visible'.
    """
    from pandas import ExcelWriter

    output = io.BytesIO()
    with ExcelWriter(output, engine="openpyxl") as xw:
        # Always write Final and Interim (using safe placeholders when needed)
        _safe_df(final, name="Final").to_excel(
            xw, index=False, sheet_name=f"Final {year}"
        )
        _safe_df(interim, name="Interim").to_excel(
            xw, index=False, sheet_name=f"Interim {year}"
        )

        # Only add Penalty when it exists and has columns/rows
        if (
            isinstance(penalty_dashboard, pd.DataFrame)
            and penalty_dashboard.shape[1] > 0
            and not penalty_dashboard.empty
        ):
            penalty_dashboard.to_excel(
                xw, index=False, sheet_name=f"Penalty Dashboard {year}"
            )

    output.seek(0)
    return output.getvalue()
