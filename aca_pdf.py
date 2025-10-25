# aca_pdf.py
from __future__ import annotations

import io
from typing import Tuple
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter

# We keep PDF filling minimal & robust. Excel writer is the critical part for your flow.

def fill_pdf_for_employee(
    base_pdf_bytes_io: io.BytesIO,
    emp_row: pd.Series,
    emp_final_df: pd.DataFrame,
    filing_year: int,
) -> Tuple[str, io.BytesIO, str, io.BytesIO]:
    """
    Minimal filler:
    - Returns a copy of the base PDF as "editable" (no field changes)
    - Returns the same as "flattened" (pages rewritten), so downstream never errors
    Replace with your detailed field mapping when needed.
    """
    base_pdf_bytes_io.seek(0)
    reader = PdfReader(base_pdf_bytes_io)
    writer_edit = PdfWriter()
    for p in reader.pages:
        writer_edit.add_page(p)

    editable_buf = io.BytesIO()
    writer_edit.write(editable_buf)
    editable_buf.seek(0)

    # "Flatten": write again (no form fields preserved)
    reader2 = PdfReader(editable_buf)
    writer_flat = PdfWriter()
    for p in reader2.pages:
        writer_flat.add_page(p)
    flat_buf = io.BytesIO()
    writer_flat.write(flat_buf)
    flat_buf.seek(0)

    emp_id = str(emp_row.get("employeeid", "empl
