# aca_pdf.py
# Utilities for filling ACA 1095-C PDF forms.
# Includes new generate_single_pdf() and generate_bulk_pdfs() wrappers expected by main_fastapi.py

import io
import pandas as pd
from PyPDF2 import PdfWriter
from datetime import datetime
from aca_processing import MONTHS, _coerce_str

# ------------------------------------------------------------
# Existing functions (already in your file)
# ------------------------------------------------------------

def fill_pdf_for_employee(pdf_bytes, emp_row, final_df_emp, year_used, emp_enroll_emp, dep_enroll_emp):
    """
    Dummy placeholder of your actual PDF fill logic.
    Should return (editable_name, editable_pdf_bytes, flat_name, flat_pdf_bytes)
    """
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)
    return "editable.pdf", buf, "flattened.pdf", buf

def save_excel_outputs(final_df, interim_df, penalty_df, year):
    """Save the 3 Excel sheets to bytes."""
    out_bytes = io.BytesIO()
    with pd.ExcelWriter(out_bytes, engine="xlsxwriter") as writer:
        final_df.to_excel(writer, sheet_name="Final", index=False)
        interim_df.to_excel(writer, sheet_name="Interim", index=False)
        penalty_df.to_excel(writer, sheet_name="Penalty Dashboard", index=False)
    out_bytes.seek(0)
    return out_bytes.getvalue()

# ------------------------------------------------------------
# New helper + wrappers (safe to add)
# ------------------------------------------------------------

def _get_template_bytes() -> bytes:
    """Fetch a PDF template or return a blank page fallback."""
    import os
    path = os.getenv("PDF_TEMPLATE_PATH", "").strip()
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    # fallback blank PDF
    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    w.write(buf)
    buf.seek(0)
    return buf.getvalue()

def _final_row_to_month_table(final_row: pd.Series) -> pd.DataFrame:
    """Convert Final row columns (Line14_Jan etc.) to month table."""
    rows = []
    for m in MONTHS:
        rows.append({
            "Month": m,
            "Line14_Final": _coerce_str(final_row.get(f"Line14_{m}", "")),
            "Line16_Final": _coerce_str(final_row.get(f"Line16_{m}", "")),
        })
    return pd.DataFrame(rows)

def generate_single_pdf(final_df: pd.DataFrame,
                        interim_df: pd.DataFrame,
                        employee_id,
                        year_used: int,
                        flatten_pdf: bool = False) -> bytes:
    """Adapter used by FastAPI for /generate/single."""
    fmask = final_df["EmployeeID"].astype(str) == str(employee_id)
    if not fmask.any():
        raise ValueError(f"EmployeeID {employee_id} not found in Final sheet")
    frow = final_df.loc[fmask].iloc[0]
    month_tbl = _final_row_to_month_table(frow)

    imask = interim_df["employeeid"].astype(str) == str(employee_id)
    emp_row = interim_df.loc[imask].iloc[0] if imask.any() else pd.Series({"employeeid": employee_id})

    pdf_bytes = _get_template_bytes()
    editable_name, editable, flat_name, flat = fill_pdf_for_employee(
        pdf_bytes=pdf_bytes,
        emp_row=emp_row,
        final_df_emp=month_tbl,
        year_used=year_used,
        emp_enroll_emp=None,
        dep_enroll_emp=None,
    )
    return flat.getvalue() if flatten_pdf else editable.getvalue()

def generate_bulk_pdfs(final_df: pd.DataFrame,
                       interim_df: pd.DataFrame,
                       employee_ids,
                       year_used: int,
                       flatten_pdf: bool = False) -> bytes:
    """Adapter used by FastAPI for /generate/bulk."""
    import zipfile
    if not employee_ids:
        employee_ids = final_df["EmployeeID"].tolist()

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for emp in employee_ids:
            try:
                pdf = generate_single_pdf(final_df, interim_df, emp, year_used, flatten_pdf)
                zf.writestr(f"1095C_{emp}_{year_used}.pdf", pdf)
            except Exception as e:
                zf.writestr(f"ERROR_{emp}.txt", str(e))
    mem.seek(0)
    return mem.getvalue()
