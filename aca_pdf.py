# aca_pdf.py
from __future__ import annotations

import io
from typing import Tuple, Optional

import pandas as pd
from pandas import ExcelWriter

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from debug_logging import get_logger

log = get_logger("pdf")


# ---------------------------
# Excel outputs (XLSX)
# ---------------------------
def _safe_sheet_name(name: str) -> str:
    bad = r'[]:*?/\\'
    out = "".join("_" if ch in bad else ch for ch in str(name))
    return out[:31] if len(out) > 31 else out


def save_excel_outputs(
    interim: pd.DataFrame,
    final: pd.DataFrame,
    year: int,
    *,
    penalty_dashboard: Optional[pd.DataFrame] = None,
) -> bytes:
    """
    Writes Interim / Final / (optional) Penalty into a single XLSX and returns raw bytes.
    Guarantees at least one visible sheet to avoid openpyxl 'At least one sheet must be visible'.
    """
    output = io.BytesIO()
    with ExcelWriter(output, engine="openpyxl") as xw:
        wrote_any = False

        if interim is not None and not interim.empty:
            interim_sorted = interim.sort_values(["EmployeeID", "MonthNum"], na_position="last")
            interim_sorted.to_excel(xw, sheet_name=_safe_sheet_name(f"Interim {year}"), index=False)
            wrote_any = True

        if final is not None and not final.empty:
            final_sorted = final.copy()
            # keep EmployeeID then Month ordering (All 12 months first if present)
            order = ["EmployeeID", "Month", "Line14_Final", "Line16_Final"]
            cols = [c for c in order if c in final_sorted.columns] + [c for c in final_sorted.columns if c not in order]
            final_sorted = final_sorted.loc[:, cols]
            final_sorted.to_excel(xw, sheet_name=_safe_sheet_name(f"Final {year}"), index=False)
            wrote_any = True

        if penalty_dashboard is not None and isinstance(penalty_dashboard, pd.DataFrame) and not penalty_dashboard.empty:
            penalty_dashboard.to_excel(xw, sheet_name=_safe_sheet_name("Penalty"), index=False)
            wrote_any = True

        # If nothing to write, create an empty "Output" sheet so workbook is valid
        if not wrote_any:
            pd.DataFrame({"Info": [f"No rows to export for year {year}"]}).to_excel(
                xw, sheet_name="Output", index=False
            )

        # Ensure the first sheet is visible/active (openpyxl does this by default, but we’re explicit)
        wb = xw.book
        if hasattr(wb, "worksheets") and wb.worksheets:
            ws0 = wb.worksheets[0]
            if getattr(ws0, "sheet_state", "visible") != "visible":
                ws0.sheet_state = "visible"
            if hasattr(wb, "active"):
                wb.active = 0

    raw = output.getvalue()
    log.info("save_excel_outputs",
             extra={"extra_data": {
                 "interim_rows": 0 if interim is None else getattr(interim, "shape", [0])[0],
                 "final_rows": 0 if final is None else getattr(final, "shape", [0])[0],
                 "penalty_rows": 0 if penalty_dashboard is None else getattr(penalty_dashboard, "shape", [0])[0],
                 "bytes": len(raw),
             }})
    return raw


# ---------------------------
# PDF builder (ReportLab)
# ---------------------------
def _month_table_from_final(emp_final: pd.DataFrame) -> Tuple[list, list]:
    """
    Convert a single-employee 'final' DataFrame (Month, Line14_Final, Line16_Final)
    into a table data + column widths for ReportLab.
    """
    cols = ["Month", "Line14_Final", "Line16_Final"]
    df = emp_final.loc[:, [c for c in cols if c in emp_final.columns]].copy()

    data = [["Month", "Line 14", "Line 16"]]
    for _, r in df.iterrows():
        data.append([str(r.get("Month", "")), str(r.get("Line14_Final", "")), str(r.get("Line16_Final", ""))])

    col_widths = [1.3 * inch, 1.2 * inch, 1.2 * inch]
    return data, col_widths


def _draw_employee_header(story, emp_row: pd.Series, year: int):
    styles = getSampleStyleSheet()
    title = Paragraph(f"<b>Form 1095-C (summary)</b> — Year {year}", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 0.15 * inch))

    # Safe getters
    def _g(field: str, default: str = "") -> str:
        try:
            v = emp_row.get(field, default)
        except Exception:
            v = default
        return "" if v is None else str(v)

    # Basic fields (adapt to your sheet columns if needed)
    lines = [
        f"<b>Employee ID:</b> {_g('employeeid')}",
        f"<b>Name:</b> {_g('employeename', _g('name', ''))}",
        f"<b>SSN/ID:</b> {_g('ssn', _g('taxid', ''))}",
        f"<b>Address:</b> {_g('address', '')} {_g('city', '')} {_g('state', '')} {_g('zip', '')}",
    ]
    for ln in lines:
        story.append(Paragraph(ln, styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))


def fill_pdf_for_employee(
    base_pdf_bytes: io.BytesIO,
    emp_row: pd.Series,
    emp_final: pd.DataFrame,
    year: int,
) -> Tuple[str, io.BytesIO, str, io.BytesIO]:
    """
    Build a printable PDF summary for one employee, using the rows in emp_final for Line14/Line16.
    Returns (editable_name, editable_pdf_bytes, flat_name, flat_pdf_bytes).
    Note: This draws a clean summary PDF; it doesn't try to write into the original form fields.
    """
    # Editable-style (just a name; content identical). Some clients keep an “editable” copy.
    editable_name = f"1095c_{emp_row.get('employeeid', 'employee')}_editable.pdf"
    flat_name = f"1095c_{emp_row.get('employeeid', 'employee')}_flat.pdf"

    # Use a SimpleDocTemplate for nicer layout
    def build_pdf() -> bytes:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=LETTER, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                                topMargin=0.6 * inch, bottomMargin=0.6 * inch)
        styles = getSampleStyleSheet()
        story = []

        # Header (employee info)
        _draw_employee_header(story, emp_row, year)

        # Month table
        data, widths = _month_table_from_final(emp_final)
        table = Table(data, colWidths=widths)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.25 * inch))

        # Footer / note
        note = Paragraph(
            "This is a generated summary based on eligibility/enrollment rules. "
            "For official IRS filing, review values and export using your standard workflow.",
            styles["Italic"]
        )
        story.append(note)

        doc.build(story)
        return buf.getvalue()

    try:
        # We ignore base_pdf_bytes content here since we draw a fresh summary.
        # (Keeping param to preserve the existing function contract with the API layer.)
        editable_bytes = io.BytesIO(build_pdf())
        flat_bytes = io.BytesIO(editable_bytes.getvalue())  # same drawing; “flattened” name
    except Exception as e:
        log.exception("fill_pdf_for_employee failed")
        # Return a minimal one-page PDF with error text to avoid HTTP 500
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=LETTER)
        c.setFont("Helvetica", 11)
        c.drawString(72, 720, "Error generating PDF for employee.")
        c.drawString(72, 705, f"Reason: {str(e)}")
        c.showPage()
        c.save()
        fallback = io.BytesIO(buf.getvalue())
        return editable_name, fallback, flat_name, io.BytesIO(fallback.getvalue())

    log.info("fill_pdf_for_employee",
             extra={"extra_data": {
                 "employeeid": str(emp_row.get("employeeid", "")),
                 "rows_final": 0 if emp_final is None else emp_final.shape[0],
                 "bytes": len(flat_bytes.getvalue())
             }})

    return editable_name, editable_bytes, flat_name, flat_bytes
