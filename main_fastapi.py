# main.py — FastAPI app on Render
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
import io, zipfile, os
import pandas as pd

# If you use your PDF filler:
# from your_pdf_module import generate_pdfs_from_summary  # your earlier Part I/II/III filler in-memory

API_KEY = os.getenv("FASTAPI_API_KEY", "")
PDF_TEMPLATE_PATH = os.getenv("PDF_TEMPLATE_PATH", "/opt/app/f1095c.pdf")
FIELDS_JSON_PATH  = os.getenv("FIELDS_JSON_PATH",  "/opt/app/pdf_acro_fields_details.json")

app = FastAPI()

def require_api_key(key: str | None):
    if not API_KEY:
        return  # no auth configured
    if not key or key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------- Helpers you plug in ----------

def build_interim_df(year: int, excel_bytes: bytes) -> pd.DataFrame:
    """
    Build the full interim table for all employees (12 months),
    including line_14 and line_16 columns.

    Replace this with your real pipeline code.
    """
    # Example skeleton: load the input workbook and call your logic.
    # Here, we just show a tiny placeholder DataFrame so the endpoint is runnable.
    # >>> Replace entirely with your real logic that we already built earlier. <<<
    df = pd.DataFrame([
        {"Employee_ID": 1001, "Name": "Jane A Doe", "Year": year, "Month": "Jan", "line_14": "1E", "line_16": "2C"},
        {"Employee_ID": 1001, "Name": "Jane A Doe", "Year": year, "Month": "Feb", "line_14": "1E", "line_16": "2C"},
    ])
    return df

def generate_all_pdfs(interim_df: pd.DataFrame) -> list[tuple[str, bytes]]:
    """
    From the interim_df (which includes Month, line_14, line_16), generate
    a PDF per employee and return as list of (filename, bytes).

    Replace this with your real PDF filler (the Part I/II/III code we wrote).
    """
    # Pseudo-implementation: you should call your existing filler using
    # PDF_TEMPLATE_PATH and FIELDS_JSON_PATH. It should *not* write to disk; return bytes.

    # For demo purposes, we return an empty list (so you can wire the real filler).
    # Example expected return: [("1095C_1001.pdf", pdf_bytes), ("1095C_1002.pdf", pdf_bytes2), ...]
    return []

# ----------------------------------------

@app.post("/pipeline")
async def pipeline(
    year: int = Form(...),
    input_excel: UploadFile = File(...),
    x_api_key: str | None = Header(None)
):
    require_api_key(x_api_key)

    try:
        excel_bytes = await input_excel.read()

        # 1) Build full interim table
        interim_df = build_interim_df(year, excel_bytes)

        # 2) Generate all PDFs from interim
        pdf_files: list[tuple[str, bytes]] = generate_all_pdfs(interim_df)

        # 3) Write an Excel (interim_full.xlsx) to memory
        interim_buf = io.BytesIO()
        with pd.ExcelWriter(interim_buf, engine="xlsxwriter") as writer:
            interim_df.to_excel(writer, index=False, sheet_name="Interim")
        interim_buf.seek(0)

        # 4) Zip: interim_full.xlsx + pdfs/
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("interim_full.xlsx", interim_buf.getvalue())
            for fname, data in pdf_files:
                z.writestr(f"pdfs/{fname}", data)
        zip_buf.seek(0)

        headers = {"Content-Disposition": 'attachment; filename="1095c_outputs.zip"'}
        return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
