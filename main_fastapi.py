# main_fastapi.py
from __future__ import annotations
import io, os, zipfile
from typing import Optional, List, Tuple
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from debug_logging import get_logger
from aca_builder import build_interim_df, load_input_workbook
from pdf_filler import generate_all_pdfs

log = get_logger("aca1095")

API_KEY = os.getenv("FASTAPI_API_KEY", "")
PDF_TEMPLATE_PATH = os.getenv("PDF_TEMPLATE_PATH", "./assets/f1095c.pdf")
FIELDS_JSON_PATH  = os.getenv("FIELDS_JSON_PATH",  "./assets/pdf_acro_fields_details.json")

app = FastAPI(title="ACA1095 Pipeline")

def _require_key(x_api_key: Optional[str]):
    if API_KEY and (x_api_key != API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/pipeline")
async def pipeline(
    year: int = Form(...),
    input_excel: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None)
):
    _require_key(x_api_key)
    try:
        excel_bytes = await input_excel.read()
        log.info(f"start year={year} file={input_excel.filename} bytes={len(excel_bytes)}")

        interim_df = build_interim_df(year, excel_bytes)
        if interim_df is None or interim_df.empty:
            raise ValueError("No data after processing (empty interim)")

        sheets = load_input_workbook(excel_bytes)
        demo = sheets.get("Emp Demographic") or sheets.get("Emp_Demographic")
        dep  = sheets.get("Dep Enrollment") or sheets.get("Dep_Enrollment")
        if isinstance(demo, pd.DataFrame) and (not demo.empty) and ("EmployeeID" in demo.columns):
            demo["EmployeeID"] = pd.to_numeric(demo["EmployeeID"], errors="coerce").astype("Int64")

        if not os.path.exists(PDF_TEMPLATE_PATH):
            raise FileNotFoundError(f"Missing PDF template at {PDF_TEMPLATE_PATH}")
        if not os.path.exists(FIELDS_JSON_PATH):
            raise FileNotFoundError(f"Missing fields JSON at {FIELDS_JSON_PATH}")

        pdfs: List[Tuple[str, bytes]] = generate_all_pdfs(
            interim_df=interim_df,
            year=year,
            template_path=PDF_TEMPLATE_PATH,
            fields_json_path=FIELDS_JSON_PATH,
            demo_df=demo if isinstance(demo, pd.DataFrame) else None,
            dep_df=dep if isinstance(dep, pd.DataFrame) else None
        )

        # build interim workbook
        interim_buf = io.BytesIO()
        with pd.ExcelWriter(interim_buf, engine="xlsxwriter") as w:
            interim_df.to_excel(w, index=False, sheet_name="Interim")
        interim_buf.seek(0)

        # zip both
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("interim_full.xlsx", interim_buf.getvalue())
            for fname, data in pdfs:
                z.writestr(f"pdfs/{fname}", data)
        zip_buf.seek(0)

        headers = {"Content-Disposition": f'attachment; filename="1095c_outputs_{year}.zip"'}
        log.info(f"done rows={len(interim_df)} pdfs={len(pdfs)}")
        return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)

    except Exception as e:
        log.exception("pipeline failed")
        return JSONResponse(status_code=400, content={"error": str(e)})
