# main_fastapi.py (only the endpoint changed)
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import io, os, zipfile, pandas as pd
from typing import Optional, List, Tuple
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

@app.post("/pipeline")
async def pipeline(
    year: str = Form(...),
    input_excel: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None)
):
    _require_key(x_api_key)
    try:
        year_int = int(year)
        if year_int < 2000 or year_int > 2100:
            raise ValueError("Year out of range")

        data = await input_excel.read()
        if not data:
            raise ValueError("Empty file")

        log.info(f"start year={year_int} file={input_excel.filename} bytes={len(data)}")

        interim_df = build_interim_df(year_int, data)
        if interim_df is None or interim_df.empty:
            raise ValueError("No data after processing (empty interim)")

        # load optional sheets for Part I/III
        sheets = load_input_workbook(data)
        demo = sheets.get("Emp Demographic") or sheets.get("Emp_Demographic") or pd.DataFrame()
        dep  = sheets.get("Dep Enrollment")  or sheets.get("Dep_Enrollment")  or pd.DataFrame()

        # PDFs
        pdfs = generate_all_pdfs(
            interim_df=interim_df,
            year=year_int,
            template_path=PDF_TEMPLATE_PATH,
            fields_json_path=FIELDS_JSON_PATH,
            demo_df=demo,
            dep_df=dep
        )

        # Excel (interim)
        interim_buf = io.BytesIO()
        with pd.ExcelWriter(interim_buf, engine="xlsxwriter") as w:
            interim_df.to_excel(w, index=False, sheet_name="Interim")
        interim_buf.seek(0)

        # ZIP everything
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("interim_full.xlsx", interim_buf.getvalue())
            for fname, data in pdfs:
                z.writestr(f"pdfs/{fname}", data)
        zip_buf.seek(0)

        headers = {"Content-Disposition": f'attachment; filename="1095c_outputs_{year_int}.zip"'}
        return StreamingResponse(zip_buf, media_type="application/zip", headers=headers)

    except ValueError as ve:
        log.exception("pipeline value error")
        return JSONResponse(status_code=422, content={{"error": str(ve)}})  # 422 for validation-ish
    except HTTPException:
        raise
    except Exception as e:
        log.exception("pipeline failed")
        return JSONResponse(status_code=400, content={"error": str(e)})
