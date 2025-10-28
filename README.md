# ACA 1095 Backend (Interim + Bulk PDFs)

FastAPI service that:
1) Builds the **full interim table** (all employees x 12 months, with Line 14/16),
2) Generates **1095-C PDFs** (Part II; Part I/III optional),
3) Returns a ZIP:
   - `interim_full.xlsx`
   - `pdfs/1095C_<EmployeeID>.pdf`

## Endpoints

- `GET /health` → `{ "ok": true }`
- `POST /pipeline` → **multipart/form-data**
  - `year`: integer (e.g., `2025`)
  - `input_excel`: file (single workbook with sheets like `Emp Demographic`, `Emp Eligibility`, `Emp Enrollment`, optional `Emp Wait Period`, optional `Dep Enrollment`)
  - **Response**: `application/zip` with interim + PDFs, or JSON `{ "error": "..." }` on failure
  - **Auth**: header `x-api-key: <FASTAPI_API_KEY>` if configured

## Environment Variables

| Name                | Required | Example                                | Description                                     |
|---------------------|----------|----------------------------------------|-------------------------------------------------|
| `FASTAPI_API_KEY`   | No       | `supersecret`                          | If set, required via `x-api-key` header         |
| `PDF_TEMPLATE_PATH` | Yes      | `/opt/app/f1095c.pdf`                  | Path to blank 1095-C template                   |
| `FIELDS_JSON_PATH`  | Yes      | `/opt/app/pdf_acro_fields_details.json`| Field mapping JSON (Part I/II/III)              |
| `LOG_LEVEL`         | No       | `INFO`                                  | Logging level                                   |
| `LOG_FILE`          | No       | `/tmp/aca1095.log`                      | Rotating log file (best effort)                 |

### Field Map JSON

Top-level keys:
- `line14` (**required**) – supports `"all"` and/or monthly keys (`"Jan"`, `"Feb"`, …)
- `line16` (**required**) – same structure as line14
- `part1` (optional) – text fields: `employee_first`, `employee_last`, `employer_name`, etc.
- `part2` (optional) – `{ "plan_start_month": "<fieldId>" }` if needed
- `part3` (optional) – `{"rows":[ { "name":..., "ssn":..., "dob":..., "all":..., "months": {"Jan":..., ...} }, ... ] }`

> If `part1/part3` are missing, the service will skip them (no error).

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export PDF_TEMPLATE_PATH=/absolute/path/to/f1095c.pdf
export FIELDS_JSON_PATH=/absolute/path/to/pdf_acro_fields_details.json
export FASTAPI_API_KEY=supersecret

uvicorn main_fastapi:app --host 0.0.0.0 --port 8000
