# ACA 1095-C Builder

End-to-end tool to turn a single Excel workbook into:

* an **Interim** month-by-month table,
* a **Final** (Line 14/16) table,
* an optional **Penalty Dashboard**, and
* pre-filled **1095-C PDFs** (single or bulk).

The codebase is split so that **input processing** and **PDF filling** stay stable, while only the **interim/penalty logic** changes when your business rules change.

---

## Project Structure

```
.
├─ aca_processing.py   # 1) Input ingestion, cleaning & shared helpers (stable)
├─ aca_builder.py      # 2) Interim + Final + Penalty logic (change here)
├─ aca_pdf.py          # 3) PDF filling & Excel writer (stable)
├─ main_fastapi.py     # FastAPI routes using the three modules
├─ requirements.txt
└─ README.md
```

---

## Quick Start

### 1) Install

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run the API

```bash
uvicorn main_fastapi:app --reload --port 8000
```

### 3) Call the endpoints

* Health check

```bash
curl -H "x-api-key: supersecret-key-123" http://localhost:8000/health
```

* Process Excel → Final + Interim (+ Penalty)

```bash
curl -X POST "http://localhost:8000/process/excel" \
  -H "x-api-key: supersecret-key-123" \
  -F "excel=@Input Data Sample.xlsx" \
  -o final_interim_penalty.xlsx
```

* Generate **single** employee PDF

```bash
curl -X POST "http://localhost:8000/generate/single" \
  -H "x-api-key: supersecret-key-123" \
  -F "excel=@Input Data Sample.xlsx" \
  -F "pdf=@1095c_template.pdf" \
  -F "employee_id=1002" \
  -F "flattened_only=true" \
  -o 1095c_1002.pdf
```

* Generate **bulk** PDFs (all employees)

```bash
curl -X POST "http://localhost:8000/generate/bulk" \
  -H "x-api-key: supersecret-key-123" \
  -F "excel=@Input Data Sample.xlsx" \
  -F "pdf=@1095c_template.pdf" \
  -o 1095c_bulk.zip
```

> 🔑 The default API key is `supersecret-key-123`. Set `API_KEYS` env var (comma-separated) to change.

---

## Data Contract (Excel)

Your workbook can name sheets flexibly (we do fuzzy matching), but the **canonical** columns are:

* **Emp Demographic**
  `employeeid, firstname, lastname, ssn, addressline1, addressline2, city, state, zipcode, role, employmentstatus, statusstartdate, statusenddate`

* **Emp Status** *(optional; if missing we derive from Demographic)*
  `employeeid, employmentstatus, role, statusstartdate, statusenddate`

* **Emp Eligibility** *(drives MV + affordability + dependents eligibility)*
  `employeeid, iseligibleforcoverage, eligibilitystartdate, eligibilityenddate, plancode, eligibilitytier, plancost`

* **Emp Enrollment**
  `employeeid, isenrolled, enrollmentstartdate, enrollmentenddate, plancode, enrollmenttier`

* **Dep Enrollment**
  `employeeid, dependentrelationship, eligible, enrolled, eligiblestartdate, eligibleenddate, enrollmentstartdate, enrollmentenddate, plancode`

* **Pay Deductions** *(kept for compatibility; not used for affordability in UAT mode)*
  `employeeid, amount, startdate, enddate`

> Dates are flexibly parsed; blanks are allowed. Employee IDs are normalized (`"1001.0"` → `"1001"`). Booleans accept `yes/no`, `y/n`, `1/0`, etc.

---

## How the Modules Work

### 1) `aca_processing.py` (stable)

* **Loads & normalizes** all sheets (case/space insensitive columns; aliases for common typos).
* **Coerces types** safely (NaN-safe integer and year parsing).
* Provides **reusable helpers**: overlap checks, month bounds, employee ID normalization, etc.
* Exposes:

  * `load_excel(bytes)`
  * `prepare_inputs(data_dict)` → cleaned DataFrames
  * `choose_report_year(emp_elig)`
  * constants: `MONTHS`, `FULL_MONTHS`, …
  * helpers used everywhere (e.g., `_coerce_str`, `_int_year`, `_safe_int`)

> You **shouldn’t** need to change this file when business rules change.

### 2) `aca_builder.py` (change here)

* Computes:

  * **Interim** table (per employee × month) with flags:

    * `employed, ft, parttime, eligibleforcoverage, eligible_allmonth, eligible_mv, offer_ee_allmonth, enrolled_allmonth, offer_spouse, offer_dependents, spouse_eligible, child_eligible, spouse_enrolled, child_enrolled, waitingperiod_month`
    * Codes: `line14_final`, `line16_final`
    * Year-level flag: `line14_all12` = `"1G"` if **never full-time** that year (per IRS rules).
  * **Final** table (Line 14 & 16 by month)
  * **Penalty Dashboard** (A=1H, B=1E+not enrolled), with friendly reasons
* **Affordability** (UAT mode): uses **Eligibility** `plancost` for **EMP** tier and compares to threshold **$50**.

  * If PlanA is offered (minimum value) and spouse+dependents are offered, we emit **1A** if affordable, otherwise **1E**.
  * If MV not offered but full-month offer exists → **1F**.
  * If no full-month offer → **1H** (unless year-level **1G** applies).

> Any change in ACA coding logic should be made **only** in this file.

### 3) `aca_pdf.py` (stable)

* Fills **Part I** (name, SSN, address) and **Part II** (Line 14 & 16).
* If an employee has `line14_all12 == "1G"`, fills **“All 12 months = 1G”** and leaves monthly boxes blank (as instructed).
* Produces both **editable** and **flattened** versions.
* `save_excel_outputs()` writes the three sheets to an output workbook.

> Field names are the 2024 1095-C PDF’s; update only if the IRS form changes.

---

## API Endpoints

* `POST /process/excel`
  **In:** `excel` (.xlsx)
  **Out:** Excel file with sheets: `Final {year}`, `Interim {year}`, `Penalty Dashboard {year}`

* `POST /generate/single`
  **In:** `excel` (.xlsx), `pdf` (1095-C template), `employee_id`, `flattened_only` (true/false)
  **Out:** PDF (flattened) or ZIP (editable + flattened)

* `POST /generate/bulk`
  **In:** `excel` (.xlsx), `pdf` (template), optional `employee_ids` (JSON array)
  **Out:** ZIP of flattened PDFs

**Notes**

* The server auto-detects the **report year** from Eligibility ranges, but you can add a `filing_year` form field and pass it through using `_int_year(...)` if you want to override.
* CORS is open by default; lock it down for production.
* API authentication uses the header `x-api-key`.

---

## Configuration

* `API_KEYS` — comma-separated list of valid keys.
  Default: `supersecret-key-123`.

---

## Troubleshooting

* **“float object cannot be interpreted as an integer”** or **“cannot convert float NaN to integer”**
  The code is **NaN-safe** (via `_int_year` / `_safe_int`). If you added manual `int(...)` casts in new code/handlers, replace them with `_int_year` or `_safe_int`.

* **Columns missing**
  The processor creates empty columns for missing sheets, but logic may require specific ones (see “Data Contract”). Verify your sheet headers and their spelling (case/space don’t matter).

* **PDF fields don’t appear**
  Ensure your template is the 1095-C PDF with the expected field names; otherwise update `F_PART1`, `F_L14`, `F_L16` in `aca_pdf.py`.

---

## Extending / Changing Rules

1. Open **`aca_builder.py`**.
2. Edit the logic inside `build_interim()` (e.g., how Line 14/16 are assigned).
3. Leave `aca_processing.py` and `aca_pdf.py` untouched unless the **input format** or **IRS PDF** changes.

---

## License

Internal use. Add your organization’s license here.

---

## Changelog (highlights)

* Split monolith into **processing / builder / pdf** modules.
* Added NaN-safe year & month handling.
* Implemented year-level **1G** behavior.
* Penalty dashboard sheet integrated into `/process/excel` output.



This is the app link
https://v0-aca-1095-generator.vercel.app/
