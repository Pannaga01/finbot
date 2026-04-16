# Smart CSV Handling for FinBot RAG Pipeline

## Problem

Currently, CSV files (like `hr_data.csv` — 500 rows × 21 columns) are processed through the Docling `DocumentConverter`, which converts the entire CSV into a single document and chunks it as generic text. This leads to:

- **Huge context windows** — full CSV rows dumped into the LLM prompt
- **Poor retrieval** — semantic search on flattened table text is inaccurate
- **No computation** — aggregation queries (count, average, sum, filter) rely on the LLM to "do math" from raw text, which is unreliable

## Proposed Approach

### Three-layer CSV strategy:

1. **Row-level chunks** → Each CSV row becomes a natural-language sentence stored in Qdrant (for lookup queries like "What is Pavan Krishnan's leave balance?")
2. **Summary chunks** → Pre-computed group statistics (by department, role, location, etc.) stored in Qdrant (for queries like "How many employees are in the Finance department?")
3. **Pandas computation engine** → For aggregation/math queries detected at query time, compute the answer with pandas and send only the compact result to the LLM for natural language formatting

---

## Proposed Changes

### New Module

#### [NEW] [csv_handler.py](file:///c:/Users/upsre/OneDrive/Documents/Learning/AI%20bootcamp/assignment_FinBot/backend/csv_handler.py)

A dedicated module with three responsibilities:

**1. `parse_csv_to_row_chunks(file_path, collection, access_roles)`**
- Reads CSV with `pandas`
- Converts each row into a natural-language sentence, e.g.:
  ```
  Employee FINEMP1001 (Pavan Krishnan) is a Male HR Executive in the HR department
  at Senior level, based in Lucknow. Salary: ₹2,026,850. Leave balance: 14,
  leaves taken: 5. Attendance: 87.5%. Performance rating: 2.
  ```
- Returns a list of chunk dicts `{"text": ..., "metadata": {...}}` compatible with the existing Qdrant pipeline

**2. `generate_csv_summary_chunks(file_path, collection, access_roles)`**
- Uses pandas to pre-compute summaries grouped by key columns (department, gender, location, designation_level, employment_type, etc.)
- Example summary chunk:
  ```
  Department summary for Technology: 95 employees, average salary ₹1,845,320,
  average performance rating 3.2, average attendance 85.1%, total leaves taken 340.
  Gender breakdown: Male 50, Female 35, Non-Binary 10.
  ```
- Also generates an overall dataset summary chunk with totals

**3. `try_csv_computation(query, csv_path)`**
- Takes a user query and attempts to answer it using pandas operations
- Uses keyword matching to detect aggregation intent (count, average, total, how many, highest, lowest, etc.)
- Maps detected entities to column names and filters
- Returns `(True, result_text)` if computation succeeded, `(False, None)` otherwise
- The result_text is a compact 1-3 line answer, not raw data

---

### Existing Module Modifications

#### [MODIFY] [rag.py](file:///c:/Users/upsre/OneDrive/Documents/Learning/AI%20bootcamp/assignment_FinBot/backend/rag.py)

**Ingestion changes** (`load_documents_chunker`):
- Detect `.csv` files separately from other document types
- For CSV files: call `parse_csv_to_row_chunks()` + `generate_csv_summary_chunks()` instead of Docling
- For non-CSV files: keep existing Docling pipeline unchanged
- Merge all chunks before embedding

**Query-time changes** (`answer`):
- Before running the full RAG pipeline, call `try_csv_computation(query, csv_path)` for any CSV files the user's role has access to
- If computation succeeds, include the computed result as a **priority context** prepended to the retrieved chunks
- The LLM then uses this pre-computed result to formulate a natural language answer

#### [MODIFY] [requirements.txt](file:///c:/Users/upsre/OneDrive/Documents/Learning/AI%20bootcamp/assignment_FinBot/backend/requirements.txt)

- Add `pandas` as a dependency

---

## What Does NOT Change

- **Frontend** — no changes needed
- **`app.py`** — API contract stays the same
- **`router.py`** — routing logic unchanged
- **`config.py`** — no new config needed (CSV paths are auto-discovered from DATA_ROOT)
- **`output_guardrails.py`** — works on the final response as before

---

## Example Query Flow (After)

**Query:** "How many female employees are in HR?"

1. Router → routes to `hr`
2. `try_csv_computation()` detects aggregation keywords ("how many"), entity ("female", "HR")
3. Pandas filters: `df[(df['gender']=='Female') & (df['department']=='HR')]` → count = 12
4. Returns: `"Based on HR data: There are 12 female employees in the HR department."`
5. This compact result is prepended to the RAG context
6. LLM receives a clean, pre-computed answer + supporting chunks → produces accurate response

**Query:** "What is the leave balance for Pavan Krishnan?"

1. Router → routes to `hr`
2. `try_csv_computation()` detects entity ("Pavan Krishnan") + field ("leave balance")
3. Pandas lookup: `df[df['full_name']=='Pavan Krishnan']['leave_balance']` → 14
4. Returns: `"Pavan Krishnan's leave balance is 14 days."`
5. Prepended to context, LLM formats the final answer

---

## Verification Plan

### Automated Tests
- Delete the existing `qdrant_data` folder to force re-ingestion
- Run the FastAPI server and test with these queries via curl/Postman:
  - `"How many female employees are in HR?"` → should return a count
  - `"What is the leave balance for Pavan Krishnan?"` → should return 14
  - `"What is the average salary for Senior HR Executives?"` → should return a computed average
  - `"How many employees are in the Finance department?"` → should return a count
  - Non-CSV queries (e.g., "What is FinSolve's hybrid work model?") → should work as before via Docling

### Manual Verification
- Compare response quality before/after the change for CSV-based questions
- Verify that non-CSV document retrieval is unaffected
