# csv_handler.py
# ----------------
# Smart CSV handling for FinBot RAG pipeline.
# Instead of sending raw CSV to the LLM, this module:
#   1. Parses CSV with pandas
#   2. Converts rows into natural-language chunks for Qdrant
#   3. Pre-computes group summaries as chunks for Qdrant
#   4. Provides a pandas computation engine for aggregation queries

import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd

from config import ROLE_ACCESS


# ---------------------------------------------------------------------------
# 1. ROW-LEVEL CHUNKING
# ---------------------------------------------------------------------------

def _safe_int(val, default="N/A"):
    """Safely convert a value to int, returning default if NaN or missing."""
    if pd.isna(val):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_val(val, default="N/A"):
    """Return val if not NaN, else default."""
    if pd.isna(val):
        return default
    return val


def _row_to_text(row: pd.Series) -> str:
    """
    Convert a single CSV row into a human-readable sentence.
    Tailored for the hr_data.csv schema. Handles NaN values gracefully.
    """
    salary = _safe_int(row.get('salary'))
    salary_fmt = f"₹{salary:,}" if isinstance(salary, int) else "N/A"

    text = (
        f"Employee {_safe_val(row.get('employee_id'))} ({_safe_val(row.get('full_name'))}) "
        f"is a {_safe_val(row.get('gender'))} {_safe_val(row.get('role'))} "
        f"in the {_safe_val(row.get('department'))} department "
        f"at {_safe_val(row.get('designation_level'))} level, "
        f"based in {_safe_val(row.get('location'))}. "
        f"Employment type: {_safe_val(row.get('employment_type'))}, "
        f"status: {_safe_val(row.get('employment_status'))}. "
        f"Salary: {salary_fmt}. "
        f"Leave balance: {_safe_val(row.get('leave_balance'))}, "
        f"leaves taken: {_safe_val(row.get('leaves_taken'))}. "
        f"Attendance: {_safe_val(row.get('attendance_pct'))}%. "
        f"Performance rating: {_safe_val(row.get('performance_rating'))}. "
        f"Date of joining: {_safe_val(row.get('date_of_joining'))}. "
        f"Manager ID: {_safe_val(row.get('manager_id'))}."
    )
    return text


def _generate_chunk_id(text: str, source: str) -> str:
    return hashlib.md5((text + source).encode()).hexdigest()


def parse_csv_to_row_chunks(
    file_path: Path,
    collection: str,
    access_roles: List[str],
) -> List[Dict[str, Any]]:
    """
    Read a CSV file and convert each row into a text chunk
    with metadata, ready for Qdrant ingestion.
    """
    df = pd.read_csv(file_path)
    filename = file_path.name
    chunks = []

    for idx, row in df.iterrows():
        text = _row_to_text(row)
        chunk_id = _generate_chunk_id(text, filename)

        chunks.append({
            "text": text,
            "metadata": {
                "chunk_id": chunk_id,
                "source_document": filename,
                "collection": collection,
                "access_roles": access_roles,
                "section_title": f"Employee Record: {row.get('full_name', 'N/A')}",
                "page_number": None,
                "chunk_type": "csv_row",
                "parent_chunk_id": None,
            },
        })

    print(f"[csv_handler] Parsed {len(chunks)} row chunks from {filename}")
    return chunks


# ---------------------------------------------------------------------------
# 2. SUMMARY CHUNKING — pre-computed group statistics
# ---------------------------------------------------------------------------

def generate_csv_summary_chunks(
    file_path: Path,
    collection: str,
    access_roles: List[str],
) -> List[Dict[str, Any]]:
    """
    Generate pre-computed summary chunks grouped by department,
    gender, location, designation level, etc.
    """
    df = pd.read_csv(file_path)
    filename = file_path.name
    summaries: List[Dict[str, Any]] = []

    # Fill NaN in numeric columns to avoid int conversion errors
    numeric_cols = ["salary", "leave_balance", "leaves_taken", "attendance_pct", "performance_rating"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # --- Overall dataset summary ---
    overall = (
        f"HR Dataset Overview ({filename}): "
        f"Total employees: {len(df)}. "
        f"Departments: {', '.join(df['department'].unique())}. "
        f"Locations: {', '.join(df['location'].unique())}. "
        f"Average salary: ₹{int(df['salary'].mean()):,}. "
        f"Average performance rating: {df['performance_rating'].mean():.2f}. "
        f"Average attendance: {df['attendance_pct'].mean():.1f}%. "
        f"Total leaves taken: {int(df['leaves_taken'].sum())}. "
        f"Gender distribution: {df['gender'].value_counts().to_dict()}. "
        f"Employment type distribution: {df['employment_type'].value_counts().to_dict()}. "
        f"Employment status distribution: {df['employment_status'].value_counts().to_dict()}."
    )
    summaries.append(_make_summary_chunk(overall, filename, collection, access_roles, "Overall HR Summary"))

    # --- Per-department summaries ---
    for dept, group in df.groupby("department"):
        text = (
            f"Department summary for {dept}: "
            f"{len(group)} employees. "
            f"Average salary: ₹{int(group['salary'].mean()):,}. "
            f"Average performance rating: {group['performance_rating'].mean():.2f}. "
            f"Average attendance: {group['attendance_pct'].mean():.1f}%. "
            f"Total leaves taken: {int(group['leaves_taken'].sum())}. "
            f"Average leave balance: {group['leave_balance'].mean():.1f}. "
            f"Gender breakdown: {group['gender'].value_counts().to_dict()}. "
            f"Designation levels: {group['designation_level'].value_counts().to_dict()}. "
            f"Locations: {group['location'].value_counts().to_dict()}."
        )
        summaries.append(_make_summary_chunk(text, filename, collection, access_roles, f"Department Summary: {dept}"))

    # --- Per-location summaries ---
    for loc, group in df.groupby("location"):
        text = (
            f"Location summary for {loc}: "
            f"{len(group)} employees. "
            f"Departments: {group['department'].value_counts().to_dict()}. "
            f"Average salary: ₹{int(group['salary'].mean()):,}. "
            f"Average performance rating: {group['performance_rating'].mean():.2f}."
        )
        summaries.append(_make_summary_chunk(text, filename, collection, access_roles, f"Location Summary: {loc}"))

    # --- Per-gender summaries ---
    for gender, group in df.groupby("gender"):
        text = (
            f"Gender summary for {gender}: "
            f"{len(group)} employees. "
            f"Departments: {group['department'].value_counts().to_dict()}. "
            f"Average salary: ₹{int(group['salary'].mean()):,}. "
            f"Average performance rating: {group['performance_rating'].mean():.2f}."
        )
        summaries.append(_make_summary_chunk(text, filename, collection, access_roles, f"Gender Summary: {gender}"))

    # --- Per designation-level summaries ---
    for level, group in df.groupby("designation_level"):
        text = (
            f"Designation level summary for {level}: "
            f"{len(group)} employees. "
            f"Departments: {group['department'].value_counts().to_dict()}. "
            f"Average salary: ₹{int(group['salary'].mean()):,}. "
            f"Average performance rating: {group['performance_rating'].mean():.2f}."
        )
        summaries.append(_make_summary_chunk(text, filename, collection, access_roles, f"Designation Summary: {level}"))

    # --- Per department + gender cross-tabulation ---
    for (dept, gender), group in df.groupby(["department", "gender"]):
        text = (
            f"There are {len(group)} {gender} employees in the {dept} department. "
            f"Average salary: ₹{int(group['salary'].mean()):,}. "
            f"Average performance rating: {group['performance_rating'].mean():.2f}."
        )
        summaries.append(_make_summary_chunk(text, filename, collection, access_roles, f"Dept-Gender: {dept} - {gender}"))

    # --- Per department + designation cross-tabulation ---
    for (dept, role), group in df.groupby(["department", "role"]):
        text = (
            f"Role summary: {len(group)} {role}(s) in the {dept} department. "
            f"Average salary: ₹{int(group['salary'].mean()):,}. "
            f"Average performance rating: {group['performance_rating'].mean():.2f}. "
            f"Average leave balance: {group['leave_balance'].mean():.1f}."
        )
        summaries.append(_make_summary_chunk(text, filename, collection, access_roles, f"Role Summary: {dept} - {role}"))

    print(f"[csv_handler] Generated {len(summaries)} summary chunks from {filename}")
    return summaries


def _make_summary_chunk(
    text: str,
    filename: str,
    collection: str,
    access_roles: List[str],
    section_title: str,
) -> Dict[str, Any]:
    return {
        "text": text,
        "metadata": {
            "chunk_id": _generate_chunk_id(text, filename),
            "source_document": filename,
            "collection": collection,
            "access_roles": access_roles,
            "section_title": section_title,
            "page_number": None,
            "chunk_type": "csv_summary",
            "parent_chunk_id": None,
        },
    }


# ---------------------------------------------------------------------------
# 3. PANDAS COMPUTATION ENGINE — for aggregation queries at query time
# ---------------------------------------------------------------------------

# Keywords that signal an aggregation/computation query
_AGGREGATION_KEYWORDS = [
    "how many", "count", "total", "number of",
    "average", "mean", "avg",
    "sum", "highest", "lowest", "maximum", "minimum",
    "top", "bottom", "most", "least", "max", "min",
    "what is the salary", "what is the leave",
    "above", "below", "greater than", "less than",
    "percentage", "ratio",
]

# Map natural-language terms to DataFrame column names
_COLUMN_MAP = {
    "salary": "salary",
    "leave balance": "leave_balance",
    "leaves taken": "leaves_taken",
    "attendance": "attendance_pct",
    "performance rating": "performance_rating",
    "performance": "performance_rating",
    "rating": "performance_rating",
    "department": "department",
    "gender": "gender",
    "location": "location",
    "role": "role",
    "designation": "designation_level",
    "designation level": "designation_level",
    "employment type": "employment_type",
    "employment status": "employment_status",
    "name": "full_name",
    "employee": "full_name",
    "manager": "manager_id",
    "joining date": "date_of_joining",
    "date of joining": "date_of_joining",
    "email": "email",
    "phone": "phone",
}

# Department names for entity detection
_DEPARTMENTS = ["HR", "Technology", "Finance", "Sales", "Marketing", "Operations", "Legal", "Administration"]
_GENDERS = ["Female", "Male", "Non-Binary"]


def _is_aggregation_query(query: str) -> bool:
    
    """Check if the query needs pandas computation."""
    q_lower = query.lower()
    return any(kw in q_lower for kw in _AGGREGATION_KEYWORDS)


def _detect_department(query: str) -> Optional[str]:
    """Detect if a department name is mentioned in the query."""
    q_lower = query.lower()
    for dept in _DEPARTMENTS:
        if dept.lower() in q_lower:
            return dept
    return None


def _detect_gender(query: str) -> Optional[str]:
    q_lower = query.lower()
    for g in _GENDERS:
        if g.lower() in q_lower:
            return g
    return None


def _detect_person(query: str, df: pd.DataFrame) -> Optional[str]:
    """Detect if a person's name is mentioned in the query."""
    q_lower = query.lower()
    for name in df["full_name"].unique():
        if name.lower() in q_lower:
            return name
    return None


def _detect_numeric_threshold(query: str) -> Optional[float]:
    """Extract a numeric threshold from the query (e.g., 'above 4')."""
    import re
    match = re.search(r'(?:above|below|greater than|less than|over|under|more than)\s+(\d+(?:\.\d+)?)', query.lower())
    if match:
        return float(match.group(1))
    return None


def try_csv_computation(query: str, csv_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Attempt to answer the query using pandas computation.
    
    Returns:
        (True, result_text) if computation succeeded
        (False, None) if this query doesn't need computation or couldn't be computed
    """
    if not csv_path.exists():
        return False, None

    q_lower = query.lower()
    df = pd.read_csv(csv_path)

    # --- Person-specific lookup ---
    person = _detect_person(query, df)
    if person:
        row = df[df["full_name"] == person]
        if row.empty:
            return True, f"No employee found with name '{person}'."

        r = row.iloc[0]
        result = (
            f"Employee details for {person} (ID: {r['employee_id']}):\n"
            f"  Role: {r['role']}, Department: {r['department']}, Level: {r['designation_level']}\n"
            f"  Location: {r['location']}, Gender: {r['gender']}\n"
            f"  Salary: ₹{int(r['salary']):,}\n"
            f"  Leave balance: {r['leave_balance']}, Leaves taken: {r['leaves_taken']}\n"
            f"  Attendance: {r['attendance_pct']}%, Performance rating: {r['performance_rating']}\n"
            f"  Employment type: {r['employment_type']}, Status: {r['employment_status']}\n"
            f"  Date of joining: {r['date_of_joining']}, Manager ID: {r['manager_id']}"
        )
        return True, result

    # --- Only try aggregation for aggregation-type queries ---
    if not _is_aggregation_query(query):
        return False, None

    dept = _detect_department(query)
    gender = _detect_gender(query)
    threshold = _detect_numeric_threshold(query)

    # Apply department filter
    filtered = df.copy()
    filter_desc = []
    if dept:
        filtered = filtered[filtered["department"] == dept]
        filter_desc.append(f"in {dept} department")
    if gender:
        filtered = filtered[filtered["gender"] == gender]
        filter_desc.append(f"who are {gender}")

    filter_str = " ".join(filter_desc) if filter_desc else "across all employees"

    if filtered.empty:
        return True, f"No employees found {filter_str}."

    # --- "How many" / count queries ---
    if any(kw in q_lower for kw in ["how many", "count", "number of", "total number"]):
        # Check for threshold-based counting
        if threshold is not None:
            if "performance" in q_lower or "rating" in q_lower:
                if "above" in q_lower or "greater" in q_lower or "over" in q_lower or "more than" in q_lower:
                    count = len(filtered[filtered["performance_rating"] > threshold])
                else:
                    count = len(filtered[filtered["performance_rating"] < threshold])
                return True, f"Number of employees {filter_str} with performance rating {'above' if 'above' in q_lower else 'below'} {threshold}: {count}"

            if "salary" in q_lower:
                if "above" in q_lower or "greater" in q_lower or "over" in q_lower or "more than" in q_lower:
                    count = len(filtered[filtered["salary"] > threshold])
                else:
                    count = len(filtered[filtered["salary"] < threshold])
                return True, f"Number of employees {filter_str} with salary {'above' if 'above' in q_lower else 'below'} ₹{int(threshold):,}: {count}"

        count = len(filtered)
        return True, f"Number of employees {filter_str}: {count}"

    # --- Average queries ---
    if any(kw in q_lower for kw in ["average", "mean", "avg"]):
        if "salary" in q_lower:
            avg = filtered["salary"].mean()
            return True, f"Average salary {filter_str}: ₹{int(avg):,}"
        if "performance" in q_lower or "rating" in q_lower:
            avg = filtered["performance_rating"].mean()
            return True, f"Average performance rating {filter_str}: {avg:.2f}"
        if "attendance" in q_lower:
            avg = filtered["attendance_pct"].mean()
            return True, f"Average attendance {filter_str}: {avg:.1f}%"
        if "leave" in q_lower:
            avg = filtered["leave_balance"].mean()
            return True, f"Average leave balance {filter_str}: {avg:.1f} days"

    # --- Total / sum queries ---
    if any(kw in q_lower for kw in ["total", "sum"]):
        if "salary" in q_lower:
            total = filtered["salary"].sum()
            return True, f"Total salary {filter_str}: ₹{int(total):,}"
        if "leave" in q_lower:
            if "taken" in q_lower:
                total = filtered["leaves_taken"].sum()
                return True, f"Total leaves taken {filter_str}: {int(total)}"
            else:
                total = filtered["leave_balance"].sum()
                return True, f"Total leave balance {filter_str}: {int(total)}"

    # --- Highest / max queries ---
    if any(kw in q_lower for kw in ["highest", "maximum", "max", "top", "most"]):
        if "salary" in q_lower:
            idx = filtered["salary"].idxmax()
            emp = filtered.loc[idx]
            return True, f"Highest salary {filter_str}: {emp['full_name']} ({emp['role']}) with ₹{int(emp['salary']):,}"
        if "performance" in q_lower or "rating" in q_lower:
            idx = filtered["performance_rating"].idxmax()
            emp = filtered.loc[idx]
            return True, f"Highest performance rating {filter_str}: {emp['full_name']} ({emp['role']}) with rating {emp['performance_rating']}"

    # --- Lowest / min queries ---
    if any(kw in q_lower for kw in ["lowest", "minimum", "min", "bottom", "least"]):
        if "salary" in q_lower:
            idx = filtered["salary"].idxmin()
            emp = filtered.loc[idx]
            return True, f"Lowest salary {filter_str}: {emp['full_name']} ({emp['role']}) with ₹{int(emp['salary']):,}"
        if "performance" in q_lower or "rating" in q_lower:
            idx = filtered["performance_rating"].idxmin()
            emp = filtered.loc[idx]
            return True, f"Lowest performance rating {filter_str}: {emp['full_name']} ({emp['role']}) with rating {emp['performance_rating']}"

    # --- Fallback: couldn't parse the aggregation ---
    return False, None


# ---------------------------------------------------------------------------
# 4. PUBLIC: Get all CSV file paths for a given collection
# ---------------------------------------------------------------------------

def get_csv_files_for_collection(data_root: Path, collection: str) -> List[Path]:
    """Return all CSV file paths in a collection folder."""
    folder = data_root / collection
    if not folder.exists():
        return []
    return [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".csv"]
