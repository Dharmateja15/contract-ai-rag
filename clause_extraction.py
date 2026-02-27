import re
import json
import pdfplumber
import sys
import os


# -----------------------------
# Preprocess
# -----------------------------

def preprocess_text(text: str) -> str:
    """
    Normalize spacing while preserving structural newlines.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# -----------------------------
# PDF Reader
# -----------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File '{pdf_path}' not found.")

    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"

    return text


# -----------------------------
# Clause Split
# -----------------------------

def split_into_clauses(text: str):
    """
    Split using:
    - Double newlines
    - Numbered headings like '1.' or '2.1'
    """
    pattern = r"(?:\n{2,}|^\d{1,2}\.\s|^\d{1,2}\.\d{1,2}\s)"
    raw_clauses = re.split(pattern, text, flags=re.MULTILINE)

    cleaned_clauses = []

    for clause in raw_clauses:
        clause = clause.strip()

        if len(clause) < 50:
            continue

        if re.fullmatch(r"[a-zA-Z]|i+|v+|x+", clause.lower()):
            continue

        # Remove internal line breaks for clean JSON output
        clause = re.sub(r"\n+", " ", clause)

        cleaned_clauses.append(clause)

    return cleaned_clauses


# -----------------------------
# Classification (Priority-based)
# -----------------------------

def classify_clause(clause: str) -> str:
    clause_lower = clause.lower()

    # 1️⃣ Termination (highest priority)
    if any(word in clause_lower for word in ["terminate", "termination"]):
        return "Termination Clause"

    # 2️⃣ Confidentiality
    if "confidential" in clause_lower or "non-disclosure" in clause_lower:
        return "Confidentiality Clause"

    # 3️⃣ Liability
    if any(word in clause_lower for word in ["liability", "liable", "indemnify", "indemnity"]):
        return "Liability Clause"

    # 4️⃣ Intellectual Property
    if any(word in clause_lower for word in ["intellectual property", "ip rights", "copyright", "trademark"]):
        return "Intellectual Property Clause"

    # 5️⃣ Governing Law (strict detection)
    if "governing law" in clause_lower or "governed by the laws" in clause_lower:
        return "Governing Law Clause"

    # 6️⃣ Notice
    if any(word in clause_lower for word in ["notice shall", "written notice", "registered mail"]):
        return "Notice Clause"

    # 7️⃣ Assignment (avoid false positive from "successors and assigns")
    if "may not assign" in clause_lower or "assign this agreement" in clause_lower:
        return "Assignment Clause"

    # 8️⃣ Payment / Compensation
    if any(
        word in clause_lower
        for word in [
            "payment",
            "salary",
            "fee",
            "compensation",
            "amount",
            "consideration",
            "wage",
            "remuneration",
        ]
    ):
        return "Payment Clause"

    return "Other"


# -----------------------------
# Main Extraction
# -----------------------------

def extract_clauses_from_pdf(pdf_path: str):
    contract_text = extract_text_from_pdf(pdf_path)
    cleaned_text = preprocess_text(contract_text)
    clauses = split_into_clauses(cleaned_text)

    extracted_data = []

    for idx, clause in enumerate(clauses, start=1):
        extracted_data.append(
            {
                "clause_number": idx,
                "clause_text": clause,
                "clause_type": classify_clause(clause),
                "confidence_score": 0.80,
            }
        )

    return extracted_data


# -----------------------------
# Run from Terminal
# -----------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clause_extraction.py <contract.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    results = extract_clauses_from_pdf(pdf_path)

    # ensure_ascii=False prevents \u2026 type output
    print(json.dumps(results, indent=4, ensure_ascii=False))