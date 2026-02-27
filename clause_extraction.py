import re
import json
import pdfplumber
import sys
import os


# -----------------------------
# Preprocess
# -----------------------------

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -----------------------------
# PDF Reader
# -----------------------------

def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File '{pdf_path}' not found.")

    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text


# -----------------------------
# Clause Split (Improved)
# -----------------------------

def split_into_clauses(text):
    # Split on headings or numbered sections
    raw_clauses = re.split(r'\n|\d+\.\s', text)

    cleaned_clauses = []

    for clause in raw_clauses:
        clause = clause.strip()

        # Remove very short fragments
        if len(clause) < 50:
            continue

        # Remove single letters or roman numerals
        if re.fullmatch(r'[a-zA-Z]|i+|v+|x+', clause.lower()):
            continue

        cleaned_clauses.append(clause)

    return cleaned_clauses


# -----------------------------
# Classification
# -----------------------------

def classify_clause(clause):
    clause_lower = clause.lower()

    # Priority matters!
    if any(word in clause_lower for word in ["terminate", "termination"]):
        return "Termination Clause"

    elif any(word in clause_lower for word in ["confidential"]):
        return "Confidentiality Clause"

    elif any(word in clause_lower for word in ["liability", "liable"]):
        return "Liability Clause"

    elif any(word in clause_lower for word in ["intellectual property", "ip rights"]):
        return "Intellectual Property Clause"

    elif any(word in clause_lower for word in ["payment", "salary", "fee", "compensation", "amount"]):
        return "Payment Clause"

    else:
        return "Other"

# -----------------------------
# Main Extraction
# -----------------------------

def extract_clauses_from_pdf(pdf_path):
    contract_text = extract_text_from_pdf(pdf_path)
    cleaned_text = preprocess_text(contract_text)
    clauses = split_into_clauses(cleaned_text)

    extracted_data = []

    for idx, clause in enumerate(clauses, start=1):
        extracted_data.append({
            "clause_number": idx,
            "clause_text": clause,
            "clause_type": classify_clause(clause),
            "confidence_score": 0.80  # placeholder for future ML
        })

    return extracted_data


# -----------------------------
# Run from Terminal (Optional)
# -----------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python clause_extraction.py <contract.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    results = extract_clauses_from_pdf(pdf_path)

    print(json.dumps(results, indent=4))