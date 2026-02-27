import os
import json

from clause_extraction import extract_clauses_from_pdf
from retrieval_engine import process_clauses as find_similarities
from llm_engine import analyze_batch_risk

# ==========================================
# REQUIRED CLAUSES PER CONTRACT TYPE
# ==========================================

REQUIRED_CLAUSES = {
    "Employment": [
        "Termination Clause",
        "Confidentiality Clause",
        "Payment Clause",
        "Notice Clause",
        "Governing Law Clause",
    ],
    "NDA": [
        "Confidentiality Clause",
        "Termination Clause",
        "Governing Law Clause",
    ],
    "Service": [
        "Payment Clause",
        "Liability Clause",
        "Confidentiality Clause",
        "Termination Clause",
        "Governing Law Clause",
        "Notice Clause",
    ],
    "Vendor": [
        "Payment Clause",
        "Liability Clause",
        "Intellectual Property Clause",
        "Termination Clause",
        "Governing Law Clause",
    ],
    "Lease": [
        "Payment Clause",
        "Termination Clause",
        "Liability Clause",
        "Governing Law Clause",
    ],
}


# ==========================================
# MAIN PIPELINE
# ==========================================

def run_analysis_pipeline(pdf_path: str, contract_type: str):
    """
    End‑to‑end pipeline:
    1) Extract clauses from PDF
    2) Retrieve similar precedent clauses (FAISS)
    3) Single batch LLM call for risk analysis
    4) Compute missing clauses for this contract type
    """

    print(f"\n--- 1. Extracting Clauses from: {pdf_path} ---")
    extracted_raw = extract_clauses_from_pdf(pdf_path)

    print(f"--- 2. Finding Similar Precedents for {contract_type} ---")
    enriched_clauses = find_similarities(extracted_raw, contract_type)

    print("--- 3. Performing Batch Risk Analysis with Groq ---")
    # ONE Groq call for all clauses
    risk_map = analyze_batch_risk(contract_type, enriched_clauses)
    # risk_map: {clause_number: {risk_level, explanation}}

    final_clauses = []
    found_types = set()

    for item in enriched_clauses:
        clause_type = item["clause_type"]
        found_types.add(clause_type)
        num = item["clause_number"]

        risk_data = risk_map.get(
            num,
            {
                "risk_level": "Unknown",
                "explanation": "No risk data returned for this clause.",
            },
        )

        final_clauses.append(
            {
                "title": clause_type,
                "risk_level": risk_data["risk_level"],
                "explanation": risk_data["explanation"],
            }
        )

    required_list = REQUIRED_CLAUSES.get(contract_type, [])
    missing = [c for c in required_list if c not in found_types]

    return {
        "contract_type": contract_type,
        "missing_clauses": missing,
        "clauses": final_clauses,
    }


# ==========================================
# EXECUTION BLOCK (CLI TEST)
# ==========================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python pipeline.py <contract.pdf> <ContractType>")
        print("Example: python pipeline.py 1.employment_contract_sample.pdf Employment")
        sys.exit(1)

    pdf_path = sys.argv[1]
    contract_type = sys.argv[2]  # e.g., Employment, NDA, Service, Vendor, Lease

    if os.path.exists(pdf_path):
        final_report = run_analysis_pipeline(pdf_path, contract_type)

        print("\n" + "=" * 50)
        print("CONTRACT ANALYSIS FINAL REPORT")
        print("=" * 50)
        print(json.dumps(final_report, indent=4, ensure_ascii=False))
    else:
        print(f"Error: Could not find '{pdf_path}' in your directory.")
