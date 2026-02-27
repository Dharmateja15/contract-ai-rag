# contract-ai-rag
extract_clauses(file) -> list of dict

check_missing_clauses(contract_type, clauses) -> dict

analyze_clause(contract_type, title, text) -> dict

analyze_contract(contract_type, file) -> final dict

final output  :
{
  "contract_type": "",
  "missing_clauses": [],
  "clauses": [
      {
          "title": "",
          "risk_level": "",
          "explanation": ""
      }
  ]
}

Module 1 – Clause Extraction (Student: Dharma)


Reads contract PDF using pdfplumber and normalizes text while preserving structural newlines.


Splits the contract into clauses using double newlines and numbered headings (e.g., 1., 2.1).


Classifies each clause into basic types (Termination, Confidentiality, Liability, IP, Governing Law, Notice, Assignment, Payment, Other) using rule-based keyword matching.


Outputs structured JSON: clause_number, clause_text, clause_type, and a placeholder confidence_score, which is consumed by the retrieval and LLM modules.