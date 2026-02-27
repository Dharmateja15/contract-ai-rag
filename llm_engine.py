import os
import json
from groq import Groq

# ==========================================
# CONFIGURATION
# ==========================================

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment. Please check your GitHub Secrets.")

client = Groq(api_key=api_key)

# Llama 3 70B is used for legal reasoning
MODEL_NAME = "openai/gpt-oss-20b"


# ==========================================
# PROMPT BUILDER (BATCH)
# ==========================================

def build_batch_prompt(contract_type, enriched_clauses):
    """
    enriched_clauses: list of items from retrieval_engine.process_clauses
      each item has: clause_number, clause, clause_type, similar_clauses
    """
    clause_blocks = []

    for item in enriched_clauses:
        num = item["clause_number"]
        ctype = item["clause_type"]
        text = item["clause"]
        sims = item.get("similar_clauses", [])

        if sims:
            sims_block = "\n".join(
                [f"- {s['text']} (Similarity: {s.get('score', 'N/A')})" for s in sims]
            )
        else:
            sims_block = "No similar precedent clauses found."

        clause_blocks.append(
            f"""Clause {num}:
Type: {ctype}
Text: {text}

Relevant Precedents:
{sims_block}
"""
        )

    clauses_section = "\n\n".join(clause_blocks)

    return f"""
You are a legal contract risk analysis assistant.

Contract Type: {contract_type}

Below are multiple clauses from this contract and their closest precedent clauses (if any).

For EACH clause, you must:
1. Compare the clause with its precedents.
2. Identify deviations that increase legal or financial risk.
3. Assign a risk level: "Low", "Medium", or "High".
4. Provide a concise explanation (maximum 2 sentences).

Return ONLY a valid JSON object with this exact structure:
{{
  "results": [
    {{
      "clause_number": <number>,
      "risk_level": "Low/Medium/High",
      "explanation": "Short explanation text"
    }},
    ...
  ]
}}

Clauses:
{clauses_section}
"""


# ==========================================
# BATCH ANALYSIS FUNCTION
# ==========================================

def analyze_batch_risk(contract_type, enriched_clauses):
    """
    Single LLM call for all clauses in a contract.

    Returns:
        risk_map: dict[int, dict]  # clause_number -> {risk_level, explanation}
    """
    if not enriched_clauses:
        return {}

    prompt = build_batch_prompt(contract_type, enriched_clauses)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that outputs only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=MODEL_NAME,
            # We expect a JSON object with a "results" array
            response_format={"type": "json_object"}
        )

        content = chat_completion.choices[0].message.content
        parsed = json.loads(content)

        # Expect: {"results": [ {...}, {...} ]}
        results = parsed.get("results", [])
        risk_map = {}

        for item in results:
            num = item.get("clause_number")
            if num is None:
                continue
            risk_map[int(num)] = {
                "risk_level": item.get("risk_level", "Unknown"),
                "explanation": item.get("explanation", "Analysis complete.")
            }

        return risk_map

    except Exception as e:
        # On error, mark all clauses as High risk with same explanation
        return {
            item["clause_number"]: {
                "risk_level": "High",
                "explanation": f"Groq Engine Error: {str(e)}"
            }
            for item in enriched_clauses
        }
