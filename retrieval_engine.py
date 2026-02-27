import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_K = 2
SIMILARITY_THRESHOLD = 0.45  # Minimum cosine similarity to accept match

# ==========================================
# LOAD EMBEDDING MODEL (ONCE)
# ==========================================

model = SentenceTransformer(MODEL_NAME)

# ==========================================
# PRECEDENT DATABASE
# (Replace with JSON/Database later)
# ==========================================

precedents = [
    # Employment Contract
    {
        "contract_type": "Employment",
        "clause_type": "Payment Clause",
        "text": "Salary must be paid within 30 days of invoice receipt."
    },
    {
        "contract_type": "Employment",
        "clause_type": "Payment Clause",
        "text": "Employee compensation includes bonus and stock options."
    },
    {
        "contract_type": "Employment",
        "clause_type": "Termination Clause",
        "text": "Either party may terminate employment with 60 days written notice."
    },
    {
        "contract_type": "Employment",
        "clause_type": "Confidentiality Clause",
        "text": "Employee shall not disclose confidential information during employment."
    },

    # NDA Contract
    {
        "contract_type": "NDA",
        "clause_type": "Confidentiality Clause",
        "text": "Confidential information shall not be disclosed for 5 years."
    },
    {
        "contract_type": "NDA",
        "clause_type": "Confidentiality Clause",
        "text": "The receiving party must protect proprietary information."
    }
]

# ==========================================
# GLOBAL STORAGE
# ==========================================

indexes = {}
metadata_store = {}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def normalize_vectors(vectors):
    """
    Normalize vectors safely for cosine similarity.
    Prevents divide-by-zero errors.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


# ==========================================
# BUILD INDEXES (ONCE)
# ==========================================

def build_indexes():
    grouped_data = defaultdict(list)

    # Group precedents by (contract_type, clause_type)
    for item in precedents:
        key = (item["contract_type"], item["clause_type"])
        grouped_data[key].append(item["text"])

    # Build FAISS index for each group
    for key, texts in grouped_data.items():
        embeddings = model.encode(texts)
        embeddings = normalize_vectors(embeddings)

        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(np.array(embeddings))

        indexes[key] = index
        metadata_store[key] = texts


# Build indexes immediately
build_indexes()


# ==========================================
# RETRIEVE SIMILAR CLAUSES
# ==========================================

def retrieve_similar(clause_text, contract_type, clause_type, top_k=TOP_K):
    if not clause_text:
        return []

    key = (contract_type, clause_type)

    if key not in indexes:
        return []

    index = indexes[key]
    stored_texts = metadata_store[key]

    # Encode query
    query_vector = model.encode([clause_text])
    query_vector = normalize_vectors(query_vector)

    scores, indices = index.search(
        np.array(query_vector),
        min(top_k, len(stored_texts))
    )

    results = []

    for idx, score in zip(indices[0], scores[0]):
        score = float(score)

        if score < SIMILARITY_THRESHOLD:
            continue

        results.append({
            "text": stored_texts[idx],
            "score": round(score, 4)
        })

    return results


# ==========================================
# MAIN FUNCTION CALLED BY PIPELINE
# ==========================================

def process_clauses(clause_list, contract_type):
    """
    Input:
        clause_list -> List[Dict] (from extraction)
        contract_type -> str (from UI selection)

    Output:
        List[Dict] enriched with similar clauses
    """

    output = []

    for clause in clause_list:
        clause_text = clause.get("clause_text", "")
        clause_type = clause.get("clause_type", "Other")

        similar_clauses = retrieve_similar(
            clause_text,
            contract_type,
            clause_type
        )

        output.append({
            "clause_number": clause.get("clause_number"),
            "clause": clause_text,
            "clause_type": clause_type,
            "confidence_score": clause.get("confidence_score"),
            "similar_clauses": similar_clauses
        })

    return output


# ==========================================
# LOCAL TEST
# ==========================================

if __name__ == "__main__":

    sample_input = [
        {
            "clause_number": 4,
            "clause_text": "Payment must be completed within 45 days.",
            "clause_type": "Payment Clause",
            "confidence_score": 0.8
        }
    ]

    result = process_clauses(sample_input, contract_type="Employment")
    print(result)