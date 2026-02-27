# ⚖️ Open Contract Validator (OCV)

AI-Powered Contract Risk Analysis using Retrieval-Augmented Generation (RAG)

---

## 📌 Overview

Open Contract Validator (OCV) is a hackathon project that analyzes legal contracts using AI.

The system:

- Extracts clauses from a contract PDF
- Classifies clauses into legal categories
- Retrieves semantically similar precedent clauses
- Uses a Large Language Model (LLM) to assess risk
- Detects missing important clauses
- Generates negotiation suggestions
- Displays an overall risk index

The project follows a Retrieval-Augmented Generation (RAG) architecture to improve reliability and reduce hallucination.

---

## 🧠 How It Works

### Step 1 — Clause Extraction
The uploaded PDF is processed using `pdfplumber`.  
Text is cleaned and split into clauses using structural patterns (numbered headings, spacing, etc.).

Each clause is classified into types such as:
- Termination Clause
- Confidentiality Clause
- Payment Clause
- Governing Law Clause
- Notice Clause
- Other

---

### Step 2 — Semantic Retrieval (RAG)

Each clause is converted into an embedding using:

- `sentence-transformers`
- Model: `all-MiniLM-L6-v2`

Embeddings are stored and compared using:

- `FAISS` (vector similarity search)

The system retrieves semantically similar precedent clauses based on cosine similarity.

This provides contextual grounding before LLM reasoning.

---

### Step 3 — LLM Risk Analysis

We use:

- Groq API
- LLaMA 3 model

The LLM receives:
- The contract clause
- Retrieved similar precedent clauses

It outputs:
- Risk Level (Low / Medium / High)
- Short explanation

---

### Step 4 — Missing Clause Detection

Based on contract type (Employment, NDA, etc.),  
the system checks whether required clause types are present.

Missing clause types are flagged in the report.

---

### Step 5 — Risk Aggregation

Each clause risk is mapped numerically:

- Low = 1
- Medium = 2
- High = 3

An overall contract risk index is calculated using the average score.

---

### Step 6 — Negotiation Suggestions

After risk analysis, the system generates:

- A short summary of contract risk posture
- Practical negotiation suggestions

---

## 🏗 Architecture

User Upload (PDF)  
↓  
Clause Extraction  
↓  
Embedding Generation (MiniLM)  
↓  
Vector Search (FAISS)  
↓  
LLM Risk Reasoning (Groq + LLaMA 3)  
↓  
Risk Report + Negotiation Suggestions  
↓  
Streamlit Dashboard  

---

## 🧰 Technologies Used

### Backend
- Python
- pdfplumber
- sentence-transformers
- FAISS
- NumPy
- Groq API

### Frontend
- Streamlit
- Plotly

---

## 📂 Project Structure
contract-ai-rag/
│
├── app.py # Streamlit UI
├── pipeline.py # End-to-end workflow
├── clause_extraction.py # Clause parsing & classification
├── retrieval_engine.py # Embeddings + FAISS search
├── llm_engine.py # Groq LLM interface
├── requirements.txt
└── README.md

---

## ⚙️ Installation

### 1. Clone the repository
git clone https://github.com/Dharmateja15/contract-ai-rag

cd contract-ai-rag

### 2. Install dependencies

pip install -r requirements.txt

### 3. Set Groq API key

export GROQ_API_KEY="your_api_key_here"

(Or configure in GitHub Codespaces secrets.)

### 4. Run the application
streamlit run app.py --server.address 0.0.0.0 --server.port 8501

---

## 📌 Limitations

- Clause classification is rule-based (not ML-trained).
- LLM usage depends on API token limits.
- Designed for structured text-based PDFs (not scanned OCR documents).
- Precedent database is small and static (for demo purposes).

---

## 🎯 Purpose

This project demonstrates how Retrieval-Augmented Generation (RAG) can be applied to legal contract analysis.

It focuses on:
- Structured AI pipelines
- Vector similarity search
- Grounded LLM reasoning
- Modular architecture

---

## 👥 Team

Code Alchemists:
Dharma Teja Yelpucherla
Varchaswi Datta Araveti
Narasapuram Surekha
Munagala Venkata Naga Sunayana
Hackathon Project – 2026
