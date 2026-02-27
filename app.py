import os
import tempfile

import streamlit as st
import plotly.express as px

from pipeline import run_analysis_pipeline
from llm_engine import client, MODEL_NAME


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Open Contract Validator",
    page_icon="⚖️",
    layout="wide",
)

# ==========================
# CUSTOM STYLING
# ==========================

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.block-card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}
.risk-low { color: #22c55e; font-weight: 700; }
.risk-medium { color: #facc15; font-weight: 700; }
.risk-high { color: #ef4444; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:
    st.title("⚖️ OCV Dashboard")
    st.markdown("Secure | Explainable | AI-Powered")
    st.markdown("---")
    st.info("Clause extraction + Vector Retrieval + LLM Risk Analysis")

# ==========================
# HEADER
# ==========================

st.markdown("<h1 style='color:#7CFCB5;'>⚖️ Open Contract Validator</h1>", unsafe_allow_html=True)
st.write("AI-powered legal contract risk analysis & compliance engine.")

# ==========================
# INPUT SECTION
# ==========================

col_left, col_right = st.columns([2, 1])

with col_left:
    uploaded_file = st.file_uploader("Upload Contract PDF", type=["pdf"])

with col_right:
    contract_type = st.selectbox(
        "Select Contract Type",
        ["Employment", "NDA", "Service", "Vendor", "Lease"],
    )

run_button = st.button("🚀 Run Full Analysis")

# ==========================
# NEGOTIATION FUNCTION
# ==========================

def generate_negotiation_tips(contract_type: str, report: dict):
    try:
        clauses_summary = "\n".join(
            [f"- {c['title']} ({c['risk_level']})" for c in report.get("clauses", [])]
        )
        missing = report.get("missing_clauses", [])

        prompt = f"""
You are a contract lawyer.

Contract type: {contract_type}

Clause risks:
{clauses_summary}

Missing clauses:
{missing}

Provide:
1. 3 bullet summary of overall risk posture.
2. 3–5 practical negotiation improvements.
Keep concise.
"""

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are concise and practical."},
                {"role": "user", "content": prompt},
            ],
            model=MODEL_NAME,
            max_tokens=300,
        )

        return response.choices[0].message.content.strip()

    except:
        return "Negotiation tips unavailable due to API limits."

# ==========================
# TRANSLATION FUNCTION
# ==========================

def translate_text(text, target_language):
    try:
        prompt = f"Translate the following text into {target_language}:\n\n{text}"

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt}
            ],
            model=MODEL_NAME,
            max_tokens=400
        )

        return response.choices[0].message.content.strip()

    except:
        return "Translation unavailable due to API limits."

# ==========================
# MAIN LOGIC
# ==========================

if run_button:
    if uploaded_file is None:
        st.error("Please upload a PDF.")
    else:
        with st.spinner("Analyzing contract..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                report = run_analysis_pipeline(tmp_path, contract_type)
            finally:
                os.remove(tmp_path)

        st.markdown("---")

        clauses = report.get("clauses", [])
        missing = report.get("missing_clauses", [])

        # ==========================
        # OVERALL RISK INDEX
        # ==========================

        if clauses:
            risk_map = {"Low": 1, "Medium": 2, "High": 3}
            scores = [risk_map.get(c["risk_level"], 2) for c in clauses]
            overall_score = round(sum(scores) / len(scores), 2)

            st.subheader("🚨 Overall Risk Index")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Risk Score (1=Low, 3=High)", overall_score)
            with col2:
                st.progress(overall_score / 3)

        # ==========================
        # MISSING CLAUSES
        # ==========================

        st.subheader("📌 Missing Clause Types")

        if missing:
            for m in missing:
                st.warning(m)
        else:
            st.success("All required clauses found.")

        # ==========================
        # CLAUSE CARDS
        # ==========================

        st.subheader("📊 Clause Risk Assessment")

        for c in clauses:
            risk = c["risk_level"]
            css_class = (
                "risk-low" if risk == "Low"
                else "risk-medium" if risk == "Medium"
                else "risk-high"
            )

            st.markdown(f"""
            <div class="block-card">
                <h4>{c['title']}</h4>
                <p>Risk Level: <span class="{css_class}">{risk}</span></p>
                <p>{c['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)

        # ==========================
        # RISK DISTRIBUTION
        # ==========================

        if clauses:
            risk_counts = {}
            for c in clauses:
                risk_counts[c["risk_level"]] = risk_counts.get(c["risk_level"], 0) + 1

            fig = px.bar(
                x=list(risk_counts.keys()),
                y=list(risk_counts.values()),
                labels={"x": "Risk Level", "y": "Count"},
                title="Risk Distribution",
                color=list(risk_counts.keys()),
            )

            st.plotly_chart(fig, use_container_width=True)

        # ==========================
        # NEGOTIATION TIPS
        # ==========================

        st.subheader("🤝 Negotiation Tips & Summary")

        tips = generate_negotiation_tips(contract_type, report)
        st.write(tips)

        # ==========================
        # LANGUAGE CONVERSION
        # ==========================

        st.subheader("🌍 Convert Summary Language")

        language = st.selectbox(
            "Select Language",
            ["English", "Hindi", "Telugu", "Spanish"]
        )

        if language != "English":
            translated = translate_text(tips, language)
            st.write(translated)