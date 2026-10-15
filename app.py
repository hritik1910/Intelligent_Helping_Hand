
import os

import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from src.assistant import explain_results
from src.rag_search import search_documents

load_dotenv()

st.set_page_config(
    page_title="Manufacturing Intelligence Copilot",
    layout="wide"
)

st.title("Manufacturing Intelligence Copilot")

st.caption(
    "Production analytics, quality insights and technical document search"
)

database_url = os.getenv("DATABASE_URL")

if not database_url:
    st.error("DATABASE_URL is missing from the .env file.")
    st.stop()

engine = create_engine(database_url)

st.warning(
    "This demonstration uses simulated manufacturing data."
)

# Section 1: Business dashboard
st.header("1. Manufacturing Analytics")

REPORTS = {
    "Production line performance": """
        SELECT
            line_id,
            SUM(units_produced) AS units_produced,
            ROUND(
                100.0 * SUM(units_passed_first_pass)
                / NULLIF(SUM(units_produced), 0),
                2
            ) AS fpy_pct
        FROM production_runs
        GROUP BY line_id
        ORDER BY fpy_pct ASC
    """,
    "Defect category analysis": """
        SELECT 'SPI' AS category,
               SUM(spi_defects) AS defect_count
        FROM production_runs

        UNION ALL

        SELECT 'AOI', SUM(aoi_defects)
        FROM production_runs

        UNION ALL

        SELECT 'ICT', SUM(ict_failures)
        FROM production_runs
    """,
    "Quality cost by product": """
        SELECT
            product_id,
            SUM(scrap_cost_inr + rework_cost_inr)
                AS quality_cost_inr
        FROM production_runs
        GROUP BY product_id
        ORDER BY quality_cost_inr DESC
    """
}

report_name = st.selectbox(
    "Choose an analysis",
    list(REPORTS.keys())
)

with engine.connect() as connection:
    results = pd.read_sql_query(
        text(REPORTS[report_name]),
        connection
    )

st.dataframe(results, use_container_width=True)

if not results.empty:
    if report_name == "Production line performance":
        st.bar_chart(
            results,
            x="line_id",
            y="fpy_pct"
        )

    elif report_name == "Defect category analysis":
        st.bar_chart(
            results,
            x="category",
            y="defect_count"
        )

    else:
        st.bar_chart(
            results,
            x="product_id",
            y="quality_cost_inr"
        )

# Section 2: AI explanation
st.header("2. Ask the Data Assistant")

question = st.text_input(
    "Ask a question about the selected analysis"
)

if st.button("Explain the results"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Analysing the results..."):
            answer = explain_results(
                question,
                results.to_dict(orient="records")
            )

        st.write(answer)
        st.caption(
            "The explanation is based on the displayed SQL results."
        )

# Section 3: Document retrieval
st.header("3. Search Technical Procedures")

document_question = st.text_input(
    "Ask a question about the sample SOP documents"
)

if st.button("Search procedures"):
    if not document_question.strip():
        st.warning("Enter a question first.")
    else:
        passages = search_documents(document_question)

        if not passages:
            st.info("No documents were found.")
        else:
            for passage in passages:
                st.markdown(
                    f"**Source:** {passage['source']} "
                    f"— Section {passage['section']}"
                )
                st.write(passage["text"])

engine.dispose()
