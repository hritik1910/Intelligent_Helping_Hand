## 1. Project Overview

**What problem does it solve?**

Manufacturing facilities generate large volumes of production data, including machine performance records, operational parameters, production trends, and technical documentation. Analyzing this information manually can be time-consuming, and finding the right information across structured datasets and technical documents can be challenging.

The Manufacturing AI Copilot aims to simplify this process by combining traditional data analytics with Generative AI and Retrieval-Augmented Generation (RAG).

The project connects manufacturing data stored in a MySQL database with an AI-powered interface built using Streamlit. It is designed to help users explore production data, understand operational trends, and retrieve relevant information from technical documents using natural-language questions.

Instead of writing SQL queries for every analysis or manually searching through lengthy PDF documents, users can interact with the application using everyday language.

**Key Capabilities**

Manufacturing Data Analytics: Analyze structured production data using Python, Pandas, and MySQL.

Natural-Language Interaction: Ask questions about manufacturing operations in plain English.

## 2. Business Objectives
The project aims to simplify factory data analysis and technical knowledge retrieval. It is designed to answer questions related to production trends, machine performance, unusual readings, and technical troubleshooting using available datasets and documentation.

## 3. GenAI and RAG Testing
The AI responses are evaluated by comparing analytical results with SQL/Python calculations and checking document-based answers against the original source material. Testing also includes questions with missing information to identify unsupported or inaccurate responses.
