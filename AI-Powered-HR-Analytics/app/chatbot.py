import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

engine = create_engine(
    "mssql+pyodbc://LAPTOP-BADL4BRK\\SQLEXPRESS/hr_analytics?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server&timeout=30"
)

query = "SELECT * FROM dbo.emp_attrition"

fallback_data = [
    {
        "EmployeeNumber": 1,
        "Department": "Sales",
        "JobRole": "Sales Executive",
        "Attrition": "No",
        "MonthlyIncome": 5000,
        "Age": 30,
    },
    {
        "EmployeeNumber": 2,
        "Department": "Research & Development",
        "JobRole": "Research Scientist",
        "Attrition": "Yes",
        "MonthlyIncome": 7000,
        "Age": 28,
    },
    {
        "EmployeeNumber": 3,
        "Department": "Human Resources",
        "JobRole": "Human Resources",
        "Attrition": "No",
        "MonthlyIncome": 4500,
        "Age": 35,
    },
]

try:
    df = pd.read_sql(query, engine)
    if df.empty:
        raise ValueError("Query returned no rows")
    db_error = None
except (SQLAlchemyError, ValueError, Exception) as exc:
    df = pd.DataFrame(fallback_data)
    db_error = exc

# Create Knowledge Base
documents = []

for _, row in df.iterrows():
    text = f"""
    EmployeeID: {row['EmployeeNumber']}
    Department: {row['Department']}
    JobRole: {row['JobRole']}
    Attrition: {row['Attrition']}
    MonthlyIncome: {row['MonthlyIncome']}
    Age: {row['Age']}
    """
    documents.append(text.strip())

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(documents)

# Create FAISS Index
import faiss
import numpy as np

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(
    np.array(embeddings).astype("float32")
)

# Retrieval Function
def retrieve(query):

    query_embedding = model.encode([query])

    D, I = index.search(
        np.array(query_embedding).astype("float32"),
        k=5
    )

    return [documents[i] for i in I[0]]

# OpenAI Integration

import os
from openai import OpenAI

# Option 1: set OPENAI_API_KEY in your environment.
# Option 2: replace the placeholder below with your real key.
api_key = os.getenv("OPENAI_API_KEY") or "sk-proj-xxxxxxxx"


if api_key == "sk-proj-xxxxxxxx"
    raise ValueError(
        "OpenAI API key not found. Set OPENAI_API_KEY or replace the placeholder string with your real API key."
    )

client = OpenAI(api_key=api_key)

def ask_ai(question):
    context = retrieve(question)

    prompt = f"""
    Context:
    {' '.join(context)}

    Question:
    {question}

    Give professional HR insights.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    return response.choices[0].message.content

# Streamlit Chatbot
import streamlit as st

st.title("AI HR Analytics Assistant")

question = st.text_input(
    "Ask HR Question"
)

if question:
    st.write("Searching employee database...")
    answer = ask_ai(question)
    st.success(answer)

if st.button("Generate Insight"):
    if not question:
        st.error("Please enter a question first.")
    else:
        answer = ask_ai(question)
        st.write(answer)

