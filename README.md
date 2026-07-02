# AI-Powered-HR-Analytics
An end-to-end AI-powered HR Analytics solution that combines SQL, Python, Machine Learning, Power BI, Streamlit, and Retrieval-Augmented Generation (RAG) to predict employee attrition, visualize workforce insights, and enable natural language HR decision support.
========================================================
  AI-Powered Employee Attrition Prediction
  HR Analytics Assistant — README
========================================================

PROJECT OVERVIEW
----------------
This project predicts employee attrition using Machine Learning
and allows HR users to ask natural language questions about
employee data through an AI-powered chatbot.

The application retrieves relevant employee records using FAISS
vector search and Sentence Transformers before generating
professional HR insights with OpenAI GPT.


FEATURES
--------
- Employee Attrition Prediction (Random Forest Classifier)
- HR Analytics Dashboard (Streamlit)
- AI HR Assistant (OpenAI GPT + RAG)
- SQL Server Database Integration
- FAISS Vector Search
- Sentence Transformers Embeddings
- Feature Importance Analysis


TECHNOLOGIES USED
-----------------
- Python
- Streamlit
- SQL Server / SQLAlchemy
- Pandas / NumPy
- Scikit-Learn (Random Forest)
- Sentence Transformers
- FAISS
- OpenAI API
- Matplotlib


PROJECT STRUCTURE
-----------------
HR-Analytics/
│
├── chatbot.py             # Main Streamlit app + AI chatbot
├── EDA.py                 # Exploratory Data Analysis
├── SQLQuery1.sql          # Database schema / queries
├── feature_importance.csv # Output from Random Forest model
├── requirements.txt       # Python dependencies
├── README.py              # This file
├── images/
│   ├── dashboard.png
│   └── chatbot.png
└── models/                # Saved ML model files


INSTALLATION
------------

1. Clone the repository:

   git clone https://github.com/yourusername/HR-Analytics.git
   cd HR-Analytics

2. Install required packages:

   pip install -r requirements.txt

3. Configure environment variable:
   Create a .env file in the project root:

   OPENAI_API_KEY=YOUR_API_KEY

4. Run the application:

   streamlit run chatbot.py


MACHINE LEARNING MODEL
----------------------
Algorithm : Random Forest Classifier
Target    : Employee Attrition (Yes / No)

Evaluation Metrics:
  - Accuracy Score
  - Classification Report (Precision, Recall, F1)
  - Confusion Matrix
  - Feature Importance

Top Attrition Predictors:
  1. Overtime           (Importance: 0.21)
  2. Monthly Income     (Importance: 0.18)
  3. Age                (Importance: 0.15)
  4. Years at Company   (Importance: 0.13)
  5. Job Role           (Importance: 0.09)


AI HR ASSISTANT — EXAMPLE QUESTIONS
-------------------------------------
The chatbot supports natural language HR queries such as:

  Q: "Which department has the highest attrition?"
  Q: "Show employee details for high-risk employees."
  Q: "Recommend retention strategies."
  Q: "Which employees are at risk of leaving?"
  Q: "Why are employees in Sales leaving?"
  Q: "How can HR reduce attrition?"


RAG PIPELINE (Retrieval-Augmented Generation)
----------------------------------------------
1. Employee records are embedded using Sentence Transformers
2. Embeddings are indexed with FAISS for fast similarity search
3. Relevant employee records are retrieved for each HR query
4. OpenAI GPT generates professional HR insights from context


BUSINESS INSIGHTS SUMMARY
--------------------------
- Overall attrition rate         : 16.1%
- Highest attrition department   : Sales (20%)
- Overtime employees attrition   : 31% vs 10% (no overtime)
- Most resignations occur in     : First 0–2 years
- Age group most likely to leave : 20–30 years


BEFORE UPLOADING TO GITHUB
---------------------------
  [x] Remove OpenAI API key from all source files
  [x] Add .env to .gitignore
  [x] Include requirements.txt
  [x] Add screenshots to /images folder
  [x] Verify SQL connection strings are not hardcoded


FUTURE IMPROVEMENTS
-------------------
- Login Authentication
- Employee Risk Score Dashboard
- SHAP Explainable AI Integration
- Chat History & Memory
- Voice Assistant
- PDF Report Generation
- Power BI Dashboard
- Docker & Cloud Deployment


AUTHOR
------
Name    : Your Name
LinkedIn: https://linkedin.com/in/yourprofile
GitHub  : https://github.com/yourusername


LICENSE
-------
MIT License — free to use and modify with attribution.
"""

# ── Quick-start helper ────────────────────────────────────────────────────────

PROJECT_INFO = {
    "name": "AI-Powered Employee Attrition Prediction & HR Analytics Assistant",
    "version": "1.0.0",
    "python_version": ">=3.8",
    "main_file": "chatbot.py",
    "dataset": "IBM HR Analytics Employee Attrition Dataset",
    "model": "Random Forest Classifier",
    "ui": "Streamlit",
}

DEPENDENCIES = [
    "streamlit",
    "pandas",
    "numpy",
    "scikit-learn",
    "sqlalchemy",
    "pyodbc",
    "sentence-transformers",
    "faiss-cpu",
    "openai",
    "matplotlib",
    "python-dotenv",
]

EXAMPLE_QUESTIONS = [
    "Which department has the highest attrition?",
    "Show me high-risk employees.",
    "Recommend retention strategies.",
    "What is the overall attrition rate?",
    "Which job roles have the most turnover?",
    "How does overtime affect attrition?",
    "What is the average income of employees who left?",
    "Which age group leaves the most?",
]


def print_info():
    """Print project information to the console."""
    print("\n" + "=" * 60)
    print(f"  {PROJECT_INFO['name']}")
    print("=" * 60)
    print(f"  Version : {PROJECT_INFO['version']}")
    print(f"  Python  : {PROJECT_INFO['python_version']}")
    print(f"  Model   : {PROJECT_INFO['model']}")
    print(f"  UI      : {PROJECT_INFO['ui']}")
    print("=" * 60)
    print("\n  Run:  streamlit run chatbot.py\n")


if __name__ == "__main__":
    print_info()
