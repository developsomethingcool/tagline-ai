import os
import streamlit as st
import joblib
import sqlite3
from sqlalchemy import create_engine
import pandas as pd

from dotenv import load_dotenv


# Load environment
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    st.error("❌ DATABASE_URL not set in .env")

# Cached resources
@st.cache_resource
def load_model():
    try:
        tfidf = joblib.load('models/tfidf.joblib')
        clf   = joblib.load('models/logreg_tfidf.joblib')

    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None, None
    
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None
    
    return tfidf, clf

@st.cache_resource
def get_db_engine():
    return create_engine(DATABASE_URL)

# Initialization
tfidf, clf = load_model()
engine = get_db_engine()

st.title("News Topic Classifier")

# User input and prediction
user_text = st.text_area("Paste a news headline or short paragraph:", height=300)
    
if st.button("Predict"):
    if not user_text.strip():
        st.warning("Please enter some text before predicting.")
    elif tfidf is None or clf is None:
        st.error("Models are not loaded, cannot predict.")
    else:
        X    = tfidf.transform([user_text])
        pred = clf.predict(X)[0]
        st.success(f"**Predicted category:** {pred}")

# Dataset overview

st.header("Dataset Overview")
if st.checkbox("Show class distribution"):
    try:
        dist = pd.read_sql(
            "SELECT category, COUNT(*) AS n "
            "FROM articles "
            "GROUP BY category "
            "ORDER BY n DESC;",
            con=engine
        )
        st.bar_chart(dist.set_index("category"))
    except Exception as e:
        st.error(f"Failed to load data from Postgres: {e}")

# Model Performance
st.header("Model Performance")
st.image("reports/cm_tfidf.png", caption="TF-IDF Confusion Matrix")