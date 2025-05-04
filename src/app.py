import streamlit as st, joblib, pandas as pd

@st.cache_resource
def load_model():
    tfidf = joblib.load('models/tfidf.joblib')
    clf   = joblib.load('models/logreg_tfidf.joblib')
    return tfidf, clf
tfidf, clf = load_model()

st.title("News Topic Classifier 📰")
user_text = st.text_area("Paste a news headline or short paragraph:")

if st.button("Predict"):
    X = tfidf.transform([user_text])
    pred = clf.predict(X)[0]
    st.success(f"**Predicted category:** {pred}")

st.header("Dataset Overview")
if st.checkbox("Show class distribution"):
    import sqlite3, pandas as pd
    con = sqlite3.connect('data/news.db')
    dist = pd.read_sql("SELECT category, COUNT(*) n FROM articles GROUP BY category ORDER BY n DESC;", con)
    st.bar_chart(dist.set_index('category'))
    con.close()

st.header("Model Performance")
st.image('reports/cm.png', caption="Confusion matrix on hold‑out set")
