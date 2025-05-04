import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from dotenv import load_dotenv

load_dotenv()
db_url = os.getenv("DATABASE_URL")

engine = create_engine(db_url)

df = pd.read_sql("SELECT category, text FROM articles", engine)

# train+val and test
train_val_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["category"], random_state=42
)

# train and val
train_df, val_df = train_test_split(
    train_val_df, test_size=0.1, stratify=train_val_df["category"], random_state=42
)

# Save train/test df
train_df.to_pickle("data/train_df.pkl")
val_df.to_pickle("data/val_df.pkl")
test_df.to_pickle("data/test_df.pkl")


tfidf = TfidfVectorizer(
    stop_words="english", max_features=40_000, ngram_range=(1, 2)
)

X_train = tfidf.fit_transform(train_df.text)
X_test = tfidf.transform(test_df.text)
y_train, y_test = train_df.category, test_df.category

clf = LogisticRegression(max_iter=300, n_jobs=-1)
clf.fit(X_train, y_train)

joblib.dump(tfidf, "models/tfidf.joblib")
joblib.dump(clf, "models/logreg_tfidf.joblib")
print("✅ TF‑IDF + LogReg model trained and saved.")