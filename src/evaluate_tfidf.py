import pandas as pd
import joblib

from eval_utils import evaluate_model

# Load test data
test_df = pd.read_pickle("data/test_df.pkl")
y_true = test_df["category"]

# Load TF-IDF vectorizer and model
tfidf = joblib.load("models/tfidf.joblib")
clf = joblib.load("models/logreg_tfidf.joblib")

# Transform test texts
X_test = tfidf.transform(test_df["text"])

# Predict
y_pred = clf.predict(X_test)

# Labels for confusion matrix
labels = sorted(y_true.unique())

# Evaluate and save metrics
evaluate_model(y_true, y_pred, labels, output_prefix="tfidf")