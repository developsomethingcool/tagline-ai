import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from eval_utils import evaluate_model

# Load test and training data for label mapping
test_df = pd.read_pickle("data/test_df.pkl")
train_df = pd.read_pickle("data/train_df.pkl")

# Load model and tokenizer (saved directory contains config with id2label)
model = AutoModelForSequenceClassification.from_pretrained("models/bert_news/")
tokenizer = AutoTokenizer.from_pretrained("models/bert_news/")

# Prepare label list
labels = sorted(train_df.category.unique())

# Map string labels to integer IDs using model.config.label2id
test_df["labels"] = test_df["category"].map(model.config.label2id)
# Keep only text and labels columns
test_df = test_df[["text", "labels"]]

# Convert to Hugging Face Dataset
ds_test = Dataset.from_pandas(test_df.reset_index(drop=True))

# Tokenize function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

ds_test = ds_test.map(tokenize, batched=True)

# Set format for PyTorch
columns = ["input_ids", "attention_mask", "labels"]
ds_test.set_format(type="torch", columns=columns)

# Create a Trainer for prediction
trainer = Trainer(model=model, tokenizer=tokenizer)

# Get predictions
predictions = trainer.predict(ds_test)
logits = predictions.predictions

# Compute predicted class indices
y_pred = np.argmax(logits, axis=-1)
# True labels
y_true = predictions.label_ids

# Evaluate and save metrics
evaluate_model(y_true, y_pred, labels, output_prefix="bert")