import torch
import os
import multiprocessing
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Fix for multiprocessing on macOS with MPS
# This prevents the "RuntimeError: An attempt has been made to start a new process" error
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure proper entry point for multiprocessing
if __name__ == "__main__":
    # Load train, validation, and test datasets
    train_df = pd.read_pickle("data/train_df.pkl")
    val_df = pd.read_pickle("data/val_df.pkl")
    test_df = pd.read_pickle("data/test_df.pkl")
    
    # Define labels mapping explicitly
    labels = sorted(train_df.category.unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # Map labels to integers for Hugging Face
    for df in [train_df, val_df, test_df]:
        df["labels"] = df["category"].map(label2id)
        df.drop(columns=['category'], inplace=True)
    
    # Convert datasets to Hugging Face format
    ds_train = Dataset.from_pandas(train_df.reset_index(drop=True))
    ds_val = Dataset.from_pandas(val_df.reset_index(drop=True))
    ds_test = Dataset.from_pandas(test_df.reset_index(drop=True))
    
    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    
    ds_train = ds_train.map(tokenize, batched=True)
    ds_val = ds_val.map(tokenize, batched=True)
    ds_test = ds_test.map(tokenize, batched=True)
    
    # Set dataset format explicitly
    columns_to_keep = ["input_ids", "attention_mask", "labels"]
    ds_train.set_format(type="torch", columns=columns_to_keep)
    ds_val.set_format(type="torch", columns=columns_to_keep)
    ds_test.set_format(type="torch", columns=columns_to_keep)
    
    # Device selection - explicit Apple Silicon MPS check
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device: {device}")
    else:
        device = torch.device("cpu")
        print(f"MPS not available, using: {device}")
    
    # Load model and move to MPS
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label
    ).to(device)
    
    # Metrics for evaluation
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro')
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro
        }
    
    # Training arguments optimized for Apple Silicon MPS
    args = TrainingArguments(
        output_dir="checkpoints/bert_mps",
        num_train_epochs=3,
        per_device_train_batch_size=8,    # Reduced batch size for MPS memory constraints
        per_device_eval_batch_size=16,    # Reduced batch size for MPS memory constraints
        logging_steps=100,                # More frequent logging
        save_steps=500,                   # Save model checkpoint every 500 steps
        fp16=False,                       # MPS does not support fp16 well
        report_to="none",                 # Remove external logging for simplicity
        dataloader_num_workers=0,         # Disable multiprocessing for MPS compatibility
        dataloader_pin_memory=False,      # Disable pin memory as it's not supported on MPS
        disable_tqdm=False                # Keep progress bars
    )
    
    # Create Trainer with validation dataset and MPS-specific settings
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        # Removing tokenizer parameter as it's deprecated and causes warnings
        # Use processing_class instead if you need it in newer transformers versions
    )
    
    # Train
    trainer.train()
    
    # Evaluate on validation set explicitly after training
    val_results = trainer.evaluate(ds_val)
    print("Validation results:", val_results)
    
    # Optionally evaluate on test set
    test_results = trainer.evaluate(ds_test)
    print("Test results:", test_results)
    
    # Save the trained model and tokenizer
    model.save_pretrained("models/bert_news/")
    tokenizer.save_pretrained("models/bert_news/")