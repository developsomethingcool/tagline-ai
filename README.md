# tagline-ai
Every story has a label.

A complete end-to-end machine learning pipeline for classifying news articles into categories using both traditional machine learning (TF-IDF + Logistic Regression) and deep learning (DistilBERT) approaches.

## Project Overview

This project implements a complete text classification pipeline that:
1. Ingests news article data into a SQL database
2. Preprocesses and cleans the text data
3. Trains both traditional ML and transformer-based models
4. Evaluates model performance with detailed metrics
5. Provides a user-friendly Streamlit web application for making predictions

## Setup Instructions

### Prerequisites

- Python 3.8+
- PostgreSQL database
- pip or conda for package management

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/developsomethingcool/tagline-ai.git
cd tagline-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the PostgreSQL database and create a `.env` file with your database connection string:
```
DATABASE_URL=postgresql://username:password@localhost:5432/news_categories
```

5. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset) and place it in the `data/` directory.

## Pipeline Workflow

### 1. Data Ingestion

The `ingest.py` script takes the raw JSON data from Kaggle, processes it, and loads it into the PostgreSQL database:

```bash
python src/ingest.py
```

This script:
- Connects to the PostgreSQL database using the connection string from the `.env` file
- Reads the News Category Dataset JSON file
- Combines headline and short description into a single text field
- Creates an 'articles' table in the database and loads the processed data

### 2. Model Training

#### TF-IDF + Logistic Regression Model

```bash
python src/train_tfidf.py
```

This script:
- Fetches data from the PostgreSQL database
- Splits data into train, validation, and test sets
- Transforms text using TF-IDF vectorization
- Trains a Logistic Regression classifier
- Saves the model and vectorizer to the models directory

#### DistilBERT Model

```bash
python src/train_distilBERT.py
```

This script:
- Uses the same train/val/test splits from the TF-IDF training
- Fine-tunes a pre-trained DistilBERT model on the news categories
- Saves the model to the models directory

### 3. Model Evaluation

#### Evaluate TF-IDF + LogReg Model

```bash
python src/evaluate_tfidf.py
```

#### Evaluate DistilBERT Model

```bash
python src/evaluate_bert.py
```

Both evaluation scripts:
- Load the trained models
- Make predictions on the test set
- Generate and save confusion matrices and classification reports

### 4. Web Application

Run the Streamlit application to interact with the model:

```bash
streamlit run src/app.py
```

The app provides:
- A text input area for entering news headlines or articles
- Prediction of the news category
- Dataset overview with class distribution
- Model evaluation metrics visualization

The project includes two models:

1. **TF-IDF + Logistic Regression**:
   - Strengths: Fast training and inference, interpretable
   - Performance: Check `reports/tfidf_metrics.json` for detailed metrics

2. **DistilBERT**:
   - Strengths: Better semantic understanding, higher accuracy
   - Performance: Check `reports/bert_metrics.json` for detailed metrics

## Customization

- To add new categories or support different classification tasks, modify the data preprocessing steps in `ingest.py`
- For different text models, update the training scripts and Streamlit app accordingly

## Troubleshooting

- **Database Connection Issues**: Verify that PostgreSQL is running and the connection string in `.env` is correct
- **Model Loading Errors**: Ensure all model files are in the correct locations
- **CUDA/MPS Errors**: The training script has specific adaptations for Apple Silicon; adjust based on your hardware

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Kaggle News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- HuggingFace Transformers library
- Scikit-learn
- Streamlit
