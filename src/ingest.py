import pandas as pd 
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os


# Load DB connection string from .env
load_dotenv()
db_url = os.getenv("DATABASE_URL")
print("DB URL:", db_url)

engine = create_engine(db_url)

# Test the connection
with engine.connect() as connection:
    result = connection.execute(text("SELECT version();"))
    print(result.fetchone())


# Read the information from database
df = pd.read_json('data/News_Category_Dataset_v3.json', lines=True)

df['text'] = df['headline'] + ' ' + df['short_description']

# Keep only category, text and date
df = df[['category', 'text', 'date']].dropna().drop_duplicates()

df.to_sql("articles", engine, if_exists="replace", index_label="id")
