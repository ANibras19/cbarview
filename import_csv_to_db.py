import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load .env file for DATABASE_URL
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env file")

engine = create_engine(DATABASE_URL)

# Mapping CSV filenames to table names
csv_files = {
    "Checklist.csv": "checklist",
    "CMF Domains.csv": "cmf_domains",
    "CMF Goals.csv": "cmf_goals",
    "CMF Strands.csv": "cmf_strands",
    "Reflist.csv": "reflist"
}

# Path to your CSV folder
CSV_FOLDER = r"C:\Users\Ahmed\OneDrive\Desktop\cbar-view\frontend\public"

with engine.begin() as conn:
    for file_name, table_name in csv_files.items():
        file_path = os.path.join(CSV_FOLDER, file_name)
        if os.path.exists(file_path):
            print(f"Importing {file_name} → {table_name}...")
            df = pd.read_csv(file_path)
            # Write to PostgreSQL (replace table if exists)
            df.to_sql(table_name, con=conn, if_exists="replace", index=False)
        else:
            print(f"⚠ File not found: {file_name}")

print("✅ All CSVs imported successfully!")
