

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Read database connection details from .env
load_dotenv()

database_url = os.getenv("DATABASE_URL")

if not database_url:
    raise ValueError(
        "DATABASE_URL is missing. Check your .env file."
    )

# Connect Python to MySQL
engine = create_engine(database_url)

# Read the cleaned CSV file
csv_path = "data/raw/production_runs.csv"
df = pd.read_csv(csv_path)

try:
    # Create the table and load the CSV records into MySQL.
    # Warning: "replace" deletes an existing table before recreating it.
    df.to_sql(
        name="production_runs",
        con=engine,
        if_exists="replace",
        index=False
    )

    print("CSV loaded successfully into MySQL!")
    print("Total rows loaded:", len(df))

finally:
    engine.dispose()
