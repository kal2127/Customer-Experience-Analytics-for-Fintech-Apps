import pandas as pd
import psycopg2
from psycopg2 import sql
import os
import re
import numpy as np # Import numpy for better NaN handling

# --- Configuration: IMPORTANT! Update with your specific PostgreSQL details ---
DB_CONFIG = {
    "host": "localhost",
    "database": "bank_reviews", 
    "user": "postgres",     # <-- CHECK THIS: Your username
    "password": "kal",        # <-- CHECK THIS: Your password
    "port": "5432"            # <-- CHECK THIS: Your port (e.g., 5432 or 5434)
}

# --- File Paths and Mappings ---
INPUT_FILE = 'data/analyzed_reviews.csv'
# CRITICAL FIX: Mapping the short codes (CBE, Dashen, BOA) found in the CSV file
# to the full bank names used for the 'banks' table.
BANKS_MAPPING = {
    'CBE': 'Commercial Bank of Ethiopia', 
    'Dashen': 'Dashen Bank',
    'BOA': 'Bank of Abyssinia'
}
BANK_ID_MAP = {} # Stores Bank Name -> Bank ID after insertion

# --- Database Connection and Setup Functions ---

def create_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        # Use autocommit=True for table creation/bank insertion robustness
        conn.autocommit = True
        print("✅ PostgreSQL connection successful.")
        return conn
    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL: {e}")
        print("Please ensure your PostgreSQL server is running and DB_CONFIG is correct.")
        return None

def create_tables(conn):
    """Creates the Banks and Reviews tables if they don't exist."""
    cur = conn.cursor()
    
    # 1. Banks Table Creation
    print("Creating 'banks' table...")
    banks_table_query = """
    CREATE TABLE IF NOT EXISTS banks (
        bank_id SERIAL PRIMARY KEY,
        bank_name VARCHAR(255) UNIQUE NOT NULL,
        app_name VARCHAR(255)
    );
    """
    cur.execute(banks_table_query)

    # 2. Reviews Table Creation
    # Ensures the column type for topic cluster is large enough (BIGINT)
    print("Creating 'reviews' table...")
    reviews_table_query = """
    CREATE TABLE IF NOT EXISTS reviews (
        review_id SERIAL PRIMARY KEY,
        bank_id INTEGER REFERENCES banks (bank_id) ON DELETE CASCADE,
        review_text TEXT NOT NULL,
        rating INTEGER NOT NULL,
        review_date DATE,
        sentiment_label VARCHAR(50),
        sentiment_score NUMERIC(5, 4), -- Target name for compound_score
        topic_cluster BIGINT,          -- CHANGED from INTEGER to BIGINT
        topic_label VARCHAR(255),
        source VARCHAR(100)
    );
    """
    cur.execute(reviews_table_query)
    conn.commit()
    print("✅ Tables created or already exist.")
    cur.close()

def insert_banks(conn):
    """Inserts unique bank records and populates the BANK_ID_MAP."""
    cur = conn.cursor()
    print("Inserting bank records...")
    
    bank_records = []
    # bank_name (value) is used for the database, short_code (key) is used for app_name field
    for short_code, bank_name in BANKS_MAPPING.items():
        bank_records.append((bank_name, short_code)) # (bank_name, app_name)
        
    for bank_name, app_name in bank_records:
        try:
            # Check if bank already exists
            cur.execute("SELECT bank_id FROM banks WHERE bank_name = %s", (bank_name,))
            result = cur.fetchone()
            
            if result is None:
                # Insert the new bank
                insert_query = "INSERT INTO banks (bank_name, app_name) VALUES (%s, %s) RETURNING bank_id;"
                cur.execute(insert_query, (bank_name, app_name))
                bank_id = cur.fetchone()[0]
                print(f"   -> Inserted: {bank_name} (ID: {bank_id})")
            else:
                bank_id = result[0]
                print(f"   -> Found existing: {bank_name} (ID: {bank_id})")
                
            # Map the FULL bank name (the value in the dict) to the ID for lookups
            BANK_ID_MAP[bank_name] = bank_id
            
        except Exception as e:
            print(f"Error inserting bank {bank_name}: {e}")
            conn.rollback() # Rollback if error occurs
            
    conn.commit()
    print("✅ Bank records insertion complete.")
    cur.close()

def load_reviews_data(conn, df):
    """Inserts the processed review data into the 'reviews' table."""
    print("\nStarting review data loading...")
    
    # 1. Map the short code (e.g., 'CBE') in the 'bank' column to the full bank name
    df['bank_name'] = df['bank'].map(BANKS_MAPPING)
    # 2. Map the full bank name to the bank_id 
    df['bank_id'] = df['bank_name'].map(BANK_ID_MAP)
    
    # Ensure all data types match the PostgreSQL schema before insertion
    df['review_date'] = pd.to_datetime(df['date'], errors='coerce').dt.date

    # Prepare data for insertion
    records_to_insert = []
    
    try:
        # Turn off autocommit just for this large insertion block for performance/safety
        conn.autocommit = False 
        
        with conn.cursor() as cur:
            for index, row in df.iterrows():
                # Data cleaning/typing before insertion
                review_text = str(row['review']) if pd.notna(row['review']) else None
                sentiment_score = float(row['sentiment_score']) if pd.notna(row['sentiment_score']) else None
                
                # --- Robust Casting for Ints ---
                
                bank_id = row['bank_id']
                if pd.notna(bank_id):
                    bank_id = int(bank_id)
                else:
                    bank_id = None
                
                rating = row['rating']
                if pd.isna(rating):
                    continue
                rating = int(rating)

                topic_cluster = row['topic_cluster']
                if pd.isna(topic_cluster):
                    topic_cluster = None
                else:
                    topic_cluster = int(topic_cluster)

                # Skip if essential data is missing (this is where the error was before)
                if review_text is None or bank_id is None:
                    continue

                records_to_insert.append((
                    bank_id,
                    review_text,
                    rating,
                    row['review_date'],
                    row['sentiment_label'],
                    sentiment_score, 
                    topic_cluster,
                    row['topic_label'],
                    'Google Play Store'
                ))
                
            # Construct the INSERT query
            insert_query = """
            INSERT INTO reviews (bank_id, review_text, rating, review_date, sentiment_label, 
                                 sentiment_score, topic_cluster, topic_label, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Execute the inserts
            cur.executemany(insert_query, records_to_insert)
            conn.commit() # Commit the transaction after all inserts
            print(f"✅ Successfully inserted {len(records_to_insert)} review records.")
            
    except Exception as e:
        print(f"❌ CRITICAL Error during review insertion. Check data types: {e}")
        conn.rollback() # Rollback the transaction on error
    finally:
        conn.autocommit = True # Reset autocommit 


def main():
    """Main function to orchestrate the ETL process."""
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Analyzed data file not found at {INPUT_FILE}.")
        print("Please ensure Task 2 (topic_modeling.py) was run successfully.")
        return

    # 1. Load data from CSV
    print(f"Loading data from {INPUT_FILE}...")
    try:
        # Define dtypes for integer columns to prevent float inference and NaN issues
        csv_dtypes = {
            'rating': 'Int64',         # Use Pandas' Integer type that handles NaN
            'topic_cluster': 'Int64'   # Use Pandas' Integer type that handles NaN
        }
        
        # Load the CSV
        df = pd.read_csv(INPUT_FILE, dtype=csv_dtypes)
        
        # --- CRITICAL: Standardize column names for DB insertion ---
        # 1. Clean column names (remove non-alphanumeric chars for robust lookup)
        df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col).lower().strip() for col in df.columns]
        
        # 2. Rename 'compound_score' to 'sentiment_score' to match the PostgreSQL schema
        if 'compound_score' in df.columns:
            df.rename(columns={'compound_score': 'sentiment_score'}, inplace=True)
        else:
            print("❌ CRITICAL: 'compound_score' column not found after cleaning. Check CSV header.")
            return

        # Drop columns not needed for DB insertion
        df = df.drop(columns=['unnamed0', 'sentimentnumeric', 'positivescore', 
                              'negativescore', 'neutralscore'], errors='ignore') 
        df.dropna(subset=['review'], inplace=True)
    except Exception as e:
        print(f"Error loading CSV or renaming columns: {e}")
        return

    # 2. Establish DB Connection
    conn = create_connection()
    if conn is None:
        return

    # 3. Create Tables and Insert Bank Data (MUST RUN FIRST)
    create_tables(conn)
    insert_banks(conn)

    # 4. Load Review Data
    if not df.empty and BANK_ID_MAP:
        load_reviews_data(conn, df)
    else:
        print("Skipping review load: DataFrame is empty or Bank IDs were not mapped.")

    # 5. Close Connection
    conn.close()
    print("\nDatabase loading process finished.")


if __name__ == "__main__":
    main()