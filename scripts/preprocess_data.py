import pandas as pd
import numpy as np
import re # We'll use this for text cleaning

# --- File Paths ---
INPUT_FILE = 'data/raw_google_play_reviews.csv'
OUTPUT_FILE = 'data/play_store_reviews_clean.csv'

print(f"Loading raw data from: {INPUT_FILE}")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: Raw data file not found at {INPUT_FILE}. Please run the scraping script first.")
    exit()

# 1. Rename Columns to match the project requirements (if they don't already)
# Note: The scraping script already handled this, but we'll include a mapping check
# just in case, ensuring we have: 'review', 'rating', 'date', 'bank', 'source'
df = df.rename(columns={'review': 'review_text'})
# Ensure all required columns exist (they should, based on the scraping script)
required_columns = ['review_text', 'rating', 'date', 'bank', 'source']
if not all(col in df.columns for col in required_columns):
    print("Error: DataFrame is missing required columns. Check scraping script output.")
    exit()
    
# 2. Convert Data Types
# Convert 'date' to datetime objects and 'rating' to integer
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['rating'] = df['rating'].astype('Int64') # Using 'Int64' to allow for NaN values

print(f"Initial row count: {len(df)}")

# 3. Handle Duplicates
df.drop_duplicates(subset=['review_text', 'bank', 'rating'], keep='first', inplace=True)
print(f"Row count after removing duplicates: {len(df)}")

#==============================Part 2 =========================================#
# 4. Handle Missing Data (Missing Data < 5% KPI)

# Drop rows where the critical 'review_text' or 'rating' is missing
df.dropna(subset=['review_text', 'rating', 'date'], inplace=True)

# Check missing data percentage
missing_count = df.isnull().sum()
total_rows = len(df)
missing_percentage = (missing_count.sum() / (total_rows * len(required_columns))) * 100

print(f"Total rows after dropping NaNs: {total_rows}")
print(f"Total missing data across required columns is {missing_percentage:.2f}%")
if missing_percentage < 5:
    print("✅ Missing data KPI met.")
else:
    print("⚠️ Warning: Missing data exceeds 5%.")


# 5. Text Cleaning (Essential for NLP)

def clean_text(text):
    """
    Cleans the review text by removing non-alphanumeric characters (except spaces)
    and converting to lowercase.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove mentions/hashtags/links (optional, but good practice)
    text = re.sub(r'@\w+|#\w+|http\S+|www\S+', '', text)
    # Remove anything that's not a word character or space (keeps letters, numbers, underscore)
    # or specifically remove punctuation (more aggressive)
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra white space
    return text

# Apply the cleaning function to the review text column
df['review_text_cleaned'] = df['review_text'].apply(clean_text)

# Drop the original 'review_text' column to save space and use the cleaned one
df.drop('review_text', axis=1, inplace=True)
df.rename(columns={'review_text_cleaned': 'review'}, inplace=True)

# Reorder columns for final output
df = df[['bank', 'source', 'date', 'rating', 'review']]

#=================================Part 3===================================#
# 6. Final Save
print(f"\nFinal cleaned dataset size: {len(df)} rows.")

# Save the final, cleaned DataFrame to the 'data' folder
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Cleaned data saved successfully to {OUTPUT_FILE}")