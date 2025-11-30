import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re

# --- NLTK VADER Lexicon Check and Download ---
# This block ensures the VADER lexicon is available before the analyzer is initialized.
try:
    # Check if the lexicon data file is already downloaded
    nltk.data.find('sentiment/vader_lexicon.zip')
    print("VADER lexicon already downloaded.")
except nltk.downloader.DownloadError:
    # If not found, download it
    print("VADER lexicon not found. Downloading...")
    nltk.download('vader_lexicon')
except LookupError:
    # Handle the case where the data folder structure is missing
    print("NLTK data path issue. Attempting download...")
    nltk.download('vader_lexicon')


# --- Configuration ---
INPUT_FILE = 'data/play_store_reviews_clean.csv'
OUTPUT_FILE = 'data/analyzed_reviews_sentiment_only.csv'

# --- Main Logic ---

print(f"\nLoading cleaned data from: {INPUT_FILE}")

# Load the cleaned data from Task 1
try:
    df = pd.read_csv(INPUT_FILE)
    # Ensure review column is treated as string and drop empty ones
    df['review'] = df['review'].astype(str).str.strip()
    df.dropna(subset=['review'], inplace=True) 
    df = df[df['review'] != '']
    print(f"Data loaded successfully. Total reviews for analysis: {len(df)}")
except FileNotFoundError:
    print(f"Error: Cleaned data file not found at {INPUT_FILE}. Please run preprocessing first.")
    exit()

# 1. Initialize the VADER Sentiment Analyzer
print("\nInitializing VADER Sentiment Analyzer...")
analyzer = SentimentIntensityAnalyzer()

# 2. Define Function to Apply Sentiment Scoring
def get_vader_sentiment(text):
    """Calculates VADER scores and determines the sentiment label based on compound score."""
    # VADER returns a dictionary with 'neg', 'neu', 'pos', and 'compound' scores
    vs = analyzer.polarity_scores(text)
    
    # Determine the label based on the standard VADER compound threshold:
    # Compound score >= 0.05 is Positive
    # Compound score <= -0.05 is Negative
    # Otherwise, it is Neutral
    
    if vs['compound'] >= 0.05:
        label = 'POSITIVE'
    elif vs['compound'] <= -0.05:
        label = 'NEGATIVE'
    else:
        label = 'NEUTRAL'
        
    return pd.Series([label, vs['compound'], vs['pos'], vs['neg'], vs['neu']])

# 3. Apply the Function and Add Columns
print("Applying VADER analysis to reviews...")
# Apply the function and expand the returned Series into new columns
df[['sentiment_label', 'compound_score', 'positive_score', 'negative_score', 'neutral_score']] = \
    df['review'].apply(get_vader_sentiment)

# 4. Create a Binary/Ternary Numeric Score (Useful for comparisons/modeling)
# Use 1 for Positive, 0 for Neutral, -1 for Negative
df['sentiment_numeric'] = df['sentiment_label'].map({'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0})


# 5. Display Summary and Save
print("\nVADER Sentiment Analysis Complete.")
print("--- Sentiment Distribution ---")
print(df['sentiment_label'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

# Save the DataFrame with the new sentiment columns
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Data with VADER sentiment scores saved to {OUTPUT_FILE}")