import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
import json # New import for saving the topic keywords

# --- NLTK Preparation (Run this part once if resources are missing) ---
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError: # Corrected exception handling
    print("NLTK resources missing. Downloading...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("NLTK resources successfully downloaded.")
    
# --- Configuration ---
INPUT_FILE = 'data/analyzed_reviews_sentiment_only.csv'
OUTPUT_FILE = 'data/analyzed_reviews.csv'
TOPIC_KEYWORDS_FILE = 'data/topic_keywords_k10.json' # NEW OUTPUT FILE
NUM_TOPICS = 10  # K value used for K-Means Clustering

# --- Advanced Text Preprocessing Function ---

# Initialize Lemmatizer and Stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Add bank names as custom stop words since they don't help identify a theme
custom_stop_words = ['cbe', 'boa', 'dashen', 'bank', 'mobile', 'app', 'birr', 'next', 'amole']
stop_words.update(custom_stop_words)


def preprocess_for_topics(text):
    """Tokenizes, removes stop words, and lemmatizes the text."""
    if pd.isna(text):
        return ""
    
    tokens = re.findall(r'\b\w+\b', str(text).lower())
    processed_tokens = []
    for word in tokens:
        if word not in stop_words and len(word) > 2:
            processed_tokens.append(lemmatizer.lemmatize(word))
            
    return " ".join(processed_tokens)

# --- Main Logic ---

print(f"Loading sentiment-analyzed data from: {INPUT_FILE}")

# 1. Load Data
try:
    df = pd.read_csv(INPUT_FILE)
    df['review'] = df['review'].astype(str).str.strip()
    df.dropna(subset=['review'], inplace=True)
    df = df[df['review'] != ''].copy()
    print(f"Data loaded successfully. Total reviews for topic modeling: {len(df)}")
except FileNotFoundError:
    print(f"Error: Sentiment data file not found at {INPUT_FILE}.")
    exit()

# 2. Apply Advanced Preprocessing
print("Applying advanced text preprocessing for topic modeling...")
df['review_processed'] = df['review'].apply(preprocess_for_topics)
df = df[df['review_processed'].str.len() > 0].copy()
print(f"Total reviews after final topic cleaning: {len(df)}")


# 3. Vectorization using TF-IDF
print(f"\nCreating TF-IDF vectors...")
# Documenting key parameters used for reproducibility
vectorizer = TfidfVectorizer(
    max_df=0.85,    
    min_df=5,       
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(df['review_processed'])
print(f"TF-IDF matrix created with {X.shape[1]} features (words/phrases).")

# 4. K-Means Clustering
print(f"Running K-Means Clustering with K={NUM_TOPICS}...")
# Documenting key parameters used for reproducibility
kmeans = KMeans(n_clusters=NUM_TOPICS, 
                init='k-means++', 
                max_iter=100, 
                n_init=1, 
                random_state=42)
kmeans.fit(X)

# Assign cluster labels back to the DataFrame
df['topic_cluster'] = kmeans.labels_


# 5. Topic Interpretation (Extracting Top Words for Each Cluster)

print("\n--- Topic Interpretation (Top 10 Keywords per Topic) ---")
feature_names = vectorizer.get_feature_names_out()
topic_names = {}
topic_documentation = {} # Dictionary to store the explicit mapping
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

for i in range(NUM_TOPICS):
    top_words = [feature_names[ind] for ind in order_centroids[i, :10]]
    topic_label = f"Topic {i+1}: {' | '.join(top_words)}"
    
    # Store the mapping for documentation
    topic_documentation[f"Topic {i+1}"] = {
        "keywords": top_words,
        "inferred_theme": None # Placeholder for manual human interpretation
    }
    
    topic_names[i] = topic_label
    print(topic_label)

# Save the explicit keyword mapping as a JSON file
with open(TOPIC_KEYWORDS_FILE, 'w') as f:
    json.dump({
        "parameters": {
            "K_clusters": NUM_TOPICS,
            "TFIDF_max_df": 0.85,
            "TFIDF_min_df": 5,
            "TFIDF_ngram_range": "(1, 2)"
        },
        "topics": topic_documentation
    }, f, indent=4)
print(f"\nExplicit keyword mapping and parameters saved to {TOPIC_KEYWORDS_FILE}")

# Map the final topic name back to the DataFrame
df['topic_label'] = df['topic_cluster'].map(topic_names)


# 6. Final Save
df.drop('review_processed', axis=1, inplace=True)

df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ All analysis complete. Final data saved to {OUTPUT_FILE}")