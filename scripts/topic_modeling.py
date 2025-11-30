import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

# --- NLTK Preparation (Simplified Block) ---
# This uses a safer check for missing resources by catching LookupError.
try:
    # Check if necessary resources are available
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    print("NLTK resources checked and found.")
except LookupError:
    print("NLTK resources missing. Downloading...")
    # Download them if they are missing
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("NLTK resources successfully downloaded.")
    
# --- Configuration ---
INPUT_FILE = 'data/analyzed_reviews_sentiment_only.csv'
OUTPUT_FILE = 'data/analyzed_reviews.csv'
NUM_TOPICS = 10  # A good starting point for K-Means clusters (K)

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
    
    # 1. Tokenize (split into words)
    tokens = re.findall(r'\b\w+\b', str(text).lower())
    
    # 2. Stop word removal and Lemmatization
    processed_tokens = []
    for word in tokens:
        if word not in stop_words and len(word) > 2: # Exclude short words
            processed_tokens.append(lemmatizer.lemmatize(word))
            
    # 3. Join tokens back into a single string
    return " ".join(processed_tokens)

# --- Main Logic ---

print(f"Loading sentiment-analyzed data from: {INPUT_FILE}")

# 1. Load Data
try:
    df = pd.read_csv(INPUT_FILE)
    df.dropna(subset=['review'], inplace=True)
    df = df[df['review'] != '']
    print(f"Data loaded successfully. Total reviews for topic modeling: {len(df)}")
except FileNotFoundError:
    print(f"Error: Sentiment data file not found at {INPUT_FILE}.")
    exit()

# 2. Apply Advanced Preprocessing
print("Applying advanced text preprocessing for topic modeling...")
df['review_processed'] = df['review'].apply(preprocess_for_topics)
# Remove rows where preprocessing resulted in an empty string
df = df[df['review_processed'].str.len() > 0].copy()
print(f"Total reviews after final topic cleaning: {len(df)}")


# 3. Vectorization using TF-IDF
print(f"\nCreating TF-IDF vectors...")
vectorizer = TfidfVectorizer(
    max_df=0.85,    # Ignore terms that appear in more than 85% of documents
    min_df=5,       # Ignore terms that appear in less than 5 documents
    ngram_range=(1, 2) # Consider 1-word and 2-word phrases
)
X = vectorizer.fit_transform(df['review_processed'])
print(f"TF-IDF matrix created with {X.shape[1]} features (words/phrases).")

# 4. K-Means Clustering
print(f"Running K-Means Clustering with K={NUM_TOPICS}...")
kmeans = KMeans(n_clusters=NUM_TOPICS, 
                init='k-means++', 
                max_iter=100, 
                n_init=1, # Setting n_init=1 is usually sufficient for large datasets
                random_state=42)
kmeans.fit(X)

# Assign cluster labels back to the DataFrame
df['topic_cluster'] = kmeans.labels_


# 5. Topic Interpretation (Extracting Top Words for Each Cluster)

print("\n--- Topic Interpretation (Top 10 Keywords per Topic) ---")
# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()
topic_names = {}
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

for i in range(NUM_TOPICS):
    # Get the top 10 keywords for the cluster
    top_words = [feature_names[ind] for ind in order_centroids[i, :10]]
    topic_label = f"Topic {i+1}: {' | '.join(top_words)}"
    topic_names[i] = topic_label
    print(topic_label)

# Map the final topic name back to the DataFrame
df['topic_label'] = df['topic_cluster'].map(topic_names)


# 6. Final Save
# Drop the temporary processed column before saving
df.drop('review_processed', axis=1, inplace=True)

# Save the final, fully analyzed DataFrame
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ All analysis complete. Final data saved to {OUTPUT_FILE}")