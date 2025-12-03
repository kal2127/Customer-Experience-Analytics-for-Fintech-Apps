import pandas as pd
import psycopg2
from psycopg2 import sql
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration: Use the same PostgreSQL details ---
DB_CONFIG = {
    "host": "localhost",
    "database": "bank_reviews", 
    "user": "postgres",     # CORRECTED: Use the user from database_load.py
    "password": "kal",        # CORRECTED: Use the password from database_load.py
    "port": "5432"            # CORRECTED: Use the port from database_load.py
}

# --- Plotting Setup ---
sns.set_theme(style="whitegrid")
REPORT_PATH = 'reports'
os.makedirs(REPORT_PATH, exist_ok=True)


def get_data_from_db(conn, query, params=None):
    """Executes a SQL query and returns the results as a pandas DataFrame."""
    try:
        df = pd.read_sql(query, conn, params=params)
        return df
    except Exception as e:
        print(f"❌ Error running SQL query: {e}")
        return pd.DataFrame()

def create_sentiment_comparison_chart(df_sentiment):
    """Visualizes sentiment distribution across all banks."""
    print("Generating Sentiment Distribution chart...")
    
    # Calculate percentage for plotting
    df_plot = df_sentiment.groupby(['bank_name', 'sentiment_label']).size().reset_index(name='count')
    df_sum = df_plot.groupby('bank_name')['count'].sum().reset_index(name='total')
    df_plot = pd.merge(df_plot, df_sum, on='bank_name')
    df_plot['percentage'] = (df_plot['count'] / df_plot['total']) * 100
    
    plt.figure(figsize=(12, 7))
    # Order sentiments logically
    sentiment_order = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    
    ax = sns.barplot(
        x='bank_name', 
        y='percentage', 
        hue='sentiment_label', 
        data=df_plot,
        order=df_plot['bank_name'].unique(),
        hue_order=sentiment_order,
        palette={'POSITIVE': 'g', 'NEUTRAL': 'y', 'NEGATIVE': 'r'}
    )
    
    plt.title('Bank Comparison: Sentiment Distribution of Mobile App Reviews', fontsize=16)
    plt.xlabel('Bank', fontsize=14)
    plt.ylabel('Percentage of Reviews', fontsize=14)
    plt.legend(title='Sentiment', loc='upper left')
    plt.xticks(rotation=0)
    
    # Add data labels
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.1f}%', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', 
                    xytext=(0, 5), 
                    textcoords='offset points',
                    fontsize=9)

    file_path = os.path.join(REPORT_PATH, 'bank_sentiment_comparison.png')
    plt.savefig(file_path)
    plt.close()
    print(f"✅ Saved chart to {file_path}")
    


def create_top_negative_themes_chart(df_negative):
    """Visualizes the most frequent topics within negative reviews."""
    print("Generating Top Negative Themes chart...")
    
    # Filter only negative reviews
    negative_reviews = df_negative[df_negative['sentiment_label'] == 'NEGATIVE'].copy()
    
    # Count the frequency of each topic label within negative reviews
    topic_counts = negative_reviews['topic_label'].value_counts().nlargest(10).reset_index()
    topic_counts.columns = ['topic_label', 'count']
    
    plt.figure(figsize=(10, 8))
    
    sns.barplot(
        x='count', 
        y='topic_label', 
        data=topic_counts,
        palette='viridis' # Use a distinct color palette
    )
    
    plt.title('Top 10 Themes Driving Negative Customer Experience', fontsize=16)
    plt.xlabel('Number of Negative Reviews', fontsize=14)
    plt.ylabel('Topic/Theme', fontsize=14)
    plt.tight_layout()
    
    # Add data labels
    for index, row in topic_counts.iterrows():
        plt.text(row['count'] + 5, index, str(row['count']), color='black', ha="left", va="center")
        
    file_path = os.path.join(REPORT_PATH, 'top_negative_themes.png')
    plt.savefig(file_path)
    plt.close()
    print(f"✅ Saved chart to {file_path}")
    


def main():
    """Main function to orchestrate the reporting and visualization."""
    conn = create_connection()
    if conn is None:
        return
        
    # --- SQL Queries to Extract Data for Reporting ---

    # 1. Query for Sentiment Comparison Chart
    print("\nExecuting query for sentiment comparison...")
    sentiment_query = """
    SELECT 
        b.bank_name, 
        r.sentiment_label
    FROM 
        reviews r
    JOIN 
        banks b ON r.bank_id = b.bank_id;
    """
    df_sentiment = get_data_from_db(conn, sentiment_query)
    
    if df_sentiment.empty:
        print("No data retrieved for sentiment comparison. Exiting.")
        conn.close()
        return

    # 2. Query for Negative Themes Chart (Fetch all data needed to filter locally)
    print("Executing query for negative theme analysis...")
    theme_query = """
    SELECT 
        sentiment_label, 
        topic_label 
    FROM 
        reviews;
    """
    df_theme = get_data_from_db(conn, theme_query)

    # --- Generate Visualizations ---
    create_sentiment_comparison_chart(df_sentiment)
    create_top_negative_themes_chart(df_theme)

    # --- Final Reporting Insights ---
    print("\n--- Key Analytical Insights for Report ---")
    
    # Example Insight 1: Highest and Lowest Positive Sentiment
    sentiment_pivot = df_sentiment.groupby(['bank_name', 'sentiment_label']).size().unstack(fill_value=0)
    sentiment_pivot['Total'] = sentiment_pivot.sum(axis=1)
    sentiment_pivot['Positive_Ratio'] = sentiment_pivot['POSITIVE'] / sentiment_pivot['Total']
    
    best_bank = sentiment_pivot['Positive_Ratio'].idxmax()
    worst_bank = sentiment_pivot['Positive_Ratio'].idxmin()
    
    print(f"🥇 Highest Positive Sentiment: {best_bank} (Ratio: {sentiment_pivot['Positive_Ratio'].max():.2f})")
    print(f"🥉 Lowest Positive Sentiment: {worst_bank} (Ratio: {sentiment_pivot['Positive_Ratio'].min():.2f})")
    
    # Example Insight 2: Worst Performing Topic (based on count)
    negative_reviews = df_theme[df_theme['sentiment_label'] == 'NEGATIVE']
    
    if not negative_reviews.empty:
        worst_topic = negative_reviews['topic_label'].value_counts().idxmax()
        worst_topic_count = negative_reviews['topic_label'].value_counts().max()
        print(f"🔥 Most Critical Pain Point (Topic): '{worst_topic}' with {worst_topic_count} Negative Mentions.")
    else:
        print("🔥 Most Critical Pain Point (Topic): No negative reviews found to analyze.")
    
    conn.close()
    print("\nReporting and Visualization Complete.")

# Reuse the connection function from the database_load script
def create_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False # Turn off autocommit for safer querying
        print("✅ PostgreSQL connection successful for reporting.")
        return conn
    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL: {e}")
        return None


if __name__ == "__main__":
    # Ensure you've replaced credentials in DB_CONFIG before running!
    main()