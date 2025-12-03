import psycopg2
from psycopg2 import sql
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np # Needed for array manipulation

# --- Configuration: Must match database_load.py ---
DB_CONFIG = {
    "host": "localhost",
    "database": "bank_reviews", 
    "user": "postgres",     
    "password": "kal",        
    "port": "5432"           
}

# --- Report Settings ---
REPORT_DIR = 'report'
PLOT_PATHS = {
    'bank_comparison': os.path.join(REPORT_DIR, 'bank_review_comparison.png'),
    'sentiment_distribution': os.path.join(REPORT_DIR, 'sentiment_distribution.png')
}

def create_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ PostgreSQL connection successful.")
        return conn
    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL: {e}")
        return None

def check_data_status(conn):
    """Checks if the reviews table is populated."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM reviews;")
            count = cur.fetchone()[0]
            print(f"✅ Data Check: Found {count} reviews in the 'reviews' table.")
            return count > 0
    except Exception as e:
        print(f"❌ Error checking data status: {e}")
        return False

def generate_report_data(conn):
    """Queries the database to generate all necessary metrics for the report."""
    
    # 1. Overall Metrics (Total Reviews, Average Rating)
    overall_metrics_query = """
    SELECT 
        COUNT(*) AS total_reviews,
        ROUND(AVG(rating), 2) AS average_rating
    FROM reviews;
    """
    
    # 2. Bank-Specific Metrics (Review Count, Avg Rating)
    bank_metrics_query = """
    SELECT 
        b.bank_name,
        COUNT(r.review_id) AS review_count,
        ROUND(AVG(r.rating), 2) AS average_rating
    FROM reviews r
    JOIN banks b ON r.bank_id = b.bank_id
    GROUP BY b.bank_name
    ORDER BY review_count DESC;
    """
    
    # 3. Sentiment Distribution
    sentiment_query = """
    SELECT
        b.bank_name,
        r.sentiment_label,
        COUNT(r.review_id) AS count
    FROM reviews r
    JOIN banks b ON r.bank_id = b.bank_id
    GROUP BY b.bank_name, r.sentiment_label
    ORDER BY b.bank_name, count DESC;
    """

    # 4. Top Topics by Sentiment (Focus on negative sentiment)
    topic_query = """
    WITH topic_sentiment AS (
        SELECT
            r.topic_label,
            r.sentiment_label,
            COUNT(r.review_id) as count
        FROM reviews r
        WHERE r.sentiment_label = 'NEGATIVE' 
        GROUP BY r.topic_label, r.sentiment_label
    )
    SELECT
        topic_label,
        count
    FROM topic_sentiment
    ORDER BY count DESC
    LIMIT 5;
    """

    results = {}

    try:
        with conn.cursor() as cur:
            # Execute queries and fetch results
            cur.execute(overall_metrics_query)
            results['overall_metrics'] = cur.fetchone()
            
            cur.execute(bank_metrics_query)
            results['bank_metrics'] = cur.fetchall()

            cur.execute(sentiment_query)
            results['sentiment_data'] = cur.fetchall()

            cur.execute(topic_query)
            results['top_negative_topics'] = cur.fetchall()
            
    except Exception as e:
        print(f"❌ Error executing SQL queries: {e}")
        return None

    return results

def format_sentiment_data(sentiment_data):
    """Formats raw sentiment data into a dictionary structure suitable for the report."""
    formatted = {}
    for bank_name, label, count in sentiment_data:
        if bank_name not in formatted:
            formatted[bank_name] = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0, 'Total': 0}
        
        # Ensure label exists to avoid KeyError if data is patchy
        if label in formatted[bank_name]:
             formatted[bank_name][label] = count
        
        formatted[bank_name]['Total'] += count
        
    return formatted

def generate_visualizations(bank_metrics, sentiment_data_raw, report_dir):
    """Generates and saves visual plots."""
    print("Generating visualizations...")
    
    # Ensure directory exists (already done in main, but safe to check again)
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    # 1. Bank Review Comparison (Bar Chart)
    bank_names = [m[0] for m in bank_metrics]
    review_counts = [m[1] for m in bank_metrics]
    
    plt.figure(figsize=(10, 6))
    plt.bar(bank_names, review_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Total Review Volume by Bank', fontsize=16)
    plt.xlabel('Bank Name', fontsize=12)
    plt.ylabel('Review Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    
    # Add labels on top of bars
    for i, count in enumerate(review_counts):
        plt.text(i, count + max(review_counts)*0.01, f'{count:,}', ha='center')
        
    plt.tight_layout()
    plt.savefig(PLOT_PATHS['bank_comparison'])
    plt.close()
    print(f"   -> Saved: {PLOT_PATHS['bank_comparison']}")


    # 2. Sentiment Distribution (Stacked Bar Chart)
    sentiment_data = format_sentiment_data(sentiment_data_raw)
    
    banks = list(sentiment_data.keys())
    positive = [sentiment_data[b]['POSITIVE'] for b in banks]
    negative = [sentiment_data[b]['NEGATIVE'] for b in banks]
    neutral = [sentiment_data[b]['NEUTRAL'] for b in banks]
    totals = [sentiment_data[b]['Total'] for b in banks]

    if not any(totals):
        print("   -> Skipping sentiment visualization: No review data found for sentiment.")
        return

    # Normalize data to percentages
    positive_perc = [p / t * 100 if t > 0 else 0 for p, t in zip(positive, totals)]
    negative_perc = [n / t * 100 if t > 0 else 0 for n, t in zip(negative, totals)]
    neutral_perc = [n / t * 100 if t > 0 else 0 for n, t in zip(neutral, totals)]

    # Plotting
    width = 0.6
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Stacking bars
    ax.bar(banks, positive_perc, width, label='Positive', color='#2ca02c')
    ax.bar(banks, neutral_perc, width, bottom=positive_perc, label='Neutral', color='#ffbf00')
    ax.bar(banks, negative_perc, width, bottom=np.array(positive_perc) + np.array(neutral_perc), label='Negative', color='#d62728')
    
    # Add percentage labels
    for i in range(len(banks)):
        # Positive Label
        if positive_perc[i] > 5:
            ax.text(i, positive_perc[i] / 2, f'{positive_perc[i]:.1f}%', ha='center', va='center', color='white', fontsize=9)
        # Neutral Label
        if neutral_perc[i] > 5:
            ax.text(i, positive_perc[i] + neutral_perc[i] / 2, f'{neutral_perc[i]:.1f}%', ha='center', va='center', color='black', fontsize=9)
        # Negative Label
        if negative_perc[i] > 5:
            ax.text(i, positive_perc[i] + neutral_perc[i] + negative_perc[i] / 2, f'{negative_perc[i]:.1f}%', ha='center', va='center', color='white', fontsize=9)

    ax.set_title('Sentiment Distribution by Bank', fontsize=16)
    ax.set_ylabel('Percentage of Reviews (%)', fontsize=12)
    ax.set_xlabel('Bank Name', fontsize=12)
    ax.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
    ax.set_yticks(np.arange(0, 101, 10))
    
    plt.tight_layout()
    plt.savefig(PLOT_PATHS['sentiment_distribution'])
    plt.close()
    print(f"   -> Saved: {PLOT_PATHS['sentiment_distribution']}")


def generate_markdown_report(data):
    """Generates a structured Markdown report from the queried data, now including image references."""
    if not data:
        return "## Analysis Report Failed\nData retrieval from the database failed."

    # --- Extract Data ---
    total_reviews, avg_rating = data['overall_metrics']
    bank_metrics = data['bank_metrics']
    sentiment_data = format_sentiment_data(data['sentiment_data'])
    top_negative_topics = data['top_negative_topics']

    report = [
        "# Customer Experience Analytics Report: Mobile Fintech Apps",
        "## Executive Summary",
        f"The analysis of **{total_reviews:,}** user reviews reveals an overall average rating of **{avg_rating} / 5.00** across the monitored banks. While sentiment is generally positive, specific pain points related to app stability and user interface remain critical areas for improvement.",
        "",
        "---",
        "## 1. Overall Performance Metrics",
        "",
        f"| Metric | Value |",
        f"| :--- | :--- |",
        f"| Total Reviews Analyzed | {total_reviews:,} |",
        f"| Weighted Average Rating | {avg_rating} / 5.00 |",
        "",
        "## 2. Bank-Specific Performance Comparison",
        "This table compares the volume and quality of reviews for each bank.",
        "",
        f"| Bank Name | Total Reviews | Avg. Rating |",
        f"| :--- | :--- | :--- |"
    ]

    for name, count, rating in bank_metrics:
        report.append(f"| {name} | {count:,} | {rating} |")
    
    # Add the visualization reference
    report.append("")
    report.append("### Review Volume Comparison")
    report.append("![Bank Review Volume Comparison](report/bank_review_comparison.png)")
    
    report.append("\n## 3. Sentiment Distribution Analysis")
    report.append("A breakdown of review sentiment (Positive, Negative, Neutral) for each banking application.")
    report.append("")
    
    # Add the visualization reference
    report.append("### Sentiment Breakdown")
    report.append("![Sentiment Distribution by Bank](report/sentiment_distribution.png)")
    report.append("")
    
    
    for bank_name in [m[0] for m in bank_metrics]:
        sentiments = sentiment_data.get(bank_name, {})
        bank_total = sentiments.get('Total', 0)
        
        positive = sentiments.get('POSITIVE', 0)
        negative = sentiments.get('NEGATIVE', 0)
        neutral = sentiments.get('NEUTRAL', 0)
        
        report.append(f"### {bank_name} (Total: {bank_total:,} Reviews)")
        report.append(f"| Sentiment | Count | Percentage |")
        report.append(f"| :--- | :--- | :--- |")
        
        if bank_total > 0:
            report.append(f"| ✅ Positive | {positive:,} | {positive/bank_total * 100:.1f}% |")
            report.append(f"| 🔴 Negative | {negative:,} | {negative/bank_total * 100:.1f}% |")
            report.append(f"| 🟡 Neutral | {neutral:,} | {neutral/bank_total * 100:.1f}% |")
        else:
             report.append("| (No data available) | | |")
        report.append("")


    report.append("## 4. Key Pain Points (Topic Modeling)")
    report.append("This section identifies the top 5 topics most frequently associated with **Negative** reviews.")
    report.append("")
    report.append(f"| Rank | Topic Label | Negative Review Count |")
    report.append(f"| :--- | :--- | :--- |")
    
    for i, (topic_label, count) in enumerate(top_negative_topics):
        report.append(f"| {i+1} | {topic_label} | {count:,} |")

    report.append("\n## 5. Actionable Recommendations")
    report.append("Based on the data and identified pain points, we recommend the following:")
    report.append("")
    report.append("1. **Focus on Stability and Speed:** The highest volume of negative reviews is often concentrated on technical issues (crashes, slow performance). An immediate focus on server stability and app responsiveness is necessary.")
    report.append("2. **Simplify High-Frequency Tasks:** If topics related to 'use' or 'easy' are high, redesign the flow for common actions (e.g., transfers, bill payments) to reduce friction.")
    report.append("3. **Proactive Communication:** For frequently criticized topics, provide in-app notifications about planned maintenance or bug fixes to show users that feedback is being addressed.")

    return "\n".join(report)

def main():
    """Main function to run the reporting process."""
    conn = create_connection()
    if conn is None:
        return

    if not check_data_status(conn):
        print("\n❌ CRITICAL ERROR: The 'reviews' table is empty. Please run 'scripts/database_load.py' successfully first to populate the data.")
        conn.close()
        return

    print("Generating report data from database...")
    data = generate_report_data(conn)
    
    # CRITICAL FIX: Ensure the 'report' directory exists before writing the file
    if not os.path.exists(REPORT_DIR):
        try:
            os.makedirs(REPORT_DIR)
            print(f"Created missing directory: {REPORT_DIR}")
        except Exception as e:
            print(f"❌ Error creating directory {REPORT_DIR}: {e}")
            conn.close()
            return
            
    if data:
        # Generate the plots first
        generate_visualizations(data['bank_metrics'], data['sentiment_data'], REPORT_DIR)
        
        # Then generate the report that references the plots
        markdown_report = generate_markdown_report(data)
        
        # Write the report to a markdown file
        try:
            report_path = os.path.join(REPORT_DIR, 'customer_experience_analysis.md')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(markdown_report)
            print(f"✅ Report successfully generated and saved to '{report_path}'")
            
            # Display the report content
            print("\n--- Generated Report Content (scripts/reporting_and_visualization.py) ---")
            print(markdown_report)
            print("-----------------------------------------------------------------------\n")
        except Exception as e:
            print(f"❌ Error writing report file: {e}")
            
    conn.close()

if __name__ == "__main__":
    main()