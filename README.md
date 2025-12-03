Customer Experience Analytics Pipeline for Mobile Fintech Apps

Project Summary

This project establishes an end-to-end Data Engineering and Analytics pipeline to assess customer satisfaction and identify pain points for major Ethiopian mobile banking applications (Commercial Bank of Ethiopia, Bank of Abyssinia, and Dashen Bank).

The pipeline systematically scrapes thousands of user reviews from the Google Play Store, processes them using Natural Language Processing (NLP) techniques for sentiment and topic analysis, loads the enriched data into a secure PostgreSQL database, and generates a final analytical report with actionable business recommendations and visualizations.

Key Objectives Achieved

Data Acquisition: Successfully scraped and cleaned raw user reviews.

NLP Analysis: Classified sentiment (Positive, Negative, Neutral) using VADER and clustered key themes (topics) using LDA to identify critical pain points.

Data Engineering: Created a robust, structured PostgreSQL database schema to persist all analyzed data.

Reporting: Generated a comprehensive, data-driven report with charts that summarize comparative performance and pinpoint areas for improvement (e.g., system stability, transaction speed).

🛠️ Technology Stack

Category

Technology

Purpose

Language

Python 3.x

Core pipeline logic, scripting, and NLP.

Web Scraping

google-play-scraper

Data collection from Google Play Store.

Data Handling

Pandas

Data cleaning, transformation, and manipulation.

NLP/ML

NLTK, Scikit-learn (LDA), VADER

Text preprocessing, Topic Modeling, and Sentiment Analysis.

Visualization

Matplotlib

Generating comparative bar charts and sentiment distribution plots.

Database

PostgreSQL

Data persistence and serving as the source for reporting.

DB Connector

Psycopg2

Python library for interacting with PostgreSQL.

⚙️ Data Pipeline Architecture

The project is structured into four sequential, modular stages:

Stage

Script

Description

1. Extraction & Cleaning

scripts/data_scraping_and_cleaning.py

Fetches raw reviews for targeted banks, standardizes text, and saves to play_store_reviews_clean.csv.

2. NLP Analysis

scripts/sentiment_and_topic_modeling.py

Applies VADER for sentiment scoring and LDA for topic clustering. The enriched data is saved to analyzed_reviews.csv.

3. Database Loading

scripts/database_load.py

Connects to PostgreSQL, defines the schema (banks, reviews tables), and bulk-loads the analyzed data for persistent storage.

4. Reporting & Visualization

scripts/reporting_and_visualization.py

Queries the PostgreSQL database, calculates metrics (Avg. Rating, Sentiment Distribution), generates image visualizations, and compiles a final markdown report (customer_experience_analysis.md).

🚀 Setup and Execution

Prerequisites

Python: Python 3.8+ must be installed.

Dependencies: Install the required Python libraries:

pip install pandas numpy nltk scikit-learn matplotlib psycopg2-binary google-play-scraper


PostgreSQL: A running PostgreSQL instance (e.g., using Docker or local installation). Database configuration is defined in the script configuration variables.

Running the Project

Execute the pipeline scripts sequentially from the project root directory:

Run Data Scraping and Cleaning:

python scripts/data_scraping_and_cleaning.py


Run NLP Analysis (This generates the topics and sentiment):

python scripts/sentiment_and_topic_modeling.py


Load Data into PostgreSQL:

python scripts/database_load.py


Generate Final Report and Visualizations:

python scripts/reporting_and_visualization.py


The final report and generated charts will be saved in the report/ directory.

💡 Key Findings

The analysis of 2,967 reviews yielded an overall average rating of 3.53/5.00, but revealed stark differences in customer experience across the banks:

Bank Name

Total Reviews

Avg. Rating

Negative Sentiment %

Commercial Bank of Ethiopia (CBE)

1,565

3.85

20.3%

Bank of Abyssinia (BOA)

992

2.84

58.8%

Dashen Bank

410

3.96

15.6%

Primary Pain Points (Derived from Topic Modeling)

The most frequent negative topics centered on:

System Reliability / Crashes: Explicit complaints involving keywords like 'fix', 'worst', 'system', and 'doesn't work properly'.

Account Access & Service Downtime: Frustrations related to 'update', 'service', 'money', and being unable to 'open' or 'access' their accounts.

⏭️ Future Enhancements

Multilingual Support: Integrate a multilingual NLP model (e.g., mBERT) for accurate analysis of Amharic and other local languages.

Time-Series Analysis: Track sentiment change over time to correlate with application updates or marketing efforts.

Interactive Dashboard: Deploy a BI tool (e.g., Streamlit, Dash) to query the PostgreSQL database and provide real-time, interactive performance monitoring.
