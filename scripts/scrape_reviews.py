import pandas as pd
from google_play_scraper import Sort, reviews_all

# --- Configuration ---
# Dictionary of the App IDs and the label we want to assign
APP_LIST = {
    "CBE": "prod.cbe.birr", # Commercial Bank of Ethiopia (CBEBirr Plus)
    "BOA": "com.boa.boaMobileBanking", # Bank of Abyssinia (BoA Mobile)
    "Dashen": "com.cr2.amolelight" # Dashen Bank (Dashen Mobile/Amole Lite)
}

# The minimum required number of reviews per bank
MIN_REVIEWS_PER_BANK = 400

# List to hold all collected review data (dictionaries)
all_reviews = []

print("Starting web scraping process...")

# --- Main Scraping Loop ---
for bank_name, app_id in APP_LIST.items():
    print(f"\n--- Scraping reviews for {bank_name} (App ID: {app_id}) ---")
    
    try:
        # Use reviews_all to efficiently scrape a large number of reviews
        # We specify the maximum number of reviews to fetch (MIN_REVIEWS_PER_BANK)
        # and sort by 'newest' for the most recent data.
        current_app_reviews = reviews_all(
            app_id,
            sleep_milliseconds=0, # Use a delay if you scrape a massive amount
            lang='en', # Prefer English reviews if possible
            country='et', # Set country to Ethiopia for relevance
            sort=Sort.NEWEST,
            filter_score_with=None # Get all ratings (1-5 stars)
        )

        # Process and store the reviews
        for review in current_app_reviews:
            # We must rename the columns to match the project requirements (review, rating, date, bank, source)
            all_reviews.append({
                'review': review['content'],
                'rating': review['score'],
                'date': review['at'],
                'bank': bank_name,
                'source': 'Google Play'
            })
            
        print(f"✅ Scraped {len(current_app_reviews)} reviews for {bank_name}.")

    except Exception as e:
        print(f"❌ An error occurred while scraping {bank_name}: {e}")
        print("Moving to the next bank.")

# --- Data Compilation ---
if all_reviews:
    # Convert the list of dictionaries into a Pandas DataFrame
    df_raw = pd.DataFrame(all_reviews)
    
    # Define the path to save the raw data in the 'data' folder
    output_path = 'data/raw_google_play_reviews.csv'
    
    # Save the raw, uncleaned data (optional, but good practice)
    df_raw.to_csv(output_path, index=False)
    
    print(f"\n🏁 Scraping complete! Total reviews collected: {len(df_raw)}.")
    print(f"Data saved to {output_path}")
    print(df_raw['bank'].value_counts())
else:
    print("\n⚠️ No reviews were collected. Check your internet connection or app IDs.")


# After running the script, the raw data is ready for Step 3: Preprocessing.