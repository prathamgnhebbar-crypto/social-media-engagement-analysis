"""
MODULE 2 - DATA CLEANING
Dataset: sentimentdataset.csv (Supplementary dataset, used only for Module 5 NLP)
"""

import pandas as pd
import re

# -----------------------------------------------------------------------
# STEP 1: Load raw data
# -----------------------------------------------------------------------
df = pd.read_csv("data/raw/raw_sentiment_dataset.csv")
print("Initial shape:", df.shape)

# -----------------------------------------------------------------------
# STEP 2: Drop junk index columns
# -----------------------------------------------------------------------
# 'Unnamed: 0.1' and 'Unnamed: 0' are leftover pandas index columns from
# whoever exported this CSV originally - they carry no information.
df = df.drop(columns=["Unnamed: 0.1", "Unnamed: 0"])

# -----------------------------------------------------------------------
# STEP 3: Strip whitespace from ALL string/object columns
# -----------------------------------------------------------------------
# Verified in Phase 2: every value in Sentiment and Platform has leading/
# trailing whitespace (e.g. " Twitter  "). This must be fixed before any
# grouping or categorical analysis, or "Twitter" and " Twitter " count as
# two different categories.
str_cols = df.select_dtypes(include="object").columns.tolist() + \
           [c for c in df.columns if df[c].dtype.name == "str"]
str_cols = list(set(str_cols))
for col in str_cols:
    df[col] = df[col].astype(str).str.strip()

# -----------------------------------------------------------------------
# STEP 4: Remove duplicate posts
# -----------------------------------------------------------------------
before = len(df)
df = df.drop_duplicates(subset="Text")
print(f"Removed {before - len(df)} duplicate Text rows")

# -----------------------------------------------------------------------
# STEP 5: Basic text cleaning (for NLP pipeline in Module 5)
# -----------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # remove URLs (defensive)
    text = re.sub(r"[^a-z0-9\s#]", "", text)          # keep letters/numbers/hashtags
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["Text_Clean"] = df["Text"].apply(clean_text)

# -----------------------------------------------------------------------
# STEP 6: Remap the 279 raw sentiment labels into 3 usable classes
# -----------------------------------------------------------------------
# The raw 'Sentiment' column has 279 unique fine-grained emotion words
# (Joy, Excitement, Despair, Hate, Contentment, etc). For a workable
# Positive/Neutral/Negative classification target, we group these using
# a manually curated, disclosed mapping. This is a real methodological
# step - not a fabrication - and must be described exactly this way in
# the report (a similar remap approach appears in published NLP work on
# this same public dataset).
positive_words = {
    "positive", "joy", "excitement", "contentment", "happy", "hopeful",
    "gratitude", "elation", "playful", "serenity", "admiration", "thrill",
    "amusement", "affection", "enthusiasm", "pride", "optimism", "love",
    "happiness", "delight", "wonder", "kind", "confident", "grateful",
    "euphoria", "hope", "empowerment", "inspiration", "calmness",
    "compassion", "tenderness", "fulfillment", "reverence", "enchantment",
    "awe", "determination", "acceptance", "creativity", "adventure",
    "freedom", "success", "motivation", "relief", "triumph", "radiance",
    "satisfaction", "blessed", "celebration", "charm", "cheerfulness",
    "coziness", "elegance", "energy", "engagement", "excited",
    "fascination", "free-spirited", "grandeur", "harmony", "imagination",
    "immersion", "innerpeace", "innocence", "inspired", "iconic",
    "joyfulreunion", "kindness", "mesmerizing", "overjoyed", "amazement",
    "pensive", "playfuljoy", "positivity", "proud", "relaxation",
    "renewedeffort", "resilience", "reverie", "romance", "rejuvenation",
    "vibrancy", "wonderment", "yearning", "zest", "arousal", "euphoric",
}
negative_words = {
    "negative", "sad", "embarrassed", "loneliness", "despair", "hate",
    "bad", "anger", "fear", "disgust", "frustration", "grief", "jealousy",
    "regret", "shame", "anxiety", "bitter", "bitterness", "disappointment",
    "hopeless", "hurt", "isolation", "betrayal", "melancholy", "numbness",
    "frustrated", "desolation", "overwhelmed", "devastated", "envious",
    "envy", "exhaustion", "heartbreak", "helplessness", "hopelessness",
    "intimidation", "loss", "miserable", "mixedemotions", "obstacle",
    "pressure", "resentment", "sadness", "suffering", "sorrow",
    "suspense", "vulnerability", "yearningsadness", "darkness",
    "disappointed", "disgusted", "dismissive", "fearful", "grievance",
    "heartache", "insecurity", "regretful", "ruins",
}

def remap_sentiment(raw_label: str) -> str:
    label = raw_label.strip().lower().replace(" ", "").replace("-", "")
    if label in positive_words or label.replace(" ", "") in positive_words:
        return "Positive"
    if label in negative_words or label.replace(" ", "") in negative_words:
        return "Negative"
    return "Neutral"   # genuinely ambiguous/mixed labels (Curiosity, Confusion,
                       # Surprise, Indifference, Ambivalence, Nostalgia, Reflection)

df["Sentiment_Raw"] = df["Sentiment"]          # keep the original for transparency
df["Sentiment_3class"] = df["Sentiment_Raw"].apply(remap_sentiment)

print("\n3-class sentiment distribution after remapping:")
print(df["Sentiment_3class"].value_counts())

# -----------------------------------------------------------------------
# STEP 7: Fix datatypes
# -----------------------------------------------------------------------
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Platform"] = df["Platform"].astype("category")
df["Country"] = df["Country"].astype("category")

# -----------------------------------------------------------------------
# STEP 8: Missing value check
# -----------------------------------------------------------------------
missing = df.isnull().sum()
print("\nMissing values per column:\n", missing[missing > 0] if missing.sum() else "None found.")

# -----------------------------------------------------------------------
# STEP 9: Save
# -----------------------------------------------------------------------
print("\nFinal shape:", df.shape)
df.to_csv("data/cleaned/cleaned_sentiment_dataset.csv", index=False)
print("Saved -> data/cleaned/cleaned_sentiment_dataset.csv")
