"""
MODULE 2 - DATA CLEANING
Dataset: social_media_engagement_dataset.csv (Primary engagement dataset)

Every step below is explained inline so it can be reused directly in the
project report / notebook, and explained verbally in the viva.
"""

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------
# STEP 1: Load raw data
# -----------------------------------------------------------------------
df = pd.read_csv("data/raw/raw_engagement_dataset.csv")
print("Initial shape:", df.shape)

# -----------------------------------------------------------------------
# STEP 2: Remove exact duplicate rows and duplicate Post_IDs
# -----------------------------------------------------------------------
before = len(df)
df = df.drop_duplicates()
df = df.drop_duplicates(subset="Post_ID", keep="first")
print(f"Removed {before - len(df)} duplicate rows (0 found, dataset was already unique)")

# -----------------------------------------------------------------------
# STEP 3: Fix data types
# -----------------------------------------------------------------------
# Timestamp was a plain string -> convert to real datetime so we can extract
# proper time-series features later (trend forecasting, Module 10).
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Has_Media / Is_Verified were already boolean - confirm and keep as bool.
df["Has_Media"] = df["Has_Media"].astype(bool)
df["Is_Verified"] = df["Is_Verified"].astype(bool)

# Categorical columns -> pandas 'category' dtype (memory efficient, and
# makes group-by operations in EDA faster and cleaner).
categorical_cols = ["Platform", "Content_Type", "Category", "Day_of_Week",
                     "Sentiment", "Influencer_Tier"]
for col in categorical_cols:
    df[col] = df[col].str.strip()          # defensive strip, even though clean here
    df[col] = df[col].astype("category")

# -----------------------------------------------------------------------
# STEP 4: Missing value check
# -----------------------------------------------------------------------
missing = df.isnull().sum()
print("\nMissing values per column:\n", missing[missing > 0] if missing.sum() else "None found.")
# Verified in Phase 2: this dataset has 0 missing values. No imputation needed.
# We still run this check as a permanent safety step in the pipeline.

# -----------------------------------------------------------------------
# STEP 5: Logical consistency checks (real issue found in Phase 2)
# -----------------------------------------------------------------------
# The provided Engagement_Rate column does NOT match the standard formula
# (Likes+Comments+Shares+Saves)/Views*100 -- verified correlation ~0.009.
# Decision (confirmed with project owner): recompute our own engagement
# rate as the metric used throughout the project, and RETAIN the original
# column under a new name purely for transparency/comparison.
df = df.rename(columns={"Engagement_Rate": "Engagement_Rate_Provided"})
df["Engagement_Rate"] = (
    (df["Likes"] + df["Comments"] + df["Shares"] + df["Saves"]) / df["Views"] * 100
)

# -----------------------------------------------------------------------
# STEP 6: Outlier identification (flag, do not silently delete)
# -----------------------------------------------------------------------
# We use the IQR method on our recomputed Engagement_Rate to flag outliers.
# Flagging instead of deleting preserves genuine high-performing ("viral")
# posts, which are exactly the rows Module 6 needs to predict.
Q1 = df["Engagement_Rate"].quantile(0.25)
Q3 = df["Engagement_Rate"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df["Is_Outlier_Engagement"] = (
    (df["Engagement_Rate"] < lower_bound) | (df["Engagement_Rate"] > upper_bound)
)
print(f"\nOutliers flagged (IQR method): {df['Is_Outlier_Engagement'].sum()} "
      f"of {len(df)} rows ({df['Is_Outlier_Engagement'].mean()*100:.1f}%)")

# -----------------------------------------------------------------------
# STEP 7: Feature engineering needed for later modules (kept here so the
# cleaned file is immediately usable without re-deriving these each time)
# -----------------------------------------------------------------------
df["Like_Rate"] = df["Likes"] / df["Views"] * 100
df["Comment_Rate"] = df["Comments"] / df["Views"] * 100
df["Share_Rate"] = df["Shares"] / df["Views"] * 100
df["Save_Rate"] = df["Saves"] / df["Views"] * 100
df["Save_to_Share_Ratio"] = df["Saves"] / df["Shares"].replace(0, np.nan)
df["Comment_to_Like_Ratio"] = df["Comments"] / df["Likes"].replace(0, np.nan)

df["Month"] = df["Timestamp"].dt.month
df["Year"] = df["Timestamp"].dt.year

# -----------------------------------------------------------------------
# STEP 8: Final sanity check + save
# -----------------------------------------------------------------------
print("\nFinal shape:", df.shape)
print("Final dtypes:\n", df.dtypes)

df.to_csv("data/cleaned/cleaned_engagement_dataset.csv", index=False)
print("\nSaved -> data/cleaned/cleaned_engagement_dataset.csv")
