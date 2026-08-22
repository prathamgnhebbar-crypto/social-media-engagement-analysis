"""
MODULE 3 - EXPLORATORY DATA ANALYSIS
Dataset: cleaned_engagement_dataset.csv

Generates the 7 required visualizations and prints the numeric evidence
behind each observation (so nothing in the report is asserted without a
number to back it up).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")
df = pd.read_csv("data/cleaned/cleaned_engagement_dataset.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

OUT = "visualizations/"

# -------------------------------------------------------------------
# 1. Distribution of engagement (recomputed Engagement_Rate)
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(df["Engagement_Rate"], bins=60, ax=axes[0], color="#4C72B0")
axes[0].set_title("Engagement Rate Distribution (raw)")
axes[0].set_xlabel("Engagement Rate (%)")

sns.histplot(np.log1p(df["Engagement_Rate"]), bins=60, ax=axes[1], color="#55A868")
axes[1].set_title("Engagement Rate Distribution (log-transformed)")
axes[1].set_xlabel("log(1 + Engagement Rate)")
plt.tight_layout()
plt.savefig(OUT + "01_engagement_distribution.png", dpi=150)
plt.close()

print("1. ENGAGEMENT DISTRIBUTION")
print(df["Engagement_Rate"].describe())
print("Skewness:", df["Engagement_Rate"].skew())
print()

# -------------------------------------------------------------------
# 2. Engagement by content type
# -------------------------------------------------------------------
order = df.groupby("Content_Type", observed=True)["Engagement_Rate"].median().sort_values(ascending=False).index
plt.figure(figsize=(11, 5))
sns.boxplot(data=df, x="Content_Type", y="Engagement_Rate", order=order, showfliers=False)
plt.xticks(rotation=45, ha="right")
plt.title("Engagement Rate by Content Type")
plt.tight_layout()
plt.savefig(OUT + "02_engagement_by_content_type.png", dpi=150)
plt.close()

print("2. ENGAGEMENT BY CONTENT TYPE (median, sorted)")
print(df.groupby("Content_Type", observed=True)["Engagement_Rate"].median().sort_values(ascending=False))
print()

# -------------------------------------------------------------------
# 3. Engagement by topic (Category)
# -------------------------------------------------------------------
order_cat = df.groupby("Category", observed=True)["Engagement_Rate"].median().sort_values(ascending=False).index
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="Category", y="Engagement_Rate", order=order_cat, showfliers=False)
plt.xticks(rotation=45, ha="right")
plt.title("Engagement Rate by Content Category / Topic")
plt.tight_layout()
plt.savefig(OUT + "03_engagement_by_category.png", dpi=150)
plt.close()

print("3. ENGAGEMENT BY CATEGORY (median, sorted)")
print(df.groupby("Category", observed=True)["Engagement_Rate"].median().sort_values(ascending=False))
print()

# -------------------------------------------------------------------
# 4. Engagement by posting time (hour of day)
# -------------------------------------------------------------------
plt.figure(figsize=(11, 5))
hourly = df.groupby("Hour_of_Day")["Engagement_Rate"].median()
sns.lineplot(x=hourly.index, y=hourly.values, marker="o", color="#C44E52")
plt.title("Median Engagement Rate by Hour of Day")
plt.xlabel("Hour of Day (0-23)")
plt.ylabel("Median Engagement Rate (%)")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig(OUT + "04_engagement_by_hour.png", dpi=150)
plt.close()

print("4. ENGAGEMENT BY HOUR (median)")
print(hourly.sort_values(ascending=False).head(5))
print()

# -------------------------------------------------------------------
# 5. Correlation heatmap
# -------------------------------------------------------------------
num_cols = ["Likes", "Comments", "Shares", "Views", "Saves", "Follower_Count",
            "Engagement_Rate", "Hashtag_Count", "Content_Length", "Hour_of_Day"]
corr = df[num_cols].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap - Numeric Features")
plt.tight_layout()
plt.savefig(OUT + "05_correlation_heatmap.png", dpi=150)
plt.close()

print("5. CORRELATION HEATMAP - key pairs")
print("Likes vs Comments:", round(corr.loc["Likes", "Comments"], 3))
print("Shares vs Likes:", round(corr.loc["Shares", "Likes"], 3))
print("Saves vs Shares:", round(corr.loc["Saves", "Shares"], 3))
print("Comments vs Engagement_Rate:", round(corr.loc["Comments", "Engagement_Rate"], 3))
print("Follower_Count vs Engagement_Rate:", round(corr.loc["Follower_Count", "Engagement_Rate"], 3))
print()

# -------------------------------------------------------------------
# 6. Top-performing content
# -------------------------------------------------------------------
top10 = df.sort_values("Engagement_Rate", ascending=False).head(10)[
    ["Post_ID", "Platform", "Content_Type", "Category", "Engagement_Rate", "Views", "Likes"]
]
print("6. TOP 10 POSTS BY ENGAGEMENT RATE")
print(top10.to_string(index=False))
print()

plt.figure(figsize=(9, 5))
top10_plot = top10.sort_values("Engagement_Rate")
plt.barh(top10_plot["Post_ID"], top10_plot["Engagement_Rate"], color="#8172B2")
plt.xlabel("Engagement Rate (%)")
plt.title("Top 10 Posts by Engagement Rate")
plt.tight_layout()
plt.savefig(OUT + "06_top_performing_content.png", dpi=150)
plt.close()

# -------------------------------------------------------------------
# 7. Save-to-Share and Comment-to-Like analysis
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(df["Save_to_Share_Ratio"].clip(upper=df["Save_to_Share_Ratio"].quantile(0.95)),
             bins=50, ax=axes[0], color="#DD8452")
axes[0].set_title("Save-to-Share Ratio Distribution (95th pct clipped)")

sns.histplot(df["Comment_to_Like_Ratio"].clip(upper=df["Comment_to_Like_Ratio"].quantile(0.95)),
             bins=50, ax=axes[1], color="#4C72B0")
axes[1].set_title("Comment-to-Like Ratio Distribution (95th pct clipped)")
plt.tight_layout()
plt.savefig(OUT + "07_save_share_comment_like_ratios.png", dpi=150)
plt.close()

print("7. SAVE-TO-SHARE / COMMENT-TO-LIKE RATIOS")
print(df[["Save_to_Share_Ratio", "Comment_to_Like_Ratio"]].describe())

print("\nAll 7 EDA visualizations saved to visualizations/")
