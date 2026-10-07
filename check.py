import os
import pandas as pd

# Original FULL Amazon interaction data
INTERACTIONS_PATH = os.path.join(
    "data",
    "amazon_books_10core.csv"
)

interactions = pd.read_csv(INTERACTIONS_PATH)

# Count interactions per item
item_counts = interactions.groupby("asin").size()

print("========== AMAZON BOOKS DATASET CHECK ==========")

print("Total interactions:", len(interactions))
print("Total items:", len(item_counts))
print("Total users:", interactions["user_id"].nunique())

print("\n========== INTERACTIONS PER ITEM ==========")

print(
    "Items >= 10 interactions:",
    (item_counts >= 10).sum()
)

print(
    "Items < 10 interactions:",
    (item_counts < 10).sum()
)

print(
    "Mean interactions/item:",
    item_counts.mean()
)

print(
    "Median interactions/item:",
    item_counts.median()
)

print(
    "Minimum interactions/item:",
    item_counts.min()
)

print(
    "Maximum interactions/item:",
    item_counts.max()
)

print("\n========== 10-CORE CHECK ==========")

if item_counts.min() >= 10:
    print("YES: Every item has at least 10 interactions.")
else:
    print("NO: Some items have fewer than 10 interactions.")

# Number of cold items under your 10% / 10% protocol
n_items = len(item_counts)

n_val = int(0.10 * n_items)
n_test = int(0.10 * n_items)

print("\n========== YOUR COLD-ITEM SPLIT ==========")

print("Total items:", n_items)
print("Cold validation items:", n_val)
print("Cold test items:", n_test)
print("Warm items:", n_items - n_val - n_test)