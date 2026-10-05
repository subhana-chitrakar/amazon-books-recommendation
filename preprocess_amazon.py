import os
import ast
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
# ============================================================
# 1. FILE PATHS
# ============================================================

DATA_DIR = "data"

INTERACTIONS_PATH = os.path.join(
    DATA_DIR,
    "amazon_books_10core.csv"
)

METADATA_PATH = os.path.join(
    DATA_DIR,
    "amazon_books_10core_metadata.csv"
)

ITEMS_PATH = os.path.join(
    DATA_DIR,
    "amazon_10core_items.txt"
)


# ============================================================
# 2. LOAD INTERACTIONS
# ============================================================

interactions = pd.read_csv(INTERACTIONS_PATH)

print("Interaction shape:", interactions.shape)
print("Interaction columns:", interactions.columns.tolist())
print(interactions.head())


# ============================================================
# 3. LOAD METADATA
# ============================================================

metadata = pd.read_csv(METADATA_PATH)

print("\nMetadata shape:", metadata.shape)
print("Metadata columns:", metadata.columns.tolist())
print(metadata.head())


# ============================================================
# 4. CHECK MISSING METADATA
# ============================================================

print("\nMissing metadata values:")
print(metadata.isnull().sum())


# ============================================================
# 5. INSPECT RAW CATEGORIES
# ============================================================

print("\nFirst 10 category values:")
print(metadata["categories"].head(10).to_string(index=False))

books_only_count = (
    metadata["categories"].astype(str).str.strip()
    == "[['Books']]"
).sum()

print("\nBooks with only [['Books']] category:")
print(books_only_count)

print("\nPercentage:")
print(books_only_count / len(metadata) * 100)


# ============================================================
# 6. INSPECT TITLE AND DESCRIPTION
# ============================================================

print("\nSample titles and descriptions:")

sample_text = metadata[
    metadata["title"].notna()
    & metadata["description"].notna()
][["title", "description"]].head(5)

for _, row in sample_text.iterrows():

    print("\nTITLE:")
    print(row["title"])

    print("DESCRIPTION:")
    print(str(row["description"])[:300])


# ============================================================
# 7. INSPECT PRICE
# ============================================================

print("\nPrice statistics:")
print(metadata["price"].describe())

print("\nLowest 5 prices:")
print(metadata["price"].nsmallest(5).tolist())

print("\nHighest 5 prices:")
print(metadata["price"].nlargest(5).tolist())


# ============================================================
# 8. CHECK TITLE + DESCRIPTION COVERAGE
# ============================================================

missing_both = (
    metadata["title"].isnull()
    & metadata["description"].isnull()
).sum()

print("\nBooks missing BOTH title and description:")
print(missing_both)

print("\nPercentage missing both:")
print(missing_both / len(metadata) * 100)


# ============================================================
# 9. VERIFY INTERACTION AND METADATA ITEMS MATCH
# ============================================================

interaction_items = set(interactions["asin"])
metadata_items = set(metadata["asin"])

print("\nUnique books in interactions:", len(interaction_items))
print("Unique books in metadata:", len(metadata_items))

print(
    "Interaction books missing from metadata:",
    len(interaction_items - metadata_items)
)

print(
    "Metadata books not in interactions:",
    len(metadata_items - interaction_items)
)


# ============================================================
# 10. VERIFY ITEM-LIST FILE
# ============================================================

with open(ITEMS_PATH, "r", encoding="utf-8") as f:
    listed_items = {
        line.strip()
        for line in f
        if line.strip()
    }

print("\nUnique books in item list:", len(listed_items))

print(
    "Interaction books missing from item list:",
    len(interaction_items - listed_items)
)

print(
    "Item-list books not in interactions:",
    len(listed_items - interaction_items)
)


# ============================================================
# 11. DATASET SUMMARY
# ============================================================

num_users = interactions["user_id"].nunique()
num_items = interactions["asin"].nunique()
num_interactions = len(interactions)

print("\nDataset summary:")
print("Users:", num_users)
print("Items:", num_items)
print("Interactions:", num_interactions)


# ============================================================
# 12. CHECK BOOKS WITH VERY WEAK METADATA
# ============================================================

weak_metadata = 0

for _, row in metadata.iterrows():

    has_title = (
        pd.notna(row["title"])
        and str(row["title"]).strip() != ""
    )

    has_description = (
        pd.notna(row["description"])
        and str(row["description"]).strip() != ""
    )

    try:
        category_paths = ast.literal_eval(
            row["categories"]
        )

        book_categories = set()

        for path in category_paths:
            for category in path:
                book_categories.add(category)

    except Exception:
        book_categories = set()

    if (
        not has_title
        and not has_description
        and book_categories == {"Books"}
    ):
        weak_metadata += 1


print("\nWeak-metadata books:")
print(weak_metadata)

print("\nWeak-metadata percentage:")
print(
    weak_metadata / len(metadata) * 100
)


# ============================================================
# 13. CLEAN TITLE AND DESCRIPTION
# ============================================================

metadata["title_clean"] = (
    metadata["title"]
    .fillna("")
    .astype(str)
)

metadata["description_clean"] = (
    metadata["description"]
    .fillna("")
    .astype(str)
)

print("\nClean text check:")

print(
    "Missing clean titles:",
    metadata["title_clean"].isna().sum()
)

print(
    "Missing clean descriptions:",
    metadata["description_clean"].isna().sum()
)


# ============================================================
# 14. COMBINE TITLE + DESCRIPTION
# ============================================================

metadata["text"] = (
    metadata["title_clean"].str.strip()
    + ". "
    + metadata["description_clean"].str.strip()
).str.strip()

print("\nCombined text examples:")

for text in metadata["text"].head(3):
    print("\n", text[:300])


# ============================================================
# 15. EXTRACT INFORMATIVE CATEGORIES
# ============================================================

def extract_informative_categories(value):

    try:
        category_paths = ast.literal_eval(value)

    except Exception:
        return ""

    categories = set()

    for path in category_paths:

        for category in path:

            category = str(category).strip()

            # "Books" alone is too generic to distinguish items.
            if category and category != "Books":
                categories.add(category)

    return " | ".join(sorted(categories))


metadata["category_text"] = (
    metadata["categories"].apply(
        extract_informative_categories
    )
)


# ============================================================
# 16. INSPECT INFORMATIVE CATEGORIES
# ============================================================

print("\nInformative category examples:")

examples = metadata[
    metadata["category_text"] != ""
][
    ["asin", "category_text"]
].head(10)

print(examples.to_string(index=False))


def build_content_text(row):
    parts = []

    title = row["title_clean"].strip()
    description = row["description_clean"].strip()
    categories = row["category_text"].strip()

    if title:
        parts.append(f"Title: {title}")

    if categories:
        parts.append(f"Categories: {categories}")

    if description:
        parts.append(f"Description: {description}")

    return ". ".join(parts)


metadata["content_text"] = metadata.apply(
    build_content_text,
    axis=1
)

print("\nFinal content-text examples:")

for text in metadata["content_text"].head(3):
    print("\n", text[:400])

print(
    "\nBooks with completely empty content:",
    (metadata["content_text"] == "").sum()
)

# ============================================================
# 17. LOAD TEXT EMBEDDING MODEL
# ============================================================

print("\nLoading SentenceTransformer model...")

text_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Model loaded.")
print("Maximum sequence length:", text_model.max_seq_length)
# ============================================================
# 18. TEST TEXT ENCODING
# ============================================================

# test_texts = metadata["content_text"].head(3).tolist()

# test_embeddings = text_model.encode(
#     test_texts,
#     convert_to_numpy=True
# )

# print("\nTest embedding shape:")
# print(test_embeddings.shape)



# ============================================================
# 19. ENCODE ALL AMAZON BOOK CONTENT
# ============================================================

# print("\nEncoding all book content...")

# content_embeddings = text_model.encode(
#     metadata["content_text"].tolist(),
#     batch_size=64,
#     show_progress_bar=True,
#     convert_to_numpy=True
# )

# print("\nContent embedding shape:")
# print(content_embeddings.shape)


# # Give books with no usable metadata a zero vector
# empty_content_mask = (
#     metadata["content_text"] == ""
# ).to_numpy()

# content_embeddings[empty_content_mask] = 0.0

# print(
#     "Zero vectors assigned to empty-content books:",
#     empty_content_mask.sum()
# )


# ============================================================
# 20. SAVE CONTENT FEATURES
# ============================================================

# PROCESSED_DIR = "processed"

# os.makedirs(
#     PROCESSED_DIR,
#     exist_ok=True
# )

# np.save(
#     os.path.join(
#         PROCESSED_DIR,
#         "amazon_content_embeddings.npy"
#     ),
#     content_embeddings
# )

# np.save(
#     os.path.join(
#         PROCESSED_DIR,
#         "amazon_content_asins.npy"
#     ),
#     metadata["asin"].astype(str).to_numpy()
# )

# print("\nSaved Amazon content embeddings.")


# ============================================================
# 21. VERIFY 10-CORE CONDITION
# ============================================================

item_counts = interactions.groupby("asin").size()

print("\nAmazon 10-core verification:")
print("Minimum interactions per book:", item_counts.min())
print("Maximum interactions per book:", item_counts.max())
print("Books with fewer than 10 interactions:", (item_counts < 10).sum())
print("Total books:", len(item_counts))


# ============================================================
# 22. CREATE STRICT COLD-ITEM SPLIT
# ============================================================

all_items = item_counts.index.to_numpy().copy()

rng = np.random.default_rng(seed=42)
rng.shuffle(all_items)

n_items = len(all_items)

n_val = int(0.10 * n_items)
n_test = int(0.10 * n_items)

cold_val_items = all_items[:n_val]

cold_test_items = all_items[
    n_val:n_val + n_test
]

warm_items = all_items[
    n_val + n_test:
]

print("\nCold-item split:")
print("Warm items:", len(warm_items))
print("Cold validation items:", len(cold_val_items))
print("Cold test items:", len(cold_test_items))

print(
    "Total:",
    len(warm_items)
    + len(cold_val_items)
    + len(cold_test_items)
)


# ============================================================
# 23. SPLIT INTERACTIONS BY ITEM GROUP
# ============================================================

train_interactions = interactions[
    interactions["asin"].isin(warm_items)
].copy()

val_interactions = interactions[
    interactions["asin"].isin(cold_val_items)
].copy()

test_interactions = interactions[
    interactions["asin"].isin(cold_test_items)
].copy()

print("\nInteraction split:")
print("Train interactions:", len(train_interactions))
print("Validation interactions:", len(val_interactions))
print("Test interactions:", len(test_interactions))

print(
    "Total interactions:",
    len(train_interactions)
    + len(val_interactions)
    + len(test_interactions)
)

# ============================================================
# 24. CHECK STRICT COLD-ITEM LEAKAGE
# ============================================================

train_item_set = set(train_interactions["asin"])
val_item_set = set(val_interactions["asin"])
test_item_set = set(test_interactions["asin"])

train_val_overlap = train_item_set & val_item_set
train_test_overlap = train_item_set & test_item_set
val_test_overlap = val_item_set & test_item_set

print("\nLeakage check:")
print(
    "Train / cold-validation item overlap:",
    len(train_val_overlap)
)

print(
    "Train / cold-test item overlap:",
    len(train_test_overlap)
)

print(
    "Cold-validation / cold-test item overlap:",
    len(val_test_overlap)
)


# ============================================================
# 25. CHECK VALIDATION/TEST USERS ARE KNOWN IN TRAINING
# ============================================================

train_users = set(train_interactions["user_id"])
val_users = set(val_interactions["user_id"])
test_users = set(test_interactions["user_id"])

unknown_val_users = val_users - train_users
unknown_test_users = test_users - train_users

print("\nKnown-user check:")
print("Training users:", len(train_users))
print("Validation users:", len(val_users))
print("Test users:", len(test_users))

print(
    "Validation users missing from training:",
    len(unknown_val_users)
)

print(
    "Test users missing from training:",
    len(unknown_test_users)
)


# ============================================================
# 26. SAVE STRICT COLD-ITEM SPLITS
# ============================================================

PROCESSED_DIR = "processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

train_interactions.to_csv(
    os.path.join(PROCESSED_DIR, "amazon_train.csv"),
    index=False
)

val_interactions.to_csv(
    os.path.join(PROCESSED_DIR, "amazon_cold_val.csv"),
    index=False
)

test_interactions.to_csv(
    os.path.join(PROCESSED_DIR, "amazon_cold_test.csv"),
    index=False
)

np.save(
    os.path.join(PROCESSED_DIR, "amazon_warm_items.npy"),
    warm_items
)

np.save(
    os.path.join(PROCESSED_DIR, "amazon_cold_val_items.npy"),
    cold_val_items
)

np.save(
    os.path.join(PROCESSED_DIR, "amazon_cold_test_items.npy"),
    cold_test_items
)

print("\nSaved strict cold-item splits.")

# ============================================================
# 27. VERIFY SAVED FILES
# ============================================================

expected_files = [
    "amazon_content_embeddings.npy",
    "amazon_content_asins.npy",
    "amazon_train.csv",
    "amazon_cold_val.csv",
    "amazon_cold_test.csv",
    "amazon_warm_items.npy",
    "amazon_cold_val_items.npy",
    "amazon_cold_test_items.npy",
]

print("\nSaved-file check:")

for filename in expected_files:
    path = os.path.join("processed", filename)
    print(filename, "->", os.path.exists(path))