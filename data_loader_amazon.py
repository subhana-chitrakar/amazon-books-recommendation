import os
import numpy as np
import pandas as pd
import torch


# ============================================================
# 1. PATHS
# ============================================================

PROCESSED_DIR = "processed"

TRAIN_PATH = os.path.join(
    PROCESSED_DIR,
    "amazon_train.csv"
)

CONTENT_PATH = os.path.join(
    PROCESSED_DIR,
    "amazon_content_embeddings.npy"
)

CONTENT_ASINS_PATH = os.path.join(
    PROCESSED_DIR,
    "amazon_content_asins.npy"
)

WARM_ITEMS_PATH = os.path.join(
    PROCESSED_DIR,
    "amazon_warm_items.npy"
)


# ============================================================
# 2. LOAD SAVED AMAZON DATA
# ============================================================

train_interactions = pd.read_csv(
    TRAIN_PATH
)

content_embeddings = np.load(
    CONTENT_PATH
)

content_asins = np.load(
    CONTENT_ASINS_PATH,
    allow_pickle=True
)

warm_items = np.load(
    WARM_ITEMS_PATH,
    allow_pickle=True
)

print("Training interactions:", train_interactions.shape)
print("All content embeddings:", content_embeddings.shape)
print("Content ASINs:", content_asins.shape)
print("Warm items:", warm_items.shape)


# ============================================================
# 3. ALIGN WARM ITEMS WITH CONTENT FEATURES
# ============================================================

asin_to_content_idx = {
    asin: idx
    for idx, asin in enumerate(content_asins)
}

warm_content_indices = [
    asin_to_content_idx[asin]
    for asin in warm_items
]

warm_item_features = content_embeddings[
    warm_content_indices
]

print("\nWarm feature alignment:")
print("Warm items:", len(warm_items))
print("Warm item feature shape:", warm_item_features.shape)

print(
    "Expected feature dimension:",
    content_embeddings.shape[1]
)


# ============================================================
# 4. CREATE USER AND WARM-ITEM INTEGER INDICES
# ============================================================

# Unique users appearing in training
train_users = train_interactions[
    "user_id"
].unique()

# user_id -> integer index
user_to_idx = {
    user_id: idx
    for idx, user_id in enumerate(train_users)
}

# warm ASIN -> integer index
warm_item_to_idx = {
    asin: idx
    for idx, asin in enumerate(warm_items)
}

print("\nInteger indexing:")
print("Number of training users:", len(user_to_idx))
print("Number of warm items:", len(warm_item_to_idx))

# ============================================================
# 5. BUILD TRAINING EDGE INDEX
# ============================================================

user_indices = train_interactions[
    "user_id"
].map(user_to_idx).to_numpy()

item_indices = train_interactions[
    "asin"
].map(warm_item_to_idx).to_numpy()

edge_index = torch.tensor(
    np.vstack([
        user_indices,
        item_indices
    ]),
    dtype=torch.long
)

print("\nTraining graph:")
print("Edge index shape:", edge_index.shape)
print("Number of training edges:", edge_index.shape[1])

print(
    "User index range:",
    edge_index[0].min().item(),
    "to",
    edge_index[0].max().item()
)

print(
    "Item index range:",
    edge_index[1].min().item(),
    "to",
    edge_index[1].max().item()
)

# ============================================================
# 6. PACKAGE DATA FOR GRAPHSAGE
# ============================================================

def load_amazon_data():

    warm_features_tensor = torch.tensor(
        warm_item_features,
        dtype=torch.float32
    )

    return {
        "warm_item_features": warm_features_tensor,
        "edge_index": edge_index,
        "num_users": len(user_to_idx),
        "num_warm_items": len(warm_item_to_idx),
        "user_to_idx": user_to_idx,
        "warm_item_to_idx": warm_item_to_idx
    }

# ============================================================
# 6. PACKAGE DATA FOR GRAPHSAGE
# ============================================================

def load_amazon_data():

    warm_features_tensor = torch.tensor(
        warm_item_features,
        dtype=torch.float32
    )

    return {
        "warm_item_features": warm_features_tensor,
        "edge_index": edge_index,
        "num_users": len(user_to_idx),
        "num_warm_items": len(warm_item_to_idx),
        "user_to_idx": user_to_idx,
        "warm_item_to_idx": warm_item_to_idx
    }