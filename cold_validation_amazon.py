import os
import numpy as np
import pandas as pd
import torch

from data_loader_amazon import load_amazon_data


# ============================================================
# 1. PATHS
# ============================================================

PROCESSED_DIR = "processed"

VAL_PATH = os.path.join(
    PROCESSED_DIR,
    "amazon_cold_val.csv"
)

COLD_VAL_ITEMS_PATH = os.path.join(
    PROCESSED_DIR,
    "amazon_cold_val_items.npy"
)

CONTENT_PATH = os.path.join(
    PROCESSED_DIR,
    "amazon_content_embeddings.npy"
)

CONTENT_ASINS_PATH = os.path.join(
    PROCESSED_DIR,
    "amazon_content_asins.npy"
)


# ============================================================
# 2. LOAD COLD-VALIDATION FEATURES
# ============================================================

def load_cold_validation_features():

    cold_val_items = np.load(
        COLD_VAL_ITEMS_PATH,
        allow_pickle=True
    )

    content_embeddings = np.load(
        CONTENT_PATH
    )

    content_asins = np.load(
        CONTENT_ASINS_PATH,
        allow_pickle=True
    )

    # ASIN -> row in the full content embedding matrix
    asin_to_content_idx = {
        asin: idx
        for idx, asin in enumerate(content_asins)
    }

    # Find the content row for every cold-validation book
    cold_rows = [
        asin_to_content_idx[asin]
        for asin in cold_val_items
    ]

    # Extract their 384-D content features
    cold_features = content_embeddings[
        cold_rows
    ]

    cold_features = torch.tensor(
        cold_features,
        dtype=torch.float32
    )

    return cold_val_items, cold_features


# ============================================================
# 3. LOAD COLD-VALIDATION GROUND TRUTH
# ============================================================

def load_cold_validation_ground_truth(
    user_to_idx
):

    val_interactions = pd.read_csv(
        VAL_PATH
    )

    ground_truth = {}

    for row in val_interactions.itertuples(
        index=False
    ):

        user_id = row.user_id
        asin = row.asin

        # Convert original Amazon user ID
        # to the model's integer user index
        user_index = user_to_idx[user_id]

        if user_index not in ground_truth:
            ground_truth[user_index] = set()

        ground_truth[user_index].add(
            asin
        )

    return ground_truth

# ============================================================
# 4. RECALL@K
# ============================================================

def recall_at_k(
    ranked_asins,
    user_indices,
    ground_truth,
    k
):

    recalls = []

    for row, user_index in enumerate(user_indices):

        true_items = ground_truth[user_index]

        recommended_items = set(
            ranked_asins[row, :k]
        )

        hits = len(
            recommended_items.intersection(
                true_items
            )
        )

        recall = hits / len(true_items)

        recalls.append(recall)

    return np.mean(recalls)


# ============================================================
# 5. NDCG@K
# ============================================================

def ndcg_at_k(
    ranked_asins,
    user_indices,
    ground_truth,
    k
):

    ndcg_scores = []

    for row, user_index in enumerate(user_indices):

        true_items = ground_truth[user_index]

        recommended_items = ranked_asins[
            row,
            :k
        ]

        # ----------------------------------------------------
        # DCG
        # ----------------------------------------------------

        dcg = 0.0

        for rank, asin in enumerate(
            recommended_items,
            start=1
        ):

            if asin in true_items:

                dcg += (
                    1.0
                    / np.log2(rank + 1)
                )

        # ----------------------------------------------------
        # Ideal DCG
        # ----------------------------------------------------

        ideal_hits = min(
            len(true_items),
            k
        )

        idcg = 0.0

        for rank in range(
            1,
            ideal_hits + 1
        ):

            idcg += (
                1.0
                / np.log2(rank + 1)
            )

        ndcg = dcg / idcg

        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores)


# ============================================================
# 6. GRAPHSAGE COLD-VALIDATION
# ============================================================

def evaluate_graphsage_amazon(
    model,
    batch_size=512
):

    # --------------------------------------------------------
    # Load warm training data
    # --------------------------------------------------------

    training_data = load_amazon_data()

    warm_item_features = training_data[
        "warm_item_features"
    ]

    edge_index = training_data[
        "edge_index"
    ]

    user_to_idx = training_data[
        "user_to_idx"
    ]

    # --------------------------------------------------------
    # Load cold-validation candidates + 384-D features
    # --------------------------------------------------------

    cold_val_items, cold_features = (
        load_cold_validation_features()
    )

    # --------------------------------------------------------
    # Load validation ground truth
    # --------------------------------------------------------

    ground_truth = (
        load_cold_validation_ground_truth(
            user_to_idx
        )
    )

    validation_users = np.array(
        list(ground_truth.keys()),
        dtype=np.int64
    )



    # --------------------------------------------------------
    # Generate final representations
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        # Final user representations from the warm graph
        user_z, _ = model(
            warm_item_features,
            edge_index
        )

        # Strict cold-validation item representations
        #
        # These items have content features,
        # but ZERO training neighbors.
        cold_z = model.encode_cold_items(
            cold_features
        )

           # ----------------------------------------------------
        # Rank cold-validation candidates in user batches
        # ----------------------------------------------------

        all_ranked_asins = []
        all_user_indices = []

        for start in range(
            0,
            len(validation_users),
            batch_size
        ):

            end = min(
                start + batch_size,
                len(validation_users)
            )

            batch_users = validation_users[
                start:end
            ]

            batch_users_tensor = torch.tensor(
                batch_users,
                dtype=torch.long
            )

            # Representations for this batch of users
            batch_user_z = user_z[
                batch_users_tensor
            ]

            # Score each user against ALL cold-validation books
            scores = torch.matmul(
                batch_user_z,
                cold_z.T
            )

            # We eventually report up to @20,
            # so only the top 20 positions are needed.
            top_indices = torch.topk(
                scores,
                k=20,
                dim=1
            ).indices

            # Candidate positions -> original Amazon ASINs
            ranked_asins = cold_val_items[
                top_indices.cpu().numpy()
            ]

            all_ranked_asins.append(
                ranked_asins
            )

            all_user_indices.extend(
                batch_users.tolist()
            )     
    # --------------------------------------------------------
    # Combine all validation batches
    # --------------------------------------------------------

    all_ranked_asins = np.concatenate(
        all_ranked_asins,
        axis=0
    )

    all_user_indices = np.array(
        all_user_indices,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Validation metrics
    # --------------------------------------------------------

    recall_5 = recall_at_k(
        all_ranked_asins,
        all_user_indices,
        ground_truth,
        k=5
    )

    ndcg_5 = ndcg_at_k(
        all_ranked_asins,
        all_user_indices,
        ground_truth,
        k=5
    )

    recall_10 = recall_at_k(
        all_ranked_asins,
        all_user_indices,
        ground_truth,
        k=10
    )

    ndcg_10 = ndcg_at_k(
        all_ranked_asins,
        all_user_indices,
        ground_truth,
        k=10
    )

    recall_20 = recall_at_k(
        all_ranked_asins,
        all_user_indices,
        ground_truth,
        k=20
    )

    ndcg_20 = ndcg_at_k(
        all_ranked_asins,
        all_user_indices,
        ground_truth,
        k=20
    )

    return {
        "Recall@5": recall_5,
        "NDCG@5": ndcg_5,
        "Recall@10": recall_10,
        "NDCG@10": ndcg_10,
        "Recall@20": recall_20,
        "NDCG@20": ndcg_20
    }