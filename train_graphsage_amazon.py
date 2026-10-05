import torch
import pandas as pd

from data_loader_amazon import load_amazon_data
from graphsage_amazon import GraphSAGEAmazon
from cold_validation_amazon import evaluate_graphsage_amazon


# ============================================================
# FAST BPR NEGATIVE SAMPLING
# ============================================================

def build_observed_pair_keys(
    user_indices,
    positive_item_indices,
    num_warm_items
):
    """
    Encode every observed warm (user, item) training pair
    as one integer key.

    These keys are used to make sure that sampled negatives
    are truly unobserved warm items for that user.
    """

    pair_keys = (
        user_indices * num_warm_items
        + positive_item_indices
    )

    observed_pair_keys = torch.unique(
        pair_keys,
        sorted=True
    )

    return observed_pair_keys


def sample_negative_items(
    user_indices,
    num_warm_items,
    observed_pair_keys
):
    """
    Sample one unobserved WARM negative item for every
    positive training interaction.

    Cold-validation and cold-test items cannot be sampled
    because item indices range only over warm items.
    """

    negative_items = torch.randint(
        low=0,
        high=num_warm_items,
        size=user_indices.shape,
        dtype=torch.long
    )

    while True:

        candidate_keys = (
            user_indices * num_warm_items
            + negative_items
        )

        positions = torch.searchsorted(
            observed_pair_keys,
            candidate_keys
        )

        safe_positions = torch.clamp(
            positions,
            max=len(observed_pair_keys) - 1
        )

        invalid = (
            observed_pair_keys[safe_positions]
            == candidate_keys
        )

        if not invalid.any():
            break

        num_invalid = int(
            invalid.sum().item()
        )

        negative_items[invalid] = torch.randint(
            low=0,
            high=num_warm_items,
            size=(num_invalid,),
            dtype=torch.long
        )

    return negative_items


# ============================================================
# BPR LOSS
# ============================================================

def bpr_loss(
    user_embeddings,
    positive_item_embeddings,
    negative_item_embeddings
):

    positive_scores = torch.sum(
        user_embeddings
        * positive_item_embeddings,
        dim=1
    )

    negative_scores = torch.sum(
        user_embeddings
        * negative_item_embeddings,
        dim=1
    )

    loss = -torch.log(
        torch.sigmoid(
            positive_scores
            - negative_scores
        )
        + 1e-8
    ).mean()

    return loss


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

SEED = 42
LEARNING_RATE = 0.001

# Full Amazon GraphSAGE training.
# Best checkpoint is selected using cold-validation NDCG@10.
NUM_EPOCHS = 30

BPR_BATCH_SIZE = 65536

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# LOAD WARM TRAINING GRAPH
# ============================================================

data = load_amazon_data()

warm_item_features = data[
    "warm_item_features"
]

edge_index = data[
    "edge_index"
]

num_users = data[
    "num_users"
]

num_warm_items = data[
    "num_warm_items"
]

user_to_idx = data[
    "user_to_idx"
]

warm_item_to_idx = data[
    "warm_item_to_idx"
]


# ============================================================
# LOAD ALL WARM TRAINING INTERACTIONS
# ============================================================

train_interactions = pd.read_csv(
    "processed/amazon_train.csv"
)

user_indices = torch.tensor(
    train_interactions["user_id"]
    .map(user_to_idx)
    .to_numpy(),
    dtype=torch.long
)

positive_item_indices = torch.tensor(
    train_interactions["asin"]
    .map(warm_item_to_idx)
    .to_numpy(),
    dtype=torch.long
)


# ============================================================
# BUILD OBSERVED USER-ITEM LOOKUP
# ============================================================

# IMPORTANT:
# This is built from ALL warm training interactions.
#
# A sampled negative is rejected if that user already
# interacted with the item in training.

observed_pair_keys = build_observed_pair_keys(
    user_indices,
    positive_item_indices,
    num_warm_items
)


# ============================================================
# DATA CHECK
# ============================================================

print("\nAmazon GraphSAGE training data:")
print("Users:", num_users)
print("Warm items:", num_warm_items)

print(
    "Training interactions:",
    len(user_indices)
)

print(
    "Observed warm user-item pairs:",
    len(observed_pair_keys)
)

print(
    "Warm item features:",
    warm_item_features.shape
)

print(
    "Training graph:",
    edge_index.shape
)


# ============================================================
# CREATE MODEL
# ============================================================

model = GraphSAGEAmazon(
    num_users=num_users,
    item_input_dim=384,
    hidden_dim=64
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# BEST VALIDATION CHECKPOINT
# ============================================================

best_val_ndcg10 = -1.0
best_epoch = -1

CHECKPOINT_PATH = (
    "best_graphsage_amazon.pt"
)


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    print(
        f"\nEpoch {epoch}/{NUM_EPOCHS}"
    )

    model.train()

    optimizer.zero_grad()

    # --------------------------------------------------------
    # 1. Sample one unobserved warm negative
    #    for every positive training interaction
    # --------------------------------------------------------

    negative_item_indices = (
        sample_negative_items(
            user_indices,
            num_warm_items,
            observed_pair_keys
        )
    )

    print(
        "Negative sampling complete."
    )

    # --------------------------------------------------------
    # 2. Full warm-graph GraphSAGE propagation
    # --------------------------------------------------------

    user_z, item_z = model(
        warm_item_features,
        edge_index
    )

    print(
        "GraphSAGE propagation complete."
    )

    # --------------------------------------------------------
    # 3. Compute BPR loss in chunks
    # --------------------------------------------------------

    total_loss = 0.0

    for start in range(
        0,
        len(user_indices),
        BPR_BATCH_SIZE
    ):

        end = min(
            start + BPR_BATCH_SIZE,
            len(user_indices)
        )

        batch_users = (
            user_indices[start:end]
        )

        batch_positive_items = (
            positive_item_indices[start:end]
        )

        batch_negative_items = (
            negative_item_indices[start:end]
        )

        batch_user_z = (
            user_z[batch_users]
        )

        batch_positive_z = (
            item_z[batch_positive_items]
        )

        batch_negative_z = (
            item_z[batch_negative_items]
        )

        batch_loss = bpr_loss(
            batch_user_z,
            batch_positive_z,
            batch_negative_z
        )

        total_loss = (
            total_loss
            + batch_loss
            * len(batch_users)
        )

    loss = (
        total_loss
        / len(user_indices)
    )

    # --------------------------------------------------------
    # 4. Backpropagation
    # --------------------------------------------------------

    loss.backward()

    optimizer.step()

    print(
        f"Training loss: "
        f"{loss.item():.6f}"
    )

    # --------------------------------------------------------
    # 5. COLD-VALIDATION ONLY
    # --------------------------------------------------------

    val_metrics = (
        evaluate_graphsage_amazon(
            model
        )
    )

    val_ndcg10 = (
        val_metrics["NDCG@10"]
    )

    print(
        "\nCold-validation metrics:"
    )

    for (
        metric_name,
        metric_value
    ) in val_metrics.items():

        print(
            f"{metric_name}: "
            f"{metric_value:.6f}"
        )

    # --------------------------------------------------------
    # 6. Checkpoint selection using VALIDATION NDCG@10
    # --------------------------------------------------------

    if (
        val_ndcg10
        > best_val_ndcg10
    ):

        best_val_ndcg10 = (
            val_ndcg10
        )

        best_epoch = epoch

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )

        print(
            f"New best checkpoint saved "
            f"(validation NDCG@10 = "
            f"{best_val_ndcg10:.6f})"
        )


# ============================================================
# FINISHED
# ============================================================

print(
    "\nTraining finished."
)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation NDCG@10: "
    f"{best_val_ndcg10:.6f}"
)

print(
    f"Best checkpoint: "
    f"{CHECKPOINT_PATH}"
)