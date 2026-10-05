import torch
import torch.optim as optim
import pandas as pd

from data_loader_amazon import load_amazon_data
from graphsage_xsimgcl_amazon import GraphSAGEXSimGCLAmazon
from cold_validation_amazon import evaluate_graphsage_amazon


# ============================================================
# Reproducibility and hyperparameters
# ============================================================

SEED = 42

LEARNING_RATE = 0.001
EPS = 0.2
EPOCHS = 30

BPR_CHUNK_SIZE = 65536

CHECKPOINT_PATH = "best_graphsage_perturbation_amazon.pt"


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    These keys are used to ensure that sampled negatives
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
# MEMORY-EFFICIENT BPR GRADIENT CALCULATION
# ============================================================

def compute_bpr_embedding_gradients(
    model,
    user_embeddings,
    item_embeddings,
    user_indices,
    positive_item_indices,
    negative_item_indices,
    chunk_size=65536
):
    """
    Compute BPR gradients with respect to the final
    perturbed user/item embeddings.

    The GraphSAGE computation graph is detached here so
    that large BPR batches do not retain the entire
    GraphSAGE graph.
    """

    user_leaf = (
        user_embeddings
        .detach()
        .requires_grad_(True)
    )

    item_leaf = (
        item_embeddings
        .detach()
        .requires_grad_(True)
    )

    total_loss = 0.0

    num_interactions = len(user_indices)

    for start in range(
        0,
        num_interactions,
        chunk_size
    ):

        end = min(
            start + chunk_size,
            num_interactions
        )

        batch_loss = model.bpr_loss(
            user_leaf,
            item_leaf,
            user_indices[start:end],
            positive_item_indices[start:end],
            negative_item_indices[start:end]
        )

        weight = (
            (end - start)
            / num_interactions
        )

        (
            batch_loss * weight
        ).backward()

        total_loss += (
            batch_loss.detach().item()
            * weight
        )

    return (
        total_loss,
        user_leaf.grad.detach(),
        item_leaf.grad.detach()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    # --------------------------------------------------------
    # Load Amazon data
    # --------------------------------------------------------

    data = load_amazon_data()

    warm_item_features = data["warm_item_features"]
    edge_index = data["edge_index"]

    num_users = data["num_users"]
    num_warm_items = data["num_warm_items"]

    # --------------------------------------------------------
    # Load positive warm training interactions
    # --------------------------------------------------------

    train_interactions = pd.read_csv(
        "processed/amazon_train.csv"
    )

    user_indices = torch.tensor(
        train_interactions["user_id"]
        .map(data["user_to_idx"])
        .to_numpy(),
        dtype=torch.long
    )

    positive_item_indices = torch.tensor(
        train_interactions["asin"]
        .map(data["warm_item_to_idx"])
        .to_numpy(),
        dtype=torch.long
    )

    print(
        "Positive interactions:",
        len(user_indices)
    )

    # --------------------------------------------------------
    # Build observed warm interaction lookup
    # --------------------------------------------------------

    observed_pair_keys = build_observed_pair_keys(
        user_indices,
        positive_item_indices,
        num_warm_items
    )

    print(
        "Observed warm pairs:",
        len(observed_pair_keys)
    )

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print("Users:", num_users)
    print("Warm items:", num_warm_items)
    print("Training edges:", edge_index.shape[1])
    print(
        "Item feature shape:",
        warm_item_features.shape
    )

    # --------------------------------------------------------
    # Create GraphSAGE + perturbation model
    # --------------------------------------------------------

    model = GraphSAGEXSimGCLAmazon(
        num_users=num_users,
        item_input_dim=384,
        hidden_dim=64,
        eps=EPS
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(
        "Perturbation-only model initialized."
    )

    # --------------------------------------------------------
    # Best validation checkpoint
    # --------------------------------------------------------

    best_ndcg10 = -1.0
    best_epoch = -1

    print("Model ready for training.")

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        model.train()

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Fresh unobserved warm negatives each epoch
        # ----------------------------------------------------

        negative_item_indices = sample_negative_items(
            user_indices,
            num_warm_items,
            observed_pair_keys
        )

        # ----------------------------------------------------
        # Perturbed two-layer GraphSAGE embeddings
        #
        # IMPORTANT:
        # We keep the perturbation.
        # We simply do NOT calculate contrastive loss.
        # ----------------------------------------------------

        (
            user_h1_tilde,
            item_h1_tilde,
            user_h2_tilde,
            item_h2_tilde,
        ) = model.forward_perturbed(
            warm_item_features,
            edge_index
        )

        print(
            "Training forward pass complete."
        )

        # ----------------------------------------------------
        # BPR loss and gradients
        #
        # BPR is calculated using the perturbed final
        # embeddings.
        # ----------------------------------------------------

        (
            bpr_loss,
            bpr_user_grad,
            bpr_item_grad
        ) = compute_bpr_embedding_gradients(
            model,
            user_h2_tilde,
            item_h2_tilde,
            user_indices,
            positive_item_indices,
            negative_item_indices,
            chunk_size=BPR_CHUNK_SIZE,
        )

        print(
            "BPR loss:",
            bpr_loss
        )

        if not torch.isfinite(
            torch.tensor(bpr_loss)
        ).item():

            raise ValueError(
                "BPR loss is NaN or infinite"
            )

        # ----------------------------------------------------
        # PERTURBATION-ONLY BACKWARD PASS
        #
        # NO contrastive loss.
        #
        # Only BPR gradients are propagated through the
        # perturbed GraphSAGE embeddings.
        # ----------------------------------------------------

        torch.autograd.backward(
            tensors=[
                user_h2_tilde,
                item_h2_tilde,
            ],
            grad_tensors=[
                bpr_user_grad,
                bpr_item_grad,
            ],
        )

        print(
            "Perturbation-only backward pass complete."
        )

        # ----------------------------------------------------
        # Optimizer update
        # ----------------------------------------------------

        optimizer.step()

        print(
            "Optimizer update complete."
        )

        # ----------------------------------------------------
        # Cold validation
        #
        # IMPORTANT:
        # We select the checkpoint using cold validation only.
        # Cold test is never used here.
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_metrics = evaluate_graphsage_amazon(
                model
            )

        print(
            "Validation metrics:",
            val_metrics
        )

        current_ndcg10 = float(
            val_metrics["NDCG@10"]
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if current_ndcg10 > best_ndcg10:

            best_ndcg10 = current_ndcg10

            best_epoch = epoch

            torch.save(
                model.state_dict(),
                CHECKPOINT_PATH
            )

            print(
                f"New best checkpoint: "
                f"epoch {best_epoch}, "
                f"NDCG@10 = {best_ndcg10:.6f}"
            )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(
        f"\nBest epoch: {best_epoch}, "
        f"best validation NDCG@10: "
        f"{best_ndcg10:.6f}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()