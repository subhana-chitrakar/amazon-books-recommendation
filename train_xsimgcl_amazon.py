import torch
import torch.optim as optim
import pandas as pd
from data_loader_amazon import load_amazon_data
from graphsage_xsimgcl_amazon import GraphSAGEXSimGCLAmazon
from cold_validation_amazon import evaluate_graphsage_amazon



# -----------------------------------------------------
# Reproducibility and hyperparameters
# -----------------------------------------------------

SEED = 42

LEARNING_RATE = 0.001
EPS = 0.2
TEMPERATURE = 0.15
LAMBDA_CL = 0.01
EPOCHS = 30

BPR_CHUNK_SIZE = 65536
CL_CHUNK_SIZE = 256

CHECKPOINT_PATH = "best_graphsage_xsimgcl_amazon.pt"


# -----------------------------------------------------
# Reproducibility
# -----------------------------------------------------

def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


  
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

def compute_bpr_embedding_gradients(
    model,
    user_embeddings,
    item_embeddings,
    user_indices,
    positive_item_indices,
    negative_item_indices,
    chunk_size=65536
):
    # Detached leaves prevent BPR batches from retaining
    # the large GraphSAGE computation graph.
    user_leaf = user_embeddings.detach().requires_grad_(True)
    item_leaf = item_embeddings.detach().requires_grad_(True)

    total_loss = 0.0
    num_interactions = len(user_indices)

    for start in range(0, num_interactions, chunk_size):
        end = min(start + chunk_size, num_interactions)

        batch_loss = model.bpr_loss(
            user_leaf,
            item_leaf,
            user_indices[start:end],
            positive_item_indices[start:end],
            negative_item_indices[start:end]
        )

        weight = (end - start) / num_interactions
        (batch_loss * weight).backward()

        total_loss += batch_loss.detach().item() * weight

    return (
        total_loss,
        user_leaf.grad.detach(),
        item_leaf.grad.detach()
    )

def main():
    set_seed(SEED)

    data = load_amazon_data()

    warm_item_features = data["warm_item_features"]
    edge_index = data["edge_index"]

    num_users = data["num_users"]
    num_warm_items = data["num_warm_items"]

    # Load positive warm training interactions
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

    print("Positive interactions:", len(user_indices))

    # Build lookup of observed warm interactions
    observed_pair_keys = build_observed_pair_keys(
        user_indices,
        positive_item_indices,
        num_warm_items
    )

    print(
        "Observed warm pairs:",
        len(observed_pair_keys)
    )

    print("Users:", num_users)
    print("Warm items:", num_warm_items)
    print("Training edges:", edge_index.shape[1])
    print("Item feature shape:", warm_item_features.shape)
    # Create proposed model
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

    print("Proposed model initialized.")

    best_ndcg10 = -1.0
    best_epoch = -1
    print("Model ready for training.")
      


    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        model.train()
        optimizer.zero_grad()

        # Fresh unobserved warm negatives each epoch.
        negative_item_indices = sample_negative_items(
            user_indices, num_warm_items, observed_pair_keys
        )

        # Perturbed two-layer GraphSAGE embeddings.
        (
            user_h1_tilde,
            item_h1_tilde,
            user_h2_tilde,
            item_h2_tilde,
        ) = model.forward_perturbed(warm_item_features, edge_index)
        print("Training forward pass complete.")

        bpr_loss, bpr_user_grad, bpr_item_grad = compute_bpr_embedding_gradients(
            model,
            user_h2_tilde,
            item_h2_tilde,
            user_indices,
            positive_item_indices,
            negative_item_indices,
            chunk_size=BPR_CHUNK_SIZE,
        )
        print("BPR loss:", bpr_loss)
        if not torch.isfinite(torch.tensor(bpr_loss)).item():
            raise ValueError("BPR loss is NaN or infinite")

        user_cl_loss, user_grad_h1, user_grad_h2 = (
            model.compute_embedding_gradients(
                user_h1_tilde,
                user_h2_tilde,
                temperature=TEMPERATURE,
                chunk_size=CL_CHUNK_SIZE,
            )
        )
        print("User contrastive loss:", user_cl_loss)

        item_cl_loss, item_grad_h1, item_grad_h2 = (
            model.compute_embedding_gradients(
                item_h1_tilde,
                item_h2_tilde,
                temperature=TEMPERATURE,
                chunk_size=CL_CHUNK_SIZE,
            )
        )
        print("Item contrastive loss:", item_cl_loss)

        # Backpropagate the exact cached embedding gradients together.
        torch.autograd.backward(
            tensors=[
                user_h1_tilde,
                user_h2_tilde,
                item_h1_tilde,
                item_h2_tilde,
            ],
            grad_tensors=[
                LAMBDA_CL * user_grad_h1,
                bpr_user_grad + LAMBDA_CL * user_grad_h2,
                LAMBDA_CL * item_grad_h1,
                bpr_item_grad + LAMBDA_CL * item_grad_h2,
            ],
        )
        print("Combined backward pass complete.")
        optimizer.step()
        print("Optimizer update complete.")

        # Select checkpoints using cold validation, never cold test.
        model.eval()
        with torch.no_grad():
            val_metrics = evaluate_graphsage_amazon(model)
        print("Validation metrics:", val_metrics)

        current_ndcg10 = float(val_metrics["NDCG@10"])
        if current_ndcg10 > best_ndcg10:
            best_ndcg10 = current_ndcg10
            best_epoch = epoch
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(
                f"New best checkpoint: epoch {best_epoch}, "
                f"NDCG@10 = {best_ndcg10:.6f}"
            )

    print(f"Best epoch: {best_epoch}, best validation NDCG@10: {best_ndcg10:.6f}")


if __name__ == "__main__":
    main()
