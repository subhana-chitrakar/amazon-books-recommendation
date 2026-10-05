
import numpy as np
import torch

from data_loader_amazon import load_amazon_data
from cold_validation_amazon import (
    load_cold_validation_features,
    load_cold_validation_ground_truth,
    recall_at_k,
    ndcg_at_k,
)


def evaluate_clcrec_amazon(
    model,
    batch_size=256,
):
    """
    Strict cold-validation evaluation for Amazon CLCRec.

    Users:
        Learned collaborative user embeddings.

    Cold items:
        Content-only embeddings from the trained encoder.

    Candidates:
        ALL Amazon cold-validation items.

    Metrics:
        Recall@5, NDCG@5
        Recall@10, NDCG@10
        Recall@20, NDCG@20

    Cold-test is never accessed.
    """

    # Use exactly the same user mapping as GraphSAGE.
    training_data = load_amazon_data()
    user_to_idx = training_data["user_to_idx"]

    # Load cold-validation item IDs and 384D features.
    cold_val_items, cold_features = (
        load_cold_validation_features()
    )

    # Same ground truth as GraphSAGE evaluation.
    ground_truth = (
        load_cold_validation_ground_truth(
            user_to_idx
        )
    )

    validation_users = np.array(
        list(ground_truth.keys()),
        dtype=np.int64,
    )

    if len(validation_users) == 0:
        raise ValueError(
            "No cold-validation users found."
        )

    if len(cold_val_items) < 20:
        raise ValueError(
            "Fewer than 20 cold-validation candidates."
        )

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            device = model.user_embedding.weight.device

            # Content-only cold-item encoding.
            # No cold item-ID embeddings are used.
            cold_z = model.encode_content(
                cold_features.to(device)
            )

            user_z = model.user_embedding.weight

            all_ranked_asins = []
            all_user_indices = []

            # Batch users to avoid a massive score matrix.
            for start in range(
                0,
                len(validation_users),
                batch_size,
            ):
                end = min(
                    start + batch_size,
                    len(validation_users),
                )

                batch_users = validation_users[
                    start:end
                ]

                batch_users_tensor = torch.as_tensor(
                    batch_users,
                    dtype=torch.long,
                    device=device,
                )

                batch_user_z = user_z[
                    batch_users_tensor
                ]

                # Rank against ALL cold-validation
                # candidates, not a sampled subset.
                scores = torch.matmul(
                    batch_user_z,
                    cold_z.T,
                )

                top_indices = torch.topk(
                    scores,
                    k=20,
                    dim=1,
                ).indices

                ranked_asins = cold_val_items[
                    top_indices.cpu().numpy()
                ]

                all_ranked_asins.append(
                    ranked_asins
                )

                all_user_indices.extend(
                    batch_users.tolist()
                )

    finally:
        if was_training:
            model.train()

    all_ranked_asins = np.concatenate(
        all_ranked_asins,
        axis=0,
    )

    all_user_indices = np.asarray(
        all_user_indices,
        dtype=np.int64,
    )

    # Reuse GraphSAGE's exact metric definitions.
    return {
        "Recall@5": recall_at_k(
            all_ranked_asins,
            all_user_indices,
            ground_truth,
            k=5,
        ),
        "NDCG@5": ndcg_at_k(
            all_ranked_asins,
            all_user_indices,
            ground_truth,
            k=5,
        ),
        "Recall@10": recall_at_k(
            all_ranked_asins,
            all_user_indices,
            ground_truth,
            k=10,
        ),
        "NDCG@10": ndcg_at_k(
            all_ranked_asins,
            all_user_indices,
            ground_truth,
            k=10,
        ),
        "Recall@20": recall_at_k(
            all_ranked_asins,
            all_user_indices,
            ground_truth,
            k=20,
        ),
        "NDCG@20": ndcg_at_k(
            all_ranked_asins,
            all_user_indices,
            ground_truth,
            k=20,
        ),
    }


if __name__ == "__main__":
    print(
        "CLCRec Amazon cold-validation evaluator "
        "imported successfully."
    )
    print(
        "No validation evaluation or training "
        "was performed."
    )
