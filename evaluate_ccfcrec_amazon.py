
import numpy as np
import torch

from data_loader_amazon import load_amazon_data

from cold_validation_amazon import (
    load_cold_validation_features,
    load_cold_validation_ground_truth,
    recall_at_k,
    ndcg_at_k,
)


def evaluate_ccfcrec_amazon(
    model,
    batch_size=256,
):
    """
    Strict cold-validation evaluation for CCFCRec.

    Uses learned user embeddings and content-only
    cold-item embeddings. Returns six metrics.
    """
    data = load_amazon_data()
    user_to_idx = data["user_to_idx"]

    cold_items, cold_features = (
        load_cold_validation_features()
    )

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

    if len(cold_items) < 20:
        raise ValueError(
            "Fewer than 20 validation candidates."
        )

    device = next(model.parameters()).device
    was_training = model.training

    model.eval()

    all_ranked_asins = []
    all_user_indices = []

    try:
        with torch.no_grad():
            cold_features = cold_features.to(device)

            # Strict cold: content only, no item IDs
            # or training interactions.
            cold_embeddings = model.encode_content(
                cold_features
            )

            for start in range(
                0,
                len(validation_users),
                batch_size,
            ):
                batch_users = validation_users[
                    start:start + batch_size
                ]

                user_tensor = torch.as_tensor(
                    batch_users,
                    dtype=torch.long,
                    device=device,
                )

                user_embeddings = (
                    model.encode_users(user_tensor)
                )

                scores = torch.matmul(
                    user_embeddings,
                    cold_embeddings.T,
                )

                top_indices = torch.topk(
                    scores,
                    k=20,
                    dim=1,
                ).indices

                ranked_asins = cold_items[
                    top_indices.cpu().numpy()
                ]

                all_ranked_asins.append(
                    ranked_asins
                )

                all_user_indices.extend(
                    batch_users.tolist()
                )

    finally:
        model.train(was_training)

    ranked_asins = np.concatenate(
        all_ranked_asins,
        axis=0,
    )

    user_indices = np.asarray(
        all_user_indices,
        dtype=np.int64,
    )

    return {
        "Recall@5": recall_at_k(
            ranked_asins,
            user_indices,
            ground_truth,
            k=5,
        ),
        "NDCG@5": ndcg_at_k(
            ranked_asins,
            user_indices,
            ground_truth,
            k=5,
        ),
        "Recall@10": recall_at_k(
            ranked_asins,
            user_indices,
            ground_truth,
            k=10,
        ),
        "NDCG@10": ndcg_at_k(
            ranked_asins,
            user_indices,
            ground_truth,
            k=10,
        ),
        "Recall@20": recall_at_k(
            ranked_asins,
            user_indices,
            ground_truth,
            k=20,
        ),
        "NDCG@20": ndcg_at_k(
            ranked_asins,
            user_indices,
            ground_truth,
            k=20,
        ),
    }



if __name__ == "__main__":
    from ccfcrec_amazon import CCFCRec

    torch.manual_seed(42)

    data = load_amazon_data()

    model = CCFCRec(
        num_users=data["num_users"],
        num_warm_items=data["num_warm_items"],
        item_input_dim=data[
            "warm_item_features"
        ].shape[1],
        embedding_dim=64,
    )

    print("Testing Amazon CCFCRec validation...")

    results = evaluate_ccfcrec_amazon(
        model,
        batch_size=256,
    )

    for name, value in results.items():
        print(f"{name}: {value:.6f}")

    assert len(results) == 6
    assert all(
        np.isfinite(value)
        for value in results.values()
    )

    print(
        "CCFCRec Amazon validation smoke test PASSED"
    )
