import torch

from data_loader_amazon import load_amazon_data
from clcrec_amazon import CLCRecAmazon
from cold_validation_clcrec_amazon import evaluate_clcrec_amazon


CHECKPOINT_PATH = "best_clcrec_amazon.pt"


def main():

    # -----------------------------------------------------
    # Load Amazon data
    # -----------------------------------------------------

    data = load_amazon_data()

    warm_features = data["warm_item_features"]

    # -----------------------------------------------------
    # Create the same CLCRec architecture
    # -----------------------------------------------------

    model = CLCRecAmazon(
        num_users=data["num_users"],
        num_warm_items=data["num_warm_items"],
        item_input_dim=warm_features.shape[1],
        embedding_dim=64,
    )

    # -----------------------------------------------------
    # Load the BEST validation checkpoint
    # -----------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    print(
        "Loaded CLCRec checkpoint:",
        CHECKPOINT_PATH
    )

    if "epoch" in checkpoint:
        print(
            "Best checkpoint epoch:",
            checkpoint["epoch"]
        )

    if "validation_metrics" in checkpoint:
        print("\nMetrics saved inside checkpoint:")
        for name, value in checkpoint["validation_metrics"].items():
            print(f"{name}: {float(value):.9f}")

    # -----------------------------------------------------
    # Evaluate on strict cold validation
    # -----------------------------------------------------

    print("\nEvaluating CLCRec on cold validation...")

    with torch.no_grad():

        metrics = evaluate_clcrec_amazon(
            model,
            batch_size=256,
        )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\nCold-validation results:")

    print(
        f"Recall@5:  {float(metrics['Recall@5']):.9f}"
    )

    print(
        f"NDCG@5:    {float(metrics['NDCG@5']):.9f}"
    )

    print(
        f"Recall@10: {float(metrics['Recall@10']):.9f}"
    )

    print(
        f"NDCG@10:   {float(metrics['NDCG@10']):.9f}"
    )

    print(
        f"Recall@20: {float(metrics['Recall@20']):.9f}"
    )

    print(
        f"NDCG@20:   {float(metrics['NDCG@20']):.9f}"
    )


if __name__ == "__main__":
    main()