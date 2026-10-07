import torch

from graphsage_xsimgcl_amazon import GraphSAGEXSimGCLAmazon
from data_loader_amazon import load_amazon_data
from cold_validation_amazon import evaluate_graphsage_amazon

CHECKPOINT_PATH = "best_graphsage_perturbation_amazon.pt"


def main():
    data = load_amazon_data()

    model = GraphSAGEXSimGCLAmazon(
        num_users=data["num_users"],
        item_input_dim=384,
        hidden_dim=64,
        eps=0.2
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    print("Loaded checkpoint:", CHECKPOINT_PATH)
    print("Evaluating perturbation-only model on Amazon cold validation...")

    with torch.no_grad():
        metrics = evaluate_graphsage_amazon(model)

    print("\nCold-validation results:")
    print(f"Recall@5:  {float(metrics['Recall@5']):.9f}")
    print(f"NDCG@5:    {float(metrics['NDCG@5']):.9f}")
    print(f"Recall@10: {float(metrics['Recall@10']):.9f}")
    print(f"NDCG@10:   {float(metrics['NDCG@10']):.9f}")
    print(f"Recall@20: {float(metrics['Recall@20']):.9f}")
    print(f"NDCG@20:   {float(metrics['NDCG@20']):.9f}")


if __name__ == "__main__":
    main()