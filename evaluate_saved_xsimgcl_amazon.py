import torch

from graphsage_xsimgcl_amazon import GraphSAGEXSimGCLAmazon
from data_loader_amazon import load_amazon_data
from cold_validation_amazon import evaluate_graphsage_amazon


CHECKPOINT_PATH = "best_graphsage_xsimgcl_amazon.pt"


def main():

    # Load the same Amazon data used during training.
    data = load_amazon_data()

    num_users = data["num_users"]

    # Create the Proposed model architecture.
    model = GraphSAGEXSimGCLAmazon(
        num_users=num_users,
        item_input_dim=384,
        hidden_dim=64,
        eps=0.2
    )

    # Load the best Proposed checkpoint.
    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict, strict=True)

    model.eval()

    print(
        "Loaded Proposed checkpoint:",
        CHECKPOINT_PATH
    )

    print(
        "Evaluating GraphSAGE + XSimGCL-inspired model "
        "on cold validation..."
    )

    # IMPORTANT:
    # This uses the normal clean forward pass.
    # Do NOT use forward_perturbed() during evaluation.
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