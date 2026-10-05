
import random
import numpy as np
import torch

from data_loader_amazon import load_amazon_data
from clcrec_amazon import CLCRecAmazon
from clcrec_dataset_amazon import (
    CLCRecAmazonTrainingDataset,
)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)

    data = load_amazon_data()

    dataset = CLCRecAmazonTrainingDataset(
        edge_index=data["edge_index"],
        num_warm_items=data["num_warm_items"],
        num_neg=128,
    )

    model = CLCRecAmazon(
        num_users=data["num_users"],
        num_warm_items=data["num_warm_items"],
        item_input_dim=384,
        embedding_dim=64,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # Exactly one real training batch.
    batch_indices = range(256)
    samples = [
        dataset[i] for i in batch_indices
    ]

    users = torch.stack([
        sample[0] for sample in samples
    ])

    items = torch.stack([
        sample[1] for sample in samples
    ])

    features =  data["warm_item_features"]

    model.train()
    optimizer.zero_grad()

    total_loss, loss1, loss2, reg_loss = (
        model.loss(
            users,
            items,
            features,
            temperature=2.0,
            lr_lambda=0.5,
            reg_weight=0.1,
            num_sample=0.5,
        )
    )

    assert torch.isfinite(total_loss)

    total_loss.backward()
    optimizer.step()

    print("One-batch training test passed.")
    print("Batch size:", len(samples))
    print("Candidate shape:", tuple(items.shape))
    print("Total loss:", total_loss.item())
    print("Loss 1:", loss1.item())
    print("Loss 2:", loss2.item())
    print("Regularization:", reg_loss.item())


if __name__ == "__main__":
    main()
