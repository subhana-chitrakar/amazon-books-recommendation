
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

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

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        drop_last=False,
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

    # Complete warm-item feature matrix.
    features = data["warm_item_features"]

    max_batches = 100
    model.train()

    start = time.perf_counter()
    completed = 0

    for users, items in loader:
        optimizer.zero_grad(set_to_none=True)

        total_loss, _, _, _ = model.loss(
            users,
            items,
            features,
            temperature=2.0,
            lr_lambda=0.5,
            reg_weight=0.1,
            num_sample=0.5,
        )

        if not torch.isfinite(total_loss):
            raise RuntimeError("Non-finite loss")

        total_loss.backward()
        optimizer.step()

        completed += 1

        if completed % 20 == 0:
            print(
                f"Completed {completed} batches; "
                f"loss={total_loss.item():.6f}",
                flush=True,
            )

        if completed >= max_batches:
            break

    elapsed = time.perf_counter() - start

    batches_per_epoch = len(loader)
    estimated_hours = (
        elapsed / completed
        * batches_per_epoch
        * 30
        / 3600
    )

    print("\nBenchmark completed.")
    print(f"Batches: {completed}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(
        f"Seconds per batch: "
        f"{elapsed / completed:.3f}"
    )
    print(
        f"Estimated 30-epoch training hours: "
        f"{estimated_hours:.2f}"
    )
    print(
        "Estimate excludes validation, startup, "
        "and checkpoint saving."
    )


if __name__ == "__main__":
    main()
