
# ============================================================
# train_clcrec_amazon.py
# Amazon Books - CLCRec Training
# Strict Cold-Item Recommendation
# ============================================================

import argparse
import os
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
from cold_validation_clcrec_amazon import (
    evaluate_clcrec_amazon,
)


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

SEED = 42

EPOCHS = 30
BATCH_SIZE = 256
NUM_NEG = 128

LEARNING_RATE = 0.001
TEMPERATURE = 2.0
LR_LAMBDA = 0.5
REG_WEIGHT = 0.1
NUM_SAMPLE = 0.5

EMBEDDING_DIM = 64

BEST_PATH = "best_clcrec_amazon.pt"
LAST_PATH = "last_clcrec_amazon.pt"


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. TRAINING
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one batch without saving checkpoints.",
    )

    args = parser.parse_args()

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is unavailable. Check your GPU "
            "and PyTorch installation."
        )

    device = torch.device(args.device)

    set_seed(SEED)
    torch.set_num_threads(4)

    print("Device:", device, flush=True)

    # --------------------------------------------------------
    # Load Amazon training data
    # --------------------------------------------------------

    data = load_amazon_data()

    warm_features = data[
        "warm_item_features"
    ].to(device)

    dataset = CLCRecAmazonTrainingDataset(
        edge_index=data["edge_index"],
        num_warm_items=data["num_warm_items"],
        num_neg=NUM_NEG,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------------
    # Initialize CLCRec
    # --------------------------------------------------------

    model = CLCRecAmazon(
        num_users=data["num_users"],
        num_warm_items=data["num_warm_items"],
        item_input_dim=warm_features.shape[1],
        embedding_dim=EMBEDDING_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # One-batch smoke test
    # --------------------------------------------------------

    if args.smoke_test:

        model.train()

        users, items = next(iter(loader))

        users = users.to(device)
        items = items.to(device)

        optimizer.zero_grad(set_to_none=True)

        total_loss, loss1, loss2, reg_loss = (
            model.loss(
                users,
                items,
                warm_features,
                temperature=TEMPERATURE,
                lr_lambda=LR_LAMBDA,
                reg_weight=REG_WEIGHT,
                num_sample=NUM_SAMPLE,
            )
        )

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                "Smoke test produced a non-finite loss."
            )

        total_loss.backward()
        optimizer.step()

        print("\nSmoke test passed.")
        print("Total loss:", total_loss.item())
        print("Loss 1:", loss1.item())
        print("Loss 2:", loss2.item())
        print("Regularization:", reg_loss.item())
        print("No checkpoint saved.")

        return

    # --------------------------------------------------------
    # Protect existing checkpoints
    # --------------------------------------------------------

    if (
        os.path.exists(BEST_PATH)
        or os.path.exists(LAST_PATH)
    ):
        raise FileExistsError(
            "An existing CLCRec checkpoint was found. "
            "Training stopped to prevent overwriting it."
        )

    best_ndcg10 = float("-inf")
    best_epoch = None

    print("\nTraining configuration:")
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Negatives:", NUM_NEG)
    print("Learning rate:", LEARNING_RATE)
    print("Batches per epoch:", len(loader))

    # ========================================================
    # 4. FULL TRAINING LOOP
    # ========================================================

    for epoch in range(1, EPOCHS + 1):

        start_time = time.perf_counter()

        model.train()

        total_sum = 0.0
        loss1_sum = 0.0
        loss2_sum = 0.0
        reg_sum = 0.0

        processed = 0

        for batch_idx, (users, items) in enumerate(
            loader,
            start=1,
        ):

            users = users.to(
                device,
                non_blocking=True,
            )

            items = items.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            total_loss, loss1, loss2, reg_loss = (
                model.loss(
                    users,
                    items,
                    warm_features,
                    temperature=TEMPERATURE,
                    lr_lambda=LR_LAMBDA,
                    reg_weight=REG_WEIGHT,
                    num_sample=NUM_SAMPLE,
                )
            )

            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, "
                    f"batch {batch_idx}."
                )

            total_loss.backward()

            optimizer.step()

            batch_count = users.shape[0]

            processed += batch_count

            total_sum += (
                total_loss.item() * batch_count
            )

            loss1_sum += (
                loss1.item() * batch_count
            )

            loss2_sum += (
                loss2.item() * batch_count
            )

            reg_sum += (
                reg_loss.item() * batch_count
            )

            if batch_idx % 1000 == 0:

                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Batch {batch_idx}/{len(loader)} | "
                    f"Loss {total_sum / processed:.6f}",
                    flush=True,
                )

        # ----------------------------------------------------
        # Epoch loss
        # ----------------------------------------------------

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        print(
            f"Total loss: "
            f"{total_sum / processed:.6f}"
        )

        print(
            f"Loss 1: "
            f"{loss1_sum / processed:.6f}"
        )

        print(
            f"Loss 2: "
            f"{loss2_sum / processed:.6f}"
        )

        print(
            f"Regularization: "
            f"{reg_sum / processed:.6f}"
        )

        # ====================================================
        # 5. STRICT COLD-VALIDATION
        # ====================================================

        metrics = evaluate_clcrec_amazon(
            model,
            batch_size=256,
        )

        print("\nCold-validation metrics:")

        for name, value in metrics.items():

            print(
                f"{name}: {float(value):.9f}"
            )

        ndcg10 = float(
            metrics["NDCG@10"]
        )

        if not np.isfinite(ndcg10):
            raise RuntimeError(
                "Non-finite validation NDCG@10."
            )

        # ----------------------------------------------------
        # Save best validation checkpoint
        # ----------------------------------------------------

        if ndcg10 > best_ndcg10:

            best_ndcg10 = ndcg10
            best_epoch = epoch

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "epoch": epoch,

                "validation_metrics": {
                    key: float(value)
                    for key, value in metrics.items()
                },

                "config": {
                    "seed": SEED,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "num_neg": NUM_NEG,
                    "learning_rate": LEARNING_RATE,
                    "temperature": TEMPERATURE,
                    "lr_lambda": LR_LAMBDA,
                    "reg_weight": REG_WEIGHT,
                    "num_sample": NUM_SAMPLE,
                    "embedding_dim": EMBEDDING_DIM,
                    "num_users": data["num_users"],
                    "num_warm_items":
                        data["num_warm_items"],
                    "item_input_dim":
                        warm_features.shape[1],
                },
            }

            torch.save(
                checkpoint,
                BEST_PATH,
            )

            print(
                "\nSaved new best checkpoint."
            )

            print(
                "Best epoch:",
                best_epoch,
            )

            print(
                "Best NDCG@10:",
                best_ndcg10,
            )

        elapsed = (
            time.perf_counter() - start_time
        )

        print(
            f"\nEpoch time: "
            f"{elapsed / 60:.2f} minutes\n",
            flush=True,
        )

    # ========================================================
    # 6. SAVE FINAL EPOCH
    # ========================================================

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "epoch": EPOCHS,
        },
        LAST_PATH,
    )

    print("\nTraining completed.")

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation NDCG@10:",
        best_ndcg10,
    )

    print(
        "Best checkpoint:",
        BEST_PATH,
    )

    print(
        "Final checkpoint:",
        LAST_PATH,
    )


if __name__ == "__main__":
    main()


# python train_clcrec_amazon.py --smoke-test