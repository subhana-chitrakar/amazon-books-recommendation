"""CCFCRec Amazon training; default is a one-batch smoke test.

Run --train only after reviewing the smoke-test output.
Cold-test is never loaded.

PC/GPU-ready:
- Automatically uses CUDA when available.
- Supports configurable DataLoader workers.
- Uses pinned memory and non-blocking transfers on CUDA.
- Keeps the research method and frozen hyperparameters unchanged.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader_amazon import load_amazon_data
from ccfcrec_amazon import CCFCRec
from ccfcrec_dataset_amazon import CCFCRecAmazonTrainingDataset
from evaluate_ccfcrec_amazon import evaluate_ccfcrec_amazon


# ============================================================
# FROZEN RESEARCH HYPERPARAMETERS
# ============================================================

SEED = 42
TAU = 0.1
LAMBDA = 0.5
EPS = 1e-8

BATCH_SIZE = 1024
EPOCHS = 30
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 0.1

NUM_POSITIVE = 10
NUM_NEGATIVE = 40
NUM_SELF_NEGATIVE = 40

EMBEDDING_DIM = 64

CHECKPOINT = "ccfcrec_amazon_30ep_best.pt"
LATEST_CHECKPOINT = "ccfcrec_amazon_latest_training.pt"


# ============================================================
# LOSSES
# ============================================================

def cooccurrence_contrastive_loss(
    content_embedding,
    positive_item_embeddings,
    negative_item_embeddings,
):
    q = content_embedding.unsqueeze(1)

    positive_similarity = (
        F.cosine_similarity(
            q,
            positive_item_embeddings,
            dim=-1,
        )
        / TAU
    )

    negative_similarity = (
        F.cosine_similarity(
            q.unsqueeze(2),
            negative_item_embeddings.unsqueeze(1),
            dim=-1,
        )
        / TAU
    )

    positive_exp = torch.exp(positive_similarity)
    negative_sum = torch.exp(negative_similarity).sum(dim=-1)

    loss = -torch.log(
        positive_exp
        / (positive_exp + negative_sum + EPS)
    )

    return loss.sum() / loss.shape[1]


def self_contrastive_loss(
    content_embedding,
    anchor_item_embedding,
    self_negative_embeddings,
):
    positive_similarity = (
        F.cosine_similarity(
            content_embedding,
            anchor_item_embedding,
            dim=-1,
        )
        / TAU
    )

    negative_similarity = (
        F.cosine_similarity(
            content_embedding.unsqueeze(1),
            self_negative_embeddings,
            dim=-1,
        )
        / TAU
    )

    positive_exp = torch.exp(positive_similarity)
    negative_sum = torch.exp(negative_similarity).sum(dim=-1)

    loss = -torch.log(
        positive_exp
        / (positive_exp + negative_sum + EPS)
    )

    return loss.sum()


def collaborative_ranking_loss(
    anchor_item_embedding,
    positive_user_embedding,
    negative_user_embedding,
):
    positive_score = (
        anchor_item_embedding * positive_user_embedding
    ).sum(dim=-1)

    negative_score = (
        anchor_item_embedding * negative_user_embedding
    ).sum(dim=-1)

    return -F.logsigmoid(
        positive_score - negative_score
    ).sum()


def content_ranking_loss(
    content_embedding,
    positive_user_embedding,
    negative_user_embedding,
):
    positive_score = (
        content_embedding * positive_user_embedding
    ).sum(dim=-1)

    negative_score = (
        content_embedding * negative_user_embedding
    ).sum(dim=-1)

    return -F.logsigmoid(
        positive_score - negative_score
    ).sum()


# ============================================================
# LOSS CALCULATION
# ============================================================

def calculate_losses(
    model,
    batch,
    warm_features,
    device,
    non_blocking=False,
):
    b = {
        key: value.to(
            device,
            non_blocking=non_blocking,
        )
        for key, value in batch.items()
    }

    anchor = model.encode_collaborative_items(
        b["anchor_item"]
    )

    positive_items = model.encode_collaborative_items(
        b["positive_items"]
    )

    negative_items = model.encode_collaborative_items(
        b["negative_items"]
    )

    self_negatives = model.encode_collaborative_items(
        b["self_negative_items"]
    )

    positive_users = model.encode_users(
        b["positive_user"]
    )

    negative_users = model.encode_users(
        b["negative_user"]
    )

    content = model.encode_content(
        warm_features[b["anchor_item"]]
    )

    contrast = cooccurrence_contrastive_loss(
        content,
        positive_items,
        negative_items,
    )

    self_loss = self_contrastive_loss(
        content,
        anchor,
        self_negatives,
    )

    collaborative = collaborative_ranking_loss(
        anchor,
        positive_users,
        negative_users,
    )

    content_rank = content_ranking_loss(
        content,
        positive_users,
        negative_users,
    )

    total = (
        LAMBDA * (contrast + self_loss)
        + (1.0 - LAMBDA)
        * (collaborative + content_rank)
    )

    return total, (
        contrast,
        self_loss,
        collaborative,
        content_rank,
    )


# ============================================================
# RANDOM SEED
# ============================================================

def set_random_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATALOADER WORKER SEEDING
# ============================================================

def seed_worker(worker_id):
    """Seed Python and NumPy RNGs inside each DataLoader worker."""
    del worker_id

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# DATALOADER
# ============================================================

def create_dataloader(dataset, batch_size, workers, device):
    pin_memory = device.type == "cuda"

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker if workers > 0 else None,
    }

    if workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(**loader_kwargs)


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_resume_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    device,
):
    # weights_only=False is required for NumPy RNG state.
    # Only load trusted local checkpoints.
    saved = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if saved.get("epoch", 0) < 1:
        raise ValueError("Invalid checkpoint epoch")

    if saved.get("epoch", 0) > EPOCHS:
        raise ValueError(
            "Checkpoint epoch is greater than configured EPOCHS"
        )

    model.load_state_dict(
        saved["model_state_dict"]
    )

    optimizer.load_state_dict(
        saved["optimizer_state_dict"]
    )

    best_ndcg = float(
        saved["best_ndcg"]
    )

    random.setstate(
        saved["python_rng_state"]
    )

    np.random.set_state(
        saved["numpy_rng_state"]
    )

    torch.set_rng_state(
        saved["torch_rng_state"].cpu()
    )

    if (
        torch.cuda.is_available()
        and saved.get("cuda_rng_states") is not None
    ):
        torch.cuda.set_rng_state_all(
            saved["cuda_rng_states"]
        )

    completed_epoch = int(
        saved["epoch"]
    )

    return completed_epoch, best_ndcg


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train CCFCRec on Amazon Books 10-core."
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run full 30-epoch training.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last completed epoch.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "DataLoader worker processes. "
            "Use 0 for safest reproducibility. "
            "On a PC, try 2-4 if CPU sampling is slow."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Training batch size. Default: {BATCH_SIZE}.",
    )

    args = parser.parse_args()

    if args.workers < 0:
        raise ValueError("--workers must be >= 0")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    # --------------------------------------------------------
    # SEEDS
    # --------------------------------------------------------

    set_random_seeds()

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = get_device()

    print(
        "Device:",
        device,
        flush=True,
    )

    if device.type == "cuda":
        print(
            "CUDA device:",
            torch.cuda.get_device_name(0),
            flush=True,
        )

    print(
        "DataLoader workers:",
        args.workers,
        flush=True,
    )

    print(
        "Batch size:",
        args.batch_size,
        flush=True,
    )

    # --------------------------------------------------------
    # LOAD AMAZON WARM TRAINING DATA
    # --------------------------------------------------------

    data = load_amazon_data()

    features = data[
        "warm_item_features"
    ].to(device)

    # --------------------------------------------------------
    # TRAINING DATASET
    # --------------------------------------------------------

    dataset = CCFCRecAmazonTrainingDataset(
        data["edge_index"],
        data["num_users"],
        data["num_warm_items"],
        num_positive=NUM_POSITIVE,
        num_negative=NUM_NEGATIVE,
        num_self_negative=NUM_SELF_NEGATIVE,
    )

    loader = create_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = CCFCRec(
        num_users=data["num_users"],
        num_warm_items=data["num_warm_items"],
        item_input_dim=features.shape[1],
        embedding_dim=EMBEDDING_DIM,
    ).to(device)

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        "Training samples:",
        len(dataset),
        flush=True,
    )

    print(
        "Positive samples per training instance:",
        NUM_POSITIVE,
        flush=True,
    )

    print(
        "Negative samples per training instance:",
        NUM_NEGATIVE,
        flush=True,
    )

    print(
        "Self-negative samples per training instance:",
        NUM_SELF_NEGATIVE,
        flush=True,
    )

    # ========================================================
    # SMOKE TEST
    # ========================================================

    if not args.train:
        model.train()

        batch = next(iter(loader))

        optimizer.zero_grad(
            set_to_none=True
        )

        loss, components = calculate_losses(
            model,
            batch,
            features,
            device,
            non_blocking=(
                device.type == "cuda"
            ),
        )

        assert bool(
            torch.isfinite(loss).item()
        ), "Nonfinite smoke-test loss"

        loss.backward()

        assert all(
            p.grad is None
            or bool(
                torch.isfinite(
                    p.grad
                ).all().item()
            )
            for p in model.parameters()
        ), "Nonfinite gradients"

        optimizer.step()

        print(
            "One-batch total loss:",
            loss.item(),
        )

        print(
            "Loss components:",
            [
                value.item()
                for value in components
            ],
        )

        print(
            "CCFCRec Amazon training smoke test PASSED"
        )

        print(
            "No checkpoint saved. Full training NOT started."
        )

        return

    # ========================================================
    # RESUME STATE
    # ========================================================

    best_ndcg = float("-inf")
    start_epoch = 1

    if args.resume:
        if not os.path.isfile(
            LATEST_CHECKPOINT
        ):
            raise FileNotFoundError(
                "Resume checkpoint not found: "
                f"{LATEST_CHECKPOINT}"
            )

        completed_epoch, best_ndcg = (
            load_resume_checkpoint(
                LATEST_CHECKPOINT,
                model,
                optimizer,
                device,
            )
        )

        start_epoch = (
            completed_epoch + 1
        )

        print(
            "Resuming from completed epoch "
            f"{completed_epoch}; "
            f"next epoch {start_epoch}",
            flush=True,
        )

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(
        start_epoch,
        EPOCHS + 1,
    ):
        model.train()

        epoch_loss = 0.0

        for step, batch in enumerate(
            loader,
            start=1,
        ):
            optimizer.zero_grad(
                set_to_none=True
            )

            loss, _ = calculate_losses(
                model,
                batch,
                features,
                device,
                non_blocking=(
                    device.type == "cuda"
                ),
            )

            if not bool(
                torch.isfinite(loss).item()
            ):
                raise FloatingPointError(
                    f"Nonfinite loss at "
                    f"epoch {epoch}, "
                    f"batch {step}"
                )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            if step % 100 == 0:
                print(
                    f"Epoch {epoch}/{EPOCHS} "
                    f"batch {step}/{len(loader)} "
                    f"loss={loss.item():.6f}",
                    flush=True,
                )

        # ----------------------------------------------------
        # STRICT COLD VALIDATION
        # ----------------------------------------------------

        metrics = evaluate_ccfcrec_amazon(
            model,
            batch_size=256,
        )

        ndcg = float(
            metrics["NDCG@10"]
        )

        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"total_loss={epoch_loss:.6f} "
            f"validation={metrics}",
            flush=True,
        )

        if not np.isfinite(ndcg):
            raise FloatingPointError(
                "Nonfinite validation NDCG@10"
            )

        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        if ndcg > best_ndcg:
            best_ndcg = ndcg

            torch.save(
                model.state_dict(),
                CHECKPOINT,
            )

            print(
                "Saved best checkpoint: "
                f"{CHECKPOINT} "
                f"(NDCG@10={best_ndcg:.9f})",
                flush=True,
            )

        # ----------------------------------------------------
        # RESUMABLE CHECKPOINT
        # ----------------------------------------------------

        latest_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_ndcg": best_ndcg,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }

        # Atomic replacement avoids leaving a partial
        # resumable checkpoint after interruption.
        temp_path = (
            LATEST_CHECKPOINT
            + ".tmp"
        )

        torch.save(
            latest_state,
            temp_path,
        )

        os.replace(
            temp_path,
            LATEST_CHECKPOINT,
        )

        print(
            "Saved resumable state after "
            f"epoch {epoch}: "
            f"{LATEST_CHECKPOINT}",
            flush=True,
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(
        f"Best validation NDCG@10: "
        f"{best_ndcg:.9f}"
    )


if __name__ == "__main__":
    main()