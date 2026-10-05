"""CCFCRec Amazon training; default is a one-batch smoke test.

Run --train only after reviewing the smoke-test output. Cold-test is never loaded.
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

SEED = 42
TAU = 0.1
LAMBDA = 0.5
EPS = 1e-8
BATCH_SIZE = 1024
EPOCHS = 30
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 0.1
CHECKPOINT = "ccfcrec_amazon_30ep_best.pt"
LATEST_CHECKPOINT = "ccfcrec_amazon_latest_training.pt"


def cooccurrence_contrastive_loss(content_embedding, positive_item_embeddings, negative_item_embeddings):
    q = content_embedding.unsqueeze(1)
    positive_similarity = F.cosine_similarity(q, positive_item_embeddings, dim=-1) / TAU
    negative_similarity = F.cosine_similarity(
        q.unsqueeze(2), negative_item_embeddings.unsqueeze(1), dim=-1
    ) / TAU
    positive_exp = torch.exp(positive_similarity)
    negative_sum = torch.exp(negative_similarity).sum(dim=-1)
    loss = -torch.log(positive_exp / (positive_exp + negative_sum + EPS))
    return loss.sum() / loss.shape[1]


def self_contrastive_loss(content_embedding, anchor_item_embedding, self_negative_embeddings):
    positive_similarity = F.cosine_similarity(content_embedding, anchor_item_embedding, dim=-1) / TAU
    negative_similarity = F.cosine_similarity(
        content_embedding.unsqueeze(1), self_negative_embeddings, dim=-1
    ) / TAU
    positive_exp = torch.exp(positive_similarity)
    negative_sum = torch.exp(negative_similarity).sum(dim=-1)
    loss = -torch.log(positive_exp / (positive_exp + negative_sum + EPS))
    return loss.sum()


def collaborative_ranking_loss(anchor_item_embedding, positive_user_embedding, negative_user_embedding):
    positive_score = (anchor_item_embedding * positive_user_embedding).sum(dim=-1)
    negative_score = (anchor_item_embedding * negative_user_embedding).sum(dim=-1)
    return -F.logsigmoid(positive_score - negative_score).sum()


def content_ranking_loss(content_embedding, positive_user_embedding, negative_user_embedding):
    positive_score = (content_embedding * positive_user_embedding).sum(dim=-1)
    negative_score = (content_embedding * negative_user_embedding).sum(dim=-1)
    return -F.logsigmoid(positive_score - negative_score).sum()


def calculate_losses(model, batch, warm_features, device):
    b = {key: value.to(device) for key, value in batch.items()}
    anchor = model.encode_collaborative_items(b["anchor_item"])
    positive_items = model.encode_collaborative_items(b["positive_items"])
    negative_items = model.encode_collaborative_items(b["negative_items"])
    self_negatives = model.encode_collaborative_items(b["self_negative_items"])
    positive_users = model.encode_users(b["positive_user"])
    negative_users = model.encode_users(b["negative_user"])
    content = model.encode_content(warm_features[b["anchor_item"]])
    contrast = cooccurrence_contrastive_loss(content, positive_items, negative_items)
    self_loss = self_contrastive_loss(content, anchor, self_negatives)
    collaborative = collaborative_ranking_loss(anchor, positive_users, negative_users)
    content_rank = content_ranking_loss(content, positive_users, negative_users)
    total = LAMBDA * (contrast + self_loss) + (1.0 - LAMBDA) * (collaborative + content_rank)
    return total, (contrast, self_loss, collaborative, content_rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run full 30-epoch training")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last completed epoch")
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    data = load_amazon_data()
    features = data["warm_item_features"].to(device)
    dataset = CCFCRecAmazonTrainingDataset(
        data["edge_index"], data["num_users"], data["num_warm_items"],
        num_positive=10, num_negative=40, num_self_negative=40,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    model = CCFCRec(
        num_users=data["num_users"],
        num_warm_items=data["num_warm_items"],
        item_input_dim=features.shape[1],
        embedding_dim=64,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print("Training samples:", len(dataset), flush=True)
    print("Batch size:", BATCH_SIZE, flush=True)

    if not args.train:
        model.train()
        batch = next(iter(loader))
        optimizer.zero_grad()
        loss, components = calculate_losses(model, batch, features, device)
        assert bool(torch.isfinite(loss).item()), "Nonfinite smoke-test loss"
        loss.backward()
        assert all(
            p.grad is None or bool(torch.isfinite(p.grad).all().item())
            for p in model.parameters()
        ), "Nonfinite gradients"
        optimizer.step()
        print("One-batch total loss:", loss.item())
        print("Loss components:", [v.item() for v in components])
        print("CCFCRec Amazon training smoke test PASSED")
        print("No checkpoint saved. Full training NOT started.")
        return

    best_ndcg = float("-inf")
    start_epoch = 1
    if args.resume:
        if not os.path.isfile(LATEST_CHECKPOINT):
            raise FileNotFoundError(f"Resume checkpoint not found: {LATEST_CHECKPOINT}")
        # weights_only=False is required for NumPy RNG state; load only trusted local checkpoints.
        saved = torch.load(LATEST_CHECKPOINT, map_location=device, weights_only=False)
        if saved.get("epoch", 0) < 1 or saved.get("epoch", 0) > EPOCHS:
            raise ValueError("Invalid checkpoint epoch")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        best_ndcg = float(saved["best_ndcg"])
        random.setstate(saved["python_rng_state"])
        np.random.set_state(saved["numpy_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        if torch.cuda.is_available() and saved.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
        start_epoch = int(saved["epoch"]) + 1
        print(f"Resuming from completed epoch {start_epoch - 1}; next epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            optimizer.zero_grad()
            loss, _ = calculate_losses(model, batch, features, device)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {step}")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if step % 100 == 0:
                print(f"Epoch {epoch}/{EPOCHS} batch {step}/{len(loader)} loss={loss.item():.6f}", flush=True)
        metrics = evaluate_ccfcrec_amazon(model, batch_size=256)
        ndcg = float(metrics["NDCG@10"])
        print(f"Epoch {epoch}/{EPOCHS} total_loss={epoch_loss:.6f} validation={metrics}", flush=True)
        if not np.isfinite(ndcg):
            raise FloatingPointError("Nonfinite validation NDCG@10")
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"Saved best checkpoint: {CHECKPOINT} (NDCG@10={best_ndcg:.9f})", flush=True)
        # Atomic replacement avoids leaving a partial latest checkpoint after interruption.
        latest_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_ndcg": best_ndcg,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        temp_path = LATEST_CHECKPOINT + ".tmp"
        torch.save(latest_state, temp_path)
        os.replace(temp_path, LATEST_CHECKPOINT)
        print(f"Saved resumable state after epoch {epoch}: {LATEST_CHECKPOINT}", flush=True)
    print(f"Best validation NDCG@10: {best_ndcg:.9f}")


if __name__ == "__main__":
    main()
