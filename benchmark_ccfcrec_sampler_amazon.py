
# benchmark_ccfcrec_sampler_amazon.py

import random
import time
import numpy as np
import torch

from data_loader_amazon import load_amazon_data
from ccfcrec_dataset_amazon import (
    CCFCRecAmazonTrainingDataset,
)

SEED = 42
NUM_SAMPLES = 2048

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Loading Amazon training data...", flush=True)
data = load_amazon_data()

print("Building CCFCRec sampler...", flush=True)
dataset = CCFCRecAmazonTrainingDataset(
    edge_index=data["edge_index"],
    num_users=data["num_users"],
    num_warm_items=data["num_warm_items"],
)

# Sample across the training dataset rather than
# repeatedly using only the first few interactions.
rng = np.random.default_rng(SEED)
indices = rng.choice(
    len(dataset),
    size=NUM_SAMPLES,
    replace=False,
)

print(
    f"Benchmarking {NUM_SAMPLES} samples...",
    flush=True,
)

start = time.perf_counter()

for count, idx in enumerate(indices, start=1):
    sample = dataset[int(idx)]

    if count % 64 == 0:
        elapsed = time.perf_counter() - start
        print(
            f"{count}/{NUM_SAMPLES} samples | "
            f"{elapsed:.2f} seconds",
            flush=True,
        )

elapsed = time.perf_counter() - start

print("\nCCFCRec sampler benchmark")
print("Samples:", NUM_SAMPLES)
print("Total seconds:", round(elapsed, 3))
print(
    "Milliseconds per sample:",
    round(elapsed * 1000 / NUM_SAMPLES, 3),
)
print(
    "Estimated sampling hours for 30 epochs:",
    round(
        elapsed / NUM_SAMPLES
        * len(dataset)
        * 30 / 3600,
        2,
    ),
)
print(
    "Cached positive pools:",
    len(dataset.positive_cache),
)
print("Benchmark completed.")

cache_bytes = sum(
    pool.nbytes
    for pool in dataset.positive_cache.values()
)

print(
    "Cache memory (MB):",
    round(cache_bytes / (1024 ** 2), 2),
)

print(
    "Average positive-pool size:",
    round(
        sum(
            len(pool)
            for pool in dataset.positive_cache.values()
        ) / len(dataset.positive_cache),
        2,
    ),
)