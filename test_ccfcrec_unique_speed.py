
import time
import numpy as np

from data_loader_amazon import load_amazon_data
from ccfcrec_dataset_amazon import (
    CCFCRecAmazonTrainingDataset,
)

data = load_amazon_data()

dataset = CCFCRecAmazonTrainingDataset(
    data["edge_index"],
    data["num_users"],
    data["num_warm_items"],
)

rng = np.random.default_rng(42)
indices = rng.choice(
    len(dataset),
    size=2048,
    replace=False,
)

unique_time = 0.0
bincount_time = 0.0

for idx in indices:
    item = int(dataset.items[int(idx)])
    users = dataset._get_item_users(item)

    pieces = [
        dataset._get_user_items(int(u))
        for u in users
    ]

    values = np.concatenate(pieces)

    start = time.perf_counter()
    original = np.unique(values)
    unique_time += time.perf_counter() - start

    start = time.perf_counter()
    counts = np.bincount(
        values,
        minlength=dataset.num_warm_items,
    )
    alternative = np.flatnonzero(counts)
    bincount_time += time.perf_counter() - start

    assert np.array_equal(original, alternative)

print("\n2048 random-anchor comparison")
print("All results identical: True")
print(
    "np.unique total seconds:",
    round(unique_time, 4),
)
print(
    "np.bincount total seconds:",
    round(bincount_time, 4),
)
print(
    "np.unique average ms:",
    round(unique_time / 2048 * 1000, 4),
)
print(
    "np.bincount average ms:",
    round(bincount_time / 2048 * 1000, 4),
)
