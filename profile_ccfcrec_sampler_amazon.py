
import cProfile
import pstats
import random

import numpy as np

from data_loader_amazon import load_amazon_data
from ccfcrec_dataset_amazon import (
    CCFCRecAmazonTrainingDataset,
)

random.seed(42)
np.random.seed(42)

data = load_amazon_data()

dataset = CCFCRecAmazonTrainingDataset(
    edge_index=data["edge_index"],
    num_users=data["num_users"],
    num_warm_items=data["num_warm_items"],
)

rng = np.random.default_rng(42)
indices = rng.choice(
    len(dataset),
    size=2048,
    replace=False,
)

profiler = cProfile.Profile()
profiler.enable()

for idx in indices:
    dataset[int(idx)]

profiler.disable()

print("\nTop 20 functions by cumulative time:")
pstats.Stats(profiler).strip_dirs().sort_stats(
    "cumulative"
).print_stats(20)
