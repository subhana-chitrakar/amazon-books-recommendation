
import numpy as np

from data_loader_amazon import load_amazon_data
from ccfcrec_dataset_amazon import (
    CCFCRecAmazonTrainingDataset,
)


data = load_amazon_data()

dataset = CCFCRecAmazonTrainingDataset(
    edge_index=data["edge_index"],
    num_users=data["num_users"],
    num_warm_items=data["num_warm_items"],
)

unique_anchors = np.unique(dataset.items)

too_few_positives = 0
too_few_negatives = 0
no_negative_users = 0

minimum_positives = float("inf")
minimum_negatives = float("inf")

for index, item in enumerate(unique_anchors):
    item = int(item)

    positives = dataset._positive_pool(item)

    num_pos = len(positives)
    num_neg = dataset.num_warm_items - num_pos - 1
    num_neg_users = (
        dataset.num_users
        - len(dataset._get_item_users(item))
    )

    minimum_positives = min(
        minimum_positives, num_pos
    )
    minimum_negatives = min(
        minimum_negatives, num_neg
    )

    if num_pos < 10:
        too_few_positives += 1

    if num_neg < 40:
        too_few_negatives += 1

    if num_neg_users < 1:
        no_negative_users += 1

    if (index + 1) % 10000 == 0:
        print(
            f"Checked {index + 1}/"
            f"{len(unique_anchors)} items",
            flush=True,
        )

print("\nCCFCRec Amazon eligibility report")
print("Warm anchors:", len(unique_anchors))
print("Minimum co-occurring positives:",
      minimum_positives)
print("Minimum non-co-occurring negatives:",
      minimum_negatives)
print("Items with fewer than 10 positives:",
      too_few_positives)
print("Items with fewer than 40 negatives:",
      too_few_negatives)
print("Items with no negative users:",
      no_negative_users)
