
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class CLCRecAmazonTrainingDataset(Dataset):
    """
    Memory-efficient CLCRec training dataset.

    For each positive interaction:
      - Candidate 0: observed warm positive item
      - Candidates 1..128: distinct unobserved warm negatives

    No cold-validation or cold-test items are sampled.
    """

    def __init__(
        self,
        edge_index,
        num_warm_items,
        num_neg=128,
    ):
        self.num_warm_items = int(num_warm_items)
        self.num_neg = int(num_neg)

        if self.num_warm_items <= self.num_neg:
            raise ValueError(
                "Not enough warm items for distinct negatives."
            )

        # CPU arrays of positive warm training interactions.
        self.users = (
            edge_index[0].cpu().numpy().astype(np.int64)
        )
        self.pos_items = (
            edge_index[1].cpu().numpy().astype(np.int64)
        )

        if len(self.users) == 0:
            raise ValueError("Training graph is empty.")

        if (
            self.users.min() < 0
            or self.pos_items.min() < 0
            or self.pos_items.max() >= self.num_warm_items
        ):
            raise ValueError("Invalid training graph indices.")

        # Compact CSR-style lookup:
        # Store only observed pairs, not negative pools.
        num_users = int(self.users.max()) + 1

        pair_keys = np.unique(
            self.users * self.num_warm_items
            + self.pos_items
        )

        sorted_users = pair_keys // self.num_warm_items
        sorted_items = pair_keys % self.num_warm_items

        counts = np.bincount(
            sorted_users,
            minlength=num_users,
        )

        self.offsets = np.empty(
            num_users + 1,
            dtype=np.int64,
        )
        self.offsets[0] = 0
        np.cumsum(
            counts,
            out=self.offsets[1:],
        )

        self.observed_items = sorted_items

        # Ensure every training user has at least 128
        # distinct unobserved warm items.
        if np.any(
            self.num_warm_items - counts < self.num_neg
        ):
            raise ValueError(
                "A user has fewer than num_neg "
                "available warm negatives."
            )

        print("CLCRec Amazon dataset ready.")
        print("Training interactions:", len(self.users))
        print("Warm items:", self.num_warm_items)
        print("Negatives per interaction:", self.num_neg)
        print("Unique observed pairs:", len(pair_keys))

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = int(self.users[index])
        positive_item = int(self.pos_items[index])

        start = self.offsets[user]
        end = self.offsets[user + 1]

        # Sorted observed items for this user.
        observed = self.observed_items[start:end]

        # Draw without replacement from the complement
        # of the observed items. Avoid materializing it.
        #
        # A rank in [0, available) identifies one of the
        # user's unobserved warm items.
        available = self.num_warm_items - len(observed)

        ranks = random.sample(
            range(available),
            self.num_neg,
        )

        ranks_array = np.asarray(
            ranks,
            dtype=np.int64,
        )

        # Map complement ranks back to warm item IDs.
        # For each observed item, subtract its position
        # to find where it falls in complement-rank space.
        #
        # Example:
        # observed = [1, 4]
        # complement = [0, 2, 3, 5, ...]
        # blocked thresholds = [1, 3]
        thresholds = (
            observed
            - np.arange(
                len(observed),
                dtype=np.int64,
            )
        )

        shifts = np.searchsorted(
            thresholds,
            ranks_array,
            side="right",
        )

        # A single searchsorted is insufficient when
        # skipped observed items form consecutive blocks.
        # Instead, resolve each sampled complement rank
        # using its position among available items.
        #
        # The kth unobserved item is the kth entry in
        # the conceptual complement, located by binary
        # search on its warm-item index.
        negatives = np.empty(
            self.num_neg,
            dtype=np.int64,
        )

        for j, rank in enumerate(ranks):
            low = rank
            high = rank + len(observed)

            while low < high:
                mid = (low + high) // 2

                observed_leq = np.searchsorted(
                    observed,
                    mid,
                    side="right",
                )

                available_leq = (
                    mid + 1 - observed_leq
                )

                if available_leq >= rank + 1:
                    high = mid
                else:
                    low = mid + 1

            negatives[j] = low

        item_tensor = torch.empty(
            self.num_neg + 1,
            dtype=torch.long,
        )

        item_tensor[0] = positive_item
        item_tensor[1:] = torch.from_numpy(
            negatives
        )

        user_tensor = torch.full(
            (self.num_neg + 1,),
            user,
            dtype=torch.long,
        )

        return user_tensor, item_tensor


if __name__ == "__main__":
    # Small synthetic test only.
    # Does not access Amazon files or cold-test.

    random.seed(42)

    edge_index = torch.tensor(
        [
            [0, 0, 0, 1, 1, 2],
            [1, 2, 4, 3, 5, 7],
        ],
        dtype=torch.long,
    )

    dataset = CLCRecAmazonTrainingDataset(
        edge_index=edge_index,
        num_warm_items=150,
        num_neg=128,
    )

    for index in range(len(dataset)):
        users, items = dataset[index]

        user = int(users[0])
        positive = int(items[0])
        negatives = items[1:].tolist()

        observed = set(
            dataset.observed_items[
                dataset.offsets[user]:
                dataset.offsets[user + 1]
            ].tolist()
        )

        assert len(negatives) == 128
        assert len(set(negatives)) == 128
        assert all(
            0 <= item < 150
            for item in negatives
        )
        assert all(
            item not in observed
            for item in negatives
        )
        assert positive in observed
        assert torch.all(users == user)

    print("All negative-sampling checks passed.")
