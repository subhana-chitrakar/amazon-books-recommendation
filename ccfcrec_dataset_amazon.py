
# ccfcrec_dataset_amazon.py

import random
from turtle import position
import numpy as np
import torch
from torch.utils.data import Dataset


class CCFCRecAmazonTrainingDataset(Dataset):
    """
    Amazon adaptation of the MovieLens CCFCRec sampler.

    Positives: other warm items sharing at least one user.
    Negatives: warm items not co-occurring with anchor.
    Self-negatives: any other warm items.
    Negative user: user who did not interact with anchor.

    All samples are drawn without replacement within each
    item group, matching the MovieLens sampling rules.
    """

    def __init__(
        self,
        edge_index,
        num_users,
        num_warm_items,
        num_positive=10,
        num_negative=40,
        num_self_negative=40,
    ):
        super().__init__()

        self.num_users = int(num_users)
        self.num_warm_items = int(num_warm_items)
        self.num_positive = int(num_positive)
        self.num_negative = int(num_negative)
        self.num_self_negative = int(num_self_negative)

        users = (
            edge_index[0].cpu().numpy().astype(
                np.int64, copy=False
            )
        )
        items = (
            edge_index[1].cpu().numpy().astype(
                np.int64, copy=False
            )
        )

        if len(users) != len(items):
            raise ValueError("Invalid edge_index")

        if (
            np.any(users < 0)
            or np.any(users >= self.num_users)
            or np.any(items < 0)
            or np.any(items >= self.num_warm_items)
        ):
            raise ValueError("Edge indices out of range")

        self.users = users
        self.items = items

        # Deduplicate interactions and build sorted CSR
        # adjacency lists in both directions.
        pairs = np.unique(
            users * self.num_warm_items + items
        )

        unique_users = pairs // self.num_warm_items
        unique_items = pairs % self.num_warm_items

        user_counts = np.bincount(
            unique_users, minlength=self.num_users
        )
        self.user_offsets = np.empty(
            self.num_users + 1, dtype=np.int64
        )
        self.user_offsets[0] = 0
        np.cumsum(
            user_counts, out=self.user_offsets[1:]
        )
        self.user_items = unique_items

        # Sort the unique pairs by item, then user.
        order = np.lexsort(
            (unique_users, unique_items)
        )
        sorted_items = unique_items[order]
        self.item_users = unique_users[order]

        item_counts = np.bincount(
            sorted_items, minlength=self.num_warm_items
        )
        self.item_offsets = np.empty(
            self.num_warm_items + 1, dtype=np.int64
        )
        self.item_offsets[0] = 0
        np.cumsum(
            item_counts, out=self.item_offsets[1:]
        )

        # Positive co-occurrence sets are computed lazily
        # and cached as sorted arrays, not Python sets.
        self.positive_cache = {}

        if self.num_self_negative > self.num_warm_items - 1:
            raise ValueError(
                "Insufficient self-negative items"
            )

    def __len__(self):
        return len(self.users)

    def _get_item_users(self, item):
        start = self.item_offsets[item]
        end = self.item_offsets[item + 1]
        return self.item_users[start:end]

    def _get_user_items(self, user):
        start = self.user_offsets[user]
        end = self.user_offsets[user + 1]
        return self.user_items[start:end]

    def _positive_pool(self, item):
        cached = self.positive_cache.get(item)
        if cached is not None:
            return cached

        users = self._get_item_users(item)

        pieces = [
            self._get_user_items(int(u))
            for u in users
        ]

        if pieces:
            values = np.concatenate(pieces)

            counts = np.bincount(
                values,
                minlength=self.num_warm_items,
            )

            positives = np.flatnonzero(counts)

            positives = positives[
                positives != item
            ]
        else:
            positives = np.empty(
                0, dtype=np.int64
            )


        # Cache at most 256 positive pools.
        # Evict the oldest entry when full.
        if len(self.positive_cache) >= 256:
            oldest_item = next(iter(self.positive_cache))
            del self.positive_cache[oldest_item]
        self.positive_cache[item] = positives
        return positives

    @staticmethod
    def _sample_complement(
        excluded,
        universe_size,
        sample_size,
    ):
        """
        Uniform sampling without replacement from
        [0, universe_size) excluding sorted unique IDs.

        Map sampled complement ranks to actual IDs.
        """
        available = universe_size - len(excluded)

        if sample_size > available:
            raise ValueError(
                "Not enough candidates for sampling"
            )

        ranks = np.asarray(
            random.sample(
                range(available), sample_size
            ),
            dtype=np.int64,
        )

        thresholds = excluded - np.arange(
            len(excluded), dtype=np.int64
        )

        shifts = np.searchsorted(
            thresholds, ranks, side="right"
        )

        return ranks + shifts

    def __getitem__(self, idx):
        positive_user = int(self.users[idx])
        anchor_item = int(self.items[idx])

        # One negative user, uniformly sampled from
        # users who never interacted with anchor.
        observed_users = self._get_item_users(
            anchor_item
        )
        negative_user = int(
            self._sample_complement(
                observed_users,
                self.num_users,
                1,
            )[0]
        )

        # Co-occurring positive warm items.
        positive_pool = self._positive_pool(
            anchor_item
        )

        if len(positive_pool) < self.num_positive:
            raise ValueError(
                f"Item {anchor_item} has only "
                f"{len(positive_pool)} co-occurring "
                "positives; cannot sample 10 distinct."
            )

        positive_indices = random.sample(
             range(len(positive_pool)),
             self.num_positive,
        )
        positive_items = positive_pool[
            positive_indices
        ].copy()

        # Exclude positives and the anchor when
        # sampling non-co-occurring negative items.
        excluded = np.empty(
            len(positive_pool) + 1,
            dtype=np.int64
        )
        position = np.searchsorted(
           positive_pool, anchor_item
        )
        excluded[:position] = positive_pool[:position]
        excluded[position] = anchor_item
        excluded[position + 1:] = positive_pool[position:]

        negative_items = (
            self._sample_complement(
                excluded,
                self.num_warm_items,
                self.num_negative,
            )
        )

        # Sample other warm items as self-negatives.
        self_negatives = (
            self._sample_complement(
                np.asarray(
                    [anchor_item], dtype=np.int64
                ),
                self.num_warm_items,
                self.num_self_negative,
            )
        )

        return {
            "positive_user": torch.tensor(
                positive_user, dtype=torch.long
            ),
            "negative_user": torch.tensor(
                negative_user, dtype=torch.long
            ),
            "anchor_item": torch.tensor(
                anchor_item, dtype=torch.long
            ),
            "positive_items": torch.tensor(
                positive_items, dtype=torch.long
            ),
            "negative_items": torch.from_numpy(
                negative_items.copy()
            ),
            "self_negative_items": torch.from_numpy(
                self_negatives.copy()
            ),
        }


if __name__ == "__main__":
    # Synthetic test only
    random.seed(42)

    num_users = 20
    num_items = 100

    edges = []

    for u in range(num_users):
        if u < 10:
            item_range = range(0, 12)
        else:
            item_range = range(12, 24)

        for i in item_range:
            edges.append((u, i))

    # Keep the rest of your synthetic test
    # indented at this same level.

    edge_index = torch.tensor(
        edges, dtype=torch.long
    ).T.contiguous()

    dataset = CCFCRecAmazonTrainingDataset(
        edge_index,
        num_users=num_users,
        num_warm_items=num_items,
        num_positive=10,
        num_negative=40,
        num_self_negative=40,
    )

    sample = dataset[0]

    anchor = sample["anchor_item"].item()
    pos = sample["positive_items"].tolist()
    neg = sample["negative_items"].tolist()
    self_neg = sample[
        "self_negative_items"
    ].tolist()
    neg_user = sample["negative_user"].item()

    assert len(pos) == len(set(pos)) == 10
    assert len(neg) == len(set(neg)) == 40
    assert len(self_neg) == len(
        set(self_neg)
    ) == 40

    assert anchor not in pos
    assert anchor not in neg
    assert anchor not in self_neg

    assert not (set(pos) & set(neg))

    assert neg_user not in (
        dataset._get_item_users(anchor)
    )

    print("CCFCRec Amazon dataset test PASSED")
