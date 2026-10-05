
# ccfcrec_amazon.py

import torch
import torch.nn as nn


class CCFCRec(nn.Module):
    """
    CCFCRec adapted for Amazon Books strict cold-item
    recommendation.

    UCE: learned user collaborative embeddings
    COCE: learned warm-item collaborative embeddings
    CBCE: content-based collaborative encoder

    Cold items are represented using content only.
    """

    def __init__(
        self,
        num_users=158650,
        num_warm_items=103153,
        item_input_dim=384,
        embedding_dim=64,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_warm_items = num_warm_items
        self.item_input_dim = item_input_dim
        self.embedding_dim = embedding_dim

        # User collaborative embeddings
        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim,
        )

        # Warm-item collaborative embeddings
        self.item_embedding = nn.Embedding(
            num_warm_items,
            embedding_dim,
        )

        # Content encoder
        self.content_encoder = nn.Sequential(
            nn.Linear(item_input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, embedding_dim),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(
            self.user_embedding.weight
        )
        nn.init.xavier_normal_(
            self.item_embedding.weight
        )

        for layer in self.content_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def encode_users(self, user_ids):
        return self.user_embedding(user_ids)

    def encode_collaborative_items(self, item_ids):
        # Only warm-item indices are allowed here.
        return self.item_embedding(item_ids)

    def encode_content(self, item_features):
        # Used for warm training and cold evaluation.
        return self.content_encoder(item_features)

    def forward(self, item_features):
        return self.encode_content(item_features)

    def score(self, user_ids, item_features):
        user_repr = self.encode_users(user_ids)
        item_repr = self.encode_content(item_features)

        return torch.sum(
            user_repr * item_repr,
            dim=-1,
        )


if __name__ == "__main__":
    torch.manual_seed(42)

    model = CCFCRec(
        num_users=10,
        num_warm_items=20,
        item_input_dim=384,
        embedding_dim=64,
    )

    users = torch.tensor([0, 1, 2])
    warm_items = torch.tensor([3, 4, 5])
    features = torch.randn(3, 384)

    user_z = model.encode_users(users)
    item_z = model.encode_collaborative_items(
        warm_items
    )
    content_z = model.encode_content(features)
    scores = model.score(users, features)

    assert user_z.shape == (3, 64)
    assert item_z.shape == (3, 64)
    assert content_z.shape == (3, 64)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()

    print("CCFCRec Amazon model test PASSED")
