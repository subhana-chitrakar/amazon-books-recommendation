import torch
import torch.nn as nn


# ============================================================
# AMAZON GRAPHSAGE
# ============================================================

class GraphSAGEAmazon(nn.Module):

    def __init__(
        self,
        num_users,
        item_input_dim=384,
        hidden_dim=64
    ):
        super().__init__()

        # ====================================================
        # Initial representations
        # ====================================================

        # Learned 64-D representation for each Amazon user
        self.user_embedding = nn.Embedding(
            num_users,
            hidden_dim
        )

        # Amazon item content: 384-D -> 64-D
        self.item_encoder = nn.Linear(
            item_input_dim,
            hidden_dim
        )

        # ====================================================
        # GraphSAGE Layer 1
        # ====================================================

        self.user_layer1 = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.item_layer1 = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        # ====================================================
        # GraphSAGE Layer 2
        # ====================================================

        self.user_layer2 = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.item_layer2 = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        # ====================================================
        # Initialization
        # ====================================================

        nn.init.xavier_uniform_(
            self.user_embedding.weight
        )

        nn.init.xavier_uniform_(
            self.item_encoder.weight
        )


    # ========================================================
    # Initial embeddings h^(0)
    # ========================================================

    def get_initial_embeddings(
        self,
        item_features
    ):

        user_h0 = self.user_embedding.weight

        item_h0 = self.item_encoder(
            item_features
        )

        return user_h0, item_h0


    # ========================================================
    # Mean neighbor aggregation
    # ========================================================

    def mean_aggregate(
        self,
        user_h,
        item_h,
        edge_index
    ):

        user_idx = edge_index[0]
        item_idx = edge_index[1]

        # ----------------------------------------------------
        # Items -> Users
        # ----------------------------------------------------

        user_neighbor_sum = torch.zeros_like(
            user_h
        )

        user_neighbor_sum.index_add_(
            0,
            user_idx,
            item_h[item_idx]
        )

        user_degree = torch.zeros(
            user_h.size(0),
            device=user_h.device,
            dtype=user_h.dtype
        )

        user_degree.index_add_(
            0,
            user_idx,
            torch.ones_like(
                user_idx,
                dtype=user_h.dtype
            )
        )

        user_neighbor_mean = (
            user_neighbor_sum
            / user_degree.clamp(min=1).unsqueeze(1)
        )

        # ----------------------------------------------------
        # Users -> Items
        # ----------------------------------------------------

        item_neighbor_sum = torch.zeros_like(
            item_h
        )

        item_neighbor_sum.index_add_(
            0,
            item_idx,
            user_h[user_idx]
        )

        item_degree = torch.zeros(
            item_h.size(0),
            device=item_h.device,
            dtype=item_h.dtype
        )

        item_degree.index_add_(
            0,
            item_idx,
            torch.ones_like(
                item_idx,
                dtype=user_h.dtype
            )
        )

        item_neighbor_mean = (
            item_neighbor_sum
            / item_degree.clamp(min=1).unsqueeze(1)
        )

        return user_neighbor_mean, item_neighbor_mean


    # ========================================================
    # GraphSAGE Layer 1
    # ========================================================

    def sage_layer1(
        self,
        user_h0,
        item_h0,
        edge_index
    ):

        user_mean, item_mean = self.mean_aggregate(
            user_h0,
            item_h0,
            edge_index
        )

        # Self representation + neighbor mean
        user_input = torch.cat(
            [user_h0, user_mean],
            dim=1
        )

        item_input = torch.cat(
            [item_h0, item_mean],
            dim=1
        )

        # Separate transformations for users and items
        user_h1 = torch.relu(
            self.user_layer1(user_input)
        )

        item_h1 = torch.relu(
            self.item_layer1(item_input)
        )

        return user_h1, item_h1


    # ========================================================
    # GraphSAGE Layer 2
    # ========================================================

    def sage_layer2(
        self,
        user_h1,
        item_h1,
        edge_index
    ):

        user_mean, item_mean = self.mean_aggregate(
            user_h1,
            item_h1,
            edge_index
        )

        # Self representation + neighbor mean
        user_input = torch.cat(
            [user_h1, user_mean],
            dim=1
        )

        item_input = torch.cat(
            [item_h1, item_mean],
            dim=1
        )

        # Separate transformations for users and items
        user_h2 = torch.relu(
            self.user_layer2(user_input)
        )

        item_h2 = torch.relu(
            self.item_layer2(item_input)
        )

        return user_h2, item_h2


    # ========================================================
    # Complete GraphSAGE forward pass
    # ========================================================

    def forward(
        self,
        item_features,
        edge_index
    ):

        # h^(0)
        user_h0, item_h0 = self.get_initial_embeddings(
            item_features
        )

        # h^(0) -> h^(1)
        user_h1, item_h1 = self.sage_layer1(
            user_h0,
            item_h0,
            edge_index
        )

        # h^(1) -> h^(2)
        user_h2, item_h2 = self.sage_layer2(
            user_h1,
            item_h1,
            edge_index
        )

        # Final representations z = h^(2)
        return user_h2, item_h2


    # ========================================================
    # Strict cold-item inference
    # ========================================================

    def encode_cold_items(
        self,
        cold_item_features
    ):

        # ----------------------------------------------------
        # Initial cold-item representation h^(0)
        # Amazon content: 384-D -> 64-D
        # ----------------------------------------------------

        item_h0 = self.item_encoder(
            cold_item_features
        )

        # ----------------------------------------------------
        # Layer 1
        #
        # Strict cold items have ZERO training neighbors.
        # ----------------------------------------------------

        zero_neighbors_h0 = torch.zeros_like(
            item_h0
        )

        item_input1 = torch.cat(
            [item_h0, zero_neighbors_h0],
            dim=1
        )

        item_h1 = torch.relu(
            self.item_layer1(item_input1)
        )

        # ----------------------------------------------------
        # Layer 2
        #
        # Cold items still have ZERO training neighbors.
        # ----------------------------------------------------

        zero_neighbors_h1 = torch.zeros_like(
            item_h1
        )

        item_input2 = torch.cat(
            [item_h1, zero_neighbors_h1],
            dim=1
        )

        item_h2 = torch.relu(
            self.item_layer2(item_input2)
        )

        return item_h2

    # ============================================================
# BPR NEGATIVE SAMPLING
# ============================================================

def sample_negative_items(
    user_indices,
    positive_item_indices,
    num_warm_items
):

    # Store each user's observed warm items
    user_positive_items = {}

    for user, item in zip(
        user_indices.tolist(),
        positive_item_indices.tolist()
    ):

        if user not in user_positive_items:
            user_positive_items[user] = set()

        user_positive_items[user].add(item)

    negative_items = []

    # One unobserved warm negative for each positive interaction
    for user in user_indices.tolist():

        while True:

            negative_item = torch.randint(
                low=0,
                high=num_warm_items,
                size=(1,)
            ).item()

            if negative_item not in user_positive_items[user]:

                negative_items.append(
                    negative_item
                )

                break

    return torch.tensor(
        negative_items,
        dtype=torch.long
    )


# ============================================================
# BPR LOSS
# ============================================================

def bpr_loss(
    user_embeddings,
    positive_item_embeddings,
    negative_item_embeddings
):

    positive_scores = torch.sum(
        user_embeddings
        * positive_item_embeddings,
        dim=1
    )

    negative_scores = torch.sum(
        user_embeddings
        * negative_item_embeddings,
        dim=1
    )

    loss = -torch.log(
        torch.sigmoid(
            positive_scores
            - negative_scores
        )
        + 1e-8
    ).mean()

    return loss