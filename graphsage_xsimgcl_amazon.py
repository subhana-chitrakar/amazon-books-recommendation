import torch
import torch.nn.functional as F

from graphsage_amazon import GraphSAGEAmazon

class GraphSAGEXSimGCLAmazon(GraphSAGEAmazon):

    def __init__(
        self,
        num_users,
        item_input_dim=384,
        hidden_dim=64,
        eps=0.2
    ):
        """
        Amazon version of the finalized proposed model.

        Backbone:
            GraphSAGEAmazon

        XSimGCL-inspired component:
            embedding perturbation

        Amazon item content:
            384-D -> 64-D

        epsilon:
            0.2
        """

        super().__init__(
            num_users=num_users,
            item_input_dim=item_input_dim,
            hidden_dim=hidden_dim
        )

        self.eps = eps

            # -----------------------------------------------------
    # XSimGCL-inspired embedding perturbation
    # -----------------------------------------------------

    def mean_aggregate(self, user_h, item_h, edge_index):
        """Exact degree-normalized neighbor means via sparse multiplication.

        Unlike dense edge gathers, sparse.mm does not materialize an
        [num_edges, hidden_dim] tensor during forward/backward.
        The graph contains only warm training interactions.
        """
        user_idx, item_idx = edge_index[0], edge_index[1]
        num_users = user_h.size(0)
        num_items = item_h.size(0)
        device = user_h.device
        dtype = user_h.dtype

        # Cache graph-only sparse matrices across layers and epochs.
        cache_key = (edge_index.data_ptr(), num_users, num_items, device, dtype)
        if getattr(self, "_adj_cache_key", None) != cache_key:
            user_degree = torch.bincount(user_idx, minlength=num_users).to(dtype=dtype)
            item_degree = torch.bincount(item_idx, minlength=num_items).to(dtype=dtype)
            user_values = user_degree.clamp(min=1).reciprocal()[user_idx]
            item_values = item_degree.clamp(min=1).reciprocal()[item_idx]

            user_adj = torch.sparse_coo_tensor(
                torch.stack((user_idx, item_idx)), user_values,
                (num_users, num_items), device=device
            ).coalesce()
            item_adj = torch.sparse_coo_tensor(
                torch.stack((item_idx, user_idx)), item_values,
                (num_items, num_users), device=device
            ).coalesce()
            self._user_adj = user_adj
            self._item_adj = item_adj
            self._adj_cache_key = cache_key

        return (
            torch.sparse.mm(self._user_adj, item_h),
            torch.sparse.mm(self._item_adj, user_h),
        )

    def perturb(self, embeddings):

        # Generate random noise with the same shape.
        random_noise = torch.rand_like(embeddings)

        # Normalize each noise vector.
        random_noise = F.normalize(
            random_noise,
            dim=-1
        )

        # Apply sign direction and epsilon scaling.
        noise = (
            torch.sign(embeddings)
            * random_noise
            * self.eps
        )

        # Add noise to the original embeddings.
        perturbed_embeddings = embeddings + noise

        return perturbed_embeddings


        # -----------------------------------------------------
    # Perturbed GraphSAGE forward propagation
    # -----------------------------------------------------

    def forward_perturbed(
        self,
        item_features,
        edge_index
    ):

        # Step 1: Initial user and item representations.
        user_h0, item_h0 = self.get_initial_embeddings(
            item_features
        )

        # Step 2: First GraphSAGE layer.
        user_h1, item_h1 = self.sage_layer1(
            user_h0,
            item_h0,
            edge_index
        )

        # Step 3: Perturb Layer-1 representations.
        user_h1_tilde = self.perturb(user_h1)
        item_h1_tilde = self.perturb(item_h1)

        # Step 4: Second GraphSAGE layer.
        # IMPORTANT: It receives the perturbed h1.
        user_h2, item_h2 = self.sage_layer2(
            user_h1_tilde,
            item_h1_tilde,
            edge_index
        )

        # Step 5: Perturb Layer-2 representations.
        user_h2_tilde = self.perturb(user_h2)
        item_h2_tilde = self.perturb(item_h2)

        return (
            user_h1_tilde,
            item_h1_tilde,
            user_h2_tilde,
            item_h2_tilde
        )

            # -----------------------------------------------------
    # Memory-safe global InfoNCE
    # -----------------------------------------------------

    def info_nce(
        self,
        view1,
        view2,
        temperature=0.15,
        chunk_size=256
    ):
        # Normalize both views.
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)

        num_nodes = view1.size(0)
        total_loss = 0.0

        # Process anchor nodes in smaller groups.
        for start in range(0, num_nodes, chunk_size):
            end = min(start + chunk_size, num_nodes)

            anchors = view1[start:end]

            # Every anchor is compared against ALL nodes.
            similarities = torch.matmul(
                anchors,
                view2.T
            ) / temperature

            # Matching nodes are positive pairs.
            row_indices = torch.arange(
                end - start,
                device=similarities.device
            )

            positive_scores = similarities[
                row_indices,
                torch.arange(
                    start,
                    end,
                    device=similarities.device
                )
            ]

            # Stable equivalent of log(sum(exp(scores))).
            log_denominator = torch.logsumexp(
                similarities,
                dim=1
            )

            batch_loss = (
                log_denominator - positive_scores
            ).sum()

            total_loss = total_loss + batch_loss

        return total_loss / num_nodes
            # -----------------------------------------------------
    # User + item contrastive loss
    # -----------------------------------------------------

    def contrastive_loss(
        self,
        user_h1_tilde,
        item_h1_tilde,
        user_h2_tilde,
        item_h2_tilde,
        user_indices,
        positive_item_indices,
        temperature=0.15
    ):

        # Remove duplicate users and positive warm items.
        unique_users = torch.unique(user_indices)
        unique_items = torch.unique(positive_item_indices)

        # Contrast the same users across Layers 1 and 2.
        user_cl_loss = self.info_nce(
            user_h1_tilde[unique_users],
            user_h2_tilde[unique_users],
            temperature
        )

        # Contrast the same items across Layers 1 and 2.
        item_cl_loss = self.info_nce(
            item_h1_tilde[unique_items],
            item_h2_tilde[unique_items],
            temperature
        )

        # Combine user-side and item-side contrastive loss.
        total_cl_loss = user_cl_loss + item_cl_loss

        return (
            total_cl_loss,
            user_cl_loss,
            item_cl_loss
        )

        # -----------------------------------------------------
    # BPR recommendation loss
    # -----------------------------------------------------

    def bpr_loss(
        self,
        user_h2_tilde,
        item_h2_tilde,
        user_indices,
        positive_item_indices,
        negative_item_indices
    ):

        # Select final perturbed user embeddings.
        user_z = user_h2_tilde[user_indices]

        # Select positive and negative item embeddings.
        positive_item_z = item_h2_tilde[
            positive_item_indices
        ]

        negative_item_z = item_h2_tilde[
            negative_item_indices
        ]

        # Calculate user-item dot-product scores.
        positive_scores = torch.sum(
            user_z * positive_item_z,
            dim=1
        )

        negative_scores = torch.sum(
            user_z * negative_item_z,
            dim=1
        )

        # Encourage positive scores to exceed negative scores.
        loss = -F.logsigmoid(
            positive_scores - negative_scores
        ).mean()

        return loss

        # -----------------------------------------------------
    # Gradient-caching helper
    # -----------------------------------------------------

    def compute_embedding_gradients(
        self,
        view1,
        view2,
        temperature=0.15,
        chunk_size=256
    ):
        # Independent copies, disconnected from GraphSAGE.
        a = view1.detach().requires_grad_(True)
        b = view2.detach().requires_grad_(True)

        # Normalize once, then treat normalized vectors
        # as independent leaves during chunk calculations.
        a_norm = F.normalize(a, dim=1)
        b_norm = F.normalize(b, dim=1)

        a_leaf = a_norm.detach().requires_grad_(True)
        b_leaf = b_norm.detach().requires_grad_(True)

        num_nodes = a.size(0)
        total_loss = 0.0

        for start in range(0, num_nodes, chunk_size):
            end = min(start + chunk_size, num_nodes)

            similarities = (
                a_leaf[start:end] @ b_leaf.T
            ) / temperature

            row_indices = torch.arange(
                end - start, device=a.device
            )
            positive_indices = torch.arange(
                start, end, device=a.device
            )

            positives = similarities[
                row_indices, positive_indices
            ]

            chunk_loss = (
                torch.logsumexp(similarities, dim=1)
                - positives
            ).sum() / num_nodes

            # Release each chunk's computation graph.
            chunk_loss.backward()

            total_loss += chunk_loss.detach().item()

        # Propagate accumulated gradients through
        # normalization exactly once.
        torch.autograd.backward(
            (a_norm, b_norm),
            (a_leaf.grad, b_leaf.grad)
        )

        return (
            total_loss,
            a.grad.detach(),
            b.grad.detach()
        )