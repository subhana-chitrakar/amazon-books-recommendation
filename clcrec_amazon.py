
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLCRecAmazon(nn.Module):
    """
    CLCRec adaptation for Amazon Books strict cold-item recommendation.

    Collaborative representations:
        User-ID embeddings: 64D
        Warm-item-ID embeddings: 64D

    Content encoder:
        384D -> 256D -> 64D

    Cold items:
        Encoded using content only, without item-ID embeddings.

    Training objectives:
        Loss 1: Collaborative-item/content-item contrast
        Loss 2: User/hybrid-item contrast
        Regularization: User/item embedding norm
    """

    def __init__(
        self,
        num_users,
        num_warm_items,
        item_input_dim=384,
        embedding_dim=64,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_warm_items = num_warm_items
        self.item_input_dim = item_input_dim
        self.embedding_dim = embedding_dim

        # Collaborative embeddings: training users and warm items only.
        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim,
        )

        self.item_embedding = nn.Embedding(
            num_warm_items,
            embedding_dim,
        )

        # Same two-layer content encoder as MovieLens CLCRec.
        # Only the input dimension changes: 403 -> 384.
        self.encoder_layer1 = nn.Linear(
            item_input_dim,
            256,
        )

        self.encoder_layer2 = nn.Linear(
            256,
            embedding_dim,
        )

        # Preserve MovieLens embedding initialization.
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def encode_content(self, item_features):
        """
        Encode item features without accessing item-ID embeddings.

        Supports:
            [N, 384]
            [batch_size, num_candidates, 384]
        """
        x = self.encoder_layer1(item_features)
        x = F.leaky_relu(x)
        x = self.encoder_layer2(x)

        return x

    def loss_contrastive(
        self,
        anchor,
        candidates,
        temperature,
    ):
        """
        Contrastive classification loss.

        anchor:
            [batch_size, embedding_dim]

        candidates:
            [batch_size, 1 + num_neg, embedding_dim]

        Candidate index 0 is the positive.

        This logsumexp formulation is mathematically equivalent
        to the MovieLens exp/sum/log implementation, but avoids
        numerical overflow or underflow.
        """
        scores = torch.sum(
            anchor.unsqueeze(1) * candidates,
            dim=2,
        )

        scores = scores / temperature

        positive_scores = scores[:, 0]

        log_denominator = torch.logsumexp(
            scores,
            dim=1,
        )

        return (
            log_denominator - positive_scores
        ).mean()

    def item_content_contrastive_loss(
        self,
        item_tensor,
        warm_item_features,
        temperature=2.0,
    ):
        """
        Loss 1:
        Positive collaborative item versus candidate content items.

        item_tensor:
            [batch_size, 1 + num_neg]

        All item indices must refer to warm training items.
        """
        positive_items = item_tensor[:, 0]

        positive_collab = self.item_embedding(
            positive_items,
        )

        positive_collab = F.normalize(
            positive_collab,
            dim=1,
        )

        candidate_features = warm_item_features[
            item_tensor
        ]

        candidate_content = self.encode_content(
            candidate_features,
        )

        candidate_content = F.normalize(
            candidate_content,
            dim=2,
        )

        return self.loss_contrastive(
            positive_collab,
            candidate_content,
            temperature,
        )

    def user_item_contrastive_loss(
        self,
        user_tensor,
        item_tensor,
        warm_item_features,
        temperature=2.0,
        num_sample=0.5,
    ):
        """
        Loss 2:
        User versus positive/negative hybrid item embeddings.

        Starts with collaborative item embeddings.
        Random candidate positions are replaced with
        content representations.

        The random sampling with replacement is intentionally
        preserved from the MovieLens implementation.
        """
        users = user_tensor[:, 0]

        user_z = self.user_embedding(
            users,
        )

        item_collab_z = self.item_embedding(
            item_tensor,
        )

        candidate_features = warm_item_features[
            item_tensor
        ]

        item_content_z = self.encode_content(
            candidate_features,
        )

        item_input = item_collab_z.clone()

        flat_item_input = item_input.reshape(
            -1,
            item_input.size(-1),
        )

        flat_content = item_content_z.reshape(
            -1,
            item_content_z.size(-1),
        )

        num_replace = int(
            flat_item_input.size(0) * num_sample
        )

        rand_index = torch.randint(
            flat_item_input.size(0),
            (num_replace,),
            device=flat_item_input.device,
        )

        flat_item_input[rand_index] = (
            flat_content[rand_index].clone()
        )

        item_input = flat_item_input.view_as(
            item_input
        )

        return self.loss_contrastive(
            user_z,
            item_input,
            temperature,
        )

    def loss(
        self,
        user_tensor,
        item_tensor,
        warm_item_features,
        temperature=2.0,
        lr_lambda=0.5,
        reg_weight=0.1,
        num_sample=0.5,
    ):
        """
        Complete CLCRec objective.

        total_loss =
            lr_lambda * loss1
            + (1 - lr_lambda) * loss2
            + reg_loss

        Expected inputs:
            user_tensor: [batch_size, 129]
            item_tensor: [batch_size, 129]
            warm_item_features: [num_warm_items, 384]

        The first item is positive; the remaining 128
        items are distinct unobserved warm negatives.
        """
        # Loss 1: collaborative item <-> content item.
        loss1 = self.item_content_contrastive_loss(
            item_tensor=item_tensor,
            warm_item_features=warm_item_features,
            temperature=temperature,
        )

        # Loss 2: user <-> hybrid item.
        loss2 = self.user_item_contrastive_loss(
            user_tensor=user_tensor,
            item_tensor=item_tensor,
            warm_item_features=warm_item_features,
            temperature=temperature,
            num_sample=num_sample,
        )

        # Same embedding-norm regularization as MovieLens.
        users = user_tensor[:, 0]

        user_z = self.user_embedding(
            users,
        )

        item_z = self.item_embedding(
            item_tensor,
        )

        user_reg = torch.sqrt(
            torch.sum(user_z ** 2, dim=-1)
        ).mean()

        item_reg = torch.sqrt(
            torch.sum(item_z ** 2, dim=-1)
        ).mean()

        reg_loss = (
            (user_reg + item_reg) / 2.0
        ) * reg_weight

        contrastive_loss = (
            lr_lambda * loss1
            + (1.0 - lr_lambda) * loss2
        )

        total_loss = contrastive_loss + reg_loss

        return (
            total_loss,
            loss1,
            loss2,
            reg_loss,
        )


if __name__ == "__main__":
    # Lightweight architecture test.
    # Does not load data, train, or access cold-test.

    torch.manual_seed(42)

    model = CLCRecAmazon(
        num_users=10,
        num_warm_items=20,
        item_input_dim=384,
        embedding_dim=64,
    )

    dummy_features = torch.randn(20, 384)

    user_tensor = torch.tensor([
        [0, 0, 0],
        [1, 1, 1],
    ], dtype=torch.long)

    item_tensor = torch.tensor([
        [0, 1, 2],
        [3, 4, 5],
    ], dtype=torch.long)

    total_loss, loss1, loss2, reg_loss = model.loss(
        user_tensor=user_tensor,
        item_tensor=item_tensor,
        warm_item_features=dummy_features,
        temperature=2.0,
        lr_lambda=0.5,
        reg_weight=0.1,
        num_sample=0.5,
    )

    assert torch.isfinite(total_loss).item()

    total_loss.backward()

    assert model.user_embedding.weight.grad is not None
    assert model.item_embedding.weight.grad is not None
    assert model.encoder_layer1.weight.grad is not None
    assert model.encoder_layer2.weight.grad is not None

    cold_features = torch.randn(4, 384)

    with torch.no_grad():
        cold_z = model.encode_content(cold_features)

    assert cold_z.shape == (4, 64)

    print("CLCRec Amazon model sanity checks passed.")
    print("Total loss:", total_loss.item())
    print("Loss 1:", loss1.item())
    print("Loss 2:", loss2.item())
    print("Regularization:", reg_loss.item())
    print("Cold embedding shape:", tuple(cold_z.shape))
