import torch
import torch.nn.functional as F
from train_xsimgcl_amazon import compute_bpr_embedding_gradients

torch.manual_seed(42)

# Tiny synthetic embeddings
users = torch.randn(5, 4)
items = torch.randn(7, 4)

u = torch.tensor([0, 1, 1, 2, 4])
pos = torch.tensor([0, 2, 3, 4, 5])
neg = torch.tensor([1, 3, 4, 5, 6])

class TinyModel:
    def bpr_loss(self, user_emb, item_emb, u, p, n):
        positive = (user_emb[u] * item_emb[p]).sum(dim=1)
        negative = (user_emb[u] * item_emb[n]).sum(dim=1)
        return -F.logsigmoid(positive - negative).mean()

model = TinyModel()

# Original direct gradients
u_direct = users.clone().requires_grad_(True)
i_direct = items.clone().requires_grad_(True)
model.bpr_loss(u_direct, i_direct, u, pos, neg).backward()

# Memory-safe chunked gradients
_, u_cached, i_cached = compute_bpr_embedding_gradients(
    model, users, items, u, pos, neg, chunk_size=2
)

print("User gradients match:",
      torch.allclose(u_direct.grad, u_cached, atol=1e-6))
print("Item gradients match:",
      torch.allclose(i_direct.grad, i_cached, atol=1e-6))