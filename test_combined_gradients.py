import torch
from graphsage_xsimgcl_amazon import GraphSAGEXSimGCLAmazon
from train_xsimgcl_amazon import compute_bpr_embedding_gradients

torch.manual_seed(42)

model = GraphSAGEXSimGCLAmazon(
    num_users=5,
    item_input_dim=4,
    hidden_dim=4,
    eps=0.2
)

# Tiny synthetic embeddings
base = [torch.randn(n, 4) for n in (5, 5, 7, 7)]

u = torch.tensor([0, 1, 1, 2, 4])
pos = torch.tensor([0, 2, 3, 4, 5])
neg = torch.tensor([1, 3, 4, 5, 6])

# Direct objective
direct = [x.clone().requires_grad_(True) for x in base]
uh1, uh2, ih1, ih2 = direct

direct_loss = (
    model.bpr_loss(uh2, ih2, u, pos, neg)
    + 0.01 * (
        model.info_nce(uh1, uh2, chunk_size=2)
        + model.info_nce(ih1, ih2, chunk_size=2)
    )
)
direct_loss.backward()

# Cached-gradient objective
cached = [x.clone().requires_grad_(True) for x in base]
uh1, uh2, ih1, ih2 = cached

_, bu, bi = compute_bpr_embedding_gradients(
    model, uh2, ih2, u, pos, neg, chunk_size=2
)

_, u1, u2 = model.compute_embedding_gradients(
    uh1, uh2, chunk_size=2
)
_, i1, i2 = model.compute_embedding_gradients(
    ih1, ih2, chunk_size=2
)

torch.autograd.backward(
    cached,
    [0.01*u1, bu+0.01*u2, 0.01*i1, bi+0.01*i2]
)

print(
    "Combined gradients match:",
    all(
        torch.allclose(a.grad, b.grad, atol=1e-6)
        for a, b in zip(direct, cached)
    )
)