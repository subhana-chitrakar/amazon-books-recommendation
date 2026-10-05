import copy
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

features = torch.randn(7, 4)
edges = torch.tensor([
    [0, 1, 1, 2, 3, 4, 4],
    [0, 1, 2, 3, 4, 5, 6]
])

users = torch.tensor([0, 1, 2, 3, 4])
positive = torch.tensor([0, 1, 3, 4, 5])
negative = torch.tensor([1, 2, 4, 5, 6])

# Direct objective
torch.manual_seed(123)
direct_model = copy.deepcopy(model)

u1, i1, u2, i2 = direct_model.forward_perturbed(
    features, edges
)

loss = (
    direct_model.bpr_loss(
        u2, i2, users, positive, negative
    )
    + 0.01 * (
        direct_model.info_nce(u1, u2, chunk_size=2)
        + direct_model.info_nce(i1, i2, chunk_size=2)
    )
)
loss.backward()

# Cached-gradient objective, using identical perturbations
torch.manual_seed(123)
cached_model = copy.deepcopy(model)

u1, i1, u2, i2 = cached_model.forward_perturbed(
    features, edges
)

_, bu, bi = compute_bpr_embedding_gradients(
    cached_model, u2, i2,
    users, positive, negative, chunk_size=2
)

_, gu1, gu2 = cached_model.compute_embedding_gradients(
    u1, u2, chunk_size=2
)
_, gi1, gi2 = cached_model.compute_embedding_gradients(
    i1, i2, chunk_size=2
)

torch.autograd.backward(
    [u1, u2, i1, i2],
    [0.01*gu1, bu+0.01*gu2,
     0.01*gi1, bi+0.01*gi2]
)

matches = [
    torch.allclose(
        a.grad, b.grad, atol=1e-5, rtol=1e-4
    )
    for a, b in zip(
        direct_model.parameters(),
        cached_model.parameters()
    )
]

print("All GraphSAGE gradients match:", all(matches))