"""
Checks for the temperature-scaled semantic loss (models/helper_funcs.semantic_infonce).

Three properties, the first of which is the whole reason the loss is written this way:

1. tau -> inf recovers the published flat mean-difference loss, so the published objective is
   the tau -> inf limit of the new one and tau is a continuous knob between them.
2. within-sequence entries (cpc_mask==2) never enter the denominator -- they compare a caption
   embedding with itself, so using them as negatives would be a false-negative gradient.
3. --semantic_dedup removes same-scene rows (identical caption embeddings) from the denominator.

Run: python test_semantic_infonce.py  (no data, no GPU)
"""
import torch

from helpers.helper_funcs import create_cpc_matrix
from models.helper_funcs import semantic_infonce, compute_losses

torch.manual_seed(0)

B, T, D = 8, 5, 32          # 8 sequences x 5 timesteps, 32-d embeddings
N = B * T


def setup(seed=0):
    """A (target) / B (prediction) pair with the shapes the semantic loss sees."""
    g = torch.Generator().manual_seed(seed)
    tgt = torch.randn(B, D, generator=g)
    tgt = tgt / tgt.norm(dim=1, keepdim=True)
    A = tgt.unsqueeze(1).repeat(1, T, 1).reshape(-1, D)
    pred = torch.randn(N, D, generator=g, requires_grad=True)
    cpc_mask = create_cpc_matrix(N, T)
    return A, pred, cpc_mask


def published_loss(sim, cpc_mask):
    return sim[cpc_mask == 3].mean() - sim[cpc_mask == 1].mean()


def grad_of(loss_fn, pred):
    if pred.grad is not None:
        pred.grad = None
    loss_fn().backward()
    return pred.grad.clone()


# ---------------------------------------------------------------- 1. the tau -> inf limit
A, pred, cpc_mask = setup()
sim = lambda: A @ (pred / pred.norm(dim=1, keepdim=True)).T

g_pub = grad_of(lambda: published_loss(sim(), cpc_mask), pred)
for tau in (0.05, 1.0, 100.0):
    g_nce = grad_of(lambda: semantic_infonce(sim(), cpc_mask, tau, norm='tau')[0], pred)
    cos = torch.nn.functional.cosine_similarity(g_pub.flatten(), g_nce.flatten(), dim=0).item()
    ratio = (g_nce.norm() / g_pub.norm()).item()
    print(f'tau={tau:<7} cos(grad, published grad)={cos:+.5f}  |grad|/|published|={ratio:.3f}')
    if tau == 100.0:
        assert cos > 0.999, cos          # same direction
        assert 0.9 < ratio < 1.1, ratio  # and the same scale, so no effective-LR shift

# -------------------------------------------------- 2. mask==2 is out of the denominator
A, pred, cpc_mask = setup(1)
# a prediction that resembles its own sequence's other timesteps must not be penalised for
# it: perturbing a cpc_mask==2 entry may not change the loss
sim_leaf = (A @ (pred / pred.norm(dim=1, keepdim=True)).T).detach().requires_grad_(True)
semantic_infonce(sim_leaf, cpc_mask, 0.1, norm='tau')[0].backward()
assert sim_leaf.grad[cpc_mask == 2].abs().max() == 0, 'cpc_mask==2 leaked into the loss'
assert sim_leaf.grad[cpc_mask == 3].abs().max() > 0
print('mask==2 gradient:', sim_leaf.grad[cpc_mask == 2].abs().max().item(),
      ' mask==3 gradient (max):', sim_leaf.grad[cpc_mask == 3].abs().max().item())

# --------------------------------------------------------------------------- 3. dedup
# sequences 0 and 3 are the same scene -> identical caption embeddings
A, pred, cpc_mask = setup(2)
A = A.reshape(B, T, D)
A[3] = A[0]
A = A.reshape(N, D)
img_n = torch.tensor([0, 1, 2, 0, 4, 5, 6, 7])
ids = img_n.repeat_interleave(T)
same_img = ids.unsqueeze(1) == ids.unsqueeze(0)

sim_leaf = (A @ (pred / pred.norm(dim=1, keepdim=True)).T).detach().requires_grad_(True)
semantic_infonce(sim_leaf, cpc_mask, 0.1, norm='tau', same_img=same_img)[0].backward()
cross = same_img & (cpc_mask == 3)
assert sim_leaf.grad[cross].abs().max() == 0, 'duplicate caption embeddings still in denominator'
assert sim_leaf.grad.diagonal().abs().min() > 0, 'dedup removed the positive'
print(f'dedup: {int(cross.sum())} duplicate entries dropped, positives intact')

# ------------------------------------------------- 4. the two variants run end to end
hyp = {'optimizer': {'losses': {'glimpse_loss': 0, 'scene_loss': 0, 'gazeloc_loss': 0,
                                'semantic_loss': 1, 'semantic_temp': 0.1,
                                'semantic_norm': 'tau', 'semantic_dedup': 1}}}
tgt = torch.randn(B, D)
outputs = [None, torch.randn(B, T, D, requires_grad=True), None, None]
for variant in (1, 2):
    hyp['optimizer']['losses']['semantic_loss'] = variant
    loss, floor, diag = compute_losses(outputs, None, None, tgt, None, cpc_mask, hyp, 1,
                                       img_n=img_n)
    print(f'semantic_loss={variant}: loss={loss.item():+.4f} floor={floor.item():+.4f} '
          f'{ {k: round(v, 4) for k, v in diag.items()} }')
    assert torch.isfinite(loss)

print('\nall checks passed')
