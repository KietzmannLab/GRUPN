"""
Verifies, on random toy data, that the per-timestep expansion compute_losses does for the
semantic_loss==1 branch is a no-op for the resulting means -- i.e. that computing the loss
directly at the sample level (no T-fold repeat, no create_cpc_matrix) gives EXACTLY the same
number as the real "slow" path. This is copied verbatim from helpers/helper_funcs.py
(create_cpc_matrix) and the semantic_loss==1 branch of models/helper_funcs.py (compute_losses),
reproduced here only to avoid needing torchvision for this standalone offline check -- the real
cluster run (random_caption_baseline.py) still calls the actual functions, and additionally
cross-checks its own fast path against them on a few real batches at run time.
"""
import torch


def create_cpc_matrix(N, m, mask=True):
    block_value, rest_value, diag_value = 2, 3, 1
    matrix = torch.full((N, N), float(rest_value))
    torch.diagonal(matrix).fill_(diag_value)
    for i in range(0, N, m):
        matrix[i:i + m, i:i + m] = torch.full((m, m), float(block_value))
        torch.diagonal(matrix[i:i + m, i:i + m]).fill_(diag_value)
    return matrix


def slow_semantic_loss(semantic_embed, B_per_sample, T):
    """Exact copy of compute_losses' semantic_loss==1 branch, with the T-fold expansion."""
    b = semantic_embed.shape[0]
    cpc_mask = create_cpc_matrix(b * T, T)

    A = semantic_embed.unsqueeze(1).repeat(1, T, 1).reshape(-1, semantic_embed.shape[1])
    B = B_per_sample.unsqueeze(1).repeat(1, T, 1).reshape(-1, B_per_sample.shape[1])
    A = A / A.norm(dim=1, keepdim=True)
    similarity_matrix = A @ (B / B.norm(dim=1, keepdim=True)).T
    loss = similarity_matrix[cpc_mask == 3].mean() - similarity_matrix[cpc_mask == 1].mean()

    oracle = A @ A.T
    floor = oracle[cpc_mask == 3].mean() - oracle[cpc_mask == 1].mean()
    return loss.item(), floor.item()


def fast_semantic_loss(semantic_embed, B_per_sample):
    """Proposed shortcut: operate at the sample level, no T-fold expansion at all."""
    A = semantic_embed / semantic_embed.norm(dim=1, keepdim=True)
    B = B_per_sample / B_per_sample.norm(dim=1, keepdim=True)
    b = A.shape[0]
    eye = torch.eye(b, dtype=torch.bool)

    sim = A @ B.T
    diag_mean = sim[eye].mean()
    offdiag_mean = sim[~eye].mean()
    loss = offdiag_mean - diag_mean

    oracle = A @ A.T
    floor = oracle[~eye].mean() - oracle[eye].mean()
    return loss.item(), floor.item()


def main():
    torch.manual_seed(0)
    for trial, (b, T, D) in enumerate([(7, 6, 16), (13, 6, 768), (512, 6, 768), (37, 1, 8)]):
        A = torch.randn(b, D)
        perm = torch.randperm(b)
        while (perm == torch.arange(b)).any():
            perm = torch.randperm(b)
        B_random = A[perm]

        loss_slow, floor_slow = slow_semantic_loss(A, B_random, T)
        loss_fast, floor_fast = fast_semantic_loss(A, B_random)

        loss_slow_oracle, _ = slow_semantic_loss(A, A, T)
        loss_fast_oracle, _ = fast_semantic_loss(A, A)

        print(f'trial {trial} (b={b}, T={T}, D={D}):')
        print(f'  random  loss  slow={loss_slow:+.6f} fast={loss_fast:+.6f} '
              f'diff={abs(loss_slow - loss_fast):.2e}')
        print(f'  floor         slow={floor_slow:+.6f} fast={floor_fast:+.6f} '
              f'diff={abs(floor_slow - floor_fast):.2e}')
        print(f'  oracle-as-B   slow={loss_slow_oracle:+.6f} fast={loss_fast_oracle:+.6f} '
              f'diff={abs(loss_slow_oracle - loss_fast_oracle):.2e}  '
              f'(should also == floor: {abs(loss_slow_oracle - floor_slow):.2e})')
        assert abs(loss_slow - loss_fast) < 1e-5
        assert abs(floor_slow - floor_fast) < 1e-5
        assert abs(loss_slow_oracle - floor_slow) < 1e-5
    print('\nALL CHECKS PASSED -- the sample-level shortcut is exact.')


if __name__ == '__main__':
    main()
