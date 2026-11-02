"""
Empirical "random caption prediction" baseline for the memorygate semantic contrastive loss.

Mirrors exactly the run logged to wandb as kietzmannlab/grupn/m8iqdcjf
(gpn_lstm_n_1024_tm_1_t_6_..._bbv6_..._semc_1_..._tr_train_515_...):
  dataset_path = /share/klab/datasets/GPN/, bbv=6, trainer=train_515, timesteps T=6,
  batch_size=512 (default, unmodified by every run_train_*.sh on the cluster -- checked).
Needs only the 'mpnet_embeddings' arrays of the GPN dataset h5 files -- no glimpses, no
checkpoint, no GPU.

compute_losses' semantic_loss==1 branch expands each sample's embedding T-fold (one row per
LSTM timestep) before building the N=batch_size*T similarity matrix. For a target-vs-target
(no real network output) computation that expansion is a pure bookkeeping no-op: every one of
the T diagonal entries in a sample's own block is a copy of the same value, and every one of
the T*T cross-block entries between two samples is a copy of that pair's single value, so none
of the masked means it feeds into change. Operating at the sample level directly (plain
batch_size x batch_size similarity matrix, diagonal = "self", off-diagonal = "other samples")
gives the identical number at ~T^2 = 36x less compute; verified against the real
create_cpc_matrix/compute_losses on random toy data (test_random_caption_baseline_shortcut.py,
diffs <1e-6) and cross-checked again below against the actual imported functions on real
batches from both splits before trusting the fast path for the full run.

For each batch:
  - floor  = mean_{i!=j} cos(A_i, A_j) - 1                         (oracle / perfect prediction;
    1 by construction, so floor depends only on cross-image caption confusability -- should
    reproduce the already-logged values, train -0.8639 / val -0.8349)
  - random = mean_{i!=j} cos(A_i, B_i) - mean_i cos(A_i, B_i)      where B = A permuted by a
    derangement (no sample mapped to itself) -- the loss a predictor with zero information
    about scene identity would get.
Both are averaged unweighted over batches within one pass (matching train_net.py's own
running-mean convention), repeated over many random derangements/shuffles to report mean +/- SD.
"""
import json
import os
import numpy as np
import h5py
import torch

from helpers.helper_funcs import create_cpc_matrix, CONDS_TEMPLATE
from models.helper_funcs import compute_losses

DATASET_PATH = '/share/klab/datasets/GPN/'
DATASET_STR = 'coco_NSD_dg3fix91_r50v6ap_7fix'
N_TRACES = 10           # dg3_fix_actvs.shape[1], confirmed via h5 metadata
TIMESTEPS = 6           # run name's t_6 / args.timesteps default
BATCH_SIZE = 512        # args.batch_size default, unmodified by any run_train_*.sh on hpc3
N_REPS = 20
N_CROSSCHECK_BATCHES = 3   # per split: verify the fast path against the real slow path
SEED = 0

HYP = {'optimizer': {'losses': {'glimpse_loss': 0, 'semantic_loss': 1,
                                 'scene_loss': 0, 'gazeloc_loss': 0}}}


def load_embeddings(split):
    """Per-image mpnet embeddings for 'train_515' or 'val', in CocoGaze's own row order."""
    if split == 'val':
        with h5py.File(f'{DATASET_PATH}{DATASET_STR}_val.h5', 'r') as f:
            return f['val']['mpnet_embeddings'][()].astype(np.float32)
    elif split == 'train_515':
        with h5py.File(f'{DATASET_PATH}{DATASET_STR}_train.h5', 'r') as f:
            train_emb = f['train']['mpnet_embeddings'][()].astype(np.float32)
        with h5py.File(f'{DATASET_PATH}{DATASET_STR}_test.h5', 'r') as f:
            test_emb_full = f['test']['mpnet_embeddings'][()].astype(np.float32)
        conds = np.load(CONDS_TEMPLATE.format(n=515))
        test_idx = np.setdiff1d(np.arange(test_emb_full.shape[0]), conds)
        return np.concatenate([train_emb, test_emb_full[test_idx]], axis=0)
    raise ValueError(split)


def derangement(n, rng):
    """Random permutation of arange(n) with no fixed points (needed for batches >= 2 rows)."""
    perm = rng.permutation(n)
    fixed = np.nonzero(perm == np.arange(n))[0]
    while len(fixed):
        if len(fixed) == 1 and n > 1:
            j = rng.integers(0, n)
            while j == fixed[0]:
                j = rng.integers(0, n)
            perm[fixed[0]], perm[j] = perm[j], perm[fixed[0]]
        else:
            shuffled = rng.permutation(fixed)
            perm[fixed] = shuffled
        fixed = np.nonzero(perm == np.arange(n))[0]
    return perm


def batches_for_split(n_images, shuffle, rng):
    flat_img_idx = np.repeat(np.arange(n_images), N_TRACES)  # matches CocoGaze idx//n_traces
    if shuffle:
        flat_img_idx = flat_img_idx[rng.permutation(len(flat_img_idx))]
    for start in range(0, len(flat_img_idx), BATCH_SIZE):
        yield flat_img_idx[start:start + BATCH_SIZE]


def fast_floor_and_random(A_batch, B_random_batch):
    """Sample-level shortcut (see module docstring) -- exact, ~36x cheaper than the T-expanded
    create_cpc_matrix/compute_losses path."""
    b = A_batch.shape[0]
    eye = torch.eye(b, dtype=torch.bool)

    A = A_batch / A_batch.norm(dim=1, keepdim=True)
    B = B_random_batch / B_random_batch.norm(dim=1, keepdim=True)

    sim_random = A @ B.T
    random_loss = sim_random[~eye].mean() - sim_random[eye].mean()

    oracle = A @ A.T
    floor = oracle[~eye].mean() - oracle[eye].mean()
    return floor.item(), random_loss.item()


def slow_floor_and_random(A_batch, B_random_batch):
    """The real create_cpc_matrix/compute_losses path, T-expanded -- used only as a per-run
    cross-check on a few batches, not for the full pass (see module docstring)."""
    b = A_batch.shape[0]
    cpc_mask = create_cpc_matrix(b * TIMESTEPS, TIMESTEPS)
    outputs_random = [None, B_random_batch.unsqueeze(1).repeat(1, TIMESTEPS, 1), None, None]
    loss_random, floor, _ = compute_losses(outputs_random, None, None, A_batch, None,
                                            cpc_mask, HYP, 1)
    return floor.item(), loss_random.item()


def run_pass(embeddings, shuffle, rng, crosscheck_remaining):
    """One pass over the split; returns (mean_floor, mean_random) over batches, unweighted,
    exactly mirroring train_net.py's own running-mean convention. `crosscheck_remaining` counts
    down how many more batches should additionally be verified against the real slow path."""
    floor_sum = random_sum = 0.0
    n_batches = 0
    for batch_img_idx in batches_for_split(embeddings.shape[0], shuffle, rng):
        b = len(batch_img_idx)
        if b < 2:
            continue  # can't build a negative pair
        A_batch = torch.from_numpy(embeddings[batch_img_idx])            # [b, 768]
        perm = derangement(b, rng)
        B_random_batch = A_batch[perm]                                   # [b, 768], no self-match

        floor, random_loss = fast_floor_and_random(A_batch, B_random_batch)

        if crosscheck_remaining[0] > 0:
            floor_slow, random_slow = slow_floor_and_random(A_batch, B_random_batch)
            assert abs(floor - floor_slow) < 1e-4, (floor, floor_slow)
            assert abs(random_loss - random_slow) < 1e-4, (random_loss, random_slow)
            crosscheck_remaining[0] -= 1
            print(f'  crosscheck ok (fast={floor:+.6f}/{random_loss:+.6f} vs '
                  f'slow={floor_slow:+.6f}/{random_slow:+.6f})', flush=True)

        floor_sum += floor
        random_sum += random_loss
        n_batches += 1

    return floor_sum / n_batches, random_sum / n_batches


def main():
    rng = np.random.default_rng(SEED)
    results = {}
    for split_name, split_key, shuffle in [('train_515', 'train_515', True), ('val', 'val', False)]:
        embeddings = load_embeddings(split_key)
        crosscheck_remaining = [N_CROSSCHECK_BATCHES]
        floors, randoms = [], []
        for rep in range(N_REPS):
            f, r = run_pass(embeddings, shuffle, rng, crosscheck_remaining)
            floors.append(f); randoms.append(r)
            print(f'[{split_name}] rep {rep}: floor={f:+.4f} random={r:+.4f}', flush=True)
        results[split_name] = {
            'n_images': int(embeddings.shape[0]),
            'floor_mean': float(np.mean(floors)), 'floor_sd': float(np.std(floors, ddof=1)),
            'random_mean': float(np.mean(randoms)), 'random_sd': float(np.std(randoms, ddof=1)),
            'n_reps': N_REPS,
        }
        print(f'[{split_name}] SUMMARY: '
              f'floor={results[split_name]["floor_mean"]:+.4f}+/-{results[split_name]["floor_sd"]:.4f}  '
              f'random={results[split_name]["random_mean"]:+.4f}+/-{results[split_name]["random_sd"]:.4f}',
              flush=True)

    out_path = '/share/klab/psulewski/psulewski/memgate/random_caption_baseline.json'
    mode = 'r+' if os.path.exists(out_path) else 'x'
    with open(out_path, mode) as fh:
        fh.seek(0)
        json.dump(results, fh, indent=2)
        fh.truncate()
    print(f'wrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
