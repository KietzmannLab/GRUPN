"""
End-to-end smoke test of AVSDurationProbe against a synthetic pack and a real
lstm_gpn / gru_gpn: feature count, scopes, saccade-unit conversion, and that the probe
leaves the network's mode and weights untouched. Needs no cluster data.

    python probes/test_probe_smoke.py [--memgate_dir <avs_gazetime/memgate>]
"""
import argparse
import os
import sys
import tempfile

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from probes.avs_duration_probe import DEFAULT_MEMGATE_DIR  # noqa: E402

_parser = argparse.ArgumentParser()
_parser.add_argument('--memgate_dir', default=DEFAULT_MEMGATE_DIR)
MEMGATE = _parser.parse_args().memgate_dir
PACK = os.path.join(tempfile.mkdtemp(prefix='probe_pack_'), 'mock_probe_pack.h5')

rng = np.random.default_rng(0)
N_SUBJ, N_SEQ, D = 3, 40, 2048

if os.path.exists(PACK):
    os.remove(PACK)
with h5py.File(PACK, 'w') as f:
    f.attrs['pack_version'] = 2
    for k, v in dict(screen_width=1024, screen_height=768, image_width=947,
                     image_height=710, train_image_size=256).items():
        f.attrs[k] = v
    for s in range(1, N_SUBJ + 1):
        lens = rng.integers(3, 12, N_SEQ)
        n = int(lens.sum())
        seq_id = np.repeat(np.arange(N_SEQ), lens).astype(np.int32)
        seq_pos = np.concatenate([np.arange(L) for L in lens]).astype(np.int16)
        embed = rng.normal(0, 1, (n, D)).astype(np.float16)
        # plant a signal: duration depends on the mean embedding, so a gate that tracks the
        # input should show a non-zero beta
        signal = embed.astype(np.float32).mean(1)
        log_dur = (5.6 + 0.4 * signal + rng.normal(0, 0.3, n)).astype(np.float32)
        keep = (seq_pos > 0)
        fix_seq_c = np.where(keep, (seq_pos - seq_pos[keep].mean()) / seq_pos[keep].std(), np.nan)
        g = f.create_group(f'as{s:02d}')
        g.create_dataset('glimpse_embed', data=embed)
        # raw eye-tracker screen-pixel displacements, as pack v2 stores them
        g.create_dataset('saccade_px', data=rng.normal(0, [187., 138.], (n, 2)).astype(np.float32))
        g.create_dataset('seq_id', data=seq_id)
        g.create_dataset('seq_pos', data=seq_pos)
        g.create_dataset('log_dur', data=log_dur)
        g.create_dataset('fix_seq_c', data=fix_seq_c.astype(np.float32))
        g.create_dataset('keep', data=keep)
        g.create_dataset('clean', data=(seq_id % 3 == 0) & keep)

from probes.avs_duration_probe import AVSDurationProbe   # noqa: E402
from models.GPN import lstm_gpn, gru_gpn                  # noqa: E402

ok = True
for name, cls in (('lstm', lstm_gpn), ('gru', gru_gpn)):
    net = cls(timestep_multiplier=1, glimpse_loss=0, semantic_loss=1, n_rnn=64,
              regularisation=1, input_dropout=0.25, rnn_dropout=0.0,
              return_all_actvs=0, input_split=0, recurrence=True, input_feats=D)
    net.train()
    probe = AVSDurationProbe(PACK, layers=(0,), provide_loc=1, device='cpu',
                             chunk_seqs=16, stats='meta', memgate_dir=MEMGATE,
                             saccade_units='train_units')
    legacy = AVSDurationProbe(PACK, layers=(0,), provide_loc=1, device='cpu',
                              chunk_seqs=16, stats='meta', memgate_dir=MEMGATE,
                              saccade_units='legacy')
    sd_aligned = np.abs(np.concatenate([d['saccade_vec'] for d in probe.subjects.values()])).std(0)
    sd_legacy = np.abs(np.concatenate([d['saccade_vec'] for d in legacy.subjects.values()])).std(0)
    print(f'  saccade sd: train_units {sd_aligned.round(2)} vs legacy {sd_legacy.round(4)} '
          f'(ratio {(sd_aligned / sd_legacy).round(0)})')
    scalars, fig = probe(net, epoch=1)

    betas = {k: v for k, v in scalars.items() if k.startswith('probe/beta/')}
    print(f'\n--- {name}: {len(scalars)} scalars, {len(betas)} betas, '
          f'wall {scalars["probe/wall_s"]:.1f}s, n_fix={scalars["probe/n_fix"]} ---')
    for k in sorted(betas):
        se = scalars[k.replace('/beta/', '/se/')]
        print(f'  {k.split("/")[-1]:14s} b={betas[k]:+.4f} se={se:.4f} '
              f'signs={scalars[k.replace("/beta/","/n_same_sign/")]}/{N_SUBJ}')

    # structural checks
    n_gates = 4 if name == 'lstm' else 3
    checks = {
        'net left in training mode': net.training,
        'beta count (gates + 2 drives x 2)': len(betas) == n_gates + 4,
        'clean-scope betas present': any(k.startswith('probe/beta_clean/') for k in scalars),
        'mean_open logged': any(k.startswith('probe/mean_open/') for k in scalars),
        'all betas finite': all(np.isfinite(v) for v in betas.values()),
        'figure returned': fig is not None,
        'train_units saccades are training-scale (sd 30-80)': bool(
            np.all((sd_aligned > 20) & (sd_aligned < 100))),
        'legacy saccades are ~250-300x smaller': bool(
            np.all((sd_aligned / sd_legacy > 200) & (sd_aligned / sd_legacy < 350))),
        'y axis flipped vs legacy': bool(np.sign(
            probe.subjects['as01']['saccade_vec'][1, 1]) != np.sign(
            legacy.subjects['as01']['saccade_vec'][1, 1])),
        'gate openings in [0,1] for sigmoid gates': all(
            0 <= scalars[f'probe/mean_open/{g}_0_main'] <= 1
            for g in (('i', 'f', 'o') if name == 'lstm' else ('r', 'z'))),
    }
    for label, good in checks.items():
        print(f'  {"PASS" if good else "FAIL"}  {label}')
        ok &= bool(good)

    # the probe must not perturb the model
    before = {k: v.clone() for k, v in net.state_dict().items()}
    probe(net, epoch=2)
    same = all(torch.equal(before[k], v) for k, v in net.state_dict().items())
    print(f'  {"PASS" if same else "FAIL"}  weights untouched by the probe')
    ok &= same

print('\n' + ('ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'))
sys.exit(0 if ok else 1)
