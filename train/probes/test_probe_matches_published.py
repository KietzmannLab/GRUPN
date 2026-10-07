"""
Does the probe's gate path equal the published extraction path?

The probe rebuilds the RNN input from the network's projection submodules instead of calling
forward, so this asserts that `joint_proj` and every gate tensor (main / ff / ctx, all layers)
come out identical to `add_gate_extraction_to_gpn(net.forward)` — i.e. that a probe beta and a
`gpn_feature_extraction.py` beta measure the same quantity.

    python probes/test_probe_matches_published.py [--memgate_dir <avs_gazetime/memgate>]
"""
import argparse
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from probes.avs_duration_probe import DEFAULT_MEMGATE_DIR  # noqa: E402

_parser = argparse.ArgumentParser()
_parser.add_argument('--memgate_dir', default=DEFAULT_MEMGATE_DIR)
MEMGATE = _parser.parse_args().memgate_dir
sys.path.insert(0, MEMGATE)

from models.GPN import lstm_gpn, gru_gpn                                      # noqa: E402
from lstm_gate_extractor import (add_gate_extraction_to_gpn,                  # noqa: E402
                                 LSTMGateExtractor, GRUGateExtractor)
from probes.avs_duration_probe import AVSDurationProbe                        # noqa: E402

torch.manual_seed(0)
D, N, H = 2048, 9, 64
ok = True

for name, cls, rnn_attr, gates in (('lstm', lstm_gpn, 'lstm', 'ifgo'), ('gru', gru_gpn, 'gru', 'rzn')):
    net = cls(timestep_multiplier=2, glimpse_loss=0, semantic_loss=1, n_rnn=H,
              regularisation=1, input_dropout=0.25, rnn_dropout=0.0,
              return_all_actvs=1, input_split=0, recurrence=True, input_feats=D)
    net.eval()
    net.forward = types.MethodType(add_gate_extraction_to_gpn(net.forward.__func__), net)

    glimpse = torch.randn(1, N, D)
    saccade = torch.randn(1, N, 2) * 0.3

    # published path: the model's own forward, then gate extraction on representations['joint_proj']
    with torch.no_grad():
        reps, _ = net(glimpse, saccade, return_gate_actvs=True)

    # probe path: joint_proj rebuilt from the submodules, extractor called directly
    probe = AVSDurationProbe.__new__(AVSDurationProbe)   # no pack needed for this comparison
    with torch.no_grad():
        joint = probe._joint_proj(net, glimpse, saccade)
        ext = LSTMGateExtractor(net.lstm) if name == 'lstm' else GRUGateExtractor(net.gru)
        gd = ext.extract_sequence_gates(joint, layers_to_extract=(0, 1))

    d_joint = (joint - reps['joint_proj']).abs().max().item()
    print(f'\n--- {name} ---\n  joint_proj max abs diff: {d_joint:.2e}')
    ok &= d_joint < 1e-6

    worst = 0.0
    for layer in (0, 1):
        for g in gates:
            for suffix in ('', '_ff_mean', '_ctx_mean'):
                key = f'{name}_{g}_{layer}{suffix}'
                a, b = gd[key], reps[key]
                diff = (a - b).abs().max().item()
                worst = max(worst, diff)
    print(f'  all gate tensors (both layers, main/ff/ctx) max abs diff: {worst:.2e}')
    ok &= worst < 1e-6

    # and the scalar the probe actually regresses: mean over hidden units
    probe_mean = gd[f'{name}_{gates[0]}_0'].mean(dim=2).numpy()
    pub_mean = reps[f'{name}_{gates[0]}_0'].mean(dim=2).numpy()
    d = float(np.abs(probe_mean - pub_mean).max())
    print(f'  per-fixation gate mean ({gates[0]}, layer 0) max abs diff: {d:.2e}')
    ok &= d < 1e-6

print('\n' + ('IDENTICAL TO THE PUBLISHED PATH' if ok else 'MISMATCH'))
sys.exit(0 if ok else 1)
