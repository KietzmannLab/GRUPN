#!/usr/bin/env python
"""
Run the AVS duration probe over every saved epoch of one checkpoint directory.

The probe normally runs inside training. A run trained with `--probe 0` therefore has no betas,
and `--save_every_epoch 1` is what makes them recoverable afterwards: every epoch's weights are
on disk, so the same probe can be applied post hoc and yields the same quantity it would have
logged live.

    python probes/probe_checkpoints.py \\
        --net_dir logs/net_params/<net_name> \\
        --pack <avs_probe_pack[_dvd]_v1.h5> \\
        --loss_npz logs/perf_logs/<net_name>/loss_<net_name>.npz \\
        --out_csv <out>.csv

Hyperparameters come from the directory name through the parser in `gpn_feature_extraction.py`,
compiled out of that source rather than imported, because importing the module pulls in
`avs_gazetime.config`. That is the same idiom `test_backbone_swap.py` uses. The state dict is
loaded strictly, so a misparsed loss flag fails loudly instead of silently probing the wrong
architecture.

The pack stores already-embedded glimpses, so `--pack` must match the backbone the checkpoint was
trained on. The `expect_bbv` guard enforces that from the `_bbv` tag in the directory name.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from probes.avs_duration_probe import DEFAULT_MEMGATE_DIR, AVSDurationProbe  # noqa: E402
from models.GPN import gru_gpn, lstm_gpn                                     # noqa: E402


def load_hyperparameter_parser(memgate_dir):
    """`parse_gru_model_name`, compiled out of gpn_feature_extraction.py."""
    src_path = os.path.join(memgate_dir, 'gpn_feature_extraction.py')
    src = open(src_path).read()
    chunk = src[src.index('def parse_gru_model_name'):src.index('def setup_local_model')]
    ns = {'re': re}
    exec(compile(chunk, src_path, 'exec'), ns)
    return ns['parse_gru_model_name']


def build_net(net_name, hyp):
    cls = gru_gpn if net_name.startswith('gpn_gru') else lstm_gpn
    return cls(timestep_multiplier=hyp['timestep_multiplier'],
               glimpse_loss=hyp['glimpse_loss'], semantic_loss=hyp['semantic_loss'],
               scene_loss=hyp['scene_loss'], gazeloc_loss=hyp['gazeloc_loss'],
               n_rnn=hyp['n_rnn'], regularisation=hyp['regularisation'],
               input_dropout=hyp['input_dropout'], rnn_dropout=hyp['rnn_dropout'],
               input_split=hyp['input_split'], recurrence=hyp['recurrence'],
               input_feats=hyp['input_feats'])


def epoch_checkpoints(net_dir, net_name):
    """(epoch, path) for every `_epoch_N.pth`, epoch -1 (random init) first."""
    out = []
    for p in glob.glob(os.path.join(net_dir, f'{net_name}_epoch_*.pth')):
        m = re.search(r'_epoch_(-?\d+)\.pth$', p)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_dir', required=True)
    parser.add_argument('--pack', required=True)
    parser.add_argument('--out_csv', required=True)
    parser.add_argument('--loss_npz', default=None,
                        help='perf_logs npz, to carry train/val loss into the same table')
    parser.add_argument('--label', default=None, help='value for the `backbone` column')
    parser.add_argument('--layers', default='0')
    parser.add_argument('--stats', default='meta')
    parser.add_argument('--saccade_units', default='train_units')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--memgate_dir', default=DEFAULT_MEMGATE_DIR)
    args = parser.parse_args()

    net_dir = args.net_dir.rstrip('/')
    net_name = os.path.basename(net_dir)
    hyp = load_hyperparameter_parser(args.memgate_dir)(net_name)
    print(f'{net_name}\n  bbv={hyp["bbv"]} n_rnn={hyp["n_rnn"]} tm={hyp["timestep_multiplier"]} '
          f'semc={hyp["semantic_loss"]} provide_loc={hyp["provide_loc"]}')

    ckpts = epoch_checkpoints(net_dir, net_name)
    if not ckpts:
        raise SystemExit(f'no {net_name}_epoch_*.pth in {net_dir} — was it trained with '
                         f'--save_every_epoch 1?')
    print(f'  {len(ckpts)} per-epoch checkpoints: epochs {[e for e, _ in ckpts]}')

    net = build_net(net_name, hyp).to(args.device)
    probe = AVSDurationProbe(pack_path=args.pack,
                             layers=tuple(int(x) for x in args.layers.split(',')),
                             provide_loc=hyp['provide_loc'], device=args.device,
                             stats=args.stats, figure=False,
                             memgate_dir=args.memgate_dir,
                             saccade_units=args.saccade_units,
                             expect_bbv=hyp['bbv'])

    loss = {}
    if args.loss_npz:
        d = np.load(args.loss_npz)
        tr, va = np.asarray(d['train_loss']), np.asarray(d['val_loss'])
        loss = {i + 1: (float(tr[i]), float(va[i])) for i in range(len(va))}
        print(f'  losses for epochs 1-{len(va)}, min val at epoch {int(np.argmin(va)) + 1}')

    rows = []
    for epoch, path in ckpts:
        net.load_state_dict(torch.load(path, map_location=args.device))
        scalars, _ = probe(net, epoch)
        row = {'backbone': args.label or f'bbv{hyp["bbv"]}', 'epoch': epoch,
               'train_loss': loss.get(epoch, (np.nan, np.nan))[0],
               'val_loss': loss.get(epoch, (np.nan, np.nan))[1]}
        row.update({k: v for k, v in scalars.items() if k.startswith('probe/')})
        rows.append(row)
        betas = ', '.join(f'{k.split("/")[-1]}={scalars[k]:+.3f}'
                          for k in sorted(scalars) if k.startswith('probe/beta/'))
        print(f'  epoch {epoch:>3}: {betas}')

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f'\nwrote {args.out_csv} ({len(rows)} epochs)')


if __name__ == '__main__':
    main()
