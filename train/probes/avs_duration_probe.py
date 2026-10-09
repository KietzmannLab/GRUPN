"""
Training-time diagnostic: the fixation-duration betas of the RNN's gates and their
feedforward / recurrent drives, measured on real AVS-MEG scene-viewing sequences.

Each call runs the current network over a fixed pack of AVS fixation sequences
(avs-gazetime/avs_gazetime/memgate/build_avs_probe_pack.py), recovers the per-fixation gate
activations, and fits the published duration model

    dur_z_global ~ gate + fix_seq_c      (random intercept per subject, N = 5)

reported in gpn_gate_duration_analysis.py / gate_exploration.ipynb. The fast per-epoch
estimator is a per-subject OLS pooled by inverse variance; passing stats='mixedlm' reproduces
the reference statsmodels fit exactly (slower — use it at milestone epochs).

THIS IS A DIAGNOSTIC. It must never inform checkpoint selection, early stopping or
hyperparameter choice: a beta that was used to pick a model is no longer an independent
result. Selection stays on the validation loss.

Saccade vectors are converted into the units the network was trained on — see
`saccade_units` below and memgate_v2/11_coordinate_alignment.md.

One convention is inherited from the published extraction so that probe betas stay comparable
with it, rather than being "more correct" here: the recurrence is run out to the full AVS
sequence length, beyond the 6 timesteps seen during training.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# the gate extractor is the single source of truth for gate recomputation; the module imports
# only torch, so it can be loaded from the avs-gazetime checkout without pulling in its package
DEFAULT_MEMGATE_DIR = '/home/student/p/psulewski/avs-gazetime/avs_gazetime/memgate'

GATE_SETS = {'lstm': ('i', 'f', 'g', 'o'), 'gru': ('r', 'z', 'n')}
# gates whose feedforward / context decomposition is logged (the two write/update gates)
DRIVE_GATES = {'lstm': ('i', 'f'), 'gru': ('r', 'z')}
GATE_COLORS = {'i': 'cornflowerblue', 'f': 'salmon', 'g': 'mediumseagreen',
               'o': 'mediumpurple', 'r': 'cornflowerblue', 'z': 'salmon',
               'n': 'mediumseagreen'}
GATE_NAMES = {'i': 'input', 'f': 'forget', 'g': 'cell', 'o': 'output',
              'r': 'reset', 'z': 'update', 'n': 'candidate'}
DRIVE_LABEL = {'main': '', 'ff': ' feedforward', 'ctx': ' recurrent'}


def _import_gate_extractors(memgate_dir):
    if memgate_dir not in sys.path:
        sys.path.insert(0, memgate_dir)
    import lstm_gate_extractor as ext
    return ext.LSTMGateExtractor, ext.GRUGateExtractor


class AVSDurationProbe:
    """
    Parameters
    ----------
    pack_path : str
        HDF5 pack written by build_avs_probe_pack.py.
    layers : tuple of int
        RNN layers to report (0 is the first).
    provide_loc : int
        Must match the training setting: 0 zeroes the saccade input, as the trainer does.
    saccade_units : {'train_units', 'legacy'}
        How the pack's raw screen-pixel displacements become network inputs.
        'train_units' (default) matches what the networks are trained on — the displacement
        as a fraction of the stimulus image, in the 256 px units of the COCO glimpse
        datasets, y increasing downward like the image array. 'legacy' reproduces
        extractions from before 2026-10-07 (dgx/1024, dgy/768), which is ~270x too small:
        on the published checkpoint it leaves 53% of coord_proj's 512 units with no
        across-fixation variance, i.e. half the RNN input frozen. Mirrors
        saccade_vectors() in avs_gazetime/memgate/gpn_feature_extraction.py.
    expect_bbv : int or None
        The `--bbv` the network is trained with. The pack stores *already-embedded* glimpses,
        so a pack built with a different backbone silently feeds the network an input
        distribution it never saw. When given, a pack whose `encoder_bbv` attribute differs
        (or is missing) is refused rather than run.
    clean_only : bool
        Restrict to scenes that were never GPN training items (the pack's `clean` flag).
        Required for a valid beta whenever the network trained on AVS scenes; with a
        `_noavs` trainer every scene is clean and both variants are reported anyway.
    stats : {'meta', 'mixedlm'}
        'meta' = per-subject OLS + inverse-variance pooling (no statsmodels needed).
    """

    def __init__(self, pack_path, layers=(0,), provide_loc=1, clean_only=False,
                 chunk_seqs=128, device='cuda', stats='meta', figure=True,
                 memgate_dir=DEFAULT_MEMGATE_DIR, saccade_units='train_units',
                 expect_bbv=None):
        import h5py

        self.layers = tuple(layers)
        self.provide_loc = provide_loc
        self.saccade_units = saccade_units
        self.clean_only = clean_only
        self.chunk_seqs = chunk_seqs
        self.device = device
        self.stats = stats
        self.figure = figure
        self.LSTMGateExtractor, self.GRUGateExtractor = _import_gate_extractors(memgate_dir)

        self.subjects = {}
        with h5py.File(pack_path, 'r') as f:
            self.pack_version = int(f.attrs.get('pack_version', -1))
            if 'saccade_px' not in f[sorted(f.keys())[0]]:
                raise ValueError(f'{pack_path} predates pack v2 (no saccade_px); rebuild it '
                                 f'with build_avs_probe_pack.py')
            self.encoder = str(f.attrs.get('encoder', 'unknown'))
            self.encoder_bbv = (int(f.attrs['encoder_bbv'])
                                if 'encoder_bbv' in f.attrs else None)
            if expect_bbv is not None and self.encoder_bbv != int(expect_bbv):
                raise ValueError(
                    f'probe pack backbone mismatch: {pack_path} was built with '
                    f'encoder_bbv={self.encoder_bbv} ({self.encoder}) but this network trains '
                    f'on bbv={int(expect_bbv)}. The pack stores already-embedded glimpses, so '
                    f'it cannot be reused across backbones — rebuild it with '
                    f'build_avs_probe_pack.py --backbone <matching> and a separate '
                    f'--pack_path. (encoder_bbv=None means the pack predates the backbone '
                    f'stamp; rebuild it.)')
            geometry = {k: float(f.attrs[k]) for k in
                        ('screen_width', 'screen_height', 'image_width', 'image_height',
                         'train_image_size')}
            for subject in sorted(f.keys()):
                g = f[subject]
                self.subjects[subject] = {k: g[k][:] for k in
                                          ('glimpse_embed', 'saccade_px', 'seq_id', 'seq_pos',
                                           'log_dur', 'fix_seq_c', 'keep', 'clean')}
        if not self.subjects:
            raise ValueError(f'no subject groups in {pack_path}')

        for d in self.subjects.values():
            d['saccade_vec'] = self._convert_saccades(d['saccade_px'], geometry)

        # dur_z_global: log duration pooled across subjects over analysed rows, as in the
        # reference pipeline (the random intercept then absorbs between-subject level)
        pooled = np.concatenate([d['log_dur'][d['keep']] for d in self.subjects.values()])
        mu, sd = pooled.mean(), pooled.std()
        for d in self.subjects.values():
            d['dur_z_global'] = ((d['log_dur'] - mu) / sd).astype(np.float32)
            d['sequences'] = self._sequence_slices(d['seq_id'])

        n_fix = sum(int(d['keep'].sum()) for d in self.subjects.values())
        n_clean = sum(int((d['keep'] & d['clean']).sum()) for d in self.subjects.values())
        sacc = np.concatenate([d['saccade_vec'] for d in self.subjects.values()])
        print(f'[probe] {len(self.subjects)} subjects, {n_fix} analysed fixations '
              f'({n_clean} on never-trained scenes), pack v{self.pack_version}, '
              f'encoder {self.encoder}, '
              f'saccade units {self.saccade_units} '
              f'(per-axis sd {np.abs(sacc).std(0).round(3).tolist()})')

    def _convert_saccades(self, px, geometry):
        """Raw eye-tracker screen displacement -> the network's input units."""
        if self.saccade_units == 'train_units':
            scale = np.array([geometry['train_image_size'] / geometry['image_width'],
                              -geometry['train_image_size'] / geometry['image_height']])
        elif self.saccade_units == 'legacy':
            scale = np.array([1.0 / geometry['screen_width'], 1.0 / geometry['screen_height']])
        else:
            raise ValueError(f'unknown saccade units {self.saccade_units!r}')
        return (px.astype(np.float32) * scale).astype(np.float32)

    @staticmethod
    def _sequence_slices(seq_id):
        """(start, stop) row ranges per scene sequence; rows are contiguous per sequence."""
        bounds = np.flatnonzero(np.diff(seq_id)) + 1
        starts = np.concatenate(([0], bounds))
        stops = np.concatenate((bounds, [len(seq_id)]))
        return list(zip(starts.tolist(), stops.tolist()))

    # ------------------------------------------------------------------ gate extraction

    def _rnn_and_type(self, net):
        if hasattr(net, 'lstm') and isinstance(net.lstm, nn.LSTM):
            return net.lstm, 'lstm'
        if hasattr(net, 'gru') and isinstance(net.gru, nn.GRU):
            return net.gru, 'gru'
        raise TypeError('probe needs an nn.LSTM or nn.GRU at net.lstm / net.gru')

    def _joint_proj(self, net, glimpse, saccade):
        """
        The RNN's own input, built from the network's projection layers — the same tensor
        `representations['joint_proj']` that the published extraction feeds the extractor.
        Computed here instead of calling net.forward so no flag on the live training module
        has to be mutated.
        """
        actv = F.relu(net.actv_proj_norm(net.actv_proj(net.actv_dropout(glimpse))))
        coord = F.relu(net.coord_proj_norm(net.coord_proj(saccade)))
        return torch.cat((actv, coord), dim=2)

    def _gate_scalars(self, net, data):
        """
        Per-fixation gate means for one subject: dict feature -> array over all rows
        (NaN where a row was not computed). Sequences are length-sorted into chunks to
        keep padding cheap; padded timesteps are dropped, never averaged over.
        """
        rnn, rnn_type = self._rnn_and_type(net)
        extractor = (self.LSTMGateExtractor(rnn) if rnn_type == 'lstm'
                     else self.GRUGateExtractor(rnn))
        gates = GATE_SETS[rnn_type]
        drives = DRIVE_GATES[rnn_type]

        features = {}
        for layer in self.layers:
            for g in gates:
                features[(g, layer, 'main')] = np.full(len(data['seq_id']), np.nan, np.float32)
            for g in drives:
                for variant in ('ff', 'ctx'):
                    features[(g, layer, variant)] = np.full(len(data['seq_id']), np.nan, np.float32)

        order = sorted(data['sequences'], key=lambda s: s[1] - s[0])
        embed, sacc = data['glimpse_embed'], data['saccade_vec']

        for start in range(0, len(order), self.chunk_seqs):
            chunk = order[start:start + self.chunk_seqs]
            lengths = [stop - beg for beg, stop in chunk]
            t_max = max(lengths)
            x = torch.zeros(len(chunk), t_max, embed.shape[1], dtype=torch.float32)
            s = torch.zeros(len(chunk), t_max, 2, dtype=torch.float32)
            for row, ((beg, stop), length) in enumerate(zip(chunk, lengths)):
                x[row, :length] = torch.from_numpy(embed[beg:stop].astype(np.float32))
                s[row, :length] = torch.from_numpy(sacc[beg:stop])
            x, s = x.to(self.device), s.to(self.device) * self.provide_loc

            joint = self._joint_proj(net, x, s)
            gate_dict = extractor.extract_sequence_gates(joint, layers_to_extract=self.layers)

            for layer in self.layers:
                for g in gates:
                    means = gate_dict[f'{rnn_type}_{g}_{layer}'].mean(dim=2).cpu().numpy()
                    self._scatter(features[(g, layer, 'main')], chunk, lengths, means)
                for g in drives:
                    for variant in ('ff', 'ctx'):
                        means = gate_dict[f'{rnn_type}_{g}_{layer}_{variant}_mean'].cpu().numpy()
                        self._scatter(features[(g, layer, variant)], chunk, lengths, means)
            del gate_dict, joint, x, s

        return features, rnn_type

    @staticmethod
    def _scatter(target, chunk, lengths, means):
        for row, ((beg, stop), length) in enumerate(zip(chunk, lengths)):
            target[beg:stop] = means[row, :length]

    # ------------------------------------------------------------------ statistics

    @staticmethod
    def _ols(y, x, covariate):
        """beta and SE for x in y ~ 1 + x + covariate."""
        X = np.column_stack([np.ones_like(x), x, covariate])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        dof = len(y) - X.shape[1]
        sigma2 = resid @ resid / dof
        xtx_inv = np.linalg.inv(X.T @ X)
        return coef[1], float(np.sqrt(sigma2 * xtx_inv[1, 1]))

    def _pool(self, per_subject):
        """Inverse-variance pooled beta + the across-subject one-sample t (N = 5)."""
        betas = np.array([b for b, _ in per_subject])
        ses = np.array([s for _, s in per_subject])
        w = 1.0 / ses ** 2
        beta = float((betas * w).sum() / w.sum())
        se = float(np.sqrt(1.0 / w.sum()))
        t_subj = float(betas.mean() / (betas.std(ddof=1) / np.sqrt(len(betas)))) if len(betas) > 1 else np.nan
        return beta, se, t_subj, betas

    def _fit(self, feature_by_subject, mask_key):
        """Per-subject OLS of dur_z_global on each feature, then pooled across subjects."""
        out = {}
        keys = next(iter(feature_by_subject.values())).keys()
        for key in keys:
            per_subject = []
            for subject, features in feature_by_subject.items():
                d = self.subjects[subject]
                mask = d['keep'] & np.isfinite(features[key])
                if mask_key == 'clean':
                    mask = mask & d['clean']
                if mask.sum() < 50:
                    continue
                x = features[key][mask].astype(np.float64)
                x = (x - x.mean()) / x.std()                      # per-subject z, as published
                per_subject.append(self._ols(d['dur_z_global'][mask].astype(np.float64), x,
                                             d['fix_seq_c'][mask].astype(np.float64)))
            if len(per_subject) >= 2:
                out[key] = self._pool(per_subject)
        return out

    def _fit_mixedlm(self, feature_by_subject, mask_key):
        """Reference estimator: statsmodels mixed LM with a per-subject random intercept."""
        import pandas as pd
        import statsmodels.formula.api as smf

        frames = []
        for subject, features in feature_by_subject.items():
            d = self.subjects[subject]
            mask = d['keep'].copy()
            if mask_key == 'clean':
                mask = mask & d['clean']
            frame = {'dur_z_global': d['dur_z_global'][mask], 'fix_seq_c': d['fix_seq_c'][mask],
                     'subject': subject}
            for key, values in features.items():
                col = values[mask].astype(np.float64)
                frame['__'.join(map(str, key))] = (col - np.nanmean(col)) / np.nanstd(col)
            frames.append(pd.DataFrame(frame))
        data = pd.concat(frames, ignore_index=True)

        out = {}
        for key in next(iter(feature_by_subject.values())).keys():
            col = '__'.join(map(str, key))
            sub = data[['dur_z_global', 'fix_seq_c', 'subject', col]].dropna()
            res = smf.mixedlm(f"dur_z_global ~ {col} + fix_seq_c", sub,
                              groups=sub['subject']).fit(reml=True)
            out[key] = (float(res.params[col]), float(res.bse[col]))
        return out

    # ------------------------------------------------------------------ reporting

    @staticmethod
    def _label(key, rnn_type):
        gate, layer, variant = key
        return f'{GATE_NAMES[gate]}{DRIVE_LABEL[variant]} L{layer}'

    def _make_figure(self, results, rnn_type):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_context("poster")
        items = sorted(results.items(), key=lambda kv: kv[1][0])
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        for position, (key, (beta, se, _, _)) in enumerate(items):
            ax.barh(position, beta, xerr=se, color=GATE_COLORS[key[0]], edgecolor='white')
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([self._label(k, rnn_type) for k, _ in items])
        ax.axvline(0, color='black')
        ax.set_xlabel('standardised β')
        sns.despine()
        plt.tight_layout()
        figure = plt.gcf()
        return figure

    def __call__(self, net, epoch):
        """Returns (scalars dict for W&B, matplotlib figure or None)."""
        import time
        t0 = time.time()
        was_training = net.training
        net.eval()
        try:
            with torch.no_grad():
                feature_by_subject, rnn_type = {}, None
                for subject, data in self.subjects.items():
                    features, rnn_type = self._gate_scalars(net, data)
                    feature_by_subject[subject] = features

                scalars = {}
                scopes = ['all'] + (['clean'] if any(d['clean'].any() for d in self.subjects.values()) else [])
                results_all = None
                for scope in scopes:
                    results = self._fit(feature_by_subject, 'clean' if scope == 'clean' else None)
                    tag = 'beta' if scope == 'all' else 'beta_clean'
                    for key, (beta, se, t_subj, betas) in results.items():
                        name = f'{key[0]}_{key[1]}_{key[2]}'
                        scalars[f'probe/{tag}/{name}'] = beta
                        scalars[f'probe/{tag.replace("beta", "se")}/{name}'] = se
                        scalars[f'probe/{tag.replace("beta", "t_subj")}/{name}'] = t_subj
                        scalars[f'probe/{tag.replace("beta", "n_same_sign")}/{name}'] = \
                            int((np.sign(betas) == np.sign(beta)).sum())
                    if scope == 'all':
                        results_all = results

                # gate operating points, to catch a collapsed or saturated gate
                for key in results_all:
                    vals = np.concatenate([feature_by_subject[s][key][self.subjects[s]['keep']]
                                           for s in self.subjects])
                    name = f'{key[0]}_{key[1]}_{key[2]}'
                    scalars[f'probe/mean_open/{name}'] = float(np.nanmean(vals))
                    scalars[f'probe/std_open/{name}'] = float(np.nanstd(vals))

                if self.stats == 'mixedlm':
                    try:
                        ref = self._fit_mixedlm(feature_by_subject, None)
                        diffs = [abs(ref[k][0] - results_all[k][0]) for k in ref if k in results_all]
                        for key, (beta, se) in ref.items():
                            scalars[f'probe/beta_mixedlm/{key[0]}_{key[1]}_{key[2]}'] = beta
                        scalars['probe/meta_vs_mixedlm_max_abs_diff'] = float(max(diffs))
                    except ImportError:
                        print('[probe] statsmodels unavailable, skipping the reference fit')

                scalars['probe/n_fix'] = int(sum(int(d['keep'].sum()) for d in self.subjects.values()))
                scalars['probe/wall_s'] = time.time() - t0
                figure = self._make_figure(results_all, rnn_type) if self.figure else None
                return scalars, figure
        finally:
            if was_training:
                net.train()
