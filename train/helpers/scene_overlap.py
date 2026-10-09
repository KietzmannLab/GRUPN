# Audit and control the scene overlap between the AVS-MEG stimulus set and the GPN training set.
#
# The glimpse-sequence datasets (coco_{dva}_dg3fix{ext}_r50v{v}ap_{n}fix_{split}.h5) are addressed
# by position, inheriting their row order from ms_coco_embeddings_deepgaze_16_fixations.h5, which
# carries a coco_ids dataset per split. That makes the mapping from an h5 index to a COCO image id
# exact, so AVS scenes can be located in the training splits and held out.
#
# Usage:
#   python -m helpers.scene_overlap --report_out /path/avs_gpn_overlap_report.csv
#   python -m helpers.scene_overlap --exclusion_dir /path/heldout_indices   # writes exclude_{split}.npy
#                                                                           # and safe_coco_ids.txt

import argparse
import os

import h5py
import numpy as np
import pandas as pd

SOURCE_H5 = '/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze_16_fixations.h5'
AVS_SCENES_CSV = '/share/klab/datasets/avs/input/scene_sampling_MEG/experiment_cocoIDs.csv'
CONDS_515 = '/share/klab/datasets/NSD_special_imgs_pythonicDatasetIndices/pythonic_conds515.npy'
GPN_DATASET_TMPL = '/share/klab/datasets/GPN/coco_{dva}_dg3fix{ext}_r50v{bbv}ap_{nfix}fix_{split}.h5'
SPLITS = ('train', 'val', 'test')


def load_avs_scenes():
    """AVS-MEG stimulus list: one row per presented scene, with the NSD membership flags."""
    avs = pd.read_csv(AVS_SCENES_CSV)
    avs['cocoID'] = avs['cocoID'].astype(np.int64)
    return avs


def load_split_coco_ids(source_h5=SOURCE_H5):
    """COCO image id for every row of every split of the source dataset."""
    with h5py.File(source_h5, 'r') as f:
        return {s: np.asarray(f[s]['coco_ids'][:]).astype(np.int64) for s in SPLITS}


def held_out_515_ids(split_ids):
    """COCO ids of the 515 images the `train_515` trainer already holds out of the test split."""
    idx = np.load(CONDS_515)
    return set(split_ids['test'][idx].tolist())


def audit(avs=None, split_ids=None):
    """
    One row per AVS scene: which h5 split holds it, whether `train_515` already holds it out,
    and whether it was therefore a training item (`leaked`).
    """
    avs = load_avs_scenes() if avs is None else avs
    split_ids = load_split_coco_ids() if split_ids is None else split_ids
    held = held_out_515_ids(split_ids)

    report = avs.copy()
    for split in SPLITS:
        report[f'in_h5_{split}'] = report['cocoID'].isin(set(split_ids[split].tolist()))
    report['in_515'] = report['cocoID'].isin(held)
    # train_515 = the whole train split + the test split minus the 515
    report['leaked'] = (report['in_h5_train'] |
                        (report['in_h5_test'] & ~report['in_515']))
    return report


def summarise(report):
    n = len(report)
    leaked = int(report['leaked'].sum())
    print(f'AVS scenes: {n}  (special100={int(report.special100.sum())}, '
          f'shared1000={int(report.shared1000.sum())})')
    for split in SPLITS:
        print(f'  in h5 {split:5s}: {int(report[f"in_h5_{split}"].sum())}')
    print(f'  held out by train_515: {int(report.in_515.sum())}')
    print(f'  LEAKED into training:  {leaked}  ({100 * leaked / n:.1f}%)')
    print(f'  never trained on:      {n - leaked}')
    unmatched = int((~report[[f'in_h5_{s}' for s in SPLITS]].any(axis=1)).sum())
    if unmatched:
        print(f'  WARNING: {unmatched} AVS scenes found in no split')


def exclusion_indices(avs=None, split_ids=None):
    """h5 row indices to drop per split so that no AVS scene is ever a training item."""
    avs = load_avs_scenes() if avs is None else avs
    split_ids = load_split_coco_ids() if split_ids is None else split_ids
    avs_ids = set(avs['cocoID'].unique().tolist())
    return {s: np.flatnonzero(np.isin(split_ids[s], list(avs_ids))).astype(np.int64)
            for s in SPLITS}


def check_dataset_alignment(split, n_source, dva='NSD', ext=91, bbv=6, nfix=7):
    """
    The exclusion indices are positions in the source dataset; they only transfer to the
    glimpse-sequence dataset if both have the same number of images in the same order.
    """
    path = GPN_DATASET_TMPL.format(dva=dva, ext=ext, bbv=bbv, nfix=nfix, split=split)
    if not os.path.exists(path):
        print(f'  {split}: glimpse dataset not found ({path}) — alignment unchecked')
        return
    with h5py.File(path, 'r') as f:
        n = f[split]['mpnet_embeddings'].shape[0]
    status = 'OK' if n == n_source else 'MISMATCH'
    print(f'  {split}: glimpse dataset n={n}, source n={n_source} — {status}')


def main():
    parser = argparse.ArgumentParser(description='AVS/GPN scene overlap audit and held-out indices')
    parser.add_argument('--report_out', type=str, default=None)
    parser.add_argument('--exclusion_dir', type=str, default=None)
    parser.add_argument('--check_alignment', action='store_true')
    # which glimpse dataset the alignment is checked against: 6 = SimCLR, 4 = DVD-B. The
    # exclusion indices themselves are backbone-independent (positions in the source h5), so
    # this only verifies that the r50v{bbv} files inherited the same row order and count.
    parser.add_argument('--bbv', type=int, default=6)
    args = parser.parse_args()

    avs = load_avs_scenes()
    split_ids = load_split_coco_ids()

    report = audit(avs, split_ids)
    summarise(report)

    if args.check_alignment:
        print('Glimpse-dataset alignment:')
        for split in SPLITS:
            check_dataset_alignment(split, len(split_ids[split]), bbv=args.bbv)

    if args.report_out:
        report.to_csv(args.report_out, index=False)
        print(f'Report written: {args.report_out}')

    if args.exclusion_dir:
        os.makedirs(args.exclusion_dir, exist_ok=True)
        for split, idx in exclusion_indices(avs, split_ids).items():
            out = os.path.join(args.exclusion_dir, f'exclude_{split}.npy')
            np.save(out, idx)
            print(f'{split}: {len(idx)} indices -> {out}')

        # COCO ids of the AVS scenes that were never training items, for the probe pack's
        # `clean` flag and for leakage_control_analysis.py
        safe = sorted(report.loc[~report['leaked'], 'cocoID'].astype(np.int64))
        out = os.path.join(args.exclusion_dir, 'safe_coco_ids.txt')
        with open(out, 'w') as fh:
            fh.write(','.join(str(i) for i in safe))
        print(f'never-trained COCO ids: {len(safe)} -> {out}')


if __name__ == '__main__':
    main()
