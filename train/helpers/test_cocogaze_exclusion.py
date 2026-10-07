"""
Mock-data check of CocoGaze: row selection for every split form, the AVS exclusion, and
parity between the in_memory and on-disk paths. Guards the data path of a multi-day
training job, so it is worth keeping.

    python helpers/test_cocogaze_exclusion.py     # from GRUPN/train, no cluster data needed
"""
import os
import sys
import tempfile

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = tempfile.mkdtemp(prefix='cocogaze_mock_') + '/'
GPN, OPT, HELD = ROOT + 'GPN/', ROOT + 'optimized_datasets/', ROOT + 'heldout/'
for _d in (GPN, OPT, HELD):
    os.makedirs(_d, exist_ok=True)

N = {'train': 12, 'val': 4, 'test': 20}
TRACES, NFIX, T = 3, 7, 6
DS = 'coco_NSD_dg3fix91_r50v6ap_7fix'
CONDS = np.array([2, 5, 9])              # stands in for pythonic_conds{N}: held out of training
EXCL = {'train': np.array([0, 1]), 'val': np.array([3]), 'test': np.array([2, 7, 13])}

# build mock h5s; every value encodes its (split, row) so misrouting is detectable
for split, n in N.items():
    with h5py.File(f'{GPN}{DS}_{split}.h5', 'w') as f:
        g = f.create_group(split)
        tag = {'train': 0, 'val': 100, 'test': 200}[split]
        base = (tag + np.arange(n)).astype(np.float32)
        a = np.zeros((n, TRACES, NFIX, 4, 8), np.float16)
        a[:] = base[:, None, None, None, None]
        a += np.arange(TRACES, dtype=np.float16)[None, :, None, None, None] * 0.5
        a[..., 1, :] = -999                      # a different gaze type must never be read
        g.create_dataset('dg3_fix_actvs', data=a)
        for name, k in (('next_fix_coords', 6), ('next_fix_rel_coords', 6)):
            c = np.zeros((n, TRACES, k, 4, 2), np.int32)
            c[:] = base.astype(np.int32)[:, None, None, None, None]
            g.create_dataset(name, data=c)
        g.create_dataset('mpnet_embeddings', data=np.tile(base[:, None], (1, 5)))
        g.create_dataset('full_image_actvs', data=np.tile(base[:, None], (1, 8)))
with h5py.File(f'{OPT}ms_coco_embeddings.h5', 'w') as f:
    for split, n in N.items():
        tag = {'train': 0, 'val': 100, 'test': 200}[split]
        f.create_group(split).create_dataset(
            'img_multi_hot', data=np.tile((tag + np.arange(n)).astype(np.int64)[:, None], (1, 3)))
for s, idx in EXCL.items():
    np.save(f'{HELD}exclude_{s}.npy', idx)

import helpers.helper_funcs as hf
hf.CONDS_TEMPLATE = ROOT + 'conds{n}.npy'
np.save(ROOT + 'conds3.npy', CONDS)

def build(split, in_memory):
    return hf.CocoGaze(split=split, dataset_path=GPN, gaze_type='dg3', timesteps=T,
                       in_memory=in_memory, bbv=6, dva_dataset='NSD', heldout_dir=HELD)

def rows_of(ds):
    """recover the (tag+row) identity of every image the dataset exposes, via __getitem__"""
    out = []
    for i in range(0, len(ds), TRACES):
        actvs = ds[i][0]
        out.append(int(round(float(actvs[0, 0]))))
    return out

ok = True
def check(name, got, want):
    global ok
    good = got == want
    ok &= good
    print(f'  {"PASS" if good else "FAIL"}  {name}')
    if not good:
        print(f'        got  {got}\n        want {want}')

print('train_3  (train + test minus conds)')
want = [0 + i for i in range(N['train'])] + \
       [200 + i for i in range(N['test']) if i not in CONDS]
for im in (1, 0):
    check(f'in_memory={im}', rows_of(build('train_3', im)), want)

print('train_3_noavs  (also minus the AVS exclusions)')
want_noavs = [0 + i for i in range(N['train']) if i not in EXCL['train']] + \
             [200 + i for i in range(N['test'])
              if i not in CONDS and i not in EXCL['test']]
for im in (1, 0):
    check(f'in_memory={im}', rows_of(build('train_3_noavs', im)), want_noavs)

print('val / val_noavs')
check('val', rows_of(build('val', 1)), [100 + i for i in range(N['val'])])
check('val_noavs', rows_of(build('val_noavs', 1)),
      [100 + i for i in range(N['val']) if i not in EXCL['val']])

print('test_3 (eval split = the conds images)')
check('test_3', rows_of(build('test_3', 1)), [200 + int(i) for i in CONDS])

print('exclusion actually removes the AVS images')
check('no excluded test row present',
      [r for r in want_noavs if r >= 200 and (r - 200) in EXCL['test']], [])
check('counts', len(want_noavs), len(want) - len(EXCL['train']) -
      len([i for i in EXCL['test'] if i not in CONDS]))

print('in_memory parity on every returned tensor, and gaze-type/trace indexing')
a, b = build('train_3_noavs', 1), build('train_3_noavs', 0)
for i in (0, 1, 2, 7, 25, len(a) - 1):
    for k, (x, y) in enumerate(zip(a[i], b[i])):
        if not torch.allclose(x.float(), y.float()):
            print(f'  FAIL  item {i} field {k}'); ok = False
print(f'  {"PASS" if ok else "FAIL"}  tensors identical across in_memory paths')
sample = a[1]            # trace 1 of the dataset's first image (= train row 2 after exclusion)
first_row = int(a.src_row[0])
check('trace offset read correctly',
      round(float(sample[0][0, 0]) - first_row, 1), 0.5)
check('no -999 (wrong gaze type) anywhere', bool((sample[0] == -999).any()), False)
check('fix_coords has the prepended centre', sample[2][0].tolist(), [0.0, 0.0])
check('shapes', [tuple(t.shape) for t in sample[:5]],
      [(T + 1, 8), (T, 2), (T + 1, 2), (5,), (8,)])

print('\n' + ('ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'))
sys.exit(0 if ok else 1)
