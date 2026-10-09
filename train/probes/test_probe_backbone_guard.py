"""
Does the probe refuse a pack whose glimpse encoder does not match the network's --bbv?

The pack stores already-embedded glimpses, so reusing a SimCLR pack for a bbv-4 network would
feed the network an input distribution it never saw and still produce plausible betas. This
asserts the mismatch aborts instead.

    python probes/test_probe_backbone_guard.py        # no cluster data, no GPU

Mock pack only; the gate path itself is covered by test_probe_matches_published.py.
"""
import argparse
import os
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from probes.avs_duration_probe import DEFAULT_MEMGATE_DIR, AVSDurationProbe  # noqa: E402

_parser = argparse.ArgumentParser()
_parser.add_argument('--memgate_dir', default=DEFAULT_MEMGATE_DIR)
MEMGATE = _parser.parse_args().memgate_dir

N, D = 12, 2048


def write_pack(path, encoder_bbv):
    with h5py.File(path, 'w') as f:
        g = f.create_group('as01')
        g['glimpse_embed'] = np.zeros((N, D), np.float16)
        g['saccade_px'] = np.zeros((N, 2), np.float32)
        g['seq_id'] = np.repeat([0, 1, 2], 4).astype(np.int32)
        g['seq_pos'] = np.tile(np.arange(4), 3).astype(np.int16)
        g['log_dur'] = np.linspace(5.0, 6.0, N).astype(np.float32)
        g['fix_seq_c'] = np.zeros(N, np.float32)
        g['keep'] = np.ones(N, bool)
        g['clean'] = np.zeros(N, bool)
        f.attrs['pack_version'] = 2
        for k, v in (('screen_width', 1024), ('screen_height', 768), ('image_width', 947),
                     ('image_height', 710), ('train_image_size', 256)):
            f.attrs[k] = v
        if encoder_bbv is not None:
            f.attrs['encoder'] = f'mock encoder bbv {encoder_bbv}'
            f.attrs['encoder_bbv'] = encoder_bbv


def build(path, expect_bbv):
    return AVSDurationProbe(pack_path=path, device='cpu', figure=False,
                            memgate_dir=MEMGATE, expect_bbv=expect_bbv)


tmp = tempfile.mkdtemp(prefix='probe_pack_')
simclr_pack = os.path.join(tmp, 'pack_bbv6.h5')
dvd_pack = os.path.join(tmp, 'pack_bbv4.h5')
unstamped = os.path.join(tmp, 'pack_unstamped.h5')
write_pack(simclr_pack, 6)
write_pack(dvd_pack, 4)
write_pack(unstamped, None)

ok = True


def check(label, cond):
    global ok
    ok &= bool(cond)
    print(f"  {'OK  ' if cond else 'FAIL'} {label}")


def refuses(path, expect_bbv):
    try:
        build(path, expect_bbv)
        return False
    except ValueError:
        return True


check('bbv 6 network + simclr pack runs', build(simclr_pack, 6).encoder_bbv == 6)
check('bbv 4 network + dvd pack runs', build(dvd_pack, 4).encoder_bbv == 4)
check('bbv 4 network + simclr pack refused', refuses(simclr_pack, 4))
check('bbv 6 network + dvd pack refused', refuses(dvd_pack, 6))
check('pack without an encoder_bbv stamp refused', refuses(unstamped, 6))
check('expect_bbv=None keeps the pre-change behaviour (no check)',
      build(unstamped, None).encoder_bbv is None)

print('\n' + ('BACKBONE GUARD FIRES' if ok else 'FAILURES ABOVE'))
sys.exit(0 if ok else 1)
