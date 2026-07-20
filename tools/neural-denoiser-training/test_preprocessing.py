#!/usr/bin/env python3
"""
test_preprocessing.py — D5/V3-7 regression: training preprocessing must match
the runtime pack shader (packages/walkaround-hybrid/src/shaders/neuralPack.wgsl).

Asserts:
  1. _decode_normal reproduces neuralPack.wgsl:57-58 exactly:
       nd_remapped = nd*2 - 1
       nrm = select(normalize(nd_remapped), (0,1,0), len(nd_remapped) < eps)
  2. The color target container (_load_hdr_np) round-trips LINEAR HDR radiance
     (values > 1.0 survive; NOT reinhard-tonemapped/clamped to [0,1]).

Run directly (no pytest required): python3 test_preprocessing.py
"""
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch

import train  # train.py in the same directory


def _wgsl_reference_decode(packed: np.ndarray) -> np.ndarray:
    """Numpy re-implementation of neuralPack.wgsl:57-58 for a [3] packed normal."""
    remapped = packed * 2.0 - 1.0
    len_sq = float(np.dot(remapped, remapped))
    if len_sq < 1e-6:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return (remapped / np.sqrt(len_sq)).astype(np.float32)


def test_decode_normal_matches_wgsl():
    # A few packed [0,1] normals + one degenerate (sky) pixel.
    packed_pixels = np.array([
        [0.5, 0.5, 1.0],    # +Z rest pose → decodes to (0,0,1)
        [1.0, 0.5, 0.5],    # +X
        [0.0, 0.0, 0.0],    # remapped (-1,-1,-1) — non-degenerate
        [0.5, 0.5, 0.5],    # remapped (0,0,0) — degenerate → up (0,1,0)
    ], dtype=np.float32)  # [4, 3]

    # Lay out as a [3, H, W] tensor (H=1, W=4) like _load_rgb produces.
    normal_t = torch.from_numpy(packed_pixels.T[:, None, :].copy())  # [3, 1, 4]
    decoded = train._decode_normal(normal_t).numpy()                 # [3, 1, 4]

    for i in range(packed_pixels.shape[0]):
        got = decoded[:, 0, i]
        want = _wgsl_reference_decode(packed_pixels[i])
        assert np.allclose(got, want, atol=1e-5), (
            f'pixel {i}: decode {got} != wgsl ref {want}'
        )
        # Non-degenerate decodes must be unit length.
        if not np.allclose(packed_pixels[i], 0.5):
            assert abs(np.linalg.norm(got) - 1.0) < 1e-5
    print('OK  _decode_normal matches neuralPack.wgsl:57-58')


def test_color_target_is_linear_hdr():
    # Write a VHDR .bin with a value > 1 and confirm it survives linearly
    # (reinhard-LDR encoding would clamp/tonemap it into [0,1]).
    w, h = 2, 1
    linear = np.array([[5.0, 0.25, 0.0], [12.0, 100.0, 3.0]], dtype=np.float32)  # [2,3]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'frame_0001.bin'
        with open(p, 'wb') as f:
            f.write(struct.pack('<IIII', train._VHDR_MAGIC, train._VHDR_VERSION, w, h))
            f.write(linear.reshape(-1).astype('<f4').tobytes())
        loaded = train._load_hdr_np(p)  # [H, W, 3]
    assert loaded.shape == (h, w, 3), loaded.shape
    assert np.allclose(loaded.reshape(-1, 3), linear, atol=1e-5), loaded
    # The load must NOT be tonemapped/clamped: HDR values > 1 survive.
    assert loaded.max() > 1.0, 'color target must be linear HDR, not LDR-clamped'
    reinhard = linear / (1.0 + linear)
    assert not np.allclose(loaded.reshape(-1, 3), reinhard), (
        'loaded color must be linear, not reinhard-tonemapped'
    )
    print('OK  color target round-trips linear HDR (not reinhard-LDR)')


if __name__ == '__main__':
    test_decode_normal_matches_wgsl()
    test_color_target_is_linear_hdr()
    print('ALL PREPROCESSING TESTS PASSED')
