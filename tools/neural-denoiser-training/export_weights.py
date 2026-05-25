#!/usr/bin/env python3
"""
export_weights.py — Convert a vitrum PyTorch checkpoint to the vitrum binary format.

Standalone: works with any UNetDenoiser checkpoint produced by train.py,
or any PyTorch model whose attribute names match LAYER_ATTR below.

Usage:
  python export_weights.py --pth model.pth --out weights.bin

Output format (.vitrum-model binary):
  Header:   [u32 magic=0xDEAF1984, u32 version=1, u32 layerCount]
  Per layer: [u32 nameLen, char[nameLen] name (UTF-8),
              u32 weightCount, f32[weightCount] weights (OIKW or IOKW),
              u32 biasCount,   f32[biasCount]   biases]

Weight layout per layer kind:
  Conv2D:          OIKW  (outputC × inputC × kH × kW)  ← PyTorch Conv2d default
  ConvTranspose2D: IOKW  (inputC × outputC × kH × kW)  ← PyTorch ConvTranspose2d default
  All others:      No weights (empty arrays).

Both PyTorch layouts are used as-is (no transposition needed):
  torch.Conv2d.weight.shape       = (outC, inC, kH, kW) = OIKW  ✓
  torch.ConvTranspose2d.weight.shape = (inC, outC, kH, kW) = IOKW ✓

This is documented in weights.ts (LayerWeightLayout) and must remain in sync.

Requirements: PyTorch >= 2.0, numpy
"""

import argparse
import os
import struct
import numpy as np
import torch


VITRUM_MODEL_MAGIC   = 0xDEAF1984
VITRUM_MODEL_VERSION = 1

# Ordered list of layer names (must match unetArchitecture.ts LayerSpec.name order).
LAYER_NAMES = [
    'enc1_conv', 'enc1_down',
    'enc2_conv', 'enc2_down',
    'enc3_conv', 'enc3_down',
    'bottleneck',
    'dec3_up',   'dec3_conv',
    'dec2_up',   'dec2_conv',
    'dec1_up',   'dec1_conv',
    'proj',
]

# Map layer name → PyTorch model attribute name (for getattr).
LAYER_ATTR: dict[str, str] = {name: name for name in LAYER_NAMES}


def export(pth_path: str, out_path: str, *, verbose: bool = True) -> None:
    """
    Load a PyTorch checkpoint and export to vitrum binary format.

    Args:
        pth_path: Path to the .pth checkpoint file.
        out_path: Path to write the .vitrum-model binary.
        verbose:  Print per-layer info when True.
    """
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=True)

    # Handle both raw state_dicts and dicts with a 'state_dict' key.
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    records: list[tuple[str, np.ndarray, np.ndarray]] = []

    for name in LAYER_NAMES:
        attr = LAYER_ATTR[name]
        weight_key = f'{attr}.weight'
        bias_key   = f'{attr}.bias'

        if weight_key not in state_dict:
            raise KeyError(
                f"Layer '{attr}' not found in checkpoint. "
                f"Expected key '{weight_key}'. "
                f"Available keys: {sorted(k for k in state_dict if '.weight' in k)}"
            )

        w = state_dict[weight_key].detach().cpu().numpy().flatten().astype(np.float32)
        b = state_dict[bias_key].detach().cpu().numpy().flatten().astype(np.float32) \
            if bias_key in state_dict else np.array([], dtype=np.float32)

        if verbose:
            print(f'  {name:20s}  weights={len(w):8,}  biases={len(b):5,}')

        records.append((name, w, b))

    with open(out_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', VITRUM_MODEL_MAGIC))
        f.write(struct.pack('<I', VITRUM_MODEL_VERSION))
        f.write(struct.pack('<I', len(records)))

        for (name, weights, biases) in records:
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<I', len(weights)))
            f.write(weights.tobytes())
            f.write(struct.pack('<I', len(biases)))
            f.write(biases.tobytes())

    total_params = sum(len(w) + len(b) for (_, w, b) in records)
    size_kb = os.path.getsize(out_path) / 1024
    if verbose:
        print(f'\nTotal parameters: {total_params:,}  ({total_params * 4 / 1024**2:.2f} MB f32)')
        print(f'Output: {out_path} ({size_kb:.1f} KB)')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Convert a vitrum PyTorch checkpoint to the vitrum binary weights format.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--pth', required=True, help='Input PyTorch checkpoint (.pth)')
    p.add_argument('--out', default='weights.bin', help='Output vitrum binary weights path')
    p.add_argument('--quiet', action='store_true', help='Suppress per-layer output')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f'Exporting {args.pth} → {args.out}')
    export(args.pth, args.out, verbose=not args.quiet)
