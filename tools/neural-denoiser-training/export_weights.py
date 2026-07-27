#!/usr/bin/env python3
"""
export_weights.py — Convert a vitrum PyTorch checkpoint to the vitrum binary format.

Standalone: works with any UNetDenoiser checkpoint produced by train.py,
or any PyTorch model whose attribute names match LAYER_ATTR below.

Usage:
  python export_weights.py --pth model.pth --out weights.bin

Output format (.vitrum-model binary, v2 by default):
  Header:   [u32 magic=0xDEAF1984, u32 version=2, u32 layerCount,
             u32 metadataLength, u8[metadataLength] canonical JSON]
  Legacy v1 (--legacy-v1) omits metadataLength and metadata bytes.
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
import hashlib
import json
import os
import struct
import numpy as np
import torch


VITRUM_MODEL_MAGIC   = 0xDEAF1984
VITRUM_MODEL_VERSION = 2
VITRUM_MODEL_LEGACY_VERSION = 1
MAX_PARAMETER_MAGNITUDE = 1024.0
F16_MAX_ABS_ERROR = 0.05
F16_MAX_MEAN_ABS_ERROR = 0.005
F16_MIN_PSNR_DB = 35.0
NEURAL_ARCHITECTURE_ID = 'vitrum-unet-9x3-v1'
NEURAL_F16_QUANTIZATION = 'f16-storage-per-logical-layer-f32-weight-bias-accumulation'
NEURAL_F16_METRIC_DOMAIN = 'postprocessed-linear-hdr'
EXPECTED_PREPROCESSING = {
    'version': 1,
    'color': 'linear-hdr-scaled',
    'radianceScale': 16.0,
    'radianceClamp': 64.0,
    'albedoRange': [0, 1],
    'normalEncoding': 'signed-world-unit',
    'nonFinite': 'zero',
}

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


def validate_mixed_precision_metadata(metadata: dict) -> None:
    tensor_storage = metadata.get('tensorStorage')
    if tensor_storage not in {None, 'f32', 'f16-compatible'}:
        raise ValueError('metadata.tensorStorage has an unknown enum value')
    report = metadata.get('mixedPrecision')
    if tensor_storage != 'f16-compatible':
        if report is not None and not isinstance(report, dict):
            raise ValueError('metadata.mixedPrecision must be an object')
        return
    if not isinstance(report, dict):
        raise ValueError(
            'f16-compatible tensorStorage requires mixedPrecision certification',
        )
    if report.get('status') != 'pass' or report.get('finiteOutputs') is not True:
        raise ValueError('mixedPrecision certification must pass with finite outputs')
    if report.get('architecture') != NEURAL_ARCHITECTURE_ID:
        raise ValueError('mixedPrecision architecture does not match the runtime U-Net')
    if report.get('preprocessing') != EXPECTED_PREPROCESSING:
        raise ValueError('mixedPrecision preprocessing does not match the runtime contract')
    if report.get('quantization') != NEURAL_F16_QUANTIZATION:
        raise ValueError('mixedPrecision quantization semantics do not match the runtime')
    if report.get('metricDomain') != NEURAL_F16_METRIC_DOMAIN:
        raise ValueError('mixedPrecision metricDomain must be postprocessed-linear-hdr')
    for digest_name in ('checkpointSha256', 'validationCorpusSha256'):
        digest = report.get(digest_name)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in '0123456789abcdef' for char in digest)
        ):
            raise ValueError(f'mixedPrecision.{digest_name} must be a lowercase SHA-256 digest')
    if report.get('accumulation') != 'f32' or report.get('weights') != 'f32':
        raise ValueError('mixedPrecision certification requires f32 accumulation and weights')
    validation_scenes = report.get('validationScenes')
    if (
        isinstance(validation_scenes, bool)
        or not isinstance(validation_scenes, int)
        or validation_scenes < 1
    ):
        raise ValueError('mixedPrecision.validationScenes must be an integer >= 1')
    metrics: dict[str, float] = {}
    for name in ('maxAbsError', 'meanAbsError', 'psnrDb', 'outputMin', 'outputMax'):
        value = report.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f'mixedPrecision.{name} must be finite')
        metrics[name] = float(value)
    if (
        metrics['maxAbsError'] < 0
        or metrics['maxAbsError'] > F16_MAX_ABS_ERROR
        or metrics['meanAbsError'] < 0
        or metrics['meanAbsError'] > F16_MAX_MEAN_ABS_ERROR
        or metrics['psnrDb'] < F16_MIN_PSNR_DB
        or metrics['outputMin'] < 0
        or metrics['outputMin'] > metrics['outputMax']
        or metrics['outputMax'] > EXPECTED_PREPROCESSING['radianceClamp']
    ):
        raise ValueError('mixedPrecision certification exceeds runtime precision/output bounds')


def validate_metadata(metadata: object) -> dict:
    if not isinstance(metadata, dict):
        raise ValueError('v2 export requires checkpoint metadata object')
    string_fields = (
        'id', 'captureSource', 'captureBackend', 'tonemap', 'hardware',
    )
    for field in string_fields:
        if not isinstance(metadata.get(field), str):
            raise ValueError(f'metadata.{field} must be a string')
    for field in ('trainingSamples', 'noisySpp', 'cleanSpp'):
        value = metadata.get(field)
        minimum = 1 if field != 'trainingSamples' else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'metadata.{field} must be an integer >= {minimum}')
    auxiliary = metadata.get('auxiliaryInputs')
    allowed_auxiliary = {'albedo', 'normal', 'depth', 'motion'}
    if (
        not isinstance(auxiliary, list)
        or len(auxiliary) != len(set(auxiliary))
        or any(value not in allowed_auxiliary for value in auxiliary)
    ):
        raise ValueError('metadata.auxiliaryInputs contains invalid or duplicate enum values')
    quality = metadata.get('qualityReport')
    if not isinstance(quality, dict) or quality.get('status') not in {'pass', 'fail', 'unknown'}:
        raise ValueError('metadata.qualityReport.status has an unknown enum value')
    if metadata.get('preprocessing') != EXPECTED_PREPROCESSING:
        raise ValueError('metadata.preprocessing does not match the runtime contract')
    validate_mixed_precision_metadata(metadata)
    # Canonical JSON rejects NaN/Infinity in optional quality metrics.
    json.dumps(metadata, allow_nan=False)
    return metadata


def canonical_metadata_bytes(metadata: dict) -> bytes:
    return json.dumps(
        validate_metadata(metadata),
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')


def validate_parameters(values: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{label} contains non-finite values')
    if values.size and float(np.max(np.abs(values))) > MAX_PARAMETER_MAGNITUDE:
        raise ValueError(f'{label} exceeds magnitude bound {MAX_PARAMETER_MAGNITUDE:g}')


def ordered_tensor_digest(records: list[tuple[str, np.ndarray, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, weights, biases in records:
        if weights.size == 0 and biases.size == 0:
            continue
        name_bytes = name.encode('utf-8')
        digest.update(struct.pack('<I', len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack('<I', len(weights)))
        digest.update(np.asarray(weights, dtype='<f4').tobytes())
        digest.update(struct.pack('<I', len(biases)))
        digest.update(np.asarray(biases, dtype='<f4').tobytes())
    return digest.hexdigest()


def export(
    pth_path: str,
    out_path: str,
    *,
    verbose: bool = True,
    legacy_v1: bool = False,
    mixed_precision_report_path: str | None = None,
) -> None:
    """
    Load a PyTorch checkpoint and export to vitrum binary format.

    Args:
        pth_path: Path to the .pth checkpoint file.
        out_path: Path to write the .vitrum-model binary.
        verbose:  Print per-layer info when True.
    """
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=True)
    version = VITRUM_MODEL_LEGACY_VERSION if legacy_v1 else VITRUM_MODEL_VERSION
    metadata_bytes = b''
    metadata: dict | None = None
    if version == VITRUM_MODEL_VERSION:
        if not isinstance(checkpoint, dict) or 'metadata' not in checkpoint:
            raise ValueError(
                'v2 export requires a train.py checkpoint with metadata; use --legacy-v1 only for compatibility',
            )
        metadata = dict(checkpoint['metadata'])
        if mixed_precision_report_path is not None:
            with open(mixed_precision_report_path, encoding='utf-8') as report_file:
                report = json.load(report_file)
            metadata['tensorStorage'] = 'f16-compatible'
            metadata['mixedPrecision'] = report
    elif mixed_precision_report_path is not None:
        raise ValueError('--mixed-precision-report requires the metadata-bearing v2 format')

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

        validate_parameters(w, f'{name}.weights')
        validate_parameters(b, f'{name}.biases')
        if verbose:
            print(f'  {name:20s}  weights={len(w):8,}  biases={len(b):5,}')

        records.append((name, w, b))

    if metadata is not None:
        actual_checkpoint_digest = ordered_tensor_digest(records)
        expected_checkpoint_digest = metadata.get('mixedPrecision', {}).get('checkpointSha256')
        if metadata.get('tensorStorage') == 'f16-compatible' and expected_checkpoint_digest != actual_checkpoint_digest:
            raise ValueError(
                'mixedPrecision.checkpointSha256 does not match the exact ordered checkpoint tensors',
            )
        metadata_bytes = canonical_metadata_bytes(metadata)

    with open(out_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', VITRUM_MODEL_MAGIC))
        f.write(struct.pack('<I', version))
        f.write(struct.pack('<I', len(records)))
        if version == VITRUM_MODEL_VERSION:
            f.write(struct.pack('<I', len(metadata_bytes)))
            f.write(metadata_bytes)

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
    p.add_argument('--legacy-v1', action='store_true', help='Write metadata-free v1 compatibility output (never production-ready)')
    p.add_argument(
        '--mixed-precision-report',
        help='Validated JSON metrics; certifies f16 tensor storage only when all runtime bounds pass',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f'Exporting {args.pth} → {args.out}')
    export(
        args.pth,
        args.out,
        verbose=not args.quiet,
        legacy_v1=args.legacy_v1,
        mixed_precision_report_path=args.mixed_precision_report,
    )
