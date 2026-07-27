#!/usr/bin/env python3
"""
train.py — vitrum neural denoiser training script.

Trains a U-Net denoiser matching the architecture in:
  packages/walkaround-hybrid/src/neural/unetArchitecture.ts

Architecture reference:
  Ronneberger, Fischer, Brox 2015 "U-Net: Convolutional Networks for
  Biomedical Image Segmentation." MICCAI. https://arxiv.org/abs/1505.04597

  Chaitanya et al. 2017 "Interactive Reconstruction of Monte Carlo Image
  Sequences using a Recurrent Denoising Autoencoder." SIGGRAPH.
  https://doi.org/10.1145/3072959.3073601

Loss: L1 + SSIM composite (Chaitanya 2017 §4: perceptual quality + L1 stability).

Usage:
  python train.py --data path/to/noisy_clean_pairs/ --epochs 50 --batch 4 --lr 1e-4

Dataset format: see dataset_spec.md

Output:
  model.pth         — PyTorch checkpoint (full model state_dict + metadata)
  weights.bin       — vitrum binary weights (same as export_weights.py output)

Requirements: PyTorch >= 2.0, torchvision, Pillow, numpy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
from PIL import Image

# NOTE: torch is imported lazily inside train() so that `--dry-run` (numpy-only
# dataset + shape + export-path validation) works on boxes WITHOUT PyTorch
# installed. Do NOT add a top-level `import torch` — it would break --dry-run.
try:                       # pragma: no cover - environment dependent
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except ImportError:        # pragma: no cover - environment dependent
    _HAS_TORCH = False
    # Provide a stub base class so the UNetDenoiser definition below still parses.
    class _StubModule:
        pass
    nn = type('nn', (), {'Module': _StubModule})  # type: ignore


# ── Runtime/training preprocessing contract ─────────────────────────────────
# Must remain byte-for-byte semantically aligned with
# packages/walkaround-hybrid/src/neural/preprocessing.ts.

NEURAL_RADIANCE_SCALE = 16.0
NEURAL_RADIANCE_CLAMP = 64.0
MAX_PARAMETER_MAGNITUDE = 1024.0
F16_MAX_ABS_ERROR = 0.05
F16_MAX_MEAN_ABS_ERROR = 0.005
F16_MIN_PSNR_DB = 35.0
NEURAL_ARCHITECTURE_ID = 'vitrum-unet-9x3-v1'
NEURAL_F16_QUANTIZATION = 'f16-storage-per-logical-layer-f32-weight-bias-accumulation'
NEURAL_F16_METRIC_DOMAIN = 'postprocessed-linear-hdr'
NEURAL_PREPROCESSING = {
    'version': 1,
    'color': 'linear-hdr-scaled',
    'radianceScale': NEURAL_RADIANCE_SCALE,
    'radianceClamp': NEURAL_RADIANCE_CLAMP,
    'albedoRange': [0, 1],
    'normalEncoding': 'signed-world-unit',
    'nonFinite': 'zero',
}


def preprocess_radiance_np(values: np.ndarray) -> np.ndarray:
    """Finite-safe linear-HDR preprocessing used for noisy inputs and clean targets."""
    finite = np.where(np.isfinite(values), values, 0.0)
    return (
        np.clip(finite, 0.0, NEURAL_RADIANCE_CLAMP) / NEURAL_RADIANCE_SCALE
    ).astype(np.float32)


def postprocess_radiance_np(values: np.ndarray) -> np.ndarray:
    """Inverse runtime output transform for tests and offline inspection."""
    finite = np.where(np.isfinite(values), values, 0.0)
    return np.clip(
        finite * NEURAL_RADIANCE_SCALE,
        0.0,
        NEURAL_RADIANCE_CLAMP,
    ).astype(np.float32)


def preprocess_radiance(values):
    """Torch mirror of preprocess_radiance_np, kept untyped for --dry-run."""
    finite = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    return torch.clamp(finite, 0.0, NEURAL_RADIANCE_CLAMP) / NEURAL_RADIANCE_SCALE


def sanitize_albedo(values):
    finite = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    return torch.clamp(finite, 0.0, 1.0)


def canonical_json_bytes(value: dict) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    ).encode('utf-8')


def validate_export_parameters(values: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{label} contains non-finite values')
    if values.size and float(np.max(np.abs(values))) > MAX_PARAMETER_MAGNITUDE:
        raise ValueError(f'{label} exceeds magnitude bound {MAX_PARAMETER_MAGNITUDE:g}')


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
        raise ValueError('f16-compatible tensorStorage requires mixedPrecision certification')
    if report.get('status') != 'pass' or report.get('finiteOutputs') is not True:
        raise ValueError('mixedPrecision certification must pass with finite outputs')
    if report.get('architecture') != NEURAL_ARCHITECTURE_ID:
        raise ValueError('mixedPrecision architecture does not match the runtime U-Net')
    if report.get('preprocessing') != NEURAL_PREPROCESSING:
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
        or metrics['outputMax'] > NEURAL_RADIANCE_CLAMP
    ):
        raise ValueError('mixedPrecision certification exceeds runtime precision/output bounds')


def validate_checkpoint_metadata(metadata: dict) -> None:
    if not isinstance(metadata, dict):
        raise ValueError('v2 export requires checkpoint metadata object')
    if metadata.get('preprocessing') != NEURAL_PREPROCESSING:
        raise ValueError('metadata.preprocessing does not match the runtime contract')
    quality = metadata.get('qualityReport')
    if not isinstance(quality, dict) or quality.get('status') not in {'pass', 'fail', 'unknown'}:
        raise ValueError('metadata.qualityReport.status has an unknown enum value')
    validate_mixed_precision_metadata(metadata)
    canonical_json_bytes(metadata)


def checkpoint_metadata(
    *,
    checkpoint_id: str,
    training_samples: int,
    noisy_spp: int,
    clean_spp: int,
    capture_source: str,
    capture_backend: str,
    hardware: str,
) -> dict:
    """Create a schema-complete v2 checkpoint record; quality starts unknown."""
    return {
        'id': checkpoint_id,
        'trainingSamples': int(training_samples),
        'noisySpp': int(noisy_spp),
        'cleanSpp': int(clean_spp),
        'auxiliaryInputs': ['albedo', 'normal'],
        'captureSource': capture_source,
        'captureBackend': capture_backend,
        'tonemap': 'linear-hdr',
        'hardware': hardware,
        'preprocessing': dict(NEURAL_PREPROCESSING),
        'tensorStorage': 'f32',
        'qualityReport': {'status': 'unknown'},
    }


# ── Architecture ─────────────────────────────────────────────────────────────
# Must match unetArchitecture.ts exactly.

class UNetDenoiser(nn.Module):
    """
    Vitrum U-Net denoiser: 9-channel input, 3-channel RGB output.

    Channel widths:
      Encoder:    9 → 24 → 48 → 96 → 192 (bottleneck)
      Decoder: 192 → 96 → 48 → 24 → 3
    Parameter count: ~535,107 (~2.04 MB f32, ~1.02 MB f16).
    (Earlier docs cited ~426,075; that figure excluded the three strided
    down-conv layers enc1_down/enc2_down/enc3_down — 109,032 params.
    Canonical source: packages/walkaround-hybrid/src/neural/unetArchitecture.ts)

    No batch norm: inference is per-frame; BN would require re-normalization
    at runtime across frames. Uses ReLU activations only.

    Skip connections: element-wise addition (not concatenation), so decoder
    channels match encoder channels exactly. This is the Bug 1 fix:
    skip-add operands have matching (H, W, C) by construction.
    """

    def __init__(self):
        super().__init__()

        # ── Encoder ───────────────────────────────────────────────────────
        # Level 1: stride-1 feature conv + stride-2 downsampler
        self.enc1_conv = nn.Conv2d(9, 24, 3, stride=1, padding=1, bias=True)
        self.enc1_down = nn.Conv2d(24, 24, 3, stride=2, padding=1, bias=True)

        # Level 2
        self.enc2_conv = nn.Conv2d(24, 48, 3, stride=1, padding=1, bias=True)
        self.enc2_down = nn.Conv2d(48, 48, 3, stride=2, padding=1, bias=True)

        # Level 3
        self.enc3_conv = nn.Conv2d(48, 96, 3, stride=1, padding=1, bias=True)
        self.enc3_down = nn.Conv2d(96, 96, 3, stride=2, padding=1, bias=True)

        # Bottleneck (stride 1, no spatial change)
        self.bottleneck = nn.Conv2d(96, 192, 3, stride=1, padding=1, bias=True)

        # ── Decoder ───────────────────────────────────────────────────────
        # Level 3: tconv from H/8 → H/4 (192→96), then conv
        # Skip source: enc3_feat (H/4 × W/4 × 96) ← matches dec3_up output ✓
        self.dec3_up   = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec3_conv = nn.Conv2d(96, 96, 3, stride=1, padding=1, bias=True)

        # Level 2: tconv from H/4 → H/2 (96→48)
        # Skip source: enc2_feat (H/2 × W/2 × 48) ✓
        self.dec2_up   = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec2_conv = nn.Conv2d(48, 48, 3, stride=1, padding=1, bias=True)

        # Level 1: tconv from H/2 → H (48→24)
        # Skip source: enc1_feat (H × W × 24) ✓
        self.dec1_up   = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec1_conv = nn.Conv2d(24, 24, 3, stride=1, padding=1, bias=True)

        # Output projection (1×1 conv, no activation — linear RGB output)
        self.proj = nn.Conv2d(24, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 9, H, W] — noisyColor (3) + albedo (3) + normals (3)
        Returns:
            [B, 3, H, W] — denoised RGB
        """
        # ── Encoder ───────────────────────────────────────────────────────
        # Level 1
        s1 = F.relu(self.enc1_conv(x))       # [B, 24, H, W]   ← skip1
        e1 = self.enc1_down(s1)              # [B, 24, H/2, W/2]

        # Level 2
        s2 = F.relu(self.enc2_conv(e1))      # [B, 48, H/2, W/2]  ← skip2
        e2 = self.enc2_down(s2)              # [B, 48, H/4, W/4]

        # Level 3
        s3 = F.relu(self.enc3_conv(e2))      # [B, 96, H/4, W/4]  ← skip3
        e3 = self.enc3_down(s3)              # [B, 96, H/8, W/8]

        # Bottleneck
        bn = F.relu(self.bottleneck(e3))     # [B, 192, H/8, W/8]

        # ── Decoder ───────────────────────────────────────────────────────
        # Level 3: tconv H/8 → H/4, skip-add enc3_feat (H/4 × 96)
        d3 = self.dec3_up(bn)                # [B, 96, H/4, W/4]
        d3 = F.relu(self.dec3_conv(d3 + s3)) # skip-add + conv + relu

        # Level 2: tconv H/4 → H/2, skip-add enc2_feat (H/2 × 48)
        d2 = self.dec2_up(d3)                # [B, 48, H/2, W/2]
        d2 = F.relu(self.dec2_conv(d2 + s2)) # skip-add + conv + relu

        # Level 1: tconv H/2 → H, skip-add enc1_feat (H × 24)
        d1 = self.dec1_up(d2)                # [B, 24, H, W]
        d1 = F.relu(self.dec1_conv(d1 + s1)) # skip-add + conv + relu

        # Output projection
        return self.proj(d1)                 # [B, 3, H, W]


# ── Dataset ──────────────────────────────────────────────────────────────────

class DenoisingDataset(Dataset if _HAS_TORCH else object):
    """
    Paired noisy / clean dataset loader.

    Directory structure (see dataset_spec.md):
      data/
        scene_name/
          noisy/
            frame_0001.bin        ← 1 spp path-traced (linear-HDR RGB, VHDR .bin)
            frame_0001_albedo.png  ← albedo G-buffer (RGB [0,1])
            frame_0001_normal.png  ← world normals (packed RGB [0,1]; decoded to [-1,1])
          clean/
            frame_0001.bin        ← high-spp reference (linear-HDR RGB, VHDR .bin)

    Encoding alignment (matches the runtime pack shader neuralPack.wgsl):
      • color inputs (noisy/clean) are linear HDR — NOT reinhard-tonemapped LDR —
        so the training target matches the runtime raw-linear color input.
      • the normal channel is decoded [0,1]->[-1,1] and renormalized in
        __getitem__ (see _decode_normal), mirroring neuralPack.wgsl:57-58.
    """

    def __init__(self, data_dir: str, patch_size: int = 256):
        self.patch_size = patch_size
        self.samples: list[tuple[Path, Path, Path, Path]] = []

        data_path = Path(data_dir)
        for scene_dir in sorted(data_path.iterdir()):
            if not scene_dir.is_dir():
                continue
            noisy_dir  = scene_dir / 'noisy'
            clean_dir  = scene_dir / 'clean'
            if not noisy_dir.exists() or not clean_dir.exists():
                continue
            # Color targets are linear-HDR .bin (see capture-dataset.mjs);
            # albedo/normal G-buffers remain [0,1] PNG.
            for noisy_path in sorted(noisy_dir.glob('frame_*.bin')):
                stem = noisy_path.stem
                albedo_path = noisy_dir / f'{stem}_albedo.png'
                normal_path = noisy_dir / f'{stem}_normal.png'
                clean_path  = clean_dir / f'{stem}.bin'
                if albedo_path.exists() and normal_path.exists() and clean_path.exists():
                    self.samples.append((noisy_path, albedo_path, normal_path, clean_path))

        if not self.samples:
            raise ValueError(
                f'No valid training samples found in {data_dir}. '
                f'Expected scene/noisy/frame_*.bin (linear-HDR color) + frame_*_albedo.png '
                f'+ frame_*_normal.png and scene/clean/frame_*.bin pairs. See dataset_spec.md.'
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        noisy_path, albedo_path, normal_path, clean_path = self.samples[idx]

        noisy  = _load_hdr(noisy_path)   # [3, H, W] float32 linear HDR
        albedo = _load_rgb(albedo_path)  # [3, H, W] float32 [0,1]
        normal = _load_rgb(normal_path)  # [3, H, W] float32 [0,1] (packed)
        clean  = _load_hdr(clean_path)   # [3, H, W] float32 linear HDR

        # Runtime alignment: the outer pack sanitizes signed world normals and
        # the internal input pack applies the versioned HDR scale/clamp exactly
        # once. Training applies the same transforms to inputs and targets.
        normal = _decode_normal(normal)
        noisy, clean = preprocess_radiance(noisy), preprocess_radiance(clean)
        albedo = sanitize_albedo(albedo)

        # Stack to 9-channel input.
        x = torch.cat([noisy, albedo, normal], dim=0)  # [9, H, W]

        # Random crop for training.
        _, H, W = x.shape
        p = self.patch_size
        if H > p and W > p:
            y0 = torch.randint(0, H - p, (1,)).item()
            x0 = torch.randint(0, W - p, (1,)).item()
            x     = x[:, y0:y0+p, x0:x0+p]
            clean = clean[:, y0:y0+p, x0:x0+p]

        return x, clean


def _load_rgb(path: Path) -> torch.Tensor:
    """Load a PNG as a float32 tensor [3, H, W] in [0, 1]."""
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
    return torch.from_numpy(arr).permute(2, 0, 1)   # [3, H, W]


# Linear-HDR color container written by capture-dataset.mjs. Header (little-endian):
#   u32 magic ('VHDR' = 0x52444856), u32 version, u32 width, u32 height,
# then width*height*3 float32 linear radiance (row-major, interleaved RGB).
_VHDR_MAGIC   = 0x52444856
_VHDR_VERSION = 1


def _load_hdr_np(path: Path) -> np.ndarray:
    """Load a .bin linear-HDR color target as a float32 array [H, W, 3] (linear)."""
    with open(path, 'rb') as f:
        header = np.frombuffer(f.read(16), dtype='<u4')
        magic, version, width, height = (int(header[0]), int(header[1]),
                                         int(header[2]), int(header[3]))
        if magic != _VHDR_MAGIC:
            raise ValueError(f'{path}: bad VHDR magic 0x{magic:08X} (expected 0x{_VHDR_MAGIC:08X})')
        if version != _VHDR_VERSION:
            raise ValueError(f'{path}: unsupported VHDR version {version} (expected {_VHDR_VERSION})')
        data = np.frombuffer(f.read(width * height * 3 * 4), dtype='<f4')
    return data.reshape(height, width, 3).astype(np.float32)  # linear, NOT tonemapped


def _load_hdr(path: Path) -> torch.Tensor:
    """Load a .bin linear-HDR color target as a float32 tensor [3, H, W] (linear)."""
    arr = _load_hdr_np(path)                          # [H, W, 3] linear
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)  # [3, H, W]


def _decode_normal(normal: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Decode a packed [0,1] normal tensor [3, H, W] to a renormalized [-1,1]
    unit-vector field, mirroring the runtime pack shader neuralPack.wgsl:57-58
    (nd_remapped = nd*2-1; select(normalize(nd_remapped), (0,1,0), len<eps)).
    Zero-length (sky/background) normals fall back to geometric-up (0,1,0).
    """
    packed = torch.where(torch.isfinite(normal), normal, torch.full_like(normal, 0.5))
    packed = torch.clamp(packed, 0.0, 1.0)
    remapped = packed * 2.0 - 1.0                     # [0,1] -> [-1,1]
    len_sq = (remapped * remapped).sum(dim=0, keepdim=True)  # [1, H, W]
    norm = torch.sqrt(torch.clamp(len_sq, min=eps))
    decoded = remapped / norm
    up = torch.zeros_like(decoded)
    up[1] = 1.0
    return torch.where(len_sq < eps, up, decoded)


# ── Loss ─────────────────────────────────────────────────────────────────────

def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Simplified SSIM loss (1 - SSIM). Chaitanya 2017 uses L1 + SSIM composite.
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    B, C, H, W = pred.shape

    # Gaussian window.
    sigma = 1.5
    kernel_1d = torch.exp(
        -torch.arange(window_size, dtype=torch.float32).sub(window_size // 2).pow(2) / (2 * sigma ** 2)
    )
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.outer(kernel_1d).unsqueeze(0).unsqueeze(0)  # [1,1,k,k]
    kernel_2d = kernel_2d.expand(C, -1, -1, -1).to(pred.device)

    pad = window_size // 2

    def conv(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel_2d, padding=pad, groups=C)

    mu1 = conv(pred)
    mu2 = conv(target)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv(pred * pred) - mu1_sq
    sigma2_sq = conv(target * target) - mu2_sq
    sigma12   = conv(pred * target) - mu1_mu2

    ssim_map = (
        (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    )
    return 1.0 - ssim_map.mean()


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 + SSIM composite loss per Chaitanya 2017."""
    l1   = F.l1_loss(pred, target)
    ssim = ssim_loss(pred, target)
    return l1 + 0.1 * ssim


# ── Vitrum binary export ──────────────────────────────────────────────────────

VITRUM_MODEL_MAGIC   = 0xDEAF1984
VITRUM_MODEL_VERSION = 2

# Canonical layer names (must match unetArchitecture.ts LayerSpec.name).
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

# Map layer name → model attribute (PyTorch nn.Module attribute name).
LAYER_ATTR: dict[str, str] = {
    'enc1_conv':  'enc1_conv',
    'enc1_down':  'enc1_down',
    'enc2_conv':  'enc2_conv',
    'enc2_down':  'enc2_down',
    'enc3_conv':  'enc3_conv',
    'enc3_down':  'enc3_down',
    'bottleneck': 'bottleneck',
    'dec3_up':    'dec3_up',
    'dec3_conv':  'dec3_conv',
    'dec2_up':    'dec2_up',
    'dec2_conv':  'dec2_conv',
    'dec1_up':    'dec1_up',
    'dec1_conv':  'dec1_conv',
    'proj':       'proj',
}


def export_vitrum_weights(model: UNetDenoiser, out_path: str, metadata: dict) -> None:
    """
    Export model weights to the vitrum binary format.
    Same format as export_weights.py — see weights.ts for byte-level layout.
    """
    records: list[tuple[str, np.ndarray, np.ndarray]] = []

    for name in LAYER_NAMES:
        attr = LAYER_ATTR[name]
        layer = getattr(model, attr)

        w = layer.weight.detach().cpu().numpy().flatten().astype(np.float32)
        b = layer.bias.detach().cpu().numpy().flatten().astype(np.float32)
        records.append((name, w, b))

    write_vitrum_binary(records, out_path, metadata)

    print(f'Exported vitrum weights → {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)')


# ── Canonical layer shapes (single source of truth, mirrors unetArchitecture.ts) ─
# Each entry: name → (weight_shape, bias_len). Conv2d weight is OIKW
# (outC, inC, kH, kW); ConvTranspose2d weight is IOKW (inC, outC, kH, kW).
# This drives BOTH the numpy dry-run export and the canonical param-count check.
CANONICAL_LAYERS: list[tuple[str, tuple[int, int, int, int], int]] = [
    ('enc1_conv',  (24,   9, 3, 3),  24),
    ('enc1_down',  (24,  24, 3, 3),  24),
    ('enc2_conv',  (48,  24, 3, 3),  48),
    ('enc2_down',  (48,  48, 3, 3),  48),
    ('enc3_conv',  (96,  48, 3, 3),  96),
    ('enc3_down',  (96,  96, 3, 3),  96),
    ('bottleneck', (192, 96, 3, 3), 192),
    ('dec3_up',    (192, 96, 2, 2),  96),   # IOKW (inC=192, outC=96)
    ('dec3_conv',  (96,  96, 3, 3),  96),
    ('dec2_up',    (96,  48, 2, 2),  48),   # IOKW
    ('dec2_conv',  (48,  48, 3, 3),  48),
    ('dec1_up',    (48,  24, 2, 2),  24),   # IOKW
    ('dec1_conv',  (24,  24, 3, 3),  24),
    ('proj',       (3,   24, 1, 1),   3),
]
CANONICAL_PARAM_COUNT = 535107   # pinned by neural.test.ts deriveParamCount.


def canonical_param_count() -> int:
    total = 0
    for _name, wshape, blen in CANONICAL_LAYERS:
        n = 1
        for d in wshape:
            n *= d
        total += n + blen
    return total


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


def write_vitrum_binary(
    records: list[tuple[str, np.ndarray, np.ndarray]],
    out_path: str,
    metadata: dict,
) -> int:
    """Write the .vitrum-model binary (loadWeightsFromArrayBuffer schema). Returns total params."""
    validate_checkpoint_metadata(metadata)
    for name, weights, biases in records:
        validate_export_parameters(weights, f'{name}.weights')
        validate_export_parameters(biases, f'{name}.biases')

    if (
        metadata.get('tensorStorage') == 'f16-compatible'
        and metadata['mixedPrecision']['checkpointSha256'] != ordered_tensor_digest(records)
    ):
        raise ValueError('mixedPrecision.checkpointSha256 does not match the exact ordered checkpoint tensors')

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<I', VITRUM_MODEL_MAGIC))
        f.write(struct.pack('<I', VITRUM_MODEL_VERSION))
        f.write(struct.pack('<I', len(records)))
        metadata_bytes = canonical_json_bytes(metadata)
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
    return sum(len(w) + len(b) for (_, w, b) in records)


def enumerate_samples(data_dir: str) -> list[tuple[Path, Path, Path, Path]]:
    """Numpy-only mirror of DenoisingDataset's sample discovery (no torch)."""
    samples: list[tuple[Path, Path, Path, Path]] = []
    data_path = Path(data_dir)
    for scene_dir in sorted(data_path.iterdir()):
        if not scene_dir.is_dir():
            continue
        noisy_dir, clean_dir = scene_dir / 'noisy', scene_dir / 'clean'
        if not noisy_dir.exists() or not clean_dir.exists():
            continue
        for noisy_path in sorted(noisy_dir.glob('frame_*.bin')):
            stem = noisy_path.stem
            albedo = noisy_dir / f'{stem}_albedo.png'
            normal = noisy_dir / f'{stem}_normal.png'
            clean  = clean_dir / f'{stem}.bin'
            if albedo.exists() and normal.exists() and clean.exists():
                samples.append((noisy_path, albedo, normal, clean))
    return samples


def dry_run(args: argparse.Namespace) -> None:
    """
    Numpy-only end-to-end validation (NO torch required):
      1. Discover noisy/clean pairs in --data.
      2. Load + stack the 9-channel input, assert shapes.
      3. Build a deterministic random checkpoint matching the canonical
         layer shapes, write the .vitrum-model binary, assert 535,107 params.
    This exercises the dataset-loading + export format path so the round-trip
    test (walkaround-hybrid) can load the result through the real weights.ts loader.
    """
    print('=== train.py --dry-run (numpy-only; PyTorch NOT required) ===')

    # 1. Dataset discovery.
    samples = enumerate_samples(args.data)
    if not samples:
        raise SystemExit(
            f'No valid pairs in {args.data}. Run capture-dataset.mjs first. See dataset_spec.md.'
        )
    print(f'Discovered {len(samples)} noisy/clean pairs.')

    # 2. Shape validation on the first sample.
    noisy_p, albedo_p, normal_p, clean_p = samples[0]
    def load_png(p: Path) -> np.ndarray:
        return np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0  # [H,W,3]
    # Apply the same finite scale/clamp and auxiliary sanitation as runtime.
    noisy  = _load_hdr_np(noisy_p)                             # [H,W,3] linear
    clean  = _load_hdr_np(clean_p)                             # [H,W,3] linear
    albedo = load_png(albedo_p)                                # [H,W,3] [0,1]
    normal_packed = load_png(normal_p)                         # [H,W,3] [0,1]
    noisy, clean = preprocess_radiance_np(noisy), preprocess_radiance_np(clean)
    albedo = np.clip(np.where(np.isfinite(albedo), albedo, 0.0), 0.0, 1.0).astype(np.float32)
    normal_packed = np.clip(
        np.where(np.isfinite(normal_packed), normal_packed, 0.5), 0.0, 1.0,
    )
    remapped = normal_packed * 2.0 - 1.0
    lensq = np.sum(remapped * remapped, axis=-1, keepdims=True)
    up = np.zeros_like(remapped); up[..., 1] = 1.0
    normal = np.where(lensq < 1e-6, up, remapped / np.sqrt(np.clip(lensq, 1e-6, None)))
    x = np.concatenate([noisy, albedo, normal], axis=-1)  # [H,W,9]
    H, W, _ = x.shape
    assert x.shape[2] == 9, f'expected 9 input channels, got {x.shape[2]}'
    assert clean.shape == (H, W, 3), f'clean shape {clean.shape} != ({H},{W},3)'
    assert noisy.shape == albedo.shape == normal.shape == (H, W, 3)
    print(f'Input stack OK: x[{H}x{W}x9] (noisy[HDR]+albedo+normal[-1,1]), '
          f'target clean[{H}x{W}x3][HDR]')

    # 3. Build deterministic random checkpoint + export.
    rng = np.random.default_rng(args.seed)
    records: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, wshape, blen in CANONICAL_LAYERS:
        fan_in = wshape[1] * wshape[2] * wshape[3]  # inC*kH*kW (He init)
        scale = (2.0 / max(1, fan_in)) ** 0.5
        w = rng.uniform(-scale, scale, size=wshape).astype(np.float32).flatten()
        b = rng.uniform(-0.01, 0.01, size=blen).astype(np.float32)
        records.append((name, w, b))

    metadata = checkpoint_metadata(
        checkpoint_id=Path(args.out_bin).stem,
        training_samples=len(samples),
        noisy_spp=args.noisy_spp,
        clean_spp=args.clean_spp,
        capture_source=str(args.data),
        capture_backend=args.capture_backend,
        hardware=args.hardware,
    )
    total = write_vitrum_binary(records, args.out_bin, metadata)
    expect = canonical_param_count()
    assert expect == CANONICAL_PARAM_COUNT, f'shape-table param count {expect} != {CANONICAL_PARAM_COUNT}'
    assert total == CANONICAL_PARAM_COUNT, (
        f'exported param count {total} != canonical {CANONICAL_PARAM_COUNT}'
    )
    size_kb = os.path.getsize(args.out_bin) / 1024
    print(f'Exported {len(records)} layers → {args.out_bin} ({size_kb:.1f} KB)')
    print(f'Param count: {total:,} == canonical {CANONICAL_PARAM_COUNT:,}  ✓')
    print('=== dry-run OK: dataset + shapes + export format all validated ===')


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    if not _HAS_TORCH:
        raise SystemExit(
            'PyTorch is not installed. Run with --dry-run for a numpy-only '
            'dataset+shape+export validation, or `pip install torch torchvision Pillow numpy` '
            'to train. See README.md.'
        )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    # --smoke: tiny end-to-end run (few epochs, small patch, single worker).
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.patch_size = min(args.patch_size, 64)
        print(f'[smoke] epochs={args.epochs} patch={args.patch_size} batch={args.batch}')

    # Dataset + loader.
    dataset = DenoisingDataset(args.data, patch_size=args.patch_size)
    workers = 0 if args.smoke else 2
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                         num_workers=workers, pin_memory=(device.type == 'cuda'))
    print(f'Dataset: {len(dataset)} samples')

    # Model.
    model     = UNetDenoiser().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {param_count:,}')
    assert param_count == CANONICAL_PARAM_COUNT, (
        f'model param count {param_count} != canonical {CANONICAL_PARAM_COUNT} — '
        f'architecture drifted from unetArchitecture.ts'
    )

    # Training loop.
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for x, y in loader:
            x = x.to(device)  # [B, 9, H, W]
            y = y.to(device)  # [B, 3, H, W]

            optimizer.zero_grad()
            pred = model(x)
            loss = combined_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)
        print(f'Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}')

    # Save checkpoint.
    metadata = checkpoint_metadata(
        checkpoint_id=Path(args.out_pth).stem,
        training_samples=len(dataset),
        noisy_spp=args.noisy_spp,
        clean_spp=args.clean_spp,
        capture_source=str(args.data),
        capture_backend=args.capture_backend,
        hardware=args.hardware,
    )
    torch.save({
        'state_dict': model.state_dict(),
        'args':       vars(args),
        'param_count': param_count,
        'metadata': metadata,
    }, args.out_pth)
    print(f'Checkpoint saved → {args.out_pth}')

    # Export vitrum binary.
    export_vitrum_weights(model, args.out_bin, metadata)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train the vitrum U-Net neural denoiser.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--data',       default='data_smoke',   help='Path to noisy/clean pair dataset directory')
    p.add_argument('--epochs',     type=int,   default=50, help='Number of training epochs')
    p.add_argument('--batch',      type=int,   default=4,  help='Batch size')
    p.add_argument('--lr',         type=float, default=1e-4, help='Initial learning rate')
    p.add_argument('--patch-size', type=int,   default=256, help='Training crop size (pixels)')
    p.add_argument('--out-pth',    default='model.pth',    help='Output PyTorch checkpoint path')
    p.add_argument('--out-bin',    default='weights.bin',  help='Output vitrum binary weights path')
    p.add_argument('--seed',       type=int,   default=1984, help='RNG seed (dry-run weight init)')
    p.add_argument('--noisy-spp', type=int, default=1, help='Noisy capture samples per pixel recorded in metadata')
    p.add_argument('--clean-spp', type=int, default=1, help='Clean target samples per pixel recorded in metadata; set this to the capture truth')
    p.add_argument('--capture-backend', default='unspecified', help='Renderer/backend that captured the dataset')
    p.add_argument('--hardware', default='unspecified', help='Capture/training hardware provenance')
    p.add_argument('--smoke',      action='store_true',    help='Tiny end-to-end run (≤2 epochs, ≤64 patch); requires torch')
    p.add_argument('--dry-run',    action='store_true',    help='Numpy-only dataset+shape+export validation (no torch)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.dry_run:
        dry_run(args)
    else:
        train(args)
