#!/usr/bin/env python3
"""
test_preprocessing.py — D5/V3-7 regression: training preprocessing must match
the runtime pack shader (packages/walkaround-hybrid/src/shaders/neuralPack.wgsl).

Asserts:
  1. _decode_normal reproduces neuralPack.wgsl:57-58 exactly:
       nd_remapped = nd*2 - 1
       nrm = select(normalize(nd_remapped), (0,1,0), len(nd_remapped) < eps)
  2. The color target container (_load_hdr_np) round-trips LINEAR HDR radiance.
  3. Training/runtime scale, clamp, non-finite handling, and v2 metadata agree.

Run directly (no pytest required): python3 test_preprocessing.py
"""
import json
import struct
import tempfile
from pathlib import Path

import numpy as np
try:
    import torch
except ImportError:
    torch = None

import train  # train.py in the same directory


def _wgsl_reference_decode(packed: np.ndarray) -> np.ndarray:
    """Numpy re-implementation of neuralPack.wgsl:57-58 for a [3] packed normal."""
    remapped = packed * 2.0 - 1.0
    len_sq = float(np.dot(remapped, remapped))
    if len_sq < 1e-6:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return (remapped / np.sqrt(len_sq)).astype(np.float32)


def test_decode_normal_matches_wgsl():
    if torch is None:
        print('SKIP _decode_normal Torch parity (PyTorch is not installed)')
        return
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


def test_scale_clamp_and_nonfinite_contract():
    values = np.array(
        [-1.0, 0.0, 16.0, 64.0, 100.0, np.nan, np.inf, -np.inf],
        dtype=np.float32,
    )
    expected = np.array([0.0, 0.0, 1.0, 4.0, 4.0, 0.0, 0.0, 0.0], dtype=np.float32)
    actual_np = train.preprocess_radiance_np(values)
    assert np.array_equal(actual_np, expected), (actual_np, expected)
    if torch is not None:
        actual_torch = train.preprocess_radiance(torch.from_numpy(values.copy())).numpy()
        assert np.array_equal(actual_torch, expected), (actual_torch, expected)
    else:
        print('SKIP Torch radiance parity (PyTorch is not installed)')
    assert np.array_equal(
        train.postprocess_radiance_np(actual_np),
        np.array([0.0, 0.0, 16.0, 64.0, 64.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    print('OK  HDR scale/clamp/non-finite behavior matches the runtime contract')


def test_nonfinite_packed_normal_falls_back_to_up():
    if torch is None:
        print('SKIP non-finite Torch normal parity (PyTorch is not installed)')
        return
    packed = torch.full((3, 1, 1), float('nan'), dtype=torch.float32)
    decoded = train._decode_normal(packed).numpy()[:, 0, 0]
    assert np.array_equal(decoded, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    print('OK  non-finite normal input is finite and falls back to geometric-up')


def test_v2_binary_embeds_deterministic_preprocessing_metadata():
    metadata = train.checkpoint_metadata(
        checkpoint_id='preprocess-test',
        training_samples=1,
        noisy_spp=1,
        clean_spp=1,
        capture_source='unit',
        capture_backend='unit',
        hardware='unit',
    )
    records = [
        (
            'proj',
            np.array([0.25, -0.5], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
        ),
    ]
    with tempfile.TemporaryDirectory() as td:
        first = Path(td) / 'first.vitrum-model'
        second = Path(td) / 'second.vitrum-model'
        train.write_vitrum_binary(records, str(first), metadata)
        train.write_vitrum_binary(records, str(second), metadata)
        first_bytes = first.read_bytes()
        second_bytes = second.read_bytes()
    assert first_bytes == second_bytes
    magic, version, layer_count, metadata_length = struct.unpack('<IIII', first_bytes[:16])
    assert magic == train.VITRUM_MODEL_MAGIC
    assert version == 2
    assert layer_count == 1
    decoded = json.loads(first_bytes[16:16 + metadata_length].decode('utf-8'))
    assert decoded['preprocessing'] == train.NEURAL_PREPROCESSING
    print('OK  v2 binary embeds deterministic preprocessing metadata')


def test_v2_writer_rejects_unsafe_payloads():
    metadata = train.checkpoint_metadata(
        checkpoint_id='unsafe-payload-test',
        training_samples=1,
        noisy_spp=1,
        clean_spp=1,
        capture_source='unit',
        capture_backend='unit',
        hardware='unit',
    )
    unsafe = [
        ('non-finite', np.array([np.nan], dtype=np.float32), 'non-finite'),
        ('magnitude', np.array([train.MAX_PARAMETER_MAGNITUDE + 1], dtype=np.float32), 'magnitude'),
    ]
    with tempfile.TemporaryDirectory() as td:
        for label, values, expected in unsafe:
            try:
                train.write_vitrum_binary(
                    [('proj', values, np.zeros(1, dtype=np.float32))],
                    str(Path(td) / f'{label}.vitrum-model'),
                    metadata,
                )
            except ValueError as error:
                assert expected in str(error), error
            else:
                raise AssertionError(f'{label} payload must be rejected')
    print('OK  v2 writer rejects non-finite and out-of-bound parameters')


def test_f16_certification_export_roundtrip():
    if torch is None:
        print('SKIP f16 certification export roundtrip (PyTorch is not installed)')
        return
    import export_weights

    metadata = train.checkpoint_metadata(
        checkpoint_id='mixed-precision-roundtrip',
        training_samples=500,
        noisy_spp=1,
        clean_spp=4096,
        capture_source='unit',
        capture_backend='walkaround-hybrid',
        hardware='unit',
    )
    metadata['qualityReport'] = {'status': 'pass', 'reportPath': 'quality.json'}
    report = {
        'status': 'pass',
        'checkpointSha256': '',
        'architecture': train.NEURAL_ARCHITECTURE_ID,
        'preprocessing': dict(train.NEURAL_PREPROCESSING),
        'quantization': train.NEURAL_F16_QUANTIZATION,
        'metricDomain': train.NEURAL_F16_METRIC_DOMAIN,
        'validationCorpusSha256': 'a' * 64,
        'validationScenes': 8,
        'maxAbsError': 0.01,
        'meanAbsError': 0.001,
        'psnrDb': 48.0,
        'finiteOutputs': True,
        'outputMin': 0.0,
        'outputMax': 64.0,
        'accumulation': 'f32',
        'weights': 'f32',
    }
    state_dict = {}
    records = []
    for name, weight_shape, bias_len in train.CANONICAL_LAYERS:
        weights = np.zeros(weight_shape, dtype=np.float32)
        biases = np.zeros(bias_len, dtype=np.float32)
        state_dict[f'{name}.weight'] = torch.from_numpy(weights.copy())
        state_dict[f'{name}.bias'] = torch.from_numpy(biases.copy())
        records.append((name, weights.flatten(), biases))
    report['checkpointSha256'] = export_weights.ordered_tensor_digest(records)

    with tempfile.TemporaryDirectory() as td:
        directory = Path(td)
        checkpoint = directory / 'model.pth'
        report_path = directory / 'mixed-precision.json'
        output = directory / 'model.vitrum-model'
        torch.save({'state_dict': state_dict, 'metadata': metadata}, checkpoint)
        report_path.write_text(json.dumps(report), encoding='utf-8')
        export_weights.export(
            str(checkpoint),
            str(output),
            verbose=False,
            mixed_precision_report_path=str(report_path),
        )
        binary = output.read_bytes()

    _magic, version, _layer_count, metadata_length = struct.unpack('<IIII', binary[:16])
    decoded = json.loads(binary[16:16 + metadata_length].decode('utf-8'))
    assert version == 2
    assert decoded['tensorStorage'] == 'f16-compatible'
    assert decoded['mixedPrecision'] == report

    certified_metadata = dict(metadata)
    certified_metadata['tensorStorage'] = 'f16-compatible'
    certified_metadata['mixedPrecision'] = report
    with tempfile.TemporaryDirectory() as td:
        try:
            train.write_vitrum_binary(
                list(reversed(records)),
                str(Path(td) / 'reordered.vitrum-model'),
                certified_metadata,
            )
        except ValueError as error:
            assert 'ordered checkpoint tensors' in str(error), error
        else:
            raise AssertionError('reordered tensors must invalidate the certificate')

        tampered_state = dict(state_dict)
        tampered_weight = tampered_state['enc1_conv.weight'].clone()
        tampered_weight.reshape(-1)[0] = 1.0
        tampered_state['enc1_conv.weight'] = tampered_weight
        stale_checkpoint = Path(td) / 'stale.pth'
        stale_output = Path(td) / 'stale.vitrum-model'
        stale_report = Path(td) / 'stale-report.json'
        torch.save({'state_dict': tampered_state, 'metadata': metadata}, stale_checkpoint)
        stale_report.write_text(json.dumps(report), encoding='utf-8')
        try:
            export_weights.export(
                str(stale_checkpoint),
                str(stale_output),
                verbose=False,
                mixed_precision_report_path=str(stale_report),
            )
        except ValueError as error:
            assert 'ordered checkpoint tensors' in str(error), error
        else:
            raise AssertionError('tampered weights with a stale report must be rejected')

    invalid = dict(metadata)
    invalid['tensorStorage'] = 'f16-compatible'
    invalid['mixedPrecision'] = {**report, 'maxAbsError': train.F16_MAX_ABS_ERROR + 0.001}
    try:
        train.validate_checkpoint_metadata(invalid)
    except ValueError as error:
        assert 'precision/output bounds' in str(error), error
    else:
        raise AssertionError('out-of-bound f16 certification must be rejected')
    print('OK  f16 certification metrics round-trip through v2 export and enforce bounds')


def test_optional_torch_dependency_contract():
    assert train._HAS_TORCH is (torch is not None)
    if torch is None:
        try:
            train.train(object())
        except SystemExit as error:
            message = str(error)
            assert 'PyTorch is not installed' in message
            assert '--dry-run' in message
            assert 'pip install torch' in message
        else:
            raise AssertionError('training without PyTorch must fail with setup guidance')
        print('OK  no-Torch import and training dependency contract is explicit; --dry-run remains available')
    else:
        assert train._HAS_TORCH is True
        print('OK  optional PyTorch training runtime is available')



if __name__ == '__main__':
    test_decode_normal_matches_wgsl()
    test_color_target_is_linear_hdr()
    test_scale_clamp_and_nonfinite_contract()
    test_nonfinite_packed_normal_falls_back_to_up()
    test_optional_torch_dependency_contract()
    test_v2_binary_embeds_deterministic_preprocessing_metadata()
    test_v2_writer_rejects_unsafe_payloads()
    test_f16_certification_export_roundtrip()
    print('ALL AVAILABLE PREPROCESSING TESTS PASSED')
