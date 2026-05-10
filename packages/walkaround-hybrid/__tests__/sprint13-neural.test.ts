/**
 * Sprint 13 (Phase 6) — Neural denoiser scaffold tests.
 *
 * Test strategy: WGSL string-content + TypeScript structural validation only.
 * No real WebGPU device required. Follows the same "defensive structural" pattern
 * as sprint9-welford.test.ts and sprint11-ppg.test.ts.
 *
 * Coverage:
 *   1. CONV2D_WGSL — entry point, bindings, Conv2DParams struct.
 *   2. TRANSPOSED_CONV2D_WGSL — entry point, bindings, TransposedConv2DParams struct.
 *   3. RELU_WGSL — entry point, bindings, ReLUParams struct.
 *   4. SKIP_CONNECTION_WGSL — entry point, bindings, SkipParams struct.
 *   5. BILINEAR_UPSAMPLE_WGSL — entry point, bindings, UpsampleParams struct.
 *   6. Conv2DParams std140 layout — 8 fields × 4 bytes = 32 bytes.
 *   7. InferenceGraph — constructor, type shape, dispose.
 *   8. WALKAROUND_DENOISER_UNET_SPEC — layer count, channel widths, tensor names.
 *   9. ModelWeights — map types.
 *  10. Architecture constants — UNET_TOTAL_PARAMETERS, UNET_WEIGHT_BYTES.
 *  11. Package index — InferenceGraph, UNET_SPEC, and shader strings exported.
 */

import { describe, it, expect, vi } from 'vitest';

// ── WGSL imports ─────────────────────────────────────────────────────────────
import { CONV2D_WGSL } from '../src/neural/wgsl/conv2d.wgsl.js';
import { TRANSPOSED_CONV2D_WGSL } from '../src/neural/wgsl/transposedConv2d.wgsl.js';
import { RELU_WGSL } from '../src/neural/wgsl/relu.wgsl.js';
import { SKIP_CONNECTION_WGSL } from '../src/neural/wgsl/skipConnection.wgsl.js';
import { BILINEAR_UPSAMPLE_WGSL } from '../src/neural/wgsl/bilinearUpsample.wgsl.js';

// ── TypeScript module imports ─────────────────────────────────────────────────
import {
  InferenceGraph,
} from '../src/neural/InferenceGraph.js';
import type {
  InferenceGraphSpec,
  ModelWeights,
  InferenceLayer,
  InferenceLayerKind,
} from '../src/neural/InferenceGraph.js';
import {
  WALKAROUND_DENOISER_UNET_SPEC,
  UNET_INPUT_CHANNELS,
  UNET_OUTPUT_CHANNELS,
  UNET_ENCODER_CHANNELS,
  UNET_DECODER_CHANNELS,
  UNET_TOTAL_PARAMETERS,
  UNET_WEIGHT_BYTES,
  UNET_INPUT_TENSOR_NAMES,
  UNET_OUTPUT_TENSOR_NAMES,
} from '../src/neural/unetArchitecture.js';

// ── WebGPU global polyfills (Node test environment) ───────────────────────────
if (typeof (globalThis as Record<string, unknown>)['GPUBufferUsage'] === 'undefined') {
  (globalThis as Record<string, unknown>)['GPUBufferUsage'] = {
    MAP_READ:      0x0001,
    MAP_WRITE:     0x0002,
    COPY_SRC:      0x0004,
    COPY_DST:      0x0008,
    INDEX:         0x0010,
    VERTEX:        0x0020,
    UNIFORM:       0x0040,
    STORAGE:       0x0080,
    INDIRECT:      0x0100,
    QUERY_RESOLVE: 0x0200,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. CONV2D_WGSL — entry point, bindings, Conv2DParams struct
// ─────────────────────────────────────────────────────────────────────────────

describe('CONV2D_WGSL — 2D convolution shader fragment', () => {
  it('is a non-empty string', () => {
    expect(typeof CONV2D_WGSL).toBe('string');
    expect(CONV2D_WGSL.length).toBeGreaterThan(0);
  });

  it('declares @compute @workgroup_size(8, 8, 1) entry point', () => {
    expect(CONV2D_WGSL).toContain('@compute @workgroup_size(8, 8, 1)');
  });

  it('contains conv2dKernel entry point', () => {
    expect(CONV2D_WGSL).toContain('fn conv2dKernel');
  });

  it('uses @builtin(global_invocation_id)', () => {
    expect(CONV2D_WGSL).toContain('@builtin(global_invocation_id)');
  });

  it('declares Conv2DParams uniform struct', () => {
    expect(CONV2D_WGSL).toContain('struct Conv2DParams');
  });

  it('Conv2DParams has inputH, inputW, inputC fields', () => {
    expect(CONV2D_WGSL).toContain('inputH:');
    expect(CONV2D_WGSL).toContain('inputW:');
    expect(CONV2D_WGSL).toContain('inputC:');
  });

  it('Conv2DParams has kernelH, kernelW, outputC, stride, dilation fields', () => {
    expect(CONV2D_WGSL).toContain('kernelH:');
    expect(CONV2D_WGSL).toContain('kernelW:');
    expect(CONV2D_WGSL).toContain('outputC:');
    expect(CONV2D_WGSL).toContain('stride:');
    expect(CONV2D_WGSL).toContain('dilation:');
  });

  it('binds conv_input at @group(0) @binding(0)', () => {
    expect(CONV2D_WGSL).toContain('@group(0) @binding(0)');
    expect(CONV2D_WGSL).toContain('conv_input');
  });

  it('binds conv_weights at @group(0) @binding(1)', () => {
    expect(CONV2D_WGSL).toContain('@group(0) @binding(1)');
    expect(CONV2D_WGSL).toContain('conv_weights');
  });

  it('binds conv_bias at @group(0) @binding(2)', () => {
    expect(CONV2D_WGSL).toContain('@group(0) @binding(2)');
    expect(CONV2D_WGSL).toContain('conv_bias');
  });

  it('binds conv_output (read_write) at @group(0) @binding(3)', () => {
    expect(CONV2D_WGSL).toContain('@group(0) @binding(3)');
    expect(CONV2D_WGSL).toContain('read_write');
    expect(CONV2D_WGSL).toContain('conv_output');
  });

  it('binds conv_params uniform at @group(0) @binding(4)', () => {
    expect(CONV2D_WGSL).toContain('@group(0) @binding(4)');
    expect(CONV2D_WGSL).toContain('conv_params');
  });

  it('uses SAME padding with zero-fill for OOB pixels', () => {
    expect(CONV2D_WGSL).toContain('return 0.0');
  });

  it('accumulates over ky, kx, ic loops', () => {
    expect(CONV2D_WGSL).toContain('for (var ky');
    expect(CONV2D_WGSL).toContain('for (var kx');
    expect(CONV2D_WGSL).toContain('for (var ic');
  });

  it('adds bias to accumulator', () => {
    expect(CONV2D_WGSL).toContain('conv_bias[oc]');
  });

  it('supports dilation via conv_params.dilation', () => {
    expect(CONV2D_WGSL).toContain('conv_params.dilation');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. TRANSPOSED_CONV2D_WGSL — entry point, bindings, struct
// ─────────────────────────────────────────────────────────────────────────────

describe('TRANSPOSED_CONV2D_WGSL — transposed (deconvolutional) 2D convolution', () => {
  it('is a non-empty string', () => {
    expect(typeof TRANSPOSED_CONV2D_WGSL).toBe('string');
    expect(TRANSPOSED_CONV2D_WGSL.length).toBeGreaterThan(0);
  });

  it('declares @compute @workgroup_size(8, 8, 1)', () => {
    expect(TRANSPOSED_CONV2D_WGSL).toContain('@compute @workgroup_size(8, 8, 1)');
  });

  it('contains transposedConv2dKernel entry point', () => {
    expect(TRANSPOSED_CONV2D_WGSL).toContain('fn transposedConv2dKernel');
  });

  it('declares TransposedConv2DParams struct', () => {
    expect(TRANSPOSED_CONV2D_WGSL).toContain('struct TransposedConv2DParams');
  });

  it('TransposedConv2DParams has stride field', () => {
    expect(TRANSPOSED_CONV2D_WGSL).toContain('stride:');
  });

  it('binds tconv_input at @group(0) @binding(0)', () => {
    expect(TRANSPOSED_CONV2D_WGSL).toContain('@group(0) @binding(0)');
    expect(TRANSPOSED_CONV2D_WGSL).toContain('tconv_input');
  });

  it('binds tconv_weights at @group(0) @binding(1)', () => {
    expect(TRANSPOSED_CONV2D_WGSL).toContain('@group(0) @binding(1)');
    expect(TRANSPOSED_CONV2D_WGSL).toContain('tconv_weights');
  });

  it('binds tconv_output (read_write) at @group(0) @binding(3)', () => {
    expect(TRANSPOSED_CONV2D_WGSL).toContain('@group(0) @binding(3)');
    expect(TRANSPOSED_CONV2D_WGSL).toContain('tconv_output');
  });

  it('uses gather formulation: output iterates over (oy, ox, oc)', () => {
    expect(TRANSPOSED_CONV2D_WGSL).toContain('let oy = gid.y');
    expect(TRANSPOSED_CONV2D_WGSL).toContain('let ox = gid.x');
  });

  it('output dimensions are inputH × stride (2× upsample for stride=2)', () => {
    expect(TRANSPOSED_CONV2D_WGSL).toContain('tconv_params.inputH * s');
    expect(TRANSPOSED_CONV2D_WGSL).toContain('tconv_params.inputW * s');
  });

  it('adds tconv_bias to output', () => {
    expect(TRANSPOSED_CONV2D_WGSL).toContain('tconv_bias[oc]');
  });

  it('uses modulo check for stride-compatible kernel positions', () => {
    expect(TRANSPOSED_CONV2D_WGSL).toContain('% s');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. RELU_WGSL — entry point, bindings
// ─────────────────────────────────────────────────────────────────────────────

describe('RELU_WGSL — ReLU activation shader fragment', () => {
  it('is a non-empty string', () => {
    expect(typeof RELU_WGSL).toBe('string');
    expect(RELU_WGSL.length).toBeGreaterThan(0);
  });

  it('declares @compute @workgroup_size(256, 1, 1)', () => {
    expect(RELU_WGSL).toContain('@compute @workgroup_size(256, 1, 1)');
  });

  it('contains reluKernel entry point', () => {
    expect(RELU_WGSL).toContain('fn reluKernel');
  });

  it('applies max(input, 0.0) activation', () => {
    expect(RELU_WGSL).toContain('max(relu_input[i], 0.0)');
  });

  it('binds relu_input at @group(0) @binding(0)', () => {
    expect(RELU_WGSL).toContain('@group(0) @binding(0)');
    expect(RELU_WGSL).toContain('relu_input');
  });

  it('binds relu_output (read_write) at @group(0) @binding(1)', () => {
    expect(RELU_WGSL).toContain('@group(0) @binding(1)');
    expect(RELU_WGSL).toContain('relu_output');
  });

  it('binds relu_params uniform at @group(0) @binding(2)', () => {
    expect(RELU_WGSL).toContain('@group(0) @binding(2)');
    expect(RELU_WGSL).toContain('relu_params');
  });

  it('declares ReLUParams struct with totalElements field', () => {
    expect(RELU_WGSL).toContain('struct ReLUParams');
    expect(RELU_WGSL).toContain('totalElements');
  });

  it('bounds-checks i against relu_params.totalElements', () => {
    expect(RELU_WGSL).toContain('relu_params.totalElements');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. SKIP_CONNECTION_WGSL — entry point, bindings
// ─────────────────────────────────────────────────────────────────────────────

describe('SKIP_CONNECTION_WGSL — UNet skip-connection (elementwise add)', () => {
  it('is a non-empty string', () => {
    expect(typeof SKIP_CONNECTION_WGSL).toBe('string');
    expect(SKIP_CONNECTION_WGSL.length).toBeGreaterThan(0);
  });

  it('declares @compute @workgroup_size(256, 1, 1)', () => {
    expect(SKIP_CONNECTION_WGSL).toContain('@compute @workgroup_size(256, 1, 1)');
  });

  it('contains skipConnectionKernel entry point', () => {
    expect(SKIP_CONNECTION_WGSL).toContain('fn skipConnectionKernel');
  });

  it('performs elementwise addition: output = inputA + inputB', () => {
    expect(SKIP_CONNECTION_WGSL).toContain('skip_inputA[i] + skip_inputB[i]');
  });

  it('binds skip_inputA at @group(0) @binding(0)', () => {
    expect(SKIP_CONNECTION_WGSL).toContain('@group(0) @binding(0)');
    expect(SKIP_CONNECTION_WGSL).toContain('skip_inputA');
  });

  it('binds skip_inputB at @group(0) @binding(1)', () => {
    expect(SKIP_CONNECTION_WGSL).toContain('@group(0) @binding(1)');
    expect(SKIP_CONNECTION_WGSL).toContain('skip_inputB');
  });

  it('binds skip_output (read_write) at @group(0) @binding(2)', () => {
    expect(SKIP_CONNECTION_WGSL).toContain('@group(0) @binding(2)');
    expect(SKIP_CONNECTION_WGSL).toContain('skip_output');
  });

  it('declares SkipParams struct with totalElements', () => {
    expect(SKIP_CONNECTION_WGSL).toContain('struct SkipParams');
    expect(SKIP_CONNECTION_WGSL).toContain('totalElements');
  });

  it('bounds-checks i against skip_params.totalElements', () => {
    expect(SKIP_CONNECTION_WGSL).toContain('skip_params.totalElements');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 5. BILINEAR_UPSAMPLE_WGSL — entry point, bindings
// ─────────────────────────────────────────────────────────────────────────────

describe('BILINEAR_UPSAMPLE_WGSL — bilinear 2× upsampling shader', () => {
  it('is a non-empty string', () => {
    expect(typeof BILINEAR_UPSAMPLE_WGSL).toBe('string');
    expect(BILINEAR_UPSAMPLE_WGSL.length).toBeGreaterThan(0);
  });

  it('declares @compute @workgroup_size(8, 8, 1)', () => {
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('@compute @workgroup_size(8, 8, 1)');
  });

  it('contains bilinearUpsampleKernel entry point', () => {
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('fn bilinearUpsampleKernel');
  });

  it('declares UpsampleParams struct', () => {
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('struct UpsampleParams');
  });

  it('UpsampleParams has inputH, inputW, channels, scale fields', () => {
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('inputH:');
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('inputW:');
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('channels:');
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('scale:');
  });

  it('binds ups_input at @group(0) @binding(0)', () => {
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('@group(0) @binding(0)');
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('ups_input');
  });

  it('binds ups_output (read_write) at @group(0) @binding(1)', () => {
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('@group(0) @binding(1)');
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('ups_output');
  });

  it('uses center-aligned coordinate mapping: (out + 0.5) / scale - 0.5', () => {
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('+ 0.5) / sf - 0.5');
  });

  it('computes bilinear weights w00, w10, w01, w11', () => {
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('w00');
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('w01');
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('w10');
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('w11');
  });

  it('clamps OOB reads to border (clamp strategy)', () => {
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('clamp(iy, 0, H - 1)');
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('clamp(ix, 0, W - 1)');
  });

  it('iterates over all channels per output pixel', () => {
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('for (var ch = 0u; ch < C; ch++)');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 6. Conv2DParams std140 layout — 8 fields × 4 bytes = 32 bytes
// ─────────────────────────────────────────────────────────────────────────────

describe('Conv2DParams std140 layout', () => {
  it('Conv2DParams struct has exactly 8 u32 fields (32 bytes total)', () => {
    // 8 fields: inputH, inputW, inputC, kernelH, kernelW, outputC, stride, dilation
    // Each u32 = 4 bytes. Total = 32 bytes — one std140 block.
    const fieldNames = ['inputH', 'inputW', 'inputC', 'kernelH', 'kernelW', 'outputC', 'stride', 'dilation'];
    for (const field of fieldNames) {
      expect(CONV2D_WGSL).toContain(field);
    }
    // Verify 8 distinct field declarations by counting "u32," occurrences after "struct Conv2DParams"
    const structIdx = CONV2D_WGSL.indexOf('struct Conv2DParams');
    expect(structIdx).toBeGreaterThan(-1);
    const structBody = CONV2D_WGSL.slice(structIdx, structIdx + 400);
    const u32Count = (structBody.match(/u32/g) ?? []).length;
    expect(u32Count).toBe(8);  // exactly 8 u32 fields
  });

  it('TransposedConv2DParams has 8 fields including 1 padding field (_pad)', () => {
    expect(TRANSPOSED_CONV2D_WGSL).toContain('_pad:');
    const structIdx = TRANSPOSED_CONV2D_WGSL.indexOf('struct TransposedConv2DParams');
    const structBody = TRANSPOSED_CONV2D_WGSL.slice(structIdx, structIdx + 400);
    const u32Count = (structBody.match(/u32/g) ?? []).length;
    expect(u32Count).toBe(8);  // 7 real fields + 1 pad = 32 bytes
  });

  it('ReLUParams has 4 u32 fields (totalElements + 3 padding = 16 bytes)', () => {
    const structIdx = RELU_WGSL.indexOf('struct ReLUParams');
    const structBody = RELU_WGSL.slice(structIdx, structIdx + 200);
    const u32Count = (structBody.match(/u32/g) ?? []).length;
    expect(u32Count).toBe(4);  // totalElements + _pad0 + _pad1 + _pad2
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 7. InferenceGraph — constructor, dispose, run guard
// ─────────────────────────────────────────────────────────────────────────────

describe('InferenceGraph — structural tests (no GPU required)', () => {
  const emptySpec: InferenceGraphSpec = {
    layers:        [],
    inputTensors:  ['noisy'],
    outputTensors: ['denoised'],
  };

  const emptyWeights: ModelWeights = {
    weights: new Map(),
    biases:  new Map(),
  };

  it('InferenceGraph constructor accepts spec + weights without throwing', () => {
    expect(() => new InferenceGraph(emptySpec, emptyWeights)).not.toThrow();
  });

  it('InferenceGraph instance is truthy after construction', () => {
    const g = new InferenceGraph(emptySpec, emptyWeights);
    expect(g).toBeTruthy();
  });

  it('dispose() does not throw on a fresh (not-initialized) instance', () => {
    const g = new InferenceGraph(emptySpec, emptyWeights);
    expect(() => g.dispose()).not.toThrow();
  });

  it('run() throws before initialize() is called', () => {
    const g = new InferenceGraph(emptySpec, emptyWeights);
    expect(() =>
      g.run(
        {} as GPUDevice,
        {} as GPUCommandEncoder,
        new Map(),
        new Map(),
      )
    ).toThrow('[InferenceGraph] run() called before initialize().');
  });

  it('dispose() twice does not throw (idempotent)', () => {
    const g = new InferenceGraph(emptySpec, emptyWeights);
    g.dispose();
    expect(() => g.dispose()).not.toThrow();
  });

  it('ModelWeights weights field is a Map', () => {
    expect(emptyWeights.weights instanceof Map).toBe(true);
  });

  it('ModelWeights biases field is a Map', () => {
    expect(emptyWeights.biases instanceof Map).toBe(true);
  });

  it('ModelWeights can hold Float32Array values', () => {
    const weights: ModelWeights = {
      weights: new Map([['enc1_conv', new Float32Array([1, 2, 3])]]),
      biases:  new Map([['enc1_conv', new Float32Array([0.1])]]),
    };
    expect(weights.weights.get('enc1_conv')!.length).toBe(3);
    expect(weights.biases.get('enc1_conv')!.length).toBe(1);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 8. WALKAROUND_DENOISER_UNET_SPEC — layer count, channel widths, tensor names
// ─────────────────────────────────────────────────────────────────────────────

describe('WALKAROUND_DENOISER_UNET_SPEC — UNet architecture spec', () => {
  it('is defined and has a layers array', () => {
    expect(WALKAROUND_DENOISER_UNET_SPEC).toBeDefined();
    expect(Array.isArray(WALKAROUND_DENOISER_UNET_SPEC.layers)).toBe(true);
  });

  it('has at least 20 layers (encoder + bottleneck + decoder + projection)', () => {
    // 3 enc × 2 (conv+relu) + 1 btn × 2 + 3 dec × 4 (tconv+skip+relu+conv+relu) + 1 proj = 22+
    expect(WALKAROUND_DENOISER_UNET_SPEC.layers.length).toBeGreaterThanOrEqual(20);
  });

  it('specifies noisy, albedo, normals as input tensor names', () => {
    const inputs = WALKAROUND_DENOISER_UNET_SPEC.inputTensors;
    expect(inputs).toContain('noisyColor');
    expect(inputs).toContain('albedo');
    expect(inputs).toContain('normals');
  });

  it('specifies denoisedColor as output tensor name', () => {
    const outputs = WALKAROUND_DENOISER_UNET_SPEC.outputTensors;
    expect(outputs).toContain('denoisedColor');
  });

  it('contains at least one conv2d layer', () => {
    const conv2dLayers = WALKAROUND_DENOISER_UNET_SPEC.layers.filter(
      (l) => l.kind === 'conv2d'
    );
    expect(conv2dLayers.length).toBeGreaterThan(0);
  });

  it('contains at least one transposed_conv2d layer (decoder upsample)', () => {
    const tconvLayers = WALKAROUND_DENOISER_UNET_SPEC.layers.filter(
      (l) => l.kind === 'transposed_conv2d'
    );
    expect(tconvLayers.length).toBeGreaterThan(0);
  });

  it('contains at least one relu layer per encoder level', () => {
    const reluLayers = WALKAROUND_DENOISER_UNET_SPEC.layers.filter(
      (l) => l.kind === 'relu'
    );
    // At least 3 (enc) + 1 (btn) + 3 (dec) = 7 relu layers minimum
    expect(reluLayers.length).toBeGreaterThanOrEqual(7);
  });

  it('contains skip layers for skip connections', () => {
    const skipLayers = WALKAROUND_DENOISER_UNET_SPEC.layers.filter(
      (l) => l.kind === 'skip'
    );
    expect(skipLayers.length).toBeGreaterThan(0);
  });

  it('encoder conv2d layers have UNET_ENCODER_CHANNELS as outputC', () => {
    const convLayers = WALKAROUND_DENOISER_UNET_SPEC.layers.filter(
      (l) => l.kind === 'conv2d' && l.output.startsWith('enc')
    );
    const outputChannels = convLayers.map((l) => (l.params as Record<string, number>)['outputC']);
    // Each enc layer should match one of the UNET_ENCODER_CHANNELS
    for (const c of outputChannels) {
      expect(UNET_ENCODER_CHANNELS).toContain(c);
    }
  });

  it('output projection layer produces denoisedColor', () => {
    const lastLayer = WALKAROUND_DENOISER_UNET_SPEC.layers[WALKAROUND_DENOISER_UNET_SPEC.layers.length - 1];
    expect(lastLayer.output).toBe('denoisedColor');
    expect(lastLayer.kind).toBe('conv2d');
  });

  it('output projection has outputC = 3 (RGB)', () => {
    const lastLayer = WALKAROUND_DENOISER_UNET_SPEC.layers[WALKAROUND_DENOISER_UNET_SPEC.layers.length - 1];
    expect((lastLayer.params as Record<string, number>)['outputC']).toBe(3);
  });

  it('all layer kinds are valid InferenceLayerKind values', () => {
    const validKinds: InferenceLayerKind[] = [
      'conv2d', 'transposed_conv2d', 'relu', 'skip', 'upsample',
    ];
    for (const layer of WALKAROUND_DENOISER_UNET_SPEC.layers) {
      expect(validKinds).toContain(layer.kind);
    }
  });

  it('every layer has a non-empty output tensor name', () => {
    for (const layer of WALKAROUND_DENOISER_UNET_SPEC.layers) {
      expect(typeof layer.output).toBe('string');
      expect(layer.output.length).toBeGreaterThan(0);
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 9. Architecture constants
// ─────────────────────────────────────────────────────────────────────────────

describe('UNet architecture constants', () => {
  it('UNET_INPUT_CHANNELS is 9 (noisy RGB + albedo RGB + normals RGB)', () => {
    expect(UNET_INPUT_CHANNELS).toBe(9);
  });

  it('UNET_OUTPUT_CHANNELS is 3 (denoised RGB)', () => {
    expect(UNET_OUTPUT_CHANNELS).toBe(3);
  });

  it('UNET_ENCODER_CHANNELS has 4 levels (enc1, enc2, enc3, bottleneck)', () => {
    expect(UNET_ENCODER_CHANNELS.length).toBe(4);
  });

  it('UNET_DECODER_CHANNELS has 3 levels (mirror encoder)', () => {
    expect(UNET_DECODER_CHANNELS.length).toBe(3);
  });

  it('UNET_ENCODER_CHANNELS values are in ascending order', () => {
    for (let i = 1; i < UNET_ENCODER_CHANNELS.length; i++) {
      expect(UNET_ENCODER_CHANNELS[i]).toBeGreaterThan(UNET_ENCODER_CHANNELS[i - 1]);
    }
  });

  it('UNET_TOTAL_PARAMETERS is 426,075', () => {
    expect(UNET_TOTAL_PARAMETERS).toBe(426_075);
  });

  it('UNET_WEIGHT_BYTES is UNET_TOTAL_PARAMETERS × 4', () => {
    expect(UNET_WEIGHT_BYTES).toBe(UNET_TOTAL_PARAMETERS * 4);
  });

  it('UNET_WEIGHT_BYTES is within the 1–3 MB DoD target', () => {
    const ONE_MB = 1024 * 1024;
    const THREE_MB = 3 * ONE_MB;
    expect(UNET_WEIGHT_BYTES).toBeGreaterThanOrEqual(ONE_MB);
    expect(UNET_WEIGHT_BYTES).toBeLessThanOrEqual(THREE_MB);
  });

  it('UNET_INPUT_TENSOR_NAMES contains noisyColor, albedo, normals', () => {
    expect(UNET_INPUT_TENSOR_NAMES).toContain('noisyColor');
    expect(UNET_INPUT_TENSOR_NAMES).toContain('albedo');
    expect(UNET_INPUT_TENSOR_NAMES).toContain('normals');
  });

  it('UNET_OUTPUT_TENSOR_NAMES contains denoisedColor', () => {
    expect(UNET_OUTPUT_TENSOR_NAMES).toContain('denoisedColor');
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 10. Package index exports
// ─────────────────────────────────────────────────────────────────────────────

describe('Sprint 13 exports from package index', () => {
  it('InferenceGraph is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['InferenceGraph']).toBe('function');
  });

  it('WALKAROUND_DENOISER_UNET_SPEC is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect((mod as Record<string, unknown>)['WALKAROUND_DENOISER_UNET_SPEC']).toBeDefined();
  });

  it('CONV2D_WGSL is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['CONV2D_WGSL']).toBe('string');
  });

  it('TRANSPOSED_CONV2D_WGSL is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['TRANSPOSED_CONV2D_WGSL']).toBe('string');
  });

  it('RELU_WGSL is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['RELU_WGSL']).toBe('string');
  });

  it('SKIP_CONNECTION_WGSL is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['SKIP_CONNECTION_WGSL']).toBe('string');
  });

  it('BILINEAR_UPSAMPLE_WGSL is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['BILINEAR_UPSAMPLE_WGSL']).toBe('string');
  });

  it('UNET_TOTAL_PARAMETERS is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['UNET_TOTAL_PARAMETERS']).toBe('number');
  });

  it('UNET_WEIGHT_BYTES is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['UNET_WEIGHT_BYTES']).toBe('number');
  });

  it('UNET_INPUT_CHANNELS is exported from package index', async () => {
    const mod = await import('../src/index.js');
    expect(typeof (mod as Record<string, unknown>)['UNET_INPUT_CHANNELS']).toBe('number');
  });
});
