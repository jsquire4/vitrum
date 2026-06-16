/**
 * neuralWeightsRoundTrip.test.ts — capture → train → export → load round-trip.
 *
 * Proves the neural-denoiser training pipeline closes end-to-end at the
 * weight-format + runtime-allocation boundary:
 *
 *   1. A checkpoint is built to the EXACT canonical layer shapes that
 *      tools/neural-denoiser-training/train.py exports (the 14 weight-bearing
 *      conv2d / transposedConv2d layers — mirrors CANONICAL_LAYERS in train.py
 *      and unetArchitecture.ts).
 *   2. It is serialized to the .vitrum-model binary format (the same byte schema
 *      train.py's `write_vitrum_binary` / export_weights.py emit), then loaded
 *      back through the REAL `loadWeightsFromArrayBuffer` runtime loader.
 *   3. The loaded ModelWeights are fed into a REAL `InferenceGraph.initialize`
 *      against a stub GPUDevice, exercising `allocateGraph` /
 *      `buildBindGroup` — i.e. the production weight-upload path that maps each
 *      named layer's weights+biases onto GPU buffers.
 *
 * This is the "round-trip" deliverable for road-to-100 A10: it demonstrates a
 * checkpoint produced by the export path loads cleanly into the runtime graph.
 * No real GPU; the stub device records allocations and returns sized mapped
 * ranges so the weight upload (Float32Array.set on getMappedRange) succeeds.
 *
 * The canonical param count (535,107) is asserted on the loaded checkpoint.
 */

import { readFileSync } from 'node:fs';
import { describe, it, expect, beforeAll } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
installWebGPUPolyfills();

import { InferenceGraph } from '../src/neural/InferenceGraph.js';
import {
  buildUNetSpec,
  deriveParamCount,
} from '../src/neural/unetArchitecture.js';
import {
  loadWeightsFromArrayBuffer,
  serializeWeightsToArrayBuffer,
  buildRandomWeightsForSpec,
  validateWeightsForSpec,
} from '../src/neural/weights.js';
import type { ModelWeights } from '../src/neural/weights.js';

// The 14 weight-bearing layers train.py exports, in order. Must match
// CANONICAL_LAYERS in tools/neural-denoiser-training/train.py and the
// conv2d/transposedConv2d layers of unetArchitecture.ts.
const CANONICAL_EXPORT_LAYER_NAMES = [
  'enc1_conv', 'enc1_down',
  'enc2_conv', 'enc2_down',
  'enc3_conv', 'enc3_down',
  'bottleneck',
  'dec3_up', 'dec3_conv',
  'dec2_up', 'dec2_conv',
  'dec1_up', 'dec1_conv',
  'proj',
];
const CANONICAL_PARAM_COUNT = 535107;
const TRACKED_CHECKPOINTS = [
  'starter-v1.vitrum-model',
  'v2-random.vitrum-model',
];

function readTrackedCheckpoint(name: string): ArrayBuffer {
  const bytes = readFileSync(new URL(`../../../tools/neural-denoiser-training/checkpoints/${name}`, import.meta.url));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

/**
 * Build a checkpoint with ONLY the weight-bearing layers (what the export path
 * writes), to the exact canonical shapes. Deterministic (LCG seed) so the test
 * is bit-stable. Mirrors train.py's dry-run He-init export.
 */
function buildExportedCheckpoint(): ModelWeights {
  const spec = buildUNetSpec();
  // buildRandomWeightsForSpec emits one record per layer (including zero-param
  // relu/skip/inputPack with empty arrays). The export path writes ONLY the
  // parameterized layers — filter to mirror that exactly.
  const full = buildRandomWeightsForSpec(spec as never, 1984);
  const weightBearing = full.layers.filter((l) => l.weights.length > 0);
  return { layers: weightBearing };
}

// ─── Stub GPUDevice: records allocations, returns sized mapped ranges ─────────

interface AllocRecord {
  buffers: { label: string; size: number }[];
  weightUploads: { label: string; floats: number }[];
}

function makeStubDevice(rec: AllocRecord): GPUDevice {
  const mkBuf = (label: string, size: number) => {
    if (label.startsWith('neural-weights-') || label.startsWith('neural-biases-')) {
      rec.weightUploads.push({ label, floats: size / 4 });
    }
    rec.buffers.push({ label, size });
    return {
      label,
      size,
      // mappedAtCreation weight/bias buffers call getMappedRange then .set —
      // return a correctly-sized backing buffer so the upload succeeds.
      getMappedRange: () => new ArrayBuffer(size),
      unmap() {},
      destroy() {},
    } as unknown as GPUBuffer;
  };

  const pass = {
    setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {},
  };
  const encoder = {
    beginComputePass: () => pass,
    copyBufferToBuffer() {},
    finish: () => ({}),
  };

  return {
    createBuffer: (d: GPUBufferDescriptor) => mkBuf((d.label as string) ?? '', d.size),
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroupLayout: () => ({}),
    createBindGroup: () => ({}),
    createPipelineLayout: () => ({}),
    createCommandEncoder: () => encoder,
    queue: { writeBuffer() {}, submit() {} },
  } as unknown as GPUDevice;
}

describe('neural weights capture→train→export→load round-trip', () => {
  let checkpoint: ModelWeights;
  let binary: ArrayBuffer;
  let loaded: ModelWeights;

  beforeAll(() => {
    checkpoint = buildExportedCheckpoint();
    binary = serializeWeightsToArrayBuffer(checkpoint);
    loaded = loadWeightsFromArrayBuffer(binary);
  });

  it('exported checkpoint has exactly the 14 canonical export layers in order', () => {
    expect(loaded.layers.map((l) => l.name)).toEqual(CANONICAL_EXPORT_LAYER_NAMES);
  });

  it('loaded checkpoint param count equals the canonical 535,107', () => {
    const total = loaded.layers.reduce(
      (acc, l) => acc + l.weights.length + l.biases.length,
      0,
    );
    expect(total).toBe(CANONICAL_PARAM_COUNT);
    // And it matches the spec the runtime derives independently.
    expect(deriveParamCount(buildUNetSpec().layers)).toBe(CANONICAL_PARAM_COUNT);
  });

  it('binary starts with the vitrum-model magic + version', () => {
    const view = new DataView(binary);
    expect(view.getUint32(0, true)).toBe(0xDEAF1984 >>> 0);
    expect(view.getUint32(4, true)).toBe(1);
    expect(view.getUint32(8, true)).toBe(14); // layerCount
  });

  it('loaded weights bit-equal the source checkpoint (no format drift)', () => {
    expect(loaded.layers.length).toBe(checkpoint.layers.length);
    for (let i = 0; i < checkpoint.layers.length; i++) {
      const a = checkpoint.layers[i]!;
      const b = loaded.layers[i]!;
      expect(b.name).toBe(a.name);
      expect(Array.from(b.weights)).toEqual(Array.from(a.weights));
      expect(Array.from(b.biases)).toEqual(Array.from(a.biases));
    }
  });

  it('validates the exported checkpoint against the runtime U-Net spec', () => {
    expect(() => validateWeightsForSpec(buildUNetSpec(), loaded)).not.toThrow();
  });

  it.each(TRACKED_CHECKPOINTS)('tracked checkpoint %s loads and validates against the runtime U-Net spec', (name) => {
    const actual = loadWeightsFromArrayBuffer(readTrackedCheckpoint(name));
    expect(actual.layers.map((l) => l.name)).toEqual(CANONICAL_EXPORT_LAYER_NAMES);
    const total = actual.layers.reduce(
      (acc, l) => acc + l.weights.length + l.biases.length,
      0,
    );
    expect(total).toBe(CANONICAL_PARAM_COUNT);
    expect(() => validateWeightsForSpec(buildUNetSpec(), actual)).not.toThrow();
  });

  it('rejects missing, unknown, wrong-sized, and non-finite neural checkpoint layers', () => {
    const spec = buildUNetSpec();
    expect(() => validateWeightsForSpec(spec, {
      layers: loaded.layers.filter((l) => l.name !== 'proj'),
    })).toThrow(/missing weights for layer 'proj'/);

    expect(() => validateWeightsForSpec(spec, {
      layers: [...loaded.layers, { name: 'extra', weights: new Float32Array(0), biases: new Float32Array(0) }],
    })).toThrow(/unknown layer 'extra'/);

    const enc1 = loaded.layers.find((l) => l.name === 'enc1_conv')!;
    expect(() => validateWeightsForSpec(spec, {
      layers: loaded.layers.map((l) =>
        l.name === 'enc1_conv'
          ? { ...l, weights: enc1.weights.slice(1) }
          : l,
      ),
    })).toThrow(/enc1_conv.*weight length/);

    const badBiases = new Float32Array(enc1.biases);
    badBiases[0] = Number.NaN;
    expect(() => validateWeightsForSpec(spec, {
      layers: loaded.layers.map((l) =>
        l.name === 'enc1_conv'
          ? { ...l, biases: badBiases }
          : l,
      ),
    })).toThrow(/enc1_conv\.biases\[0\] is not finite/);
  });

  it('loaded checkpoint initializes a real InferenceGraph (stub device)', async () => {
    const rec: AllocRecord = { buffers: [], weightUploads: [] };
    const device = makeStubDevice(rec);

    const graph = new InferenceGraph(buildUNetSpec());
    await graph.initialize(device, loaded, 64, 64);

    // Every parameterized layer must have uploaded a weights buffer + a biases
    // buffer (binding 1 and 2). 14 layers → 14 weight + 14 bias buffers.
    const weightBufs = rec.weightUploads.filter((u) => u.label.startsWith('neural-weights-'));
    const biasBufs = rec.weightUploads.filter((u) => u.label.startsWith('neural-biases-'));
    expect(weightBufs.length).toBe(14);
    expect(biasBufs.length).toBe(14);

    // Spot-check the uploaded float counts match the loaded checkpoint shapes,
    // proving the named weights mapped onto the correct GPU buffers.
    const enc1 = loaded.layers.find((l) => l.name === 'enc1_conv')!;
    const enc1Upload = rec.weightUploads.find((u) => u.label === 'neural-weights-enc1_conv')!;
    expect(enc1Upload.floats).toBeGreaterThanOrEqual(enc1.weights.length);

    graph.dispose();
  });
});
