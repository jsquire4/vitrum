/**
 * inferenceGraphDimsOnce.test.ts — Task 4.5 Theme-I characterization gate for the
 * InferenceGraph decomposition (TensorDimSolver + LayerResourceAllocator extract)
 * AND the dims-once perf change.
 *
 * BEHAVIOR-IDENTITY PIN. The decomposition must leave the executed compute graph
 * byte-for-byte identical:
 *   - the per-layer dispatchWorkgroups(gx,gy,gz) sequence,
 *   - the per-layer uniform-buffer write bytes,
 *   - the input-pack dispatch,
 *   - the final denoised→output copyBufferToBuffer extent,
 * are all captured by a recording mock GPUDevice and pinned to GOLDEN values
 * computed independently in THIS file (the same forward-dim simulation the
 * production graph runs).
 *
 * The dims-once change (compute the tensor-dim map ONCE at initialize() instead
 * of per-layer per-frame inside run()) is a PERF change that MUST be output
 * identical — so the golden dispatch + uniform bytes here are exactly what the
 * pre-refactor code emitted. If the refactor changed any dispatched value this
 * test fails.
 *
 * No real GPU: a recording mock device with the minimal WebGPU surface the
 * InferenceGraph touches (createBuffer / createShaderModule /
 * createComputePipelineAsync / createBindGroup / command-encoder / queue).
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
installWebGPUPolyfills();

import { InferenceGraph } from '../src/neural/InferenceGraph.js';
import { buildUNetSpec } from '../src/neural/unetArchitecture.js';
import type { UNetSpec, LayerSpec } from '../src/neural/unetArchitecture.js';
import type { ModelWeights, LayerWeights } from '../src/neural/weights.js';

// ─── Recording mock GPUDevice ────────────────────────────────────────────────
// Captures exactly the host-issued GPU calls InferenceGraph makes. Every buffer
// gets a monotone id + the label; dispatch + uniform writes are recorded against
// the compute pass label so we can attribute them to a layer.

interface Recorded {
  dispatches: { label: string; gx: number; gy: number; gz: number }[];
  uniformWrites: { bufLabel: string; bytes: number[] }[];
  copies: { srcLabel: string; dstLabel: string; size: number }[];
}

function makeRecordingDevice(rec: Recorded): GPUDevice {
  let bufId = 0;
  const mkBuf = (label: string, size: number): GPUBuffer => {
    const b = {
      label: label || `buf${bufId++}`,
      size,
      destroy() {},
      getMappedRange() { return new ArrayBuffer(size); },
      unmap() {},
    };
    return b as unknown as GPUBuffer;
  };

  const queue = {
    writeBuffer(buf: GPUBuffer, _off: number, data: ArrayBuffer | ArrayBufferView) {
      // Snapshot the written bytes so the uniform write can be compared.
      let view: Uint8Array;
      if (data instanceof ArrayBuffer) view = new Uint8Array(data);
      else view = new Uint8Array(
        (data as ArrayBufferView).buffer,
        (data as ArrayBufferView).byteOffset,
        (data as ArrayBufferView).byteLength,
      );
      const label = (buf as unknown as { label: string }).label;
      // Only record uniform buffers (the layer params + the inputPack count).
      if (label.includes('uniform') || label.includes('inputPack')) {
        rec.uniformWrites.push({ bufLabel: label, bytes: Array.from(view) });
      }
    },
    submit() {},
  };

  const mkPass = (label: string) => ({
    setPipeline() {},
    setBindGroup() {},
    dispatchWorkgroups(gx: number, gy = 1, gz = 1) {
      rec.dispatches.push({ label, gx, gy, gz });
    },
    end() {},
  });

  const mkEncoder = () => ({
    beginComputePass(desc?: { label?: string }) { return mkPass(desc?.label ?? ''); },
    copyBufferToBuffer(src: GPUBuffer, _so: number, dst: GPUBuffer, _do: number, size: number) {
      rec.copies.push({
        srcLabel: (src as unknown as { label: string }).label,
        dstLabel: (dst as unknown as { label: string }).label,
        size,
      });
    },
    finish() { return {}; },
  });

  const device = {
    createBuffer(desc: { label?: string; size: number }) {
      return mkBuf(desc.label ?? '', desc.size);
    },
    createShaderModule() {
      return { getCompilationInfo: async () => ({ messages: [] }) };
    },
    async createComputePipelineAsync() {
      return { getBindGroupLayout: () => ({}) };
    },
    createComputePipeline() {
      return { getBindGroupLayout: () => ({}) };
    },
    createBindGroup() { return {}; },
    createCommandEncoder() { return mkEncoder(); },
    queue,
    limits: {
      maxComputeWorkgroupStorageSize: 32768,
      maxComputeWorkgroupsPerDimension: 65_535,
    },
  };
  return device as unknown as GPUDevice;
}

// ─── Golden: independent forward-dim simulation (mirror of the production one) ──

interface Dims { H: number; W: number; C: number; }

function goldenDims(spec: UNetSpec, W: number, H: number): Map<string, Dims> {
  const dims = new Map<string, Dims>();
  dims.set('noisyColor', { H, W, C: 3 });
  dims.set('albedo', { H, W, C: 3 });
  dims.set('normals', { H, W, C: 3 });
  for (const layer of spec.layers) {
    const inDims = layer.inputs.length > 0 ? dims.get(layer.inputs[0]!) : undefined;
    if (!inDims && layer.kind !== 'inputPack') continue;
    let outH = inDims?.H ?? H;
    let outW = inDims?.W ?? W;
    let outC = layer.params.outC;
    switch (layer.kind) {
      case 'inputPack':
        dims.set(layer.output, { H, W, C: 9 });
        continue;
      case 'conv2d': {
        const kH = layer.params.kH ?? 3, kW = layer.params.kW ?? 3;
        const s = layer.params.stride ?? 1;
        const p = layer.params.padding ?? (kH === 3 && kW === 3 ? 1 : 0);
        outH = Math.floor((outH + 2 * p - kH) / s) + 1;
        outW = Math.floor((outW + 2 * p - kW) / s) + 1;
        break;
      }
      case 'transposedConv2d': {
        const kH = layer.params.kH ?? 2, kW = layer.params.kW ?? 2;
        const s = layer.params.stride ?? 2, p = layer.params.padding ?? 0;
        const d = layer.params.dilation ?? 1, op = layer.params.outputPadding ?? 0;
        outH = (outH - 1) * s - 2 * p + d * (kH - 1) + op + 1;
        outW = (outW - 1) * s - 2 * p + d * (kW - 1) + op + 1;
        break;
      }
      case 'relu':
      case 'skipAdd':
        outC = inDims!.C;
        break;
    }
    dims.set(layer.output, { H: outH, W: outW, C: outC });
  }
  return dims;
}

function goldenDispatch(kind: LayerSpec['kind'], d: Dims): [number, number, number] {
  switch (kind) {
    case 'conv2d':
    case 'transposedConv2d':
      return [Math.ceil(d.H / 8), Math.ceil(d.W / 8), d.C];
    case 'relu':
    case 'skipAdd':
      return [Math.ceil((d.H * d.W * d.C) / 256), 1, 1];
    default:
      return [1, 1, 1];
  }
}

// Pack the uniform bytes the production graph writes for one layer.
function goldenUniformBytes(layer: LayerSpec, dimsMap: Map<string, Dims>, H: number, W: number): number[] {
  const u32 = new Uint32Array(12);
  const inDims = layer.inputs.length > 0 ? dimsMap.get(layer.inputs[0]!) : undefined;
  switch (layer.kind) {
    case 'conv2d':
      u32[0] = inDims?.H ?? H; u32[1] = inDims?.W ?? W;
      u32[2] = layer.params.inC; u32[3] = layer.params.outC;
      u32[4] = layer.params.kH ?? 3; u32[5] = layer.params.kW ?? 3;
      u32[6] = layer.params.stride ?? 1;
      u32[7] = layer.params.padding ?? (u32[4] === 3 && u32[5] === 3 ? 1 : 0);
      break;
    case 'transposedConv2d':
      u32[0] = inDims?.H ?? H; u32[1] = inDims?.W ?? W;
      u32[2] = layer.params.inC; u32[3] = layer.params.outC;
      u32[4] = layer.params.kH ?? 2; u32[5] = layer.params.kW ?? 2;
      u32[6] = layer.params.stride ?? 2; u32[7] = layer.params.padding ?? 0;
      u32[8] = layer.params.dilation ?? 1;
      u32[9] = layer.params.outputPadding ?? 0;
      break;
    case 'relu':
    case 'skipAdd': {
      const count = (inDims?.H ?? H) * (inDims?.W ?? W) * (inDims?.C ?? layer.params.inC);
      u32[0] = count;
      u32[1] = Math.ceil(count / 256);
      break;
    }
    default:
      break;
  }
  return Array.from(new Uint8Array(u32.buffer));
}

// ─── A small synthetic weights blob for the spec (zeros — values never matter
//     for the dispatch/uniform structure this test pins). ──

function makeWeights(spec: UNetSpec): ModelWeights {
  const layers: LayerWeights[] = [];
  for (const layer of spec.layers) {
    if (layer.kind === 'conv2d' || layer.kind === 'transposedConv2d') {
      const kH = layer.params.kH ?? 3, kW = layer.params.kW ?? 3;
      const wlen = layer.params.outC * layer.params.inC * kH * kW;
      layers.push({
        name: layer.name,
        weights: new Float32Array(wlen),
        biases: new Float32Array(layer.params.outC),
      });
    }
  }
  return { layers, modelName: 'test', totalParams: 0 } as unknown as ModelWeights;
}

describe('InferenceGraph — dims-once + decomposition behavior-identity (Task 4.5 Theme I)', () => {
  const W = 32, H = 32;
  let spec: UNetSpec;

  beforeAll(() => { spec = buildUNetSpec(); });

  // Allocate a labelled input buffer on the recording mock (cast: the mock's
  // createBuffer only needs {label,size} but the GPUDevice type wants a full
  // GPUBufferDescriptor).
  const mkBuf = (device: GPUDevice, label: string): GPUBuffer =>
    device.createBuffer({ label, size: H * W * 3 * 4 } as GPUBufferDescriptor);

  it('emits the GOLDEN per-layer dispatch sequence (inputPack + each compute layer)', async () => {
    const rec: Recorded = { dispatches: [], uniformWrites: [], copies: [] };
    const device = makeRecordingDevice(rec);
    const graph = new InferenceGraph(spec);
    await graph.initialize(device, makeWeights(spec), W, H);

    const noisy = mkBuf(device, 'noisy');
    const albedo = mkBuf(device, 'albedo');
    const normals = mkBuf(device, 'normals');
    const out = mkBuf(device, 'out');

    graph.run(noisy, albedo, normals, out);

    // Build the golden dispatch list.
    const dimsMap = goldenDims(spec, W, H);
    const golden: { label: string; gx: number; gy: number; gz: number }[] = [];
    // inputPack first.
    golden.push({ label: 'neural-inputPack', gx: Math.ceil((H * W) / 256), gy: 1, gz: 1 });
    for (const layer of spec.layers) {
      if (layer.kind === 'inputPack') continue;
      if (!('conv2d transposedConv2d relu skipAdd'.includes(layer.kind))) continue;
      const od = dimsMap.get(layer.output);
      if (!od) continue;
      const [gx, gy, gz] = goldenDispatch(layer.kind, od);
      golden.push({ label: `neural-${layer.name}`, gx, gy, gz });
    }

    expect(rec.dispatches).toEqual(golden);
  });

  it('writes the GOLDEN uniform bytes for every layer (dims-once must not change them)', async () => {
    const rec: Recorded = { dispatches: [], uniformWrites: [], copies: [] };
    const device = makeRecordingDevice(rec);
    const graph = new InferenceGraph(spec);
    await graph.initialize(device, makeWeights(spec), W, H);

    const noisy = mkBuf(device, 'noisy');
    const albedo = mkBuf(device, 'albedo');
    const normals = mkBuf(device, 'normals');
    const out = mkBuf(device, 'out');

    // Capture only the run()-phase uniform writes (the per-frame re-writes).
    const initWriteCount = rec.uniformWrites.length;
    graph.run(noisy, albedo, normals, out);
    const runWrites = rec.uniformWrites.slice(initWriteCount);

    const dimsMap = goldenDims(spec, W, H);
    // run() re-writes each compute layer's uniform (NOT the inputPack uniform,
    // which is written once at init). Build the expected per-layer byte list.
    const expected: { bufLabel: string; bytes: number[] }[] = [];
    for (const layer of spec.layers) {
      if (layer.kind === 'inputPack') continue;
      if (!('conv2d transposedConv2d relu skipAdd'.includes(layer.kind))) continue;
      expected.push({
        bufLabel: `neural-uniform-${layer.name}`,
        bytes: goldenUniformBytes(layer, dimsMap, H, W),
      });
    }
    expect(runWrites).toEqual(expected);
  });

  it('copies the denoised tensor to output with the GOLDEN extent', async () => {
    const rec: Recorded = { dispatches: [], uniformWrites: [], copies: [] };
    const device = makeRecordingDevice(rec);
    const graph = new InferenceGraph(spec);
    await graph.initialize(device, makeWeights(spec), W, H);

    const noisy = mkBuf(device, 'noisy');
    const albedo = mkBuf(device, 'albedo');
    const normals = mkBuf(device, 'normals');
    const out = mkBuf(device, 'out');
    graph.run(noisy, albedo, normals, out);

    const dimsMap = goldenDims(spec, W, H);
    const dd = dimsMap.get('denoised')!;
    expect(rec.copies).toHaveLength(1);
    expect(rec.copies[0]!.size).toBe(dd.H * dd.W * dd.C * 4);
    expect(rec.copies[0]!.srcLabel).toMatch(
      /^neural\/slot-\d+:.*(?:^|,)denoised(?:,|$)/,
    );
    expect(rec.copies[0]!.dstLabel).toBe('out');
  });

  it('zero-pads an odd logical extent and dispatches an exact output crop', async () => {
    const logicalW = 9;
    const logicalH = 8;
    const rec: Recorded = { dispatches: [], uniformWrites: [], copies: [] };
    const device = makeRecordingDevice(rec);
    const graph = new InferenceGraph(spec);
    await graph.initialize(device, makeWeights(spec), logicalW, logicalH);

    const bytes = logicalW * logicalH * 3 * 4;
    const buffer = (label: string): GPUBuffer => device.createBuffer({
      label,
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    graph.run(buffer('odd-noisy'), buffer('odd-albedo'), buffer('odd-normals'), buffer('odd-out'));

    expect(graph.inferenceWidth).toBe(16);
    expect(graph.inferenceHeight).toBe(8);
    expect(rec.dispatches[0]).toEqual({
      label: 'neural-inputPack',
      gx: 1,
      gy: 1,
      gz: 1,
    });
    expect(rec.dispatches.at(-1)).toEqual({
      label: 'neural-outputCrop',
      gx: 1,
      gy: 1,
      gz: 1,
    });
    expect(rec.copies).toHaveLength(0);
    const u32 = (label: string): number[] => {
      const write = rec.uniformWrites.find(entry => entry.bufLabel === label);
      expect(write).toBeDefined();
      return Array.from(new Uint32Array(Uint8Array.from(write!.bytes).buffer));
    };
    expect(u32('neural-uniform-inputPack')).toEqual([9, 8, 16, 8]);
    expect(u32('neural-uniform-outputCrop')).toEqual([9, 8, 16, 0]);
  });

  it('two runs emit an identical dispatch sequence (idempotent per-frame)', async () => {
    const rec: Recorded = { dispatches: [], uniformWrites: [], copies: [] };
    const device = makeRecordingDevice(rec);
    const graph = new InferenceGraph(spec);
    await graph.initialize(device, makeWeights(spec), W, H);

    const noisy = mkBuf(device, 'noisy');
    const albedo = mkBuf(device, 'albedo');
    const normals = mkBuf(device, 'normals');
    const out = mkBuf(device, 'out');

    graph.run(noisy, albedo, normals, out);
    const firstRun = rec.dispatches.slice();
    rec.dispatches.length = 0;
    graph.run(noisy, albedo, normals, out);
    expect(rec.dispatches).toEqual(firstRun);
  });
});
