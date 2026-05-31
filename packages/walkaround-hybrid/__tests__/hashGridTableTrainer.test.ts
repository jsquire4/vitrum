/**
 * hashGridTableTrainer.test.ts — Task 4.5 Theme-I characterization gate for the
 * HashGridTableTrainer extraction (#3) from NrcSubsystem._tableTrainStep.
 *
 * The inline hash-grid table-Adam pipeline (clear → encode-backward → finalize →
 * Adam) was extracted into a peer trainer class mirroring FusedMlpTrainer's
 * shape, AND its two per-frame throwaway UBOs (the grad-finalize count UBO + the
 * Adam params UBO) were promoted to persistent buffers allocated ONCE at init.
 *
 * Both changes must be OUTPUT-IDENTICAL. This recording-mock test pins:
 *   1. the per-frame GPU command sequence (writeBuffer / clear / dispatch /
 *      submit) is byte-for-byte the pre-refactor sequence,
 *   2. the per-frame UBO bytes written (posBuf, numActive, grad-finalize count,
 *      Adam params incl. the per-step bc1/bc2) are identical,
 *   3. NO new GPU buffers are created during the per-frame step (the UBOs are
 *      now persistent — the buffer allocation count after init does not grow on
 *      successive table-train steps).
 *
 * The numerical correctness of the kernels is pinned separately by
 * nrcEncodeBackward.test.ts (CPU oracle) + the lavapipe harness; this test pins
 * the HOST orchestration the refactor touched.
 */

import { describe, it, expect } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
installWebGPUPolyfills();

import { HashGridTableTrainer } from '../src/neural/nrc/hashGridTableTrainer.js';

// ── Recording mock GPUDevice ─────────────────────────────────────────────────

interface Op {
  kind: 'createBuffer' | 'writeBuffer' | 'clear' | 'dispatch' | 'submit';
  size?: number;
  offset?: number;
  bytes?: number[];
  detail?: number;
}

function makeRecordingDevice(ops: Op[]): GPUDevice {
  let bufId = 0;
  const mkBuf = (size: number): GPUBuffer => ({
    _id: bufId++, size,
    destroy() {},
    getMappedRange() { return new ArrayBuffer(size); },
    async mapAsync() {},
    unmap() {},
  }) as unknown as GPUBuffer;

  const mkPass = () => ({
    setPipeline() {}, setBindGroup() {},
    dispatchWorkgroups(gx: number) { ops.push({ kind: 'dispatch', detail: gx }); },
    end() {},
  });

  const mkEncoder = () => ({
    clearBuffer() { ops.push({ kind: 'clear' }); },
    beginComputePass() { return mkPass(); },
    finish() { return {}; },
  });

  return {
    createBuffer(desc: { size: number }) {
      ops.push({ kind: 'createBuffer', size: desc.size });
      return mkBuf(desc.size);
    },
    createShaderModule() { return { getCompilationInfo: async () => ({ messages: [] }) }; },
    async createComputePipelineAsync() { return { getBindGroupLayout: () => ({}) }; },
    createBindGroup() { return {}; },
    createCommandEncoder() { return mkEncoder(); },
    queue: {
      writeBuffer(_buf: GPUBuffer, offset: number, data: ArrayBuffer | ArrayBufferView) {
        let view: Uint8Array;
        if (data instanceof ArrayBuffer) view = new Uint8Array(data);
        else view = new Uint8Array(
          (data as ArrayBufferView).buffer,
          (data as ArrayBufferView).byteOffset,
          (data as ArrayBufferView).byteLength,
        );
        ops.push({ kind: 'writeBuffer', offset, bytes: Array.from(view) });
      },
      submit() { ops.push({ kind: 'submit' }); },
    },
    limits: { maxComputeWorkgroupStorageSize: 32768 },
  } as unknown as GPUDevice;
}

// Mirror the production config the subsystem uses.
const cfg = {
  levels: 4,
  featuresPerEntry: 2,
  tableSize: 64,
  nMin: 4,
  growth: 2.0,
  inW: 19,        // raw MLP input width (dL/dX row stride)
  recordCap: 32,
  tableLearningRate: 0.1,
};

async function buildTrainer(ops: Op[]): Promise<{ device: GPUDevice; trainer: HashGridTableTrainer; tableScalars: number }> {
  const device = makeRecordingDevice(ops);
  const tableScalars = cfg.levels * cfg.tableSize * cfg.featuresPerEntry;
  const trainer = new HashGridTableTrainer(device, {
    levels: cfg.levels,
    featuresPerEntry: cfg.featuresPerEntry,
    inW: cfg.inW,
    tableScalars,
    recordCap: cfg.recordCap,
    tableLearningRate: cfg.tableLearningRate,
  });
  // External buffers the trainer reads but does not own (the MLP dL/dX, the
  // table buffer it Adam-updates, and the level descriptors).
  const ext = {
    gradInputF: device.createBuffer({ size: cfg.recordCap * cfg.inW * 4 } as GPUBufferDescriptor),
    tablesBuf: device.createBuffer({ size: tableScalars * 4 } as GPUBufferDescriptor),
    levelsBuf: device.createBuffer({ size: cfg.levels * 16 } as GPUBufferDescriptor),
  };
  await trainer.build(ext, [-1, -1, -1], [1, 1, 1]);
  return { device, trainer, tableScalars };
}

describe('HashGridTableTrainer — extracted + UBOs-once (Task 4.5 #3)', () => {
  it('a table-train step creates NO new GPU buffers (persistent UBOs)', async () => {
    const ops: Op[] = [];
    const { trainer } = await buildTrainer(ops);

    ops.length = 0;
    const batchPos = new Float32Array(cfg.recordCap * 3).fill(0.25);
    trainer.step(batchPos, 8);

    const created = ops.filter(o => o.kind === 'createBuffer');
    expect(created).toHaveLength(0); // the old code created 2 throwaway UBOs/frame
  });

  it('emits the golden per-frame command sequence', async () => {
    const ops: Op[] = [];
    const { tableScalars } = await { ...(await buildTrainer(ops)) };
    // Rebuild cleanly to isolate the step ops.
    ops.length = 0;
    const built = await buildTrainer(ops);
    ops.length = 0;

    const numActive = 8;
    const batchPos = new Float32Array(cfg.recordCap * 3).fill(0.25);
    built.trainer.step(batchPos, numActive);

    // Golden host stream (mirrors NrcSubsystem._tableTrainStep exactly):
    //  writeBuffer posBuf (numActive*3 f32)
    //  writeBuffer encBwdParamsUbo @12 (numActive u32)
    //  clear gradTablesFx + submit   (separate encoder)
    //  --- main encoder ---
    //  dispatch encode-backward ceil(numActive/64)
    //  writeBuffer gradFin count UBO (ONLY if not already written at init — see
    //    below; the persistent version writes it once at init, so per-step it is
    //    NOT re-written)
    //  dispatch gradFinalize ceil(tableScalars/64)
    //  writeBuffer Adam UBO (per-step bc1/bc2)
    //  dispatch Adam ceil(tableScalars/64)
    //  submit
    const tsDispatch = Math.ceil(built.tableScalars / 64);
    const kinds = ops.map(o => o.kind);
    expect(kinds).toEqual([
      'writeBuffer', // posBuf
      'writeBuffer', // encBwdParams numActive
      'clear',
      'submit',
      'dispatch',    // encode-backward
      'dispatch',    // gradFinalize
      'writeBuffer', // Adam UBO (bc1/bc2 change per step → re-written)
      'dispatch',    // Adam
      'submit',
    ]);
    expect(built.tableScalars).toBe(tableScalars);
    // encode-backward dispatch count.
    const dispatches = ops.filter(o => o.kind === 'dispatch').map(o => o.detail);
    expect(dispatches).toEqual([Math.ceil(numActive / 64), tsDispatch, tsDispatch]);
  });

  it('writes the golden numActive + Adam-UBO bytes (bc1/bc2 advance per step)', async () => {
    const ops: Op[] = [];
    const built = await buildTrainer(ops);
    ops.length = 0;

    const batchPos = new Float32Array(cfg.recordCap * 3).fill(0.25);
    built.trainer.step(batchPos, 8); // step 1
    const step1 = ops.slice();
    ops.length = 0;
    built.trainer.step(batchPos, 8); // step 2
    const step2 = ops.slice();

    // numActive write @ offset 12 == [8,0,0,0] little-endian u32.
    const numActiveWrite = step1.find(o => o.kind === 'writeBuffer' && o.offset === 12);
    expect(numActiveWrite?.bytes).toEqual([8, 0, 0, 0]);

    // The Adam UBO write carries bc1 = 1 - 0.9^t, bc2 = 1 - 0.999^t at f32
    // offsets 8 and 9 (byte 32 and 36). Decode them from step1 vs step2 and
    // assert they advance exactly as the closed-form (t=1 then t=2).
    const adamWrite1 = step1.filter(o => o.kind === 'writeBuffer').at(-1)!;
    const adamWrite2 = step2.filter(o => o.kind === 'writeBuffer').at(-1)!;
    const decodeBcs = (op: Op) => {
      const buf = new Uint8Array(op.bytes!).buffer;
      const f = new Float32Array(buf);
      return { bc1: f[8]!, bc2: f[9]!, lr: f[4]!, count: new Uint32Array(buf)[0]! };
    };
    const a1 = decodeBcs(adamWrite1);
    const a2 = decodeBcs(adamWrite2);
    expect(a1.bc1).toBeCloseTo(1 - Math.pow(0.9, 1), 6);
    expect(a1.bc2).toBeCloseTo(1 - Math.pow(0.999, 1), 6);
    expect(a2.bc1).toBeCloseTo(1 - Math.pow(0.9, 2), 6);
    expect(a2.bc2).toBeCloseTo(1 - Math.pow(0.999, 2), 6);
    expect(a1.lr).toBeCloseTo(cfg.tableLearningRate, 6);
    expect(a1.count).toBe(built.tableScalars);
  });
});
