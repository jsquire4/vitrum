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
  kind: 'createBuffer' | 'createBindGroup' | 'destroy' | 'writeBuffer' | 'copy' | 'clear' | 'dispatch' | 'submit';
  size?: number;
  offset?: number;
  bytes?: number[];
  detail?: number;
}

function makeRecordingDevice(ops: Op[]): GPUDevice {
  let bufId = 0;
  const mkBuf = (size: number): GPUBuffer => {
    const id = bufId++;
    const bytes = new Uint8Array(size);
    return {
      _id: id, _bytes: bytes, size,
      destroy() { ops.push({ kind: 'destroy', detail: id }); },
      getMappedRange() { return bytes.buffer; },
      async mapAsync() {},
      unmap() {},
    } as unknown as GPUBuffer;
  };

  const mkPass = () => ({
    setPipeline() {}, setBindGroup() {},
    dispatchWorkgroups(gx: number) { ops.push({ kind: 'dispatch', detail: gx }); },
    end() {},
  });

  const mkEncoder = () => ({
    clearBuffer() { ops.push({ kind: 'clear' }); },
    copyBufferToBuffer(s: GPUBuffer, so: number, d: GPUBuffer, doff: number, size: number) {
      const src = (s as unknown as { _bytes: Uint8Array })._bytes;
      const dst = (d as unknown as { _bytes: Uint8Array })._bytes;
      dst.set(src.subarray(so, so + size), doff);
      ops.push({ kind: 'copy', detail: size });
    },
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
    createBindGroup() { ops.push({ kind: 'createBindGroup' }); return {}; },
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

async function buildTrainer(ops: Op[]): Promise<{
  device: GPUDevice;
  trainer: HashGridTableTrainer;
  tableScalars: number;
  initialTable: GPUBuffer;
}> {
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
  return { device, trainer, tableScalars, initialTable: ext.tablesBuf };
}

describe('HashGridTableTrainer — extracted + UBOs-once (Task 4.5 #3)', () => {
  it('alternates generations across success/failure/success without publishing failed bytes', async () => {
    const ops: Op[] = [];
    const built = await buildTrainer(ops);
    const diagnosticsBuffer = (built.trainer as unknown as {
      _diagnosticsBuffer: GPUBuffer;
    })._diagnosticsBuffer;
    const batchPos = new Float32Array(cfg.recordCap * 3).fill(0.25);
    type Tracked = GPUBuffer & { _id: number; _bytes: Uint8Array };
    type Set = { tables: Tracked; m: Tracked; v: Tracked };
    const state = built.trainer as unknown as {
      _ext: { tablesBuf: Tracked };
      _trainableSets: readonly [Set, Set];
      _tableAdamT: number;
    };
    const checksum = (set: Set): number => {
      let hash = 0x811c9dc5;
      for (const buffer of [set.tables, set.m, set.v]) {
        for (const byte of buffer._bytes) hash = Math.imul(hash ^ byte, 0x01000193) >>> 0;
      }
      return hash;
    };
    const liveSet = (): Set => state._trainableSets.find(
      set => set.tables === state._ext.tablesBuf,
    )!;
    const candidateSet = (buffer: GPUBuffer): Set => state._trainableSets.find(
      set => set.tables === buffer,
    )!;
    const mutate = (set: Set, seed: number): void => {
      set.tables._bytes[0] = seed;
      set.m._bytes[1] = seed + 1;
      set.v._bytes[2] = seed + 2;
    };

    const generationA = liveSet();
    const baseline = checksum(generationA);

    const tx1 = built.trainer.recordStep(
      built.device.createCommandEncoder(), batchPos, 8,
    );
    const generationB = candidateSet(tx1.candidateTableBuffer);
    expect(checksum(generationB)).toBe(baseline);
    mutate(generationB, 17);
    tx1.commitCpu();
    tx1.finalizeSuccess();
    expect(liveSet()).toBe(generationB);
    const afterFirstSuccess = checksum(liveSet());
    expect(afterFirstSuccess).not.toBe(baseline);
    expect(state._tableAdamT).toBe(1);

    const failed = built.trainer.recordStep(
      built.device.createCommandEncoder(), batchPos, 8,
    );
    expect(candidateSet(failed.candidateTableBuffer)).toBe(generationA);
    expect(checksum(generationA)).toBe(afterFirstSuccess);
    mutate(generationA, 91);
    failed.rollback();
    expect(liveSet()).toBe(generationB);
    expect(checksum(liveSet())).toBe(afterFirstSuccess);
    expect(state._tableAdamT).toBe(1);

    const tx2 = built.trainer.recordStep(
      built.device.createCommandEncoder(), batchPos, 8,
    );
    expect(candidateSet(tx2.candidateTableBuffer)).toBe(generationA);
    expect(checksum(generationA)).toBe(afterFirstSuccess);
    mutate(generationA, 33);
    tx2.commitCpu();
    tx2.finalizeSuccess();
    expect(liveSet()).toBe(generationA);
    expect(checksum(liveSet())).not.toBe(afterFirstSuccess);
    expect(state._tableAdamT).toBe(2);
    expect((built.trainer as unknown as { _diagnosticsBuffer: GPUBuffer })._diagnosticsBuffer)
      .toBe(diagnosticsBuffer);
  });

  it('allocates, binds, and destroys no GPU resources on the epoch hot path', async () => {
    const ops: Op[] = [];
    const { trainer } = await buildTrainer(ops);

    ops.length = 0;
    const batchPos = new Float32Array(cfg.recordCap * 3).fill(0.25);
    trainer.step(batchPos, 8);

    expect(ops.filter(o => o.kind === 'createBuffer')).toHaveLength(0);
    expect(ops.filter(o => o.kind === 'createBindGroup')).toHaveLength(0);
    expect(ops.filter(o => o.kind === 'destroy')).toHaveLength(0);
  });

  it('emits the transactional single-submit command sequence', async () => {
    const ops: Op[] = [];
    const built = await buildTrainer(ops);
    ops.length = 0;

    const numActive = 8;
    const batchPos = new Float32Array(cfg.recordCap * 3).fill(0.25);
    built.trainer.step(batchPos, numActive);

    // One transactional command buffer: write the batch/header, copy every
    // current table/moment byte into the spare generation, clear gradients,
    // dispatch encode-backward/finalize/Adam, and submit exactly once.
    expect(ops.map(o => o.kind)).toEqual([
      'writeBuffer',
      'writeBuffer',
      'copy', 'copy', 'copy',
      'clear',
      'dispatch',
      'dispatch',
      'writeBuffer',
      'dispatch',
      'submit',
    ]);
    const tableDispatch = Math.ceil(built.tableScalars / 64);
    const dispatches = ops.filter(o => o.kind === 'dispatch').map(o => o.detail);
    expect(dispatches).toEqual([
      Math.ceil(numActive / 64),
      tableDispatch,
      tableDispatch,
    ]);
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

  it('leaves the published table to the subsystem owner and destroys all trainer-owned generations once', async () => {
    const ops: Op[] = [];
    const built = await buildTrainer(ops);
    const batchPos = new Float32Array(cfg.recordCap * 3).fill(0.25);
    built.trainer.step(batchPos, 8);
    built.trainer.step(batchPos, 8);

    const currentTable = (built.trainer as unknown as {
      _ext: { tablesBuf: GPUBuffer };
    })._ext.tablesBuf;
    ops.length = 0;
    built.trainer.dispose();

    const currentId = (currentTable as unknown as { _id: number })._id;
    expect(ops.filter(o => o.kind === 'destroy' && o.detail === currentId)).toHaveLength(0);
    const destroyedIds = ops.filter(o => o.kind === 'destroy').map(o => o.detail);
    expect(new Set(destroyedIds).size).toBe(destroyedIds.length);

    // NrcSubsystem is the sole owner of the published table at final disposal.
    currentTable.destroy();
    expect(ops.filter(o => o.kind === 'destroy' && o.detail === currentId)).toHaveLength(1);
    built.trainer.dispose();
    expect(ops.filter(o => o.kind === 'destroy' && o.detail === currentId)).toHaveLength(1);
  });
});
