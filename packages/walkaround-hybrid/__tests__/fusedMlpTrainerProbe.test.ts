/**
 * fusedMlpTrainerProbe.test.ts — Task 4.5 Theme-I characterization gate for the
 * FusedMlpTrainerProbe extraction (#4).
 *
 * The FD/loss/readback debug helpers (computeGradsStep / computeLoss / readGrads
 * / readInputGrads) moved off FusedMlpTrainer into FusedMlpTrainerProbe. This is
 * a pure code move — the probe must issue the EXACT same GPU command sequence the
 * methods issued when they lived on the trainer.
 *
 * vitest has no real GPU, so we record the host-issued GPU calls against a
 * recording mock device and pin the dispatch / clear / copy / readback sequence
 * that `computeGradsStep` emits to a GOLDEN sequence derived from the trainer's
 * layer plan. (The numerical correctness of the kernels is proven separately by
 * the lavapipe harness; this test pins the HOST orchestration the refactor
 * touched.)
 */

import { describe, it, expect } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
installWebGPUPolyfills();

import { FusedMlpTrainer, type FusedNetSpec, planLayers } from '../src/neural/nrc/fusedMlpTrainer.js';
import { FusedMlpTrainerProbe } from '../src/neural/nrc/fusedMlpTrainerProbe.js';

// ── Recording mock GPUDevice ─────────────────────────────────────────────────
// Records the host-visible command stream: buffer creations, clearBuffer,
// dispatchWorkgroups, copyBufferToBuffer, and queue submits.

interface Op {
  kind: 'clear' | 'dispatch' | 'copy' | 'submit';
  detail?: number;
}

function makeRecordingDevice(ops: Op[]): GPUDevice {
  const mkBuf = (size: number): GPUBuffer => ({
    size,
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
    copyBufferToBuffer(_s: GPUBuffer, _so: number, _d: GPUBuffer, _do: number, size: number) {
      ops.push({ kind: 'copy', detail: size });
    },
    finish() { return {}; },
  });

  return {
    createBuffer(desc: { size: number }) { return mkBuf(desc.size); },
    createShaderModule() { return { getCompilationInfo: async () => ({ messages: [] }) }; },
    async createComputePipelineAsync() { return { getBindGroupLayout: () => ({}) }; },
    createBindGroup() { return {}; },
    createCommandEncoder() { return mkEncoder(); },
    queue: {
      writeBuffer() {},
      submit() { ops.push({ kind: 'submit' }); },
    },
    limits: { maxComputeWorkgroupStorageSize: 32768 },
  } as unknown as GPUDevice;
}

describe('FusedMlpTrainerProbe — extracted debug surface issues identical commands (Task 4.5 #4)', () => {
  const spec: FusedNetSpec = { inW: 4, W: 8, outW: 3, hidden: 2 };
  const tileB = 8;
  const B = 5;

  it('computeGradsStep emits the golden clear→forward→backward→3×finalize sequence', async () => {
    const ops: Op[] = [];
    const device = makeRecordingDevice(ops);
    const trainer = new FusedMlpTrainer(device, spec, { useF16: false, tileB });
    await trainer.build(B);
    const probe = new FusedMlpTrainerProbe(trainer);

    ops.length = 0; // discard build-time ops
    probe.computeGradsStep();

    const plan = planLayers(spec);
    const numTiles = Math.ceil(B / tileB);
    // Golden sequence (the EXACT host stream the pre-split method issued):
    //  enc0: 3 clears + submit
    //  enc:  forward dispatch (numTiles) + backward dispatch (numTiles)
    //        + 3 grad-finalize dispatches (ceil(count/64)) + submit
    const fin = (count: number) => Math.ceil(count / 64);
    const golden: Op[] = [
      { kind: 'clear' }, { kind: 'clear' }, { kind: 'clear' },
      { kind: 'submit' },
      { kind: 'dispatch', detail: numTiles },
      { kind: 'dispatch', detail: numTiles },
      { kind: 'dispatch', detail: fin(plan.totalW) },
      { kind: 'dispatch', detail: fin(plan.totalB) },
      { kind: 'dispatch', detail: fin(B * spec.inW) },
      { kind: 'submit' },
    ];
    expect(ops).toEqual(golden);
  });

  it('readGrads issues two readback copies sized to totalW and totalB', async () => {
    const ops: Op[] = [];
    const device = makeRecordingDevice(ops);
    const trainer = new FusedMlpTrainer(device, spec, { useF16: false, tileB });
    await trainer.build(B);
    const probe = new FusedMlpTrainerProbe(trainer);

    ops.length = 0;
    await probe.readGrads();

    const plan = planLayers(spec);
    const copies = ops.filter(o => o.kind === 'copy').map(o => o.detail);
    expect(copies).toEqual([
      Math.max(16, plan.totalW * 4),
      Math.max(16, plan.totalB * 4),
    ]);
  });

  it('readInputGrads reads back [numSamples × inW] f32', async () => {
    const ops: Op[] = [];
    const device = makeRecordingDevice(ops);
    const trainer = new FusedMlpTrainer(device, spec, { useF16: false, tileB });
    await trainer.build(B);
    const probe = new FusedMlpTrainerProbe(trainer);

    ops.length = 0;
    await probe.readInputGrads();

    const copies = ops.filter(o => o.kind === 'copy').map(o => o.detail);
    expect(copies).toEqual([Math.max(16, B * spec.inW * 4)]);
  });

  it('production trainer surface no longer exposes the debug helpers', async () => {
    const device = makeRecordingDevice([]);
    const trainer = new FusedMlpTrainer(device, spec, { useF16: false, tileB });
    await trainer.build(B);
    const anyTrainer = trainer as unknown as Record<string, unknown>;
    expect(anyTrainer['computeGradsStep']).toBeUndefined();
    expect(anyTrainer['computeLoss']).toBeUndefined();
    expect(anyTrainer['readGrads']).toBeUndefined();
    expect(anyTrainer['readInputGrads']).toBeUndefined();
    // The production hot-path surface remains.
    expect(typeof trainer.trainStep).toBe('function');
    expect(typeof trainer.setBatch).toBe('function');
    expect(typeof trainer.setWeights).toBe('function');
    expect(typeof trainer.dispose).toBe('function');
  });
});

// ── BUG-1 fix: persistent params UBO characterization ────────────────────────
//
// Before the fix, paramsUniform() / recordGradFinalize() / recordAdam() /
// recordDowncast() all called d.createBuffer() on EVERY call — 9–11 throwaway
// GPUBuffers PER trainStep(). The fix allocates a fixed set of persistent UBOs
// in build() and rewrites them in-place (or uses them read-only for constant
// values). This characterization test pins that no new GPU buffers are created
// during trainStep() calls after build() completes.

describe('FusedMlpTrainer — persistent params UBO: no new buffers per trainStep (BUG-1 fix)', () => {
  const spec: FusedNetSpec = { inW: 4, W: 8, outW: 3, hidden: 2 };
  const tileB = 8;
  const B = 5;

  function makeTrackingDevice(createBufferCalls: number[], resourceOps: string[] = []): GPUDevice {
    const mkBuf = (size: number): GPUBuffer => {
      const id = createBufferCalls.length;
      return {
        _bytes: new Uint8Array(size),
        _id: id,
        size,
        destroy() { resourceOps.push(`destroy:${id}`); },
        getMappedRange() { return (this as unknown as { _bytes: Uint8Array })._bytes.buffer; },
        async mapAsync() {},
        unmap() {},
      } as unknown as GPUBuffer;
    };

    return {
      createBuffer(desc: { size: number }) {
        createBufferCalls.push(desc.size);
        return mkBuf(desc.size);
      },
      createShaderModule() { return { getCompilationInfo: async () => ({ messages: [] }) }; },
      async createComputePipelineAsync() { return { getBindGroupLayout: () => ({}) }; },
      createBindGroup() { resourceOps.push('bind'); return {}; },
      createCommandEncoder() {
        return {
          clearBuffer() {},
          beginComputePass() {
            return { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
          },
          copyBufferToBuffer(s: GPUBuffer, so: number, d: GPUBuffer, doff: number, size: number) {
            const src = (s as unknown as { _bytes: Uint8Array })._bytes;
            const dst = (d as unknown as { _bytes: Uint8Array })._bytes;
            dst.set(src.subarray(so, so + size), doff);
          },
          finish() { return {}; },
        };
      },
      queue: { writeBuffer() {}, submit() {} },
      limits: { maxComputeWorkgroupStorageSize: 32768 },
    } as unknown as GPUDevice;
  }

  it('no new GPUBuffers are created by trainStep() after build() completes (f32 path)', async () => {
    const buildCalls: number[] = [];
    const resourceOps: string[] = [];
    const device = makeTrackingDevice(buildCalls, resourceOps);
    const trainer = new FusedMlpTrainer(device, spec, { useF16: false, tileB });
    await trainer.build(B);

    resourceOps.length = 0;
    const buildAllocCount = buildCalls.length;
    expect(buildAllocCount).toBeGreaterThan(0); // sanity: build does allocate

    // Record createBuffer calls during 3 consecutive trainStep() calls.
    const postBuildCalls: number[] = [];
    // Swap in a fresh tracking layer by replacing the createBuffer method.
    const origCreate = (device as unknown as { createBuffer: (d: { size: number }) => GPUBuffer }).createBuffer;
    (device as unknown as { createBuffer: (d: { size: number }) => GPUBuffer }).createBuffer = (desc) => {
      postBuildCalls.push(desc.size);
      return origCreate.call(device, desc);
    };

    trainer.trainStep(0.01);
    trainer.trainStep(0.01);
    trainer.trainStep(0.01);

    // The fix: ZERO new buffers created during any trainStep.
    expect(postBuildCalls).toHaveLength(0);
    expect(resourceOps).toEqual([]);
  });

  it('the persistent params UBO field identity is stable across trainStep() calls', async () => {
    const calls: number[] = [];
    const device = makeTrackingDevice(calls);
    const trainer = new FusedMlpTrainer(device, spec, { useF16: false, tileB });
    await trainer.build(B);

    const uboAfterBuild = trainer._paramsUbo;
    expect(uboAfterBuild).toBeDefined();

    trainer.trainStep(0.01);
    trainer.trainStep(0.01);

    // Same object reference: the buffer was not re-created during trainStep.
    expect(trainer._paramsUbo).toBe(uboAfterBuild);
  });

  it('the persistent grad-finalize UBO fields survive multiple steps', async () => {
    const calls: number[] = [];
    const device = makeTrackingDevice(calls);
    const trainer = new FusedMlpTrainer(device, spec, { useF16: false, tileB });
    await trainer.build(B);

    const uboW = trainer._gradFinUboW;
    const uboB = trainer._gradFinUboB;
    const uboX = trainer._gradFinUboX;
    expect(uboW).toBeDefined();
    expect(uboB).toBeDefined();
    expect(uboX).toBeDefined();

    trainer.trainStep(0.01);
    trainer.trainStep(0.01);

    expect(trainer._gradFinUboW).toBe(uboW);
    expect(trainer._gradFinUboB).toBe(uboB);
    expect(trainer._gradFinUboX).toBe(uboX);
  });

  it('alternates parameter/moment generations across success/failure/success', async () => {
    const calls: number[] = [];
    const device = makeTrackingDevice(calls);
    const trainer = new FusedMlpTrainer(device, spec, { useF16: false, tileB });
    await trainer.build(B);
    const diagnosticsBuffer = trainer.diagnosticsBuffer;
    type Tracked = GPUBuffer & { _bytes: Uint8Array };
    type Set = {
      weights: Tracked; biases: Tracked;
      wMasterGpu: Tracked; bMasterGpu: Tracked;
      mW: Tracked; vW: Tracked; mB: Tracked; vB: Tracked;
    };
    const state = trainer as unknown as {
      weights: Tracked; wMasterGpu: Tracked;
      _trainableSets: readonly [Set, Set];
      adamT: number;
    };
    const checksum = (set: Set): number => {
      let hash = 0x811c9dc5;
      for (const buffer of [
        set.wMasterGpu, set.bMasterGpu, set.mW, set.vW, set.mB, set.vB,
      ]) {
        for (const byte of buffer._bytes) hash = Math.imul(hash ^ byte, 0x01000193) >>> 0;
      }
      return hash;
    };
    const liveSet = (): Set => state._trainableSets.find(
      set => set.wMasterGpu === state.wMasterGpu,
    )!;
    const candidateSet = (buffer: GPUBuffer): Set => state._trainableSets.find(
      set => set.wMasterGpu === buffer,
    )!;
    const mutate = (set: Set, seed: number): void => {
      set.wMasterGpu._bytes[0] = seed;
      set.bMasterGpu._bytes[1] = seed + 1;
      set.mW._bytes[2] = seed + 2;
      set.vW._bytes[3] = seed + 3;
      set.mB._bytes[4] = seed + 4;
      set.vB._bytes[5] = seed + 5;
    };

    const generationA = liveSet();
    const baseline = checksum(generationA);

    const tx1 = trainer.recordTrainStep(device.createCommandEncoder(), 0.01)!;
    const generationB = candidateSet(tx1.candidateWeightBuffer);
    expect(checksum(generationB)).toBe(baseline);
    mutate(generationB, 19);
    tx1.commitCpu();
    tx1.finalizeSuccess();
    expect(liveSet()).toBe(generationB);
    const afterFirstSuccess = checksum(liveSet());
    expect(afterFirstSuccess).not.toBe(baseline);
    expect(state.adamT).toBe(1);

    const failed = trainer.recordTrainStep(device.createCommandEncoder(), 0.01)!;
    expect(candidateSet(failed.candidateWeightBuffer)).toBe(generationA);
    expect(checksum(generationA)).toBe(afterFirstSuccess);
    mutate(generationA, 93);
    failed.rollback();
    expect(liveSet()).toBe(generationB);
    expect(checksum(liveSet())).toBe(afterFirstSuccess);
    expect(state.adamT).toBe(1);

    const tx2 = trainer.recordTrainStep(device.createCommandEncoder(), 0.01)!;
    expect(candidateSet(tx2.candidateWeightBuffer)).toBe(generationA);
    expect(checksum(generationA)).toBe(afterFirstSuccess);
    mutate(generationA, 37);
    tx2.commitCpu();
    tx2.finalizeSuccess();
    expect(liveSet()).toBe(generationA);
    expect(checksum(liveSet())).not.toBe(afterFirstSuccess);
    expect(state.adamT).toBe(2);
    expect(trainer.diagnosticsBuffer).toBe(diagnosticsBuffer);
  });

  it('destroys both preallocated generations exactly once', async () => {
    const calls: number[] = [];
    const resourceOps: string[] = [];
    const device = makeTrackingDevice(calls, resourceOps);
    const trainer = new FusedMlpTrainer(device, spec, { useF16: false, tileB });
    await trainer.build(B);
    trainer.trainStep(0.01);
    trainer.trainStep(0.01);

    resourceOps.length = 0;
    trainer.dispose();
    const destroyed = resourceOps.filter(op => op.startsWith('destroy:'));
    expect(new Set(destroyed).size).toBe(destroyed.length);
    expect(destroyed).toHaveLength(calls.length);

    trainer.dispose();
    expect(resourceOps.filter(op => op.startsWith('destroy:'))).toHaveLength(calls.length);
  });

  it('dispose() after build() destroys all persistent UBOs', async () => {
    const calls: number[] = [];
    const device = makeTrackingDevice(calls);
    const trainer = new FusedMlpTrainer(device, spec, { useF16: false, tileB });
    await trainer.build(B);

    // Spy on the persistent UBO objects to confirm destroy() is called.
    const ubos = [
      trainer._paramsUbo,
      trainer._gradFinUboW, trainer._gradFinUboB, trainer._gradFinUboX,
      trainer._adamUboW, trainer._adamUboB,
    ].filter((u): u is GPUBuffer => u !== undefined); // D7.5: fields are now honestly `GPUBuffer | undefined`
    const destroyCounts = new Map(ubos.map(u => [u, 0]));
    for (const ubo of ubos) {
      const orig = ubo.destroy.bind(ubo);
      ubo.destroy = () => { destroyCounts.set(ubo, (destroyCounts.get(ubo) ?? 0) + 1); orig(); };
    }

    trainer.dispose();

    for (const [, count] of destroyCounts) {
      expect(count).toBe(1);
    }
    // Idempotent: second dispose is a no-op.
    trainer.dispose();
    for (const [, count] of destroyCounts) {
      expect(count).toBe(1);
    }
  });
});
