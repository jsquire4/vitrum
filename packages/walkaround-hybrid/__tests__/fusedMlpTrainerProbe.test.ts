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
