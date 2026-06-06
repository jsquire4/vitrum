/**
 * Characterization test (Task 4.1 — WalkaroundGPUPipeline god-orchestrator
 * extraction). Pins the EXACT registered pass set + the topologically-sorted
 * dispatch order that `initialize()`'s pass-registration block produces, so the
 * internal `registerPasses(...)` free-function extraction is provably
 * behavior-identical (same registry contents, same PASS_ORDER, byte-for-byte).
 *
 * This builds the registry the SAME way `initialize()` does — instantiating the
 * concrete Pass classes with stub pipelines/refs and calling `register()` in the
 * source order — then snapshots:
 *   • the registered ids (registration order, before topo sort)
 *   • the topologically-sorted pass-id order (the PASS_ORDER the registry emits)
 *   • the flattened passLabels sequence in sorted order (the timestamp layout)
 *
 * The goldens below were captured from the CURRENT code BEFORE the refactor.
 * If the extraction changes the registered set or order, these assertions fail.
 *
 * No GPUDevice is needed: Pass-class construction + registry topo-sort are pure.
 */

import { describe, it, expect } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
import {
  AtrousIndirectPass,
  CompositePass,
  DenoiserAdapterPass,
  GTAOPass,
  GTAOUpsamplePass,
  IndirectCombinePass,
  IndirectTemporalAccumPass,
  MotionVectorsPass,
  PPGUpdatePass,
  ReGIRBuildPass,
  ResolvePass,
  RISGIPass,
  RISPass,
  SampleBudgetPass,
  ShadePass,
  SpatialGIReservoirPass,
  SpatialReservoirPass,
  TemporalAccumPass,
  TemporalGIReservoirPass,
  TemporalReservoirPass,
} from '../src/pipeline/passes/index.js';
import { PassRegistry } from '../src/pipeline/PassRegistry.js';
import {
  DENOISER_READY_STATE,
  type Denoiser,
} from '../src/pipeline/denoisers/index.js';

const stubPipeline = {} as GPUComputePipeline;
const stubRenderPipeline = {} as GPURenderPipeline;
const stubUboRef = { buf: undefined };
const stubPingPong = { value: 0 };

const makeStubDenoiser = (id: string, labels: readonly string[]): Denoiser =>
  ({
    id: id as Denoiser['id'],
    passLabels: labels as Denoiser['passLabels'],
    state: () => DENOISER_READY_STATE,
    initialize: async () => {},
    dispatch: () => null,
    resize: () => {},
    dispose: () => {},
  } as unknown as Denoiser);

/**
 * Reproduce the registration sequence of `WalkaroundGPUPipeline.initialize()`
 * (default options: PPG compiled+on, ReGIR compiled+on, both spatial counts 2,
 * NRC off, restirPtReuse off) so the golden captures the full pass set the
 * orchestrator wires. The ReGIRBuildPass getter is a no-op closure here.
 */
function buildPipelineRegistry(): PassRegistry {
  const reg = new PassRegistry();
  reg.register(new SampleBudgetPass(stubPipeline, stubUboRef, stubUboRef));
  reg.register(new RISPass(stubPipeline));
  reg.register(new TemporalReservoirPass(stubPipeline));
  reg.register(new SpatialReservoirPass(stubPipeline, 2));
  reg.register(new RISGIPass(stubPipeline, undefined));
  reg.register(new TemporalGIReservoirPass(stubPipeline, false));
  reg.register(new SpatialGIReservoirPass(stubPipeline, 2, false));
  reg.register(new ShadePass(stubPipeline));
  reg.register(new MotionVectorsPass(stubPipeline));
  reg.register(new GTAOPass(stubPipeline));
  reg.register(new GTAOUpsamplePass(stubPipeline));
  reg.register(new DenoiserAdapterPass(
    () => makeStubDenoiser('atrous-variance', ['welford-temporal', 'atrous-variance-variance']),
    () => stubPipeline,
    () => true,
  ));
  reg.register(new IndirectTemporalAccumPass(stubPipeline, stubPingPong));
  reg.register(new AtrousIndirectPass(stubPipeline, stubUboRef));
  reg.register(new IndirectCombinePass(stubPipeline));
  reg.register(new TemporalAccumPass(stubPipeline, stubUboRef));
  reg.register(new ResolvePass(stubPipeline, stubUboRef));
  reg.register(new CompositePass(stubRenderPipeline));
  reg.register(new PPGUpdatePass(stubPipeline));
  reg.register(new ReGIRBuildPass(
    stubPipeline,
    // ReGIRCoordinator + BGLCache + getter — only metadata matters for ordering.
    {} as never,
    {} as never,
    () => ({}) as never,
  ));
  return reg;
}

// ── GOLDENS — captured from current code (pre-refactor) ─────────────────────

const GOLDEN_REGISTERED_IDS = [
  'sample-budget',
  'ris',
  'temporal',
  'spatial-2',
  'gi-ris',
  'gi-temporal',
  'gi-spatial-2',
  'shade',
  'motion-vectors',
  'gtao',
  'gtao-upsample',
  'denoiser-adapter',
  'indirect-temporal-accum',
  'atrous-indirect-3',
  'indirect-combine',
  'temporalAccum',
  'resolve',
  'composite',
  'ppg-update',
  'regir-build',
];

type PipelineUboRefShape = {
  _atrousIndirectUboRef: object;
  _accumUboRef: object;
  _sampleBudgetUboRef: object;
  _sampleCountUboRef: object;
  _resolveUboRef: object;
  readonly _perPassUboRefs: readonly object[];
};

describe('WalkaroundGPUPipeline — pass-registration characterization', () => {
  it('registers exactly the expected pass set in source order', () => {
    const reg = buildPipelineRegistry();
    expect(reg.ids()).toEqual(GOLDEN_REGISTERED_IDS);
    expect(reg.size()).toBe(GOLDEN_REGISTERED_IDS.length);
  });

  it('emits a stable topologically-sorted PASS_ORDER', () => {
    const reg = buildPipelineRegistry();
    const order = reg.sortedPasses().map((p) => p.id);

    // Pin the full sorted order as a golden — any reordering breaks this.
    expect(order).toEqual([
      'regir-build',
      'sample-budget',
      'ris',
      'temporal',
      'spatial-2',
      'gi-ris',
      'gi-temporal',
      'gi-spatial-2',
      'shade',
      'gtao',
      'gtao-upsample',
      'denoiser-adapter',
      'indirect-temporal-accum',
      'atrous-indirect-3',
      'indirect-combine',
      'motion-vectors',
      'ppg-update',
      'temporalAccum',
      'resolve',
      'composite',
    ]);
  });

  it('is deterministic across repeated sorts', () => {
    const reg = buildPipelineRegistry();
    const a = reg.sortedPasses().map((p) => p.id);
    const b = reg.sortedPasses().map((p) => p.id);
    expect(a).toEqual(b);
  });

  it('flattened passLabels sequence (timestamp layout) is byte-stable', () => {
    const reg = buildPipelineRegistry();
    const labels = reg.sortedPasses().flatMap((p) => p.passLabels);
    expect(labels).toEqual([
      'regir-build',
      'sample-budget',
      'ris',
      'temporal',
      'spatial-1',
      'spatial-2',
      'gi-ris',
      'gi-temporal',
      'gi-spatial-1',
      'gi-spatial-2',
      'shade',
      'gtao',
      'gtao-upsample',
      'welford-temporal',
      'atrous-variance-variance',
      'indirect-temporal-accum',
      'atrous-indirect-0',
      'atrous-indirect-1',
      'atrous-indirect-2',
      'atrous-indirect-3',
      'indirect-combine',
      'motion-vectors',
      'ppg-update',
      'temporalAccum',
      'resolve',
      'composite',
    ]);
  });

  it('keeps the indirect atrous UBO out of the eager pipeline-owned dispose list', async () => {
    installWebGPUPolyfills();
    const { WalkaroundGPUPipeline } = await import('../src/pipeline/WalkaroundGPUPipeline.js');
    const pipeline = Object.create(WalkaroundGPUPipeline.prototype) as PipelineUboRefShape;
    const atrousIndirect = { buf: 'atrous' };
    const accum = { buf: 'accum' };
    const sampleBudget = { buf: 'sample-budget' };
    const sampleCount = { buf: 'sample-count' };
    const resolve = { buf: 'resolve' };
    pipeline._atrousIndirectUboRef = atrousIndirect;
    pipeline._accumUboRef = accum;
    pipeline._sampleBudgetUboRef = sampleBudget;
    pipeline._sampleCountUboRef = sampleCount;
    pipeline._resolveUboRef = resolve;

    expect(pipeline._perPassUboRefs).toEqual([accum, sampleBudget, sampleCount, resolve]);
    expect(pipeline._perPassUboRefs).not.toContain(atrousIndirect);
  });
});
