/**
 * Pass-class shape tests — verify each W1-R5 Pass entry declares stable
 * metadata (id, dependencies, passLabels, gates) without requiring a GPU
 * device. These exercise the static identity of each Pass; GPU dispatch
 * is exercised by the live integration tests.
 */

import { describe, it, expect } from 'vitest';
import {
  AtrousIndirectPass,
  CompositePass,
  DenoiserAdapterPass,
  GTAOPass,
  GTAOUpsamplePass,
  IndirectCombinePass,
  IndirectTemporalAccumPass,
  PPGUpdatePass,
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
import type { PassGateOptions } from '../src/pipeline/Pass.js';
import {
  DENOISER_READY_STATE,
  type Denoiser,
} from '../src/pipeline/denoisers/index.js';

// Stub GPU resources — we never call dispatch in these tests, so the
// pipelines/refs only need to be present, not real.
const stubPipeline = {} as GPUComputePipeline;
const stubRenderPipeline = {} as GPURenderPipeline;
const stubUboRef = { buf: undefined };
const stubPingPong = { value: 0 };
// Minimal Denoiser stub for DenoiserAdapterPass tests — only `id` and
// `passLabels` are inspected by the shape tests; the dispatch path is
// covered by integration tests with a real GPU device.
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

const DEFAULT_GATE: PassGateOptions = {
  denoiserMode: 'atrous-variance',
  ppgEnabled: false,
};

describe('Pass entries — W1-R5 shape invariants', () => {
  it('SampleBudgetPass: id, no deps, single label, always-on', () => {
    const p = new SampleBudgetPass(stubPipeline, stubUboRef, stubUboRef);
    expect(p.id).toBe('sample-budget');
    expect(p.dependencies).toEqual([]);
    expect(p.passLabels).toEqual(['sample-budget']);
    // SampleBudgetPass.gates() ignores opts (always-on); call without args.
    expect(p.gates()).toBe(true);
  });

  it('RISPass: depends on sample-budget', () => {
    const p = new RISPass(stubPipeline);
    expect(p.id).toBe('ris');
    expect(p.dependencies).toEqual(['sample-budget']);
    expect(p.passLabels).toEqual(['ris']);
    // RISPass.gates() ignores opts (always-on); call without args.
    expect(p.gates()).toBe(true);
  });

  it('TemporalReservoirPass: depends on ris', () => {
    const p = new TemporalReservoirPass(stubPipeline);
    expect(p.id).toBe('temporal');
    expect(p.dependencies).toEqual(['ris']);
  });

  it('SpatialReservoirPass: owns spatial-1 + spatial-2', () => {
    const p = new SpatialReservoirPass(stubPipeline);
    expect(p.id).toBe('spatial-2');
    expect(p.dependencies).toEqual(['temporal']);
    expect(p.passLabels).toEqual(['spatial-1', 'spatial-2']);
  });

  it('RISGIPass: depends on spatial-2', () => {
    const p = new RISGIPass(stubPipeline);
    expect(p.id).toBe('gi-ris');
    expect(p.dependencies).toEqual(['spatial-2']);
  });

  it('TemporalGIReservoirPass: depends on gi-ris', () => {
    const p = new TemporalGIReservoirPass(stubPipeline);
    expect(p.id).toBe('gi-temporal');
    expect(p.dependencies).toEqual(['gi-ris']);
  });

  it('SpatialGIReservoirPass: owns gi-spatial-1 + gi-spatial-2', () => {
    const p = new SpatialGIReservoirPass(stubPipeline);
    expect(p.id).toBe('gi-spatial-2');
    expect(p.dependencies).toEqual(['gi-temporal']);
    expect(p.passLabels).toEqual(['gi-spatial-1', 'gi-spatial-2']);
  });

  it('ShadePass: depends on both spatial-2 AND gi-spatial-2', () => {
    const p = new ShadePass(stubPipeline);
    expect(p.id).toBe('shade');
    expect(p.dependencies).toEqual(['spatial-2', 'gi-spatial-2']);
  });

  it('GTAOPass: depends on shade', () => {
    const p = new GTAOPass(stubPipeline);
    expect(p.id).toBe('gtao');
    expect(p.dependencies).toEqual(['shade']);
  });

  it('GTAOUpsamplePass: depends on gtao', () => {
    const p = new GTAOUpsamplePass(stubPipeline);
    expect(p.id).toBe('gtao-upsample');
    expect(p.dependencies).toEqual(['gtao']);
  });

  it('DenoiserAdapterPass: depends on gtao-upsample; passLabels forward from active denoiser', () => {
    const stubDenoiser = makeStubDenoiser('atrous-variance', ['welford-temporal', 'atrous-variance-variance']);
    const p = new DenoiserAdapterPass(() => stubDenoiser, () => stubPipeline);
    expect(p.id).toBe('denoiser-adapter');
    expect(p.dependencies).toEqual(['gtao-upsample']);
    expect(p.passLabels).toEqual(['welford-temporal', 'atrous-variance-variance']);
  });

  it('DenoiserAdapterPass: gates() returns false for the NoneDenoiser', () => {
    const noneStub = makeStubDenoiser('none', []);
    const p = new DenoiserAdapterPass(() => noneStub, () => stubPipeline);
    expect(p.gates()).toBe(false);
  });

  it('DenoiserAdapterPass: gates() returns true for non-pass-through denoisers', () => {
    const realStub = makeStubDenoiser('atrous-variance', []);
    const p = new DenoiserAdapterPass(() => realStub, () => stubPipeline);
    expect(p.gates()).toBe(true);
  });

  it('DenoiserAdapterPass: gates() returns false when runtime bypass is off', () => {
    const realStub = makeStubDenoiser('atrous-variance', []);
    const p = new DenoiserAdapterPass(() => realStub, () => stubPipeline, () => false);
    expect(p.gates()).toBe(false);
  });

  it('IndirectTemporalAccumPass: depends on shade, gtao-upsample, denoiser-adapter', () => {
    const p = new IndirectTemporalAccumPass(stubPipeline, stubPingPong);
    expect(p.id).toBe('indirect-temporal-accum');
    expect(p.dependencies).toEqual(['shade', 'gtao-upsample', 'denoiser-adapter']);
  });

  it('AtrousIndirectPass: 4 labels, id = atrous-indirect-3', () => {
    const p = new AtrousIndirectPass(stubPipeline, stubUboRef);
    expect(p.id).toBe('atrous-indirect-3');
    expect(p.dependencies).toEqual(['indirect-temporal-accum']);
    expect(p.passLabels).toEqual([
      'atrous-indirect-0',
      'atrous-indirect-1',
      'atrous-indirect-2',
      'atrous-indirect-3',
    ]);
  });

  it('IndirectCombinePass: depends on atrous-indirect-3', () => {
    const p = new IndirectCombinePass(stubPipeline);
    expect(p.id).toBe('indirect-combine');
    expect(p.dependencies).toEqual(['atrous-indirect-3']);
  });

  it('TemporalAccumPass: depends on indirect-combine', () => {
    const p = new TemporalAccumPass(stubPipeline, stubUboRef);
    expect(p.id).toBe('temporalAccum');
    expect(p.dependencies).toEqual(['indirect-combine']);
  });

  it('ResolvePass: depends on temporalAccum', () => {
    const p = new ResolvePass(stubPipeline, stubUboRef, false);
    expect(p.id).toBe('resolve');
    expect(p.dependencies).toEqual(['temporalAccum']);
  });

  it('CompositePass: depends on resolve; render pipeline exposed', () => {
    const p = new CompositePass(stubRenderPipeline, stubUboRef);
    expect(p.id).toBe('composite');
    expect(p.dependencies).toEqual(['resolve']);
    expect(p.pipeline).toBe(stubRenderPipeline);
  });

  it('PPGUpdatePass: gated on ppgEnabled', () => {
    const p = new PPGUpdatePass(stubPipeline);
    expect(p.id).toBe('ppg-update');
    expect(p.dependencies).toEqual(['shade']);
    expect(p.gates({ ...DEFAULT_GATE, ppgEnabled: false })).toBe(false);
    expect(p.gates({ ...DEFAULT_GATE, ppgEnabled: true })).toBe(true);
  });
});

describe('Pass entries — topological registration', () => {
  it('all 18 passes register + sort with no cycles', () => {
    const reg = new PassRegistry();
    reg.register(new SampleBudgetPass(stubPipeline, stubUboRef, stubUboRef));
    reg.register(new RISPass(stubPipeline));
    reg.register(new TemporalReservoirPass(stubPipeline));
    reg.register(new SpatialReservoirPass(stubPipeline));
    reg.register(new RISGIPass(stubPipeline));
    reg.register(new TemporalGIReservoirPass(stubPipeline));
    reg.register(new SpatialGIReservoirPass(stubPipeline));
    reg.register(new ShadePass(stubPipeline));
    reg.register(new GTAOPass(stubPipeline));
    reg.register(new GTAOUpsamplePass(stubPipeline));
    reg.register(new DenoiserAdapterPass(() => makeStubDenoiser('atrous-variance', []), () => stubPipeline));
    reg.register(new IndirectTemporalAccumPass(stubPipeline, stubPingPong));
    reg.register(new AtrousIndirectPass(stubPipeline, stubUboRef));
    reg.register(new IndirectCombinePass(stubPipeline));
    reg.register(new TemporalAccumPass(stubPipeline, stubUboRef));
    reg.register(new ResolvePass(stubPipeline, stubUboRef, false));
    reg.register(new CompositePass(stubRenderPipeline, stubUboRef));
    reg.register(new PPGUpdatePass(stubPipeline));
    expect(reg.size()).toBe(18);
    const order = reg.sortedPasses().map((p) => p.id);
    // Spot-check the topo: sample-budget first, composite last.
    expect(order[0]).toBe('sample-budget');
    expect(order[order.length - 1]).toBe('composite');
    // shade comes after both spatial-2 and gi-spatial-2.
    const idx = (id: string) => order.indexOf(id);
    expect(idx('shade')).toBeGreaterThan(idx('spatial-2'));
    expect(idx('shade')).toBeGreaterThan(idx('gi-spatial-2'));
    // denoiser-adapter sits between gtao-upsample and indirect-temporal-accum.
    expect(idx('denoiser-adapter')).toBeGreaterThan(idx('gtao-upsample'));
    expect(idx('denoiser-adapter')).toBeLessThan(idx('indirect-temporal-accum'));
    // indirect-combine comes after atrous-indirect-3.
    expect(idx('indirect-combine')).toBeGreaterThan(idx('atrous-indirect-3'));
    // temporalAccum after indirect-combine.
    expect(idx('temporalAccum')).toBeGreaterThan(idx('indirect-combine'));
    // resolve after temporalAccum.
    expect(idx('resolve')).toBeGreaterThan(idx('temporalAccum'));
    // composite after resolve.
    expect(idx('composite')).toBeGreaterThan(idx('resolve'));
  });

  it('activePasses with ppgEnabled=false excludes ppg passes', () => {
    const reg = new PassRegistry();
    reg.register(new ShadePass(stubPipeline));
    reg.register(new SpatialReservoirPass(stubPipeline));
    reg.register(new TemporalReservoirPass(stubPipeline));
    reg.register(new RISPass(stubPipeline));
    reg.register(new SampleBudgetPass(stubPipeline, stubUboRef, stubUboRef));
    reg.register(new RISGIPass(stubPipeline));
    reg.register(new TemporalGIReservoirPass(stubPipeline));
    reg.register(new SpatialGIReservoirPass(stubPipeline));
    reg.register(new PPGUpdatePass(stubPipeline));
    const active = reg.activePasses({ denoiserMode: 'atrous-variance', ppgEnabled: false }).map((p) => p.id);
    expect(active).not.toContain('ppg-update');
    expect(active).toContain('shade');
  });
});
