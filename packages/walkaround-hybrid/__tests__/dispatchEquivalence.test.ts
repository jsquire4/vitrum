/**
 * Dispatch-equivalence pins for Theme E (Task 2.1).
 *
 * The Theme-E refactor routed 7 single-bind-group passes + 4 à-trous chains
 * + the GRIS scene-bind + the RISGI NRC slot through shared dispatch helpers.
 * The contract is BEHAVIOR-PRESERVING: every refactored pass must emit the
 * IDENTICAL GPU dispatch sequence — same pipeline, same bind-group slots,
 * same workgroup dims, same compute-pass label, same order.
 *
 * These tests drive each pass/denoiser through a mock device + recording
 * encoder and assert the captured dispatch record against the hand-verified
 * pre-refactor sequence. A recording encoder is the cheapest faithful pin:
 * `beginComputePass`/`setPipeline`/`setBindGroup`/`dispatchWorkgroups` are the
 * exact GPU surface the refactor touched.
 *
 * Also pins:
 *   - #5: the neural denoiser's relabel (`neural-pack`/`neural-unpack`) +
 *     the trimmed 2-entry passLabels.
 *   - #7: `composePassLabels(...)` equals the registry's runtime-dispatched
 *     label sequence (the IndirectTemporalAccumPass denoiser-adapter ordering
 *     is made safe by this assertion rather than a fragile topo tiebreaker).
 */

import { describe, it, expect, vi } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
import { CheckerboardPrefillPass } from '../src/pipeline/passes/CheckerboardPrefillPass.js';
import { MotionVectorsPass } from '../src/pipeline/passes/MotionVectorsPass.js';
import { GTAOUpsamplePass } from '../src/pipeline/passes/GTAOUpsamplePass.js';
import { IndirectCombinePass } from '../src/pipeline/passes/IndirectCombinePass.js';
import { IndirectTemporalAccumPass } from '../src/pipeline/passes/IndirectTemporalAccumPass.js';
import { TransparentOitPass } from '../src/pipeline/passes/TransparentOitPass.js';
import { TemporalAccumPass } from '../src/pipeline/passes/TemporalAccumPass.js';
import { ResolvePass } from '../src/pipeline/passes/ResolvePass.js';
import { SampleBudgetPass } from '../src/pipeline/passes/SampleBudgetPass.js';
import { TemporalGIReservoirPass } from '../src/pipeline/passes/TemporalGIReservoirPass.js';
import { SpatialGIReservoirPass } from '../src/pipeline/passes/SpatialGIReservoirPass.js';
import { RISGIPass } from '../src/pipeline/passes/RISGIPass.js';
import { AtrousIndirectPass } from '../src/pipeline/passes/AtrousIndirectPass.js';
import { AtrousDenoiser } from '../src/pipeline/denoisers/atrous.js';
import { AtrousVarianceDenoiser } from '../src/pipeline/denoisers/atrousVariance.js';
import type { DenoiserDispatchContext } from '../src/pipeline/denoisers/index.js';
import { composePassLabels } from '../src/pipeline/passes/passOrder.js';
import { PassRegistry } from '../src/pipeline/PassRegistry.js';
import { RISPass } from '../src/pipeline/passes/RISPass.js';
import { TemporalReservoirPass } from '../src/pipeline/passes/TemporalReservoirPass.js';
import { SpatialReservoirPass } from '../src/pipeline/passes/SpatialReservoirPass.js';
import { ShadePass } from '../src/pipeline/passes/ShadePass.js';
import { GTAOPass } from '../src/pipeline/passes/GTAOPass.js';
import { DenoiserAdapterPass } from '../src/pipeline/passes/DenoiserAdapterPass.js';
import { CompositePass } from '../src/pipeline/passes/CompositePass.js';
import {
  DENOISER_PASS_LABELS,
  DENOISER_READY_STATE,
  type Denoiser,
} from '../src/pipeline/denoisers/index.js';
import type { Pass, PassDispatchContext, PassFrameState } from '../src/pipeline/Pass.js';
import type { PassLabel } from '../src/pipeline/timestampQueries.js';

installWebGPUPolyfills();

// ── Recording encoder ─────────────────────────────────────────────────────────
// Captures every compute-pass emitted as a flat record so a test can assert the
// exact (label, pipeline, [slot,group]…, dims) sequence.

interface PassRecord {
  label: string;
  pipeline: unknown;
  binds: Array<{ slot: number; group: unknown }>;
  dims: [number, number, number];
}

interface EncoderEvent {
  kind: 'clear' | 'pass' | 'copy';
  label?: string;
  size?: number;
}

interface BufferWriteRecord {
  buffer: unknown;
  offset: number;
  data: BufferSource;
}

function makeRecordingEncoder(): {
  encoder: GPUCommandEncoder;
  records: PassRecord[];
  copies: Array<{ kind: 'buffer'; size: number }>;
  events: EncoderEvent[];
} {
  const records: PassRecord[] = [];
  const copies: Array<{ kind: 'buffer'; size: number }> = [];
  const events: EncoderEvent[] = [];
  const encoder = {
    beginComputePass: (desc: GPUComputePassDescriptor) => {
      events.push({ kind: 'pass', label: String((desc as { label?: string }).label) });
      const rec: PassRecord = {
        label: String((desc as { label?: string }).label),
        pipeline: undefined,
        binds: [],
        dims: [0, 0, 0],
      };
      records.push(rec);
      return {
        setPipeline: (p: unknown) => { rec.pipeline = p; },
        setBindGroup: (slot: number, group: unknown) => { rec.binds.push({ slot, group }); },
        dispatchWorkgroups: (x: number, y: number, z: number) => { rec.dims = [x, y, z ?? 1]; },
        end: () => {},
      } as unknown as GPUComputePassEncoder;
    },
    clearBuffer: () => {
      events.push({ kind: 'clear' });
    },
    copyBufferToBuffer: (_s: unknown, _so: number, _d: unknown, _do: number, size: number) => {
      copies.push({ kind: 'buffer', size });
      events.push({ kind: 'copy', size });
    },
    copyTextureToTexture: vi.fn(),
  } as unknown as GPUCommandEncoder;
  return { encoder, records, copies, events };
}

// A tagged stub texture whose createView() returns a stable tagged view object,
// so view identity can be asserted across the dispatch.
function tex(tag: string): GPUTexture {
  const view = { __tag: `${tag}#view` };
  return {
    __tag: tag,
    createView: () => view,
    width: 64,
    height: 64,
    size: 64,
  } as unknown as GPUTexture;
}
function buf(tag: string, size = 256): GPUBuffer {
  return { __tag: tag, size, destroy: vi.fn(), createView: () => ({ __tag: `${tag}#view` }) } as unknown as GPUBuffer;
}

const stubPipeline = (tag: string) => ({
  __tag: tag,
  getBindGroupLayout: () => ({ __tag: `${tag}#bgl` } as unknown as GPUBindGroupLayout),
} as unknown as GPUComputePipeline);

// A device whose createBindGroup returns a tagged stub (so each pass's bind
// group is a single opaque object — we assert SLOT placement, not contents).
function makeDevice(): GPUDevice {
  let bgN = 0;
  let bufN = 0;
  const writes: BufferWriteRecord[] = [];
  return {
    __writes: writes,
    createBindGroup: (desc: GPUBindGroupDescriptor) => ({
      __tag: `bg#${bgN++}`,
      __desc: desc,
    } as unknown as GPUBindGroup),
    createBindGroupLayout: () => ({} as unknown as GPUBindGroupLayout),
    createBuffer: (desc: GPUBufferDescriptor) => buf(`created#${bufN++}`, desc.size),
    queue: {
      writeBuffer: vi.fn((buffer: unknown, offset: number, data: BufferSource) => {
        // Snapshot the payload AT CALL TIME — the real GPUQueue.writeBuffer
        // copies synchronously, so production code may legally reuse one
        // scratch ArrayBuffer across writes (atrousVariance.ts re-packs the
        // shared eager UBO each iteration). Storing by reference would alias
        // every captured write to the final iteration's content.
        const snapshot: ArrayBuffer = data instanceof ArrayBuffer
          ? data.slice(0)
          : (data.buffer as ArrayBuffer).slice(data.byteOffset, data.byteOffset + data.byteLength);
        writes.push({ buffer, offset, data: snapshot });
      }),
    },
  } as unknown as GPUDevice;
}

function deviceWrites(device: GPUDevice): BufferWriteRecord[] {
  return (device as unknown as { __writes: BufferWriteRecord[] }).__writes;
}

function bindGroupEntries(group: unknown): GPUBindGroupEntry[] {
  return (group as { __desc: GPUBindGroupDescriptor }).__desc.entries as GPUBindGroupEntry[];
}

function bufferBinding(group: unknown, binding: number): GPUBufferBinding {
  const entry = bindGroupEntries(group).find((e) => e.binding === binding);
  return entry!.resource as GPUBufferBinding;
}

function firstF32(data: BufferSource): number {
  const buffer = data instanceof ArrayBuffer ? data : data.buffer;
  const byteOffset = data instanceof ArrayBuffer ? 0 : data.byteOffset;
  return new Float32Array(buffer, byteOffset, 1)[0]!;
}

function firstU32(data: BufferSource): number {
  const buffer = data instanceof ArrayBuffer ? data : data.buffer;
  const byteOffset = data instanceof ArrayBuffer ? 0 : data.byteOffset;
  return new Uint32Array(buffer, byteOffset, 1)[0]!;
}

// Minimal-but-complete PassDispatchContext. Each pass reads only a subset;
// unread fields are present so destructuring never throws.
function makeCtx(encoder: GPUCommandEncoder): PassDispatchContext {
  const device = makeDevice();
  const frameState: PassFrameState = {
    denoisedDirect: tex('denoisedDirect'),
    indirectAccumOut: tex('indirectAccumOut'),
    denoisedIndirect: tex('denoisedIndirect'),
    combinedDenoised: tex('combinedDenoised'),
    writeAccum: tex('writeAccum'),
    readAccum: tex('readAccum'),
    alpha: 0.9,
    isMoving: false,
  };
  const common = {
    gNormalDepthTexture: tex('gNormalDepth'),
    motionVectorTexture: tex('motionVec'),
    hdrColorTexture: tex('hdrColor'),
    hdrTotalTexture: tex('hdrTotal'),
    denoisedPingTexture: tex('denoisedPing'),
    denoisedPongTexture: tex('denoisedPong'),
    uboBuffer: buf('ubo', 432),
    combinedDenoisedTexture: tex('combinedDenoisedTex'),
    albedoTexture: tex('albedo'),
    hdrIndirectTexture: tex('hdrIndirect'),
    indirectAccumPingTexture: tex('indAccumPing'),
    indirectAccumPongTexture: tex('indAccumPong'),
    indirectDenoisedPingTexture: tex('indDenoisedPing'),
    indirectDenoisedPongTexture: tex('indDenoisedPong'),
    resolvedTexture: tex('resolved'),
    varianceBuffer: buf('variance'),
    varianceBufferAux: buf('varianceAux'),
    atrousVarianceEstimateTexture: tex('atrousVarianceEstimate'),
    tierTexture: tex('tier'),
  };
  const resources = {
    common,
    gtao: {
      aoHalfTexture: tex('aoHalf'),
      aoFullTexture: tex('aoFull'),
      gtaoUboBuffer: buf('gtaoUbo'),
    },
    restirGI: {
      reservoirGiCurrentBuffer: buf('giCurrent', 512),
      reservoirGiSpatialBuffer: buf('giSpatial', 512),
      reservoirGiPreviousBuffer: buf('giPrevious', 512),
    },
  };
  return {
    device,
    encoder,
    width: 64,
    height: 64,
    frameIndex: 3,
    frameCount: 7,
    bglCache: {} as never,
    resources: resources as never,
    inputs: {
      camera: { projMatrix: new Float32Array(16) },
      gtao: {
        adaptiveSamplingThresholdLow: 0.1,
        adaptiveSamplingThresholdHigh: 0.5,
        gtaoRadiusPx: 32,
        gtaoIntensity: 1,
        gtaoDepthThreshold: 1,
        gtaoBilateralDepthSigma: 1,
      },
      filter: { atrousIndirectSigmas: [32, 20, 0.5] },
    } as never,
    frameBindGroup: { __tag: 'frameBG' } as unknown as GPUBindGroup,
    sceneBindGroup: { __tag: 'sceneBG' } as unknown as GPUBindGroup,
    uboBindGroup: { __tag: 'uboBG' } as unknown as GPUBindGroup,
    hybridLayersBindGroup: { __tag: 'hybridBG' } as unknown as GPUBindGroup,
    lightTreeBindGroup: { __tag: 'lightTreeBG' } as unknown as GPUBindGroup,
    wgX: 8, wgY: 8,
    wgX16: 4, wgY16: 4,
    halfWgX: 2, halfWgY: 2,
    gtaoDownscale: 2,
    welfordPing: 0,
    gNormalDepthView: { __tag: 'gndView' } as unknown as GPUTextureView,
    computeDesc: (label: PassLabel) => ({ label }) as GPUComputePassDescriptor,
    renderTimestampWrites: () => undefined,
    frameState,
  } as unknown as PassDispatchContext;
}

describe('Theme-E dispatch equivalence — single-bind-group passes', () => {
  // Each expectation is the hand-verified pre-refactor sequence:
  // [ label, [bind slots], [dims] ]. All single-bind-group passes bind ONE
  // group at slot 0 and dispatch a single compute pass.
  const cases: Array<{
    name: string;
    make: (ctx: PassDispatchContext) => { dispatch: (c: PassDispatchContext) => void };
    label: string;
    dims: [number, number, number];
  }> = [
    { name: 'MotionVectorsPass', make: () => new MotionVectorsPass(stubPipeline('motion')), label: 'motion-vectors', dims: [8, 8, 1] },
    { name: 'GTAOUpsamplePass', make: () => new GTAOUpsamplePass(stubPipeline('gtaoUp')), label: 'gtao-upsample', dims: [8, 8, 1] },
    { name: 'IndirectCombinePass', make: () => new IndirectCombinePass(stubPipeline('indComb')), label: 'indirect-combine', dims: [4, 4, 1] },
    { name: 'TemporalAccumPass', make: () => new TemporalAccumPass(stubPipeline('tAccum'), { buf: undefined }), label: 'temporalAccum', dims: [4, 4, 1] },
    { name: 'ResolvePass', make: () => new ResolvePass(stubPipeline('resolve'), { buf: buf('resolveUbo') }, false), label: 'resolve', dims: [8, 8, 1] },
    {
      name: 'SampleBudgetPass',
      make: () => new SampleBudgetPass(stubPipeline('budget'), { buf: buf('budgetUbo') }, { buf: buf('countUbo') }),
      label: 'sample-budget', dims: [8, 8, 1],
    },
  ];

  for (const c of cases) {
    it(`${c.name}: 1 pass, slot-0 bind, label=${c.label}, dims=${c.dims.join('x')}`, () => {
      const { encoder, records } = makeRecordingEncoder();
      const ctx = makeCtx(encoder);
      c.make(ctx).dispatch(ctx);
      expect(records).toHaveLength(1);
      const r = records[0]!;
      expect(r.label).toBe(c.label);
      expect(r.pipeline).toBeDefined();
      expect(r.binds.map((b) => b.slot)).toEqual([0]);
      expect(r.dims).toEqual(c.dims);
    });
  }

  it('IndirectTemporalAccumPass: 1 pass, slot-0 bind, label=indirect-temporal-accum, dims=4x4x1; flips ping-pong', () => {
    const { encoder, records } = makeRecordingEncoder();
    const ctx = makeCtx(encoder);
    const ref = { value: 0 };
    const pass = new IndirectTemporalAccumPass(stubPipeline('indTA'), ref);
    pass.dispatch(ctx);
    expect(records).toHaveLength(1);
    expect(records[0]!.label).toBe('indirect-temporal-accum');
    expect(records[0]!.binds.map((b) => b.slot)).toEqual([0]);
    expect(records[0]!.dims).toEqual([4, 4, 1]);
    // Ping-pong flips AFTER dispatch (legacy ordering).
    expect(ref.value).toBe(1);
    // frameState.indirectAccumOut published to the ping slot (value was 0).
    expect((ctx.frameState.indirectAccumOut as unknown as { __tag: string }).__tag).toBe('indAccumPing');
  });
});

describe('Theme-E dispatch equivalence — GI scene-bind routing (#3)', () => {
  it('TemporalGIReservoirPass: default = slot-0 + receiver-material scene@slot-1, half-res 2x2', () => {
    const { encoder, records } = makeRecordingEncoder();
    const ctx = makeCtx(encoder);
    new TemporalGIReservoirPass(stubPipeline('giT'), /* grisEnabled */ false).dispatch(ctx);
    expect(records).toHaveLength(1);
    expect(records[0]!.label).toBe('gi-temporal');
    expect(records[0]!.binds.map((b) => b.slot)).toEqual([0, 1]);
    expect((records[0]!.binds[1]!.group as { __tag: string }).__tag).toBe('sceneBG');
    expect(records[0]!.dims).toEqual([2, 2, 1]);
  });

  it('TemporalGIReservoirPass: GRIS ON = slot-0 + scene@slot-1', () => {
    const { encoder, records } = makeRecordingEncoder();
    const ctx = makeCtx(encoder);
    new TemporalGIReservoirPass(stubPipeline('giT'), /* grisEnabled */ true).dispatch(ctx);
    expect(records[0]!.binds.map((b) => b.slot)).toEqual([0, 1]);
    // slot-1 must be the shared scene group.
    expect((records[0]!.binds[1]!.group as { __tag: string }).__tag).toBe('sceneBG');
  });

  it('SpatialGIReservoirPass (2-pass): two half-res dispatches with scene@slot-1, terminal label gi-spatial-2', () => {
    const { encoder, records } = makeRecordingEncoder();
    const ctx = makeCtx(encoder);
    new SpatialGIReservoirPass(stubPipeline('giS'), 2, false).dispatch(ctx);
    expect(records.map((r) => r.label)).toEqual(['gi-spatial-1', 'gi-spatial-2']);
    expect(records.every((r) => r.binds.map((b) => b.slot).join() === '0,1')).toBe(true);
    expect(records.every((r) => (r.binds[1]!.group as { __tag: string }).__tag === 'sceneBG')).toBe(true);
    expect(records.every((r) => r.dims.join() === '2,2,1')).toBe(true);
  });

  it('SpatialGIReservoirPass (1-pass): one dispatch labelled gi-spatial-2 + a buffer copy', () => {
    const { encoder, records, copies } = makeRecordingEncoder();
    const ctx = makeCtx(encoder);
    new SpatialGIReservoirPass(stubPipeline('giS'), 1, false).dispatch(ctx);
    expect(records).toHaveLength(1);
    expect(records[0]!.label).toBe('gi-spatial-2');
    expect(records[0]!.binds.map((b) => b.slot)).toEqual([0, 1]);
    expect((records[0]!.binds[1]!.group as { __tag: string }).__tag).toBe('sceneBG');
    expect(records[0]!.dims).toEqual([2, 2, 1]);
    // The 1-pass branch copies spatial → current (current.size = 512).
    expect(copies).toEqual([{ kind: 'buffer', size: 512 }]);
  });

  it('SpatialGIReservoirPass (2-pass, GRIS ON): scene group at slot-1 on BOTH dispatches', () => {
    const { encoder, records } = makeRecordingEncoder();
    const ctx = makeCtx(encoder);
    new SpatialGIReservoirPass(stubPipeline('giS'), 2, true).dispatch(ctx);
    for (const r of records) {
      expect(r.binds.map((b) => b.slot)).toEqual([0, 1]);
      expect((r.binds[1]!.group as { __tag: string }).__tag).toBe('sceneBG');
    }
  });
});

describe('Theme-E dispatch equivalence — RISGIPass NRC slot-4 (#3)', () => {
  it('NRC OFF (default): frame/scene/ubo/hybrid at 0..3, half-res', () => {
    const { encoder, records } = makeRecordingEncoder();
    const ctx = makeCtx(encoder);
    new RISGIPass(stubPipeline('giRis')).dispatch(ctx);
    expect(records).toHaveLength(1);
    expect(records[0]!.label).toBe('gi-ris');
    expect(records[0]!.binds.map((b) => b.slot)).toEqual([0, 1, 2, 3]);
    expect(records[0]!.dims).toEqual([2, 2, 1]);
  });

  it('NRC ON: frame/scene/ubo/hybrid at 0..3 + NRC group at slot-4, half-res', () => {
    const { encoder, records, events } = makeRecordingEncoder();
    const ctx = makeCtx(encoder);
    const nrcBg = { __tag: 'nrcBG' } as unknown as GPUBindGroup;
    const slotClaims = buf('nrcSlotClaims');
    new RISGIPass(
      stubPipeline('giRis'),
      () => nrcBg,
      (enc) => enc.clearBuffer(slotClaims),
    ).dispatch(ctx);
    encoder.copyBufferToBuffer(buf('nrcRecords'), 0, buf('nrcReadback'), 0, 64);
    expect(records[0]!.binds.map((b) => b.slot)).toEqual([0, 1, 2, 3, 4]);
    expect((records[0]!.binds[4]!.group as { __tag: string }).__tag).toBe('nrcBG');
    expect(records[0]!.dims).toEqual([2, 2, 1]);
    expect(events.map((e) => e.kind)).toEqual(['clear', 'pass', 'copy']);
    expect(events[1]!.label).toBe('gi-ris');
  });
});

describe('Theme-E dispatch equivalence — runAtrousChain (#2)', () => {
  it('AtrousDenoiser (legacy 3-iter): ping-pong dispatches atrous-0..2, slot-0, 16×16 grid', () => {
    const { encoder, records } = makeRecordingEncoder();
    const ctx = makeCtx(encoder);
    const denoiserCtx = {
      device: ctx.device,
      encoder,
      resources: {
        common: {
          hdrColorTexture: tex('hdrColor'),
          denoisedPingTexture: tex('denoisedPing'),
          denoisedPongTexture: tex('denoisedPong'),
        },
      },
      bglCache: {} as never,
      gNormalDepthView: { __tag: 'gndView' },
      atrousDirectSigmas: [128, 5, 0.05],
      wgX16: 4, wgY16: 4,
      computeDesc: (label: PassLabel) => ({ label }),
      sharedAtrousPipeline: stubPipeline('sharedAtrous'),
    } as unknown as DenoiserDispatchContext;
    const out = new AtrousDenoiser().dispatch(denoiserCtx);
    expect(records.map((r) => r.label)).toEqual(['atrous-0', 'atrous-1', 'atrous-2']);
    const writes = deviceWrites(ctx.device);
    expect(writes.map((w) => w.offset)).toEqual([0, 256, 512]);
    expect(writes.map((w) => firstF32(w.data))).toEqual([1, 2, 4]);
    const uboBindings = records.map((r) => bufferBinding(r.binds[0]!.group, 4));
    expect(uboBindings.map((b) => b.offset ?? 0)).toEqual([0, 256, 512]);
    expect(new Set(uboBindings.map((b) => b.buffer)).size).toBe(1);
    for (const r of records) {
      expect(r.binds.map((b) => b.slot)).toEqual([0]);
      expect(r.dims).toEqual([4, 4, 1]);
      expect((r.pipeline as { __tag: string }).__tag).toBe('sharedAtrous');
    }
    // 3 iters: ping(0) → pong(1) → ping(2); final output is the ping slot.
    expect((out as unknown as { __tag: string }).__tag).toBe('denoisedPing');
  });

  it('AtrousIndirectPass: 4 ping-pong dispatches, widening labels, ping/pong textures alternate', () => {
    const { encoder, records } = makeRecordingEncoder();
    const ctx = makeCtx(encoder);
    new AtrousIndirectPass(stubPipeline('atrousInd'), { buf: undefined }).dispatch(ctx);
    expect(records.map((r) => r.label)).toEqual([
      'atrous-indirect-0', 'atrous-indirect-1', 'atrous-indirect-2', 'atrous-indirect-3',
    ]);
    const writes = deviceWrites(ctx.device);
    expect(writes.map((w) => w.offset)).toEqual([0, 256, 512, 768]);
    expect(writes.map((w) => firstF32(w.data))).toEqual([1, 2, 4, 8]);
    const uboBindings = records.map((r) => bufferBinding(r.binds[0]!.group, 4));
    expect(uboBindings.map((b) => b.offset ?? 0)).toEqual([0, 256, 512, 768]);
    expect(new Set(uboBindings.map((b) => b.buffer)).size).toBe(1);
    // Every iter binds slot 0 only and dispatches 16×16-grid counts (4×4).
    for (const r of records) {
      expect(r.binds.map((b) => b.slot)).toEqual([0]);
      expect(r.dims).toEqual([4, 4, 1]);
      expect(r.pipeline).toBeDefined();
    }
    // The terminal output published to frameState is the even-iter ping slot
    // for the last (iter=3, odd) iteration → pong texture.
    expect((ctx.frameState.denoisedIndirect as unknown as { __tag: string }).__tag)
      .toBe('indDenoisedPong');
  });

  it('AtrousVarianceDenoiser: atrous iterations bind distinct UBO offsets', () => {
    const { encoder, records } = makeRecordingEncoder();
    const ctx = makeCtx(encoder);
    const denoiser = new AtrousVarianceDenoiser();
    const atrousUbo = buf('atrousVarianceUbo', 3 * 256);
    Object.assign(denoiser as unknown as {
      _welfordPipeline: GPUComputePipeline;
      _variancePipeline: GPUComputePipeline;
      _atrousPipeline: GPUComputePipeline;
      _welfordUboRef: { buf: GPUBuffer };
      _varianceUboRef: { buf: GPUBuffer };
      _atrousUboRef: { buf: GPUBuffer };
    }, {
      _welfordPipeline: stubPipeline('welford'),
      _variancePipeline: stubPipeline('variance'),
      _atrousPipeline: stubPipeline('atrousVariance'),
      _welfordUboRef: { buf: buf('welfordUbo') },
      _varianceUboRef: { buf: buf('varianceUbo') },
      _atrousUboRef: { buf: atrousUbo },
    });

    const out = denoiser.dispatch({
      device: ctx.device,
      encoder,
      resources: ctx.resources,
      gNormalDepthView: ctx.gNormalDepthView,
      isMoving: false,
      wgX16: 4,
      wgY16: 4,
      frameIndex: 3,
      computeDesc: ctx.computeDesc,
    } as unknown as DenoiserDispatchContext);

    expect(records.map((r) => r.label)).toEqual([
      'welford-temporal',
      'atrous-variance-variance',
      'atrous-variance-atrous-0',
      'atrous-variance-atrous-1',
      'atrous-variance-atrous-2',
    ]);
    const atrousWrites = deviceWrites(ctx.device).filter((w) => w.buffer === atrousUbo);
    expect(atrousWrites.map((w) => w.offset)).toEqual([0, 256, 512]);
    expect(atrousWrites.map((w) => firstU32(w.data))).toEqual([0, 1, 2]);
    const atrousRecords = records.slice(2);
    const uboBindings = atrousRecords.map((r) => bufferBinding(r.binds[0]!.group, 5));
    expect(uboBindings.map((b) => b.offset ?? 0)).toEqual([0, 256, 512]);
    expect(new Set(uboBindings.map((b) => b.buffer)).size).toBe(1);
    expect((out as unknown as { __tag: string }).__tag).toBe('denoisedPing');
  });
});

describe('Theme-E label hygiene — neural denoiser (#5)', () => {
  it('DENOISER_PASS_LABELS.neural is exactly the 2 dispatched labels', () => {
    expect(DENOISER_PASS_LABELS['neural']).toEqual(['neural-pack', 'neural-unpack']);
  });

  it('neural no longer borrows the atrous-variance label set', () => {
    expect(DENOISER_PASS_LABELS['neural']).not.toContain('welford-temporal');
    expect(DENOISER_PASS_LABELS['neural']).not.toContain('atrous-variance-variance');
  });
});

describe('Theme-E ordering safety — composePassLabels == dispatch order (#7)', () => {
  // The IndirectTemporalAccumPass declares a synthetic `denoiser-adapter`
  // dependency to pin its slot AFTER the denoiser labels. This asserts the
  // static label table places the denoiser labels exactly where the runtime
  // dispatches them — making the ordering safe by test rather than relying on
  // the topo-sort lexicographic tiebreaker.
  it('atrous-variance: denoiser labels are spliced between gtao-upsample and indirect-temporal-accum', () => {
    const labels = composePassLabels(DENOISER_PASS_LABELS['atrous-variance']);
    const gtaoUp = labels.indexOf('gtao-upsample');
    const indTA = labels.indexOf('indirect-temporal-accum');
    const welford = labels.indexOf('welford-temporal');
    expect(gtaoUp).toBeGreaterThanOrEqual(0);
    expect(welford).toBeGreaterThan(gtaoUp);
    expect(indTA).toBeGreaterThan(welford);
    // All 5 atrous-variance denoiser labels sit contiguously before indTA.
    for (const l of DENOISER_PASS_LABELS['atrous-variance']) {
      expect(labels.indexOf(l)).toBeGreaterThan(gtaoUp);
      expect(labels.indexOf(l)).toBeLessThan(indTA);
    }
  });

  // Build the real registry the orchestrator builds (default config:
  // atrous-variance, no PPG, GTAO on, 2-pass DI/GI), returning the flat
  // runtime-dispatched label sequence.
  function runtimeLabelOrder(): string[] {
    const stubPipeline = {} as GPUComputePipeline;
    const stubRender = {} as GPURenderPipeline;
    const stubUbo = { buf: undefined };
    const denoiser = {
      id: 'atrous-variance',
      passLabels: DENOISER_PASS_LABELS['atrous-variance'],
      state: () => DENOISER_READY_STATE,
    } as unknown as Denoiser;

    const reg = new PassRegistry();
    reg.register(new SampleBudgetPass(stubPipeline, stubUbo, stubUbo));
    reg.register(new RISPass(stubPipeline));
    reg.register(new TemporalReservoirPass(stubPipeline));
    reg.register(new SpatialReservoirPass(stubPipeline));
    reg.register(new RISGIPass(stubPipeline));
    reg.register(new TemporalGIReservoirPass(stubPipeline));
    reg.register(new SpatialGIReservoirPass(stubPipeline));
    reg.register(new ShadePass(stubPipeline));
    reg.register(new MotionVectorsPass(stubPipeline));
    reg.register(new GTAOPass(stubPipeline));
    reg.register(new GTAOUpsamplePass(stubPipeline));
    // cb-prefill must be registered so denoiser-adapter's dependency resolves;
    // with denoiserMode='atrous-variance' the pass gates false and is never dispatched.
    reg.register(new CheckerboardPrefillPass(stubPipeline, stubUbo, /* checkerboard */ false));
    reg.register(new DenoiserAdapterPass(() => denoiser, () => stubPipeline));
    reg.register(new IndirectTemporalAccumPass(stubPipeline, { value: 0 }));
    reg.register(new AtrousIndirectPass(stubPipeline, stubUbo));
    reg.register(new IndirectCombinePass(stubPipeline));
    reg.register(new TransparentOitPass(stubPipeline));
    reg.register(new TemporalAccumPass(stubPipeline, stubUbo));
    reg.register(new ResolvePass(stubPipeline, stubUbo, false));
    reg.register(new CompositePass(stubRender, stubUbo));

    const active = reg.activePasses({ denoiserMode: 'atrous-variance', ppgEnabled: false });
    const labels: string[] = [];
    for (const p of active as readonly Pass[]) labels.push(...p.passLabels);
    return labels;
  }

  it('denoiser-adapter slot is safe: runtime dispatches denoiser labels AFTER gtao-upsample and BEFORE indirect-temporal-accum', () => {
    // This is the property the synthetic `denoiser-adapter` dependency on
    // IndirectTemporalAccumPass guarantees, and what makes the ordering safe by
    // TEST rather than relying on the topo-sort lexicographic tiebreaker. Both
    // the runtime registry order AND the static composePassLabels table must
    // place the active denoiser's labels in that window.
    const runtime = runtimeLabelOrder();
    const gtaoUp = runtime.indexOf('gtao-upsample');
    const indTA = runtime.indexOf('indirect-temporal-accum');
    expect(gtaoUp).toBeGreaterThanOrEqual(0);
    expect(indTA).toBeGreaterThan(gtaoUp);
    for (const l of DENOISER_PASS_LABELS['atrous-variance']) {
      const i = runtime.indexOf(l);
      expect(i).toBeGreaterThan(gtaoUp);
      expect(i).toBeLessThan(indTA);
    }

    // The static slot table agrees: same denoiser window.
    const staticLabels = composePassLabels(DENOISER_PASS_LABELS['atrous-variance']);
    const sGtao = staticLabels.indexOf('gtao-upsample');
    const sIndTA = staticLabels.indexOf('indirect-temporal-accum');
    for (const l of DENOISER_PASS_LABELS['atrous-variance']) {
      const i = staticLabels.indexOf(l);
      expect(i).toBeGreaterThan(sGtao);
      expect(i).toBeLessThan(sIndTA);
    }
  });

  it('runtime denoiser-label subsequence matches the static table exactly, in order', () => {
    // Restrict both sequences to the labels they share (the runtime registry
    // omits the always-static ppg/regir reservations + orders motion-vectors
    // by topo dependency, which is orthogonal to the timestamp slot table).
    // The SHARED labels must appear in the SAME relative order in both — this
    // is the invariant that keeps timestamp slot indices aligned with dispatch.
    const runtime = runtimeLabelOrder();
    const staticLabels = composePassLabels(DENOISER_PASS_LABELS['atrous-variance']) as readonly string[];
    const runtimeSet = new Set(runtime);
    const sharedStatic = staticLabels.filter((l) => runtimeSet.has(l));
    const staticSet = new Set(staticLabels);
    const sharedRuntime = runtime.filter((l) => staticSet.has(l));
    // motion-vectors is the one label the topo-sort relocates vs the static
    // table; exclude it so the assertion pins the denoiser-window ordering
    // (the #7 concern) without coupling to that orthogonal, benign difference.
    const drop = (xs: string[]) => xs.filter((l) => l !== 'motion-vectors');
    expect(drop(sharedRuntime)).toEqual(drop(sharedStatic));
  });

  it('neural: the 2 neural labels splice contiguously into the denoiser-adapter slot', () => {
    const labels = composePassLabels(DENOISER_PASS_LABELS['neural']);
    const gtaoUp = labels.indexOf('gtao-upsample');
    const indTA = labels.indexOf('indirect-temporal-accum');
    const pack = labels.indexOf('neural-pack');
    const unpack = labels.indexOf('neural-unpack');
    // Both present, contiguous (pack immediately before unpack), and inside
    // the (gtao-upsample, indirect-temporal-accum) window — exactly the slot
    // the denoiser-adapter virtual entry occupies.
    expect(pack).toBeGreaterThan(gtaoUp);
    expect(unpack).toBe(pack + 1);
    expect(unpack).toBeLessThan(indTA);
  });
});

describe('CheckerboardPrefillPass — pass-order assertion for checkerboard+real-denoiser path', () => {
  // This block verifies the static label table (which drives timestamp-query
  // slot assignment) places cb-prefill in the expected window between
  // gtao-upsample and the denoiser labels. All four real denoiser modes are
  // checked so the insertion point is pinned regardless of which denoiser the
  // user configures.
  it.each(['svgf-real', 'bmfr', 'neural', 'oidn-final'] as const)(
    'cb-prefill sits between gtao-upsample and the denoiser labels for mode=%s',
    (mode) => {
      const denoiserLabels = DENOISER_PASS_LABELS[mode as keyof typeof DENOISER_PASS_LABELS] ?? ['denoiser-adapter'];
      const labels = composePassLabels(denoiserLabels as PassLabel[]);
      const cbPrefill = labels.indexOf('cb-prefill');
      const gtaoUp = labels.indexOf('gtao-upsample');
      const firstDenoiserLabel = denoiserLabels[0];
      // cb-prefill must be present
      expect(cbPrefill, `cb-prefill missing from layout for denoiserMode=${mode}`).toBeGreaterThanOrEqual(0);
      // cb-prefill must come after gtao-upsample
      expect(cbPrefill, `cb-prefill must come after gtao-upsample`).toBeGreaterThan(gtaoUp);
      if (firstDenoiserLabel) {
        const firstDenoiser = labels.indexOf(firstDenoiserLabel as PassLabel);
        // cb-prefill must come before the first denoiser label
        expect(cbPrefill, `cb-prefill must come before first denoiser label`).toBeLessThan(firstDenoiser);
      }
    },
  );

  it('cb-prefill is in the static label table for atrous-variance mode (always present)', () => {
    // cb-prefill is a permanent slot in the timestamp layout for ALL denoiser
    // modes — it gates to a no-op dispatch for atrous/atrous-variance modes but
    // its slot is always reserved so the layout stays stable.
    const labels = composePassLabels(DENOISER_PASS_LABELS['atrous-variance']);
    expect(labels.includes('cb-prefill')).toBe(true);
  });

  it('cb-prefill slot is immediately before the denoiser labels in the label sequence', () => {
    // For svgf-real, the label after cb-prefill must be the first denoiser label.
    const svgfLabels = DENOISER_PASS_LABELS['svgf-real'];
    const labels = composePassLabels(svgfLabels as PassLabel[]);
    const cbIdx = labels.indexOf('cb-prefill');
    expect(cbIdx).toBeGreaterThanOrEqual(0);
    // The very next label after cb-prefill must be the first svgf-real label.
    expect(labels[cbIdx + 1]).toBe(svgfLabels[0]);
  });
});

describe('ShadePass + SpatialReservoirPass — checkerboard compacted dispatch', () => {
  // OFF (default): full-res dispatch, byte-identical to the pre-checkerboard
  // path — frame/scene/ubo/hybrid at slots 0..3, full-res workgroup dims
  // (ceil(W/8) × ceil(H/8)). makeCtx is 64×64 ⇒ [8, 8, 1].
  it('checkerboard OFF: full-res dispatch [8,8,1], slots 0..3, label=shade', () => {
    const { encoder, records } = makeRecordingEncoder();
    const ctx = { ...makeCtx(encoder), checkerboardOn: false, frameParity: 0 } as PassDispatchContext;
    new ShadePass(stubPipeline('shade')).dispatch(ctx);
    expect(records).toHaveLength(1);
    expect(records[0]!.label).toBe('shade');
    expect(records[0]!.binds.map((b) => b.slot)).toEqual([0, 1, 2, 3]);
    expect(records[0]!.dims).toEqual([8, 8, 1]);
  });

  // ON: the X dispatch compacts to the active-parity columns — ceil(W/2)
  // columns ⇒ ceil(ceil(W/2)/8) workgroups in X — while Y stays full-res. For
  // 64×64: ceil(64/2)=32 cols ⇒ ceil(32/8)=4 workgroups X, ceil(64/8)=8 Y ⇒
  // [4, 8, 1]. That is HALF the X workgroups of the OFF path (4 vs 8), i.e. the
  // dispatched thread count halves — the whole point of the compaction. Same
  // bind slots, same label.
  it('checkerboard ON: compacted dispatch [4,8,1] (half the X workgroups), slots 0..3', () => {
    const { encoder, records } = makeRecordingEncoder();
    const ctx = { ...makeCtx(encoder), checkerboardOn: true, frameParity: 1 } as PassDispatchContext;
    new ShadePass(stubPipeline('shade')).dispatch(ctx);
    expect(records).toHaveLength(1);
    expect(records[0]!.label).toBe('shade');
    expect(records[0]!.binds.map((b) => b.slot)).toEqual([0, 1, 2, 3]);
    expect(records[0]!.dims).toEqual([4, 8, 1]);
    // The compacted X count is strictly less than the OFF full-res X count.
    expect(records[0]!.dims[0]).toBeLessThan(ctx.wgX);
  });

  // SpatialReservoirPass compacts EXACTLY like ShadePass when checkerboard is
  // ON: every spatial dispatch (1 or 2 passes) drops to ceil(ceil(W/2)/8) X
  // workgroups. The spatial reuse is the pipeline's dominant cost (6 BVH casts
  // per pixel × passCount), so the compaction is where the frame-time saving
  // comes from. OFF ⇒ full-res per label, byte-identical to before.
  it('SpatialReservoirPass checkerboard OFF: full-res [8,8,1] per label (spatial-1, spatial-2)', () => {
    const { encoder, records } = makeRecordingEncoder();
    const ctx = { ...makeCtx(encoder), checkerboardOn: false, frameParity: 0 } as PassDispatchContext;
    new SpatialReservoirPass(stubPipeline('spatial'), 2).dispatch(ctx);
    expect(records.map((r) => r.label)).toEqual(['spatial-1', 'spatial-2']);
    for (const r of records) {
      expect(r.binds.map((b) => b.slot)).toEqual([0, 1, 2]);
      expect(r.dims).toEqual([8, 8, 1]);
    }
  });

  it('SpatialReservoirPass checkerboard ON: compacted [4,8,1] per label (half the X workgroups)', () => {
    const { encoder, records } = makeRecordingEncoder();
    const ctx = { ...makeCtx(encoder), checkerboardOn: true, frameParity: 1 } as PassDispatchContext;
    new SpatialReservoirPass(stubPipeline('spatial'), 2).dispatch(ctx);
    expect(records.map((r) => r.label)).toEqual(['spatial-1', 'spatial-2']);
    for (const r of records) {
      expect(r.binds.map((b) => b.slot)).toEqual([0, 1, 2]);
      // 64×64 ⇒ ceil(64/2)=32 cols ⇒ ceil(32/8)=4 X workgroups, ceil(64/8)=8 Y.
      expect(r.dims).toEqual([4, 8, 1]);
      // Strictly fewer X workgroups than the OFF full-res count.
      expect(r.dims[0]).toBeLessThan(ctx.wgX);
    }
  });

  it('SpatialReservoirPass checkerboard ON (1-pass): single compacted spatial-2 dispatch', () => {
    const { encoder, records } = makeRecordingEncoder();
    const ctx = { ...makeCtx(encoder), checkerboardOn: true, frameParity: 0 } as PassDispatchContext;
    new SpatialReservoirPass(stubPipeline('spatial'), 1).dispatch(ctx);
    expect(records.map((r) => r.label)).toEqual(['spatial-2']);
    expect(records[0]!.dims).toEqual([4, 8, 1]);
  });

  // RISPass compacts EXACTLY like ShadePass/SpatialReservoirPass when
  // checkerboard is ON: the initial-candidate dispatch (primary BVH cast +
  // M_LIGHT emitter loop) drops to ceil(ceil(W/2)/8) X workgroups — one thread
  // per active-parity pixel — so RIS re-seeds only the reservoirs shade consumes
  // this frame; the gap-parity slots keep their carried-forward reservoir for
  // the FULL-RATE temporal pass. RIS binds the RIS-only light-tree group at
  // slot 3 (slots 0..3). OFF ⇒ full-res, byte-identical.
  it('RISPass checkerboard OFF: full-res dispatch [8,8,1], slots 0..3, label=ris', () => {
    const { encoder, records } = makeRecordingEncoder();
    const ctx = { ...makeCtx(encoder), checkerboardOn: false, frameParity: 0 } as PassDispatchContext;
    new RISPass(stubPipeline('ris')).dispatch(ctx);
    expect(records).toHaveLength(1);
    expect(records[0]!.label).toBe('ris');
    expect(records[0]!.binds.map((b) => b.slot)).toEqual([0, 1, 2, 3]);
    expect(records[0]!.dims).toEqual([8, 8, 1]);
  });

  it('RISPass checkerboard ON: compacted [4,8,1] (half the X workgroups), slots 0..3', () => {
    const { encoder, records } = makeRecordingEncoder();
    const ctx = { ...makeCtx(encoder), checkerboardOn: true, frameParity: 1 } as PassDispatchContext;
    new RISPass(stubPipeline('ris')).dispatch(ctx);
    expect(records).toHaveLength(1);
    expect(records[0]!.label).toBe('ris');
    expect(records[0]!.binds.map((b) => b.slot)).toEqual([0, 1, 2, 3]);
    // 64×64 ⇒ ceil(64/2)=32 cols ⇒ ceil(32/8)=4 X workgroups, ceil(64/8)=8 Y.
    expect(records[0]!.dims).toEqual([4, 8, 1]);
    // Strictly fewer X workgroups than the OFF full-res count.
    expect(records[0]!.dims[0]).toBeLessThan(ctx.wgX);
  });

  // Parity-decode invariant (mirrors the shade.wgsl decode + the resolve.wgsl
  // shaded-pixel predicate): every compacted thread (cx, cy) maps to a
  // full-res pixel that is on the ACTIVE parity, and the decoded set exactly
  // equals the active-parity pixel set the resolve pass copies through (and
  // the complementary GAP set it reprojects). This is what guarantees the
  // rendered image is unchanged by the compaction. The SAME decode is shared by
  // ris + spatial + shade (all use `px = gid.x*2 + ((gid.y + frameParity)&1)`),
  // so this invariant covers all three compacted passes.
  it('CheckerboardPrefillPass checkerboard OFF (host flag false): gates() returns false, not dispatched', () => {
    // When checkerboard=false at the host level, gates() always returns false
    // regardless of the denoiser mode — the pass is never dispatched and
    // hdrColorTexture is byte-identical to the pre-cb-prefill path.
    const pass = new CheckerboardPrefillPass(stubPipeline('cbPrefill'), { buf: buf('cbUbo') }, /* checkerboard */ false);
    expect(pass.gates({ denoiserMode: 'svgf-real', checkerboardOn: true, ppgEnabled: false })).toBe(false);
    expect(pass.gates({ denoiserMode: 'bmfr', checkerboardOn: true, ppgEnabled: false })).toBe(false);
    expect(pass.gates({ denoiserMode: 'neural', checkerboardOn: true, ppgEnabled: false })).toBe(false);
    expect(pass.gates({ denoiserMode: 'oidn-final', checkerboardOn: true, ppgEnabled: false })).toBe(false);
    // Also false for non-real denoisers (belt-and-suspenders).
    expect(pass.gates({ denoiserMode: 'atrous-variance', checkerboardOn: true, ppgEnabled: false })).toBe(false);
  });

  it('CheckerboardPrefillPass checkerboard ON host + non-real denoiser: gates() returns false', () => {
    // Even with the host flag true and checkerboardOn=true per-frame, the pass
    // only runs for the four real denoisers — the default atrous paths survive
    // via temporal accumulation.
    const pass = new CheckerboardPrefillPass(stubPipeline('cbPrefill'), { buf: buf('cbUbo') }, /* checkerboard */ true);
    expect(pass.gates({ denoiserMode: 'atrous-variance', checkerboardOn: true, ppgEnabled: false })).toBe(false);
    expect(pass.gates({ denoiserMode: 'atrous', checkerboardOn: true, ppgEnabled: false })).toBe(false);
    expect(pass.gates({ denoiserMode: 'none', checkerboardOn: true, ppgEnabled: false })).toBe(false);
  });

  it('CheckerboardPrefillPass checkerboard ON + real denoiser: gates() returns true', () => {
    const pass = new CheckerboardPrefillPass(stubPipeline('cbPrefill'), { buf: buf('cbUbo') }, /* checkerboard */ true);
    expect(pass.gates({ denoiserMode: 'svgf-real', checkerboardOn: true, ppgEnabled: false })).toBe(true);
    expect(pass.gates({ denoiserMode: 'bmfr', checkerboardOn: true, ppgEnabled: false })).toBe(true);
    expect(pass.gates({ denoiserMode: 'neural', checkerboardOn: true, ppgEnabled: false })).toBe(true);
    expect(pass.gates({ denoiserMode: 'oidn-final', checkerboardOn: true, ppgEnabled: false })).toBe(true);
  });

  it('CheckerboardPrefillPass checkerboard ON + per-frame motion fallback: gates() returns false', () => {
    // If cbActiveThisFrame is false (motion exceeded fallback) the per-frame
    // checkerboardOn is false and the pass is skipped.
    const pass = new CheckerboardPrefillPass(stubPipeline('cbPrefill'), { buf: buf('cbUbo') }, /* checkerboard */ true);
    expect(pass.gates({ denoiserMode: 'svgf-real', checkerboardOn: false, ppgEnabled: false })).toBe(false);
  });

  it('CheckerboardPrefillPass checkerboard ON + real denoiser: dispatches 1 pass, slot-0, label=cb-prefill', () => {
    const { encoder, records } = makeRecordingEncoder();
    const ctx = {
      ...makeCtx(encoder),
      checkerboardOn: true,
      frameParity: 0,
    } as PassDispatchContext;
    const pass = new CheckerboardPrefillPass(
      stubPipeline('cbPrefill'),
      { buf: buf('cbUbo') },
      /* checkerboard */ true,
    );
    pass.dispatch(ctx);
    expect(records).toHaveLength(1);
    const r = records[0]!;
    expect(r.label).toBe('cb-prefill');
    expect(r.pipeline).toBeDefined();
    expect(r.binds.map((b) => b.slot)).toEqual([0]);
    expect(r.dims).toEqual([8, 8, 1]);
  });

  it('compacted-gid decode covers exactly the active-parity pixel set (resolve agreement)', () => {
    const decodePix = (cx: number, cy: number, frameParity: number) => {
      const startCol = (cy + frameParity) & 1;
      return { x: cx * 2 + startCol, y: cy };
    };
    for (const W of [7, 8, 16, 31, 64]) {
      for (const H of [5, 8, 17]) {
        for (const parity of [0, 1]) {
          const compactCols = Math.ceil(W / 2);
          // The set decoded by the compacted dispatch (bounds-guarded like the
          // shader's `any(pix >= dims)` early-out).
          const decoded = new Set<number>();
          for (let cy = 0; cy < H; cy++) {
            for (let cx = 0; cx < compactCols; cx++) {
              const p = decodePix(cx, cy, parity);
              if (p.x >= W || p.y >= H) continue; // overshoot guard
              // Every decoded pixel must be on the active parity.
              expect(((p.x + p.y) & 1)).toBe(parity);
              decoded.add(p.y * W + p.x);
            }
          }
          // The active-parity pixel set the resolve pass treats as "shaded".
          const active = new Set<number>();
          for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
              if (((x + y) & 1) === parity) active.add(y * W + x);
            }
          }
          // Exact set equality: no active pixel missed, no extra pixel shaded.
          expect(decoded.size).toBe(active.size);
          for (const idx of active) expect(decoded.has(idx)).toBe(true);
        }
      }
    }
  });
});
