import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../bindGroupBuilders.js', async () => {
  const actual = await vi.importActual<typeof import('../bindGroupBuilders.js')>('../bindGroupBuilders.js');
  return {
    ...actual,
    buildGTAOUpsampleBindGroup: vi.fn(() => ({ label: 'gtao-upsample-bg' })),
    buildPreparedAccumBindGroup: vi.fn(() => ({ label: 'temporal-accum-bg' })),
    buildSampleBudgetBindGroup: vi.fn(() => ({ label: 'sample-budget-bg' })),
    buildIndirectTemporalAccumBindGroup: vi.fn(() => ({ label: 'indirect-temporal-bg' })),
    buildTransparentOitBindGroup: vi.fn(() => ({ label: 'transparent-oit-bg' })),
    buildPpgUpdateBindGroups: vi.fn(() => [{ label: 'ppg-bg0' }, { label: 'ppg-bg1' }]),
    buildRegirBuildBindGroup: vi.fn(() => ({ label: 'regir-bg' })),
  };
});

import { PipelineResourceCache } from '../PipelineResourceCache.js';
import {
  buildGTAOUpsampleBindGroup,
  buildIndirectTemporalAccumBindGroup,
  buildPreparedAccumBindGroup,
  buildPpgUpdateBindGroups,
  buildRegirBuildBindGroup,
  buildSampleBudgetBindGroup,
  buildTransparentOitBindGroup,
} from '../bindGroupBuilders.js';
import { GTAOUpsamplePass } from '../passes/GTAOUpsamplePass.js';
import { IndirectTemporalAccumPass } from '../passes/IndirectTemporalAccumPass.js';
import { PPGUpdatePass } from '../passes/PPGUpdatePass.js';
import { ReGIRBuildPass } from '../passes/ReGIRBuildPass.js';
import { SampleBudgetPass } from '../passes/SampleBudgetPass.js';
import { TemporalAccumPass } from '../passes/TemporalAccumPass.js';
import { TransparentOitPass } from '../passes/TransparentOitPass.js';

function texture(label: string): GPUTexture & { createView: ReturnType<typeof vi.fn> } {
  return {
    label,
    createView: vi.fn(() => ({ label: `${label}-view` })),
  } as unknown as GPUTexture & { createView: ReturnType<typeof vi.fn> };
}

function buffer(label: string): GPUBuffer {
  return { label, size: 1024 } as unknown as GPUBuffer;
}

function computePassRecorder() {
  return {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
    end: vi.fn(),
  };
}

function baseCtx(overrides: Record<string, unknown> = {}) {
  const pass = computePassRecorder();
  const common = {
    varianceBuffer: texture('variance-a'),
    varianceBufferAux: texture('variance-b'),
    tierTexture: texture('tier'),
    gNormalDepthTexture: texture('gnormal'),
    motionVectorTexture: texture('motion'),
    resolvedTexture: texture('resolved'),
    hdrIndirectTexture: texture('hdr-indirect'),
    transparentCompositeTexture: texture('transparent-composite'),
    indirectAccumPingTexture: texture('indirect-ping'),
    indirectAccumPongTexture: texture('indirect-pong'),
    uboBuffer: buffer('ubo'),
  };
  const ctx = {
    device: {
      queue: { writeBuffer: vi.fn() },
    },
    encoder: {
      beginComputePass: vi.fn(() => pass),
      copyBufferToBuffer: vi.fn(),
    },
    width: 64,
    height: 32,
    frameIndex: 3,
    frameCount: 9,
    bglCache: {},
    resources: {
      common,
      gtao: {
        aoHalfTexture: texture('ao-half'),
        aoFullTexture: texture('ao-full'),
        gtaoUboBuffer: buffer('gtao-ubo'),
      },
      restirGI: {
        reservoirGiCurrentBuffer: buffer('gi-current'),
        reservoirGiPreviousBuffer: buffer('gi-prev'),
      },
      ppg: {
        fluxAtomicsBuf: buffer('ppg-flux'),
        sTreeBuf: buffer('ppg-stree'),
        dTreeBuf: buffer('ppg-dtree'),
        dTreeOffsetsBuf: buffer('ppg-offsets'),
        cellSampleCountsBuf: buffer('ppg-counts'),
        updateUboBuffer: buffer('ppg-ubo'),
      },
    },
    inputs: {
      gtao: {
        adaptiveSamplingThresholdLow: 0.01,
        adaptiveSamplingThresholdHigh: 0.2,
      },
    },
    frameState: {},
    welfordPing: 0,
    wgX: 8,
    wgY: 4,
    wgX16: 4,
    wgY16: 2,
    halfWgX: 4,
    halfWgY: 2,
    checkerboardWgX: 4,
    checkerboardWgY: 4,
    computeDesc: vi.fn((label: string) => ({ label })),
    resourceCache: new PipelineResourceCache(),
    ...overrides,
  };
  return { ctx: ctx as never, pass, common };
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('pass bind-group cache', () => {
  it('reuses descriptor-free texture views until the cache is cleared', () => {
    const cache = new PipelineResourceCache();
    const tex = texture('aux-buffer');

    const first = cache.textureView(tex);
    const second = cache.textureView(tex);
    cache.clear();
    const afterClear = cache.textureView(tex);

    expect(first).toBe(second);
    expect(afterClear).not.toBe(first);
    expect(tex.createView).toHaveBeenCalledTimes(2);
  });

  it('reuses stable GTAO upsample bind groups and texture views', () => {
    const { ctx, common } = baseCtx();
    const pass = new GTAOUpsamplePass({} as GPUComputePipeline);

    pass.dispatch(ctx);
    pass.dispatch(ctx);

    expect(buildGTAOUpsampleBindGroup).toHaveBeenCalledTimes(1);
    expect(common.gNormalDepthTexture.createView).toHaveBeenCalledTimes(1);
  });

  it('keeps per-frame sample-budget UBO writes while reusing the bind group', () => {
    const budgetUbo = { buf: buffer('budget-ubo') };
    const sampleCountUbo = { buf: buffer('sample-count-ubo') };
    const { ctx } = baseCtx();
    const pass = new SampleBudgetPass({} as GPUComputePipeline, budgetUbo, sampleCountUbo);

    pass.dispatch(ctx);
    pass.dispatch(ctx);

    expect(buildSampleBudgetBindGroup).toHaveBeenCalledTimes(1);
    expect((ctx as { device: { queue: { writeBuffer: ReturnType<typeof vi.fn> } } }).device.queue.writeBuffer)
      .toHaveBeenCalledTimes(4);
  });

  it('keeps temporal alpha UBO writes while reusing the accum bind group', () => {
    const accumUbo = { buf: buffer('accum-ubo') };
    const { ctx } = baseCtx({
      frameState: {
        combinedDenoised: texture('combined'),
        readAccum: texture('accum-read'),
        writeAccum: texture('accum-write'),
        alpha: 0.25,
      },
    });
    const pass = new TemporalAccumPass({} as GPUComputePipeline, accumUbo);

    pass.dispatch(ctx);
    (ctx as { frameState: { alpha: number } }).frameState.alpha = 0.5;
    pass.dispatch(ctx);

    expect(buildPreparedAccumBindGroup).toHaveBeenCalledTimes(1);
    expect((ctx as { device: { queue: { writeBuffer: ReturnType<typeof vi.fn> } } }).device.queue.writeBuffer)
      .toHaveBeenCalledTimes(2);
  });

  it('caches both indirect temporal ping-pong bind-group variants', () => {
    const pingPong = { value: 0 as 0 | 1 };
    const { ctx } = baseCtx();
    const pass = new IndirectTemporalAccumPass({} as GPUComputePipeline, pingPong);

    pass.dispatch(ctx);
    pass.dispatch(ctx);
    pass.dispatch(ctx);

    expect(buildIndirectTemporalAccumBindGroup).toHaveBeenCalledTimes(2);
  });

  it('publishes the transparent composition texture and reuses its bind group', () => {
    const combined = texture('combined');
    const { ctx, common, pass: computePass } = baseCtx({
      frameState: { combinedDenoised: combined },
      frameBindGroup: { label: 'frame-bg' },
      sceneBindGroup: { label: 'scene-bg' },
      uboBindGroup: { label: 'ubo-bg' },
    });
    const pass = new TransparentOitPass({} as GPUComputePipeline);

    pass.dispatch(ctx);
    (ctx as { frameState: { combinedDenoised: GPUTexture } }).frameState.combinedDenoised = combined;
    pass.dispatch(ctx);

    expect(buildTransparentOitBindGroup).toHaveBeenCalledTimes(1);
    expect((ctx as { frameState: { combinedDenoised: unknown } }).frameState.combinedDenoised)
      .toBe(common.transparentCompositeTexture);
    expect(computePass.setBindGroup).toHaveBeenCalledWith(3, { label: 'transparent-oit-bg' });
  });

  it('caches tuple-valued PPG update bind groups', () => {
    const { ctx, pass: computePass } = baseCtx();
    const pass = new PPGUpdatePass({
      getBindGroupLayout: vi.fn((i: number) => ({ label: `layout-${i}` })),
    } as unknown as GPUComputePipeline);

    pass.dispatch(ctx);
    pass.dispatch(ctx);

    expect(buildPpgUpdateBindGroups).toHaveBeenCalledTimes(1);
    expect(computePass.setBindGroup).toHaveBeenCalledWith(0, { label: 'ppg-bg0' });
    expect(computePass.setBindGroup).toHaveBeenCalledWith(1, { label: 'ppg-bg1' });
  });

  it('rebuilds ReGIR after an emitter buffer identity change', () => {
    const resources = {
      combinedLightTreeBuffer: buffer('tree'),
      emitterBuffer: buffer('emitters-a'),
      uboBuffer: buffer('ubo'),
    };
    const cache = new PipelineResourceCache();
    const pass = new ReGIRBuildPass(
      {} as GPUComputePipeline,
      { live: true, cellCount: 1, config: { survivorsPerCell: 1 } } as never,
      {},
      () => resources,
    );
    const { ctx } = baseCtx({ resourceCache: cache });

    pass.dispatch(ctx);
    pass.dispatch(ctx);
    resources.emitterBuffer = buffer('emitters-b');
    pass.dispatch(ctx);

    expect(buildRegirBuildBindGroup).toHaveBeenCalledTimes(2);
  });
});
