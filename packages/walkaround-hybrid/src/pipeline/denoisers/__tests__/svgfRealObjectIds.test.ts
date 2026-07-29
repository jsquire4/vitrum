import { describe, expect, it, vi } from 'vitest';

import { PipelineResourceCache } from '../../PipelineResourceCache.js';
import { SVGFRealDenoiser } from '../svgfReal.js';

type TestTexture = GPUTexture & {
  readonly label: string;
  readonly view: GPUTextureView;
};

function tex(label: string, width = 8, height = 4): TestTexture {
  const view = { label: `${label}-view` } as unknown as GPUTextureView;
  return {
    label,
    view,
    width,
    height,
    createView: vi.fn(() => view),
  } as unknown as TestTexture;
}

function pipeline(label: string): GPUComputePipeline {
  return {
    label,
    getBindGroupLayout: vi.fn(() => ({ label: `${label}-bgl` })),
  } as unknown as GPUComputePipeline;
}

describe('SVGFRealDenoiser object IDs', () => {
  it('binds real current/previous object-id textures and copies current to previous', () => {
    const bindGroups: GPUBindGroupDescriptor[] = [];
    const device = {
      queue: { writeBuffer: vi.fn() },
      createBindGroup: vi.fn((desc: GPUBindGroupDescriptor) => {
        bindGroups.push(desc);
        return { label: desc.label } as unknown as GPUBindGroup;
      }),
    } as unknown as GPUDevice;
    const pass = {
      setPipeline: vi.fn(),
      setBindGroup: vi.fn(),
      dispatchWorkgroups: vi.fn(),
      end: vi.fn(),
    };
    const encoder = {
      beginComputePass: vi.fn(() => pass),
      copyTextureToTexture: vi.fn(),
    } as unknown as GPUCommandEncoder;

    const denoiser = new SVGFRealDenoiser();
    const internals = denoiser as unknown as {
      _reprojPipeline: GPUComputePipeline;
      _momentsPipeline: GPUComputePipeline;
      _fallbackPipeline: GPUComputePipeline;
      _atrousPipeline: GPUComputePipeline;
      _reprojUboRef: { buf?: GPUBuffer };
      _atrousUboRefs: Array<{ buf?: GPUBuffer }>;
    };
    const ubo = { label: 'ubo' } as unknown as GPUBuffer;
    internals._reprojPipeline = pipeline('reproj');
    internals._momentsPipeline = pipeline('moments');
    internals._fallbackPipeline = pipeline('fallback');
    internals._atrousPipeline = pipeline('atrous');
    internals._reprojUboRef.buf = ubo;
    for (const ref of internals._atrousUboRefs) ref.buf = ubo;

    const currentObjectId = tex('current-object-id');
    const previousObjectId = tex('previous-object-id');
    const gNormalDepth = tex('g-normal-depth');
    const prevNormalDepth = tex('prev-normal-depth');
    const denoisedPing = tex('ping');
    const radianceA = tex('rad-a');
    const radianceB = tex('rad-b');
    const resources = {
      common: {
        hdrColorTexture: tex('hdr'),
        motionVectorTexture: tex('motion'),
        gNormalDepthTexture: gNormalDepth,
        denoisedPingTexture: denoisedPing,
        denoisedPongTexture: tex('pong'),
      },
      svgf: {
        svgfCurrentObjectIdTexture: currentObjectId,
        svgfPreviousObjectIdTexture: previousObjectId,
        svgfPrevNormalDepthTexture: prevNormalDepth,
        svgfHistoryLengthTextureA: tex('hist-a'),
        svgfHistoryLengthTextureB: tex('hist-b'),
        svgfMomentsTextureA: tex('mom-a'),
        svgfMomentsTextureB: tex('mom-b'),
        svgfPrevRadianceTextureA: radianceA,
        svgfPrevRadianceTextureB: radianceB,
        svgfVarianceTexture: tex('variance'),
        svgfVarianceMomentsIntermedTexture: tex('variance-intermed'),
      },
    };

    denoiser.dispatch({
      device,
      encoder,
      resources,
      gNormalDepthView: gNormalDepth.view,
      atrousDirectSigmas: [128, 5, 0.05],
      atrousIndirectSigmas: [32, 20, 0.5],
      wgX16: 1,
      wgY16: 1,
      computeDesc: (label: string) => ({ label }),
    } as never);

    const reprojBindGroup = bindGroups.find((bg) => bg.label === 'svgf-real-reproj-bg')!;
    const reprojEntries = Array.from(reprojBindGroup.entries);
    const currObj = reprojEntries.find((entry: GPUBindGroupEntry) => entry.binding === 5)!;
    const prevObj = reprojEntries.find((entry: GPUBindGroupEntry) => entry.binding === 8)!;
    expect(currObj.resource).toBe(currentObjectId.view);
    expect(prevObj.resource).toBe(previousObjectId.view);

    // First à-trous output is the next frame's radiance history (§4.3).
    expect(encoder.copyTextureToTexture).toHaveBeenCalledWith(
      { texture: denoisedPing },
      { texture: radianceB },
      { width: denoisedPing.width, height: denoisedPing.height, depthOrArrayLayers: 1 },
    );
    expect(encoder.copyTextureToTexture).toHaveBeenCalledWith(
      { texture: gNormalDepth },
      { texture: prevNormalDepth },
      { width: gNormalDepth.width, height: gNormalDepth.height, depthOrArrayLayers: 1 },
    );
    expect(encoder.copyTextureToTexture).toHaveBeenCalledWith(
      { texture: currentObjectId },
      { texture: previousObjectId },
      { width: currentObjectId.width, height: currentObjectId.height, depthOrArrayLayers: 1 },
    );
  });

  it('uses the pipeline resource cache for repeated SVGF texture views', () => {
    const device = {
      queue: { writeBuffer: vi.fn() },
      createBindGroup: vi.fn(
        (desc: GPUBindGroupDescriptor) => ({ label: desc.label }) as unknown as GPUBindGroup,
      ),
    } as unknown as GPUDevice;
    const pass = {
      setPipeline: vi.fn(),
      setBindGroup: vi.fn(),
      dispatchWorkgroups: vi.fn(),
      end: vi.fn(),
    };
    const encoder = {
      beginComputePass: vi.fn(() => pass),
      copyTextureToTexture: vi.fn(),
    } as unknown as GPUCommandEncoder;

    const denoiser = new SVGFRealDenoiser();
    const internals = denoiser as unknown as {
      _reprojPipeline: GPUComputePipeline;
      _momentsPipeline: GPUComputePipeline;
      _fallbackPipeline: GPUComputePipeline;
      _atrousPipeline: GPUComputePipeline;
      _reprojUboRef: { buf?: GPUBuffer };
      _atrousUboRefs: Array<{ buf?: GPUBuffer }>;
    };
    const ubo = { label: 'ubo' } as unknown as GPUBuffer;
    internals._reprojPipeline = pipeline('reproj');
    internals._momentsPipeline = pipeline('moments');
    internals._fallbackPipeline = pipeline('fallback');
    internals._atrousPipeline = pipeline('atrous');
    internals._reprojUboRef.buf = ubo;
    for (const ref of internals._atrousUboRefs) ref.buf = ubo;

    const gNormalDepth = tex('g-normal-depth');
    const prevNormalDepth = tex('prev-normal-depth');
    const hdr = tex('hdr');
    const histB = tex('hist-b');
    const momB = tex('mom-b');
    const varianceIntermed = tex('variance-intermed');
    const variance = tex('variance');
    const currentObjectId = tex('current-object-id');
    const previousObjectId = tex('previous-object-id');
    const resources = {
      common: {
        hdrColorTexture: hdr,
        motionVectorTexture: tex('motion'),
        gNormalDepthTexture: gNormalDepth,
        denoisedPingTexture: tex('ping'),
        denoisedPongTexture: tex('pong'),
      },
      svgf: {
        svgfCurrentObjectIdTexture: currentObjectId,
        svgfPreviousObjectIdTexture: previousObjectId,
        svgfPrevNormalDepthTexture: prevNormalDepth,
        svgfHistoryLengthTextureA: tex('hist-a'),
        svgfHistoryLengthTextureB: histB,
        svgfMomentsTextureA: tex('mom-a'),
        svgfMomentsTextureB: momB,
        svgfPrevRadianceTextureA: tex('rad-a'),
        svgfPrevRadianceTextureB: tex('rad-b'),
        svgfVarianceTexture: variance,
        svgfVarianceMomentsIntermedTexture: varianceIntermed,
      },
    };

    denoiser.dispatch({
      device,
      encoder,
      resources,
      gNormalDepthView: gNormalDepth.view,
      atrousDirectSigmas: [128, 5, 0.05],
      atrousIndirectSigmas: [32, 20, 0.5],
      wgX16: 1,
      wgY16: 1,
      computeDesc: (label: string) => ({ label }),
      resourceCache: new PipelineResourceCache(),
    } as never);

    expect(gNormalDepth.createView).toHaveBeenCalledTimes(1);
    expect(prevNormalDepth.createView).toHaveBeenCalledTimes(1);
    expect(hdr.createView).toHaveBeenCalledTimes(1);
    expect(histB.createView).toHaveBeenCalledTimes(1);
    expect(momB.createView).toHaveBeenCalledTimes(1);
    expect(varianceIntermed.createView).toHaveBeenCalledTimes(1);
    expect(variance.createView).toHaveBeenCalledTimes(1);
  });
});
