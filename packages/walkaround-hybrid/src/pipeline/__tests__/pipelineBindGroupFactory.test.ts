import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../bindGroupBuilders.js', () => ({
  buildFrameBindGroup: vi.fn(() => ({ label: 'frame' })),
  buildRisGiFrameBindGroup: vi.fn(() => ({ label: 'ris-gi-frame' })),
  buildSceneBindGroup: vi.fn(() => ({ label: 'scene' })),
  buildUboBindGroup: vi.fn(() => ({ label: 'ubo' })),
  buildCompositeBindGroup: vi.fn(() => ({ label: 'composite' })),
}));

import {
  buildCompositePresentBindGroup,
  buildPerFrameBindGroups,
} from '../pipelineBindGroupFactory.js';
import { PipelineResourceCache } from '../PipelineResourceCache.js';
import {
  buildCompositeBindGroup,
  buildFrameBindGroup,
  buildRisGiFrameBindGroup,
  buildSceneBindGroup,
  buildUboBindGroup,
} from '../bindGroupBuilders.js';

beforeEach(() => {
  vi.clearAllMocks();
});

describe('pipelineBindGroupFactory', () => {
  it('buildPerFrameBindGroups wires frame, scene, ubo, and hybrid layers', () => {
    const device = {} as GPUDevice;
    const cache = {};
    const placeholderView = {} as GPUTextureView;
    const scene = {
      bvhNodesBuffer: {} as GPUBuffer,
      bvhIndexBuffer: {} as GPUBuffer,
      bvhPositionBuffer: {} as GPUBuffer,
      emitterBuffer: {} as GPUBuffer,
      emitterCdfBuffer: {} as GPUBuffer,
      bvhBeerTextureView: {} as GPUTextureView,
      bvhNormalBuffer: {} as GPUBuffer,
      bvhEmissiveTextureView: {} as GPUTextureView,
      bvhRoughMetalTextureView: {} as GPUTextureView,
      tlasNodesBuffer: {} as GPUBuffer,
      tlasInstanceIndicesBuffer: {} as GPUBuffer,
      tlasBlasRootsBuffer: {} as GPUBuffer,
      tlasInstanceWorldToLocalBuffer: {} as GPUBuffer,
      tlasInstanceLocalToWorldBuffer: {} as GPUBuffer,
      analyticLightsTextureView: {} as GPUTextureView,
      envMapTextureView: {} as GPUTextureView,
      envMarginalTextureView: {} as GPUTextureView,
      envConditionalTextureView: {} as GPUTextureView,
      envSampler: {} as GPUSampler,
      envParamsBuffer: {} as GPUBuffer,
    };
    const ddgi = {
      buildBindGroup: vi.fn(() => ({ label: 'hybrid' })),
      buildShadeBindGroup: vi.fn(() => ({ label: 'shade-hybrid' })),
    };
    const resources = {
      common: {
        hdrColorTexture: { createView: () => placeholderView },
        nearestSampler: {},
        gNormalDepthTexture: {},
        uboBuffer: {},
        tierTexture: { createView: () => placeholderView },
        compositeSampler: {},
        resolvedTexture: { createView: () => placeholderView },
      },
      restirDI: {
        reservoirCurrentBuffer: {},
        reservoirPreviousBuffer: {},
        reservoirSpatialBuffer: {},
      },
      restirGI: { reservoirGiCurrentBuffer: {} },
      gtao: { aoFullTexture: { createView: () => placeholderView } },
      svgf: { svgfCurrentObjectIdTexture: { createView: () => placeholderView } },
    } as never;

    const groups = buildPerFrameBindGroups(device, cache, resources, scene, ddgi as never, placeholderView);
    expect(groups.frame).toEqual({ label: 'frame' });
    expect(groups.scene).toEqual({ label: 'scene' });
    expect(groups.ubo).toEqual({ label: 'ubo' });
    expect(groups.hybridLayers).toEqual({ label: 'hybrid' });
    expect(ddgi.buildBindGroup).toHaveBeenCalledOnce();
  });

  it('buildCompositePresentBindGroup delegates to buildCompositeBindGroup', () => {
    const tex = { createView: vi.fn(() => ({})) } as unknown as GPUTexture;
    const bg = buildCompositePresentBindGroup({} as GPUDevice, {}, tex, {} as GPUSampler, {} as GPUBuffer);
    expect(bg).toEqual({ label: 'composite' });
    expect(tex.createView).toHaveBeenCalled();
  });

  it('reuses stable per-frame bind groups when resource identities are unchanged', () => {
    const device = {} as GPUDevice;
    const cache = {};
    const resourceCache = new PipelineResourceCache();
    const placeholderView = {} as GPUTextureView;
    const scene = {
      bvhNodesBuffer: {} as GPUBuffer,
      bvhIndexBuffer: {} as GPUBuffer,
      bvhPositionBuffer: {} as GPUBuffer,
      emitterBuffer: {} as GPUBuffer,
      emitterCdfBuffer: {} as GPUBuffer,
      bvhBeerTextureView: {} as GPUTextureView,
      bvhNormalBuffer: {} as GPUBuffer,
      bvhEmissiveTextureView: {} as GPUTextureView,
      bvhRoughMetalTextureView: {} as GPUTextureView,
      tlasNodesBuffer: {} as GPUBuffer,
      tlasInstanceIndicesBuffer: {} as GPUBuffer,
      tlasBlasRootsBuffer: {} as GPUBuffer,
      tlasInstanceWorldToLocalBuffer: {} as GPUBuffer,
      tlasInstanceLocalToWorldBuffer: {} as GPUBuffer,
      analyticLightsTextureView: {} as GPUTextureView,
      envMapTextureView: {} as GPUTextureView,
      envMarginalTextureView: {} as GPUTextureView,
      envConditionalTextureView: {} as GPUTextureView,
      envSampler: {} as GPUSampler,
      envParamsBuffer: {} as GPUBuffer,
    };
    const ddgi = {
      buildBindGroup: vi.fn(() => ({ label: 'hybrid' })),
      buildShadeBindGroup: vi.fn(() => ({ label: 'shade-hybrid' })),
    };
    const textureView = {} as GPUTextureView;
    const resources = {
      common: {
        hdrColorTexture: { createView: vi.fn(() => textureView) },
        nearestSampler: {},
        gNormalDepthTexture: { createView: vi.fn(() => textureView) },
        uboBuffer: {},
        tierTexture: { createView: vi.fn(() => textureView) },
        compositeSampler: {},
        resolvedTexture: { createView: vi.fn(() => textureView) },
        hdrIndirectTexture: { createView: vi.fn(() => textureView) },
        hdrTotalTexture: { createView: vi.fn(() => textureView) },
        albedoTexture: { createView: vi.fn(() => textureView) },
      },
      restirDI: {
        reservoirCurrentBuffer: {},
        reservoirPreviousBuffer: {},
        reservoirSpatialBuffer: {},
      },
      restirGI: { reservoirGiCurrentBuffer: {} },
      gtao: { aoFullTexture: { createView: vi.fn(() => textureView) } },
      svgf: { svgfCurrentObjectIdTexture: { createView: vi.fn(() => textureView) } },
    } as never;

    const first = buildPerFrameBindGroups(
      device,
      cache,
      resources,
      scene,
      ddgi as never,
      placeholderView,
      resourceCache,
    );
    const second = buildPerFrameBindGroups(
      device,
      cache,
      resources,
      scene,
      ddgi as never,
      placeholderView,
      resourceCache,
    );

    expect(second.frame).toBe(first.frame);
    expect(second.scene).toBe(first.scene);
    expect(second.ubo).toBe(first.ubo);
    expect(buildFrameBindGroup).toHaveBeenCalledTimes(1);
    expect(buildSceneBindGroup).toHaveBeenCalledTimes(1);
    expect(buildUboBindGroup).toHaveBeenCalledTimes(1);
    expect(ddgi.buildBindGroup).toHaveBeenCalledTimes(2);
  });

  it('rebuilds a cached per-frame group after a resource identity changes', () => {
    const device = {} as GPUDevice;
    const cache = {};
    const resourceCache = new PipelineResourceCache();
    const placeholderView = {} as GPUTextureView;
    const textureView = {} as GPUTextureView;
    const scene = {
      bvhNodesBuffer: {} as GPUBuffer,
      bvhIndexBuffer: {} as GPUBuffer,
      bvhPositionBuffer: {} as GPUBuffer,
      emitterBuffer: {} as GPUBuffer,
      emitterCdfBuffer: {} as GPUBuffer,
      bvhBeerTextureView: {} as GPUTextureView,
      bvhNormalBuffer: {} as GPUBuffer,
      bvhEmissiveTextureView: {} as GPUTextureView,
      bvhRoughMetalTextureView: {} as GPUTextureView,
      tlasNodesBuffer: {} as GPUBuffer,
      tlasInstanceIndicesBuffer: {} as GPUBuffer,
      tlasBlasRootsBuffer: {} as GPUBuffer,
      tlasInstanceWorldToLocalBuffer: {} as GPUBuffer,
      tlasInstanceLocalToWorldBuffer: {} as GPUBuffer,
      analyticLightsTextureView: {} as GPUTextureView,
      envMapTextureView: {} as GPUTextureView,
      envMarginalTextureView: {} as GPUTextureView,
      envConditionalTextureView: {} as GPUTextureView,
      envSampler: {} as GPUSampler,
      envParamsBuffer: {} as GPUBuffer,
    };
    const ddgi = {
      buildBindGroup: vi.fn(() => ({ label: 'hybrid' })),
      buildShadeBindGroup: vi.fn(() => ({ label: 'shade-hybrid' })),
    };
    const base = {
      common: {
        hdrColorTexture: { createView: vi.fn(() => textureView) },
        nearestSampler: {},
        gNormalDepthTexture: { createView: vi.fn(() => textureView) },
        uboBuffer: {},
        tierTexture: { createView: vi.fn(() => textureView) },
        compositeSampler: {},
        resolvedTexture: { createView: vi.fn(() => textureView) },
        hdrIndirectTexture: { createView: vi.fn(() => textureView) },
        hdrTotalTexture: { createView: vi.fn(() => textureView) },
        albedoTexture: { createView: vi.fn(() => textureView) },
      },
      restirDI: {
        reservoirCurrentBuffer: {},
        reservoirPreviousBuffer: {},
        reservoirSpatialBuffer: {},
      },
      restirGI: { reservoirGiCurrentBuffer: {} },
      gtao: { aoFullTexture: { createView: vi.fn(() => textureView) } },
      svgf: { svgfCurrentObjectIdTexture: { createView: vi.fn(() => textureView) } },
    };
    const resized = {
      ...base,
      common: {
        ...base.common,
        hdrColorTexture: { createView: vi.fn(() => textureView) },
      },
    };

    const first = buildPerFrameBindGroups(device, cache, base as never, scene, ddgi as never, placeholderView, resourceCache);
    const second = buildPerFrameBindGroups(device, cache, resized as never, scene, ddgi as never, placeholderView, resourceCache);

    expect(second.frame).not.toBe(first.frame);
    expect(second.scene).toBe(first.scene);
    expect(buildFrameBindGroup).toHaveBeenCalledTimes(2);
    expect(buildSceneBindGroup).toHaveBeenCalledTimes(1);
  });

  it('rebuilds a cached frame group after the SVGF current object-id texture changes', () => {
    const device = {} as GPUDevice;
    const cache = {};
    const resourceCache = new PipelineResourceCache();
    const placeholderView = {} as GPUTextureView;
    const textureView = {} as GPUTextureView;
    const scene = {
      bvhNodesBuffer: {} as GPUBuffer,
      bvhIndexBuffer: {} as GPUBuffer,
      bvhPositionBuffer: {} as GPUBuffer,
      emitterBuffer: {} as GPUBuffer,
      emitterCdfBuffer: {} as GPUBuffer,
      bvhBeerTextureView: {} as GPUTextureView,
      bvhNormalBuffer: {} as GPUBuffer,
      bvhEmissiveTextureView: {} as GPUTextureView,
      bvhRoughMetalTextureView: {} as GPUTextureView,
      tlasNodesBuffer: {} as GPUBuffer,
      tlasInstanceIndicesBuffer: {} as GPUBuffer,
      tlasBlasRootsBuffer: {} as GPUBuffer,
      tlasInstanceWorldToLocalBuffer: {} as GPUBuffer,
      tlasInstanceLocalToWorldBuffer: {} as GPUBuffer,
      analyticLightsTextureView: {} as GPUTextureView,
      envMapTextureView: {} as GPUTextureView,
      envMarginalTextureView: {} as GPUTextureView,
      envConditionalTextureView: {} as GPUTextureView,
      envSampler: {} as GPUSampler,
      envParamsBuffer: {} as GPUBuffer,
    };
    const ddgi = {
      buildBindGroup: vi.fn(() => ({ label: 'hybrid' })),
      buildShadeBindGroup: vi.fn(() => ({ label: 'shade-hybrid' })),
    };
    const objectIdA = { createView: vi.fn(() => textureView) };
    const objectIdB = { createView: vi.fn(() => textureView) };
    const base = {
      common: {
        hdrColorTexture: { createView: vi.fn(() => textureView) },
        nearestSampler: {},
        gNormalDepthTexture: { createView: vi.fn(() => textureView) },
        uboBuffer: {},
        tierTexture: { createView: vi.fn(() => textureView) },
        compositeSampler: {},
        resolvedTexture: { createView: vi.fn(() => textureView) },
        hdrIndirectTexture: { createView: vi.fn(() => textureView) },
        hdrTotalTexture: { createView: vi.fn(() => textureView) },
        albedoTexture: { createView: vi.fn(() => textureView) },
      },
      restirDI: {
        reservoirCurrentBuffer: {},
        reservoirPreviousBuffer: {},
        reservoirSpatialBuffer: {},
      },
      restirGI: { reservoirGiCurrentBuffer: {} },
      gtao: { aoFullTexture: { createView: vi.fn(() => textureView) } },
      svgf: { svgfCurrentObjectIdTexture: objectIdA },
    };
    const resized = {
      ...base,
      svgf: { svgfCurrentObjectIdTexture: objectIdB },
    };

    const first = buildPerFrameBindGroups(device, cache, base as never, scene, ddgi as never, placeholderView, resourceCache);
    const second = buildPerFrameBindGroups(device, cache, resized as never, scene, ddgi as never, placeholderView, resourceCache);

    expect(second.frame).not.toBe(first.frame);
    expect(second.scene).toBe(first.scene);
    expect(buildFrameBindGroup).toHaveBeenCalledTimes(2);
    expect(buildSceneBindGroup).toHaveBeenCalledTimes(1);
  });

  it('reuses composite-present bind groups and default texture views', () => {
    const resourceCache = new PipelineResourceCache();
    const tex = { createView: vi.fn(() => ({})) } as unknown as GPUTexture;
    const sampler = {} as GPUSampler;

    const compositeUbo = {} as GPUBuffer;
    const first = buildCompositePresentBindGroup({} as GPUDevice, {}, tex, sampler, compositeUbo, resourceCache);
    const second = buildCompositePresentBindGroup({} as GPUDevice, {}, tex, sampler, compositeUbo, resourceCache);

    expect(second).toBe(first);
    expect(tex.createView).toHaveBeenCalledTimes(1);
    expect(buildCompositeBindGroup).toHaveBeenCalledTimes(1);
  });
});
