import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../bindGroupBuilders.js', () => ({
  buildFrameBindGroup: vi.fn((
    _device: unknown,
    _cache: unknown,
    resources: { reservoirSpatialBuffer: GPUBuffer },
  ) => ({
    label: 'frame',
    // buildFrameBindGroup maps reservoirSpatialBuffer to @binding(7).
    binding7Buffer: resources.reservoirSpatialBuffer,
  })),
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

function makeSceneResources() {
  return {
    sceneStorageArenaBuffers: [
      {} as GPUBuffer,
      {} as GPUBuffer,
      {} as GPUBuffer,
    ] as const,
    bvhBeerTextureView: {} as GPUTextureView,
    bvhEmissiveTextureView: {} as GPUTextureView,
    bvhRoughMetalTextureView: {} as GPUTextureView,
    materialTextureAtlasView: {} as GPUTextureView,
    baseColorMapMetaTextureView: {} as GPUTextureView,
    bvhTangentTextureView: {} as GPUTextureView,
    bvhVertexColorTextureView: {} as GPUTextureView,
    analyticLightsTextureView: {} as GPUTextureView,
    envMapTextureView: {} as GPUTextureView,
    envMarginalTextureView: {} as GPUTextureView,
    envConditionalTextureView: {} as GPUTextureView,
    envPdfTextureView: {} as GPUTextureView,
    envParamsBuffer: {} as GPUBuffer,
  };
}

describe('pipelineBindGroupFactory', () => {
  it('buildPerFrameBindGroups wires frame, scene, ubo, and hybrid layers', () => {
    const device = {} as GPUDevice;
    const cache = {};
    const placeholderView = {} as GPUTextureView;
    const scene = makeSceneResources();
    const ddgi = {
      buildBindGroup: vi.fn(() => ({ label: 'hybrid' })),
      buildShadeBindGroup: vi.fn(() => ({ label: 'shade-hybrid' })),
    };
    const reservoirCurrentBuffer = { label: 'di-current' } as GPUBuffer;
    const reservoirPreviousBuffer = { label: 'di-previous' } as GPUBuffer;
    const reservoirSpatialBuffer = { label: 'di-spatial' } as GPUBuffer;
    const resources = {
      common: {
        hdrColorTexture: { createView: () => placeholderView },
        nearestSampler: {},
        gNormalDepthTexture: {},
        uboBuffer: {},
        tierTexture: { createView: () => placeholderView },
        resolvedTexture: { createView: () => placeholderView },
      },
      restirDI: {
        reservoirCurrentBuffer,
        reservoirPreviousBuffer,
        reservoirSpatialBuffer,
      },
      restirGI: { reservoirGiCurrentBuffer: {} },
      gtao: { aoFullTexture: { createView: () => placeholderView } },
      svgf: { svgfCurrentObjectIdTexture: { createView: () => placeholderView } },
    } as never;

    const groups = buildPerFrameBindGroups(device, cache, resources, scene, ddgi as never);
    expect(groups.frame).toMatchObject({
      label: 'frame',
      binding7Buffer: reservoirSpatialBuffer,
    });
    expect(groups.diSpatialReverse).toMatchObject({
      label: 'frame',
      binding7Buffer: reservoirCurrentBuffer,
    });
    expect(buildFrameBindGroup).toHaveBeenNthCalledWith(
      1,
      device,
      cache,
      expect.objectContaining({
        reservoirCurrentBuffer,
        reservoirPreviousBuffer,
        reservoirSpatialBuffer,
      }),
      undefined,
    );
    expect(buildFrameBindGroup).toHaveBeenNthCalledWith(
      2,
      device,
      cache,
      expect.objectContaining({
        reservoirCurrentBuffer: reservoirSpatialBuffer,
        reservoirPreviousBuffer,
        reservoirSpatialBuffer: reservoirCurrentBuffer,
      }),
      undefined,
    );
    expect(groups.scene).toEqual({ label: 'scene' });
    expect(groups.ubo).toEqual({ label: 'ubo' });
    expect(groups.hybridLayers).toEqual({ label: 'hybrid' });
    expect(ddgi.buildBindGroup).toHaveBeenCalledOnce();
    expect(buildRisGiFrameBindGroup).toHaveBeenCalledWith(
      device,
      cache,
      expect.anything(),
    );
  });

  it('buildCompositePresentBindGroup delegates to buildCompositeBindGroup', () => {
    const tex = { createView: vi.fn(() => ({})) } as unknown as GPUTexture;
    const bg = buildCompositePresentBindGroup({} as GPUDevice, {}, tex, {} as GPUBuffer);
    expect(bg).toEqual({ label: 'composite' });
    expect(tex.createView).toHaveBeenCalled();
  });

  it('reuses stable per-frame bind groups when resource identities are unchanged', () => {
    const device = {} as GPUDevice;
    const cache = {};
    const resourceCache = new PipelineResourceCache();
    const scene = makeSceneResources();
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
      resourceCache,
    );
    const second = buildPerFrameBindGroups(
      device,
      cache,
      resources,
      scene,
      ddgi as never,
      resourceCache,
    );

    expect(second.frame).toBe(first.frame);
    expect(second.scene).toBe(first.scene);
    expect(second.ubo).toBe(first.ubo);
    expect(buildFrameBindGroup).toHaveBeenCalledTimes(2);
    expect(buildSceneBindGroup).toHaveBeenCalledTimes(1);
    expect(buildUboBindGroup).toHaveBeenCalledTimes(1);
    expect(ddgi.buildBindGroup).toHaveBeenCalledTimes(2);
  });

  it('rebuilds a cached per-frame group after a resource identity changes', () => {
    const device = {} as GPUDevice;
    const cache = {};
    const resourceCache = new PipelineResourceCache();
    const textureView = {} as GPUTextureView;
    const scene = makeSceneResources();
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

    const first = buildPerFrameBindGroups(device, cache, base as never, scene, ddgi as never, resourceCache);
    const second = buildPerFrameBindGroups(device, cache, resized as never, scene, ddgi as never, resourceCache);

    expect(second.frame).not.toBe(first.frame);
    expect(second.scene).toBe(first.scene);
    expect(buildFrameBindGroup).toHaveBeenCalledTimes(4);
    expect(buildSceneBindGroup).toHaveBeenCalledTimes(1);
  });

  it('rebuilds a cached frame group after the SVGF current object-id texture changes', () => {
    const device = {} as GPUDevice;
    const cache = {};
    const resourceCache = new PipelineResourceCache();
    const textureView = {} as GPUTextureView;
    const scene = makeSceneResources();
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

    const first = buildPerFrameBindGroups(device, cache, base as never, scene, ddgi as never, resourceCache);
    const second = buildPerFrameBindGroups(device, cache, resized as never, scene, ddgi as never, resourceCache);

    expect(second.frame).not.toBe(first.frame);
    expect(second.scene).toBe(first.scene);
    expect(buildFrameBindGroup).toHaveBeenCalledTimes(4);
    expect(buildSceneBindGroup).toHaveBeenCalledTimes(1);
  });

  it('reuses composite-present bind groups and default texture views', () => {
    const resourceCache = new PipelineResourceCache();
    const tex = { createView: vi.fn(() => ({})) } as unknown as GPUTexture;

    const compositeUbo = {} as GPUBuffer;
    const first = buildCompositePresentBindGroup({} as GPUDevice, {}, tex, compositeUbo, resourceCache);
    const second = buildCompositePresentBindGroup({} as GPUDevice, {}, tex, compositeUbo, resourceCache);

    expect(second).toBe(first);
    expect(tex.createView).toHaveBeenCalledTimes(1);
    expect(buildCompositeBindGroup).toHaveBeenCalledTimes(1);
  });
});
