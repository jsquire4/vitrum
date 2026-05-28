import { describe, expect, it, vi } from 'vitest';

vi.mock('../bindGroupBuilders.js', () => ({
  buildFrameBindGroup: vi.fn(() => ({ label: 'frame' })),
  buildSceneBindGroup: vi.fn(() => ({ label: 'scene' })),
  buildUboBindGroup: vi.fn(() => ({ label: 'ubo' })),
  buildCompositeBindGroup: vi.fn(() => ({ label: 'composite' })),
}));

import {
  buildCompositePresentBindGroup,
  buildPerFrameBindGroups,
} from '../pipelineBindGroupFactory.js';

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
      bvhBeerBuffer: {} as GPUBuffer,
      tlasNodesBuffer: {} as GPUBuffer,
      tlasInstanceIndicesBuffer: {} as GPUBuffer,
      tlasBlasRootsBuffer: {} as GPUBuffer,
      tlasInstanceWorldToLocalBuffer: {} as GPUBuffer,
      tlasInstanceLocalToWorldBuffer: {} as GPUBuffer,
    };
    const ddgi = {
      buildBindGroup: vi.fn(() => ({ label: 'hybrid' })),
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
    const bg = buildCompositePresentBindGroup({} as GPUDevice, {}, tex, {} as GPUSampler);
    expect(bg).toEqual({ label: 'composite' });
    expect(tex.createView).toHaveBeenCalled();
  });
});
