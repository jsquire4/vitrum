import { describe, expect, it, vi } from 'vitest';
import { padTriangleIndicesToVec4 } from '../probeUpdateMaterials.js';
import { rebuildProbeBvhFromScene } from '../probeUpdateBvhBuffers.js';

describe('probeUpdateBvhBuffers', () => {
  it('padTriangleIndicesToVec4 pads stride-3 indices', () => {
    const out = padTriangleIndicesToVec4(new Uint32Array([0, 1, 2, 3, 4, 5]));
    expect(Array.from(out)).toEqual([0, 1, 2, 0, 3, 4, 5, 0]);
  });

  it('rebuildProbeBvhFromScene replaces BVH buffers', () => {
    const destroyed: GPUBuffer[] = [];
    const device = {
      createBuffer: vi.fn(() => ({
        destroy: () => {
          /* tracked via destroyed length */
        },
      })),
      queue: { writeBuffer: vi.fn() },
    } as unknown as GPUDevice;
    const old = {
      destroy: () => destroyed.push(old as unknown as GPUBuffer),
    } as unknown as GPUBuffer;
    const g = {
      bvhBuf: old,
      posBuf: old,
      idxBuf: old,
      normBuf: old,
      matIdBuf: old,
      tlasNodesBuf: old,
      tlasInstIdxBuf: old,
      tlasBlasRootsBuf: old,
      tlasW2lBuf: old,
      tlasL2wBuf: old,
    };
    const buffers = {
      bvhNodes: { buffer: new ArrayBuffer(32) },
      positions: { buffer: new ArrayBuffer(32) },
      indices: new Uint32Array([0, 1, 2]),
      normals: { buffer: new ArrayBuffer(32) },
      triMaterialId: { buffer: new ArrayBuffer(16) },
    } as never;
    rebuildProbeBvhFromScene(device, g, buffers);
    expect(device.createBuffer).toHaveBeenCalled();
    expect(g.bvhBuf).not.toBe(old);
  });
});
