import { describe, expect, it, vi } from 'vitest';
import { padTriangleIndicesToVec4 } from '../probeUpdateMaterials.js';
import {
  rebuildProbeBvhFromRestir,
  rebuildProbeBvhFromScene,
} from '../probeUpdateBvhBuffers.js';

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
      destroy: () => destroyed.push(old),
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

  it('rebuildProbeBvhFromRestir uses a BVHNode-sized TLAS placeholder when TLAS is absent', () => {
    const created: GPUBufferDescriptor[] = [];
    const device = {
      createBuffer: vi.fn((desc: GPUBufferDescriptor) => {
        created.push(desc);
        return { size: desc.size, destroy: vi.fn() } as unknown as GPUBuffer;
      }),
      queue: { writeBuffer: vi.fn() },
    } as unknown as GPUDevice;
    const old = {
      destroy: vi.fn(),
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
    const snap = {
      bvhNodes: new ArrayBuffer(32),
      positions: new ArrayBuffer(64),
      bvhIndex: new ArrayBuffer(32),
      normals: new ArrayBuffer(64),
      triMaterialIds: new ArrayBuffer(16),
    } as never;

    rebuildProbeBvhFromRestir(device, g, snap);

    expect(created[5]?.size).toBe(32);
    expect(created[6]?.size).toBe(16);
    expect(created[7]?.size).toBe(16);
    expect(created[8]?.size).toBe(16);
    expect(created[9]?.size).toBe(16);
  });
});
