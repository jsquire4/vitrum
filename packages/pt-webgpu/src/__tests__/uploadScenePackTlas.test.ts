import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import {
  buildPackedScene,
  CWBVH_ROOT_PAIR_WORDS,
  isValidCwbvhRootPair,
  rebuildTlasForSceneTransforms,
  uploadPackedScene,
  uploadScenePackTlasOnly,
} from '../scene/uploadSceneBuffers.js';
import { asMat4 } from '@vitrum/core';
import { installGpuConstStubs, textureStubMethods } from './gpuStub.js';
import { CWBVH_CHILD_COUNT_INVALID } from '@vitrum/shared-bvh';

function installWebGpuConstStubs(): void {
  installGpuConstStubs();
}

function twoMeshScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'mesh-a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
      },
      {
        kind: 'mesh',
        id: 'mesh-b',
        positions: new Float32Array([0, 0, 1, 1, 0, 1, 0, 1, 1]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
        transform: asMat4(new Float32Array([
          1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
        ])),
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('uploadScenePackTlasOnly', () => {
  it('allocates full BVH-node placeholders for primitive-less full-tier scenes', () => {
    installWebGpuConstStubs();
    const bufferDescs: GPUBufferDescriptor[] = [];
    const device = {
      queue: { writeBuffer: vi.fn(), writeTexture: vi.fn() },
      createBuffer: vi.fn((desc: GPUBufferDescriptor) => {
        bufferDescs.push(desc);
        return {
          label: desc.label,
          size: desc.size,
          destroy: vi.fn(),
        };
      }),
      ...textureStubMethods(),
      limits: { maxTextureDimension2D: 8192 },
    } as unknown as GPUDevice;

    const scene: Scene = {
      primitives: [],
      emitters: [],
      environment: {
        kind: 'hdri',
        hdri: {
          width: 1,
          height: 1,
          data: new Float32Array([1, 1, 1]),
        },
      },
    };
    const packed = buildPackedScene(scene);
    expect(packed.bvhNodes.byteLength).toBe(0);
    expect(packed.tlasNodes.byteLength).toBe(0);

    const uploaded = uploadPackedScene(device, packed);
    const sizesByLabel = new Map(bufferDescs.map((desc) => [desc.label, desc.size]));

    expect(uploaded.bvhNodesBuffer).toMatchObject({ size: 32 });
    expect(uploaded.tlasNodesBuffer).toMatchObject({ size: 32 });
    expect(sizesByLabel.get('vitrum.pt-webgpu.scene.bvhNodes')).toBe(32);
    expect(sizesByLabel.get('vitrum.pt-webgpu.scene.tlasNodes')).toBe(32);
  });

  it('writes only TLAS GPU buffers and CPU mirrors', () => {
    installWebGpuConstStubs();
    const writeBuffer = vi.fn();
    const device = {
      queue: { writeBuffer, writeTexture: vi.fn() },
      createBuffer: vi.fn((desc: GPUBufferDescriptor) => ({
        label: desc.label,
        size: desc.size,
        destroy: vi.fn(),
      })),
      ...textureStubMethods(),
      limits: { maxTextureDimension2D: 8192 },
    } as unknown as GPUDevice;

    const packed = buildPackedScene(twoMeshScene());
    const sb = uploadPackedScene(device, packed);
    const beforeRootPairs = new Uint32Array(sb.cwbvhTlasBlasRoots);
    writeBuffer.mockClear();

    const moved: Scene = {
      ...twoMeshScene(),
      primitives: twoMeshScene().primitives.map((p) =>
        p.id === 'mesh-b' && p.kind === 'mesh'
          ? {
              ...p,
              transform: asMat4(new Float32Array([
                1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1,
              ])),
            }
          : p,
      ),
    };
    const tlas = rebuildTlasForSceneTransforms(
      moved,
      sb.primitiveTlasBindings,
      {
        tlasNodes: sb.tlasNodes,
        tlasInstanceIndices: sb.tlasInstanceIndices,
        tlasBlasRoots: sb.tlasBlasRoots,
        tlasInstanceWorldToLocal: sb.tlasInstanceWorldToLocal,
      },
    );
    expect(tlas.ok).toBe(true);
    if (!tlas.ok) return;

    uploadScenePackTlasOnly(device, sb, {
      tlasNodes: tlas.tlasNodes,
      tlasInstanceIndices: tlas.tlasInstanceIndices,
      tlasBlasRoots: tlas.tlasBlasRoots,
      tlasInstanceWorldToLocal: tlas.tlasInstanceWorldToLocal,
      tlasInstanceLocalToWorld: tlas.tlasInstanceLocalToWorld,
      tlasNodeCount: Math.floor(tlas.tlasNodes.length / 8),
      primitiveTlasBindings: sb.primitiveTlasBindings,
    });

    // 5 binary TLAS writes + the CWBVH TLAS-root mirror.
    expect(writeBuffer).toHaveBeenCalledTimes(6);
    const labels = writeBuffer.mock.calls.map((c) => (c[0] as GPUBuffer).label ?? '');
    expect(labels.some((l) => l.includes('positions'))).toBe(false);
    expect(labels.some((l) => l.includes('bvhNodes'))).toBe(false);
    expect(labels.some((l) => l.includes('tlas'))).toBe(true);
    expect(labels.some((l) => l === 'vitrum.pt-webgpu.scene.cwbvhTlasBlasRoots')).toBe(true);
    expect(sb.cwbvhTlasBlasRoots).toEqual(beforeRootPairs);
    for (let i = 0; i < sb.tlasBlasRoots.length; i += 1) {
      const offset = i * CWBVH_ROOT_PAIR_WORDS;
      expect(isValidCwbvhRootPair(sb.cwbvhTlasBlasRoots, offset)).toBe(true);
      expect(sb.cwbvhTlasBlasRoots[offset + 1]).toBe(sb.tlasBlasRoots[i]);
      expect(sb.cwbvhTlasBlasRoots[offset + 2]).toBeLessThan(sb.cwbvhNodeCount);
    }
  });

  it('rolls binary TLAS writes back when hostile root remapping fails before publication', () => {
    installWebGpuConstStubs();
    const writeBuffer = vi.fn();
    const device = {
      queue: { writeBuffer, writeTexture: vi.fn() },
      createBuffer: vi.fn((desc: GPUBufferDescriptor) => ({
        label: desc.label,
        size: desc.size,
        destroy: vi.fn(),
      })),
      ...textureStubMethods(),
      limits: { maxTextureDimension2D: 8192 },
    } as unknown as GPUDevice;

    const packed = buildPackedScene(twoMeshScene());
    const sb = uploadPackedScene(device, packed);
    const before = {
      tlasNodes: new Uint32Array(sb.tlasNodes),
      tlasInstanceIndices: new Uint32Array(sb.tlasInstanceIndices),
      tlasBlasRoots: new Uint32Array(sb.tlasBlasRoots),
      cwbvhRoots: new Uint32Array(sb.cwbvhTlasBlasRoots),
    };
    const hostileRoot = packed.tlasBlasRoots[0]!;
    sb.cwbvhBinaryRootToWideRoot[hostileRoot] = CWBVH_CHILD_COUNT_INVALID;
    writeBuffer.mockClear();

    expect(() => uploadScenePackTlasOnly(device, sb, {
      tlasNodes: packed.tlasNodes,
      tlasInstanceIndices: packed.tlasInstanceIndices,
      tlasBlasRoots: packed.tlasBlasRoots,
      tlasInstanceWorldToLocal: packed.tlasInstanceWorldToLocal,
      tlasInstanceLocalToWorld: packed.tlasInstanceLocalToWorld,
      tlasNodeCount: Math.floor(packed.tlasNodes.length / 8),
      primitiveTlasBindings: packed.primitiveTlasBindings,
    })).toThrow(/no valid CWBVH root mapping/);

    expect(writeBuffer).toHaveBeenCalledTimes(11);
    const labels = writeBuffer.mock.calls.map((call) => (call[0] as GPUBuffer).label ?? '');
    expect(labels.filter((label) => label === 'vitrum.pt-webgpu.scene.cwbvhTlasBlasRoots')).toHaveLength(1);
    expect(labels.at(-1)).toBe('vitrum.pt-webgpu.scene.cwbvhTlasBlasRoots');
    expect(Array.from(sb.tlasNodes)).toEqual(Array.from(before.tlasNodes));
    expect(Array.from(sb.tlasInstanceIndices)).toEqual(Array.from(before.tlasInstanceIndices));
    expect(Array.from(sb.tlasBlasRoots)).toEqual(Array.from(before.tlasBlasRoots));
    expect(Array.from(sb.cwbvhTlasBlasRoots)).toEqual(Array.from(before.cwbvhRoots));
  });
});
