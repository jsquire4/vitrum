import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import {
  buildPackedScene,
  rebuildTlasForSceneTransforms,
  uploadPackedScene,
  uploadScenePackTlasOnly,
} from '../scene/uploadSceneBuffers.js';
import { asMat4 } from '@vitrum/core';
import { installGpuConstStubs, textureStubMethods } from './gpuStub.js';

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
        destroy: vi.fn(),
      })),
      ...textureStubMethods(),
      limits: { maxTextureDimension2D: 8192 },
    } as unknown as GPUDevice;

    const packed = buildPackedScene(twoMeshScene());
    const sb = uploadPackedScene(device, packed);
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

    expect(writeBuffer).toHaveBeenCalledTimes(5);
    const labels = writeBuffer.mock.calls.map((c) => (c[0] as GPUBuffer).label ?? '');
    expect(labels.some((l) => l.includes('positions'))).toBe(false);
    expect(labels.some((l) => l.includes('bvhNodes'))).toBe(false);
    expect(labels.some((l) => l.includes('tlas'))).toBe(true);
  });
});
