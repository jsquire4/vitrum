import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import {
  buildPackedScene,
  scenePackResultFromPacked,
  uploadPackedScene,
  uploadScenePackBlasOnly,
} from '../scene/uploadSceneBuffers.js';
import { rebuildPrimitiveBlas } from '@vitrum/shared-bvh';
import { asMat4 } from '@vitrum/core';

function installWebGpuConstStubs(): void {
  const g = globalThis as unknown as { GPUBufferUsage?: Record<string, number> };
  if (g.GPUBufferUsage == null) {
    g.GPUBufferUsage = { STORAGE: 1 << 0, COPY_DST: 1 << 1 };
  }
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

describe('uploadScenePackBlasOnly', () => {
  it('writes BLAS buffers but not TLAS GPU buffers', () => {
    installWebGpuConstStubs();
    const writeBuffer = vi.fn();
    const device = {
      queue: { writeBuffer },
      createBuffer: vi.fn((desc: GPUBufferDescriptor) => ({
        label: desc.label,
        destroy: vi.fn(),
      })),
    } as unknown as GPUDevice;

    const scene = twoMeshScene();
    const packed = buildPackedScene(scene);
    const sb = uploadPackedScene(device, packed);
    const geoPack = scenePackResultFromPacked(packed);
    writeBuffer.mockClear();

    const shifted = new Float32Array([0.05, 0, 0, 1.05, 0, 0, 0.05, 1, 0]);
    const nextScene: Scene = {
      ...scene,
      primitives: scene.primitives.map((p) =>
        p.id === 'mesh-a' && p.kind === 'mesh'
          ? { ...p, positions: shifted }
          : p,
      ),
    };
    const rebuilt = rebuildPrimitiveBlas(nextScene, 'mesh-a', geoPack, {
      tlas: true,
      resolveMaterialId: () => 0,
    });
    expect(rebuilt.ok).toBe(true);
    if (!rebuilt.ok) return;

    uploadScenePackBlasOnly(device, sb, rebuilt.pack);

    expect(writeBuffer).toHaveBeenCalledTimes(5);
    const labels = writeBuffer.mock.calls.map((c) => String((c[0] as GPUBuffer).label ?? ''));
    expect(labels.some((l) => l.includes('tlas'))).toBe(false);
    expect(labels.some((l) => l.includes('positions'))).toBe(true);
    expect(labels.some((l) => l.includes('bvhNodes'))).toBe(true);
  });
});
