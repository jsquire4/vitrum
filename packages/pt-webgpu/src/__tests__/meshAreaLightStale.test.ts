/**
 * meshAreaLightStale.test.ts — H11 stale mesh-area emitter triangles test.
 *
 * Verifies that when a geometry/transform fast path moves the world-space
 * positions of a primitive backed by a mesh-area emitter, the emitter arrays
 * are re-packed and re-uploaded so the GPU NEE data stays in sync.
 *
 * The test uses `hasMeshAreaEmitterForPrimitive` (the H11 gate helper) and
 * `packEmitterArrays` (to build expected world-space triangle data) and checks
 * that `SceneMutationRouter.updatePrimitive` triggers a re-upload of the
 * meshAreaLightsBuffer when the primitive's positions change.
 */

import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { hasMeshAreaEmitterForPrimitive } from '../scene/emitterPacking.js';
import { buildPackedScene, scenePackResultFromPacked } from '../scene/uploadSceneBuffers.js';
import { SceneMutationRouter } from '../sceneMutationRouter.js';
import type { MutationHost } from '../sceneMutationRouter.js';
import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';

// ─── helpers ──────────────────────────────────────────────────────────────────

function emitterScene(positions: Float32Array): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'emitter-panel',
        positions,
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
      },
    ],
    emitters: [
      {
        kind: 'mesh-area' as const,
        id: 'area',
        meshId: 'emitter-panel',
        color: [1, 1, 1],
        intensity: 1,
      },
    ],
    environment: { kind: 'none' },
  };
}

function sceneWithoutEmitter(positions: Float32Array): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'plain-mesh',
        positions,
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

// ─── hasMeshAreaEmitterForPrimitive unit tests ────────────────────────────────

describe('hasMeshAreaEmitterForPrimitive (H11 gate helper)', () => {
  it('returns true when the primitive backs a mesh-area emitter', () => {
    const scene = emitterScene(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]));
    expect(hasMeshAreaEmitterForPrimitive(scene, 'emitter-panel')).toBe(true);
  });

  it('returns false for a primitive with no emitter backing', () => {
    const scene = emitterScene(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]));
    expect(hasMeshAreaEmitterForPrimitive(scene, 'some-other-id')).toBe(false);
  });

  it('returns false in a scene with no emitters', () => {
    const scene = sceneWithoutEmitter(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]));
    expect(hasMeshAreaEmitterForPrimitive(scene, 'plain-mesh')).toBe(false);
  });
});

// ─── SceneMutationRouter — geometry patch triggers emitter re-pack ────────────

function makeHostWithEmitterScene(scene: Scene): {
  host: MutationHost;
  sceneRef: { current: Scene };
  meshAreaLightsData: Float32Array;
  meshAreaLightsWriteCalls: Float32Array[];
} {
  const packed = buildPackedScene(scene, {});
  // Extract a real geoPack so the geometry fast paths (which gate on `geoPack != null`) fire.
  const geoPack = scenePackResultFromPacked(packed);
  // Mutable copy so the H11 logic can update it in place.
  const meshAreaLightsData = new Float32Array(packed.meshAreaLightsData);

  const meshAreaLightsWriteCalls: Float32Array[] = [];

  // Stub GPU buffer handle for meshAreaLights — only writeBuffer matters here.
  const meshAreaLightsBuffer = {
    size: Math.max(16, meshAreaLightsData.byteLength),
    destroy: vi.fn(),
  } as unknown as GPUBuffer;

  // Minimal sceneBuffers stub — H11 only reads/writes the emitter-related fields
  // and their GPU buffer handles.
  const sceneBuffers = {
    ...packed,
    meshAreaLightsData,
    meshAreaLightsBuffer,
    pointLightsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
    spotLightsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
    rectAreaLightsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
    lightTreeBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
    bvhNodeCount: Math.floor(packed.bvhNodes.length / 8),
    tlasNodeCount: Math.floor(packed.tlasNodes.length / 8),
    materialCount: 1,
    positionsBuffer: { size: Math.max(16, packed.positions.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
    normalsBuffer: { size: Math.max(16, packed.normals.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
    indicesBuffer: { size: Math.max(16, packed.indices.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
    triMaterialIdsBuffer: { size: Math.max(16, packed.triMaterialIds.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
    materialsBuffer: { size: Math.max(16, packed.materials.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
    bvhNodesBuffer: { size: Math.max(16, packed.bvhNodes.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
    analyticHeadersBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
    analyticParamsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
    analyticLocalToWorldBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
    analyticWorldToLocalBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
    environmentMapTexelsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
    environmentMapCdfBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
    tlasNodesBuffer: { size: Math.max(16, packed.tlasNodes.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
    tlasInstanceIndicesBuffer: { size: Math.max(16, packed.tlasInstanceIndices.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
    tlasBlasRootsBuffer: { size: Math.max(16, packed.tlasBlasRoots.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
    tlasInstanceWorldToLocalBuffer: { size: Math.max(16, packed.tlasInstanceWorldToLocal.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
    tlasInstanceLocalToWorldBuffer: { size: Math.max(16, packed.tlasInstanceLocalToWorld.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
    uvsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
    materialTexDescriptorsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
    materialTexture: {} as GPUTexture,
    materialTextureView: {} as GPUTextureView,
    materialTextureSampler: {} as GPUSampler,
    materialLinearTexture: {} as GPUTexture,
    materialLinearTextureView: {} as GPUTextureView,
    gpuMemoryBytes: () => ({ bufferBytes: 0, textureBytesByFormat: {} }),
    destroy: vi.fn(),
  } as unknown as UploadedSceneBuffers;

  const sceneRef = { current: scene };

  const host: MutationHost = {
    device: {
      queue: {
        writeBuffer: vi.fn(
          (buf: unknown, byteOffset: number, data: ArrayBuffer, srcOffset: number, length: number) => {
            // Capture writes to the mesh-area lights buffer only.
            if (buf === meshAreaLightsBuffer) {
              meshAreaLightsWriteCalls.push(new Float32Array(data, srcOffset, length / 4));
            }
            // Also update the in-memory mirror directly (simulate GPU upload).
            if (buf === meshAreaLightsBuffer && byteOffset === 0) {
              const src = new Float32Array(data, srcOffset, length / 4);
              meshAreaLightsData.set(src);
            }
          },
        ),
      },
    } as unknown as GPUDevice,
    assertLive: vi.fn(),
    getScene: () => sceneRef.current,
    setSceneState: vi.fn((s: Scene) => { sceneRef.current = s; }),
    getSceneBuffers: () => sceneBuffers,
    getGeoPack: () => geoPack,
    setGeoPack: vi.fn(),
    invalidateBindGroups: vi.fn(),
    supportedAnalyticShapes: () => new Set<string>(),
    cameraVisibleEmitters: () => false,
    // Fall-through setScene: full repack is NOT called in these fast-path tests.
    repackScene: vi.fn((s: Scene) => { sceneRef.current = s; }),
    setScene: vi.fn((s: Scene) => { sceneRef.current = s; }),
    reset: vi.fn(),
  };

  return { host, sceneRef, meshAreaLightsData, meshAreaLightsWriteCalls };
}

describe('SceneMutationRouter — H11 mesh-area emitter triangle staleness', () => {
  it('positions patch on emitter-backed mesh → meshAreaLightsBuffer carries new world-space triangles', () => {
    const origPositions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const scene = emitterScene(origPositions);
    const { host, meshAreaLightsWriteCalls } = makeHostWithEmitterScene(scene);

    // After initial pack, the mesh-area triangles hold the original positions.
    const router = new SceneMutationRouter(host);

    // Patch the positions — moves the triangle to Z=5.
    const newPositions = new Float32Array([0, 0, 5, 1, 0, 5, 0, 1, 5]);
    router.updatePrimitive('emitter-panel', {
      positions: newPositions,
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    });

    // At least one write to meshAreaLightsBuffer must have occurred.
    expect(meshAreaLightsWriteCalls.length).toBeGreaterThan(0);

    // The written data should contain the new Z=5 world-space position.
    // The first triangle vertex in the packed mesh-area array is at float 0..2.
    const written = meshAreaLightsWriteCalls[meshAreaLightsWriteCalls.length - 1]!;
    // meshAreaLightsBuffer layout: stride 16 floats per triangle:
    //   [vertex A (vec4f)] [vertex B (vec4f)] [vertex C (vec4f)] [radiance (vec4f)]
    // vertex A.z is at float index 2.
    const vertAz = written[2];
    expect(vertAz).toBeCloseTo(5, 4);
  });

  it('non-emitter-backed mesh position patch does NOT trigger emitter re-upload', () => {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const scene = sceneWithoutEmitter(positions);
    const { host, meshAreaLightsWriteCalls } = makeHostWithEmitterScene(scene);

    const router = new SceneMutationRouter(host);
    const newPositions = new Float32Array([0, 0, 5, 1, 0, 5, 0, 1, 5]);
    router.updatePrimitive('plain-mesh', {
      positions: newPositions,
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    });

    // No emitter re-upload should occur (meshAreaLightCount=0 so no buffer to write).
    expect(meshAreaLightsWriteCalls.length).toBe(0);
  });
});
