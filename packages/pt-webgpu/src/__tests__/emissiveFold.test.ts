/**
 * emissiveFold.test.ts — H10 emissive-fold desync tests.
 *
 * Verifies that the fold (emitter color * intensity re-attached to the
 * primitive's material) is:
 *   1. Applied by `buildPackedScene` with `cameraVisibleEmitters: true`.
 *   2. Preserved after `updateEmitter` patches the emitter color.
 *   3. Preserved after `updatePrimitive` patches a non-emissive material field
 *      (roughness) via the material fast path.
 *   4. Applied to InverseSession stubs (future) — covered by the packFoldedMaterialEntry
 *      unit test checking both fold and no-fold paths.
 *
 * The emissive channel lives at float offset 4..6 of the packed material
 * (vec4 #1: emissive.rgb + metallic — see materialPacking.ts line 14).
 */

import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { buildPackedScene, packFoldedMaterialEntry } from '../scene/uploadSceneBuffers.js';
import { MATERIAL_FLOAT_STRIDE } from '../scene/materialPacking.js';
import { SceneMutationRouter } from '../sceneMutationRouter.js';
import type { MutationHost } from '../sceneMutationRouter.js';
import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';
import { installGpuConstStubs } from './gpuStub.js';

installGpuConstStubs();

// ─── helpers ──────────────────────────────────────────────────────────────────

/** Smallest possible scene: one triangle mesh + one mesh-area emitter. */
function meshEmitterScene(
  emitterColor: [number, number, number],
  emitterIntensity: number,
): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'panel',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.8, metallic: 0,
                    emissive: [0, 0, 0], emissiveIntensity: 0 },
      },
    ],
    emitters: [
      {
        kind: 'mesh-area' as const,
        id: 'light',
        meshId: 'panel',
        color: emitterColor,
        intensity: emitterIntensity,
      },
    ],
    environment: { kind: 'none' },
  };
}

function implicitEmissiveScene(
  emissive: [number, number, number],
  emissiveIntensity: number,
): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'panel',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: {
          baseColor: [0.5, 0.5, 0.5],
          roughness: 0.8,
          metallic: 0,
          emissive,
          emissiveIntensity,
        },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function movableMeshEmitterScene(): Scene {
  const triangle = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'panel-a',
        positions: triangle,
        normals,
        material: {
          baseColor: [0.5, 0.5, 0.5],
          roughness: 0.8,
          metallic: 0,
          emissive: [0.25, 0, 0],
          emissiveIntensity: 2,
        },
      },
      {
        kind: 'mesh',
        id: 'panel-b',
        positions: triangle,
        normals,
        material: {
          baseColor: [0.5, 0.5, 0.5],
          roughness: 0.8,
          metallic: 0,
          emissive: [0, 0.5, 0],
          emissiveIntensity: 3,
        },
      },
    ],
    emitters: [{
      kind: 'mesh-area',
      id: 'light',
      meshId: 'panel-a',
      color: [1, 1, 1],
      intensity: 4,
    }],
    environment: { kind: 'none' },
  };
}

/** Float offset of emissive.rgb in a packed material. */
const EMISSIVE_OFFSET = 4; // vec4 #1: emissive.rgb + metallic
const MESH_AREA_RADIANCE_OFFSET = 12; // first packed mesh-area triangle, vec4 #3

function emissiveFromMaterials(materials: Float32Array): [number, number, number] {
  return [materials[EMISSIVE_OFFSET]!, materials[EMISSIVE_OFFSET + 1]!, materials[EMISSIVE_OFFSET + 2]!];
}

// ─── packFoldedMaterialEntry unit tests ───────────────────────────────────────

describe('packFoldedMaterialEntry (H10 helper)', () => {
  const prim = {
    id: 'panel',
    material: { baseColor: [0.5, 0.5, 0.5] as [number,number,number], roughness: 0.8, metallic: 0,
                emissive: [0, 0, 0] as [number,number,number], emissiveIntensity: 0 },
  };

  const scene = meshEmitterScene([2, 0.5, 0.1], 3);

  it('applies the fold when cameraVisibleEmitters=true', () => {
    const packed = packFoldedMaterialEntry(prim, scene, true);
    expect(packed.length).toBe(MATERIAL_FLOAT_STRIDE);
    // Emissive = color * intensity = [2*3, 0.5*3, 0.1*3] = [6, 1.5, 0.3]
    expect(packed[EMISSIVE_OFFSET]).toBeCloseTo(6, 5);
    expect(packed[EMISSIVE_OFFSET + 1]).toBeCloseTo(1.5, 5);
    expect(packed[EMISSIVE_OFFSET + 2]).toBeCloseTo(0.3, 5);
  });

  it('folds camera-visible radiance with the exact staged Float32 product', () => {
    const authored = 1.0000002;
    const intensity = 1.0000002;
    const directBinary64Product = Math.fround(authored * intensity);
    const expected = Math.fround(
      Math.fround(authored) * Math.fround(intensity),
    );
    expect(expected).not.toBe(directBinary64Product);
    const stagedScene = meshEmitterScene(
      [authored, authored, authored],
      intensity,
    );
    const packed = packFoldedMaterialEntry(prim, stagedScene, true);
    expect(packed.slice(EMISSIVE_OFFSET, EMISSIVE_OFFSET + 3)).toEqual([
      expected,
      expected,
      expected,
    ]);
  });

  it('does NOT apply the fold when cameraVisibleEmitters=false', () => {
    const packed = packFoldedMaterialEntry(prim, scene, false);
    expect(packed[EMISSIVE_OFFSET]).toBeCloseTo(0, 5);
    expect(packed[EMISSIVE_OFFSET + 1]).toBeCloseTo(0, 5);
    expect(packed[EMISSIVE_OFFSET + 2]).toBeCloseTo(0, 5);
  });

  it('zeros an explicit emitter-owned material when visibility is off, preventing a mismatched hit integrand', () => {
    const mismatched = meshEmitterScene([2, 0.5, 0.1], 3);
    const primitive = mismatched.primitives[0]!;
    if (primitive.kind === 'analytic') throw new Error('test fixture must be a mesh');
    const authored = {
      ...primitive,
      material: {
        ...primitive.material,
        emissive: [9, 8, 7] as [number, number, number],
        emissiveIntensity: 4,
        emissiveMap: { handle: { width: 1, height: 1, data: new Float32Array([0.25, 0.5, 1, 1]) } },
      },
    };
    const sceneWithMismatch: Scene = { ...mismatched, primitives: [authored] };

    const off = packFoldedMaterialEntry(authored, sceneWithMismatch, false);
    expect(off.slice(EMISSIVE_OFFSET, EMISSIVE_OFFSET + 3)).toEqual([0, 0, 0]);

    const on = packFoldedMaterialEntry(authored, sceneWithMismatch, true);
    expect(on[EMISSIVE_OFFSET]).toBeCloseTo(6, 5);
    expect(on[EMISSIVE_OFFSET + 1]).toBeCloseTo(1.5, 5);
    expect(on[EMISSIVE_OFFSET + 2]).toBeCloseTo(0.3, 5);

    const offScene = buildPackedScene(sceneWithMismatch, { cameraVisibleEmitters: false });
    expect(emissiveFromMaterials(offScene.materials)).toEqual([0, 0, 0]);
    // NEE remains owned by the explicit emitter and carries its base radiance;
    // exact texture evaluation occurs in sampleMeshAreaLightRadiance on the GPU.
    expect(offScene.meshAreaLightsData[24]).toBeCloseTo(6, 5);
    expect(offScene.meshAreaLightsData[25]).toBeCloseTo(1.5, 5);
    expect(offScene.meshAreaLightsData[26]).toBeCloseTo(0.3, 5);
  });

  it('rejects multiple explicit mesh-area emitters that claim the same primitive', () => {
    const duplicateScene: Scene = {
      ...scene,
      emitters: [
        ...scene.emitters,
        { kind: 'mesh-area', id: 'light-duplicate', meshId: 'panel', color: [0, 1, 0], intensity: 2 },
      ],
    };
    expect(() => packFoldedMaterialEntry(prim, duplicateScene, true)).toThrow(/multiple mesh-area emitters/);
    expect(() => buildPackedScene(duplicateScene, { cameraVisibleEmitters: false })).toThrow(
      /multiple mesh-area emitters/,
    );
  });

  it('does NOT apply the fold when the primitive has no mesh-area emitter backing', () => {
    const otherPrim = { id: 'other', material: prim.material };
    const packed = packFoldedMaterialEntry(otherPrim, scene, true);
    expect(packed[EMISSIVE_OFFSET]).toBeCloseTo(0, 5);
  });
});

// ─── buildPackedScene fold tests ─────────────────────────────────────────────

describe('buildPackedScene emissive fold (H10)', () => {
  it('fold present after setScene with cameraVisibleEmitters:true', () => {
    const scene = meshEmitterScene([1, 0.5, 0.25], 4);
    const packed = buildPackedScene(scene, { cameraVisibleEmitters: true });
    // mat slot 0 (first primitive = 'panel')
    const em = emissiveFromMaterials(packed.materials);
    expect(em[0]).toBeCloseTo(4, 5);     // 1 * 4
    expect(em[1]).toBeCloseTo(2, 5);     // 0.5 * 4
    expect(em[2]).toBeCloseTo(1, 5);     // 0.25 * 4
  });

  it('fold absent when cameraVisibleEmitters is off (default)', () => {
    const scene = meshEmitterScene([1, 0.5, 0.25], 4);
    const packed = buildPackedScene(scene, { cameraVisibleEmitters: false });
    const em = emissiveFromMaterials(packed.materials);
    expect(em[0]).toBeCloseTo(0, 5);
    expect(em[1]).toBeCloseTo(0, 5);
    expect(em[2]).toBeCloseTo(0, 5);
  });
});

// ─── SceneMutationRouter fold-preservation tests ─────────────────────────────

/** Build a minimal MutationHost with an in-memory scene + materials buffer. */
function makeHostWithScene(scene: Scene, cameraVisible: boolean): {
  host: MutationHost;
  sceneRef: { current: Scene };
  sceneBuffers: UploadedSceneBuffers;
  materialsBuffer: Float32Array;
  meshAreaLightsData: Float32Array;
  writeCalls: Array<{ byteOffset: number; data: Float32Array }>;
  stagedCopy: {
    copyBufferToBuffer: ReturnType<typeof vi.fn>;
    submit: ReturnType<typeof vi.fn>;
  };
} {
  const packed = buildPackedScene(scene, { cameraVisibleEmitters: cameraVisible });
  const materialsBuffer = new Float32Array(packed.materials);
  const meshAreaLightsData = new Float32Array(packed.meshAreaLightsData);
  const writeCalls: Array<{ byteOffset: number; data: Float32Array }> = [];
  const copyBufferToBuffer = vi.fn();
  const submit = vi.fn();

  // Stub UploadedSceneBuffers — only the materials fields matter for H10 tests.
  const sceneBuffers = {
    ...packed,
    materials: materialsBuffer,
    meshAreaLightsData,
    materialsBuffer: {
      // Capture writes so we can assert the correct bytes were uploaded.
      destroy: vi.fn(),
    } as unknown as GPUBuffer,
    bvhNodeCount: 0,
    tlasNodeCount: 0,
    materialCount: packed.materials.length / MATERIAL_FLOAT_STRIDE,
    // GPU buffer stubs — not used by the material fast path.
    positionsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    normalsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    indicesBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    triMaterialIdsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    bvhNodesBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    analyticHeadersBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    analyticParamsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    analyticLocalToWorldBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    analyticWorldToLocalBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    environmentMapTexelsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    environmentMapCdfBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    pointLightsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    spotLightsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    rectAreaLightsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    meshAreaLightsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    lightTreeBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    tlasNodesBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    tlasInstanceIndicesBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    tlasBlasRootsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    tlasInstanceWorldToLocalBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    tlasInstanceLocalToWorldBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    uvsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
    materialTexDescriptorsBuffer: { destroy: vi.fn() } as unknown as GPUBuffer,
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
      createBuffer: vi.fn((desc: GPUBufferDescriptor) => ({
        label: desc.label,
        size: Number(desc.size),
        destroy: vi.fn(),
      })),
      createCommandEncoder: vi.fn(() => ({
        copyBufferToBuffer,
        finish: vi.fn(() => ({})),
      })),
      queue: {
        writeBuffer: vi.fn(
          (_buf: unknown, byteOffset: number, buffer: ArrayBuffer, offset: number, length: number) => {
            writeCalls.push({
              byteOffset,
              data: new Float32Array(buffer, offset, length / 4),
            });
          },
        ),
        submit,
      },
    } as unknown as GPUDevice,
    assertLive: vi.fn(),
    validatePrimitiveCandidate: vi.fn(),
    validateEmitterCandidate: vi.fn(),
    validateEnvironmentCandidate: vi.fn(),
    validateEmittersCandidate: vi.fn(),
    getScene: () => sceneRef.current,
    setSceneState: vi.fn((s: Scene) => { sceneRef.current = s; }),
    getSceneBuffers: () => sceneBuffers,
    getGeoPack: () => null,
    setGeoPack: vi.fn(),
    invalidateBindGroups: vi.fn(),
    supportedAnalyticShapes: () => new Set<string>(),
    cameraVisibleEmitters: () => cameraVisible,
    repackScene: vi.fn((s: Scene) => { sceneRef.current = s; }),
    setScene: vi.fn((s: Scene) => { sceneRef.current = s; }),
    reset: vi.fn(),
  };

  return {
    host, sceneRef, sceneBuffers, materialsBuffer, meshAreaLightsData, writeCalls,
    stagedCopy: { copyBufferToBuffer, submit },
  };
}

describe('SceneMutationRouter fold-preservation (H10)', () => {
  it('fold preserved after updateEmitter color patch (cameraVisibleEmitters=true)', () => {
    const scene = meshEmitterScene([1, 0, 0], 2);
    const { host, sceneBuffers, stagedCopy } = makeHostWithScene(scene, true);
    // Initial fold = [2, 0, 0]
    expect(sceneBuffers.materials[EMISSIVE_OFFSET]).toBeCloseTo(2, 5);

    const router = new SceneMutationRouter(host);
    // Patch the emitter to a new color
    router.updateEmitter('light', { color: [0, 1, 0] });
    expect(stagedCopy.copyBufferToBuffer).toHaveBeenCalled();
    expect(stagedCopy.submit).toHaveBeenCalledTimes(1);

    // New fold = [0, 2, 0] (new color [0,1,0] * intensity 2)
    expect(sceneBuffers.materials[EMISSIVE_OFFSET]).toBeCloseTo(0, 5);
    expect(sceneBuffers.materials[EMISSIVE_OFFSET + 1]).toBeCloseTo(2, 5);
    expect(sceneBuffers.materials[EMISSIVE_OFFSET + 2]).toBeCloseTo(0, 5);
  });

  it('fold preserved after updatePrimitive roughness patch (material fast path)', () => {
    const scene = meshEmitterScene([1, 0.5, 0], 4);
    const { host, sceneBuffers, stagedCopy } = makeHostWithScene(scene, true);
    // Initial fold = [4, 2, 0]
    expect(sceneBuffers.materials[EMISSIVE_OFFSET]).toBeCloseTo(4, 5);

    const router = new SceneMutationRouter(host);
    // Patch roughness (material-only fast path) — must NOT strip the fold
    router.updatePrimitive('panel', { material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.2, metallic: 0 } });
    expect(stagedCopy.copyBufferToBuffer).toHaveBeenCalled();
    expect(stagedCopy.submit).toHaveBeenCalledTimes(1);

    // Fold must still be [4, 2, 0] after the roughness patch
    expect(sceneBuffers.materials[EMISSIVE_OFFSET]).toBeCloseTo(4, 5);
    expect(sceneBuffers.materials[EMISSIVE_OFFSET + 1]).toBeCloseTo(2, 5);
    expect(sceneBuffers.materials[EMISSIVE_OFFSET + 2]).toBeCloseTo(0, 5);
  });

  it('emissive material patch re-packs the implicit mesh-area emitter', () => {
    const scene = implicitEmissiveScene([1, 0, 0], 2);
    const { host, sceneBuffers, stagedCopy } = makeHostWithScene(scene, true);
    // Initial implicit NEE emitter radiance = emissive * intensity = [2, 0, 0].
    expect(sceneBuffers.meshAreaLightsData[MESH_AREA_RADIANCE_OFFSET]).toBeCloseTo(2, 5);
    expect(sceneBuffers.meshAreaLightsData[MESH_AREA_RADIANCE_OFFSET + 1]).toBeCloseTo(0, 5);

    const router = new SceneMutationRouter(host);
    router.updatePrimitive('panel', {
      material: {
        baseColor: [0.5, 0.5, 0.5],
        roughness: 0.8,
        metallic: 0,
        emissive: [0, 1, 0],
        emissiveIntensity: 3,
      },
    });

    // Material camera-hit emission
    expect(stagedCopy.copyBufferToBuffer).toHaveBeenCalled();
    expect(stagedCopy.submit).toHaveBeenCalledTimes(1);

    // Material camera-hit emission and the synthesized mesh-area NEE emitter
    // both move to the new green radiance (material packing stores
    // emissive * emissiveIntensity).
    expect(sceneBuffers.materials[EMISSIVE_OFFSET]).toBeCloseTo(0, 5);
    expect(sceneBuffers.materials[EMISSIVE_OFFSET + 1]).toBeCloseTo(3, 5);
    expect(sceneBuffers.materials[EMISSIVE_OFFSET + 2]).toBeCloseTo(0, 5);
    expect(sceneBuffers.meshAreaLightsData[MESH_AREA_RADIANCE_OFFSET]).toBeCloseTo(0, 5);
    expect(sceneBuffers.meshAreaLightsData[MESH_AREA_RADIANCE_OFFSET + 1]).toBeCloseTo(3, 5);
    expect(sceneBuffers.meshAreaLightsData[MESH_AREA_RADIANCE_OFFSET + 2]).toBeCloseTo(0, 5);
  });

  it('fold NOT applied when cameraVisibleEmitters=false (control case)', () => {
    const scene = meshEmitterScene([1, 0.5, 0], 4);
    const { host, sceneBuffers, stagedCopy } = makeHostWithScene(scene, false);
    // No fold — emissive stays zero
    expect(sceneBuffers.materials[EMISSIVE_OFFSET]).toBeCloseTo(0, 5);

    const router = new SceneMutationRouter(host);
    router.updateEmitter('light', { color: [0, 1, 0] });
    expect(stagedCopy.copyBufferToBuffer).toHaveBeenCalled();
    expect(stagedCopy.submit).toHaveBeenCalledTimes(1);

    // Still no fold
    expect(sceneBuffers.materials[EMISSIVE_OFFSET]).toBeCloseTo(0, 5);
  });

  it.each([
    { cameraVisible: true, nextOwnerEmission: [4, 4, 4] },
    { cameraVisible: false, nextOwnerEmission: [0, 0, 0] },
  ])(
    'reconciles old and new owner slots when meshId moves (visible=$cameraVisible)',
    ({ cameraVisible, nextOwnerEmission }) => {
      const scene = movableMeshEmitterScene();
      const { host, sceneBuffers } = makeHostWithScene(scene, cameraVisible);
      const router = new SceneMutationRouter(host);

      router.updateEmitter('light', { meshId: 'panel-b' });

      expect(Array.from(
        sceneBuffers.materials.slice(EMISSIVE_OFFSET, EMISSIVE_OFFSET + 3),
      )).toEqual([0.5, 0, 0]);
      const panelBOffset = MATERIAL_FLOAT_STRIDE + EMISSIVE_OFFSET;
      expect(Array.from(
        sceneBuffers.materials.slice(panelBOffset, panelBOffset + 3),
      )).toEqual(nextOwnerEmission);
    },
  );

  it('reconciles removed and added owner slots for whole-list lighting updates', () => {
    const scene = movableMeshEmitterScene();
    const { host, sceneBuffers } = makeHostWithScene(scene, false);
    const router = new SceneMutationRouter(host);

    router.updateLighting({
      emitters: [{
        kind: 'mesh-area',
        id: 'replacement',
        meshId: 'panel-b',
        color: [1, 1, 1],
        intensity: 4,
      }],
    });

    expect(Array.from(
      sceneBuffers.materials.slice(EMISSIVE_OFFSET, EMISSIVE_OFFSET + 3),
    )).toEqual([0.5, 0, 0]);
    const panelBOffset = MATERIAL_FLOAT_STRIDE + EMISSIVE_OFFSET;
    expect(Array.from(
      sceneBuffers.materials.slice(panelBOffset, panelBOffset + 3),
    )).toEqual([0, 0, 0]);
  });
});
