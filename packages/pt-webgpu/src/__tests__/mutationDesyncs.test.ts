/**
 * mutationDesyncs.test.ts — Item 2 cluster: mutation-desync fixes.
 *
 * 2a. canFastPathMaterialPatch rejects TextureRef fields.
 * 2b. hasMeshAreaEmitterForPrimitive covers implicit emissive-mesh emitters.
 * 2c. Emissive-field material patch triggers emitter re-pack.
 * 2d. directionalAngularDiameter is updated by applyEmitterCountMutation.
 * 2e. clearReservoirBuffers is callable and clears allocated buffers.
 * 2f. topology/resource fast paths invalidate cached bind groups before reset.
 */

import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning, Scene } from '@vitrum/core';
import { asMat4, asTextureRef } from '@vitrum/core';
import { canFastPathMaterialPatch, materialPatchRepackFields } from '../scene/incrementalPatch.js';
import { hasMeshAreaEmitterForPrimitive } from '../scene/emitterPacking.js';
import {
  applyEmitterCountMutation,
  buildPackedScene,
  scenePackResultFromPacked,
} from '../scene/uploadSceneBuffers.js';
import { SceneMutationRouter } from '../sceneMutationRouter.js';
import { GpuResources } from '../gpuResources.js';
import type { MutationHost } from '../sceneMutationRouter.js';
import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';
import { installGpuConstStubs } from './gpuStub.js';

interface StubGpuBuffer {
  readonly label: string;
  readonly size: number;
  readonly destroy: ReturnType<typeof vi.fn>;
}

// ─── 2a: canFastPathMaterialPatch rejects TextureRef fields ───────────────────

describe('canFastPathMaterialPatch — Item 2a: TextureRef fields route to setScene', () => {
  it('accepts a scalar-only material patch', () => {
    expect(
      canFastPathMaterialPatch({ material: { roughness: 0.5, metallic: 0.1 } } as never),
    ).toBe(true);
  });

  it('rejects a patch containing baseColorMap (TextureRef)', () => {
    const handle = {};
    expect(
      canFastPathMaterialPatch({
        material: { roughness: 0.5, baseColorMap: asTextureRef(handle) },
      } as never),
    ).toBe(false);
  });

  it('rejects a patch containing normalMap', () => {
    expect(
      canFastPathMaterialPatch({
        material: { normalMap: asTextureRef({}) },
      } as never),
    ).toBe(false);
  });

  it('rejects a patch containing emissiveMap', () => {
    expect(
      canFastPathMaterialPatch({
        material: { emissive: [1, 0, 0], emissiveMap: asTextureRef({}) },
      } as never),
    ).toBe(false);
  });

  it('rejects a patch containing thicknessMap', () => {
    expect(
      canFastPathMaterialPatch({
        material: { thickness: 0.25, thicknessMap: asTextureRef({}) },
      } as never),
    ).toBe(false);
  });

  it('accepts an emissive-only patch without any map', () => {
    expect(
      canFastPathMaterialPatch({
        material: { emissive: [1, 0.5, 0], emissiveIntensity: 2 },
      } as never),
    ).toBe(true);
  });

  it('rejects when non-material facets are also touched (baseline check)', () => {
    expect(
      canFastPathMaterialPatch({ material: { roughness: 0.2 }, positions: new Float32Array(9) } as never),
    ).toBe(false);
  });

  it('classifies material fields that require a full descriptor/texture repack', () => {
    expect(
      materialPatchRepackFields({
        material: {
          baseColorMap: asTextureRef({}),
          opacity: 0.5,
          frontLayer: { normalMap: asTextureRef({}), normalScale: 0.25 },
        },
      } as never),
    ).toEqual({
      textureFields: ['baseColorMap'],
      descriptorScalarFields: ['opacity'],
      layerDescriptorFields: ['frontLayer.normalMap', 'frontLayer.normalScale'],
    });
  });
});

// ─── 2b: hasMeshAreaEmitterForPrimitive covers implicit emitters ──────────────

describe('hasMeshAreaEmitterForPrimitive — Item 2b: implicit emissive-mesh emitters', () => {
  it('returns true for a non-analytic primitive with non-zero emissive (no explicit emitter)', () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'glow',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, emissive: [2, 2, 2] },
        },
      ],
      emitters: [], // NO explicit mesh-area emitter
      environment: { kind: 'none' },
    };
    expect(hasMeshAreaEmitterForPrimitive(scene, 'glow')).toBe(true);
  });

  it('returns false for a mesh with emissive = [0,0,0] (luminance below threshold)', () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'dark',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, emissive: [0, 0, 0] },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    expect(hasMeshAreaEmitterForPrimitive(scene, 'dark')).toBe(false);
  });

  it('returns false for an analytic primitive even with non-zero emissive', () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'analytic',
          id: 'sphere-glow',
          shape: 'sphere',
          params: new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]),
          material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, emissive: [3, 3, 3] },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    expect(hasMeshAreaEmitterForPrimitive(scene, 'sphere-glow')).toBe(false);
  });

  it('still returns true via explicit mesh-area emitter (baseline)', () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'panel',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
        },
      ],
      emitters: [{ kind: 'mesh-area', id: 'e', meshId: 'panel', color: [1, 1, 1], intensity: 1 }],
      environment: { kind: 'none' },
    };
    expect(hasMeshAreaEmitterForPrimitive(scene, 'panel')).toBe(true);
  });
});

// ─── 2c: emissive-field patch triggers implicit emitter re-pack ───────────────

function makeHostWithEmissiveScene(scene: Scene): {
  host: MutationHost;
  sceneRef: { current: Scene };
  sceneBuffers: UploadedSceneBuffers;
  meshAreaLightsWriteCalls: Float32Array[];
  writeBuffer: { readonly mock: { readonly calls: unknown[][] }; mockClear: () => void };
} {
  const packed = buildPackedScene(scene, {});
  const geoPack = scenePackResultFromPacked(packed);
  const meshAreaLightsData = new Float32Array(packed.meshAreaLightsData);
  const meshAreaLightsWriteCalls: Float32Array[] = [];
  const writeBuffer = vi.fn((
    buf: unknown,
    _byteOffset: number,
    data: ArrayBuffer,
    srcOffset = 0,
    length?: number,
  ) => {
    if (buf === meshAreaLightsBuffer) {
      const byteLength = length ?? data.byteLength - srcOffset;
      meshAreaLightsWriteCalls.push(new Float32Array(data, srcOffset, Math.floor(byteLength / 4)));
    }
  });
  const createBuffer = vi.fn((desc: { label?: string; size?: number } | undefined): StubGpuBuffer => ({
    label: desc?.label ?? '',
    size: desc?.size ?? 16,
    destroy: vi.fn(),
  }));

  const buffer = (label: string, data?: ArrayBufferView): StubGpuBuffer => ({
    label,
    size: Math.max(16, data?.byteLength ?? 16),
    destroy: vi.fn(),
  });

  const meshAreaLightsBuffer = buffer('vitrum.pt-webgpu.scene.meshAreaLights', meshAreaLightsData);

  const sceneBuffers: UploadedSceneBuffers = {
    ...packed,
    meshAreaLightsData,
    meshAreaLightsBuffer: meshAreaLightsBuffer as unknown as GPUBuffer,
    positionsBuffer: buffer('vitrum.pt-webgpu.scene.positions', packed.positions) as unknown as GPUBuffer,
    normalsBuffer: buffer('vitrum.pt-webgpu.scene.normals', packed.normals) as unknown as GPUBuffer,
    indicesBuffer: buffer('vitrum.pt-webgpu.scene.indices', packed.indices) as unknown as GPUBuffer,
    triMaterialIdsBuffer: buffer('vitrum.pt-webgpu.scene.triMaterialIds', packed.triMaterialIds) as unknown as GPUBuffer,
    materialsBuffer: buffer('vitrum.pt-webgpu.scene.materials', packed.materials) as unknown as GPUBuffer,
    bvhNodesBuffer: buffer('vitrum.pt-webgpu.scene.bvhNodes', packed.bvhNodes) as unknown as GPUBuffer,
    analyticHeadersBuffer: buffer('vitrum.pt-webgpu.scene.analyticHeaders', packed.analyticHeaders) as unknown as GPUBuffer,
    analyticParamsBuffer: buffer('vitrum.pt-webgpu.scene.analyticParams', packed.analyticParams) as unknown as GPUBuffer,
    analyticLocalToWorldBuffer: buffer('vitrum.pt-webgpu.scene.analyticLocalToWorld', packed.analyticLocalToWorld) as unknown as GPUBuffer,
    analyticWorldToLocalBuffer: buffer('vitrum.pt-webgpu.scene.analyticWorldToLocal', packed.analyticWorldToLocal) as unknown as GPUBuffer,
    environmentMapTexelsBuffer: buffer('vitrum.pt-webgpu.scene.environmentMapTexels', packed.environmentMapTexels) as unknown as GPUBuffer,
    environmentMapCdfBuffer: buffer('vitrum.pt-webgpu.scene.environmentMapCdf', packed.environmentMapCdf) as unknown as GPUBuffer,
    directionalLightsBuffer: buffer('vitrum.pt-webgpu.scene.directionalLights', packed.directionalLightsData) as unknown as GPUBuffer,
    pointLightsBuffer: buffer('vitrum.pt-webgpu.scene.pointLights', packed.pointLightsData) as unknown as GPUBuffer,
    spotLightsBuffer: buffer('vitrum.pt-webgpu.scene.spotLights', packed.spotLightsData) as unknown as GPUBuffer,
    rectAreaLightsBuffer: buffer('vitrum.pt-webgpu.scene.rectAreaLights', packed.rectAreaLightsData) as unknown as GPUBuffer,
    lightTreeBuffer: buffer('vitrum.pt-webgpu.scene.lightTree', packed.lightTreeNodes) as unknown as GPUBuffer,
    tlasNodesBuffer: buffer('vitrum.pt-webgpu.scene.tlasNodes', packed.tlasNodes) as unknown as GPUBuffer,
    tlasInstanceIndicesBuffer: buffer('vitrum.pt-webgpu.scene.tlasInstanceIndices', packed.tlasInstanceIndices) as unknown as GPUBuffer,
    tlasBlasRootsBuffer: buffer('vitrum.pt-webgpu.scene.tlasBlasRoots', packed.tlasBlasRoots) as unknown as GPUBuffer,
    tlasInstanceWorldToLocalBuffer: buffer('vitrum.pt-webgpu.scene.tlasInstanceWorldToLocal', packed.tlasInstanceWorldToLocal) as unknown as GPUBuffer,
    tlasInstanceLocalToWorldBuffer: buffer('vitrum.pt-webgpu.scene.tlasInstanceLocalToWorld', packed.tlasInstanceLocalToWorld) as unknown as GPUBuffer,
    uvsBuffer: buffer('vitrum.pt-webgpu.scene.uvs', packed.uvs) as unknown as GPUBuffer,
    tangentsBuffer: buffer('vitrum.pt-webgpu.scene.tangents', packed.tangents) as unknown as GPUBuffer,
    colorsBuffer: buffer('vitrum.pt-webgpu.scene.colors', packed.colors) as unknown as GPUBuffer,
    materialTexDescriptorsBuffer: buffer('vitrum.pt-webgpu.scene.materialTexDescriptors', packed.materialTexDescriptors) as unknown as GPUBuffer,
    materialTexture: {} as GPUTexture,
    materialTextureView: {} as GPUTextureView,
    materialTextureSampler: {} as GPUSampler,
    materialLinearTexture: {} as GPUTexture,
    materialLinearTextureView: {} as GPUTextureView,
    bvhNodeCount: Math.floor(packed.bvhNodes.length / 8),
    tlasNodeCount: Math.floor(packed.tlasNodes.length / 8),
    materialCount: 1,
    gpuMemoryBytes: () => ({ bufferBytes: 0, textureBytesByFormat: {} }),
    destroy: vi.fn(),
  } as unknown as UploadedSceneBuffers;

  const sceneRef = { current: scene };

  const host: MutationHost = {
    device: {
      createBuffer,
      queue: { writeBuffer },
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
    repackScene: vi.fn((s: Scene) => { sceneRef.current = s; }),
    setScene: vi.fn((s: Scene) => { sceneRef.current = s; }),
    reset: vi.fn(),
  };

  return { host, sceneRef, sceneBuffers, meshAreaLightsWriteCalls, writeBuffer };
}

describe('SceneMutationRouter — Item 2c: emissive-field material patch triggers emitter re-pack', () => {
  it('emissive patch on implicit-emitter mesh → meshAreaLightsBuffer is re-uploaded', () => {
    // Start with a non-emissive mesh (luminance = 0 → no implicit emitter yet).
    // Then patch emissive to non-zero; the implicit emitter should be synthesized
    // and the buffer re-uploaded.
    const emissiveScene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'glow-mesh',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, emissive: [1, 1, 1] },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const { host, meshAreaLightsWriteCalls } = makeHostWithEmissiveScene(emissiveScene);
    const router = new SceneMutationRouter(host);

    // Patch emissiveIntensity — triggers Item 2c path.
    router.updatePrimitive('glow-mesh', {
      material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, emissive: [2, 2, 2], emissiveIntensity: 3 },
    } as never);

    // meshAreaLightsBuffer should have been written (emitter re-pack ran).
    expect(meshAreaLightsWriteCalls.length).toBeGreaterThan(0);
  });

  it('emissive-to-zero patch on implicit-emitter mesh removes stale mesh-area data', () => {
    installGpuConstStubs();
    const emissiveScene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'glow-mesh',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, emissive: [4, 4, 4] },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const { host, sceneBuffers } = makeHostWithEmissiveScene(emissiveScene);
    expect(sceneBuffers.meshAreaLightCount).toBeGreaterThan(0);
    expect(sceneBuffers.meshAreaLightsData.length).toBeGreaterThan(0);

    const router = new SceneMutationRouter(host);
    router.updatePrimitive('glow-mesh', {
      material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0, emissive: [0, 0, 0], emissiveIntensity: 1 },
    } as never);

    expect(sceneBuffers.meshAreaLightCount).toBe(0);
    expect(sceneBuffers.meshAreaLightsData.length).toBe(0);
    expect(host.invalidateBindGroups).toHaveBeenCalled();
  });

  it('non-emissive material patch (roughness only) does NOT trigger emitter re-pack for non-emissive mesh', () => {
    // A mesh with zero emissive — hasMeshAreaEmitterForPrimitive returns false,
    // so no emitter re-pack should happen.
    const plainScene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'plain',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };

    const { host, meshAreaLightsWriteCalls } = makeHostWithEmissiveScene(plainScene);
    const router = new SceneMutationRouter(host);

    router.updatePrimitive('plain', {
      material: { baseColor: [1, 1, 1], roughness: 0.9, metallic: 0 },
    } as never);

    expect(meshAreaLightsWriteCalls.length).toBe(0);
  });
});

// ─── 2d: directionalAngularDiameter in applyEmitterCountMutation ──────────────

describe('applyEmitterCountMutation — Item 2d: directionalAngularDiameter sync', () => {
  it('updates directionalAngularDiameter on the mutable buffer struct', () => {
    // Create a minimal UploadedSceneBuffers stub that exposes the field.
    const stubSb = {
      pointLightCount: 0,
      spotLightCount: 0,
      rectAreaLightCount: 0,
      meshAreaLightCount: 0,
      directionalLight: [0, 1, 0] as const,
      directionalIrradiance: [0, 0, 0] as const,
      directionalAngularDiameter: 0,
    } as unknown as UploadedSceneBuffers;

    applyEmitterCountMutation(stubSb, {
      directionalLightCount: 1,
      pointLightCount: 1,
      spotLightCount: 0,
      rectAreaLightCount: 0,
      meshAreaLightCount: 0,
      directionalLight: [0, 1, 0],
      directionalIrradiance: [1, 0.9, 0.8],
      directionalAngularDiameter: 0.009271, // ~sun angular diameter in radians
    });

    // The mutable cast in applyEmitterCountMutation writes directly to the struct.
    const mutable = stubSb as unknown as { directionalAngularDiameter: number };
    expect(mutable.directionalAngularDiameter).toBeCloseTo(0.009271, 5);
  });

  it('zero (default) is written when not supplied explicitly', () => {
    // Verify that directionalAngularDiameter defaults to 0 when passed as 0.
    const stubSb = {
      pointLightCount: 0,
      spotLightCount: 0,
      rectAreaLightCount: 0,
      meshAreaLightCount: 0,
      directionalLight: [0, 1, 0] as const,
      directionalIrradiance: [0, 0, 0] as const,
      directionalAngularDiameter: 0.5, // initial non-zero value
    } as unknown as UploadedSceneBuffers;

    applyEmitterCountMutation(stubSb, {
      directionalLightCount: 0,
      pointLightCount: 0,
      spotLightCount: 0,
      rectAreaLightCount: 0,
      meshAreaLightCount: 0,
      directionalLight: [0, 1, 0],
      directionalIrradiance: [0, 0, 0],
      directionalAngularDiameter: 0,
    });

    const mutable = stubSb as unknown as { directionalAngularDiameter: number };
    expect(mutable.directionalAngularDiameter).toBe(0);
  });

  it('preserves the signed castShadow:false mirror value', () => {
    const stubSb = {
      pointLightCount: 0,
      spotLightCount: 0,
      rectAreaLightCount: 0,
      meshAreaLightCount: 0,
      directionalLight: [0, 1, 0] as const,
      directionalIrradiance: [0, 0, 0] as const,
      directionalAngularDiameter: 0,
    } as unknown as UploadedSceneBuffers;

    applyEmitterCountMutation(stubSb, {
      directionalLightCount: 1,
      pointLightCount: 0,
      spotLightCount: 0,
      rectAreaLightCount: 0,
      meshAreaLightCount: 0,
      directionalLight: [0, 1, 0],
      directionalIrradiance: [1, 1, 1],
      directionalAngularDiameter: -1.009271,
    });

    const mutable = stubSb as unknown as { directionalAngularDiameter: number };
    expect(mutable.directionalAngularDiameter).toBeCloseTo(-1.009271, 5);
  });
});

// ─── 2e: clearReservoirBuffers clears allocated buffers ───────────────────────

describe('GpuResources.clearReservoirBuffers — Item 2e: reservoir history cleared on scene change', () => {
  it('clears all three reservoir buffers when allocated', () => {
    installGpuConstStubs();
    const clearBuffer = vi.fn();
    const finishStub = vi.fn(() => 'cmd');
    const encoder = { clearBuffer, finish: finishStub };
    const submit = vi.fn();
    const device = {
      createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
      createCommandEncoder: vi.fn(() => encoder),
      queue: { submit },
    } as unknown as GPUDevice;

    const buf = (label: string) => ({ label });
    const rptReservoirCur = buf('cur');
    const rptReservoirPrev = buf('prev');
    const rptReservoirSpatial = buf('spatial');

    const gpu = new GpuResources(device, 'full', false, true);
    gpu.rptReservoirCur = rptReservoirCur as unknown as GPUBuffer;
    gpu.rptReservoirPrev = rptReservoirPrev as unknown as GPUBuffer;
    gpu.rptReservoirSpatial = rptReservoirSpatial as unknown as GPUBuffer;
    gpu.clearReservoirBuffers();

    // Three clearBuffer calls — Cur, Prev, Spatial.
    expect(clearBuffer).toHaveBeenCalledTimes(3);
    expect(clearBuffer).toHaveBeenCalledWith(rptReservoirCur);
    expect(clearBuffer).toHaveBeenCalledWith(rptReservoirPrev);
    expect(clearBuffer).toHaveBeenCalledWith(rptReservoirSpatial);
    expect(device.createCommandEncoder).toHaveBeenCalledWith({
      label: 'vitrum.pt-webgpu.restirPt.clearReservoirs',
    });
    expect(finishStub).toHaveBeenCalledOnce();
    expect(submit).toHaveBeenCalledTimes(1);
    expect(submit).toHaveBeenCalledWith(['cmd']);
  });

  it('is a no-op when rptReservoirCur is null', () => {
    installGpuConstStubs();
    const submit = vi.fn();
    const device = {
      createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
      createCommandEncoder: vi.fn(),
      queue: { submit },
    } as unknown as GPUDevice;
    const gpu = new GpuResources(device, 'full', false, true);

    gpu.clearReservoirBuffers();

    expect(device.createCommandEncoder).not.toHaveBeenCalled();
    expect(submit).not.toHaveBeenCalled();
  });
});

// ─── 2d + updateEmitter: directionalAngularDiameter flows through updateEmitter ─

describe('SceneMutationRouter — Item 2d: updateEmitter syncs directionalAngularDiameter', () => {
  it('updateEmitter on directional emitter updates directionalAngularDiameter in sceneBuffers', () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'floor',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
        },
      ],
      emitters: [
        {
          kind: 'directional',
          id: 'sun',
          direction: [0, -1, 0],
          color: [1, 0.9, 0.8],
          intensity: 1,
          angularDiameter: 0,
        },
      ],
      environment: { kind: 'none' },
    };

    const packed = buildPackedScene(scene, {});
    const geoPack = scenePackResultFromPacked(packed);

    // Expose the mutable directionalAngularDiameter on the stub.
    const sbState = {
      directionalAngularDiameter: 0,
      directionalLight: packed.directionalLight,
      directionalIrradiance: packed.directionalIrradiance,
      pointLightCount: packed.pointLightCount,
      spotLightCount: packed.spotLightCount,
      rectAreaLightCount: packed.rectAreaLightCount,
      meshAreaLightCount: packed.meshAreaLightCount,
      pointLightsData: packed.pointLightsData,
      spotLightsData: packed.spotLightsData,
      rectAreaLightsData: packed.rectAreaLightsData,
      meshAreaLightsData: packed.meshAreaLightsData,
    };
    const sceneBuffers: UploadedSceneBuffers = {
      ...packed,
      ...sbState,
      positionsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      normalsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      indicesBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      triMaterialIdsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      materialsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      bvhNodesBuffer: { size: Math.max(16, packed.bvhNodes.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
      analyticHeadersBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      analyticParamsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      analyticLocalToWorldBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      analyticWorldToLocalBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      environmentMapTexelsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      environmentMapCdfBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      pointLightsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      spotLightsBuffer: { size: Math.max(16, packed.spotLightsData.byteLength), destroy: vi.fn() } as unknown as GPUBuffer,
      rectAreaLightsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      meshAreaLightsBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
      lightTreeBuffer: { size: 16, destroy: vi.fn() } as unknown as GPUBuffer,
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
      bvhNodeCount: Math.floor(packed.bvhNodes.length / 8),
      tlasNodeCount: Math.floor(packed.tlasNodes.length / 8),
      materialCount: 1,
      gpuMemoryBytes: () => ({ bufferBytes: 0, textureBytesByFormat: {} }),
      destroy: vi.fn(),
    } as unknown as UploadedSceneBuffers;

    const sceneRef = { current: scene };

    const host: MutationHost = {
      device: { queue: { writeBuffer: vi.fn() } } as unknown as GPUDevice,
      assertLive: vi.fn(),
      getScene: () => sceneRef.current,
      setSceneState: vi.fn((s: Scene) => { sceneRef.current = s; }),
      getSceneBuffers: () => sceneBuffers,
      getGeoPack: () => geoPack,
      setGeoPack: vi.fn(),
      invalidateBindGroups: vi.fn(),
      supportedAnalyticShapes: () => new Set<string>(),
      cameraVisibleEmitters: () => false,
      repackScene: vi.fn((s: Scene) => { sceneRef.current = s; }),
      setScene: vi.fn((s: Scene) => { sceneRef.current = s; }),
      reset: vi.fn(),
    };

    const router = new SceneMutationRouter(host);

    // Patch the directional emitter to set a non-zero angular diameter.
    router.updateEmitter('sun', { angularDiameter: 0.009271 });

    // After updateEmitter, the directionalAngularDiameter on sceneBuffers should
    // have been updated via applyEmitterCountMutation.
    const mutable = sceneBuffers as unknown as { directionalAngularDiameter: number };
    expect(mutable.directionalAngularDiameter).toBeCloseTo(0.009271, 4);

    router.updateEmitter('sun', { castShadow: false });
    expect(mutable.directionalAngularDiameter).toBeCloseTo(-1.009271, 4);
  });
});

describe('SceneMutationRouter — Phase 5C mutation observability', () => {
  it('warns when a material texture/descriptor patch falls back to a full scene repack', () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'floor',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const { host } = makeHostWithEmissiveScene(scene);
    const warnings: EngineWarning[] = [];
    const hostWithWarnings: MutationHost = {
      ...host,
      warn: vi.fn((warning: EngineWarning) => warnings.push(warning)),
    };
    const router = new SceneMutationRouter(hostWithWarnings);

    router.updatePrimitive('floor', {
      material: {
        baseColorMap: asTextureRef({ width: 1, height: 1, data: new Uint8Array([255, 255, 255, 255]) }),
        opacity: 0.75,
      },
    } as never);

    expect(hostWithWarnings.setScene).toHaveBeenCalledTimes(1);
    expect(hostWithWarnings.setSceneState).not.toHaveBeenCalled();
    expect(warnings).toEqual([
      expect.objectContaining({
        code: 'pt-webgpu.primitive-material-repack',
        backend: 'pt-webgpu',
        phase: 'mutation',
        method: 'updatePrimitive',
        details: expect.objectContaining({
          id: 'floor',
          fallbackReason: 'material-texture-descriptor-repack',
          nativePatchMissing: 'targeted-material-texture-descriptor-update',
          textureFields: ['baseColorMap'],
          descriptorScalarFields: ['opacity'],
          layerDescriptorFields: [],
        }),
      }),
    ]);
  });

  it('updateEmitter writes the emitter buffer, commits scene state, and resets accumulation', () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'floor',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
        },
      ],
      emitters: [
        {
          kind: 'point',
          id: 'lamp',
          position: [0, 2, 0],
          color: [1, 0.9, 0.7],
          intensity: 1,
        },
      ],
      environment: { kind: 'none' },
    };
    const { host, writeBuffer, sceneBuffers } = makeHostWithEmissiveScene(scene);
    const router = new SceneMutationRouter(host);
    writeBuffer.mockClear();

    router.updateEmitter('lamp', { intensity: 4, position: [2, 3, 4] });

    const writtenLabels = writeBuffer.mock.calls.map((c) => (c[0] as StubGpuBuffer | undefined)?.label ?? '');
    expect(writtenLabels).toContain('vitrum.pt-webgpu.scene.pointLights');
    expect(sceneBuffers.pointLightCount).toBe(1);
    expect(host.setSceneState).toHaveBeenCalledTimes(1);
    expect(host.setScene).not.toHaveBeenCalled();
    expect(host.reset).toHaveBeenCalledTimes(1);
  });

  it('updateEnvironment writes same-sized HDRI buffers, commits scene state, and resets accumulation', () => {
    const hdri = (scale: number) => ({
      width: 2,
      height: 1,
      data: new Float32Array([
        1 * scale, 0.5 * scale, 0.25 * scale,
        0.25 * scale, 0.5 * scale, 1 * scale,
      ]),
    });
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'floor',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'hdri', hdri: hdri(1), intensity: 1, rotationY: 0 },
    };
    const { host, writeBuffer, sceneBuffers } = makeHostWithEmissiveScene(scene);
    const router = new SceneMutationRouter(host);
    writeBuffer.mockClear();

    router.updateEnvironment({ kind: 'hdri', hdri: hdri(2), intensity: 0.4, rotationY: 0.25 });

    const writtenLabels = writeBuffer.mock.calls.map((c) => (c[0] as StubGpuBuffer | undefined)?.label ?? '');
    expect(writtenLabels).toContain('vitrum.pt-webgpu.scene.environmentMapTexels');
    expect(writtenLabels).toContain('vitrum.pt-webgpu.scene.environmentMapCdf');
    expect(sceneBuffers.hasEnvironmentMap).toBe(true);
    expect(sceneBuffers.environmentHdriIntensity).toBe(0.4);
    expect(sceneBuffers.environmentHdriRotationY).toBe(0.25);
    expect(host.setSceneState).toHaveBeenCalledTimes(1);
    expect(host.setScene).not.toHaveBeenCalled();
    expect(host.reset).toHaveBeenCalledTimes(1);
  });
});

describe('SceneMutationRouter — cached bind-group invalidation for reallocating mutations', () => {
  it('vertex/index-count topology patches invalidate cached bind groups before committing', () => {
    installGpuConstStubs();
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'resizable',
          positions: new Float32Array([0, 0, 0, 0.35, 0, 0, 0, 0.35, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [1, 0, 0], roughness: 0.5, metallic: 0 },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const { host } = makeHostWithEmissiveScene(scene);
    const router = new SceneMutationRouter(host);

    router.updatePrimitive('resizable', {
      positions: new Float32Array([
        -0.5, -0.5, 0,
         0.5, -0.5, 0,
         0.5,  0.5, 0,
        -0.5,  0.5, 0,
      ]),
      normals: new Float32Array([
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
      ]),
      indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    });

    expect(host.invalidateBindGroups).toHaveBeenCalledTimes(1);
    expect(host.setSceneState).toHaveBeenCalledTimes(1);
    expect(host.reset).toHaveBeenCalledTimes(1);
    expect(host.setScene).not.toHaveBeenCalled();
  });

  it('instanced-mesh count changes invalidate cached bind groups before committing', () => {
    installGpuConstStubs();
    const scene: Scene = {
      primitives: [
        {
          kind: 'instanced-mesh',
          id: 'instanced',
          positions: new Float32Array([0, 0, 0, 0.35, 0, 0, 0, 0.35, 0]),
          normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          material: { baseColor: [1, 0, 0], roughness: 0.5, metallic: 0 },
          instances: [
            asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -0.5, 0, 0, 1])),
            asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,  0.5, 0, 0, 1])),
          ],
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const { host } = makeHostWithEmissiveScene(scene);
    const router = new SceneMutationRouter(host);

    router.updatePrimitive('instanced', {
      instances: [
        asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -0.5, 0, 0, 1])),
      ],
    });

    expect(host.invalidateBindGroups).toHaveBeenCalledTimes(1);
    expect(host.setSceneState).toHaveBeenCalledTimes(1);
    expect(host.reset).toHaveBeenCalledTimes(1);
    expect(host.setScene).not.toHaveBeenCalled();
  });
});
