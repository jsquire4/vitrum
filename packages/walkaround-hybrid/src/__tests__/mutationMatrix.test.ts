import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { MockInstance } from 'vitest';
import { asMat4, type EngineWarning, type Scene, type SceneEmitter, type ScenePrimitive } from '@vitrum/core';
import { HybridEngine, type HybridEngineOptions } from '../HybridEngine.js';
import {
  buildReSTIRSceneBVHForCoreScene,
  type ReSTIRBvhMode,
  type SceneBVHBuffers,
} from '../restir/bvhCore.js';

function makeDeviceStub(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
    createTexture: vi.fn(() => ({ createView: vi.fn(() => ({})), destroy: vi.fn() })),
    createBindGroup: vi.fn(),
    queue: { submit: vi.fn(), writeBuffer: vi.fn(), writeTexture: vi.fn() },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function makeOpts(warnings: EngineWarning[] = []): HybridEngineOptions {
  return {
    device: makeDeviceStub(),
    width: 64,
    height: 64,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 2,
    skyTint: [1, 1, 1],
    skyIrradiance: 1,
    onWarning: (warning) => warnings.push(warning),
  };
}

function mat4Translate(x: number, y = 0, z = 0) {
  return asMat4(new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, y, z, 1,
  ]));
}

function mesh(id: string, x: number, baseColor: [number, number, number]): ScenePrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor, roughness: 0.5, metallic: 0 },
    transform: mat4Translate(x),
  };
}

function pointEmitter(intensity = 1): SceneEmitter {
  return {
    kind: 'point',
    id: 'lamp',
    position: [0, 2, 0],
    color: [1, 0.8, 0.6],
    intensity,
  };
}

function baseColorMapHandle(r: number): { width: number; height: number; data: Uint8Array; __vitrum_hint__: { channels: 4; dataType: 'uint8' } } {
  return {
    width: 1,
    height: 1,
    data: new Uint8Array([r, 255, 255, 255]),
    __vitrum_hint__: { channels: 4, dataType: 'uint8' },
  };
}

const WALKAROUND_PERMANENT_UNSUPPORTED_MATERIAL: Record<string, unknown> = {
  displacementMap: { handle: { id: 'height' } },
  displacementScale: 0.2,
  displacementBias: -0.1,
  spectralAttenuation: {
    wavelengthStart: 380,
    wavelengthEnd: 700,
    values: new Float32Array([0.1, 0.2, 0.3]),
  },
  dispersionAbbeNumber: 42,
  scatteringCoefficient: 0.15,
  scatteringAnisotropy: 0.25,
  scatteringCoefficientRGB: [0.1, 0.2, 0.3],
  frontLayer: { transmission: [1, 0.5, 0.25] },
  backLayer: { transmission: [0.25, 0.5, 1] },
  thinFilmStack: { layers: [{ ior: 1.4, thicknessNm: 300 }] },
};

const WALKAROUND_PERMANENT_UNSUPPORTED_FIELDS = [
  'backLayer',
  'dispersionAbbeNumber',
  'displacementBias',
  'displacementMap',
  'displacementScale',
  'frontLayer',
  'scatteringAnisotropy',
  'scatteringCoefficient',
  'scatteringCoefficientRGB',
  'spectralAttenuation',
  'thinFilmStack',
];

function baseScene(emitters: readonly SceneEmitter[] = []): Scene {
  return {
    primitives: [
      mesh('mesh-a', 0, [0.8, 0.2, 0.2]),
      mesh('mesh-b', 3, [0.2, 0.2, 0.8]),
    ],
    emitters,
    environment: { kind: 'none' },
  };
}

function makePipeline() {
  return {
    dispose: vi.fn(),
    refreshBvhMaterialSlice: vi.fn(),
    refreshBvhNormalsSlice: vi.fn(),
    refreshBvhRefit: vi.fn(),
    refreshBvhFullRebuild: vi.fn(),
    refreshTlasRefit: vi.fn(),
    updateEmitters: vi.fn(),
    updateAnalyticLights: vi.fn(),
    updateDirectionalEnvironment: vi.fn(),
    getEnvBindings: vi.fn((): { textureView: GPUTextureView; sampler: GPUSampler } | null => null),
    requestAccumReset: vi.fn(),
    resize: vi.fn(),
  };
}

function makeDdgi() {
  return {
    invalidateProbeCache: vi.fn(),
    markInstancesDirty: vi.fn(),
    syncRestirBvhBuffers: vi.fn(),
    setSkyParams: vi.fn(),
    setLights: vi.fn(),
    setSunIntensityMultiplier: vi.fn(),
    setEmitterTris: vi.fn(),
    setEnvironment: vi.fn(),
    dispose: vi.fn(),
  };
}

function makeRc() {
  return {
    refreshMaterialsFromCore: vi.fn(),
    invalidateBindings: vi.fn(),
    dispose: vi.fn(),
    refitCascadeBounds: vi.fn(),
    syncRestirBvhBuffers: vi.fn(),
    setSceneFromCore: vi.fn(),
    refitMergedInstance: vi.fn(() => false),
  };
}

interface SeededEngine {
  engine: HybridEngine;
  buffers: SceneBVHBuffers;
  pipeline: ReturnType<typeof makePipeline>;
  ddgi: ReturnType<typeof makeDdgi>;
  rc: ReturnType<typeof makeRc> | null;
  warnings: EngineWarning[];
}

interface HybridEngineInternals {
  _state: string;
  _lastScene: Scene | null;
  _renderScene: Scene | null;
  _bvhBuffers: SceneBVHBuffers | null;
  _pipeline: unknown;
  _ddgi: unknown;
  _rc: unknown;
}

function seedEngine(
  scene: Scene,
  opts: { bvhMode?: ReSTIRBvhMode; rc?: boolean } = {},
): SeededEngine {
  const warnings: EngineWarning[] = [];
  const engine = new HybridEngine(makeOpts(warnings));
  const pipeline = makePipeline();
  const ddgi = makeDdgi();
  const rc = opts.rc === true ? makeRc() : null;
  const buffers = buildReSTIRSceneBVHForCoreScene(scene, {
    bvhMode: opts.bvhMode ?? 'tlas',
  });
  const internals = engine as unknown as HybridEngineInternals;
  internals._state = 'ready';
  internals._lastScene = scene;
  internals._renderScene = scene;
  internals._bvhBuffers = buffers;
  internals._pipeline = pipeline;
  internals._ddgi = ddgi;
  internals._rc = rc;
  return { engine, buffers, pipeline, ddgi, rc, warnings };
}

function storedScene(engine: HybridEngine): Scene {
  const scene = (engine as unknown as HybridEngineInternals)._lastScene;
  if (scene == null) throw new Error('expected seeded scene');
  return scene;
}

function unpackUvFromVec4W(stream: Float32Array, vertexIndex: number): [number, number] {
  const words = new Uint32Array(stream.buffer, stream.byteOffset, stream.byteLength / 4);
  const word = words[vertexIndex * 4 + 3] ?? 0;
  return [
    (word & 0xFFFF) / 0xFFFF,
    ((word >>> 16) & 0xFFFF) / 0xFFFF,
  ];
}

describe('HybridEngine mutation matrix (non-GPU seam)', () => {
  let warnSpy: MockInstance;

  beforeEach(() => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  it.each<ReSTIRBvhMode>(['merged', 'tlas'])(
    'packs uv1 into bvh normal .w for texCoord 1 baseColorMap in %s mode',
    (bvhMode) => {
      const primitive = {
        kind: 'mesh',
        id: 'uv1-mesh',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
        uv1: new Float32Array([0.125, 0.25, 0.5, 0.625, 0.75, 0.875]),
        material: {
          baseColor: [1, 1, 1] as [number, number, number],
          roughness: 0.5,
          metallic: 0,
          baseColorMap: { handle: baseColorMapHandle(255), texCoord: 1 },
        },
        transform: mat4Translate(0),
      } satisfies ScenePrimitive;
      const scene: Scene = {
        primitives: [primitive],
        emitters: [],
        environment: { kind: 'none' },
      };

      const buffers = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode });
      const normals = new Float32Array(buffers.bvhNormals.cpuData);
      const uv0 = unpackUvFromVec4W(normals, 0);
      const uv1 = unpackUvFromVec4W(normals, 1);
      const uv2 = unpackUvFromVec4W(normals, 2);

      expect(uv0[0]).toBeCloseTo(0.125, 4);
      expect(uv0[1]).toBeCloseTo(0.25, 4);
      expect(uv1[0]).toBeCloseTo(0.5, 4);
      expect(uv1[1]).toBeCloseTo(0.625, 4);
      expect(uv2[0]).toBeCloseTo(0.75, 4);
      expect(uv2[1]).toBeCloseTo(0.875, 4);
      expect(buffers.materialTextureAtlas.baseColorMetaData[0]).toBe(0);
      expect(buffers.materialTextureAtlas.baseColorMetaData[1]).toBe(16);
    },
  );

  it('updatePrimitive(transform) refits TLAS, resets accumulation, and re-syncs DDGI BVH without invalidating probes', () => {
    const { engine, pipeline, ddgi } = seedEngine(baseScene(), { bvhMode: 'tlas' });
    try {
      const moved = mat4Translate(2, 0.5, 0);

      engine.updatePrimitive('mesh-a', { transform: moved });

      expect(pipeline.refreshTlasRefit).toHaveBeenCalledTimes(1);
      expect(pipeline.requestAccumReset).toHaveBeenCalledTimes(1);
      expect(ddgi.markInstancesDirty).toHaveBeenCalledTimes(1);
      expect(ddgi.syncRestirBvhBuffers).toHaveBeenCalledTimes(1);
      expect(ddgi.invalidateProbeCache).not.toHaveBeenCalled();
      expect(Array.from((storedScene(engine).primitives[0] as { transform: Float32Array }).transform))
        .toEqual(Array.from(moved));
    } finally {
      engine.dispose();
    }
  });

  it('updatePrimitive(positions+normals) uses TLAS refit and uploads normals without full rebuild', () => {
    const { engine, pipeline } = seedEngine(baseScene(), { bvhMode: 'tlas' });
    try {
      const scene = storedScene(engine);
      const prim = scene.primitives[0];
      if (prim == null || prim.kind !== 'mesh') throw new Error('expected mesh');
      const primNormals = prim.normals;
      if (primNormals == null) throw new Error('expected mesh normals');
      const positions = prim.positions.slice();
      for (let i = 0; i < positions.length; i += 3) positions[i] = (positions[i] ?? 0) + 0.125;
      const normals = new Float32Array(primNormals.length);
      for (let i = 0; i < normals.length; i += 3) normals[i] = 1;

      engine.updatePrimitive('mesh-a', { positions, normals });

      expect(pipeline.refreshBvhRefit).toHaveBeenCalledTimes(1);
      expect(pipeline.refreshBvhNormalsSlice).toHaveBeenCalledTimes(1);
      expect(pipeline.refreshTlasRefit).toHaveBeenCalledTimes(1);
      expect(pipeline.refreshBvhFullRebuild).not.toHaveBeenCalled();

      const [normalsSlice] = pipeline.refreshBvhNormalsSlice.mock.calls[0] as [
        { byteOffset: number; data: ArrayBuffer },
      ];
      expect(normalsSlice.byteOffset).toBe(0);
      const f32 = new Float32Array(normalsSlice.data);
      expect(f32[0]).toBeCloseTo(1, 5);
      expect(f32[1]).toBeCloseTo(0, 5);
      expect(f32[2]).toBeCloseTo(0, 5);
      expect(f32[3]).toBeCloseTo(0, 5);
    } finally {
      engine.dispose();
    }
  });

  it('updatePrimitive(material) uploads material resources, refreshes RC materials, and skips geometry GI propagation', () => {
    const { engine, pipeline, ddgi, rc } = seedEngine(baseScene(), { bvhMode: 'tlas', rc: true });
    try {
      engine.updatePrimitive('mesh-a', {
        material: { baseColor: [0.1, 0.9, 0.3], roughness: 0.25, metallic: 0 },
      });

      expect(pipeline.refreshBvhMaterialSlice).toHaveBeenCalledTimes(1);
      expect(pipeline.requestAccumReset).toHaveBeenCalledTimes(1);
      expect(rc?.refreshMaterialsFromCore).toHaveBeenCalledTimes(1);
      expect(ddgi.syncRestirBvhBuffers).not.toHaveBeenCalled();
      expect(ddgi.invalidateProbeCache).not.toHaveBeenCalled();
      expect(pipeline.updateEmitters).not.toHaveBeenCalled();
      expect((storedScene(engine).primitives[0] as { material: { roughness: number } }).material.roughness)
        .toBe(0.25);
    } finally {
      engine.dispose();
    }
  });

  it('updatePrimitive(material) emits structured warnings for every permanently unsupported material field', () => {
    const { engine, warnings } = seedEngine(baseScene(), { bvhMode: 'tlas' });
    try {
      engine.updatePrimitive('mesh-a', {
        material: {
          baseColor: [0.6, 0.6, 0.6],
          roughness: 0.35,
          metallic: 0,
          ...WALKAROUND_PERMANENT_UNSUPPORTED_MATERIAL,
          envMapIntensity: 0.35,
        },
      });

      const materialWarning = warnings.find((w) =>
        w.code === 'walkaround-hybrid.unconsumed-material-fields',
      );
      expect(materialWarning?.method).toBe('updatePrimitive');
      expect(materialWarning?.details?.fields).toEqual(WALKAROUND_PERMANENT_UNSUPPORTED_FIELDS);
    } finally {
      engine.dispose();
    }
  });

  it('updatePrimitive(material) emits a structured warning for residual alpha blend approximation', () => {
    const { engine, warnings } = seedEngine(baseScene(), { bvhMode: 'tlas' });
    try {
      const transparentMaterial = {
        baseColor: [0.6, 0.6, 0.6] as [number, number, number],
        roughness: 0.35,
        metallic: 0,
        alphaMode: 'blend' as const,
        opacity: 0.5,
      };

      engine.updatePrimitive('mesh-a', { material: transparentMaterial });
      engine.updatePrimitive('mesh-a', { material: transparentMaterial });

      const alphaWarnings = warnings.filter((w) =>
        w.code === 'walkaround-hybrid.alpha-blend-approximation',
      );
      expect(alphaWarnings).toHaveLength(1);
      expect(alphaWarnings[0]?.method).toBe('updatePrimitive');
      expect(alphaWarnings[0]?.details?.primitiveIds).toEqual(['mesh-a']);
    } finally {
      engine.dispose();
    }
  });

  it('updatePrimitive(material) rebuilds material texture atlas when atlas-backed maps change', () => {
    const { engine, pipeline, ddgi } = seedEngine(baseScene(), { bvhMode: 'tlas' });
    try {
      engine.updatePrimitive('mesh-a', {
        material: {
          baseColor: [1, 1, 1],
          roughness: 0.5,
          metallic: 0,
          baseColorMap: { handle: baseColorMapHandle(128) },
          normalMap: { handle: baseColorMapHandle(96), texCoord: 1 },
          normalScale: 0.5,
          roughnessMap: { handle: baseColorMapHandle(64) },
          aoMap: { handle: baseColorMapHandle(32), wrapS: 'clamp-to-edge' },
          aoMapIntensity: 0.5,
          alphaMode: 'mask',
          alphaCutoff: 0.25,
          alphaMap: { handle: baseColorMapHandle(16), texCoord: 1 },
          emissive: [2, 1, 0.5],
          emissiveIntensity: 3,
          emissiveMap: { handle: baseColorMapHandle(8), wrapT: 'mirrored-repeat' },
          transmission: 1,
          transmissionMap: { handle: baseColorMapHandle(4), wrapS: 'clamp-to-edge' },
          lightMap: { handle: baseColorMapHandle(2), texCoord: 1 },
          lightMapIntensity: 2,
          specularColorMap: { handle: baseColorMapHandle(224), wrapS: 'mirrored-repeat' },
          specularIntensityMap: { handle: baseColorMapHandle(240), texCoord: 1 },
          clearcoatMap: { handle: baseColorMapHandle(192), wrapS: 'clamp-to-edge' },
          clearcoatRoughnessMap: { handle: baseColorMapHandle(208), texCoord: 1 },
          clearcoatNormalMap: { handle: baseColorMapHandle(144), texCoord: 1 },
          clearcoatNormalScale: 0.25,
          sheenColorMap: { handle: baseColorMapHandle(176), wrapT: 'mirrored-repeat' },
          sheenRoughnessMap: { handle: baseColorMapHandle(160), texCoord: 1 },
          anisotropy: 0.5,
          anisotropyRotation: 0.25,
          anisotropyMap: { handle: baseColorMapHandle(112), texCoord: 1 },
          iridescence: 0.5,
          iridescenceIor: 2,
          iridescenceThicknessRange: [200, 800],
          iridescenceMap: { handle: baseColorMapHandle(72), wrapT: 'mirrored-repeat' },
          iridescenceThicknessMap: { handle: baseColorMapHandle(88), texCoord: 1 },
        },
      });

      expect(pipeline.refreshBvhFullRebuild).toHaveBeenCalledTimes(1);
      expect(pipeline.refreshBvhMaterialSlice).not.toHaveBeenCalled();
      expect(pipeline.requestAccumReset).toHaveBeenCalledTimes(1);
      expect(ddgi.invalidateProbeCache).toHaveBeenCalledTimes(1);
      const [rebuilt] = pipeline.refreshBvhFullRebuild.mock.calls[0] as [SceneBVHBuffers];
      expect(rebuilt.materialTextureAtlas.readableBaseColorLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableNormalLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableRoughnessLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableAoLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableAlphaLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableEmissiveLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableTransmissionLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableLightLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableSpecularColorLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableSpecularIntensityLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableClearcoatLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableClearcoatRoughnessLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableClearcoatNormalLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableSheenColorLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableSheenRoughnessLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableAnisotropyLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableIridescenceLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableIridescenceThicknessLayerCount).toBe(1);
      const material = (storedScene(engine).primitives[0] as {
        material: {
          baseColorMap?: unknown;
          normalMap?: unknown;
          normalScale?: unknown;
          roughnessMap?: unknown;
          aoMap?: unknown;
          alphaMap?: unknown;
          emissiveMap?: unknown;
          transmissionMap?: unknown;
          lightMap?: unknown;
          lightMapIntensity?: unknown;
          specularColorMap?: unknown;
          specularIntensityMap?: unknown;
          clearcoatMap?: unknown;
          clearcoatRoughnessMap?: unknown;
          clearcoatNormalMap?: unknown;
          clearcoatNormalScale?: unknown;
          sheenColorMap?: unknown;
          sheenRoughnessMap?: unknown;
          anisotropy?: unknown;
          anisotropyRotation?: unknown;
          anisotropyMap?: unknown;
          iridescence?: unknown;
          iridescenceIor?: unknown;
          iridescenceThicknessRange?: unknown;
          iridescenceMap?: unknown;
          iridescenceThicknessMap?: unknown;
        };
      }).material;
      expect(material.baseColorMap).toBeDefined();
      expect(material.normalMap).toBeDefined();
      expect(material.normalScale).toBe(0.5);
      expect(material.roughnessMap).toBeDefined();
      expect(material.aoMap).toBeDefined();
      expect(material.alphaMap).toBeDefined();
      expect(material.emissiveMap).toBeDefined();
      expect(material.transmissionMap).toBeDefined();
      expect(material.lightMap).toBeDefined();
      expect(material.lightMapIntensity).toBe(2);
      expect(material.specularColorMap).toBeDefined();
      expect(material.specularIntensityMap).toBeDefined();
      expect(material.clearcoatMap).toBeDefined();
      expect(material.clearcoatRoughnessMap).toBeDefined();
      expect(material.clearcoatNormalMap).toBeDefined();
      expect(material.clearcoatNormalScale).toBe(0.25);
      expect(material.sheenColorMap).toBeDefined();
      expect(material.sheenRoughnessMap).toBeDefined();
      expect(material.anisotropy).toBe(0.5);
      expect(material.anisotropyRotation).toBe(0.25);
      expect(material.anisotropyMap).toBeDefined();
      expect(material.iridescence).toBe(0.5);
      expect(material.iridescenceIor).toBe(2);
      expect(material.iridescenceThicknessRange).toEqual([200, 800]);
      expect(material.iridescenceMap).toBeDefined();
      expect(material.iridescenceThicknessMap).toBeDefined();
    } finally {
      engine.dispose();
    }
  });

  it('updatePrimitive(material) rebuilds material texture atlas when atlas metadata scalars change', () => {
    const lightHandle = baseColorMapHandle(192);
    const alphaHandle = baseColorMapHandle(64);
    const base = baseScene();
    const prim = base.primitives[0];
    if (prim == null || prim.kind !== 'mesh') throw new Error('expected mesh');
    const seededScene: Scene = {
      ...base,
      primitives: [
        {
          ...prim,
          material: {
            baseColor: [1, 1, 1],
            roughness: 0.5,
            metallic: 0,
            alphaMode: 'mask',
            alphaCutoff: 0.5,
            opacity: 1,
            alphaMap: { handle: alphaHandle },
            lightMap: { handle: lightHandle },
            lightMapIntensity: 1,
            specularColor: [1, 1, 1],
            specularIntensity: 1,
            clearcoat: 0,
            clearcoatRoughness: 0,
            sheen: 0,
            sheenRoughness: 0,
            sheenColor: [0, 0, 0],
            iridescence: 0,
            iridescenceIor: 1.3,
            iridescenceThicknessRange: [100, 400],
            envMapIntensity: 1,
          },
        },
        ...base.primitives.slice(1),
      ],
    };
    const { engine, pipeline } = seedEngine(seededScene, { bvhMode: 'tlas' });
    try {
      engine.updatePrimitive('mesh-a', {
        material: {
          baseColor: [1, 1, 1],
          roughness: 0.5,
          metallic: 0,
          alphaMode: 'mask',
          alphaCutoff: 0.25,
          opacity: 0.75,
          alphaMap: { handle: alphaHandle },
          lightMap: { handle: lightHandle },
          lightMapIntensity: 3,
          specularColor: [0.25, 0.5, 0.75],
          specularIntensity: 0.4,
          clearcoat: 0.6,
          clearcoatRoughness: 0.2,
          sheen: 0.8,
          sheenRoughness: 0.3,
          sheenColor: [0.9, 0.4, 0.2],
          iridescence: 0.65,
          iridescenceIor: 2,
          iridescenceThicknessRange: [250, 750],
          envMapIntensity: 0.35,
        },
      });

      expect(pipeline.refreshBvhFullRebuild).toHaveBeenCalledTimes(1);
      expect(pipeline.refreshBvhMaterialSlice).not.toHaveBeenCalled();
      const [rebuilt] = pipeline.refreshBvhFullRebuild.mock.calls[0] as [SceneBVHBuffers];
      expect(rebuilt.materialTextureAtlas.readableAlphaLayerCount).toBe(1);
      expect(rebuilt.materialTextureAtlas.readableLightLayerCount).toBe(1);
    } finally {
      engine.dispose();
    }
  });

  it('updatePrimitive(material) rebuilds material texture atlas when only envMapIntensity changes', () => {
    const base = baseScene();
    const prim = base.primitives[0];
    if (prim == null || prim.kind !== 'mesh') throw new Error('expected mesh');
    const seededScene: Scene = {
      ...base,
      primitives: [
        {
          ...prim,
          material: {
            baseColor: [1, 1, 1],
            roughness: 0.5,
            metallic: 0,
            envMapIntensity: 1,
          },
        },
        ...base.primitives.slice(1),
      ],
    };
    const { engine, pipeline } = seedEngine(seededScene, { bvhMode: 'tlas' });
    try {
      engine.updatePrimitive('mesh-a', {
        material: {
          baseColor: [1, 1, 1],
          roughness: 0.5,
          metallic: 0,
          envMapIntensity: 0.25,
        },
      });

      expect(pipeline.refreshBvhFullRebuild).toHaveBeenCalledTimes(1);
      expect(pipeline.refreshBvhMaterialSlice).not.toHaveBeenCalled();
      const [rebuilt] = pipeline.refreshBvhFullRebuild.mock.calls[0] as [SceneBVHBuffers];
      expect(rebuilt.materialTextureAtlas.baseColorMetaData[208]).toBeCloseTo(0.25, 5);
    } finally {
      engine.dispose();
    }
  });

  it('updateEmitter repacks emitters, invalidates GI lighting state, invalidates RC bindings, and resets accumulation', () => {
    const { engine, pipeline, ddgi, rc } = seedEngine(baseScene([pointEmitter()]), {
      bvhMode: 'tlas',
      rc: true,
    });
    try {
      engine.updateEmitter('lamp', { intensity: 4, position: [2, 3, 4] } as Partial<SceneEmitter>);

      expect(pipeline.updateEmitters).toHaveBeenCalledTimes(1);
      expect(rc?.invalidateBindings).toHaveBeenCalledTimes(1);
      expect(ddgi.setSunIntensityMultiplier).toHaveBeenCalledTimes(1);
      expect(ddgi.setLights).toHaveBeenCalledTimes(1);
      expect(ddgi.setEmitterTris).toHaveBeenCalledTimes(1);
      expect(ddgi.invalidateProbeCache).toHaveBeenCalledTimes(1);
      expect(pipeline.updateAnalyticLights).toHaveBeenCalledTimes(1);
      expect(pipeline.updateDirectionalEnvironment).toHaveBeenCalledWith(null, 0, 0);
      expect(ddgi.setEnvironment).toHaveBeenCalledWith(null, null, 0, 0, false);
      expect(pipeline.requestAccumReset).toHaveBeenCalledTimes(1);
      expect((storedScene(engine).emitters[0] as { intensity: number }).intensity).toBe(4);
    } finally {
      engine.dispose();
    }
  });

  it('updateEnvironment bakes procedural-sky, stores the env, updates directional IBL, invalidates DDGI, and resets accumulation', () => {
    const { engine, pipeline, ddgi, warnings } = seedEngine(baseScene(), { bvhMode: 'tlas' });
    const envTextureView = {} as GPUTextureView;
    const envSampler = {} as GPUSampler;
    pipeline.getEnvBindings.mockReturnValue({
      textureView: envTextureView,
      sampler: envSampler,
    });
    try {
      engine.updateEnvironment({
        kind: 'procedural-sky',
        sunDirection: [0, 1, 0],
        turbidity: 2,
        rayleigh: 1,
        mieCoefficient: 0.01,
        mieDirectionalG: 0.8,
        intensity: 3,
      });

      expect(warnings).toHaveLength(0);
      expect(storedScene(engine).environment.kind).toBe('procedural-sky');
      expect(ddgi.setSkyParams).toHaveBeenCalledWith(expect.any(Array), expect.any(Number));
      expect(ddgi.invalidateProbeCache).toHaveBeenCalledTimes(1);
      expect(pipeline.requestAccumReset).toHaveBeenCalledTimes(1);
      expect(pipeline.updateDirectionalEnvironment).toHaveBeenCalledWith(
        expect.objectContaining({ width: 256, height: 128 }),
        0,
        1,
      );
      expect(ddgi.setEnvironment).toHaveBeenCalledWith(envTextureView, envSampler, 0, 1, true);
    } finally {
      engine.dispose();
    }
  });

  it('updateLighting warns on unknown keys and invalidates sky lighting without rebuilding resources', () => {
    const { engine, pipeline, ddgi, warnings } = seedEngine(baseScene(), { bvhMode: 'tlas' });
    try {
      engine.updateLighting({
        skyTint: [0.4, 0.5, 0.6],
        typoIntensity: 8,
      } as never);

      expect(warnings.some((w) =>
        w.code === 'walkaround-hybrid.unknown-lighting-key' &&
        w.details?.key === 'typoIntensity',
      )).toBe(true);
      expect(ddgi.setSkyParams).toHaveBeenCalledWith([0.4, 0.5, 0.6], 1);
      expect(ddgi.invalidateProbeCache).toHaveBeenCalledTimes(1);
      expect(pipeline.requestAccumReset).toHaveBeenCalledTimes(1);
      expect(pipeline.resize).not.toHaveBeenCalled();
    } finally {
      engine.dispose();
    }
  });

  it('setSize resizes frame resources only; zero/same-size calls are no-ops and DDGI is preserved', () => {
    const { engine, pipeline, ddgi } = seedEngine(baseScene(), { bvhMode: 'tlas' });
    try {
      engine.setSize(64, 64);
      engine.setSize(0, 128);
      expect(pipeline.resize).not.toHaveBeenCalled();

      engine.setSize(128, 80);

      expect(pipeline.resize).toHaveBeenCalledTimes(1);
      expect(pipeline.resize).toHaveBeenCalledWith(128, 80);
      expect(ddgi.invalidateProbeCache).not.toHaveBeenCalled();
      expect(pipeline.requestAccumReset).not.toHaveBeenCalled();
    } finally {
      engine.dispose();
    }
  });
});
