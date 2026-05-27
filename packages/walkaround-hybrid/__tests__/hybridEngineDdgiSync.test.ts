/**
 * C2 — eager DDGI ReSTIR BVH sync after updatePrimitive (not only renderFrame).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

vi.mock('../src/ddgi/DDGI.js', () => ({
  DDGI: class MockDDGI {
    ready = true;
    pass = {
      setSunIntensityMultiplier: vi.fn(),
      setGlassMixScale: vi.fn(),
      getReadAtlasGPUTextures: vi.fn(),
      setRestirBvhSnapshot: vi.fn(),
    };
    syncRestirBvhBuffers = vi.fn();
    invalidateProbeCache = vi.fn();
    markInstancesDirty = vi.fn();
    setLights = vi.fn();
    updateFrame = vi.fn();
    dispose = vi.fn();
  },
}));

type SyncState = {
  pipelineInitDeferreds: Array<{ promise: Promise<void>; resolve: () => void }>;
  pipelineConstructed: Array<{ refreshBvhRefit: ReturnType<typeof vi.fn>; requestAccumReset: ReturnType<typeof vi.fn> }>;
  buildBVHCalls: unknown[];
};

vi.mock('../src/pipeline/WalkaroundGPUPipeline.js', () => {
  const g = globalThis as unknown as { __HYBRID_DDGI_SYNC__?: SyncState };
  if (!g.__HYBRID_DDGI_SYNC__) {
    g.__HYBRID_DDGI_SYNC__ = { pipelineInitDeferreds: [], pipelineConstructed: [], buildBVHCalls: [] };
  }
  const state = g.__HYBRID_DDGI_SYNC__!;
  class MockPipeline {
    readonly index: number;
    refreshBvhRefit = vi.fn();
    refreshBvhFullRebuild = vi.fn();
    requestAccumReset = vi.fn();
    updateEmitters = vi.fn();
    setDDGIInputs = vi.fn();
    setRCInputs = vi.fn();
    renderFrame = vi.fn();
    dispose = vi.fn();
    resize = vi.fn();
    constructor() {
      this.index = state.pipelineInitDeferreds.length;
      let resolve!: () => void;
      const promise = new Promise<void>((res) => { resolve = res; });
      state.pipelineInitDeferreds.push({ promise, resolve });
      state.pipelineConstructed.push(this);
    }
    async initialize(): Promise<void> {
      await state.pipelineInitDeferreds[this.index]!.promise;
    }
  }
  return { WalkaroundGPUPipeline: MockPipeline, HYBRID_WEBGPU_REQUIRED_LIMITS: {}, HYBRID_WEBGPU_REQUIRED_FEATURES: [] };
});

vi.mock('../src/restir/bvhCompute.js', () => {
  const g = globalThis as unknown as { __HYBRID_DDGI_SYNC__?: SyncState };
  if (!g.__HYBRID_DDGI_SYNC__) {
    g.__HYBRID_DDGI_SYNC__ = { pipelineInitDeferreds: [], pipelineConstructed: [], buildBVHCalls: [] };
  }
  const state = g.__HYBRID_DDGI_SYNC__!;
  const makeFakeBuffers = () => {
    const nodeBuf = new ArrayBuffer(32);
    const nodeU32 = new Uint32Array(nodeBuf);
    nodeU32[7] = 0xffff0001;
    return {
      bvhNodes: { cpuData: nodeBuf, count: 1, byteLength: 32 },
      bvhIndex: { cpuData: new ArrayBuffer(16), count: 1, byteLength: 16 },
      bvhBeerColors: { cpuData: new ArrayBuffer(16), count: 1, byteLength: 16 },
      bvhPositions: { cpuData: new ArrayBuffer(64), count: 4, byteLength: 64 },
      emitters: { cpuData: new ArrayBuffer(16), count: 0, byteLength: 16 },
      emitterCdf: { cpuData: new ArrayBuffer(16), count: 0, byteLength: 16 },
      emitterCount: 0,
      totalEmissivePower: 0,
      mergedGeometry: new THREE.BufferGeometry(),
      meshVertexRanges: [{
        name: 'mesh-a', vertexStart: 0, vertexCount: 3, triStart: 0, triCount: 1,
        matrixWorldAtBuild: new Float32Array([
          1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
        ]),
      }],
      bvhIndicesStride3: new Uint32Array([0, 1, 2]),
      triangleMaterialIds: { cpuData: new Uint32Array(1).buffer, byteLength: 4, count: 1 },
      buildMaterials: [new THREE.MeshStandardMaterial()],
      emitterNormals: new Float32Array(16),
      bvhMode: 'merged' as const,
      primitiveTlasBindings: [],
    };
  };
  const buildFn = vi.fn(() => {
    state.buildBVHCalls.push({});
    return makeFakeBuffers();
  });
  return {
    buildReSTIRSceneBVH: buildFn,
    buildReSTIRSceneBVHForScene: buildFn,
    rebuildEmitterBuffersFromSceneRoots: vi.fn(() => ({
      emitters: { cpuData: new ArrayBuffer(80), byteLength: 80, count: 0 },
      emitterCdf: { cpuData: new ArrayBuffer(4), byteLength: 4, count: 0 },
      emitterCount: 0,
      totalEmissivePower: 0,
    })),
    disposeSceneBVH: vi.fn(),
  };
});

vi.mock('@vitrum/three-bindings', () => {
  const scene = new THREE.Scene();
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3));
  const mesh = new THREE.Mesh(geom, new THREE.MeshBasicMaterial());
  mesh.name = 'mesh-a';
  scene.add(mesh);
  return {
    vitrumSceneToThree: vi.fn(() => scene),
    disposeVitrumThreeSceneRoot: vi.fn(),
    applyVitrumMaterialToMesh: vi.fn(),
  };
});

import { HybridEngine } from '../src/HybridEngine.js';
import { asMat4, type Scene } from '@vitrum/core';

const SCENE: Scene = {
  primitives: [{
    id: 'mesh-a',
    kind: 'mesh',
    mesh: {
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      indices: new Uint32Array([0, 1, 2]),
    },
    material: { kind: 'lambertian', albedo: [1, 1, 1] },
  } as unknown as Scene['primitives'][number]],
  emitters: [],
  environment: { kind: 'none' },
};

function state(): SyncState {
  return (globalThis as unknown as { __HYBRID_DDGI_SYNC__: SyncState }).__HYBRID_DDGI_SYNC__;
}

beforeEach(() => {
  const s = state();
  s.pipelineInitDeferreds.length = 0;
  s.pipelineConstructed.length = 0;
  s.buildBVHCalls.length = 0;
});

describe('HybridEngine DDGI eager sync (C2)', () => {
  it('syncRestirBvhBuffers runs on transform refit before renderFrame', async () => {
    const engine = new HybridEngine({
      device: { queue: { writeBuffer: vi.fn(), submit: vi.fn() } } as unknown as GPUDevice,
      width: 64,
      height: 64,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      skyTint: [1, 1, 1],
      skyIrradiance: 1,
    });
    engine.setScene(SCENE);
    while (state().pipelineInitDeferreds.length < 1) {
      await new Promise((r) => setTimeout(r, 5));
    }
    state().pipelineInitDeferreds[0]!.resolve();
    for (let i = 0; i < 30; i++) await Promise.resolve();
    await new Promise((r) => setTimeout(r, 50));

    const ddgi = (engine as unknown as { _ddgi: { syncRestirBvhBuffers: ReturnType<typeof vi.fn> } })._ddgi;
    ddgi.syncRestirBvhBuffers.mockClear();

    engine.updatePrimitive!('mesh-a', {
      transform: asMat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1]),
    });

    expect(ddgi.syncRestirBvhBuffers).toHaveBeenCalledTimes(1);
  });
});
