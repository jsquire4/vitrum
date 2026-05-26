/**
 * PR-1 — material-only `updatePrimitive` fast path (no setScene / full init).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

type MatUpdateState = {
  pipelineInitDeferreds: Array<{ promise: Promise<void>; resolve: () => void; reject: (e: unknown) => void }>;
  pipelineConstructed: unknown[];
  buildBVHCalls: unknown[];
};

vi.mock('../src/ddgi/DDGI.js', () => ({
  DDGI: class MockDDGI {
    ready = true;
    pass = { setSunIntensityMultiplier: vi.fn(), getReadAtlasGPUTextures: vi.fn() };
    invalidateProbeCache = vi.fn();
    setLights = vi.fn();
    updateFrame = vi.fn();
    dispose = vi.fn();
  },
}));

vi.mock('../src/restir/bvhCompute.js', async () => {
  const g = globalThis as unknown as { __HYBRID_MAT_STATE__?: MatUpdateState };
  if (!g.__HYBRID_MAT_STATE__) {
    g.__HYBRID_MAT_STATE__ = {
      pipelineInitDeferreds: [],
      pipelineConstructed: [],
      buildBVHCalls: [],
    };
  }
  const state = g.__HYBRID_MAT_STATE__;

  function makeFakeBuffers() {
    const indexBuf = new Uint32Array(4);
    indexBuf[0] = 0;
    indexBuf[1] = 1;
    indexBuf[2] = 2;
    indexBuf[3] = (153 << 24) | (148 << 16) | (140 << 8);
    const nodeBuf = new ArrayBuffer(32);
    const nodeU32 = new Uint32Array(nodeBuf);
    nodeU32[6] = 0;
    nodeU32[7] = 0xffff0001;

    return {
      bvhNodes: { cpuData: nodeBuf, count: 1, byteLength: 32 },
      bvhIndex: { cpuData: indexBuf.buffer, count: 1, byteLength: 16 },
      bvhBeerColors: { cpuData: new ArrayBuffer(4), count: 1, byteLength: 4 },
      bvhPositions: { cpuData: new ArrayBuffer(64), count: 4, byteLength: 64 },
      emitters: { cpuData: new ArrayBuffer(16), count: 0, byteLength: 16 },
      emitterCdf: { cpuData: new ArrayBuffer(16), count: 0, byteLength: 16 },
      emitterCount: 0,
      totalEmissivePower: 0,
      mergedGeometry: new THREE.BufferGeometry(),
      meshVertexRanges: [{
        name: 'mesh-a',
        vertexStart: 0,
        vertexCount: 3,
        triStart: 0,
        triCount: 1,
        matrixWorldAtBuild: new Float32Array(16),
      }],
      bvhIndicesStride3: new Uint32Array([0, 1, 2]),
      triangleMaterialIds: { cpuData: new Uint32Array([0]).buffer, byteLength: 4, count: 1 },
      buildMaterials: [new THREE.MeshStandardMaterial({ color: 0x99948c })],
      emitterNormals: new Float32Array(16),
      bvhMode: 'merged' as const,
      primitiveTlasBindings: [],
    };
  }

  const buildFn = vi.fn(() => {
    state.buildBVHCalls.push({});
    return makeFakeBuffers();
  });

  return {
    buildReSTIRSceneBVH: buildFn,
    buildReSTIRSceneBVHForScene: buildFn,
    rebuildEmitterBuffersFromSceneRoots: vi.fn(() => ({
      emitters: { cpuData: new ArrayBuffer(80), byteLength: 80, count: 1 },
      emitterCdf: { cpuData: new Float32Array(1).buffer, byteLength: 4, count: 1 },
      emitterCount: 1,
      totalEmissivePower: 2,
    })),
    disposeSceneBVH: vi.fn(),
  };
});

vi.mock('@vitrum/three-bindings', async () => {
  const THREE = await import('three');
  function makeSyntheticScene(): THREE.Scene {
    const scene = new THREE.Scene();
    const geom = new THREE.BufferGeometry();
    geom.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), 3),
    );
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array([0, 1, 2]), 1));
    const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ color: 0x99948c }));
    mesh.name = 'mesh-a';
    scene.add(mesh);
    return scene;
  }
  return {
    vitrumSceneToThree: vi.fn(() => makeSyntheticScene()),
    disposeVitrumThreeSceneRoot: vi.fn(),
    applyVitrumMaterialToMesh: vi.fn((mesh: THREE.Mesh, material: { baseColor: number[] }) => {
      const m = mesh.material as THREE.MeshStandardMaterial;
      m.color.setRGB(material.baseColor[0]!, material.baseColor[1]!, material.baseColor[2]!);
    }),
  };
});

vi.mock('../src/pipeline/WalkaroundGPUPipeline.js', async () => {
  const g = globalThis as unknown as { __HYBRID_MAT_STATE__?: MatUpdateState };
  if (!g.__HYBRID_MAT_STATE__) {
    g.__HYBRID_MAT_STATE__ = {
      pipelineInitDeferreds: [],
      pipelineConstructed: [],
      buildBVHCalls: [],
    };
  }
  const state = g.__HYBRID_MAT_STATE__;

  class MockWalkaroundGPUPipeline {
    public readonly index: number;
    public initialized = false;
    public disposed = false;
    public requestAccumReset = vi.fn();
    public refreshBvhMaterialSlice = vi.fn();
    public refreshBvhFullRebuild = vi.fn();
    public updateEmitters = vi.fn();
    public dispose: ReturnType<typeof vi.fn>;
    public resize = vi.fn();
    public presentLastFrame = vi.fn();
    public setDDGIInputs = vi.fn();
    public renderFrame = vi.fn();
    public refreshBvhRefit = vi.fn();

    constructor(_device: GPUDevice, _w: number, _h: number) {
      this.index = state.pipelineInitDeferreds.length;
      let resolve!: () => void;
      let reject!: (e: unknown) => void;
      const promise = new Promise<void>((res, rej) => { resolve = res; reject = rej; });
      state.pipelineInitDeferreds.push({ promise, resolve, reject });
      this.dispose = vi.fn(() => { this.disposed = true; }) as ReturnType<typeof vi.fn>;
      state.pipelineConstructed.push(this);
    }

    async initialize(): Promise<void> {
      await state.pipelineInitDeferreds[this.index]!.promise;
      this.initialized = true;
    }
  }

  return {
    WalkaroundGPUPipeline: MockWalkaroundGPUPipeline,
    HYBRID_WEBGPU_REQUIRED_LIMITS: {},
    HYBRID_WEBGPU_REQUIRED_FEATURES: [],
  };
});

import { HybridEngine } from '../src/HybridEngine.js';
import { asMat4, type Scene } from '@vitrum/core';
import { buildReSTIRSceneBVH } from '../src/restir/bvhCompute.js';

function getState(): MatUpdateState {
  const g = globalThis as unknown as { __HYBRID_MAT_STATE__?: MatUpdateState };
  if (!g.__HYBRID_MAT_STATE__) {
    g.__HYBRID_MAT_STATE__ = {
      pipelineInitDeferreds: [],
      pipelineConstructed: [],
      buildBVHCalls: [],
    };
  }
  return g.__HYBRID_MAT_STATE__;
}

function makeMockDevice(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;
}

function makeScene(): Scene {
  return {
    primitives: [{
      id: 'mesh-a',
      kind: 'mesh',
      transform: asMat4(new Float32Array(16)),
      positions: new Float32Array(9),
      normals: new Float32Array(9),
      material: { baseColor: [0.6, 0.57, 0.55], roughness: 0.8, metallic: 0 },
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

async function waitForPipelineCount(n: number, timeoutMs = 1000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (getState().pipelineInitDeferreds.length < n) {
    if (Date.now() > deadline) throw new Error(`waitForPipelineCount(${n}) timed out`);
    await new Promise((r) => setTimeout(r, 10));
  }
}

async function drainMicrotasks(): Promise<void> {
  for (let i = 0; i < 20; i++) await Promise.resolve();
  await new Promise((r) => setTimeout(r, 50));
  for (let i = 0; i < 20; i++) await Promise.resolve();
}

async function initEngine(engine: HybridEngine): Promise<void> {
  engine.setScene(makeScene());
  await waitForPipelineCount(1);
  getState().pipelineInitDeferreds[0]!.resolve();
  await drainMicrotasks();
}

describe('HybridEngine.updatePrimitive — material patch (PR-1)', () => {
  beforeEach(() => {
    const s = getState();
    s.pipelineInitDeferreds = [];
    s.pipelineConstructed = [];
    s.buildBVHCalls = [];
    vi.clearAllMocks();
  });

  it('T-1.1: material patch does not call setScene / second BVH build', async () => {
    const engine = new HybridEngine({
      device: makeMockDevice(),
      width: 64,
      height: 64,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      skyTint: [1, 1, 1],
      skyIrradiance: 1,
    });
    await initEngine(engine);
    const buildsAfterInit = vi.mocked(buildReSTIRSceneBVH).mock.calls.length;

    engine.updatePrimitive!('mesh-a', {
      material: { baseColor: [1, 0, 0], roughness: 0.5, metallic: 0 },
    });

    expect(vi.mocked(buildReSTIRSceneBVH).mock.calls.length).toBe(buildsAfterInit);
    const pipeline = getState().pipelineConstructed[0] as {
      refreshBvhMaterialSlice: ReturnType<typeof vi.fn>;
      refreshBvhFullRebuild: ReturnType<typeof vi.fn>;
    };
    expect(pipeline.refreshBvhMaterialSlice).toHaveBeenCalled();
    expect(pipeline.refreshBvhFullRebuild).not.toHaveBeenCalled();
  });

  it('T-1.2: bvhIndex.w byte lane changes for the affected triangle', async () => {
    const engine = new HybridEngine({
      device: makeMockDevice(),
      width: 64,
      height: 64,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      skyTint: [1, 1, 1],
      skyIrradiance: 1,
    });
    await initEngine(engine);

    const bvhBefore = (engine as unknown as { _bvhBuffers: { bvhIndex: { cpuData: ArrayBuffer } } })._bvhBuffers;
    const wBefore = new Uint32Array(bvhBefore.bvhIndex.cpuData)[3]!;

    engine.updatePrimitive!('mesh-a', {
      material: { baseColor: [1, 0, 0], roughness: 0.5, metallic: 0 },
    });

    const bvhAfter = (engine as unknown as { _bvhBuffers: { bvhIndex: { cpuData: ArrayBuffer } } })._bvhBuffers;
    const wAfter = new Uint32Array(bvhAfter.bvhIndex.cpuData)[3]!;
    expect(wAfter).not.toBe(wBefore);
    expect((wAfter >> 24) & 0xff).toBe(255);
  });

  it('T-1.3: transmission 0→0.5 invalidates DDGI probe cache', async () => {
    const engine = new HybridEngine({
      device: makeMockDevice(),
      width: 64,
      height: 64,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 1,
      skyTint: [1, 1, 1],
      skyIrradiance: 1,
    });
    await initEngine(engine);

    const ddgi = (engine as unknown as { _ddgi: { invalidateProbeCache: ReturnType<typeof vi.fn> } })._ddgi;
    ddgi.invalidateProbeCache.mockClear();

    engine.updatePrimitive!('mesh-a', {
      material: {
        baseColor: [0.9, 0.9, 0.9],
        roughness: 0,
        metallic: 0,
        transmission: 0.5,
        ior: 1.5,
      },
    });

    expect(ddgi.invalidateProbeCache).toHaveBeenCalled();
  });
});
