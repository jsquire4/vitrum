import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import { buildCornellBoxThreeScene } from '@vitrum-examples/shared';
import { sceneFromThreeJS } from '@vitrum/three-bindings';
import { createPTEngine_WebGL2 } from '@vitrum/pt-webgl';
import { createWalkaroundEngine_Hybrid } from '@vitrum/walkaround-hybrid';
import type { Scene } from '@vitrum/core';

type UniformRef = { value: unknown };

interface SmokeState {
  readonly pipelines: Array<{ initialized: boolean; disposed: boolean }>;
  readonly bvhBuilds: unknown[];
}

function smokeState(): SmokeState {
  const g = globalThis as unknown as { __TWO_ENGINES_BACKEND_SMOKE__?: SmokeState };
  if (!g.__TWO_ENGINES_BACKEND_SMOKE__) {
    g.__TWO_ENGINES_BACKEND_SMOKE__ = { pipelines: [], bvhBuilds: [] };
  }
  return g.__TWO_ENGINES_BACKEND_SMOKE__;
}

vi.mock('three-gpu-pathtracer', () => {
  function makeUniforms(): Record<string, UniformRef> {
    const table = [
      'uCmfX',
      'uCmfY',
      'uCmfZ',
      'uXCmfCdf',
      'uYCmfCdf',
      'uZCmfCdf',
      'uXCmfIntegral',
      'uYCmfIntegral',
      'uZCmfIntegral',
      'uSpectralRendering',
      'uRadianceClamp',
      'u_jakobCoeffs',
      'uCausticStrategy',
      'uMneeMaxIterations',
      'uMneeMaxChainLength',
      'uBdptEnabled',
      'uBdptLightPathTex',
      'uBdptMaxLightBounces',
    ];
    return Object.fromEntries(
      table.map((name) => [
        name,
        {
          value:
            name === 'u_jakobCoeffs'
              ? { x: 0, y: 0, z: 0, set(x: number, y: number, z: number): void {
                this.x = x;
                this.y = y;
                this.z = z;
              } }
              : null,
        },
      ]),
    );
  }

  class WebGLPathTracer {
    public readonly _pathTracer = {
      material: {
        uniforms: makeUniforms(),
        setDefine: vi.fn(),
      },
    };

    public readonly target = { texture: {} };
    public readonly tiles = { set: vi.fn() };
    public renderDelay = 0;
    public minSamples = 0;
    public fadeDuration = 0;
    public samples = 0;
    public setScene = vi.fn(() => { this.samples = 0; });
    public setCamera = vi.fn();
    public renderSample = vi.fn(() => { this.samples += 1; });
    public reset = vi.fn(() => { this.samples = 0; });
    public dispose = vi.fn();
    public updateEnvironment = vi.fn(() => { this.samples = 0; });
  }

  return { WebGLPathTracer };
});

vi.mock('../../../packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.js', () => {
  class MockWalkaroundGPUPipeline {
    public initialized = false;
    public disposed = false;
    public frameResources = null;
    public gpuMemoryExternalSections = {};
    public requestAccumReset = vi.fn();
    public presentLastFrame = vi.fn();
    public renderFrame = vi.fn();
    public resize = vi.fn();
    public setDDGIInputs = vi.fn();
    public setDenoiserPassEnabled = vi.fn();
    public dispose = vi.fn(() => { this.disposed = true; });
    public getDebugTextures = vi.fn(() => null);

    constructor() {
      smokeState().pipelines.push(this);
    }

    async initialize(): Promise<void> {
      this.initialized = true;
    }
  }

  return {
    WalkaroundGPUPipeline: MockWalkaroundGPUPipeline,
    HYBRID_WEBGPU_REQUIRED_LIMITS: {},
    HYBRID_LITE_LIMITS: {},
    HYBRID_WEBGPU_REQUIRED_FEATURES: [],
  };
});

vi.mock('../../../packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts', () => {
  class MockWalkaroundGPUPipeline {
    public initialized = false;
    public disposed = false;
    public frameResources = null;
    public gpuMemoryExternalSections = {};
    public requestAccumReset = vi.fn();
    public presentLastFrame = vi.fn();
    public renderFrame = vi.fn();
    public resize = vi.fn();
    public setDDGIInputs = vi.fn();
    public setDenoiserPassEnabled = vi.fn();
    public dispose = vi.fn(() => { this.disposed = true; });
    public getDebugTextures = vi.fn(() => null);

    constructor() {
      smokeState().pipelines.push(this);
    }

    async initialize(): Promise<void> {
      this.initialized = true;
    }
  }

  return {
    WalkaroundGPUPipeline: MockWalkaroundGPUPipeline,
    HYBRID_WEBGPU_REQUIRED_LIMITS: {},
    HYBRID_LITE_LIMITS: {},
    HYBRID_WEBGPU_REQUIRED_FEATURES: [],
  };
});

// THREE-decouple (2026-06-08): the live walkaround setScene core path builds its
// BVH via restir/bvhCore.ts (buildReSTIRSceneBVHForCoreScene), NOT the removed
// restir/bvhCompute.{js,ts}. Mock the module the engine actually imports so the
// `bvhBuilds` smoke signal fires and so the real (CPU+GPU) BVH build is skipped.
function makeBvhCoreMock(): Record<string, unknown> {
  const buf = (bytes: number, count = 1): unknown => ({
    cpuData: new ArrayBuffer(bytes),
    count,
    byteLength: bytes,
  });
  function makeBuffers(): unknown {
    const state = smokeState();
    state.bvhBuilds.push({});
    return {
      bvhMode: 'merged' as const,
      bvhNodes: buf(32),
      bvhIndex: buf(16),
      bvhPositions: buf(16),
      triangleMaterialIds: buf(4, 0),
      bvhBeerColors: buf(16),
      bvhEmissiveLe: buf(16),
      bvhNormals: buf(16),
      emitters: buf(16, 0),
      emitterCdf: buf(16, 0),
      emitterCount: 0,
      totalEmissivePower: 0,
      lightTree: buf(16, 0),
      lightTreeNodeCount: 0,
      lightTreeEnabled: false,
      // THREE-free merged-geometry handle (RestirMergedGeometryLike).
      mergedGeometry: { boundingBox: null, computeBoundingBox: () => undefined, dispose: () => undefined },
      meshVertexRanges: [],
      bvhIndicesStride3: new Uint32Array(0),
      buildMaterials: [],
      coreMaterials: [],
      emitterNormals: new Float32Array(0),
    };
  }
  return {
    buildReSTIRSceneBVHForCoreScene: vi.fn(makeBuffers),
    rebuildReSTIRSceneBVHPrimitiveCore: vi.fn(makeBuffers),
    resolveReSTIRBvhMode: vi.fn(() => 'merged' as const),
    rebuildEmitterBuffersFromCoreScene: vi.fn(() => ({
      emitters: buf(16, 0),
      emitterCdf: buf(16, 0),
      emitterCount: 0,
      totalEmissivePower: 0,
    })),
    disposeSceneBVH: vi.fn(),
  };
}

vi.mock('../../../packages/walkaround-hybrid/src/restir/bvhCore.js', () => makeBvhCoreMock());
vi.mock('../../../packages/walkaround-hybrid/src/restir/bvhCore.ts', () => makeBvhCoreMock());

class FakeWebGL2RenderingContext {
  readonly MAX_FRAGMENT_UNIFORM_VECTORS = 0x8dfd;
  readonly MAX_TEXTURE_SIZE = 0x0d33;
  readonly MAX_RENDERBUFFER_SIZE = 0x84e8;
  readonly RENDERER = 0x1f01;

  getExtension(_name: string): null {
    return null;
  }

  getParameter(param: number): number | string {
    if (param === this.MAX_FRAGMENT_UNIFORM_VECTORS) return 512;
    if (param === this.MAX_TEXTURE_SIZE || param === this.MAX_RENDERBUFFER_SIZE) return 8192;
    if (param === this.RENDERER) return 'Fake WebGL2';
    return 0;
  }
}

function makeRendererStub(): THREE.WebGLRenderer {
  return {
    getContext: () => new FakeWebGL2RenderingContext(),
    domElement: { addEventListener: vi.fn(), removeEventListener: vi.fn() },
    setSize: vi.fn(),
  } as unknown as THREE.WebGLRenderer;
}

function makeGpuDeviceStub(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(() => ({ finish: vi.fn() })),
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;
}

function makeCoreScene(): Scene {
  return sceneFromThreeJS(buildCornellBoxThreeScene());
}

async function drain(): Promise<void> {
  for (let i = 0; i < 20; i += 1) await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

describe('two-engines-one-scene backend smoke', () => {
  beforeAll(() => {
    (globalThis as unknown as { WebGL2RenderingContext?: unknown }).WebGL2RenderingContext =
      FakeWebGL2RenderingContext;
  });

  afterAll(() => {
    delete (globalThis as unknown as { WebGL2RenderingContext?: unknown }).WebGL2RenderingContext;
  });

  beforeEach(() => {
    smokeState().pipelines.length = 0;
    smokeState().bvhBuilds.length = 0;
  });

  it('passes the shared core Scene through the pt-webgl backend setScene path', async () => {
    const scene = makeCoreScene();
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() });

    engine.setScene(scene);

    expect(engine.state).toBe('ready');
    expect(engine.getScene()?.primitives.map((p) => p.id)).toEqual(
      scene.primitives.map((p) => p.id),
    );
    expect(engine.getSceneTlasAudit()?.meshLikePrimitiveCount).toBeGreaterThan(0);
    engine.dispose();
  });

  it('passes the shared core Scene through the walkaround-hybrid backend setScene path', async () => {
    const scene = makeCoreScene();
    const engine = await createWalkaroundEngine_Hybrid({
      device: makeGpuDeviceStub(),
      width: 64,
      height: 64,
      primaryLightDir: [0, -1, 0],
      primaryLightIntensity: 0,
      skyTint: [1, 1, 1],
      skyIrradiance: 0,
    });

    engine.setScene(scene);
    await drain();

    expect(engine.getScene()?.primitives.map((p) => p.id)).toEqual(
      scene.primitives.map((p) => p.id),
    );
    expect(smokeState().bvhBuilds.length).toBeGreaterThan(0);
    expect(smokeState().pipelines.some((p) => p.initialized)).toBe(true);
    engine.dispose();
  });
});
