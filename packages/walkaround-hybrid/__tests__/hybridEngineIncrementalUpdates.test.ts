/**
 * Items A3 — HybridEngine.updatePrimitive / updateEmitter tests.
 *
 * Pre-A3: both methods were typed `never`. Any host wanting to scrub a
 * material colour or rotate an emitter had to call `setScene()` (which
 * triggers `_teardownPipeline()` + `_initPipeline()` synchronously — full
 * BVH rebuild + shader recompile + DDGI atlas teardown) or recreate the
 * engine. ~500-2000 ms of stall per slider tick on Cornell-scale scenes.
 *
 * A3: material-only fast path that re-uploads per-triangle material bytes
 * (`bvhIndex.w` + `bvhBeerColors`) and the emitter list + power CDF via
 * `queue.writeBuffer` / destroy+reupload — no BVH rebuild, no shader
 * recompile. Geometry / topology patches still throw (audit-approved
 * conservative behaviour for round 1).
 *
 * These tests assert:
 *   - `EngineCapabilities.supportsIncrementalScene === true`.
 *   - updatePrimitive(material-only) does NOT trigger _initPipeline
 *     (no new pipeline constructed, no buildReSTIRSceneBVH call).
 *   - The pipeline's `updateMaterialsBytes` was called → GPU buffers got
 *     fresh bytes.
 *   - Accumulator + DDGI atlas were reset so the change converges visually.
 *   - updatePrimitive on a geometry patch throws with a setScene pointer.
 *   - updatePrimitive on an unknown id throws.
 *   - updateEmitter(directional) routes to updateLighting (sun-state path).
 *   - updateEmitter(rect-area) calls pipeline.updateEmitters.
 *   - updateEmitter(mesh-area) throws with an updatePrimitive pointer.
 *   - The on-disk bvhIndex bytes contain the new baseColor (R*255 in the
 *     high byte of the .w slot) — full round-trip through real packing.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';
import type { Scene, MeshPrimitive } from '@vitrum/core';

// ── Module-level state for the mocks ────────────────────────────────────────

interface State {
  pipelineInitDeferreds: Array<{ promise: Promise<void>; resolve: () => void; reject: (e: unknown) => void }>;
  pipelineConstructed: MockPipeline[];
  buildBVHCalls: number;
  updateMaterialsBytesCalls: Array<{ pipelineIdx: number; bvhIndexBytes: ArrayBuffer; beerBytes: ArrayBuffer }>;
  updateEmittersCalls: Array<{ pipelineIdx: number }>;
  requestAccumResetCalls: number;
}

interface MockPipeline {
  index: number;
  initialized: boolean;
  disposed: boolean;
  dispose: ReturnType<typeof vi.fn>;
  resize: ReturnType<typeof vi.fn>;
  requestAccumReset: ReturnType<typeof vi.fn>;
  updateMaterialsBytes: ReturnType<typeof vi.fn>;
  updateEmitters: ReturnType<typeof vi.fn>;
  setDDGIInputs: ReturnType<typeof vi.fn>;
  renderFrame: ReturnType<typeof vi.fn>;
  presentLastFrame: ReturnType<typeof vi.fn>;
}

vi.mock('../src/pipeline/WalkaroundGPUPipeline.js', async () => {
  const g = globalThis as unknown as { __A3_STATE__?: State };
  if (!g.__A3_STATE__) {
    g.__A3_STATE__ = {
      pipelineInitDeferreds: [],
      pipelineConstructed: [],
      buildBVHCalls: 0,
      updateMaterialsBytesCalls: [],
      updateEmittersCalls: [],
      requestAccumResetCalls: 0,
    };
  }
  const state = g.__A3_STATE__;

  class MockWalkaroundGPUPipeline implements MockPipeline {
    public readonly index: number;
    public initialized = false;
    public disposed = false;
    public dispose: ReturnType<typeof vi.fn>;
    public resize: ReturnType<typeof vi.fn>;
    public requestAccumReset: ReturnType<typeof vi.fn>;
    public updateMaterialsBytes: ReturnType<typeof vi.fn>;
    public updateEmitters: ReturnType<typeof vi.fn>;
    public setDDGIInputs = vi.fn();
    public renderFrame = vi.fn();
    public presentLastFrame = vi.fn();

    constructor(_device: GPUDevice, _w: number, _h: number) {
      this.index = state.pipelineInitDeferreds.length;
      let resolve!: () => void;
      let reject!: (e: unknown) => void;
      const promise = new Promise<void>((res, rej) => { resolve = res; reject = rej; });
      state.pipelineInitDeferreds.push({ promise, resolve, reject });
      this.dispose = vi.fn(() => { this.disposed = true; });
      this.resize = vi.fn();
      this.requestAccumReset = vi.fn(() => { state.requestAccumResetCalls++; });
      const idx = this.index;
      this.updateMaterialsBytes = vi.fn((bvhIndexBytes: ArrayBuffer, beerBytes: ArrayBuffer) => {
        state.updateMaterialsBytesCalls.push({ pipelineIdx: idx, bvhIndexBytes, beerBytes });
      });
      this.updateEmitters = vi.fn(() => {
        state.updateEmittersCalls.push({ pipelineIdx: idx });
      });
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

// Mock buildReSTIRSceneBVH so we don't need a real GPU. The returned struct
// uses real-shape data (real-sized typed arrays + a real THREE.Material
// reference inside materialsLut) so the engine's repack path runs against
// real `packBVHIndexW` / `packBVHBeerColors`.
vi.mock('../src/restir/bvhCompute.js', async () => {
  const g = globalThis as unknown as { __A3_STATE__?: State };
  if (g.__A3_STATE__) g.__A3_STATE__.buildBVHCalls = 0;

  return {
    buildReSTIRSceneBVH: vi.fn((roots: THREE.Object3D[]) => {
      if (g.__A3_STATE__) g.__A3_STATE__.buildBVHCalls++;
      // Collect THREE.Material instances from the root in mesh-traversal order
      // so the materialsLut order matches what a real build would produce.
      const materials: THREE.Material[] = [];
      const matIdByInstance = new Map<THREE.Material, number>();
      for (const root of roots) {
        root.traverse((obj) => {
          if (!(obj as THREE.Mesh).isMesh) return;
          const m = (obj as THREE.Mesh).material;
          const mat = Array.isArray(m) ? m[0] : m;
          if (!mat || matIdByInstance.has(mat)) return;
          matIdByInstance.set(mat, materials.length);
          materials.push(mat);
        });
      }
      // Two-triangle BVH where tri 0 → material 0, tri 1 → material 1
      // (matching the 2-primitive test scene).
      const triCount = Math.max(1, materials.length);
      const triMaterialIds = new Uint32Array(triCount);
      for (let i = 0; i < triCount; i++) triMaterialIds[i] = i;
      // Real packing produces 16 bytes/tri (4 u32 lanes). We seed with zeros
      // and let the engine's repack produce the real bytes on update.
      const bvhIndexBytes = new Uint32Array(triCount * 4);
      const beerBytes = new Uint32Array(triCount);
      // Stride-3 indices (one tri = vertices 0,1,2 → 3,4,5 → ...).
      const mergedIndices = new Uint32Array(triCount * 3);
      for (let t = 0; t < triCount; t++) {
        mergedIndices[t * 3 + 0] = t * 3 + 0;
        mergedIndices[t * 3 + 1] = t * 3 + 1;
        mergedIndices[t * 3 + 2] = t * 3 + 2;
      }
      const vertCount = triCount * 3;
      const mergedPositionsStride4 = new Float32Array(vertCount * 4);
      const mergedNormalsStride4 = new Float32Array(vertCount * 4);
      // Fill positions with a tiny triangle per primitive so emitter-build
      // math doesn't divide by zero. Triangle t at world position (t, 0, 0).
      for (let t = 0; t < triCount; t++) {
        const base = t * 3 * 4;
        mergedPositionsStride4[base + 0] = t;       // v0.x
        mergedPositionsStride4[base + 4] = t + 1;   // v1.x
        mergedPositionsStride4[base + 9] = 1;       // v2.y
        // Normal pointing +y so it can serve as a top-down light surface.
        mergedNormalsStride4[base + 1] = 1;
        mergedNormalsStride4[base + 5] = 1;
        mergedNormalsStride4[base + 9] = 1;
      }
      return {
        bvhNodes:        { cpuData: new ArrayBuffer(32), byteLength: 32, count: 1 },
        bvhIndex:        { cpuData: bvhIndexBytes.buffer, byteLength: bvhIndexBytes.byteLength, count: triCount },
        bvhBeerColors:   { cpuData: beerBytes.buffer, byteLength: beerBytes.byteLength, count: triCount },
        bvhPositions:    { cpuData: mergedPositionsStride4.buffer, byteLength: mergedPositionsStride4.byteLength, count: vertCount },
        triangleMaterialIds: { cpuData: triMaterialIds.buffer, byteLength: triMaterialIds.byteLength, count: triCount },
        emitters:        { cpuData: new ArrayBuffer(80), byteLength: 80, count: 1 },
        emitterCdf:      { cpuData: new ArrayBuffer(4),  byteLength: 4,  count: 1 },
        emitterCount:    1,
        totalEmissivePower: 1,
        mergedGeometry:  new THREE.BufferGeometry(),
        materialsLut: materials,
        mergedIndices,
        mergedPositionsStride4,
        mergedNormalsStride4,
      };
    }),
    disposeSceneBVH: vi.fn(),
  };
});

// Use the real vitrumSceneToThree path so the synthesized THREE root has
// real meshes / materials / RectAreaLights with `.name === primitive.id`
// — the updatePrimitive / updateEmitter lookups depend on that wiring.

import { HybridEngine } from '../src/HybridEngine.js';

function getState(): State {
  return (globalThis as unknown as { __A3_STATE__: State }).__A3_STATE__;
}

function makeMockDevice(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
  } as unknown as GPUDevice;
}

function makeEngine(): HybridEngine {
  return new HybridEngine({
    device:                makeMockDevice(),
    width:                 64,
    height:                64,
    primaryLightDir:       [0, -1, 0],
    primaryLightIntensity: 1.0,
    skyTint:               [1, 1, 1],
    skyIrradiance:         1.0,
  });
}

function makeTwoPrimScene(): Scene {
  const prim0: MeshPrimitive = {
    kind: 'mesh',
    id: 'prim-0',
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals:   new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    indices:   new Uint32Array([0, 1, 2]),
    material: { baseColor: [0, 1, 0], roughness: 0.5, metallic: 0 },
  };
  const prim1: MeshPrimitive = {
    kind: 'mesh',
    id: 'prim-1',
    positions: new Float32Array([2, 0, 0, 3, 0, 0, 2, 1, 0]),
    normals:   new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    indices:   new Uint32Array([0, 1, 2]),
    material: { baseColor: [0, 0, 1], roughness: 0.5, metallic: 0 },
  };
  return {
    primitives: [prim0, prim1],
    emitters: [
      { kind: 'directional', id: 'sun', color: [1, 1, 1], intensity: 1.0, direction: [0, -1, 0] },
      {
        kind: 'rect-area', id: 'panel',
        color: [1, 1, 1], intensity: 5.0,
        position: [0, 5, 0],
        uAxis: [1, 0, 0],
        vAxis: [0, 0, 1],
      },
    ],
    environment: { kind: 'none' },
  };
}

async function waitForPipelineCount(n: number, timeoutMs = 1000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (getState().pipelineConstructed.length < n) {
    if (Date.now() > deadline) throw new Error(`waitForPipelineCount(${n}) timed out`);
    await new Promise((r) => setTimeout(r, 10));
  }
}

async function drainMicrotasks(): Promise<void> {
  for (let i = 0; i < 20; i++) await Promise.resolve();
  await new Promise((r) => setTimeout(r, 50));
  for (let i = 0; i < 20; i++) await Promise.resolve();
}

async function bootEngineWithScene(scene: Scene): Promise<HybridEngine> {
  const engine = makeEngine();
  const s = getState();
  // The HybridEngine ctor does not auto-bootstrap (that's the factory's job),
  // so setScene drives the single pipeline init.
  engine.setScene(scene);
  await waitForPipelineCount(1);
  for (const d of s.pipelineInitDeferreds) d.resolve();
  await drainMicrotasks();
  return engine;
}

beforeEach(() => {
  const s = getState();
  s.pipelineInitDeferreds.length = 0;
  s.pipelineConstructed.length = 0;
  s.updateMaterialsBytesCalls.length = 0;
  s.updateEmittersCalls.length = 0;
  s.buildBVHCalls = 0;
  s.requestAccumResetCalls = 0;
});

// ─────────────────────────────────────────────────────────────────────────────
// Capability surface
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine capability flag', () => {
  it('reports supportsIncrementalScene === true', () => {
    const engine = makeEngine();
    expect(engine.capabilities.supportsIncrementalScene).toBe(true);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// updatePrimitive — material-only fast path
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine.updatePrimitive — material-only fast path', () => {
  it('exists as a real method (not `never`)', () => {
    const engine = makeEngine();
    expect(typeof engine.updatePrimitive).toBe('function');
  });

  it('throws when called before setScene', () => {
    const engine = makeEngine();
    expect(() =>
      engine.updatePrimitive('prim-0', { material: { baseColor: [1, 0, 0], roughness: 0.5, metallic: 0 } }),
    ).toThrow(/no scene set/);
  });

  it('throws on unknown primitive id', async () => {
    const engine = await bootEngineWithScene(makeTwoPrimScene());
    expect(() =>
      engine.updatePrimitive('nope', { material: { baseColor: [1, 0, 0], roughness: 0.5, metallic: 0 } }),
    ).toThrow(/not found/);
  });

  it('throws when patch contains geometry fields (clear setScene pointer)', async () => {
    const engine = await bootEngineWithScene(makeTwoPrimScene());
    expect(() =>
      engine.updatePrimitive('prim-0', { positions: new Float32Array([0, 0, 0]) } as never),
    ).toThrow(/setScene/);
    expect(() =>
      engine.updatePrimitive('prim-0', { transform: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] } as never),
    ).toThrow(/setScene/);
  });

  it('does NOT trigger a pipeline rebuild on a material-only patch', async () => {
    const engine = await bootEngineWithScene(makeTwoPrimScene());
    const s = getState();

    const pipelinesBefore = s.pipelineConstructed.length;
    const buildsBefore = s.buildBVHCalls;

    engine.updatePrimitive('prim-0', {
      material: { baseColor: [1, 0, 0], roughness: 0.5, metallic: 0 },
    });

    // No new pipeline constructed, no new BVH build.
    expect(s.pipelineConstructed.length).toBe(pipelinesBefore);
    expect(s.buildBVHCalls).toBe(buildsBefore);
  });

  it('calls pipeline.updateMaterialsBytes with non-empty buffers', async () => {
    const engine = await bootEngineWithScene(makeTwoPrimScene());
    const s = getState();
    const live = s.pipelineConstructed[s.pipelineConstructed.length - 1]!;

    engine.updatePrimitive('prim-0', {
      material: { baseColor: [1, 0, 0], roughness: 0.5, metallic: 0 },
    });

    expect(live.updateMaterialsBytes).toHaveBeenCalledTimes(1);
    const lastCall = s.updateMaterialsBytesCalls.at(-1)!;
    expect(lastCall.bvhIndexBytes.byteLength).toBeGreaterThan(0);
    expect(lastCall.beerBytes.byteLength).toBeGreaterThan(0);
  });

  it('re-uploads bvhIndex bytes that reflect the new baseColor (R=255)', async () => {
    const engine = await bootEngineWithScene(makeTwoPrimScene());
    const s = getState();

    engine.updatePrimitive('prim-0', {
      material: { baseColor: [1, 0, 0], roughness: 0.5, metallic: 0 },
    });

    const { bvhIndexBytes } = s.updateMaterialsBytesCalls.at(-1)!;
    // bvhIndex[0*4 + 3] is the .w lane of triangle 0 — packed as
    // (r << 24) | (g << 16) | (b << 8) | (trans4 | isMetal | texType).
    // R=255 → high byte === 0xFF.
    const u32 = new Uint32Array(bvhIndexBytes);
    const tri0W = u32[3]!;
    const r = (tri0W >>> 24) & 0xFF;
    const g = (tri0W >>> 16) & 0xFF;
    const b = (tri0W >>>  8) & 0xFF;
    expect(r).toBe(255);
    expect(g).toBe(0);
    expect(b).toBe(0);
  });

  it('resets the temporal accumulator on material patch', async () => {
    const engine = await bootEngineWithScene(makeTwoPrimScene());
    const s = getState();
    const live = s.pipelineConstructed[s.pipelineConstructed.length - 1]!;
    live.requestAccumReset.mockClear();

    engine.updatePrimitive('prim-0', {
      material: { baseColor: [1, 0, 0], roughness: 0.5, metallic: 0 },
    });

    expect(live.requestAccumReset).toHaveBeenCalledTimes(1);
  });

  it('invalidates the DDGI probe cache on material patch', async () => {
    const engine = await bootEngineWithScene(makeTwoPrimScene());
    const e = engine as unknown as Record<string, unknown>;
    const ddgi = e['_ddgi'] as Record<string, unknown>;
    // Simulate a converged DDGI.
    ddgi['_frame'] = 16;
    ddgi['_ready'] = true;

    engine.updatePrimitive('prim-0', {
      material: { baseColor: [1, 0, 0], roughness: 0.5, metallic: 0 },
    });

    expect(ddgi['_frame']).toBe(0);
    expect(ddgi['_ready']).toBe(false);
  });

  it('is a safe no-op when patch.material is undefined', async () => {
    const engine = await bootEngineWithScene(makeTwoPrimScene());
    const s = getState();
    const live = s.pipelineConstructed[s.pipelineConstructed.length - 1]!;
    live.updateMaterialsBytes.mockClear();
    live.requestAccumReset.mockClear();

    engine.updatePrimitive('prim-0', {});

    expect(live.updateMaterialsBytes).not.toHaveBeenCalled();
    expect(live.requestAccumReset).not.toHaveBeenCalled();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// updateEmitter
// ─────────────────────────────────────────────────────────────────────────────

describe('HybridEngine.updateEmitter', () => {
  it('exists as a real method (not `never`)', () => {
    const engine = makeEngine();
    expect(typeof engine.updateEmitter).toBe('function');
  });

  it('throws when called before setScene', () => {
    const engine = makeEngine();
    expect(() =>
      engine.updateEmitter('sun', { intensity: 2 } as never),
    ).toThrow(/no scene set/);
  });

  it('throws on unknown emitter id', async () => {
    const engine = await bootEngineWithScene(makeTwoPrimScene());
    expect(() =>
      engine.updateEmitter('nope', { intensity: 2 } as never),
    ).toThrow(/not found/);
  });

  it('directional emitter: routes intensity through updateLighting (engine field updated)', async () => {
    const engine = await bootEngineWithScene(makeTwoPrimScene());
    const e = engine as unknown as Record<string, unknown>;
    expect(e['_primaryLightIntensity']).toBe(1.0);

    engine.updateEmitter('sun', { intensity: 3.5 } as never);

    expect(e['_primaryLightIntensity']).toBe(3.5);
  });

  it('directional emitter: direction patch updates _primaryLightDir', async () => {
    const engine = await bootEngineWithScene(makeTwoPrimScene());
    const e = engine as unknown as Record<string, unknown>;

    engine.updateEmitter('sun', { direction: [0.5, -0.866, 0] } as never);

    expect(e['_primaryLightDir']).toEqual([0.5, -0.866, 0]);
  });

  it('rect-area emitter: calls pipeline.updateEmitters and resets accum', async () => {
    const engine = await bootEngineWithScene(makeTwoPrimScene());
    const s = getState();
    const live = s.pipelineConstructed[s.pipelineConstructed.length - 1]!;
    live.requestAccumReset.mockClear();

    engine.updateEmitter('panel', { intensity: 12.0 } as never);

    expect(live.updateEmitters).toHaveBeenCalledTimes(1);
    expect(live.requestAccumReset).toHaveBeenCalledTimes(1);
  });

  it('mesh-area emitter: throws with an updatePrimitive pointer', async () => {
    const scene: Scene = {
      primitives: [
        {
          kind: 'mesh',
          id: 'panel-mesh',
          positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
          normals:   new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
          indices:   new Uint32Array([0, 1, 2]),
          material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
        } as MeshPrimitive,
      ],
      emitters: [
        { kind: 'mesh-area', id: 'area-1', color: [1, 1, 1], intensity: 1, meshId: 'panel-mesh' },
      ],
      environment: { kind: 'none' },
    };
    const engine = await bootEngineWithScene(scene);
    expect(() =>
      engine.updateEmitter('area-1', { intensity: 2 } as never),
    ).toThrow(/updatePrimitive/);
  });

  it('does NOT trigger a pipeline rebuild on rect-area emitter patch', async () => {
    const engine = await bootEngineWithScene(makeTwoPrimScene());
    const s = getState();

    const pipelinesBefore = s.pipelineConstructed.length;
    const buildsBefore = s.buildBVHCalls;

    engine.updateEmitter('panel', { intensity: 12 } as never);

    expect(s.pipelineConstructed.length).toBe(pipelinesBefore);
    expect(s.buildBVHCalls).toBe(buildsBefore);
  });
});
