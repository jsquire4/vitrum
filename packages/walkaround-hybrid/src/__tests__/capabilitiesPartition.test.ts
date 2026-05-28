/**
 * Capability-set reconciliation + partitionSceneBySupport wiring for
 * walkaround-hybrid.
 *
 * Pins two invariants the 2026-05-28 reconciliation locked in:
 *   1. The declared `supported*Kinds` match what the engine genuinely
 *      RENDERS, not just what it ingests:
 *        - primitives: mesh / skinned-mesh / instanced-mesh (the
 *          vitrumSceneToThree BVH+DDGI path; instanced-mesh is genuine here
 *          via the TLAS per-instance traversal); NOT analytic (no THREE
 *          conversion path — the converter throws on it).
 *        - emitters: rect-area / disc-area / point / spot / mesh-area
 *          (rect/disc → ReSTIR-DI tris + DDGI fixtures; mesh-area → mesh
 *          emissive; point/spot → DDGI fixture lights). `directional` is NOT
 *          supported: the DDGI sun is config-driven via the constructor /
 *          updateLighting, not a scene emitter — coreEmittersToDDGILights
 *          returns null for directional, so a scene directional produces no
 *          light and must be warn-skipped rather than silently dropped.
 *   2. `setScene` filters the scene through `partitionSceneBySupport(scene,
 *      this.capabilities)` BEFORE any vitrumSceneToThree conversion, so an
 *      unsupported node (`analytic` primitive / `directional` emitter) is
 *      warn-skipped (NOT thrown / NOT silently flowed through) and
 *      `_lastScene` holds only the supported subset.
 *
 * The engine is constructed directly (not via the factory, which bootstraps
 * with an empty scene) against a minimal duck-typed GPUDevice stub. Only the
 * SYNCHRONOUS `setScene` filter + the declared capability sets are asserted;
 * the fire-and-forget async init chain is aborted via dispose() right after.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { MockInstance } from 'vitest';
import * as THREE from 'three';
import type { Scene, ScenePrimitive, SceneEmitter } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { HybridEngine } from '../HybridEngine.js';
import type { HybridEngineOptions } from '../HybridEngine.js';

// The partition filter under test runs SYNCHRONOUSLY in setScene, before the
// fire-and-forget async init chain touches three-bindings. Stub the converter
// (returns a real empty THREE.Scene — valid for the CPU BVH builder) + the
// disposer (no-op) so the background chain + dispose() can't surface unhandled
// rejections from real THREE texture disposal. solveSkin / material helpers are
// not hit on this path but must exist for module resolution.
vi.mock('@vitrum/three-bindings', () => ({
  vitrumSceneToThree: () => new THREE.Scene(),
  disposeVitrumThreeSceneRoot: () => undefined,
  solveSkin: () => ({ positions: new Float32Array(0), normals: new Float32Array(0) }),
  applyVitrumMaterialToMesh: () => undefined,
  findMeshByPrimitiveId: () => null,
}));

/** Minimal GPUDevice stub — HybridEngine's constructor only stores the device
 *  (DDGI is CPU-side) and the factory's duck-type check needs
 *  `createCommandEncoder`. The async init chain is aborted via dispose() before
 *  it touches any of these. */
function makeDeviceStub(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    createBuffer: vi.fn(),
    createTexture: vi.fn(),
    createBindGroup: vi.fn(),
    queue: { submit: vi.fn(), writeBuffer: vi.fn() },
  } as unknown as GPUDevice;
}

function makeOpts(): HybridEngineOptions {
  return {
    device: makeDeviceStub(),
    width: 64,
    height: 64,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 3,
    skyTint: [1, 1, 1],
    skyIrradiance: 1,
  };
}

function meshPrimitive(id: string): ScenePrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
  };
}

function instancedMeshPrimitive(id: string): ScenePrimitive {
  return {
    kind: 'instanced-mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0 },
    instances: [
      asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
    ],
  };
}

function analyticPrimitive(id: string): ScenePrimitive {
  return {
    kind: 'analytic',
    id,
    shape: 'sphere',
    params: new Float32Array([0, 0, 0, 1]),
    material: { baseColor: [1, 0, 0], roughness: 0.3, metallic: 0 },
  };
}

function rectEmitter(id: string): SceneEmitter {
  return {
    kind: 'rect-area',
    id,
    position: [0, 5, 0],
    uAxis: [1, 0, 0],
    vAxis: [0, 0, 1],
    color: [1, 1, 1],
    intensity: 4,
  };
}

function directionalEmitter(id: string): SceneEmitter {
  return {
    kind: 'directional',
    id,
    direction: [0, -1, 0],
    color: [1, 1, 1],
    intensity: 3,
  };
}

function sceneWith(primitives: ScenePrimitive[], emitters: SceneEmitter[] = []): Scene {
  return { primitives, emitters, environment: { kind: 'none' } };
}

describe('walkaround-hybrid capability/partition reconciliation', () => {
  let warnSpy: MockInstance | undefined;
  // The fire-and-forget async init chain fails in this non-WebGPU test env
  // (`GPUBufferUsage` is undefined) and logs via console.error before the
  // chain's catch sets state to 'error'. That's expected and orthogonal to
  // the synchronous partition behaviour under test — silence it so the
  // expected noise doesn't pollute output. (The partition *warnings* go
  // through console.warn, which the relevant tests spy on explicitly.)
  let errorSpy: MockInstance | undefined;
  beforeEach(() => {
    errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
  });
  afterEach(() => {
    warnSpy?.mockRestore();
    errorSpy?.mockRestore();
  });

  it('declares mesh / skinned-mesh / instanced-mesh and NOT analytic', () => {
    const engine = new HybridEngine(makeOpts());
    try {
      const kinds = engine.capabilities.supportedPrimitiveKinds!;
      expect(kinds.has('mesh')).toBe(true);
      expect(kinds.has('skinned-mesh')).toBe(true);
      expect(kinds.has('instanced-mesh')).toBe(true);
      expect(kinds.has('analytic')).toBe(false);
    } finally {
      engine.dispose();
    }
  });

  it('declares rect-area / disc-area / point / spot / mesh-area emitters and NOT directional', () => {
    const engine = new HybridEngine(makeOpts());
    try {
      const kinds = engine.capabilities.supportedEmitterKinds;
      for (const k of ['rect-area', 'disc-area', 'point', 'spot', 'mesh-area'] as const) {
        expect(kinds.has(k)).toBe(true);
      }
      // directional is config-driven (constructor/updateLighting), not a scene
      // emitter — coreEmittersToDDGILights returns null for it.
      expect(kinds.has('directional')).toBe(false);
    } finally {
      engine.dispose();
    }
  });

  it('keeps an instanced-mesh + rect-area scene intact (instanced-mesh is genuine via TLAS)', () => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = new HybridEngine(makeOpts());
    try {
      engine.setScene(sceneWith([instancedMeshPrimitive('inst-a')], [rectEmitter('rect-a')]));

      // No skip-warning for the supported nodes.
      expect(warnSpy.mock.calls.flat().some((m) => String(m).includes('not supported'))).toBe(false);

      // The stored scene retains the supported primitive + emitter.
      const stored = (engine as unknown as { _lastScene: Scene })._lastScene;
      expect(stored.primitives.map((p) => String(p.id))).toContain('inst-a');
      expect(stored.emitters.map((e) => String(e.id))).toContain('rect-a');
    } finally {
      engine.dispose();
    }
  });

  it('warn-skips a directional emitter instead of silently dropping it, keeping supported emitters', () => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = new HybridEngine(makeOpts());
    try {
      engine.setScene(
        sceneWith([meshPrimitive('mesh-a')], [rectEmitter('rect-a'), directionalEmitter('sun-a')]),
      );

      // A warning was emitted naming the directional emitter — the DDGI sun is
      // config-driven (constructor/updateLighting), so a scene directional
      // produces no light and is dropped rather than flowed through to a no-op.
      const warned = warnSpy.mock.calls.flat().map(String);
      expect(warned.some((m) => m.includes('sun-a') && m.includes('not supported'))).toBe(true);

      // The stored scene keeps the supported rect-area emitter, drops directional.
      const stored = (engine as unknown as { _lastScene: Scene })._lastScene;
      expect(stored.emitters.map((e) => String(e.id))).toEqual(['rect-a']);
      expect(stored.emitters.some((e) => e.kind === 'directional')).toBe(false);
    } finally {
      engine.dispose();
    }
  });

  it('warn-skips an analytic primitive instead of throwing, keeping supported nodes', () => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = new HybridEngine(makeOpts());
    try {
      // Behavior change: analytic was effectively a THROW (vitrumSceneToThree at
      // BVH build); now it is warn-skipped at setScene before conversion.
      expect(() =>
        engine.setScene(sceneWith([meshPrimitive('mesh-a'), analyticPrimitive('sphere-a')])),
      ).not.toThrow();

      const warned = warnSpy.mock.calls.flat().map(String);
      expect(warned.some((m) => m.includes('sphere-a') && m.includes('not supported'))).toBe(true);

      // The stored scene holds only the supported mesh — analytic dropped
      // BEFORE the converter runs, so its throw never fires.
      const stored = (engine as unknown as { _lastScene: Scene })._lastScene;
      expect(stored.primitives.map((p) => String(p.id))).toEqual(['mesh-a']);
      expect(stored.primitives.some((p) => p.kind === 'analytic')).toBe(false);
    } finally {
      engine.dispose();
    }
  });
});
