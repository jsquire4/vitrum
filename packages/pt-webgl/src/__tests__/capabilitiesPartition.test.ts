/**
 * Capability-set reconciliation + partitionSceneBySupport wiring for pt-webgl.
 *
 * Pins these invariants:
 *   1. The declared `supported*Kinds` match what pt-webgl GENUINELY RENDERS,
 *      not just what `vitrumSceneToThree` ingests. `instanced-mesh` IS
 *      supported: the converter builds a single THREE.InstancedMesh, and
 *      pt-webgl's `setScene` expands it into N baked THREE.Mesh instances
 *      (`expandInstancedMeshesInScene`) BEFORE the fork's geometry generator
 *      runs, so each instance renders at its real per-instance world
 *      transform. `analytic` IS accepted by pt-webgl through a generated mesh
 *      fallback before the THREE conversion path.
 *   2. `setScene` filters the scene through `partitionSceneBySupport(scene,
 *      this.capabilities)` BEFORE calling `vitrumSceneToThree`, so a now-
 *      supported `instanced-mesh` is KEPT (flows through to the converter)
 *      and a now-supported `analytic` is converted to a MeshPrimitive fallback
 *      that the converter can ingest.
 *
 * `vitrumSceneToThree` is mocked with a spy that records the scene it
 * receives so we can assert the converter only ever sees the supported
 * subset. The fork + WebGL2 context are stubbed (no live GPU needed).
 */
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import type { MockInstance } from 'vitest';
import type { Scene, ScenePrimitive } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { installWebGL2GlobalStub, makeRendererStub } from './testUtils.js';

vi.mock('three/addons/lights/RectAreaLightUniformsLib.js', () => ({
  RectAreaLightUniformsLib: { init: vi.fn() },
}));

/** Records every scene `vitrumSceneToThree` is called with, so tests can
 *  assert the converter only sees the supported subset. */
const seenScenes: Scene[] = [];
vi.mock('@vitrum/three-bindings', () => ({
  vitrumSceneToThree: (scene: Scene) => {
    seenScenes.push(scene);
    return { traverse: () => undefined };
  },
}));

vi.mock('three-gpu-pathtracer', () => {
  class WebGLPathTracer {
    readonly target = { texture: {} };
    samples = 0;
    bounces = 0;
    transmissiveBounces = 0;
    filterGlossyFactor = 0;
    renderDelay = 0;
    minSamples = 0;
    dynamicLowRes = false;
    multipleImportanceSampling = false;
    tileRepeatFactors: Uint8Array | null = null;
    configureAdditiveAccumulation = vi.fn();
    readonly tiles = { set: vi.fn() };
    readonly _pathTracer = { material: { uniforms: {} } };
    setScene(): void {}
    setCamera(): void {}
    renderSample(): void { this.samples += 1; }
    reset(): void {}
    dispose(): void {}
    updateEnvironment(): void {}
  }
  return { WebGLPathTracer };
});

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
      asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1])),
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

function sceneWith(primitives: ScenePrimitive[]): Scene {
  return { primitives, emitters: [], environment: { kind: 'none' } };
}

describe('pt-webgl capability/partition reconciliation', () => {
  let teardownGlobalStub: (() => void) | null = null;
  let warnSpy: MockInstance | undefined;
  beforeAll(() => {
    teardownGlobalStub = installWebGL2GlobalStub();
  });
  afterAll(() => {
    teardownGlobalStub?.();
    teardownGlobalStub = null;
  });
  afterEach(() => {
    seenScenes.length = 0;
    warnSpy?.mockRestore();
  });

  it('declares mesh / skinned-mesh / instanced-mesh and analytic fallback support', async () => {
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    const kinds = engine.capabilities.supportedPrimitiveKinds!;
    expect(kinds.has('mesh')).toBe(true);
    expect(kinds.has('skinned-mesh')).toBe(true);
    expect(kinds.has('instanced-mesh')).toBe(true);
    expect(kinds.has('analytic')).toBe(true);
    expect(engine.capabilities.supportedAnalyticShapes).toEqual(
      new Set(['sphere', 'box', 'capsule', 'cylinder', 'h-channel-came']),
    );
  });

  it('keeps an instanced-mesh (now supported) flowing through to the converter, not warn-skipped', async () => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });

    engine.setScene(sceneWith([meshPrimitive('mesh-a'), instancedMeshPrimitive('inst-a')]));

    // No "not supported" warning for the instanced-mesh: pt-webgl now expands
    // it into N baked meshes pt-webgl-side, so partitionSceneBySupport keeps
    // it rather than dropping it.
    const warned = warnSpy.mock.calls.flat().map(String);
    expect(warned.some((m) => m.includes('inst-a') && m.includes('not supported'))).toBe(false);

    // The converter saw BOTH the mesh and the instanced-mesh — the latter is
    // no longer filtered out before vitrumSceneToThree.
    const last = seenScenes.at(-1)!;
    expect(last.primitives.map((p) => String(p.id))).toEqual(['mesh-a', 'inst-a']);
    expect(last.primitives.some((p) => p.kind === 'instanced-mesh')).toBe(true);
  });

  it('converts an analytic primitive to a generated mesh fallback before THREE conversion', async () => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });

    expect(() =>
      engine.setScene(sceneWith([meshPrimitive('mesh-a'), analyticPrimitive('sphere-a')])),
    ).not.toThrow();

    const warned = warnSpy.mock.calls.flat().map(String);
    expect(warned.some((m) => m.includes('sphere-a') && m.includes('not supported'))).toBe(false);

    // The converter sees the authored mesh plus a generated MeshPrimitive for
    // the analytic sphere. The authored scene is still cached by the engine for
    // future params/shape patches; this is only the render-ingestion view.
    const last = seenScenes.at(-1)!;
    expect(last.primitives.map((p) => String(p.id))).toEqual(['mesh-a', 'sphere-a']);
    expect(last.primitives.some((p) => p.kind === 'analytic')).toBe(false);
    const generated = last.primitives.find((p) => p.id === 'sphere-a');
    expect(generated?.kind).toBe('mesh');
    if (generated?.kind !== 'mesh') throw new Error('expected analytic fallback to be a mesh');
    expect(generated.positions.length).toBeGreaterThan(0);
    expect(generated.normals.length).toBe(generated.positions.length);
    expect(generated.indices?.length).toBeGreaterThan(0);
    expect(generated.material.baseColor).toEqual([1, 0, 0]);
  });
});
