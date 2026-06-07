// Verifies the optional `Engine.getScene()` contract on the WebGL2 backend:
//   • null before the first setScene,
//   • returns the RETAINED canonical core Scene after setScene (same reference,
//     no defensive copy),
//   • returns the capability-FILTERED supported scene (warn-and-skip applies),
//   • the reference survives dispose (pt-webgl frees GPU resources, not the JS
//     scene object).

import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import type { MeshPrimitive, Scene } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { installWebGL2GlobalStub, makeRendererStub } from './testUtils.js';

vi.mock('three/addons/lights/RectAreaLightUniformsLib.js', () => ({
  RectAreaLightUniformsLib: { init: vi.fn() },
}));

vi.mock('@vitrum/three-bindings', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@vitrum/three-bindings')>();
  return {
    ...actual,
    vitrumSceneToThree: vi.fn((scene: Scene) => {
      const meshes = scene.primitives.map((p) => ({
        isMesh: true as const,
        name: String(p.id),
        uuid: String(p.id),
        geometry: { dispose: vi.fn() },
        material: { dispose: vi.fn() },
      }));
      return {
        updateMatrixWorld: vi.fn(),
        traverse: (cb: (o: unknown) => void) => meshes.forEach(cb),
        traverseVisible: (cb: (o: unknown) => void) => meshes.forEach(cb),
      };
    }),
    applyVitrumMaterialToMesh: vi.fn(),
  };
});

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
    renderSample(): void {
      this.samples += 1;
    }
    reset(): void {}
    dispose(): void {}
    updateEnvironment(): void {}
  }
  return { WebGLPathTracer };
});

function tri(id: string): MeshPrimitive {
  return {
    id,
    kind: 'mesh',
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor: [0.8, 0.8, 0.8], roughness: 0.5, metallic: 0 },
  };
}

describe('PTEngineWebGL2.getScene (Engine.getScene contract)', () => {
  let teardownGlobalStub: (() => void) | null = null;
  beforeAll(() => {
    teardownGlobalStub = installWebGL2GlobalStub();
  });
  afterAll(() => {
    teardownGlobalStub?.();
    teardownGlobalStub = null;
  });

  it('exposes getScene as a function', async () => {
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    expect(typeof engine.getScene).toBe('function');
  });

  it('returns null before the first setScene', async () => {
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    expect(engine.getScene!()).toBeNull();
  });

  it('returns the retained core Scene after setScene', async () => {
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    const scene: Scene = {
      primitives: [tri('mesh-a'), tri('mesh-b')],
      emitters: [],
      environment: { kind: 'none' },
    };
    engine.setScene(scene);
    const read = engine.getScene!();
    expect(read).not.toBeNull();
    expect(read!.primitives.map((p) => p.id)).toEqual(['mesh-a', 'mesh-b']);
    // It is the canonical core Scene, not a THREE host object.
    expect((read as unknown as { isScene?: boolean }).isScene).toBeUndefined();
    expect(read!.environment.kind).toBe('none');
  });

  it('returns the supported (capability-filtered) scene, not the raw input', async () => {
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    // A bogus emitter kind is warn-skipped by partitionSceneBySupport; the
    // retained scene reflects the filter, so getScene is the honest "what is
    // being rendered" answer.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const scene = {
      primitives: [tri('mesh-a')],
      emitters: [{ id: 'bad', kind: 'not-a-real-kind' } as never],
      environment: { kind: 'none' },
    } as Scene;
    engine.setScene(scene);
    const read = engine.getScene!();
    expect(read!.emitters).toEqual([]); // the unsupported emitter was dropped
    warn.mockRestore();
  });

  it('keeps the scene reference available across dispose (GPU teardown only)', async () => {
    const engine = await createPTEngine_WebGL2({ device: makeRendererStub() as never });
    const scene: Scene = {
      primitives: [tri('mesh-a')],
      emitters: [],
      environment: { kind: 'none' },
    };
    engine.setScene(scene);
    engine.dispose();
    // The concrete engine retains the JS scene object across dispose; only the
    // @vitrum/engine facade wrapper nulls it post-dispose (covered there).
    expect(engine.getScene!()!.primitives.map((p) => p.id)).toEqual(['mesh-a']);
  });
});
