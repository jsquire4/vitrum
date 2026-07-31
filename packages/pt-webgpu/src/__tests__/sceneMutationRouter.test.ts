import { describe, expect, it, vi } from 'vitest';
import { asMat4, type Scene, type ScenePrimitive, type SceneEmitter } from '@vitrum/core';
import type { ScenePackResult } from '@vitrum/shared-bvh';
import { SceneMutationRouter, type MutationHost } from '../sceneMutationRouter.js';
import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';
import {
  assertSpectralSceneSupported,
  assertThinFilmSceneSupported,
  assertVolumeSceneSupported,
} from '../spectralSceneValidation.js';

/**
 * Characterization tests for SceneMutationRouter against the MutationHost seam
 * (Task 4.3, Theme A). The full incremental-update behavior is covered end-to-
 * end by updatePrimitiveIncremental / updateEmitterIncremental /
 * updateEnvironmentIncremental / addRemovePrimitive through the public engine.
 *
 * These pin the ROUTING CONTRACT directly: which host op fires, the throws, and
 * that no engine state is duplicated (the router only reaches state through the
 * host). They exercise the "no scene buffers yet → fall through to setScene" and
 * the whole-primitive add/remove → repackScene paths, which need no GPU.
 */

function baseScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'a',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [0.8, 0.2, 0.1], roughness: 0.3, metallic: 0.1 },
      },
    ],
    emitters: [
      {
        kind: 'point',
        id: 'e',
        position: [0, 1, 0],
        intensity: 1,
        color: [1, 1, 1],
      },
    ],
    environment: { kind: 'none' },
  };
}

interface HostState {
  scene: Scene | null;
  sceneBuffers: UploadedSceneBuffers | null;
  geoPack: ScenePackResult | null;
}

function makeHost(over: Partial<HostState> = {}) {
  const state: HostState = {
    scene: baseScene(),
    sceneBuffers: null,
    geoPack: null,
    ...over,
  };
  const calls = {
    assertLive: vi.fn<[string], void>(),
    repackScene: vi.fn<[Scene, { warnOnEmpty: boolean }], void>(),
    setScene: vi.fn<[Scene], void>(),
    reset: vi.fn(),
    invalidateBindGroups: vi.fn(),
    setSceneState: vi.fn<[Scene], void>((s) => {
      state.scene = s;
    }),
    validatePrimitiveCandidate: vi.fn<[Scene, string], void>(),
    validateEmitterCandidate: vi.fn<[Scene, string], void>(),
    validateEnvironmentCandidate: vi.fn<[Scene['environment']], void>(),
    validateEmittersCandidate: vi.fn<[readonly SceneEmitter[], readonly ScenePrimitive[]], void>(),
    setGeoPack: vi.fn<[ScenePackResult], void>((g) => {
      state.geoPack = g;
    }),
    supportedAnalyticShapes: vi.fn(() => new Set<string>(['sphere'])),
  };
  const host: MutationHost = {
    device: { queue: { writeBuffer: vi.fn() } } as unknown as GPUDevice,
    assertLive: calls.assertLive,
    getScene: () => state.scene,
    setSceneState: calls.setSceneState,
    getSceneBuffers: () => state.sceneBuffers,
    getGeoPack: () => state.geoPack,
    setGeoPack: calls.setGeoPack,
    invalidateBindGroups: calls.invalidateBindGroups,
    supportedAnalyticShapes: calls.supportedAnalyticShapes,
    validatePrimitiveCandidate: calls.validatePrimitiveCandidate,
    validateEmitterCandidate: calls.validateEmitterCandidate,
    validateEnvironmentCandidate: calls.validateEnvironmentCandidate,
    validateEmittersCandidate: calls.validateEmittersCandidate,
    cameraVisibleEmitters: () => false,
    repackScene: calls.repackScene,
    setScene: calls.setScene,
    reset: calls.reset,
  };
  return { host, state, calls };
}

describe('SceneMutationRouter — routing contract (pt-webgpu Task 4.3)', () => {
  it('addPrimitive: asserts live, tail-appends, and full-repacks (warnOnEmpty:false)', () => {
    const { host, calls } = makeHost();
    const router = new SceneMutationRouter(host);
    const prim: ScenePrimitive = {
      kind: 'mesh',
      id: 'b',
      positions: new Float32Array([0, 0, 1, 1, 0, 1, 0, 1, 1]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: { baseColor: [0.1, 0.4, 0.9], roughness: 0.6, metallic: 0.2 },
    };
    router.addPrimitive(prim);
    expect(calls.assertLive).toHaveBeenCalledWith('addPrimitive');
    expect(calls.repackScene).toHaveBeenCalledTimes(1);
    const [repacked, opts] = calls.repackScene.mock.calls[0]!;
    expect(opts).toEqual({ warnOnEmpty: false });
    expect(repacked.primitives.map((p) => p.id)).toEqual(['a', 'b']);
    expect(calls.setScene).not.toHaveBeenCalled();
  });

  it('addPrimitive: throws on duplicate id without repacking', () => {
    const { host, calls } = makeHost();
    const router = new SceneMutationRouter(host);
    const dup: ScenePrimitive = {
      kind: 'mesh',
      id: 'a',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: { baseColor: [1, 1, 1], roughness: 1, metallic: 0 },
    };
    expect(() => router.addPrimitive(dup)).toThrow(/already exists/);
    expect(calls.repackScene).not.toHaveBeenCalled();
  });

  it('removePrimitive: drops the primitive and full-repacks', () => {
    const { host, calls } = makeHost();
    const router = new SceneMutationRouter(host);
    router.removePrimitive('a');
    expect(calls.assertLive).toHaveBeenCalledWith('removePrimitive');
    expect(calls.repackScene).toHaveBeenCalledTimes(1);
    const [repacked] = calls.repackScene.mock.calls[0]!;
    expect(repacked.primitives).toEqual([]);
  });

  it('removePrimitive: throws when id is absent', () => {
    const { host, calls } = makeHost();
    const router = new SceneMutationRouter(host);
    expect(() => router.removePrimitive('nope')).toThrow(/no primitive with id/);
    expect(calls.repackScene).not.toHaveBeenCalled();
  });

  it('updatePrimitive: with no sceneBuffers falls through to setScene (no fast path eligible)', () => {
    const { host, calls } = makeHost({ sceneBuffers: null, geoPack: null });
    const router = new SceneMutationRouter(host);
    router.updatePrimitive('a', {
      material: { baseColor: [0.2, 0.7, 0.9], roughness: 0.05, metallic: 0.4 },
    });
    expect(calls.assertLive).toHaveBeenCalledWith('updatePrimitive');
    // Every fast path guards on sceneBuffers != null, so all miss → setScene.
    expect(calls.setScene).toHaveBeenCalledTimes(1);
    expect(calls.validatePrimitiveCandidate).toHaveBeenCalledTimes(1);
    expect(calls.reset).not.toHaveBeenCalled();
    expect(calls.invalidateBindGroups).not.toHaveBeenCalled();
  });

  it('updatePrimitive: own undefined clears an optional field instead of becoming a no-op', () => {
    const scene = baseScene();
    const primitive = scene.primitives[0]!;
    if (primitive.kind !== 'mesh') throw new Error('expected mesh primitive');
    const transformedScene: Scene = {
      ...scene,
      primitives: [
        {
          ...primitive,
          transform: asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 4, 0, 0, 1])),
        },
      ],
    };
    const { host, calls } = makeHost({
      scene: transformedScene,
      sceneBuffers: null,
      geoPack: null,
    });
    const router = new SceneMutationRouter(host);

    router.updatePrimitive('a', { transform: undefined });

    expect(calls.validatePrimitiveCandidate).toHaveBeenCalledTimes(1);
    expect(calls.setScene).toHaveBeenCalledTimes(1);
    const [nextScene] = calls.setScene.mock.calls[0]!;
    const nextPrimitive = nextScene.primitives[0];
    expect(nextPrimitive?.kind).toBe('mesh');
    if (nextPrimitive?.kind === 'mesh') expect(nextPrimitive.transform).toBeUndefined();
  });

  it('updatePrimitive: rejects every own id/kind field before no-op routing', () => {
    const { host, calls } = makeHost({ sceneBuffers: null, geoPack: null });
    const router = new SceneMutationRouter(host);

    expect(() =>
      router.updatePrimitive('a', {
        id: 'a',
      } as never),
    ).toThrow(/id cannot be changed or supplied/);
    expect(() =>
      router.updatePrimitive('a', {
        kind: undefined,
      } as never),
    ).toThrow(/kind cannot change or be supplied/);
    expect(calls.validatePrimitiveCandidate).not.toHaveBeenCalled();
    expect(calls.setScene).not.toHaveBeenCalled();
  });

  it('updatePrimitive: mapped analytics conservatively full-repack their fallback mesh layout', () => {
    const mappedAnalytic: Scene = {
      primitives: [
        {
          kind: 'analytic',
          id: 'mapped-sphere',
          shape: 'sphere',
          params: new Float32Array([0, 0, 0, 1]),
          fallbackMesh: {
            positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
            normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
            uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
          },
          material: {
            baseColor: [1, 1, 1],
            roughness: 0.4,
            metallic: 0,
            baseColorMap: { handle: { id: 'albedo' } },
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const { host, calls } = makeHost({
      scene: mappedAnalytic,
      sceneBuffers: null,
      geoPack: null,
    });
    const router = new SceneMutationRouter(host);

    router.updatePrimitive('mapped-sphere', {
      material: {
        ...mappedAnalytic.primitives[0]!.material,
        roughness: 0.2,
      },
    });

    expect(calls.repackScene).toHaveBeenCalledTimes(1);
    expect(calls.setScene).not.toHaveBeenCalled();
  });

  it('rejects invalid thin-film mutations transactionally and accepts the supported domain', () => {
    const { host, state, calls } = makeHost({ sceneBuffers: null, geoPack: null });
    calls.validatePrimitiveCandidate.mockImplementation((candidate) => {
      assertSpectralSceneSupported(candidate);
      assertThinFilmSceneSupported(candidate);
    });
    const router = new SceneMutationRouter(host);
    const previous = state.scene;
    expect(() =>
      router.updatePrimitive('a', {
        material: {
          ...previous!.primitives[0]!.material,
          thinFilmStack: {
            layers: [{ ior: 1.4, thicknessNm: -1 }],
          },
        },
      }),
    ).toThrow(/thinFilmStack\.layers\[0\]\.thicknessNm must be > 0/);
    expect(state.scene).toBe(previous);
    expect(calls.setScene).not.toHaveBeenCalled();
    expect(calls.reset).not.toHaveBeenCalled();

    // A coherent stack overrides the RGB-only iridescence model and now
    // participates in the authored rough/metallic finite BSDF instead of
    // requiring a smooth, fully transmissive special case.
    expect(() =>
      router.updatePrimitive('a', {
        material: {
          ...previous!.primitives[0]!.material,
          iridescence: 1,
          thinFilmStack: {
            layers: [{ ior: 1.4, thicknessNm: 320 }],
            angleDependent: true,
          },
        },
      }),
    ).not.toThrow();
    expect(calls.setScene).toHaveBeenCalledTimes(1);
    const [accepted] = calls.setScene.mock.calls[0]!;
    expect(accepted.primitives[0]!.material.iridescence).toBe(1);
    expect(accepted.primitives[0]!.material.roughness).toBe(0.3);
    expect(accepted.primitives[0]!.material.metallic).toBe(0.1);
    expect(accepted.primitives[0]!.material.thinFilmStack?.layers).toHaveLength(1);
    expect(calls.reset).not.toHaveBeenCalled();
  });

  it('rejects invalid volume mutations before incremental publication', () => {
    const { host, state, calls } = makeHost({ sceneBuffers: null, geoPack: null });
    calls.validatePrimitiveCandidate.mockImplementation(assertVolumeSceneSupported);
    const router = new SceneMutationRouter(host);
    const previous = state.scene;

    expect(() =>
      router.updatePrimitive('a', {
        material: {
          ...previous!.primitives[0]!.material,
          transmission: 1,
          scatteringCoefficientRGB: [0.1, -0.2, 0.3],
        },
      }),
    ).toThrow(/scatteringCoefficientRGB/);

    expect(state.scene).toBe(previous);
    expect(calls.setScene).not.toHaveBeenCalled();
    expect(calls.repackScene).not.toHaveBeenCalled();
    expect(calls.reset).not.toHaveBeenCalled();
    expect(calls.invalidateBindGroups).not.toHaveBeenCalled();
  });

  it('updateEmitter: with no sceneBuffers falls through to setScene', () => {
    const { host, calls } = makeHost({ sceneBuffers: null });
    const router = new SceneMutationRouter(host);
    const patch: Partial<SceneEmitter> = { intensity: 2 };
    router.updateEmitter('e', patch);
    expect(calls.assertLive).toHaveBeenCalledWith('updateEmitter');
    expect(calls.setScene).toHaveBeenCalledTimes(1);
    expect(calls.validateEmitterCandidate).toHaveBeenCalledTimes(1);
  });

  it('updateEnvironment: with no sceneBuffers falls through to setScene with normalized env', () => {
    const { host, calls } = makeHost({ sceneBuffers: null });
    const router = new SceneMutationRouter(host);
    router.updateEnvironment(null);
    expect(calls.assertLive).toHaveBeenCalledWith('updateEnvironment');
    expect(calls.setScene).toHaveBeenCalledTimes(1);
    expect(calls.validateEnvironmentCandidate).toHaveBeenCalledTimes(1);
    const [nextScene] = calls.setScene.mock.calls[0]!;
    // null env normalizes to { kind: 'none' } before the setScene fall-through.
    expect(nextScene.environment).toEqual({ kind: 'none' });
  });

  it('updateEnvironment: validates its candidate before deriving unrelated geometry', () => {
    const poisonedScene: Scene = {
      primitives: [
        {
          kind: 'analytic',
          id: 'poisoned-analytic',
          shape: 'sphere',
          params: new Float32Array([0, 0, 0, 1]),
          material: {
            baseColor: [1, 1, 1],
            roughness: 0.5,
            metallic: 0,
            baseColorMap: { handle: { id: 'missing-fallback-map' } },
          },
        },
      ],
      emitters: [],
      environment: { kind: 'none' },
    };
    const { host, calls } = makeHost({
      scene: poisonedScene,
      sceneBuffers: null,
    });
    calls.validateEnvironmentCandidate.mockImplementation(() => {
      throw new TypeError('malformed environment candidate');
    });
    const router = new SceneMutationRouter(host);

    expect(() =>
      router.updateEnvironment({
        kind: 'hdri',
        hdri: {},
      }),
    ).toThrow('malformed environment candidate');
    expect(calls.validateEnvironmentCandidate).toHaveBeenCalledTimes(1);
    expect(calls.setScene).not.toHaveBeenCalled();
  });
});
