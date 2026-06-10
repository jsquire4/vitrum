import { describe, expect, it, vi } from 'vitest';
import type { Scene, ScenePrimitive, SceneEmitter } from '@vitrum/core';
import type { ScenePackResult } from '@vitrum/shared-bvh';
import { SceneMutationRouter, type MutationHost } from '../sceneMutationRouter.js';
import type { UploadedSceneBuffers } from '../scene/uploadSceneBuffers.js';

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
      } as SceneEmitter,
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
    expect(calls.reset).not.toHaveBeenCalled();
    expect(calls.invalidateBindGroups).not.toHaveBeenCalled();
  });

  it('updateEmitter: with no sceneBuffers falls through to setScene', () => {
    const { host, calls } = makeHost({ sceneBuffers: null });
    const router = new SceneMutationRouter(host);
    const patch: Partial<SceneEmitter> = { intensity: 2 } as Partial<SceneEmitter>;
    router.updateEmitter('e', patch);
    expect(calls.assertLive).toHaveBeenCalledWith('updateEmitter');
    expect(calls.setScene).toHaveBeenCalledTimes(1);
  });

  it('updateEnvironment: with no sceneBuffers falls through to setScene with normalized env', () => {
    const { host, calls } = makeHost({ sceneBuffers: null });
    const router = new SceneMutationRouter(host);
    router.updateEnvironment(null);
    expect(calls.assertLive).toHaveBeenCalledWith('updateEnvironment');
    expect(calls.setScene).toHaveBeenCalledTimes(1);
    const [nextScene] = calls.setScene.mock.calls[0]!;
    // null env normalizes to { kind: 'none' } before the setScene fall-through.
    expect(nextScene.environment).toEqual({ kind: 'none' });
  });
});
