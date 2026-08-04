import { describe, expect, it, vi } from 'vitest';
import { asMat4, type Scene } from '@vitrum/core';
import { HybridEngine, type HybridEngineOptions } from '../HybridEngine.js';
import {
  buildReSTIRSceneBVHForCoreScene,
  type SceneBVHBuffers,
} from '../restir/bvhCore.js';
import {
  SceneMutationFinalizationError,
  type PreparedSceneMutation,
} from '../SceneMutationTransaction.js';

type MutationOwner = 'ddgi' | 'rc' | 'pipeline';
type MutationPhase = 'prepare' | 'commit' | 'finalize';

interface InjectedFault {
  readonly owner: MutationOwner;
  readonly phase: MutationPhase;
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

function makeDeviceStub(): GPUDevice {
  return {
    createCommandEncoder: vi.fn(),
    createBuffer: vi.fn(() => ({ destroy: vi.fn() })),
    createTexture: vi.fn(() => ({
      createView: vi.fn(() => ({})),
      destroy: vi.fn(),
    })),
    createBindGroup: vi.fn(),
    queue: {
      submit: vi.fn(),
      writeBuffer: vi.fn(),
      writeTexture: vi.fn(),
    },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function identityMat4() {
  return asMat4(new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]));
}

function emitterScene(intensity = 1): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'triangle',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: {
        baseColor: [0.7, 0.7, 0.7],
        roughness: 0.5,
        metallic: 0,
      },
      transform: identityMat4(),
    }],
    emitters: [{
      kind: 'point',
      id: 'lamp',
      position: [0, 2, 0],
      color: [1, 0.8, 0.6],
      intensity,
    }],
    environment: { kind: 'none' },
  };
}

function meshEmitterScene(intensity = 1): Scene {
  const scene = emitterScene();
  return {
    ...scene,
    emitters: [{
      kind: 'mesh-area',
      id: 'panel',
      meshId: 'triangle',
      color: [0.25, 0.5, 1],
      intensity,
    }],
  };
}

function makeOptions(): HybridEngineOptions {
  return {
    device: makeDeviceStub(),
    width: 64,
    height: 64,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 2,
    skyTint: [1, 1, 1],
    skyIrradiance: 1,
  };
}

function preparedParticipant(
  owner: MutationOwner,
  events: string[],
  fault: InjectedFault | undefined,
): PreparedSceneMutation {
  return {
    commit: vi.fn(() => {
      events.push(`${owner}:commit`);
      if (fault?.owner === owner && fault.phase === 'commit') {
        throw new Error(`${owner} commit fault`);
      }
    }),
    rollback: vi.fn(() => {
      events.push(`${owner}:rollback`);
    }),
    finalize: vi.fn(() => {
      events.push(`${owner}:finalize`);
      if (fault?.owner === owner && fault.phase === 'finalize') {
        throw new Error(`${owner} finalize fault`);
      }
    }),
  };
}

function seedEngine(fault?: InjectedFault, scene: Scene = emitterScene()) {
  const engine = new HybridEngine(makeOptions());
  const bvh = buildReSTIRSceneBVHForCoreScene(scene, { bvhMode: 'tlas' });
  const events: string[] = [];

  const prepare = (
    owner: MutationOwner,
  ): PreparedSceneMutation => {
    events.push(`${owner}:prepare`);
    if (fault?.owner === owner && fault.phase === 'prepare') {
      throw new Error(`${owner} prepare fault`);
    }
    return preparedParticipant(owner, events, fault);
  };

  const ddgi = {
    prepareLightingMutation: vi.fn((
      _inputs: { readonly lights: readonly { readonly intensity: number }[] },
    ) => prepare('ddgi')),
    dispose: vi.fn(),
  };
  const rc = {
    prepareBindingInvalidation: vi.fn(() => prepare('rc')),
    dispose: vi.fn(),
  };
  const pipeline = {
    prepareEmitterLightingMutation: vi.fn((
      _nextBvh: SceneBVHBuffers,
      _nextScene: Scene,
    ) => prepare('pipeline')),
    dispose: vi.fn(),
  };

  const internals = engine as unknown as HybridEngineInternals;
  internals._state = 'ready';
  internals._lastScene = scene;
  internals._renderScene = scene;
  internals._bvhBuffers = bvh;
  internals._ddgi = ddgi;
  internals._rc = rc;
  internals._pipeline = pipeline;

  return {
    engine,
    internals,
    scene,
    bvh,
    events,
    ddgi,
    rc,
    pipeline,
  };
}

function updateLamp(engine: HybridEngine): void {
    engine.updateEmitter('lamp', {
      intensity: 4,
      position: [2, 3, 4],
    });
}

describe('HybridEngine.updateEmitter transaction', () => {
  it('rejects emitter patches after disposal before preparing candidates', () => {
    const f = seedEngine();
    try {
      f.internals._state = 'disposed';

      expect(() => updateLamp(f.engine)).toThrow(
        'HybridEngine.updateEmitter: engine is disposed.',
      );
      expect(f.events).toEqual([]);
      expect(f.ddgi.prepareLightingMutation).not.toHaveBeenCalled();
      expect(f.rc.prepareBindingInvalidation).not.toHaveBeenCalled();
      expect(f.pipeline.prepareEmitterLightingMutation).not.toHaveBeenCalled();
      expect(f.internals._lastScene).toBe(f.scene);
      expect(f.internals._bvhBuffers).toBe(f.bvh);
    } finally {
      f.internals._state = 'ready';
      f.engine.dispose();
    }
  });

  it('rejects the published-BVH initialization window before preparing candidates', () => {
    const f = seedEngine();
    try {
      f.internals._state = 'initializing';

      expect(() => updateLamp(f.engine)).toThrow(
        'engine is initializing',
      );
      expect(f.events).toEqual([]);
      expect(f.ddgi.prepareLightingMutation).not.toHaveBeenCalled();
      expect(f.rc.prepareBindingInvalidation).not.toHaveBeenCalled();
      expect(f.pipeline.prepareEmitterLightingMutation).not.toHaveBeenCalled();
      expect(f.internals._lastScene).toBe(f.scene);
      expect(f.internals._renderScene).toBe(f.scene);
      expect(f.internals._bvhBuffers).toBe(f.bvh);
    } finally {
      f.internals._state = 'ready';
      f.engine.dispose();
    }
  });
  it('publishes scene, BVH, DDGI, RC, and pipeline as one ordered mutation', () => {
    const f = seedEngine();
    try {
      updateLamp(f.engine);

      expect(f.events).toEqual([
        'ddgi:prepare',
        'rc:prepare',
        'pipeline:prepare',
        'ddgi:commit',
        'rc:commit',
        'pipeline:commit',
        'pipeline:finalize',
        'rc:finalize',
        'ddgi:finalize',
      ]);
      expect(f.internals._lastScene).not.toBe(f.scene);
      expect(f.internals._renderScene).not.toBe(f.scene);
      expect(f.internals._bvhBuffers).not.toBe(f.bvh);
      expect(f.internals._lastScene?.emitters[0]?.intensity).toBe(4);

      const [lightingInputs] = f.ddgi.prepareLightingMutation.mock.calls[0]!;
      expect(lightingInputs.lights).toHaveLength(1);
      expect(lightingInputs.lights[0]?.intensity).toBe(4);

      const [pipelineBvh, pipelineScene] =
        f.pipeline.prepareEmitterLightingMutation.mock.calls[0]!;
      expect(pipelineBvh).toBe(f.internals._bvhBuffers);
      expect(pipelineScene).toBe(f.internals._renderScene);
    } finally {
      f.engine.dispose();
    }
  });

  it('keeps the prior generation when the last participant fails to prepare', () => {
    const f = seedEngine({ owner: 'pipeline', phase: 'prepare' });
    try {
      expect(() => updateLamp(f.engine)).toThrow('pipeline prepare fault');

      expect(f.internals._lastScene).toBe(f.scene);
      expect(f.internals._renderScene).toBe(f.scene);
      expect(f.internals._bvhBuffers).toBe(f.bvh);
      expect(f.events).toEqual([
        'ddgi:prepare',
        'rc:prepare',
        'pipeline:prepare',
        'rc:rollback',
        'ddgi:rollback',
      ]);
    } finally {
      f.engine.dispose();
    }
  });

  it('rolls every participant back when pipeline publication fails', () => {
    const f = seedEngine({ owner: 'pipeline', phase: 'commit' });
    try {
      expect(() => updateLamp(f.engine)).toThrow('pipeline commit fault');

      expect(f.internals._lastScene).toBe(f.scene);
      expect(f.internals._renderScene).toBe(f.scene);
      expect(f.internals._bvhBuffers).toBe(f.bvh);
      expect(f.events).toEqual([
        'ddgi:prepare',
        'rc:prepare',
        'pipeline:prepare',
        'ddgi:commit',
        'rc:commit',
        'pipeline:commit',
        'pipeline:rollback',
        'rc:rollback',
        'ddgi:rollback',
      ]);
    } finally {
      f.engine.dispose();
    }
  });

  it('restores the prior emissive ownership bytes when publication fails', () => {
    const f = seedEngine(
      { owner: 'pipeline', phase: 'commit' },
      meshEmitterScene(2),
    );
    try {
      const previousBytes = new Uint8Array(
        f.bvh.bvhEmissiveLe.cpuData,
      ).slice();
      const previousFloats = new Float32Array(previousBytes.buffer);
      expect(Array.from(previousFloats)).toEqual([0.5, 1, 2, 1]);

      expect(() => f.engine.updateEmitter('panel', { intensity: 0 }))
        .toThrow('pipeline commit fault');

      expect(f.internals._bvhBuffers).toBe(f.bvh);
      const restoredBytes = new Uint8Array(
        f.internals._bvhBuffers!.bvhEmissiveLe.cpuData,
      );
      expect(Array.from(restoredBytes)).toEqual(Array.from(previousBytes));
      expect(Array.from(new Float32Array(
        f.internals._bvhBuffers!.bvhEmissiveLe.cpuData,
      ))).toEqual([0.5, 1, 2, 1]);

      const [candidate] =
        f.pipeline.prepareEmitterLightingMutation.mock.calls[0]!;
      expect(Array.from(new Float32Array(candidate.bvhEmissiveLe.cpuData)))
        .toEqual([0, 0, 0, 0]);
    } finally {
      f.engine.dispose();
    }
  });

  it('keeps the committed generation and continues retirement after a finalizer fails', () => {
    const f = seedEngine({ owner: 'ddgi', phase: 'finalize' });
    try {
      let thrown: unknown;
      try {
        updateLamp(f.engine);
      } catch (error) {
        thrown = error;
      }
      expect(thrown).toBeInstanceOf(SceneMutationFinalizationError);
      expect((thrown as SceneMutationFinalizationError).errors).toHaveLength(1);
      expect(
        (thrown as SceneMutationFinalizationError).errors[0],
      ).toEqual(new Error('ddgi finalize fault'));

      expect(f.internals._lastScene).not.toBe(f.scene);
      expect(f.internals._renderScene).not.toBe(f.scene);
      expect(f.internals._bvhBuffers).not.toBe(f.bvh);
      expect(f.internals._lastScene?.emitters[0]?.intensity).toBe(4);
      expect(f.events).toEqual([
        'ddgi:prepare',
        'rc:prepare',
        'pipeline:prepare',
        'ddgi:commit',
        'rc:commit',
        'pipeline:commit',
        'pipeline:finalize',
        'rc:finalize',
        'ddgi:finalize',
      ]);
    } finally {
      f.engine.dispose();
    }
  });
});
