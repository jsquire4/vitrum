import { afterEach, describe, expect, it, vi } from 'vitest';
import type { EngineError, Scene } from '@vitrum/core';
import { DDGI } from '../DDGI.js';
import type { SceneBVHBuffers } from '../../restir/bvhCore.js';
import { packMaterialTextureAtlas } from '../../pipeline/materialTextureAtlas.js';

function makeBoxScene(extent = 1): Scene {
  return {
    primitives: [{
      kind: 'mesh',
      id: 'box',
      positions: new Float32Array([
        -extent, -extent, -extent,
         extent, -extent, -extent,
        -extent,  extent, -extent,
         extent,  extent,  extent,
      ]),
      normals: new Float32Array([
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
        0, 0, 1,
      ]),
      indices: new Uint32Array([0, 1, 2, 1, 3, 2]),
      material: { baseColor: [1, 1, 1], roughness: 1, metallic: 0 },
    }],
    emitters: [],
    environment: { kind: 'none' },
  };
}

function deferred<T>(): {
  readonly promise: Promise<T>;
  readonly resolve: (value: T) => void;
  readonly reject: (reason: unknown) => void;
} {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const device = {} as GPUDevice;
const scene = makeBoxScene();

function makeRestirBuffers(
  blasSeed: number,
  tlasSeed?: number,
): SceneBVHBuffers {
  const cpu = (values: readonly number[]) => ({
    cpuData: new Uint32Array(values).buffer,
  });
  const tlas = tlasSeed === undefined ? undefined : {
    nodes: cpu([tlasSeed, 1, 2, 3, 4, 5, 6, 7]),
    instanceIndices: cpu([0]),
    blasRoots: cpu([0]),
    worldToLocal: cpu([tlasSeed, 0, 0, 0]),
    localToWorld: cpu([tlasSeed, 0, 0, 0]),
    nodeCount: 1,
  };
  return {
    bvhMode: tlas == null ? 'merged' : 'tlas',
    primitiveTlasBindings: [],
    mergedGeometry: {
      boundingBox: {
        min: { x: -1, y: -1, z: -1 },
        max: { x: 1, y: 1, z: 1 },
      },
      computeBoundingBox: vi.fn(),
    },
    bvhNodes: cpu([blasSeed, 1, 2, 3, 4, 5, 6, 7]),
    bvhPositions: cpu([blasSeed, 1, 2, 3]),
    bvhIndex: cpu([0, 1, 2, 0]),
    opticalTriangleIdentity: cpu([0, 0]),
    opticalInstanceBoundaryIdBasePlusOne: cpu([1]),
    bvhNormals: cpu([0, 0, 1, 0]),
    bvhTangents: cpu([1, 0, 0, 1]),
    bvhColors: cpu([1, 1, 1, 1]),
    triangleMaterialIds: cpu([0]),
    buildMaterials: [],
    coreMaterials: [],
    materialTextureAtlas: packMaterialTextureAtlas([], new Uint32Array([0]), 1),
    ...(tlas == null ? {} : { tlas }),
  } as unknown as SceneBVHBuffers;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('DDGI owned update lifecycle', () => {
  it('coalesces concurrent callers onto one owned initialization/update task', async () => {
    const ddgi = new DDGI();
    const initResult = deferred<boolean>();
    const init = vi.spyOn(ddgi.pass, 'init').mockImplementation(
      () => initResult.promise,
    );
    const runFrame = vi.spyOn(ddgi.pass, 'runFrame').mockResolvedValue(true);

    const first = ddgi.updateFrame({ coreScene: scene, device, enabled: true });
    const second = ddgi.updateFrame({ coreScene: scene, device, enabled: true });

    expect(second).toBe(first);
    expect(init).toHaveBeenCalledTimes(1);
    expect(runFrame).not.toHaveBeenCalled();
    expect((ddgi as unknown as { _inited: boolean })._inited).toBe(false);

    initResult.resolve(true);
    await first;

    expect(init).toHaveBeenCalledTimes(1);
    expect(runFrame).toHaveBeenCalledTimes(1);
    expect(ddgi.warmupFrame).toBe(1);
    ddgi.dispose();
  });

  it('does not start work for disabled frames', async () => {
    const ddgi = new DDGI();
    const init = vi.spyOn(ddgi.pass, 'init').mockResolvedValue(true);

    await ddgi.updateFrame({ coreScene: scene, device, enabled: false });

    expect(init).not.toHaveBeenCalled();
    expect(ddgi.warmupFrame).toBe(0);
    ddgi.dispose();
  });

  it('advances warmup and readiness only for an accepted probe submission', async () => {
    const ddgi = new DDGI();
    vi.spyOn(ddgi.pass, 'init').mockResolvedValue(true);
    const runFrame = vi.spyOn(ddgi.pass, 'runFrame')
      .mockResolvedValueOnce(false)
      .mockResolvedValueOnce(true);
    ddgi.setProbeUpdateDivisor(1);

    await ddgi.updateFrame({ coreScene: scene, device, enabled: true });
    expect(ddgi.warmupFrame).toBe(0);
    expect(ddgi.ready).toBe(false);

    // Bypass the wall-clock cap to exercise the next owned frame deterministically.
    (ddgi as unknown as { _lastFrameTs: number })._lastFrameTs = 0;
    await ddgi.updateFrame({ coreScene: scene, device, enabled: true });

    expect(runFrame).toHaveBeenCalledTimes(2);
    expect(ddgi.warmupFrame).toBe(1);
    expect(ddgi.ready).toBe(true);
    ddgi.dispose();
  });

  it('restarts warmup and arms every stratum when standalone bounds replace the atlas cohort', async () => {
    const ddgi = new DDGI();
    vi.spyOn(ddgi.pass, 'init').mockResolvedValue(true);
    const runFrame = vi.spyOn(ddgi.pass, 'runFrame').mockResolvedValue(true);
    const reallocate = vi.spyOn(ddgi.pass, 'reallocateGridAtlases');
    ddgi.setProbeUpdateDivisor(2);

    await ddgi.updateFrame({ coreScene: scene, device, enabled: true });
    const lifecycle = ddgi as unknown as {
      _frame: number;
      _ready: boolean;
      _lastFrameTs: number;
    };
    lifecycle._frame = 2;
    lifecycle._ready = true;
    const previousGeneration = ddgi.pass.fullBlendGeneration;

    lifecycle._lastFrameTs = 0;
    await ddgi.updateFrame({
      coreScene: makeBoxScene(8),
      device,
      enabled: true,
    });

    expect(reallocate).toHaveBeenCalledTimes(2);
    expect(runFrame.mock.calls.at(-1)?.slice(1)).toEqual([0, 2]);
    expect(ddgi.warmupFrame).toBe(1);
    expect(ddgi.ready).toBe(false);
    expect(ddgi.pass.fullBlendGeneration).not.toBe(previousGeneration);
    expect(ddgi.pass.pendingFullBlendCount).toBe(2);
    ddgi.dispose();
  });

  it('wraps the long-running frame index as an unsigned 32-bit cadence counter', () => {
    const ddgi = new DDGI();
    const lifecycle = ddgi as unknown as {
      _frame: number;
      _advanceFrameIndex(): void;
    };
    lifecycle._frame = 0xffff_ffff;

    lifecycle._advanceFrameIndex();

    expect(ddgi.warmupFrame).toBe(0);
    ddgi.dispose();
  });

  it('restarts partial stratum coverage when the probe divisor changes', async () => {
    const ddgi = new DDGI();
    vi.spyOn(ddgi.pass, 'init').mockResolvedValue(true);
    const runFrame = vi.spyOn(ddgi.pass, 'runFrame').mockResolvedValue(true);

    // Accept half of the default stride=8 cycle: offsets 0,1,2,3 only.
    for (let frame = 0; frame < 4; frame += 1) {
      (ddgi as unknown as { _lastFrameTs: number })._lastFrameTs = 0;
      await ddgi.updateFrame({ coreScene: scene, device, enabled: true });
    }
    expect(runFrame.mock.calls.map((call) => call.slice(1))).toEqual([
      [0, 8], [1, 8], [2, 8], [3, 8],
    ]);
    expect(ddgi.warmupFrame).toBe(4);
    expect(ddgi.ready).toBe(false);

    ddgi.setProbeUpdateDivisor(2);
    expect(ddgi.warmupFrame).toBe(0);
    expect(ddgi.ready).toBe(false);

    for (let frame = 0; frame < 2; frame += 1) {
      (ddgi as unknown as { _lastFrameTs: number })._lastFrameTs = 0;
      await ddgi.updateFrame({ coreScene: scene, device, enabled: true });
      expect(ddgi.ready).toBe(frame === 1);
    }
    expect(runFrame.mock.calls.slice(4).map((call) => call.slice(1)))
      .toEqual([[0, 2], [1, 2]]);
    expect(ddgi.warmupFrame).toBe(2);

    // Once a complete atlas is ready, a cadence-only change does not erase it.
    ddgi.setProbeUpdateDivisor(4);
    expect(ddgi.warmupFrame).toBe(2);
    expect(ddgi.ready).toBe(true);
    ddgi.dispose();
  });

  it('contains runFrame rejection and leaves frame/readiness unpublished', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const errors: EngineError[] = [];
    const ddgi = new DDGI({ onError: (error) => errors.push(error) });
    vi.spyOn(ddgi.pass, 'init').mockResolvedValue(true);
    vi.spyOn(ddgi.pass, 'runFrame').mockRejectedValue(
      new Error('injected submit failure'),
    );
    ddgi.setProbeUpdateDivisor(1);

    await expect(ddgi.updateFrame({
      coreScene: scene,
      device,
      enabled: true,
    })).resolves.toBeUndefined();

    expect(ddgi.warmupFrame).toBe(0);
    expect(ddgi.ready).toBe(false);
    expect(errors).toHaveLength(1);
    expect(errors[0]?.message).toContain('injected submit failure');
    ddgi.dispose();
  });

  it('keeps allocation failures retryable at the DDGI lifecycle boundary', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const errors: EngineError[] = [];
    const ddgi = new DDGI({ onError: (error) => errors.push(error) });
    const init = vi.spyOn(ddgi.pass, 'init')
      .mockRejectedValueOnce(new Error('injected allocation failure'))
      .mockResolvedValueOnce(true);
    const runFrame = vi.spyOn(ddgi.pass, 'runFrame').mockResolvedValue(true);
    ddgi.setProbeUpdateDivisor(1);

    await ddgi.updateFrame({ coreScene: scene, device, enabled: true });

    expect(ddgi.state()).toBe('initializing');
    expect(ddgi.warmupFrame).toBe(0);
    expect(runFrame).not.toHaveBeenCalled();
    expect(errors[0]?.message).toContain('injected allocation failure');

    (ddgi as unknown as { _lastFrameTs: number })._lastFrameTs = 0;
    await ddgi.updateFrame({ coreScene: scene, device, enabled: true });

    expect(init).toHaveBeenCalledTimes(2);
    expect(runFrame).toHaveBeenCalledTimes(1);
    expect(ddgi.warmupFrame).toBe(1);
    expect(ddgi.ready).toBe(true);
    ddgi.dispose();
  });

  it('retries a false result when the probe pass reports a transient init failure', async () => {
    const ddgi = new DDGI();
    const init = vi.spyOn(ddgi.pass, 'init')
      .mockResolvedValueOnce(false)
      .mockResolvedValueOnce(true);
    const runFrame = vi.spyOn(ddgi.pass, 'runFrame').mockResolvedValue(true);
    ddgi.setProbeUpdateDivisor(1);

    // ProbeUpdatePass clears this guard when pipeline compilation fails, which
    // distinguishes that retryable false result from a definitive no-device
    // result where the guard remains set.
    (ddgi.pass as unknown as { _initAttempted: boolean })._initAttempted = false;
    await ddgi.updateFrame({ coreScene: scene, device, enabled: true });

    expect(ddgi.state()).toBe('initializing');
    expect(runFrame).not.toHaveBeenCalled();

    (ddgi as unknown as { _lastFrameTs: number })._lastFrameTs = 0;
    await ddgi.updateFrame({ coreScene: scene, device, enabled: true });

    expect(init).toHaveBeenCalledTimes(2);
    expect(runFrame).toHaveBeenCalledTimes(1);
    expect(ddgi.ready).toBe(true);
    ddgi.dispose();
  });

  const contentMutations: readonly [
    string,
    (ddgi: DDGI) => void,
  ][] = [
    ['cache invalidation', (ddgi) => ddgi.invalidateProbeCache()],
    ['scene publication', (ddgi) => {
      const mutation = ddgi.prepareSceneMutation(null, undefined, {
        invalidate: true,
        instancesDirty: false,
      });
      mutation.commit();
      mutation.finalize();
    }],
    ['lighting publication', (ddgi) => {
      const mutation = ddgi.prepareLightingMutation({
        lights: [],
        sunIntensityMultiplier: 2,
        emitterTris: new Float32Array(0),
        emitterCount: 0,
      });
      mutation.commit();
      mutation.finalize();
    }],
    ['probe-stride change', (ddgi) => ddgi.setProbeUpdateDivisor(2)],
    ['standalone light replacement', (ddgi) => ddgi.setLights([{
      id: 'new-light', kind: 'fixture', intensity: 3, on: true,
      position: { x: 1, y: 2, z: 3 },
    }])],
    ['standalone emitter replacement', (ddgi) => {
      ddgi.setEmitterTris(new Float32Array(20).fill(2), 1);
    }],
    ['standalone environment replacement', (ddgi) => {
      ddgi.setEnvironment(
        {} as GPUTextureView,
        {} as GPUSampler,
        0.25,
        1.5,
        true,
      );
    }],
    ['standalone sun multiplier replacement', (ddgi) => {
      ddgi.setSunIntensityMultiplier(3);
    }],
    ['standalone sky replacement', (ddgi) => {
      ddgi.setSkyParams([0.2, 0.3, 0.4], 1.5);
    }],
    ['standalone feedback replacement', (ddgi) => {
      ddgi.setIndirectFeedback(false);
    }],
    ['standalone glass mix replacement', (ddgi) => {
      ddgi.setGlassMixScale(0.25);
    }],
  ];

  it.each(contentMutations)(
    'does not let an older in-flight submission advance warmup after %s',
    async (_name, mutate) => {
      const ddgi = new DDGI();
      vi.spyOn(ddgi.pass, 'init').mockResolvedValue(true);
      const submission = deferred<boolean>();
      const runFrame = vi.spyOn(ddgi.pass, 'runFrame')
        .mockImplementationOnce(() => submission.promise)
        .mockResolvedValueOnce(true);
      ddgi.setProbeUpdateDivisor(1);

      const oldUpdate = ddgi.updateFrame({ coreScene: scene, device, enabled: true });
      await vi.waitFor(() => expect(runFrame).toHaveBeenCalledTimes(1));
      mutate(ddgi);

      submission.resolve(true);
      await oldUpdate;

      expect(ddgi.warmupFrame).toBe(0);
      expect(ddgi.ready).toBe(false);

      (ddgi as unknown as { _lastFrameTs: number })._lastFrameTs = 0;
      await ddgi.updateFrame({ coreScene: scene, device, enabled: true });

      expect(runFrame).toHaveBeenCalledTimes(2);
      expect(ddgi.warmupFrame).toBe(1);
      if (_name === 'probe-stride change') {
        expect(runFrame.mock.calls[1]?.slice(1)).toEqual([0, 2]);
      }
      ddgi.dispose();
    },
  );

  it('invalidates an older submission when syncRestirBvhBuffers publishes new geometry', async () => {
    const ddgi = new DDGI();
    ddgi.setProbeUpdateDivisor(1);
    ddgi.syncRestirBvhBuffers(makeRestirBuffers(1));
    vi.spyOn(ddgi.pass, 'init').mockResolvedValue(true);
    const submission = deferred<boolean>();
    const runFrame = vi.spyOn(ddgi.pass, 'runFrame')
      .mockImplementationOnce(() => submission.promise)
      .mockResolvedValueOnce(true);

    const oldUpdate = ddgi.updateFrame({ device, enabled: true });
    await vi.waitFor(() => expect(runFrame).toHaveBeenCalledTimes(1));
    ddgi.syncRestirBvhBuffers(makeRestirBuffers(2));
    expect(ddgi.warmupFrame).toBe(0);
    expect(ddgi.ready).toBe(false);

    submission.resolve(true);
    await oldUpdate;
    expect(ddgi.warmupFrame).toBe(0);

    (ddgi as unknown as { _lastFrameTs: number })._lastFrameTs = 0;
    await ddgi.updateFrame({ device, enabled: true });
    expect(runFrame.mock.calls[1]?.slice(1)).toEqual([0, 1]);
    expect(ddgi.warmupFrame).toBe(1);
    ddgi.dispose();
  });

  it('treats an equal ReSTIR version tuple as a per-frame synchronization no-op', () => {
    const ddgi = new DDGI();
    ddgi.syncRestirBvhBuffers(makeRestirBuffers(3));
    const lifecycle = ddgi as unknown as {
      _frame: number;
      _ready: boolean;
      _contentEpoch: number;
    };
    lifecycle._frame = 7;
    lifecycle._ready = true;
    const epoch = lifecycle._contentEpoch;
    const invalidate = vi.spyOn(ddgi.pass, 'requestFullBlend');

    ddgi.syncRestirBvhBuffers(makeRestirBuffers(3));

    expect(lifecycle._frame).toBe(7);
    expect(lifecycle._ready).toBe(true);
    expect(lifecycle._contentEpoch).toBe(epoch);
    expect(invalidate).not.toHaveBeenCalled();
    ddgi.dispose();
  });

  it('uses the lightweight dirty-instance cadence for a TLAS-only version bump', () => {
    const ddgi = new DDGI();
    ddgi.syncRestirBvhBuffers(makeRestirBuffers(4, 1));
    const lifecycle = ddgi as unknown as {
      _frame: number;
      _ready: boolean;
      _contentEpoch: number;
    };
    lifecycle._frame = 8;
    lifecycle._ready = true;
    const epoch = lifecycle._contentEpoch;
    const invalidate = vi.spyOn(ddgi.pass, 'requestFullBlend');

    ddgi.syncRestirBvhBuffers(makeRestirBuffers(4, 2));

    expect(lifecycle._frame).toBe(4);
    expect(lifecycle._ready).toBe(true);
    expect(lifecycle._contentEpoch).toBe(epoch + 1);
    expect(invalidate).not.toHaveBeenCalled();
    ddgi.dispose();
  });

  it('dispose during init invalidates the task and self-disposes late resources', async () => {
    const ddgi = new DDGI();
    const initResult = deferred<boolean>();
    let allocated = false;
    const init = vi.spyOn(ddgi.pass, 'init').mockImplementation(async () => {
      const result = await initResult.promise;
      allocated = result;
      return result;
    });
    const disposePass = vi.spyOn(ddgi.pass, 'dispose').mockImplementation(() => {
      allocated = false;
    });
    const runFrame = vi.spyOn(ddgi.pass, 'runFrame').mockResolvedValue(true);

    const update = ddgi.updateFrame({ coreScene: scene, device, enabled: true });
    ddgi.dispose();
    expect(disposePass).toHaveBeenCalledTimes(1);

    initResult.resolve(true);
    await update;

    expect(init).toHaveBeenCalledTimes(1);
    expect(disposePass).toHaveBeenCalledTimes(2);
    expect(runFrame).not.toHaveBeenCalled();
    expect(allocated).toBe(false);
    expect(ddgi.warmupFrame).toBe(0);
    expect(ddgi.ready).toBe(false);

    await ddgi.updateFrame({ coreScene: scene, device, enabled: true });
    expect(init).toHaveBeenCalledTimes(1);
  });

  const idempotentSetters: readonly [
    string,
    (ddgi: DDGI) => void,
    (ddgi: DDGI) => void,
  ][] = [
    [
      'lights',
      (ddgi) => ddgi.setLights([{
        id: 'same-light', kind: 'fixture', intensity: 2, on: true,
        position: { x: 1, y: 2, z: 3 }, color: { r: 0.5, g: 0.6, b: 0.7 },
      }]),
      (ddgi) => ddgi.setLights([{
        id: 'same-light', kind: 'fixture', intensity: 2, on: true,
        position: { x: 1, y: 2, z: 3 }, color: { r: 0.5, g: 0.6, b: 0.7 },
      }]),
    ],
    [
      'emitter triangles',
      (ddgi) => ddgi.setEmitterTris(new Float32Array(20).fill(4), 1),
      (ddgi) => ddgi.setEmitterTris(new Float32Array(20).fill(4), 1),
    ],
    [
      'environment',
      (ddgi) => {
        const view = {} as GPUTextureView;
        const sampler = {} as GPUSampler;
        (ddgi as unknown as { _testEnv: [GPUTextureView, GPUSampler] })._testEnv = [view, sampler];
        ddgi.setEnvironment(view, sampler, 0.5, 2, true);
      },
      (ddgi) => {
        const [view, sampler] =
          (ddgi as unknown as { _testEnv: [GPUTextureView, GPUSampler] })._testEnv;
        ddgi.setEnvironment(view, sampler, 0.5, 2, true);
      },
    ],
    ['sun multiplier', (ddgi) => ddgi.setSunIntensityMultiplier(4),
      (ddgi) => ddgi.setSunIntensityMultiplier(4)],
    ['sky', (ddgi) => ddgi.setSkyParams([0.1, 0.2, 0.3], 4),
      (ddgi) => ddgi.setSkyParams([0.1, 0.2, 0.3], 4)],
    ['indirect feedback', (ddgi) => ddgi.setIndirectFeedback(false),
      (ddgi) => ddgi.setIndirectFeedback(false)],
    ['glass mix', (ddgi) => ddgi.setGlassMixScale(0.3),
      (ddgi) => ddgi.setGlassMixScale(0.3)],
  ];

  it.each(idempotentSetters)(
    'does not invalidate warmup for an idempotent %s synchronization',
    (_name, configure, repeat) => {
      const ddgi = new DDGI();
      const lifecycle = ddgi as unknown as {
        _frame: number;
        _ready: boolean;
        _contentEpoch: number;
      };
      configure(ddgi);
      lifecycle._frame = 7;
      lifecycle._ready = true;
      const epoch = lifecycle._contentEpoch;
      const fullBlend = ddgi.pass.captureFullBlendState();
      const invalidate = vi.spyOn(ddgi.pass, 'requestFullBlend');

      repeat(ddgi);

      expect(ddgi.warmupFrame).toBe(7);
      expect(ddgi.ready).toBe(true);
      expect(lifecycle._contentEpoch).toBe(epoch);
      expect(ddgi.pass.captureFullBlendState()).toEqual(fullBlend);
      expect(invalidate).not.toHaveBeenCalled();
      ddgi.dispose();
    },
  );

  it('completes independent teardown and terminal state publication after hostile disposers', async () => {
    const ddgi = new DDGI();
    const lifecycle = ddgi as unknown as {
      _grid: { dispose(): void };
      _bvh: { dispose(): void };
      _ready: boolean;
      _inited: boolean;
      _gpuOk: boolean;
    };
    lifecycle._ready = true;
    lifecycle._inited = true;
    lifecycle._gpuOk = true;
    const passDispose = vi.spyOn(ddgi.pass, 'dispose').mockImplementation(() => {
      throw new Error('hostile pass teardown');
    });
    const gridDispose = vi.spyOn(lifecycle._grid, 'dispose').mockImplementation(() => {
      throw new Error('hostile grid teardown');
    });
    const bvhDispose = vi.spyOn(lifecycle._bvh, 'dispose').mockImplementation(() => {
      throw new Error('hostile BVH teardown');
    });

    expect(() => ddgi.dispose()).not.toThrow();
    expect(passDispose).toHaveBeenCalledTimes(1);
    expect(gridDispose).toHaveBeenCalledTimes(1);
    expect(bvhDispose).toHaveBeenCalledTimes(1);
    expect(ddgi.ready).toBe(false);
    expect(ddgi.state()).toBe('initializing');
    await ddgi.updateFrame({ coreScene: scene, device, enabled: true });
    expect(passDispose).toHaveBeenCalledTimes(1);
  });
});
