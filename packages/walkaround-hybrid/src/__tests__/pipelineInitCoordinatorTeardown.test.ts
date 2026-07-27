import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  PipelineInitCoordinator,
  type PipelineInitHost,
} from '../HybridEngineLifecycle.js';
import { DEFAULT_NRC_CONFIG } from '../neural/nrc/nrcSubsystem.js';

/**
 * V1-4 regression: a dispose that races the scene-readiness poll must still
 * complete the deferred teardown — releasing pipeline/BVH/DDGI AND RC +
 * skinning (mirroring the synchronous dispose). Two historic defects:
 *   1. the poll-exit `return` sat BEFORE the phase-machine try/finally, so the
 *      finally's deferred-teardown finalisation never ran → `_pendingTeardown`
 *      never completed when dispose raced init.
 *   2. the deferred teardown only tore down pipeline+BVH+DDGI, leaking RC and
 *      skinning.
 */

interface TeardownSpies {
  teardownPipeline: ReturnType<typeof vi.fn>;
  disposeDdgi: ReturnType<typeof vi.fn>;
  disposeRc: ReturnType<typeof vi.fn>;
  disposeSkinning: ReturnType<typeof vi.fn>;
  setState: ReturnType<typeof vi.fn>;
}

/** Build a full PipelineInitHost whose scene never becomes ready (so the
 *  coordinator stays in the readiness poll) plus teardown spies. */
function makePollingHost(): { host: PipelineInitHost; spies: TeardownSpies } {
  const spies: TeardownSpies = {
    teardownPipeline: vi.fn(),
    disposeDdgi: vi.fn(),
    disposeRc: vi.fn(),
    disposeSkinning: vi.fn(),
    setState: vi.fn(),
  };
  const noop = () => {};
  const host = {
    device: {} as GPUDevice,
    width: 4,
    height: 4,
    // lastScene null + never-ready ⇒ poll keeps looping until dispose races.
    lastScene: null,
    primaryLightDir: [0, 1, 0] as const,
    primaryLightIntensity: 1,
    restirBvhModeOverride: undefined,
    denoiser: 'none' as const,
    neuralWeights: undefined,
    neuralTensorStorage: 'auto',
    oidnModelUrl: undefined,
    oidnExecutionProviders: undefined,
    verbose: false,
    debug: false,
    cameraMoveResetThresholdSq: 0,
    temporalAccumAlpha: 0,
    checkerboardMotionThresholdSq: 0,
    ctorLights: [],
    ddgi: {} as PipelineInitHost['ddgi'],
    hostSunWarningState: { warned: false },
    preferredSwapChainFormat: 'bgra8unorm' as GPUTextureFormat,
    gtaoMode: 'off' as const,
    diSpatialPasses: 1 as const,
    giSpatialPasses: 1 as const,
    grisReuse: false,
    nrcEnabled: false,
    nrcConfig: DEFAULT_NRC_CONFIG,
    ppgEnabled: false,
    ppgMaxSpatialCells: undefined,
    ppgMaxDTreeNodesPerCell: undefined,
    ppgMixAlpha: 0,
    checkerboard: false,
    ppgDispatchInterval: 1,
    regirConfig: undefined,
    isSceneReadyForBvh: () => false,
    coreSceneSuppliesMeshes: () => false,
    publishBvh: noop,
    publishPipeline: noop,
    rollbackBvh: noop,
    setState: spies.setState,
    reportError: noop,
    reportWarning: noop,
    teardownPipeline: spies.teardownPipeline,
    disposeDdgi: spies.disposeDdgi,
    disposeRc: spies.disposeRc,
    disposeSkinning: spies.disposeSkinning,
    recordInitStart: noop,
    recordInitComplete: noop,
    currentBvhBuffers: null,
  } satisfies PipelineInitHost;
  return { host, spies };
}

describe('PipelineInitCoordinator — deferred teardown on dispose-races-poll', () => {
  beforeEach(() => { vi.useFakeTimers(); });
  afterEach(() => { vi.useRealTimers(); });

  it('finalises the deferred teardown (pipeline+DDGI+RC+skinning) when dispose races the readiness poll', async () => {
    const { host, spies } = makePollingHost();
    const coord = new PipelineInitCoordinator(host);

    coord.startInit();
    expect(coord.initRunning).toBe(true);

    // Dispose arrives while the chain is parked in the scene-readiness poll
    // (isSceneReadyForBvh() === false ⇒ the loop keeps awaiting setTimeout(50)).
    const teardownNow = coord.requestTeardown();
    // An init is in flight ⇒ teardown is deferred to the chain, not run by the caller.
    expect(teardownNow).toBe(false);
    expect(coord.pendingTeardown).toBe(true);

    // Drain the poll's setTimeout(50) + the microtask queue so the chain
    // observes _disposed and hits the poll-exit path.
    await vi.advanceTimersByTimeAsync(60);

    // The deferred teardown must have completed via the poll-exit path — and it
    // must mirror the synchronous dispose: pipeline+BVH, DDGI, RC, AND skinning.
    expect(spies.teardownPipeline).toHaveBeenCalledTimes(1);
    expect(spies.disposeDdgi).toHaveBeenCalledTimes(1);
    expect(spies.disposeRc).toHaveBeenCalledTimes(1);
    expect(spies.disposeSkinning).toHaveBeenCalledTimes(1);
    expect(spies.setState).toHaveBeenCalledWith('disposed');
    // The chain has finished; it is no longer running.
    expect(coord.initRunning).toBe(false);
  });
});
