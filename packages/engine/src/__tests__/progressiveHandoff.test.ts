import { describe, expect, it, vi } from 'vitest';
import { asMat4, type Engine, type FrameInput, type FrameOutput, type Scene, type ScenePrimitive } from '@vitrum/core';
import {
  ProgressiveHandoffCoordinator,
  type ProgressiveHandoffController,
} from '../progressiveHandoff.js';

/** Minimal stub engine that records renderFrame / reset and reports an
 *  incrementing sample count (reset → 0). */
function makeStubEngine(convergeAt = Infinity) {
  let samples = 0;
  const renderFrame = vi.fn((_input: FrameInput): FrameOutput => {
    samples += 1;
    // Report accumulating samples so settle-behind can gate on them.
    return {
      kind: 'rendered',
      samplesAccumulated: samples,
      isConverged: samples >= convergeAt,
      primaryRadiance: {},
    } as unknown as FrameOutput;
  });
  const reset = vi.fn(() => {
    samples = 0;
  });
  const setScene = vi.fn();
  const updatePrimitive = vi.fn();
  const addPrimitive = vi.fn();
  const removePrimitive = vi.fn();
  const engine = {
    renderFrame, reset, setScene, updatePrimitive, addPrimitive, removePrimitive,
    get samplesAccumulated() { return samples; },
  } as unknown as Engine;
  return { engine, renderFrame, reset, setScene, updatePrimitive, addPrimitive, removePrimitive };
}

function input(x: number): FrameInput {
  // Encode the camera position in the X translation so motion is controllable.
  return {
    viewMatrix: asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1])),
    projMatrix: asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
    cameraPosition: [x, 0, 0],
  } as unknown as FrameInput;
}

function meshPrimitive(id = 'p'): ScenePrimitive {
  return {
    kind: 'mesh',
    id,
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    material: { baseColor: [1, 1, 1], roughness: 1, metallic: 0 },
  } as unknown as ScenePrimitive;
}

function sceneWithPrimitive(primitive = meshPrimitive()): Scene {
  return {
    primitives: [primitive],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('ProgressiveHandoffCoordinator', () => {
  it('renders the real-time engine on the first frame and while the camera moves', () => {
    const rt = makeStubEngine();
    const cv = makeStubEngine();
    const c = new ProgressiveHandoffCoordinator({ realtime: rt.engine, converged: cv.engine, stillFramesBeforeHandoff: 3 });

    expect(c.frame(input(0)).phase).toBe('realtime');       // first frame
    expect(c.frame(input(1)).phase).toBe('realtime');       // moved
    expect(c.frame(input(2)).phase).toBe('realtime');       // moved
    expect(rt.renderFrame).toHaveBeenCalledTimes(3);
    expect(cv.renderFrame).not.toHaveBeenCalled();
  });

  it('settles, then hands off to the converged engine after the threshold (reset once on entry)', () => {
    const rt = makeStubEngine();
    const cv = makeStubEngine();
    const c = new ProgressiveHandoffCoordinator({ realtime: rt.engine, converged: cv.engine, stillFramesBeforeHandoff: 3 });

    c.frame(input(5));                                       // frame 1 — moved (from null) → realtime
    expect(c.frame(input(5)).phase).toBe('settling');       // still 1
    expect(c.frame(input(5)).phase).toBe('settling');       // still 2
    const handoff = c.frame(input(5));                      // still 3 → threshold
    expect(handoff.phase).toBe('converging');
    expect(handoff.active).toBe(cv.engine);
    expect(cv.reset).toHaveBeenCalledTimes(1);              // reset on entry
    expect(cv.renderFrame).toHaveBeenCalledTimes(1);

    // Subsequent still frames keep accumulating WITHOUT another reset.
    c.frame(input(5));
    c.frame(input(5));
    expect(cv.reset).toHaveBeenCalledTimes(1);
    expect(cv.renderFrame).toHaveBeenCalledTimes(3);
  });

  it('snaps back to real-time on camera motion and re-resets the converged accumulator next settle', () => {
    const rt = makeStubEngine();
    const cv = makeStubEngine();
    const c = new ProgressiveHandoffCoordinator({ realtime: rt.engine, converged: cv.engine, stillFramesBeforeHandoff: 2 });

    c.frame(input(0));            // realtime
    c.frame(input(0));            // settling (still 1)
    expect(c.frame(input(0)).phase).toBe('converging'); // still 2 → handoff
    expect(cv.reset).toHaveBeenCalledTimes(1);

    // Camera moves → back to real-time.
    expect(c.frame(input(9)).phase).toBe('realtime');
    expect(rt.renderFrame).toHaveBeenCalledTimes(3); // 2 settle/realtime + this one

    // Settle again → converged accumulator reset AGAIN (stale from the prior camera).
    c.frame(input(9));
    expect(c.frame(input(9)).phase).toBe('converging');
    expect(cv.reset).toHaveBeenCalledTimes(2);
  });

  it('treats sub-epsilon camera jitter as still', () => {
    const rt = makeStubEngine();
    const cv = makeStubEngine();
    const c = new ProgressiveHandoffCoordinator({ realtime: rt.engine, converged: cv.engine, stillFramesBeforeHandoff: 2, cameraEpsilon: 1e-4 });
    c.frame(input(0));
    c.frame(input(1e-6));            // jitter below epsilon → still
    expect(c.frame(input(2e-6)).phase).toBe('converging');
  });

  it('settleBehindRealtime: accumulates converged BEHIND real-time, then switches the display when clean', () => {
    const rt = makeStubEngine();
    const cv = makeStubEngine();
    const c = new ProgressiveHandoffCoordinator({
      realtime: rt.engine, converged: cv.engine,
      stillFramesBeforeHandoff: 1, settleBehindRealtime: true, convergedDisplaySamples: 3,
    });

    c.frame(input(0));                                  // realtime (moved from null)
    // Now still: converged renders behind, display stays real-time until 3 samples.
    let r = c.frame(input(0));
    expect(r.phase).toBe('prerolling');                 // converged sample 1 — display real-time
    expect(r.active).toBe(rt.engine);
    expect(r.behindOutput?.samplesAccumulated).toBe(1); // converged accumulating behind, exposed for crossfade
    r = c.frame(input(0));
    expect(r.phase).toBe('prerolling');                 // converged sample 2
    r = c.frame(input(0));
    expect(r.phase).toBe('converging');                 // converged sample 3 ≥ display threshold → switch
    expect(r.active).toBe(cv.engine);
    // The converged engine rendered every frame since handoff (3×); real-time only
    // during the two pre-roll frames.
    expect(cv.renderFrame).toHaveBeenCalledTimes(3);
    expect(cv.reset).toHaveBeenCalledTimes(1);
  });

  it('settleBehindRealtime: switches early when the converged engine reports isConverged', () => {
    const rt = makeStubEngine();
    const cv = makeStubEngine(2); // converges (isConverged) at 2 samples
    const c = new ProgressiveHandoffCoordinator({
      realtime: rt.engine, converged: cv.engine,
      stillFramesBeforeHandoff: 1, settleBehindRealtime: true, convergedDisplaySamples: 1000,
    });
    c.frame(input(0));                                  // realtime
    expect(c.frame(input(0)).phase).toBe('prerolling'); // sample 1, not converged
    expect(c.frame(input(0)).phase).toBe('converging');  // sample 2 → isConverged → switch
  });

  it('scene authority: forwards scene mutations to BOTH engines and restarts at real-time', () => {
    const rt = makeStubEngine();
    const cv = makeStubEngine();
    const c = new ProgressiveHandoffCoordinator({ realtime: rt.engine, converged: cv.engine, stillFramesBeforeHandoff: 1 });

    // Settle into converging first.
    c.frame(input(0));
    expect(c.frame(input(0)).phase).toBe('converging');

    const scene = sceneWithPrimitive();
    c.setScene(scene);
    expect(rt.setScene).toHaveBeenCalledWith(scene);
    expect(cv.setScene).toHaveBeenCalledWith(scene);
    expect(c.phase).toBe('realtime'); // scene change → back to real-time

    c.updatePrimitive('p', { transform: undefined } as never);
    expect(rt.updatePrimitive).toHaveBeenCalledTimes(1);
    expect(cv.updatePrimitive).toHaveBeenCalledTimes(1);

    const prim = { kind: 'mesh', id: 'q' } as unknown as Parameters<typeof c.addPrimitive>[0];
    c.addPrimitive(prim);
    expect(rt.addPrimitive).toHaveBeenCalledWith(prim);
    expect(cv.addPrimitive).toHaveBeenCalledWith(prim);

    c.removePrimitive('q');
    expect(rt.removePrimitive).toHaveBeenCalledWith('q');
    expect(cv.removePrimitive).toHaveBeenCalledWith('q');
  });

  it('falls back to setScene on both engines when an incremental primitive method is absent', () => {
    const rt = makeStubEngine();
    const cv = makeStubEngine();
    delete (cv.engine as Partial<Engine>).updatePrimitive;
    const scene = sceneWithPrimitive();
    const nextPositions = new Float32Array([9, 0, 0, 1, 0, 0, 0, 1, 0]);
    const c = new ProgressiveHandoffCoordinator({
      realtime: rt.engine,
      converged: cv.engine,
      scene,
      stillFramesBeforeHandoff: 1,
    });

    c.updatePrimitive('p', { positions: nextPositions } as Partial<ScenePrimitive>);

    expect(rt.updatePrimitive).not.toHaveBeenCalled();
    expect(rt.setScene).toHaveBeenCalledTimes(1);
    expect(cv.setScene).toHaveBeenCalledTimes(1);
    const patched = rt.setScene.mock.calls[0]![0] as Scene;
    expect((patched.primitives[0] as { positions: Float32Array }).positions).toBe(nextPositions);
    expect(cv.setScene).toHaveBeenCalledWith(patched);
    expect(c.phase).toBe('realtime');
  });

  it('throws before mutating either engine when no scene fallback exists and an incremental method is absent', () => {
    const rt = makeStubEngine();
    const cv = makeStubEngine();
    delete (cv.engine as Partial<Engine>).updatePrimitive;
    const c = new ProgressiveHandoffCoordinator({
      realtime: rt.engine,
      converged: cv.engine,
      stillFramesBeforeHandoff: 1,
    });

    expect(() => {
      c.updatePrimitive('p', { positions: new Float32Array(9) } as Partial<ScenePrimitive>);
    }).toThrow(/both engines must implement updatePrimitive/);
    expect(rt.updatePrimitive).not.toHaveBeenCalled();
    expect(cv.setScene).not.toHaveBeenCalled();
  });

  it('preserves core patch invariants when building a scene fallback', () => {
    const rt = makeStubEngine();
    const cv = makeStubEngine();
    delete (cv.engine as Partial<Engine>).updatePrimitive;
    const c = new ProgressiveHandoffCoordinator({
      realtime: rt.engine,
      converged: cv.engine,
      scene: sceneWithPrimitive(),
      stillFramesBeforeHandoff: 1,
    });

    expect(() => {
      c.updatePrimitive('missing', { positions: new Float32Array(9) } as Partial<ScenePrimitive>);
    }).toThrow(/primitive "missing" not found/);
    expect(rt.setScene).not.toHaveBeenCalled();
    expect(cv.setScene).not.toHaveBeenCalled();
  });

  it('falls back to setScene on both engines when a primitive fast path rejects', () => {
    const rt = makeStubEngine();
    const cv = makeStubEngine();
    cv.updatePrimitive.mockImplementation(() => {
      throw new Error('backend cannot patch this primitive');
    });
    const scene = sceneWithPrimitive();
    const nextPositions = new Float32Array([4, 0, 0, 1, 0, 0, 0, 1, 0]);
    const c = new ProgressiveHandoffCoordinator({
      realtime: rt.engine,
      converged: cv.engine,
      scene,
      stillFramesBeforeHandoff: 1,
    });

    c.updatePrimitive('p', { positions: nextPositions } as Partial<ScenePrimitive>);

    expect(rt.updatePrimitive).toHaveBeenCalledTimes(1);
    expect(cv.updatePrimitive).toHaveBeenCalledTimes(1);
    expect(rt.setScene).toHaveBeenCalledTimes(1);
    expect(cv.setScene).toHaveBeenCalledTimes(1);
    const patched = cv.setScene.mock.calls[0]![0] as Scene;
    expect((patched.primitives[0] as { positions: Float32Array }).positions).toBe(nextPositions);
  });

  it('reset() forces back to real-time and re-arms the converged reset', () => {
    const rt = makeStubEngine();
    const cv = makeStubEngine();
    const c = new ProgressiveHandoffCoordinator({ realtime: rt.engine, converged: cv.engine, stillFramesBeforeHandoff: 1 });
    c.frame(input(0));
    expect(c.frame(input(0)).phase).toBe('converging');
    c.reset();
    expect(c.phase).toBe('realtime');
    expect(c.frame(input(0)).phase).toBe('realtime'); // first frame after reset is "moved"
  });

  it('advances a scene controller once per frame and routes primitive patches to both engines before render', () => {
    const rt = makeStubEngine();
    const cv = makeStubEngine();
    const materialPatch = {
      material: { baseColor: [0.25, 0.5, 0.75], roughness: 1, metallic: 0 },
    } as Partial<ScenePrimitive>;
    const advance = vi.fn(
      (_deltaSeconds: number, options?: Parameters<ProgressiveHandoffController['advance']>[1]) => {
        options?.engine?.updatePrimitive?.('p', materialPatch);
      },
    );
    const controller: ProgressiveHandoffController = { animations: [{}], advance };
    const c = new ProgressiveHandoffCoordinator({
      realtime: rt.engine,
      converged: cv.engine,
      scene: sceneWithPrimitive(),
      stillFramesBeforeHandoff: 1,
      controller,
    });

    const result = c.frame(input(0));

    expect(result.phase).toBe('realtime');
    expect(advance).toHaveBeenCalledTimes(1);
    expect(advance).toHaveBeenCalledWith(1 / 60, {
      engine: expect.objectContaining({ setScene: expect.any(Function), updatePrimitive: expect.any(Function) }),
      loop: true,
    });
    expect(rt.updatePrimitive).toHaveBeenCalledWith('p', materialPatch);
    expect(cv.updatePrimitive).toHaveBeenCalledWith('p', materialPatch);
    expect(rt.updatePrimitive.mock.invocationCallOrder[0] ?? 0)
      .toBeLessThan(rt.renderFrame.mock.invocationCallOrder[0] ?? 0);
  });

  it('uses the host controller delta callback and loop flag', () => {
    const rt = makeStubEngine();
    const cv = makeStubEngine();
    const advance = vi.fn();
    const controller: ProgressiveHandoffController = { animations: [{}], advance };
    const controllerDeltaSeconds = vi.fn((_frame: FrameInput, state: { phase: string; stillFrames: number }) =>
      state.phase === 'realtime' && state.stillFrames === 0 ? 0.125 : 0.25,
    );
    const c = new ProgressiveHandoffCoordinator({
      realtime: rt.engine,
      converged: cv.engine,
      controller,
      controllerDeltaSeconds,
      controllerLoop: false,
    });

    c.frame(input(0));

    expect(controllerDeltaSeconds).toHaveBeenCalledTimes(1);
    expect(advance).toHaveBeenCalledWith(0.125, {
      engine: expect.any(Object),
      loop: false,
    });
  });

  it('skips glTF-style controllers that report no animations', () => {
    const rt = makeStubEngine();
    const cv = makeStubEngine();
    const advance = vi.fn(() => {
      throw new Error('should not advance');
    });
    const controller: ProgressiveHandoffController = { animations: [], advance };
    const c = new ProgressiveHandoffCoordinator({
      realtime: rt.engine,
      converged: cv.engine,
      controller,
      stillFramesBeforeHandoff: 1,
    });

    expect(() => c.frame(input(0))).not.toThrow();
    expect(advance).not.toHaveBeenCalled();
    expect(rt.renderFrame).toHaveBeenCalledTimes(1);
  });

  it('falls back to setScene on both engines when controller patches need the scene fallback', () => {
    const rt = makeStubEngine();
    const cv = makeStubEngine();
    delete (cv.engine as Partial<Engine>).updatePrimitive;
    const nextPositions = new Float32Array([3, 0, 0, 1, 0, 0, 0, 1, 0]);
    const advance = vi.fn(
      (_deltaSeconds: number, options?: Parameters<ProgressiveHandoffController['advance']>[1]) => {
        options?.engine?.updatePrimitive?.('p', { positions: nextPositions } as Partial<ScenePrimitive>);
      },
    );
    const controller: ProgressiveHandoffController = { animations: [{}], advance };
    const c = new ProgressiveHandoffCoordinator({
      realtime: rt.engine,
      converged: cv.engine,
      scene: sceneWithPrimitive(),
      controller,
      stillFramesBeforeHandoff: 1,
    });

    c.frame(input(0));

    expect(rt.updatePrimitive).not.toHaveBeenCalled();
    expect(rt.setScene).toHaveBeenCalledTimes(1);
    expect(cv.setScene).toHaveBeenCalledTimes(1);
    const patched = rt.setScene.mock.calls[0]![0] as Scene;
    expect((patched.primitives[0] as { positions: Float32Array }).positions).toBe(nextPositions);
    expect(cv.setScene).toHaveBeenCalledWith(patched);
  });
});

describe('ProgressiveHandoffCoordinator — seed-on-handoff (P8 increment 2)', () => {
  function seedSource() {
    const s = makeStubEngine();
    const fakeTex = { __seedTex: true };
    const getProgressiveSeedTexture = vi.fn(() => ({ texture: fakeTex, width: 320, height: 180 }));
    (s.engine as unknown as { getProgressiveSeedTexture: unknown }).getProgressiveSeedTexture = getProgressiveSeedTexture;
    return { ...s, getProgressiveSeedTexture, fakeTex };
  }
  function seedSink(convergeAt = Infinity) {
    const s = makeStubEngine(convergeAt);
    const seedAccumulator = vi.fn();
    (s.engine as unknown as { seedAccumulator: unknown }).seedAccumulator = seedAccumulator;
    return { ...s, seedAccumulator };
  }

  it('seeds the converged accumulator from the real-time source on handoff, AFTER reset', () => {
    const rt = seedSource();
    const cv = seedSink();
    const c = new ProgressiveHandoffCoordinator({
      realtime: rt.engine, converged: cv.engine, stillFramesBeforeHandoff: 2,
      seedFromRealtime: true, seedWeight: 7,
    });
    c.frame(input(0)); c.frame(input(0)); c.frame(input(0)); // handoff on the 3rd
    expect(cv.reset).toHaveBeenCalledTimes(1);
    expect(rt.getProgressiveSeedTexture).toHaveBeenCalledTimes(1);
    expect(cv.seedAccumulator).toHaveBeenCalledTimes(1);
    expect(cv.seedAccumulator).toHaveBeenCalledWith(rt.fakeTex, { weight: 7, width: 320, height: 180 });
    // reset must run BEFORE the seed (the seed is the sole prior on a cleared accum).
    expect(cv.reset.mock.invocationCallOrder[0] ?? 0).toBeLessThan(cv.seedAccumulator.mock.invocationCallOrder[0] ?? 0);
  });

  it('does NOT seed when seedFromRealtime is false (default) — resets to black', () => {
    const rt = seedSource();
    const cv = seedSink();
    const c = new ProgressiveHandoffCoordinator({ realtime: rt.engine, converged: cv.engine, stillFramesBeforeHandoff: 2 });
    c.frame(input(0)); c.frame(input(0)); c.frame(input(0));
    expect(cv.reset).toHaveBeenCalledTimes(1);
    expect(cv.seedAccumulator).not.toHaveBeenCalled();
    expect(rt.getProgressiveSeedTexture).not.toHaveBeenCalled();
  });

  it('is a graceful no-op when the engines lack the optional seed methods', () => {
    const rt = makeStubEngine(); // no getProgressiveSeedTexture
    const cv = makeStubEngine(); // no seedAccumulator
    const c = new ProgressiveHandoffCoordinator({ realtime: rt.engine, converged: cv.engine, stillFramesBeforeHandoff: 2, seedFromRealtime: true });
    expect(() => { c.frame(input(0)); c.frame(input(0)); c.frame(input(0)); }).not.toThrow();
    expect(cv.reset).toHaveBeenCalledTimes(1);
  });

  it('re-seeds on each settle after camera motion invalidates the converged accumulator', () => {
    const rt = seedSource();
    const cv = seedSink();
    const c = new ProgressiveHandoffCoordinator({ realtime: rt.engine, converged: cv.engine, stillFramesBeforeHandoff: 2, seedFromRealtime: true });
    c.frame(input(0)); c.frame(input(0)); c.frame(input(0)); // handoff 1 → seed
    expect(cv.seedAccumulator).toHaveBeenCalledTimes(1);
    c.frame(input(9));                                        // moved → stale
    c.frame(input(9)); c.frame(input(9));                     // settle → handoff 2 → re-seed
    expect(cv.seedAccumulator).toHaveBeenCalledTimes(2);
  });

  it('Bug2 fix — passes DESTINATION (viewport) dims to seedAccumulator, not source dims', () => {
    // When source and destination have different resolutions (e.g. different
    // resolutionFactors), the seed call must carry the DESTINATION accumulator
    // dims so the backend resamples the seed to the right output size.
    // We inject a viewport that is DIFFERENT from the source texture dims.
    const rt = seedSource(); // source: 320×180
    const cv = seedSink();
    const c = new ProgressiveHandoffCoordinator({
      realtime: rt.engine, converged: cv.engine, stillFramesBeforeHandoff: 2,
      seedFromRealtime: true, seedWeight: 4,
    });
    // Build frames with an explicit viewport of 640×360 (2× the source).
    function inputWithViewport(x: number): FrameInput {
      return {
        viewMatrix: asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1])),
        projMatrix: asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
        cameraPosition: [x, 0, 0],
        viewport: { width: 640, height: 360, devicePixelRatio: 1 },
      } as unknown as FrameInput;
    }
    c.frame(inputWithViewport(0)); // moved (first frame)
    c.frame(inputWithViewport(0)); // still #1
    c.frame(inputWithViewport(0)); // still #2 → handoff
    // width/height must be VIEWPORT dims (640×360), NOT source dims (320×180).
    expect(cv.seedAccumulator).toHaveBeenCalledWith(rt.fakeTex, { weight: 4, width: 640, height: 360 });
  });
});
