import { describe, expect, it, vi } from 'vitest';
import { asMat4, type Engine, type FrameInput, type FrameOutput } from '@vitrum/core';
import { ProgressiveHandoffCoordinator } from '../progressiveHandoff.js';

/** Minimal stub engine that records renderFrame / reset and reports an
 *  incrementing sample count (reset → 0). */
function makeStubEngine() {
  let samples = 0;
  const renderFrame = vi.fn((_input: FrameInput): FrameOutput => {
    samples += 1;
    return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
  });
  const reset = vi.fn(() => {
    samples = 0;
  });
  const engine = { renderFrame, reset, get samplesAccumulated() { return samples; } } as unknown as Engine;
  return { engine, renderFrame, reset };
}

function input(x: number): FrameInput {
  // Encode the camera position in the X translation so motion is controllable.
  return {
    viewMatrix: asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1])),
    projMatrix: asMat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])),
    cameraPosition: [x, 0, 0],
  } as unknown as FrameInput;
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
});
