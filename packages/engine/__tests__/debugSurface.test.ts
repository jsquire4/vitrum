// Verifies the EngineDebugSurface contract from @vitrum/core compiles
// + the @vitrum/engine proxy forwards the `debug` field when the wrapped
// engine implements it.
//
// Functional verification (HybridEngine.debug.atlasTexture() returning
// the live DDGI atlas, etc.) requires a real WebGPU device + a built
// scene; that coverage lives in the shader-compile-ci end-to-end smoke
// + the @vitrum/dev overlay tests once they're wired up.

import { describe, it, expect } from 'vitest';
import type {
  Engine,
  EngineCapabilities,
  EngineDebugSurface,
  EngineState,
  FrameInput,
  FrameOutput,
  Scene,
} from '@vitrum/core';

const NULL_CAPS: EngineCapabilities = {
  supportsIncrementalScene: false,
  supportsMotionBlur: false,
  supportsAuxBuffers: false,
  accumulates: false,
  maxSamplesPerPixel: Infinity,
  maxBounces: 1,
  supportedAnalyticShapes: new Set(),
  supportedEmitterKinds: new Set(),
  causticStrategy: 'none',
};

class DebuggableFakeEngine implements Engine {
  readonly state: EngineState = 'ready';
  readonly capabilities = NULL_CAPS;
  atlasCalls = 0;
  bvhCalls = 0;

  setScene(_: Scene): void {}
  renderFrame(_: FrameInput): FrameOutput {
    return { kind: 'skipped', reason: 'no-scene' };
  }
  reset(): void {}
  pause(): void {}
  resume(): void {}
  dispose(): void {}

  readonly debug: EngineDebugSurface = {
    atlasTexture: () => {
      this.atlasCalls++;
      return null;
    },
    bvhNodes: () => {
      this.bvhCalls++;
      return new Float32Array([
        -1, -1, -1, +1, +1, +1, 0, 0,
      ]);
    },
  };
}

describe('EngineDebugSurface contract', () => {
  it('all fields are optional — an empty {} satisfies the type', () => {
    const empty: EngineDebugSurface = {};
    expect(empty.atlasTexture).toBeUndefined();
    expect(empty.bvhNodes).toBeUndefined();
  });

  it('a partial implementation type-checks (atlasTexture but no bvhNodes)', () => {
    const partial: EngineDebugSurface = {
      atlasTexture: () => null,
    };
    expect(typeof partial.atlasTexture).toBe('function');
    expect(partial.atlasTexture!()).toBe(null);
  });

  it('bvhNodes returns 8 floats per node — [min, max, depth, pad]', () => {
    const e = new DebuggableFakeEngine();
    const nodes = e.debug.bvhNodes!();
    expect(nodes).not.toBeNull();
    expect(nodes!.length).toBe(8);
    expect(nodes!.length % 8).toBe(0);
    expect(nodes!.subarray(0, 3)).toEqual(new Float32Array([-1, -1, -1]));
    expect(nodes!.subarray(3, 6)).toEqual(new Float32Array([+1, +1, +1]));
  });

  it('giSignalTextures result fields are nullable per channel', () => {
    const surface: EngineDebugSurface = {
      giSignalTextures: () => ({
        direct: null,
        indirect: null,
        ao: null,
        total: null,
      }),
    };
    const sig = surface.giSignalTextures!();
    expect(sig).not.toBeNull();
    expect(sig!.direct).toBeNull();
  });

  it('subscribers can call methods through the engine.debug indirection', () => {
    const e = new DebuggableFakeEngine();
    e.debug.atlasTexture!();
    e.debug.atlasTexture!();
    e.debug.bvhNodes!();
    expect(e.atlasCalls).toBe(2);
    expect(e.bvhCalls).toBe(1);
  });
});
