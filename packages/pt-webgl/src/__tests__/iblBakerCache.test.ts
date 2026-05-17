/**
 * iblBakerCache.test.ts
 *
 * Verifies the per-instance ownership contract of {@link IblBakerCache}:
 * baking on cache A must not populate cache B. Replaces the test-coverage
 * the deleted `_skyEquirectCacheSize` back-channel provided for the previous
 * module-level singleton; with the singleton removed, the back-channel is
 * moot (W6-E2 in plan/premium-grade-refactor-20260517.md).
 *
 * Strategy: stub `three` and `three/examples/jsm/objects/Sky.js` so we never
 * touch GL — bake() runs the cache lookup, then walks the (now-no-op) GL
 * pipeline, then stores a sentinel into the cache. We assert via the public
 * `size` accessor that the two cache instances are independent.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('three/examples/jsm/objects/Sky.js', () => {
  class Sky {
    scale = { setScalar: vi.fn() };
    material = {
      uniforms: {
        turbidity: { value: 0 },
        rayleigh: { value: 0 },
        mieCoefficient: { value: 0 },
        mieDirectionalG: { value: 0 },
        sunPosition: { value: { set: vi.fn() } },
      },
      dispose: vi.fn(),
    };
    geometry = { dispose: vi.fn() };
  }
  return { Sky };
});

vi.mock('three', () => {
  // Sentinel constants so the bake() code path runs untouched. The DataTexture
  // returned by bake() doubles as the cache value — we don't care about its
  // shape for these tests, only that two caches don't share it.
  class Scene {
    add = vi.fn();
    remove = vi.fn();
  }
  class WebGLCubeRenderTarget {
    texture = {};
    constructor(_size: number, _opts: unknown) {}
    dispose = vi.fn();
  }
  class CubeCamera {
    constructor(_n: number, _f: number, _t: unknown) {}
    update = vi.fn();
  }
  class WebGLRenderTarget {
    constructor(_w: number, _h: number, _opts: unknown) {}
    dispose = vi.fn();
  }
  class ShaderMaterial {
    constructor(_opts: unknown) {}
    dispose = vi.fn();
  }
  class PlaneGeometry {
    constructor(_w: number, _h: number) {}
    dispose = vi.fn();
  }
  class Mesh {
    constructor(_g: unknown, _m: unknown) {}
  }
  class OrthographicCamera {
    constructor(..._args: number[]) {}
  }
  // bake() mutates fields on the DataTexture instance, so it needs to be
  // shaped enough to accept assignments without throwing.
  class DataTexture {
    mapping = 0;
    minFilter = 0;
    magFilter = 0;
    wrapS = 0;
    wrapT = 0;
    colorSpace = 0;
    needsUpdate = false;
    constructor(
      public data: unknown,
      public width: number,
      public height: number,
      public format: unknown,
      public type: unknown,
    ) {}
    dispose = vi.fn();
  }
  return {
    Scene,
    WebGLCubeRenderTarget,
    CubeCamera,
    WebGLRenderTarget,
    ShaderMaterial,
    PlaneGeometry,
    Mesh,
    OrthographicCamera,
    DataTexture,
    HalfFloatType: 0,
    RGBAFormat: 0,
    EquirectangularReflectionMapping: 0,
    LinearFilter: 0,
    RepeatWrapping: 0,
    ClampToEdgeWrapping: 0,
    NoColorSpace: 0,
  };
});

// Imports MUST come after vi.mock calls so the mocked modules are wired in.
import { IblBakerCache } from '../iblBaker.js';
import type { SkyParams } from '../skyParams.js';
import type { WebGLRenderer } from 'three';

function makeRendererStub(): WebGLRenderer {
  return {
    getRenderTarget: vi.fn(() => null),
    setRenderTarget: vi.fn(),
    clear: vi.fn(),
    render: vi.fn(),
    readRenderTargetPixels: vi.fn(),
    autoClear: true,
  } as unknown as WebGLRenderer;
}

function makeSkyParams(timeOfDayBucket: number): SkyParams {
  return {
    sunPosition: [Math.cos(timeOfDayBucket), Math.sin(timeOfDayBucket), 0.5],
    turbidity: 3.5,
    rayleigh: 2.1,
    mieCoefficient: 0.005,
    mieDirectionalG: 0.8,
  };
}

describe('IblBakerCache', () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it('a fresh cache reports size 0', () => {
    const cache = new IblBakerCache();
    expect(cache.size).toBe(0);
  });

  it('bake populates only the cache it was called on', () => {
    const cacheA = new IblBakerCache();
    const cacheB = new IblBakerCache();
    const renderer = makeRendererStub();
    const params = makeSkyParams(0.25);

    expect(cacheA.size).toBe(0);
    expect(cacheB.size).toBe(0);

    cacheA.bake(renderer, params);

    expect(cacheA.size).toBe(1);
    // Critical instance-isolation invariant: baking on A must not leak into B.
    expect(cacheB.size).toBe(0);

    cacheB.bake(renderer, params);
    expect(cacheA.size).toBe(1);
    expect(cacheB.size).toBe(1);
  });

  it('bake returns the same DataTexture on second call with same key (LRU hit)', () => {
    const cache = new IblBakerCache();
    const renderer = makeRendererStub();
    const params = makeSkyParams(0.25);

    const first = cache.bake(renderer, params);
    const second = cache.bake(renderer, params);

    expect(second).toBe(first);
    expect(cache.size).toBe(1);
  });

  it('clear() empties the cache and is per-instance', () => {
    const cacheA = new IblBakerCache();
    const cacheB = new IblBakerCache();
    const renderer = makeRendererStub();

    cacheA.bake(renderer, makeSkyParams(0.1));
    cacheA.bake(renderer, makeSkyParams(0.2));
    cacheB.bake(renderer, makeSkyParams(0.3));

    expect(cacheA.size).toBe(2);
    expect(cacheB.size).toBe(1);

    cacheA.clear();

    expect(cacheA.size).toBe(0);
    // Clearing A must not touch B.
    expect(cacheB.size).toBe(1);
  });

  it('evicts LRU entries when capacity is exceeded', () => {
    const cache = new IblBakerCache(2);
    const renderer = makeRendererStub();

    cache.bake(renderer, makeSkyParams(0.1));
    cache.bake(renderer, makeSkyParams(0.2));
    expect(cache.size).toBe(2);

    cache.bake(renderer, makeSkyParams(0.3));
    expect(cache.size).toBe(2);
  });
});
