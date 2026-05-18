import * as THREE from 'three';
import { Sky } from 'three/examples/jsm/objects/Sky.js';
import type { SkyParams } from './skyParams.js';

/**
 * Bake the analytic Preetham sky shader into a CPU-readable equirect
 * `DataTexture` suitable for both:
 *   - `scene.environment` under three-gpu-pathtracer's `WebGLPathTracer`
 *     (env importance sampling reads `image.data` to build CDF/PDF tables)
 *   - raster reflections via three's MeshStandardMaterial envMap path
 *     (the renderer auto-PMREMs an equirect texture assigned to `scene.environment`)
 *
 * Pipeline: Sky → CubeCamera → 2D equirect render target → readPixels
 * → `DataTexture(Uint16Array, RGBA, HalfFloat)`. The Float buffer is the
 * keepalive; intermediate GPU targets are disposed inside the bake function.
 *
 * Q-PT-1 resolution (Option A — raw non-unit Preetham position):
 * THREE.Sky.uniforms.sunPosition expects the raw non-unit Preetham sun position
 * vector, not a unit direction. ProceduralSkyEnvironment.sunDirection is a unit
 * vector and is therefore insufficient; the baker takes SkyParams directly so
 * callers can pass the raw arc position (magnitude ~1.12–1.41). Passing a unit
 * direction would shift the Preetham atmospheric scattering computation and
 * change sky color across the day cycle.
 *
 * Cache: bake outputs are keyed by quantised SkyParams (1 decimal on
 * sun-position components, 2 decimals on scalar params). Cycle of timeOfDay
 * [0,1] produces ~25 unique sun-position buckets at .toFixed(1) precision; the
 * LRU cache holds 32 entries (≥ 1 day cycle + the static night entry).
 *
 * The caller MUST NOT dispose the returned texture — eviction handles disposal.
 *
 * Multi-renderer hosts: instantiate one {@link IblBakerCache} per
 * `WebGLRenderer` and feed it through your engine wrapper. A baked sky texture
 * is bound to the GL context that produced it, so caches MUST NOT be shared
 * across renderers. The free-function {@link bakeSkyEquirect} retains a
 * process-global cache for back-compat and is deprecated.
 */

const CUBE_SIZE = 256;
const SKY_MESH_SCALE = 500; // inside the CubeCamera's default far=1000
const EQUIRECT_WIDTH = 512;
const EQUIRECT_HEIGHT = 256;
/** Default LRU capacity for {@link IblBakerCache}. Sized to cover one full
 *  timeOfDay cycle (≈25 unique sun-position buckets at .toFixed(1)) plus the
 *  static night entry, with headroom for atmospheric-param variation. */
const DEFAULT_CACHE_CAPACITY = 32;

interface CachedBake {
  /** CPU-readable equirect HDR — `image.data` is a `Uint16Array` (HalfFloat).
   *  `EquirectHdrInfoUniform.updateFrom` reads this on the PT side; raster
   *  uses it directly via three's auto-PMREM. */
  texture: THREE.DataTexture;
}

function quantiseKey(params: SkyParams): string {
  const [sx, sy, sz] = params.sunPosition;
  // Sun-position uses .toFixed(1) for cache-friendly bucketing across timeOfDay
  // scrubs (full day cycle ≈ 25 unique buckets). Scalar atmospheric params use
  // .toFixed(2) — they vary slowly and the higher precision keeps each distinct
  // atmospheric look in its own bucket.
  return [
    sx.toFixed(1),
    sy.toFixed(1),
    sz.toFixed(1),
    params.turbidity.toFixed(2),
    params.rayleigh.toFixed(2),
    params.mieCoefficient.toFixed(4),
    params.mieDirectionalG.toFixed(2),
  ].join('|');
}

const CUBE_TO_EQUIRECT_FRAG = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  uniform samplerCube tCube;
  const float PI = 3.141592653589793;
  void main() {
    // Equirectangular projection: vUv.x → azimuth, vUv.y → elevation.
    // Convention matches three's EquirectangularReflectionMapping —
    // u=0..1 wraps azimuth, v=0 at top pole (+Y), v=1 at bottom pole (-Y).
    float phi = (vUv.x - 0.5) * 2.0 * PI;
    float theta = (1.0 - vUv.y) * PI;
    vec3 dir = vec3(
      sin(theta) * cos(phi),
      cos(theta),
      sin(theta) * sin(phi)
    );
    gl_FragColor = textureCube(tCube, dir);
  }
`;

const CUBE_TO_EQUIRECT_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

export interface IblBakerCacheOptions {
  /** LRU capacity for baked sky textures. Defaults to {@link DEFAULT_CACHE_CAPACITY}
   *  (32 entries — covers one full timeOfDay cycle plus headroom). Must be >= 1. */
  readonly maxEntries?: number;
}

/**
 * Per-instance LRU cache + baker for analytic Preetham sky equirects.
 *
 * Hosts that own a single `WebGLRenderer` should construct one of these and
 * funnel all sky bakes through {@link bake}. The cache holds GPU-bound
 * `DataTexture` outputs that are valid only for the GL context that produced
 * them, so a multi-renderer host MUST instantiate one cache per renderer.
 *
 * On dispose() all retained DataTextures are released and the cache is emptied.
 * The instance is single-use after dispose(): subsequent {@link bake} calls
 * will throw.
 */
export class IblBakerCache {
  readonly #capacity: number;
  readonly #entries = new Map<string, CachedBake>();
  #disposed = false;

  constructor(opts?: IblBakerCacheOptions) {
    const requested = opts?.maxEntries ?? DEFAULT_CACHE_CAPACITY;
    if (!Number.isFinite(requested) || requested < 1) {
      throw new RangeError(
        `IblBakerCache: maxEntries must be a finite integer >= 1 (got ${requested})`,
      );
    }
    this.#capacity = Math.floor(requested);
  }

  /** Number of entries currently held — for tests and debug overlays. */
  get size(): number {
    return this.#entries.size;
  }

  /** Configured LRU capacity. */
  get capacity(): number {
    return this.#capacity;
  }

  /**
   * Bake (or fetch the cached bake of) an analytic sky equirect for `params`.
   * Returns a CPU-readable HalfFloat `DataTexture` whose `image.data` buffer
   * is the keepalive. The caller MUST NOT dispose the returned texture —
   * eviction (or {@link clear}/{@link dispose}) handles disposal.
   */
  bake(renderer: THREE.WebGLRenderer, params: SkyParams): THREE.DataTexture {
    if (this.#disposed) {
      throw new Error('IblBakerCache: bake() called after dispose()');
    }
    const key = quantiseKey(params);
    const cached = this.#entries.get(key);
    if (cached) {
      // Move to end of insertion order (LRU promotion).
      this.#entries.delete(key);
      this.#entries.set(key, cached);
      return cached.texture;
    }

    // 1. Procedural sky → cube target via CubeCamera.
    const sky = new Sky();
    sky.scale.setScalar(SKY_MESH_SCALE);
    const u = sky.material.uniforms;
    // THREE.Sky always has these uniforms — non-null assertions are safe here.
    u['turbidity']!.value = params.turbidity;
    u['rayleigh']!.value = params.rayleigh;
    u['mieCoefficient']!.value = params.mieCoefficient;
    u['mieDirectionalG']!.value = params.mieDirectionalG;
    u['sunPosition']!.value.set(...params.sunPosition);

    const tempScene = new THREE.Scene();
    tempScene.add(sky);

    const cubeTarget = new THREE.WebGLCubeRenderTarget(CUBE_SIZE, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      generateMipmaps: false,
    });
    const cubeCamera = new THREE.CubeCamera(0.1, 1000, cubeTarget);

    const prevTarget = renderer.getRenderTarget();
    const prevAutoClear = renderer.autoClear;
    renderer.autoClear = true;
    cubeCamera.update(renderer, tempScene);

    // 2. Cube → 2D equirect via fullscreen quad + custom shader.
    const equirectRT = new THREE.WebGLRenderTarget(EQUIRECT_WIDTH, EQUIRECT_HEIGHT, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      generateMipmaps: false,
    });

    const quadMat = new THREE.ShaderMaterial({
      uniforms: { tCube: { value: cubeTarget.texture } },
      vertexShader: CUBE_TO_EQUIRECT_VERT,
      fragmentShader: CUBE_TO_EQUIRECT_FRAG,
      depthWrite: false,
      depthTest: false,
    });
    const quadGeom = new THREE.PlaneGeometry(2, 2);
    const quad = new THREE.Mesh(quadGeom, quadMat);
    const quadScene = new THREE.Scene();
    quadScene.add(quad);
    const quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    renderer.setRenderTarget(equirectRT);
    renderer.clear();
    renderer.render(quadScene, quadCamera);

    // 3. Read pixels into a Uint16Array (HalfFloat raw bits).
    const buffer = new Uint16Array(EQUIRECT_WIDTH * EQUIRECT_HEIGHT * 4);
    renderer.readRenderTargetPixels(
      equirectRT,
      0,
      0,
      EQUIRECT_WIDTH,
      EQUIRECT_HEIGHT,
      buffer,
    );

    renderer.setRenderTarget(prevTarget);
    renderer.autoClear = prevAutoClear;

    // 4. Build the CPU-readable DataTexture. The buffer is the keepalive;
    //    EquirectHdrInfoUniform.updateFrom reads `image.data` directly.
    const dataTex = new THREE.DataTexture(
      buffer,
      EQUIRECT_WIDTH,
      EQUIRECT_HEIGHT,
      THREE.RGBAFormat,
      THREE.HalfFloatType,
    );
    dataTex.mapping = THREE.EquirectangularReflectionMapping;
    dataTex.minFilter = THREE.LinearFilter;
    dataTex.magFilter = THREE.LinearFilter;
    dataTex.wrapS = THREE.RepeatWrapping;
    dataTex.wrapT = THREE.ClampToEdgeWrapping;
    dataTex.colorSpace = THREE.NoColorSpace;
    dataTex.needsUpdate = true;

    // 5. Tear down GPU intermediates — the DataTexture (with CPU buffer) survives.
    quadMat.dispose();
    quadGeom.dispose();
    equirectRT.dispose();
    cubeTarget.dispose();
    sky.material.dispose();
    sky.geometry.dispose();
    tempScene.remove(sky);

    const entry: CachedBake = { texture: dataTex };
    this.#entries.set(key, entry);
    this.#evictOldestIfNeeded();
    return dataTex;
  }

  /**
   * Drop every cached sky bake, disposing the underlying DataTextures. The
   * instance remains usable — subsequent {@link bake} calls will repopulate.
   */
  clear(): void {
    for (const entry of this.#entries.values()) {
      entry.texture.dispose();
    }
    this.#entries.clear();
  }

  /**
   * Release all retained DataTextures and mark this instance as disposed.
   * Subsequent {@link bake} calls throw. Idempotent.
   */
  dispose(): void {
    if (this.#disposed) return;
    this.clear();
    this.#disposed = true;
  }

  #evictOldestIfNeeded(): void {
    while (this.#entries.size > this.#capacity) {
      const oldestKey = this.#entries.keys().next().value as string | undefined;
      if (!oldestKey) break;
      const entry = this.#entries.get(oldestKey);
      this.#entries.delete(oldestKey);
      if (entry) entry.texture.dispose();
    }
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Back-compat free-function API (process-global singleton).
//
// WARNING: Module-level singleton — shared across all WebGLRenderer instances
// in the same JS process. A baked sky texture is bound to the GL context that
// produced it, so consumers that create multiple renderers concurrently MUST
// NOT use this path. Instantiate one {@link IblBakerCache} per renderer
// instead.
// ────────────────────────────────────────────────────────────────────────────

let processGlobalCache: IblBakerCache | null = null;

function getProcessGlobalCache(): IblBakerCache {
  if (processGlobalCache == null) {
    processGlobalCache = new IblBakerCache();
  }
  return processGlobalCache;
}

/**
 * @deprecated Use `new IblBakerCache()` and call `cache.bake(renderer, params)`
 * instead. The free-function form reads/writes a process-global cache, which
 * is unsafe in multi-renderer hosts because baked DataTextures are bound to
 * the GL context that produced them. This wrapper is retained only for
 * back-compat with single-renderer callers and will be removed in a future
 * release.
 */
export function bakeSkyEquirect(
  renderer: THREE.WebGLRenderer,
  params: SkyParams,
): THREE.DataTexture {
  return getProcessGlobalCache().bake(renderer, params);
}

/**
 * @deprecated Use `IblBakerCache#clear()` or `IblBakerCache#dispose()` on the
 * per-instance cache instead. Drops every entry held by the process-global
 * cache used by {@link bakeSkyEquirect}.
 */
export function clearSkyEquirectCache(): void {
  if (processGlobalCache != null) {
    processGlobalCache.clear();
  }
}

/**
 * @deprecated Use `IblBakerCache#size` on the per-instance cache instead.
 * Inspect the process-global cache size — for tests and debug overlays only.
 */
export function _skyEquirectCacheSize(): number {
  return processGlobalCache?.size ?? 0;
}
