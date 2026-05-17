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
 * Ownership: the cache is INSTANCE-level via {@link IblBakerCache}. Each
 * `WebGLRenderer` should be paired with its own `IblBakerCache` — a baked sky
 * texture is bound to the GL context that produced it, so sharing a cache
 * across renderers risks cross-context texture reads. `PTEngineWebGL2` owns one
 * cache per engine instance; hosts that bake outside the engine (raster
 * reflections, debug overlays) instantiate their own.
 *
 * The caller MUST NOT dispose the returned texture — eviction handles disposal.
 */

const CUBE_SIZE = 256;
const SKY_MESH_SCALE = 500; // inside the CubeCamera's default far=1000
const EQUIRECT_WIDTH = 512;
const EQUIRECT_HEIGHT = 256;
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

/**
 * Per-instance sky-equirect bake cache. One instance per renderer / engine.
 *
 * Backed by an insertion-ordered `Map<string, CachedBake>` with LRU promotion
 * on hit; eviction disposes the underlying `DataTexture`. Capacity defaults to
 * 32 entries — comfortably above the ~25 unique sun-position buckets a full
 * timeOfDay cycle produces at `.toFixed(1)` precision, plus the static night
 * entry, with margin for atmospheric-parameter variation.
 */
export class IblBakerCache {
  readonly #cache = new Map<string, CachedBake>();
  readonly #capacity: number;

  constructor(capacity: number = DEFAULT_CACHE_CAPACITY) {
    this.#capacity = Math.max(1, Math.floor(capacity));
  }

  /** Number of currently cached bakes. */
  get size(): number {
    return this.#cache.size;
  }

  /**
   * Bake (or look up) an equirect sky for the given params. See module
   * documentation for the pipeline and ownership contract.
   *
   * The returned `DataTexture` is owned by the cache — callers MUST NOT
   * dispose it. Eviction (LRU on capacity overflow) handles disposal; call
   * {@link clear} on app teardown to release everything.
   */
  bake(renderer: THREE.WebGLRenderer, params: SkyParams): THREE.DataTexture {
    const key = quantiseKey(params);
    const cached = this.#cache.get(key);
    if (cached) {
      // Move to end of insertion order (LRU promotion).
      this.#cache.delete(key);
      this.#cache.set(key, cached);
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
    this.#cache.set(key, entry);
    this.#evictOldestIfNeeded();
    return dataTex;
  }

  /** Drop every cached bake. Call on app teardown / hot-reload to release
   *  the `DataTexture` buffers held by this cache. */
  clear(): void {
    for (const entry of this.#cache.values()) {
      entry.texture.dispose();
    }
    this.#cache.clear();
  }

  #evictOldestIfNeeded(): void {
    while (this.#cache.size > this.#capacity) {
      const oldestKey = this.#cache.keys().next().value as string | undefined;
      if (oldestKey === undefined) break;
      const entry = this.#cache.get(oldestKey);
      this.#cache.delete(oldestKey);
      if (entry) entry.texture.dispose();
    }
  }
}
