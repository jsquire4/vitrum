import type { EngineOptions } from '@vitrum/core';
import type { WebGl2TraceTier } from './traceTier.js';
import type { OIDNBridgeLoader, OidnReadbackFn } from './denoise/oidnFinalDispatcher.js';

// `WebGl2TraceTier` is owned by ./traceTier.ts (the tier-selection module);
// re-exported here so consumers of the options surface keep importing it from
// one place. (Previously this module re-declared an identical union — D9 dedup.)
/** @public — public option type for pt-webgl2 consumers; controls quality tier selection. */
export type { WebGl2TraceTier };

/**
 * Options for {@link createPTEngine_WebGL2}. Narrows the core `EngineOptions.device`
 * (typed `unknown`) to a host-owned `WebGL2RenderingContext`. The engine allocates
 * GL resources against it but NEVER loses or destroys the context — the host owns
 * the lifecycle (design principle #2).
 */
export interface PTEngineWebGL2Options extends EngineOptions {
  readonly device: WebGL2RenderingContext;
  /**
   * Maximum scattering depth. WebGL2 shader loops are statically bounded, so
   * this must be an integer in the supported range 1..32. Default 8; 32 remains
   * available as an explicit high-depth opt-in.
   */
  readonly maxBounces?: number;
  /**
   * Peak byte budget for frame-sized GPU render targets. The estimator includes
   * the raw accumulator, four NEE candidate attachments, two running-mean blend
   * slots, the present target, optional auxiliary G-buffer/OIDN targets, and the
   * transactional overlap between the current published frame and a replacement.
   * Default: 512 MiB. Requests above the budget throw before GL allocation.
   */
  readonly maxRenderTargetBytes?: number;
  /** Select the output profile; default `full`. `lite` omits albedo/normal MRTs. */
  readonly traceTier?: WebGl2TraceTier;
  /** Spectral hero-wavelength rendering (S3). Default false. */
  readonly spectral?: boolean;
  /** Bidirectional path tracing (S3). Default false. */
  readonly bdpt?: boolean;
  /**
    * Primary path-sampling sequence. Default `'pcg'` preserves the historical random
    * stream. `'sobol'` composes the dedicated Sobol sampler and uploads a real
    * 256x256 RGBA32F direction texture.
    */
  readonly sampling?: 'pcg' | 'sobol';
  /** BDPT tuning — requires `bdpt:true` or `causticStrategy:'bdpt'`. */
  readonly bdptOptions?: {
    /**
     * Maximum stored light-subpath vertices, including the sampled emitter
     * endpoint. The bounded general-BDPT walk accepts 1..8 vertices; default 4.
     * Finite emitters connect from c=0; c>=1 extends through stored surface or
     * participating-medium vertices with an eight-entry nested homogeneous-
     * medium stack, exact Beer visibility, and authored HG phase PDFs.
     * Directional/environment paths
     * use a disjoint partition: primary c=0 NEE, camera/delta forward escape,
     * and c>=1 BDPT connections.
     */
    readonly maxLightBounces?: number;
  };
  /**
   * Optional texture-fetch LOD by bounce depth. `0` (default) disables the optimization
   * and samples material textures at every bounce, preserving the historical highest-
   * fidelity behavior. A positive value samples textures only while
   * `pathDepth <= materialLodDepth`; deeper bounces use flat material constants.
   */
  readonly materialLodDepth?: number;
  /**
   * Select the bounded general-BDPT estimator as the named caustic strategy.
   * This is ordinary bidirectional path tracing with Veach MIS, not MNEE or
   * photon mapping. Selecting it enables the same estimator as `bdpt:true`.
   */
  readonly causticStrategy?: 'bdpt';
  /** The BDPT caustic strategy has no separate tuning bag; use `bdptOptions`. */
  readonly causticOptions?: never;
  /**
   * Opacity of directly-visible background (sky/HDRI) pixels, in [0, 1].
   * Default 1 (opaque — the environment is visible behind the scene). Values < 1
   * make the background partially/fully transparent. The renderer's portable
   * alpha-aware running-mean compositor preserves that coverage on every WebGL2
   * device. Before this existed the uniform defaulted to 0 and directly-visible
   * env never accumulated (rendered black) — see items_to_fix §H3.
   */
  readonly backgroundAlpha?: number;
  /**
   * Background/environment blur radius for directly-visible miss rays. Default 0
   * preserves the sharp environment lookup; positive values opt into the fork's
   * stochastic background blur approximation.
   */
  readonly backgroundBlur?: number;
  /**
   * Camera projection model (flag-plumbing audit 2026-06-10). The GLSL
   * `getCameraRay` fully implements all three; the host declares which one its
   * `projMatrix` represents (the matrix alone can't disambiguate equirect, which
   * is not a linear projection). Default `'perspective'` — byte-identical to the
   * prior fixed CAMERA_TYPE=0 path.
   *   • 'perspective'     — standard pinhole (CAMERA_TYPE 0)
   *   • 'orthographic'    — parallel rays along camera −Z (CAMERA_TYPE 1)
   *   • 'equirectangular' — 360° panoramic capture (CAMERA_TYPE 2)
   */
  readonly cameraType?: 'perspective' | 'orthographic' | 'equirectangular';
  /**
   * Thin-lens depth of field (flag-plumbing audit 2026-06-10). The aperture
   * sampler activates only when `bokehSize > 0` after float32 packing. Active
   * thin-lens DOF is incompatible with `cameraType:'equirectangular'`, which has
   * no coherent full-sphere focal plane. A DOF object with `bokehSize:0` remains
   * an exact pinhole and is valid
   * for every camera type. The object carries PhysicalCamera
   * uniforms. Omitted (default) → pinhole camera, byte-identical to the prior
   * fixed FEATURE_DOF=0 path. All distances are in scene units; `bokehSize` is the
   * aperture diameter in millimetres (matches the fork's `bokehSize·0.5·1e-3` scale).
   */
  readonly dof?: {
    readonly focusDistance: number;
    /** Aperture diameter (mm). Larger = shallower depth of field. */
    readonly bokehSize: number;
    /** Aperture polygon sides; 0 = perfect circle (default 0). */
    readonly apertureBlades?: number;
    /** Aperture rotation (radians, default 0). */
    readonly apertureRotation?: number;
    /** Anamorphic squeeze ratio (1 = spherical, default 1). */
    readonly anamorphicRatio?: number;
  };
  /**
   * Intel Open Image Denoise final-pass config. Required when
   * `denoiser: 'oidn-final'`.
   *
   * The host must provide both the ONNX model asset and the optional
   * `onnxruntime-web` peer dependency. The backend reads its linear HDR
   * accumulator plus available MRT aux buffers into CPU Float32 tensors and
   * runs the shared OIDN bridge asynchronously once the PT accumulation
   * reaches the requested SPP target.
   */
  readonly oidn?: {
    readonly modelUrl: string;
    readonly executionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;
  };
  /** Test-only: inject a mock OIDN bridge. */
  readonly oidnBridgeLoader?: OIDNBridgeLoader;
  /** Test-only: override the WebGL attachment readback before OIDN inference. */
  readonly oidnReadbackFn?: OidnReadbackFn;
}
