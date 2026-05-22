import {
  CIE_X_TABLE,
  CIE_Y_TABLE,
  CIE_Z_TABLE,
  X_CMF_CDF,
  Y_CMF_CDF,
  Z_CMF_CDF,
  X_CMF_INTEGRAL,
  Y_CMF_INTEGRAL,
  Z_CMF_INTEGRAL,
} from '@vitrum/shared-samplers';

interface UniformRef<T> {
  value: T;
}

interface PathTracerMaterialLike {
  uniforms?: Record<string, UniformRef<unknown>>;
}

export interface ForkBridgeCausticOptions {
  readonly strategy: 'none' | 'manifold-nee' | 'photon-map';
  readonly mneeMaxIterations: number;
  readonly mneeMaxChainLength: number;
  readonly spectralRendering?: boolean;
  readonly radianceClamp?: number;
}

/**
 * BDPT options for Sprint 10c — bidirectional path tracing for caustic-heavy
 * scenes (stainedGlass, crystal, underwater).
 *
 * Pass this in addition to {@link ForkBridgeCausticOptions} to enable BDPT
 * in the PT_FINAL accumulation mode. BDPT adds explicit eye↔light vertex
 * connections via a ping-pong light-subpath texture.
 *
 * - `enabled` maps to the fork's `uBdptEnabled` uniform + `FEATURE_BDPT` define.
 *   When false (default) the connection GLSL is compiled out — zero overhead.
 * - `maxLightBounces` controls how many stored light vertices to attempt connections
 *   with (1–3; default 3 = BDPT_MAX_LIGHT_BOUNCES constant in fork).
 *   Reducing to 1 cuts per-sample shadow-ray cost by ~2/3 at the expense of
 *   caustic depth beyond the first emitter bounce.
 * - `lightPathTex` is the RGBA32F texture the host writes via the light-subpath
 *   draw pass. If null, BDPT is automatically disabled as a safety guard.
 *
 * References: Veach 1997 §10.3; sprint-10c-pt-fork-patch.md.
 */
export interface ForkBridgeBdptOptions {
  /** Enable BDPT integrator path (default false). */
  readonly enabled: boolean;
  /**
   * Number of light-subpath bounces to store and connect (1–3; default 3).
   * Must match the light-subpath draw pass loop count in the host renderer.
   */
  readonly maxLightBounces?: number;
  /**
   * Light-subpath ping-pong texture (RGBA32F, width=maxLightBounces, height=3).
   * Written by the host's light-subpath draw pass before the main accumulation loop.
   * null → BDPT disabled even if enabled=true (safety guard for uninitialized hosts).
   */
  readonly lightPathTex?: unknown | null;
}

function sanitizePositiveFinite(value: number, fallback: number, max: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(1, Math.min(max, Math.floor(value)));
}

const missingUniformWarnings = new Set<string>();

function setUniform<T>(material: PathTracerMaterialLike | null, name: string, value: T): void {
  const u = material?.uniforms?.[name] as UniformRef<T> | undefined;
  if (u != null) {
    u.value = value;
    return;
  }
  if (!missingUniformWarnings.has(name)) {
    missingUniformWarnings.add(name);
    console.warn(
      `@vitrum/pt-webgl: fork uniform "${name}" missing on path tracer material; check fork/bridge compatibility.`,
    );
  }
}

/**
 * Convert one of the canonical Float64Array CMF CDFs from
 * `@vitrum/shared-samplers` into a Float32Array suitable for GLSL uniform
 * upload (WebGL2 uniform arrays require Float32). The shared package owns the
 * trapezoidal-rule integral + normalised-CDF construction (see
 * `wavelengthSampling.ts::buildIntegralAndCdf`) — this bridge only narrows
 * the precision for shader use.
 *
 * W2-C14 dedup: the previous local `computeCmfIntegralWavelengthSpace` +
 * `buildCmfCdf` reimplementations were mathematically equivalent to the
 * shared-samplers versions (both forms of pair-sum trapezoidal rule resolve
 * to `step·(0.5·f[0] + f[1] + … + f[N-2] + 0.5·f[N-1])`) but maintained in
 * two places, drifting silently if either side updated boundary handling.
 */
function cdfToFloat32(cdf: Readonly<Float64Array>): Float32Array {
  const out = new Float32Array(cdf.length);
  for (let i = 0; i < cdf.length; i += 1) {
    out[i] = cdf[i] ?? 0;
  }
  return out;
}

const X_CMF_CDF_FLOAT32 = cdfToFloat32(X_CMF_CDF);
const Y_CMF_CDF_FLOAT32 = cdfToFloat32(Y_CMF_CDF);
const Z_CMF_CDF_FLOAT32 = cdfToFloat32(Z_CMF_CDF);

export function driveForkMaterialUniforms(
  pathTracer: unknown,
  causticOptions?: ForkBridgeCausticOptions,
  bdptOptions?: ForkBridgeBdptOptions,
): void {
  const tracer = pathTracer as { _pathTracer?: { material?: PathTracerMaterialLike } };
  const material = tracer._pathTracer?.material ?? null;
  if (material == null) return;

  setUniform(material, 'uCmfX', CIE_X_TABLE);
  setUniform(material, 'uCmfY', CIE_Y_TABLE);
  setUniform(material, 'uCmfZ', CIE_Z_TABLE);
  // CDFs for the GLSL MIS hero-wavelength sampler (Wilkie 2014 §3.3).
  // Y-only sampling collapses blue/red to near-zero at low SPP because Y(λ)
  // is heavily concentrated near 555 nm; uploading X and Z CDFs lets the
  // shader pick from any of three strategies with balance-heuristic mixture pdf.
  setUniform(material, 'uXCmfCdf', X_CMF_CDF_FLOAT32);
  setUniform(material, 'uYCmfCdf', Y_CMF_CDF_FLOAT32);
  setUniform(material, 'uZCmfCdf', Z_CMF_CDF_FLOAT32);
  // Trapezoidal integrals in table-space; shader/host treat these as relative
  // normalization constants. Computed at module init from CIE_X/Y/Z_TABLE so
  // the values stay in sync with the source data automatically.
  setUniform(material, 'uXCmfIntegral', X_CMF_INTEGRAL);
  setUniform(material, 'uYCmfIntegral', Y_CMF_INTEGRAL);
  setUniform(material, 'uZCmfIntegral', Z_CMF_INTEGRAL);
  setUniform(material, 'uSpectralRendering', causticOptions?.spectralRendering === true ? 1 : 0);
  setUniform(material, 'uRadianceClamp', causticOptions?.radianceClamp ?? 0);
  if (causticOptions != null) {
    const strategyCode =
      causticOptions.strategy === 'manifold-nee'
        ? 1
        : causticOptions.strategy === 'photon-map'
          ? 2
          : 0;
    setUniform(material, 'uCausticStrategy', strategyCode);
    setUniform(material, 'uMneeMaxIterations', sanitizePositiveFinite(causticOptions.mneeMaxIterations, 8, 16));
    setUniform(material, 'uMneeMaxChainLength', sanitizePositiveFinite(causticOptions.mneeMaxChainLength, 3, 8));
  }

  // Sprint 10c — BDPT uniform bridge.
  // Threads the host-side BDPT options to the fork's uBdptEnabled / uBdptMaxLightBounces /
  // uBdptLightPathTex uniforms. The fork's FEATURE_BDPT define is synced automatically via
  // onBeforeRender() from uBdptEnabled.
  //
  // Safety guard: if lightPathTex is null (uninitialized host), force enabled=false to
  // prevent the shader from sampling an unbound texture slot.
  if (bdptOptions != null) {
    const lightPathTex = bdptOptions.lightPathTex ?? null;
    const effectivelyEnabled = bdptOptions.enabled && lightPathTex != null;
    setUniform(material, 'uBdptEnabled', effectivelyEnabled);
    setUniform(material, 'uBdptLightPathTex', lightPathTex);
    const maxBounces = bdptOptions.maxLightBounces != null
      ? sanitizePositiveFinite(bdptOptions.maxLightBounces, 3, 3)
      : 3;
    setUniform(material, 'uBdptMaxLightBounces', maxBounces);
  } else {
    // BDPT not requested — ensure it's off (idempotent; safe to call every frame).
    setUniform(material, 'uBdptEnabled', false);
  }

  // RFE-09 stabilization: per-material scalar drives now come from the fork
  // MaterialsTexture packing path. The bridge only uploads global spectral tables.
}
