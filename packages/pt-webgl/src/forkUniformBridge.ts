import { CIE_X_TABLE, CIE_Y_TABLE, CIE_Z_TABLE } from '@vitrum/shared-samplers';

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

function setUniform<T>(material: PathTracerMaterialLike | null, name: string, value: T): void {
  const u = material?.uniforms?.[name] as UniformRef<T> | undefined;
  if (u != null) {
    u.value = value;
  }
}

/**
 * Wavelength-space trapezoidal integral of the CIE-Y CMF (step × Σ sample
 * pairs / 2). Computed once at module init from the same source table the
 * shader uniforms read, so the value is guaranteed to stay in sync — the
 * previously hand-baked literal `106.857` would silently fall out of sync
 * if the CIE_Y_TABLE source data ever changed.
 */
function computeCmfIntegralWavelengthSpace(
  table: Float32Array | Readonly<Float32Array>,
  stepNm: number,
): number {
  let total = 0;
  for (let i = 1; i < table.length; i += 1) {
    const v0 = table[i - 1] ?? 0;
    const v1 = table[i] ?? 0;
    total += (v0 + v1) * 0.5;
  }
  return total * stepNm;
}

const X_CMF_INTEGRAL = computeCmfIntegralWavelengthSpace(CIE_X_TABLE, 5);
const Y_CMF_INTEGRAL = computeCmfIntegralWavelengthSpace(CIE_Y_TABLE, 5);
const Z_CMF_INTEGRAL = computeCmfIntegralWavelengthSpace(CIE_Z_TABLE, 5);
// Detect silent drift in the source CMF table. The legacy hardcoded value
// was 106.857 (CIE 1931 standard observer, 380–780 nm, 5 nm step). If the
// computed integral drifts >0.05 from that, the source table changed and
// downstream normalization needs review.
if (typeof console !== 'undefined' && console.warn) {
  const CMF_INTEGRAL_DRIFTS: Array<[string, number]> = [
    ['X', X_CMF_INTEGRAL],
    ['Y', Y_CMF_INTEGRAL],
    ['Z', Z_CMF_INTEGRAL],
  ];
  for (const [name, val] of CMF_INTEGRAL_DRIFTS) {
    if (Math.abs(val - 106.857) > 0.05) {
      console.warn(
        `[forkUniformBridge] CIE-${name} integral drift: computed ${val.toFixed(3)}, ` +
          `expected ~106.857. Update host call sites that hard-code the constant.`,
      );
    }
  }
}

/**
 * buildCmfCdf — piecewise-linear normalised CDF for one CMF table. Output
 * length = `table.length + 1` (CDF[0] = 0, CDF[N] = 1). Used to populate
 * `uXCmfCdf`/`uYCmfCdf`/`uZCmfCdf` for the GLSL MIS hero-wavelength sampler.
 */
function buildCmfCdf(table: Float32Array | Readonly<Float32Array>): Float32Array {
  const cdf = new Float32Array(table.length + 1);
  cdf[0] = 0;
  let total = 0;
  for (let i = 1; i < cdf.length; i += 1) {
    const v0 = table[i - 1] ?? 0;
    const v1 = i < table.length ? (table[i] ?? 0) : 0;
    total += (v0 + v1) * 0.5;
    cdf[i] = total;
  }
  if (total > 0) {
    for (let i = 0; i < cdf.length; i += 1) {
      cdf[i] = (cdf[i] ?? 0) / total;
    }
  } else {
    cdf[cdf.length - 1] = 1;
  }
  return cdf;
}

const X_CMF_CDF_FLOAT32 = buildCmfCdf(CIE_X_TABLE);
const Y_CMF_CDF_FLOAT32 = buildCmfCdf(CIE_Y_TABLE);
const Z_CMF_CDF_FLOAT32 = buildCmfCdf(CIE_Z_TABLE);

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
  // CDFs for the GLSL MIS hero-wavelength sampler (Wilkie 2015 §3.3).
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
    setUniform(
      material,
      'uMneeMaxIterations',
      sanitizePositiveFinite(causticOptions.mneeMaxIterations, 8, 16),
    );
    setUniform(
      material,
      'uMneeMaxChainLength',
      sanitizePositiveFinite(causticOptions.mneeMaxChainLength, 3, 8),
    );
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
    const maxBounces =
      bdptOptions.maxLightBounces != null
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
