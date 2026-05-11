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
function computeYIntegralWavelengthSpace(
  y: Float32Array | Readonly<Float32Array>,
  stepNm: number,
): number {
  let total = 0;
  for (let i = 1; i < y.length; i += 1) {
    const y0 = y[i - 1] ?? 0;
    const y1 = y[i] ?? 0;
    total += (y0 + y1) * 0.5;
  }
  return total * stepNm;
}

const Y_CMF_INTEGRAL = computeYIntegralWavelengthSpace(CIE_Y_TABLE, 5);
// Detect silent drift in the source CMF table. The legacy hardcoded value
// was 106.857 (CIE 1931 standard observer, 380–780 nm, 5 nm step). If the
// computed integral drifts >0.05 from that, the source table changed and
// downstream normalization needs review.
if (typeof console !== 'undefined' && console.warn) {
  if (Math.abs(Y_CMF_INTEGRAL - 106.857) > 0.05) {
    console.warn(
      `[forkUniformBridge] CIE-Y integral drift: computed ${Y_CMF_INTEGRAL.toFixed(3)}, ` +
        `expected ~106.857. Update host call sites that hard-code the constant.`,
    );
  }
}

function buildYCdf(y: Float32Array): Float32Array {
  const cdf = new Float32Array(y.length + 1);
  cdf[0] = 0;
  let total = 0;
  for (let i = 1; i < cdf.length; i += 1) {
    const y0 = y[i - 1] ?? 0;
    const y1 = i < y.length ? (y[i] ?? 0) : 0;
    total += (y0 + y1) * 0.5;
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

export function driveForkMaterialUniforms(
  pathTracer: unknown,
  causticOptions?: ForkBridgeCausticOptions,
): void {
  const tracer = pathTracer as { _pathTracer?: { material?: PathTracerMaterialLike } };
  const material = tracer._pathTracer?.material ?? null;
  if (material == null) return;
  const yCdf = buildYCdf(CIE_Y_TABLE);

  setUniform(material, 'uCmfX', CIE_X_TABLE);
  setUniform(material, 'uCmfY', CIE_Y_TABLE);
  setUniform(material, 'uCmfZ', CIE_Z_TABLE);
  setUniform(material, 'uYCmfCdf', yCdf);
  // Trapezoidal integral in table-space; shader/host treat this as relative
  // normalization constant. Computed at module init from CIE_Y_TABLE so the
  // value stays in sync with the source data automatically.
  setUniform(material, 'uYCmfIntegral', Y_CMF_INTEGRAL);
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
  // RFE-09 stabilization: per-material scalar drives now come from the fork
  // MaterialsTexture packing path. The bridge only uploads global spectral tables.
}
