import type { Scene as ThreeScene } from 'three';
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
  scene: ThreeScene,
  causticOptions?: ForkBridgeCausticOptions,
): void {
  const tracer = pathTracer as { _pathTracer?: { material?: PathTracerMaterialLike } };
  const material = tracer._pathTracer?.material ?? null;
  if (material == null) return;
  void scene;
  const yCdf = buildYCdf(CIE_Y_TABLE);

  setUniform(material, 'uCmfX', CIE_X_TABLE);
  setUniform(material, 'uCmfY', CIE_Y_TABLE);
  setUniform(material, 'uCmfZ', CIE_Z_TABLE);
  setUniform(material, 'uYCmfCdf', yCdf);
  // Trapezoidal integral in table-space; shader/host treat this as relative normalization constant.
  setUniform(material, 'uYCmfIntegral', 106.857);
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
