import { Mesh, type MeshPhysicalMaterial, type Scene as ThreeScene } from 'three';
import { CIE_X_TABLE, CIE_Y_TABLE, CIE_Z_TABLE } from '@vitrum/shared-samplers';
import { rgbToSpectralCoefficients } from '@vitrum/shared-samplers';
import {
  FRAUNHOFER_C_NM,
  FRAUNHOFER_F_NM,
} from '@vitrum/shared-samplers';

interface UniformRef<T> {
  value: T;
}

interface PathTracerMaterialLike {
  uniforms?: Record<string, UniformRef<unknown>>;
}

interface DriveSourceMaterial {
  readonly ior: number;
  readonly abbe: number;
  readonly scatteringCoeff: number;
  readonly scatteringAnisotropy: number;
  readonly scatteringCoeffRgb?: readonly [number, number, number];
  readonly baseColor: readonly [number, number, number];
  readonly thinFilmStack?: {
    readonly layers?: ReadonlyArray<{ readonly ior: number; readonly thicknessNm: number }>;
  };
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

function isMeshPhysicalMaterial(m: unknown): m is MeshPhysicalMaterial {
  return (
    m != null &&
    typeof m === 'object' &&
    'isMeshPhysicalMaterial' in m &&
    (m as { isMeshPhysicalMaterial?: boolean }).isMeshPhysicalMaterial === true
  );
}

function materialSourcesFromScene(scene: ThreeScene): DriveSourceMaterial[] {
  const out: DriveSourceMaterial[] = [];
  scene.traverse((o) => {
    if (!(o instanceof Mesh)) return;
    const materials = Array.isArray(o.material) ? o.material : [o.material];
    for (const m of materials) {
      if (!isMeshPhysicalMaterial(m)) continue;
      const ud = (m.userData ?? {}) as Record<string, unknown>;
      const abbe = typeof ud['vitrumDispersionAbbeNumber'] === 'number' ? ud['vitrumDispersionAbbeNumber'] : 0;
      const scatter = typeof ud['vitrumScatteringCoefficient'] === 'number' ? ud['vitrumScatteringCoefficient'] : 0;
      const anisotropy = typeof ud['vitrumScatteringAnisotropy'] === 'number' ? ud['vitrumScatteringAnisotropy'] : 0;
      const scatterRgbRaw = ud['vitrumScatteringCoefficientRGB'];
      const scatterRgb =
        Array.isArray(scatterRgbRaw) && scatterRgbRaw.length === 3
          ? [
              Number(scatterRgbRaw[0] ?? 0),
              Number(scatterRgbRaw[1] ?? 0),
              Number(scatterRgbRaw[2] ?? 0),
            ] as const
          : undefined;
      const thinFilmStack =
        ud['vitrumThinFilmStack'] != null && typeof ud['vitrumThinFilmStack'] === 'object'
          ? (ud['vitrumThinFilmStack'] as { layers?: ReadonlyArray<{ ior: number; thicknessNm: number }> })
          : undefined;
      const src: DriveSourceMaterial = {
        ior: m.ior,
        abbe,
        scatteringCoeff: scatter,
        scatteringAnisotropy: anisotropy,
        baseColor: [m.color.r, m.color.g, m.color.b],
        ...(scatterRgb !== undefined ? { scatteringCoeffRgb: scatterRgb } : {}),
        ...(thinFilmStack !== undefined ? { thinFilmStack } : {}),
      };
      out.push(src);
    }
  });
  return out;
}

function dispersionStrengthFromAbbe(ior: number, abbe: number): number {
  if (abbe <= 0 || ior <= 1) return 0;
  const denom = 1 / (FRAUNHOFER_F_NM * FRAUNHOFER_F_NM) - 1 / (FRAUNHOFER_C_NM * FRAUNHOFER_C_NM);
  if (Math.abs(denom) < 1e-12) return 0;
  return Math.max(0, (ior - 1) / (abbe * denom));
}

/**
 * Pick a single "representative" material to drive the fork's global uniforms.
 *
 * SINGLE-MATERIAL LIMITATION: The fork's PhysicalPathTracingMaterial carries
 * scene-level (not per-instance) uniforms for scattering, dispersion, and
 * spectral coefficients. There is no per-material binding today. We therefore
 * aggregate by choosing the material with the highest combined scattering +
 * dispersion signal. Future work (RFE-09 follow-up) can upgrade this once the
 * fork supports per-material uniform arrays or a MaterialsTexture binding for
 * these fields.
 */
function dominantSource(sources: readonly DriveSourceMaterial[]): DriveSourceMaterial | null {
  let best: DriveSourceMaterial | null = null;
  let score = -Infinity;
  for (const s of sources) {
    const dispersion = dispersionStrengthFromAbbe(s.ior, s.abbe);
    const thisScore = s.scatteringCoeff + dispersion * 10;
    if (thisScore > score) {
      score = thisScore;
      best = s;
    }
  }
  return best;
}

export function driveForkMaterialUniforms(pathTracer: unknown, scene: ThreeScene): void {
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
  // RFE-09 stabilization: per-material scalar drives now come from the fork
  // MaterialsTexture packing path. The bridge only uploads global spectral tables.
}
