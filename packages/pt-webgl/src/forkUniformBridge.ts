import { Mesh, type MeshPhysicalMaterial, type Scene as ThreeScene, type Vector3Tuple } from 'three';
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

  const sources = materialSourcesFromScene(scene);
  const source = dominantSource(sources);
  const yCdf = buildYCdf(CIE_Y_TABLE);

  setUniform(material, 'uCmfX', CIE_X_TABLE);
  setUniform(material, 'uCmfY', CIE_Y_TABLE);
  setUniform(material, 'uCmfZ', CIE_Z_TABLE);
  setUniform(material, 'uYCmfCdf', yCdf);
  // Trapezoidal integral in table-space; shader/host treat this as relative normalization constant.
  setUniform(material, 'uYCmfIntegral', 106.857);

  if (source == null) {
    setUniform(material, 'u_volumeDensity', 0);
    setUniform(material, 'u_sssSigmaT', 0);
    setUniform(material, 'u_anisotropyG', 0);
    setUniform(material, 'u_scatterAlbedo', [0.8, 0.85, 0.9] as Vector3Tuple);
    setUniform(material, 'u_sssAlbedo', [0.9, 0.9, 0.9] as Vector3Tuple);
    setUniform(material, 'u_sssAnisotropyG', 0);
    setUniform(material, 'u_ior0', 1.5);
    setUniform(material, 'u_dispersionStrength', 0);
    setUniform(material, 'u_jakobCoeffs', [0, 0, 0] as Vector3Tuple);
    setUniform(material, 'iorCauchyA', 1.5);
    setUniform(material, 'iorCauchyB', 0);
    setUniform(material, 'iorCauchyC', 0);
    setUniform(material, 'uThinFilmEnabled', 0);
    setUniform(material, 'uThinFilmLayerCount', 0);
    setUniform(material, 'uThinFilmLayerIors', new Float32Array(35));
    setUniform(material, 'uThinFilmLayerThicknessNm', new Float32Array(35));
    // TODO: gated on RFE-13 — spectral Beer-Lambert per-λ attenuation upload.
    // Once RFE-13 (ray-payload restructure) lands, upload vitrumSpectralAttenuation
    // curve values here to drive the Beer-Lambert extinction path in the shader.
    return;
  }

  const dispersionStrength = dispersionStrengthFromAbbe(source.ior, source.abbe);
  const [c0, c1, c2] = rgbToSpectralCoefficients(
    source.baseColor[0],
    source.baseColor[1],
    source.baseColor[2],
  );

  setUniform(material, 'u_volumeDensity', source.scatteringCoeff);
  setUniform(material, 'u_sssSigmaT', source.scatteringCoeff);
  setUniform(material, 'u_anisotropyG', source.scatteringAnisotropy);
  setUniform(material, 'u_scatterAlbedo', (source.scatteringCoeffRgb ?? [0.9, 0.9, 0.9]) as Vector3Tuple);
  // u_sssAlbedo = single-scatter albedo (σ_s / σ_t per channel); u_sssAnisotropyG mirrors u_anisotropyG.
  setUniform(material, 'u_sssAlbedo', (source.scatteringCoeffRgb ?? [0.9, 0.9, 0.9]) as Vector3Tuple);
  setUniform(material, 'u_sssAnisotropyG', source.scatteringAnisotropy);
  setUniform(material, 'u_ior0', source.ior);
  setUniform(material, 'u_dispersionStrength', dispersionStrength);
  setUniform(material, 'u_jakobCoeffs', [c0, c1, c2] as Vector3Tuple);
  setUniform(material, 'iorCauchyA', source.ior);
  setUniform(material, 'iorCauchyB', dispersionStrength * 1e-6);
  setUniform(material, 'iorCauchyC', 0);

  const layerIors = new Float32Array(35);
  const layerThicknessNm = new Float32Array(35);
  let layerCount = 0;
  const layers = source.thinFilmStack?.layers ?? [];
  for (let i = 0; i < Math.min(35, layers.length); i += 1) {
    const layer = layers[i];
    if (layer == null) continue;
    layerIors[i] = layer.ior;
    layerThicknessNm[i] = layer.thicknessNm;
    layerCount += 1;
  }
  setUniform(material, 'uThinFilmEnabled', layerCount > 0 ? 1 : 0);
  setUniform(material, 'uThinFilmLayerCount', layerCount);
  setUniform(material, 'uThinFilmLayerIors', layerIors);
  setUniform(material, 'uThinFilmLayerThicknessNm', layerThicknessNm);

  // TODO: gated on RFE-13 — spectral Beer-Lambert per-λ attenuation upload.
  // Once RFE-13 (ray-payload restructure) lands, read source.spectralAttenuation
  // (vitrumSpectralAttenuation userData key) and upload it to the fork's
  // spectral extinction uniform so the Beer-Lambert path activates per wavelength.
}
