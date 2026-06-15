import type { Scene } from '@vitrum/core';
import { bakePreethamSkyEquirect, luminance } from '@vitrum/shared-samplers';

// ---------------------------------------------------------------------------
// Preetham analytic daylight model
// Ref: Preetham, Shirley, Smits, "A Practical Analytic Model for Daylight",
//      SIGGRAPH 1999. https://doi.org/10.1145/311535.311545
//
// Overview:
//   The sky luminance distribution is factored as
//     L(θ,γ) = Yz · F(θ,γ) / F(0,θs)
//   where F is the Perez sky-luminance distribution function with 5 turbidity-
//   derived coefficients A..E, θ is the zenith angle of the sample direction,
//   γ is the angle between the sample direction and the sun, and θs is the
//   solar zenith angle.  The same Perez form is applied to CIE x and y
//   chromaticities, converting the result to XYZ and then to linear-RGB.
//
// ProceduralSkyEnvironment field → Preetham mapping
//   turbidity        → turbidity T (1..∞; typical range 2..10).  Controls the
//                       A..E Perez coefficients and the zenith-luminance
//                       polynomial directly (Table 1 + Eq. 5 of the paper).
//   rayleigh         → Rayleigh scattering scale in [0..∞], default 1.  Used
//                       as a factor on the zenith luminance Yz: higher rayleigh
//                       brightens the sky and makes it bluer at low turbidities.
//                       Physically: rayleigh optical depth ∝ wavelength^-4; a
//                       unit multiplier here preserves the Preetham calibration.
//   mieCoefficient   → Mie haze scale in [0..∞], default 0.005.  Scales the
//                       Mie contribution to sky brightness around the sun via
//                       the C, E Perez coefficients (higher mieCoefficient →
//                       broader, brighter sun corona). Value 0.005 matches the
//                       default in most three.js SKY shader presets (itself
//                       derived from the Preetham formulation).
//   mieDirectionalG  → Henyey-Greenstein asymmetry g ∈ [-1, 1].  Controls the
//                       angular width of the sun aureole added on top of the
//                       Preetham sky: HG(cosγ, g) = (1−g²) / [4π (1+g²−2g cosγ)^1.5].
//                       g=0.8 gives a tight forward-scattering peak (clear sky
//                       white sun); g→1 narrows it to a delta sun disk.
//   sunDirection     → unit vector pointing toward the sun (world-space).
//                       Derivation: θs = acos(sunDirection.y).
//
// Strategy: bake to a 256×128 equirectangular float map, then feed it through
//   the existing HDRI importanace-sampling path (same CDF build, same texel
//   layout, same sun-extraction).  Resolution 256×128 was chosen so that:
//   - The pixel solid angle dΩ ≈ (2π/256)·(π/128) ≈ 1.9×10^-3 sr fits
//     ~4 sun-disk pixels (solar subtense ≈ 5.4×10^-5 sr), giving the
//     existing sun-extraction a clear maximum.
//   - CDF cost = 32 768 floats ≈ 128 kB — same order as a 128×64 HDR thumbnail;
//     rebuilding on every scene change is sub-millisecond on a modern CPU.
// ---------------------------------------------------------------------------

interface EnvironmentParams {
  readonly tint: readonly [number, number, number];
  readonly sunDirection: readonly [number, number, number];
  readonly sunStrength: number;
  /**
   * H14-E: HDRI radiance intensity multiplier, separate from `sunStrength`.
   * `scene.environment.intensity ?? 1` when a valid HDRI is present; 0 otherwise.
   * Uploaded to `params.environmentHdriIntensity` so the equirect lookup is NOT
   * gated by the procedural-sky sun strength.
   */
  readonly hdriIntensity: number;
  /**
   * H6: HDRI environment rotation around the world +Y axis, in radians (default 0).
   * Uploaded to `params.environmentTint.w` (the previously-zero .w lane of
   * environmentTint — no layout change).  The WGSL lookups apply rotateYNeg by this
   * value before UV indexing, and rotateYPos after CDF sampling.
   * Non-HDRI environments keep this at 0 (rotationY does not apply to procedural sky).
   */
  readonly hdriRotationY: number;
  readonly hdriWidth: number;
  readonly hdriHeight: number;
  readonly hasHdri: boolean;
  readonly hdriTexels: Float32Array;
  readonly hdriCdf: Float32Array;
  readonly warnings: readonly string[];
}

/** Empty / no-environment slot — neutral tint, no HDRI sampling.
 *  File-local — only consumed by `environmentParams()` below; 2026-05-18
 *  dead-code sweep verified zero external consumers. */
function emptyEnvironmentParams(): EnvironmentParams {
  return {
    tint: [1, 1, 1],
    sunDirection: [0, 1, 0],
    sunStrength: 0,
    hdriIntensity: 0,
    hdriRotationY: 0,
    hdriWidth: 0,
    hdriHeight: 0,
    hasHdri: false,
    hdriTexels: new Float32Array(0),
    hdriCdf: new Float32Array(0),
    warnings: [],
  };
}

/**
 * Build environment params from a procedural-sky scene environment using the
 * Preetham 1999 analytic daylight model.
 *
 * Strategy: bake the model into a 256×128 equirectangular float map and route
 * it through the existing HDRI importance-sampling path (same CDF build, same
 * texel layout, same sun-extraction).  This way the shader sees a real
 * physically-derived sky with full importance sampling — no WGSL changes needed.
 *
 * Provenance: Preetham, Shirley, Smits, "A Practical Analytic Model for
 * Daylight", SIGGRAPH 1999 (doi:10.1145/311535.311545).  See the
 * shared-samplers' `bakePreethamSkyEquirect` helper for the field→model mapping and
 * per-field citations.
 */
// File-local — only consumed by `environmentParams()` below; 2026-05-18
// dead-code sweep verified zero external consumers.
function buildProceduralSkyEnvironmentParams(
  env: Extract<Scene['environment'], { kind: 'procedural-sky' }>,
): EnvironmentParams {
  // Defensive defaults: the TypeScript type marks all fields required, but hosts
  // (especially @ts-nocheck scripts) may pass a partial object with only `kind`.
  // Math.max/min with `undefined` produces NaN (not the clamped value), so every
  // Preetham output would be NaN → GPU buffer full of NaN → black output with
  // zero GPU errors.  Apply per-field defaults here matching the core contract's
  // documented defaults so a bare `{ kind: 'procedural-sky' }` renders correctly.
  const rawD = env.sunDirection ?? [0, 1, 0];
  const len = Math.hypot(rawD[0] ?? 0, rawD[1] ?? 0, rawD[2] ?? 0);
  const sunDir: readonly [number, number, number] =
    len < 1e-8 ? [0, 1, 0] : [(rawD[0] ?? 0) / len, (rawD[1] ?? 0) / len, (rawD[2] ?? 0) / len];
  const intensity = env.intensity ?? 1;

  const { texels, cdf, width, height } = bakePreethamSkyEquirect({
    sunDirection: sunDir,
    turbidity: env.turbidity ?? 2,
    rayleigh: env.rayleigh ?? 1,
    mieCoefficient: env.mieCoefficient ?? 0.005,
    mieDirectionalG: env.mieDirectionalG ?? 0.8,
    intensity,
  });

  return {
    // Tint is white: the radiance is already baked into the equirect texels.
    tint: [1, 1, 1],
    // sunDirection / sunStrength: the procedural-sky sun gate (environmentSun.w)
    // is NOT needed here because we route through the HDRI path (hasHdri=true).
    // Set sunStrength=0 so the sky NEE branch doesn't double-fire alongside the
    // HDRI CDF sampling (the baked map includes the sun disk in the CDF).
    sunDirection: sunDir,
    sunStrength: 0,
    // H14-E: hdriIntensity is always 1 here — intensity was already folded into
    // the baked radiance values by bakePreethamSkyEquirect.  Multiplying again would
    // double-apply it.  The HDRI gate (hasHdri=true) picks this up correctly.
    hdriIntensity: 1,
    // rotationY does not apply to procedural sky (the sun direction is world-space).
    hdriRotationY: 0,
    hdriWidth: width,
    hdriHeight: height,
    hasHdri: true,
    hdriTexels: texels,
    hdriCdf: cdf,
    warnings: [],
  };
}

export function environmentParams(scene: Scene): EnvironmentParams {
  if (scene.environment.kind === 'none') {
    return emptyEnvironmentParams();
  }
  if (scene.environment.kind === 'procedural-sky') {
    return buildProceduralSkyEnvironmentParams(scene.environment);
  }
  type HdriLike = { width?: number; height?: number; data?: ArrayLike<number> };
  const hdri = scene.environment.hdri as HdriLike;
  const width = Number(hdri?.width ?? 0);
  const height = Number(hdri?.height ?? 0);
  const data = hdri?.data;
  const pixelCount = width * height;
  const hasRgb =
    pixelCount > 0 &&
    data != null &&
    typeof data.length === 'number' &&
    data.length >= pixelCount * 3;
  if (Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0 && hasRgb) {
    // Detect whether the caller provided RGBA (stride 4) or RGB (stride 3) data.
    // A w·h·4-length buffer passes the >= w·h·3 gate but must be decoded at
    // stride 4 or every pixel after the first will read from the wrong offset.
    const isRgba = data.length >= pixelCount * 4;
    const stride = isRgba ? 4 : 3;
    // Warn once on ambiguous / unexpected RGBA input so the host is aware of the
    // implicit stride detection (there is no authoritative flag in the scene API).
    const warnings: string[] = [];
    if (isRgba) {
      warnings.push(
        '[vitrum/pt-webgpu] HDRI data.length matches a w×h×4 RGBA layout; ' +
          'decoding at stride 4. Pass a w×h×3 RGB array to suppress this warning.',
      );
    }
    const texels = new Float32Array(pixelCount * 4);
    const cdf = new Float32Array(pixelCount + 1);
    let totalWeight = 0;
    for (let i = 0; i < pixelCount; i += 1) {
      const r = Number(data[i * stride] ?? 0);
      const g = Number(data[i * stride + 1] ?? 0);
      const b = Number(data[i * stride + 2] ?? 0);
      texels[i * 4] = r;
      texels[i * 4 + 1] = g;
      texels[i * 4 + 2] = b;
      const y = (i / width) | 0;
      const theta = ((y + 0.5) / height) * Math.PI;
      const weight = Math.max(0, luminance(r, g, b) * Math.sin(theta));
      totalWeight += weight;
      cdf[i + 1] = totalWeight;
    }
    if (totalWeight > 1e-12) {
      const dOmegaBase = ((2 * Math.PI) / width) * (Math.PI / height);
      for (let i = 0; i < pixelCount; i += 1) {
        cdf[i + 1] = (cdf[i + 1] ?? 0) / totalWeight;
        const y = (i / width) | 0;
        const theta = ((y + 0.5) / height) * Math.PI;
        const sinTheta = Math.max(Math.sin(theta), 1e-5);
        const pmf = Math.max((cdf[i + 1] ?? 0) - (cdf[i] ?? 0), 0);
        texels[i * 4 + 3] = pmf / (dOmegaBase * sinTheta);
      }
      cdf[0] = 0;
      cdf[pixelCount] = 1;
      const hdriIntensity = scene.environment.intensity ?? 1;
      // H6: pass rotationY through; default 0 (identity rotation).
      const hdriRotationY = scene.environment.rotationY ?? 0;
      return {
        tint: [1, 1, 1],
        sunDirection: [0, 1, 0],
        // H14-E: sunStrength drives the procedural-sky sun gate (environmentSun.w).
        // For a pure-HDRI scene, set sun.w to 0 so the sky NEE branch doesn't fire.
        // The HDRI radiance uses its own hdriIntensity lane (→ params.environmentHdriIntensity).
        sunStrength: 0,
        hdriIntensity,
        hdriRotationY,
        hdriWidth: width,
        hdriHeight: height,
        hasHdri: true,
        hdriTexels: texels,
        hdriCdf: cdf,
        warnings,
      };
    }
    // All pixels are black (totalWeight ≤ 1e-12) — the HDRI data was valid but
    // has zero luminance. Report accurately rather than misattributing this to
    // missing pixel data.
    return {
      tint: [1, 1, 1],
      sunDirection: [0, 1, 0],
      sunStrength: 0,
      hdriIntensity: 0,
      hdriRotationY: 0,
      hdriWidth: 0,
      hdriHeight: 0,
      hasHdri: false,
      hdriTexels: new Float32Array(0),
      hdriCdf: new Float32Array(0),
      warnings: [
        ...warnings,
        'HDRI environment has zero total luminance (all-black or transparent pixels); using black no-environment fallback.',
      ],
    };
  }
  return {
    tint: [1, 1, 1],
    sunDirection: [0, 1, 0],
    sunStrength: 0,
    hdriIntensity: 0,
    hdriRotationY: 0,
    hdriWidth: 0,
    hdriHeight: 0,
    hasHdri: false,
    hdriTexels: new Float32Array(0),
    hdriCdf: new Float32Array(0),
    warnings: [
      'HDRI environment lacks CPU pixel data (width/height/data); using black no-environment fallback.',
    ],
  };
}
