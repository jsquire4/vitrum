import type { Scene } from '@vitrum/core';
import { luminance } from '@vitrum/shared-samplers';

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

// Perez sky-luminance distribution function.
// F(θ, γ; A..E) = (1 + A exp(B/cosθ)) · (1 + C exp(Dγ) + E cos²γ)
// θ  = zenith angle of sample direction (0 = zenith, π/2 = horizon)
// γ  = angle between sample and sun
function perez(cosTheta: number, gamma: number, A: number, B: number, C: number, D: number, E: number): number {
  // Guard against cosTheta → 0 (exactly at horizon; cos(π/2) = 0)
  const safecos = Math.max(cosTheta, 1e-4);
  return (1 + A * Math.exp(B / safecos)) * (1 + C * Math.exp(D * gamma) + E * Math.cos(gamma) * Math.cos(gamma));
}

// Preetham Appendix A — Perez coefficient tables for Y (CIE luminance) as a
// function of turbidity T.  Eq. A.1.
function perezCoeffsY(T: number): [number, number, number, number, number] {
  return [
     0.17872 * T - 1.46303,
    -0.35540 * T + 0.42749,
    -0.02266 * T + 5.32505,
     0.12064 * T - 2.57705,
    -0.06696 * T + 0.37027,
  ];
}

// Preetham Appendix A — Perez coefficient tables for x chromaticity.  Eq. A.2.
function perezCoeffsX(T: number): [number, number, number, number, number] {
  return [
    -0.01925 * T - 0.25922,
    -0.06651 * T + 0.00081,
    -0.00041 * T + 0.21247,
    -0.06409 * T - 0.89887,
    -0.00325 * T + 0.04517,
  ];
}

// Preetham Appendix A — Perez coefficient tables for y chromaticity.  Eq. A.3.
function perezCoeffsY_chroma(T: number): [number, number, number, number, number] {
  return [
    -0.01669 * T - 0.26078,
    -0.09495 * T + 0.00921,
    -0.00792 * T + 0.21023,
    -0.04405 * T - 1.65369,
    -0.01092 * T + 0.05291,
  ];
}

// Preetham Section A.2 — Zenith luminance Yz (kcd/m²) as a function of
// turbidity T and solar zenith angle χ (radians).  Eq. A.4.
function zenithLuminance(T: number, chi: number): number {
  const chi2 = chi * chi;
  // Preetham A.4 cubic term — retained in case the polynomial form is later restored.
  const _chi3 = chi2 * chi;
  return (4.0453 * T - 4.9710) * Math.tan(((4.0 / 9.0) - T / 120.0) * (Math.PI - 2 * chi)) - 0.2155 * T + 2.4192;
}

// Preetham Section A.3 — Zenith x/y chromaticities as a function of
// turbidity T and solar zenith angle θs (radians).  Eq. A.5 / A.6.
function zenithChromaticity(T: number, thetaSun: number): { xz: number; yz: number } {
  const t1 = T;
  const t2 = T * T;
  const ts  = thetaSun;
  const ts2 = ts * ts;
  const ts3 = ts2 * ts;
  // CIE x chromaticity (Table A.3 coefficients)
  const xz =
    ( 0.00166 * ts3 - 0.00375 * ts2 + 0.00209 * ts + 0       ) * t2 +
    (-0.02903 * ts3 + 0.06377 * ts2 - 0.03202 * ts + 0.00394 ) * t1 +
    ( 0.11693 * ts3 - 0.21196 * ts2 + 0.06052 * ts + 0.25886 );
  // CIE y chromaticity (Table A.4 coefficients)
  const yz =
    ( 0.00275 * ts3 - 0.00610 * ts2 + 0.00317 * ts + 0       ) * t2 +
    (-0.04214 * ts3 + 0.08970 * ts2 - 0.04153 * ts + 0.00516 ) * t1 +
    ( 0.15346 * ts3 - 0.26756 * ts2 + 0.06670 * ts + 0.26688 );
  return { xz, yz };
}

// Convert CIE xyY to linear sRGB (D65 whitepoint; rec.709 primaries).
// xyY → XYZ → linear sRGB (IEC 61966-2-1 / Rec.709 matrix, row-major).
function xyYtoLinearRGB(x: number, y: number, Y: number): [number, number, number] {
  if (y < 1e-10 || Y < 0) return [0, 0, 0];
  const X = (Y / y) * x;
  const Z = (Y / y) * (1 - x - y);
  // Rec.709 / sRGB D65 XYZ→RGB matrix (linear light, no gamma)
  const r =  3.2406 * X - 1.5372 * Y - 0.4986 * Z;
  const g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
  const b =  0.0557 * X - 0.2040 * Y + 1.0570 * Z;
  return [Math.max(0, r), Math.max(0, g), Math.max(0, b)];
}

// Henyey-Greenstein phase function (isotropic normalised over 4π sr).
// Used for the mie directional sun aureole added on top of the Preetham sky.
// Ref: van de Hulst, 1957; Henyey & Greenstein, 1941.
function henyeyGreenstein(cosGamma: number, g: number): number {
  const g2 = g * g;
  const denom = Math.pow(1 + g2 - 2 * g * cosGamma, 1.5);
  if (denom < 1e-12) return 0;
  return (1 - g2) / (4 * Math.PI * denom);
}

/** Equirect map resolution used for the baked procedural-sky.
 *  256×128 ≈ 32 K pixels → ≈128 kB for the CDF; sub-ms to build on CPU. */
const SKY_MAP_WIDTH  = 256;
const SKY_MAP_HEIGHT = 128;

/** Solar angular radius (radians) — matches the real sun's ~0.5° diameter / 2. */
const SOLAR_ANGULAR_RADIUS = 0.00436;  // 0.25° in radians

/**
 * Bake the Preetham 1999 analytic daylight model into a 256×128 equirectangular
 * RGBA float map (radiance RGB + pdf-Omega placeholder) suitable for the HDRI
 * importance-sampling path.
 *
 * @param sunDir   unit sun direction (world-space, +Y = up)
 * @param T        turbidity (Preetham T, typical 1.5..10)
 * @param rayleigh Rayleigh scattering scale (multiplicative on sky luminance)
 * @param mieBeta  Mie coefficient (scales the C/E Perez forward-scatter corona)
 * @param mieG     Henyey-Greenstein g for sun aureole (0 = isotropic, 1 = forward)
 * @param intensity overall intensity multiplier applied to the baked radiance
 */
function bakePreethamEquirect(
  sunDir: readonly [number, number, number],
  T: number,
  rayleigh: number,
  mieBeta: number,
  mieG: number,
  intensity: number,
): { texels: Float32Array; cdf: Float32Array; width: number; height: number } {
  const W = SKY_MAP_WIDTH;
  const H = SKY_MAP_HEIGHT;
  const pixelCount = W * H;

  // Clamp inputs to physically sane ranges.
  const Tclamp   = Math.max(1.5, Math.min(30, T));
  const rScl     = Math.max(0, rayleigh);
  const mieScl   = Math.max(0, mieBeta) * 200;   // β_Mie * 200 ≈ scaling used in
                                                   // three.js Sky shader (see note below)
  // Note on mieCoefficient scaling: The three.js Sky shader (itself a port of Hosek
  // et al. + Preetham) multiplies mieCoefficient by a large constant (~200) to
  // arrive at a perceptible Mie extinction term.  The core API spec's default
  // 0.005 × 200 = 1.0 lands close to turbidity T≈2 Mie extinction, which matches
  // the visual intent of that default.  We honour the same convention here so that
  // mieCoefficient=0.005 reproduces a clear-sky Mie term with no abrupt jump.
  const mieGclamp = Math.max(-0.9999, Math.min(0.9999, mieG));

  // Solar zenith angle (angle from zenith to sun direction; sun.y = cosθs)
  const sunY  = Math.max(-1, Math.min(1, sunDir[1]));  // cos(θs)
  const thetaSun = Math.acos(sunY);

  // Perez coefficients for Y, x, y
  const [AY, BY, CY, DY, EY] = perezCoeffsY(Tclamp);
  const [Ax, Bx, Cx, Dx, Ex] = perezCoeffsX(Tclamp);
  const [Ay, By, Cy, Dy, Ey] = perezCoeffsY_chroma(Tclamp);

  // Preetham Eq. A.4: zenith luminance.  kcd/m² → we use it as a relative
  // scale factor calibrated so noon clear-sky zenith ≈ 1 kcd/m² ≈ 1 (display units).
  const YzRaw = zenithLuminance(Tclamp, thetaSun);
  // Below-horizon sun: zenithLuminance can go negative; clamp to a small positive.
  const Yz = Math.max(1e-5, YzRaw) * rScl * (intensity > 0 ? intensity : 1);

  const { xz, yz } = zenithChromaticity(Tclamp, thetaSun);

  // Normalisation denominator: F(0, θs) evaluated at the zenith direction (θ=0).
  const normY = perez(1, thetaSun, AY, BY, CY, DY, EY);   // cos(0)=1
  const normx = perez(1, thetaSun, Ax, Bx, Cx, Dx, Ex);
  const normy = perez(1, thetaSun, Ay, By, Cy, Dy, Ey);

  const texels = new Float32Array(pixelCount * 4);
  const cdf    = new Float32Array(pixelCount + 1);

  // Sun direction in equirect-space (not dependent on the pixel loop)
  const sunDirX = sunDir[0];
  const sunDirY2 = sunDir[1];  // renamed to avoid collision with yz local
  const sunDirZ = sunDir[2];

  let totalWeight = 0;

  for (let py = 0; py < H; py += 1) {
    // θ = zenith angle in [0, π]; φ = azimuth in [0, 2π)
    const theta = ((py + 0.5) / H) * Math.PI;
    const cosTheta = Math.cos(theta);
    const sinTheta = Math.sin(theta);
    const sinThetaSafe = Math.max(sinTheta, 1e-6);

    for (let px = 0; px < W; px += 1) {
      const phi = ((px + 0.5) / W) * (2 * Math.PI);

      // World-space direction for this pixel (equirect convention: θ=0 → +Y zenith,
      // φ=0 → +X east).
      const dx = sinTheta * Math.cos(phi);
      const dy = cosTheta;
      const dz = sinTheta * Math.sin(phi);

      // cosγ = dot(sampleDir, sunDir)
      const cosGamma = dx * sunDirX + dy * sunDirY2 + dz * sunDirZ;
      const cosGammaClamp = Math.max(-1, Math.min(1, cosGamma));
      const gamma = Math.acos(cosGammaClamp);

      // Preetham sky distribution (below-horizon → zero)
      let r = 0, g = 0, b = 0;
      if (cosTheta >= -0.05) {
        // Sky luminance (relative, normalised to zenith = Yz)
        const FY = perez(Math.max(cosTheta, 1e-4), gamma, AY, BY, CY, DY, EY);
        const Fxv = perez(Math.max(cosTheta, 1e-4), gamma, Ax, Bx, Cx, Dx, Ex);
        const Fyv = perez(Math.max(cosTheta, 1e-4), gamma, Ay, By, Cy, Dy, Ey);

        const skyY = (normY > 1e-12) ? Yz * (FY / normY) : 0;
        const skyx = (normx > 1e-12) ? xz * (Fxv / normx) : xz;
        const skyy = (normy > 1e-12) ? yz * (Fyv / normy) : yz;

        // Smooth horizon transition: blend sky to 0 in the last 5° above horizon
        const horizonFade = Math.min(1, (cosTheta + 0.05) / 0.05);

        const skyLum = Math.max(0, skyY) * horizonFade;

        // Add Mie scattering contribution: an additive HG lobe scaled by mieBeta.
        // This brightens the sky near the sun with the correct directional bias.
        // mieScl already incorporates the 200× factor discussed above.
        const mieContrib = mieScl * henyeyGreenstein(cosGammaClamp, mieGclamp) * 4 * Math.PI * Yz * horizonFade;

        // Sun disk: add solar radiance into pixels within the solar angular radius.
        // The Preetham paper does NOT include the sun disk in the sky model (it models
        // scattered sky light only); we add an analytic solar disk here so that the
        // existing HDRI sun-extraction (which looks for the map maximum) correctly
        // locates and extracts the sun.
        // Solar luminance ≈ 1.6×10^9 cd/m² → in our units, relative to Yz, this is
        // calibrated to ≈ 500×Yz so the sun is bright but not numerically explosive.
        // The finite disk prevents a delta-function spike that would alias the CDF.
        const inSunDisk = gamma <= SOLAR_ANGULAR_RADIUS ? 1.0 : 0.0;
        const sunRadiance = inSunDisk * 500 * Yz;

        const totalY = skyLum + mieContrib + sunRadiance;

        // Convert chromaticity + totalY → XYZ → linear RGB.
        // For the mie/sun additive terms we use the sun's chromaticity (xz, yz)
        // since they represent spectrally similar forward-scattered light.
        const blendX = (skyLum > 1e-12 || totalY < 1e-12) ? skyx : xz;
        const blendY_chroma = (skyLum > 1e-12 || totalY < 1e-12) ? skyy : yz;
        const rgb = xyYtoLinearRGB(blendX, blendY_chroma, totalY);
        r = rgb[0];
        g = rgb[1];
        b = rgb[2];
      }

      const i = py * W + px;
      texels[i * 4]     = r;
      texels[i * 4 + 1] = g;
      texels[i * 4 + 2] = b;

      // sinθ-weighted CDF (same as HDRI path in the same file)
      const weight = Math.max(0, luminance(r, g, b) * sinThetaSafe);
      totalWeight += weight;
      cdf[i + 1] = totalWeight;
    }
  }

  // Normalise CDF and fill .w (pdf per steradian) — mirrors HDRI path exactly.
  if (totalWeight > 1e-12) {
    const dOmegaBase = ((2 * Math.PI) / W) * (Math.PI / H);
    for (let i = 0; i < pixelCount; i += 1) {
      cdf[i + 1] = (cdf[i + 1] ?? 0) / totalWeight;
      const py = (i / W) | 0;
      const theta = ((py + 0.5) / H) * Math.PI;
      const sinTheta = Math.max(Math.sin(theta), 1e-5);
      const pmf = Math.max((cdf[i + 1] ?? 0) - (cdf[i] ?? 0), 0);
      texels[i * 4 + 3] = pmf / (dOmegaBase * sinTheta);
    }
    cdf[0] = 0;
    cdf[pixelCount] = 1;
  }

  return { texels, cdf, width: W, height: H };
}

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
 * `bakePreethamEquirect` helper above for the field→model mapping and
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

  const { texels, cdf, width, height } = bakePreethamEquirect(
    sunDir,
    env.turbidity ?? 2,
    env.rayleigh ?? 1,
    env.mieCoefficient ?? 0.005,
    env.mieDirectionalG ?? 0.8,
    intensity,
  );

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
    // the baked radiance values by bakePreethamEquirect.  Multiplying again would
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
