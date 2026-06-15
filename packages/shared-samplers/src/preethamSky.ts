import { luminance } from './luminance.js';

export interface PreethamSkyBakeOptions {
  readonly sunDirection?: readonly [number, number, number];
  readonly turbidity?: number;
  readonly rayleigh?: number;
  readonly mieCoefficient?: number;
  readonly mieDirectionalG?: number;
  readonly intensity?: number;
  readonly width?: number;
  readonly height?: number;
}

export interface PreethamSkyBake {
  readonly texels: Float32Array;
  readonly cdf: Float32Array;
  readonly width: number;
  readonly height: number;
  readonly sunDirection: readonly [number, number, number];
}

const DEFAULT_SKY_WIDTH = 256;
const DEFAULT_SKY_HEIGHT = 128;
const SOLAR_ANGULAR_RADIUS = 0.00436;

function perez(
  cosTheta: number,
  gamma: number,
  A: number,
  B: number,
  C: number,
  D: number,
  E: number,
): number {
  const safeCos = Math.max(cosTheta, 1e-4);
  return (
    (1 + A * Math.exp(B / safeCos)) *
    (1 + C * Math.exp(D * gamma) + E * Math.cos(gamma) * Math.cos(gamma))
  );
}

function perezCoeffsY(T: number): [number, number, number, number, number] {
  return [
    0.17872 * T - 1.46303,
    -0.3554 * T + 0.42749,
    -0.02266 * T + 5.32505,
    0.12064 * T - 2.57705,
    -0.06696 * T + 0.37027,
  ];
}

function perezCoeffsX(T: number): [number, number, number, number, number] {
  return [
    -0.01925 * T - 0.25922,
    -0.06651 * T + 0.00081,
    -0.00041 * T + 0.21247,
    -0.06409 * T - 0.89887,
    -0.00325 * T + 0.04517,
  ];
}

function perezCoeffsYChroma(T: number): [number, number, number, number, number] {
  return [
    -0.01669 * T - 0.26078,
    -0.09495 * T + 0.00921,
    -0.00792 * T + 0.21023,
    -0.04405 * T - 1.65369,
    -0.01092 * T + 0.05291,
  ];
}

function zenithLuminance(T: number, chi: number): number {
  return (
    (4.0453 * T - 4.971) * Math.tan((4.0 / 9.0 - T / 120.0) * (Math.PI - 2 * chi)) -
    0.2155 * T +
    2.4192
  );
}

function zenithChromaticity(T: number, thetaSun: number): { xz: number; yz: number } {
  const t1 = T;
  const t2 = T * T;
  const ts = thetaSun;
  const ts2 = ts * ts;
  const ts3 = ts2 * ts;
  const xz =
    (0.00166 * ts3 - 0.00375 * ts2 + 0.00209 * ts) * t2 +
    (-0.02903 * ts3 + 0.06377 * ts2 - 0.03202 * ts + 0.00394) * t1 +
    (0.11693 * ts3 - 0.21196 * ts2 + 0.06052 * ts + 0.25886);
  const yz =
    (0.00275 * ts3 - 0.0061 * ts2 + 0.00317 * ts) * t2 +
    (-0.04214 * ts3 + 0.0897 * ts2 - 0.04153 * ts + 0.00516) * t1 +
    (0.15346 * ts3 - 0.26756 * ts2 + 0.0667 * ts + 0.26688);
  return { xz, yz };
}

function xyYtoLinearRGB(x: number, y: number, Y: number): [number, number, number] {
  if (y < 1e-10 || Y < 0) return [0, 0, 0];
  const X = (Y / y) * x;
  const Z = (Y / y) * (1 - x - y);
  return [
    Math.max(0, 3.2406 * X - 1.5372 * Y - 0.4986 * Z),
    Math.max(0, -0.9689 * X + 1.8758 * Y + 0.0415 * Z),
    Math.max(0, 0.0557 * X - 0.204 * Y + 1.057 * Z),
  ];
}

function henyeyGreenstein(cosGamma: number, g: number): number {
  const g2 = g * g;
  const denom = Math.pow(1 + g2 - 2 * g * cosGamma, 1.5);
  if (denom < 1e-12) return 0;
  return (1 - g2) / (4 * Math.PI * denom);
}

function normalizeSunDirection(
  input: readonly [number, number, number] | undefined,
): readonly [number, number, number] {
  const raw = input ?? [0, 1, 0];
  const len = Math.hypot(raw[0] ?? 0, raw[1] ?? 0, raw[2] ?? 0);
  return len < 1e-8 ? [0, 1, 0] : [(raw[0] ?? 0) / len, (raw[1] ?? 0) / len, (raw[2] ?? 0) / len];
}

/**
 * Bake the Preetham 1999 analytic daylight model into an equirectangular
 * float RGBA map. The `.w` channel stores pdf per steradian for consumers that
 * use direct CDF sampling; consumers with an existing HDRI CDF builder can use
 * the RGB channels and rebuild their own distribution.
 *
 * Provenance: Preetham, Shirley, Smits, "A Practical Analytic Model for
 * Daylight", SIGGRAPH 1999 (doi:10.1145/311535.311545).
 */
export function bakePreethamSkyEquirect(opts: PreethamSkyBakeOptions = {}): PreethamSkyBake {
  const width = Math.max(1, Math.floor(opts.width ?? DEFAULT_SKY_WIDTH));
  const height = Math.max(1, Math.floor(opts.height ?? DEFAULT_SKY_HEIGHT));
  const pixelCount = width * height;
  const sunDir = normalizeSunDirection(opts.sunDirection);
  const turbidity = Math.max(1.5, Math.min(30, opts.turbidity ?? 2));
  const rayleigh = Math.max(0, opts.rayleigh ?? 1);
  const mieScale = Math.max(0, opts.mieCoefficient ?? 0.005) * 200;
  const mieG = Math.max(-0.9999, Math.min(0.9999, opts.mieDirectionalG ?? 0.8));
  const intensity = Math.max(0, opts.intensity ?? 1);

  const thetaSun = Math.acos(Math.max(-1, Math.min(1, sunDir[1])));
  const [AY, BY, CY, DY, EY] = perezCoeffsY(turbidity);
  const [Ax, Bx, Cx, Dx, Ex] = perezCoeffsX(turbidity);
  const [Ay, By, Cy, Dy, Ey] = perezCoeffsYChroma(turbidity);
  const yzRaw = zenithLuminance(turbidity, thetaSun);
  const zenithY = Math.max(1e-5, yzRaw) * rayleigh * intensity;
  const { xz, yz } = zenithChromaticity(turbidity, thetaSun);
  const normY = perez(1, thetaSun, AY, BY, CY, DY, EY);
  const normX = perez(1, thetaSun, Ax, Bx, Cx, Dx, Ex);
  const normy = perez(1, thetaSun, Ay, By, Cy, Dy, Ey);
  const texels = new Float32Array(pixelCount * 4);
  const cdf = new Float32Array(pixelCount + 1);
  let totalWeight = 0;

  for (let py = 0; py < height; py += 1) {
    const theta = ((py + 0.5) / height) * Math.PI;
    const cosTheta = Math.cos(theta);
    const sinTheta = Math.sin(theta);
    const sinThetaSafe = Math.max(sinTheta, 1e-6);

    for (let px = 0; px < width; px += 1) {
      const phi = ((px + 0.5) / width) * (2 * Math.PI);
      const dx = sinTheta * Math.cos(phi);
      const dy = cosTheta;
      const dz = sinTheta * Math.sin(phi);
      const cosGamma = Math.max(-1, Math.min(1, dx * sunDir[0] + dy * sunDir[1] + dz * sunDir[2]));
      const gamma = Math.acos(cosGamma);
      let r = 0;
      let g = 0;
      let b = 0;

      if (cosTheta >= -0.05) {
        const safeCosTheta = Math.max(cosTheta, 1e-4);
        const skyY =
          normY > 1e-12 ? zenithY * (perez(safeCosTheta, gamma, AY, BY, CY, DY, EY) / normY) : 0;
        const skyx =
          normX > 1e-12 ? xz * (perez(safeCosTheta, gamma, Ax, Bx, Cx, Dx, Ex) / normX) : xz;
        const skyy =
          normy > 1e-12 ? yz * (perez(safeCosTheta, gamma, Ay, By, Cy, Dy, Ey) / normy) : yz;
        const horizonFade = Math.min(1, (cosTheta + 0.05) / 0.05);
        const skyLum = Math.max(0, skyY) * horizonFade;
        const mieContrib =
          mieScale * henyeyGreenstein(cosGamma, mieG) * 4 * Math.PI * zenithY * horizonFade;
        const sunRadiance = (gamma <= SOLAR_ANGULAR_RADIUS ? 1 : 0) * 500 * zenithY;
        const totalY = skyLum + mieContrib + sunRadiance;
        const blendX = skyLum > 1e-12 || totalY < 1e-12 ? skyx : xz;
        const blendY = skyLum > 1e-12 || totalY < 1e-12 ? skyy : yz;
        [r, g, b] = xyYtoLinearRGB(blendX, blendY, totalY);
      }

      const i = py * width + px;
      texels[i * 4] = r;
      texels[i * 4 + 1] = g;
      texels[i * 4 + 2] = b;
      const weight = Math.max(0, luminance(r, g, b) * sinThetaSafe);
      totalWeight += weight;
      cdf[i + 1] = totalWeight;
    }
  }

  if (totalWeight > 1e-12) {
    const dOmegaBase = ((2 * Math.PI) / width) * (Math.PI / height);
    for (let i = 0; i < pixelCount; i += 1) {
      cdf[i + 1] = (cdf[i + 1] ?? 0) / totalWeight;
      const py = (i / width) | 0;
      const theta = ((py + 0.5) / height) * Math.PI;
      const sinTheta = Math.max(Math.sin(theta), 1e-5);
      const pmf = Math.max((cdf[i + 1] ?? 0) - (cdf[i] ?? 0), 0);
      texels[i * 4 + 3] = pmf / (dOmegaBase * sinTheta);
    }
    cdf[0] = 0;
    cdf[pixelCount] = 1;
  }

  return { texels, cdf, width, height, sunDirection: sunDir };
}
