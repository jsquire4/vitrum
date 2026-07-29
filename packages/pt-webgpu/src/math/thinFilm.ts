import type { ThinFilmLayer } from '@vitrum/core';
import {
  CIE_D65_TABLE, CIE_LAMBDA_MIN, CIE_LAMBDA_STEP, CIE_TABLE_LENGTH,
  CIE_X_TABLE, CIE_Y_TABLE, CIE_Z_TABLE, xyzToLinearSRGB,
} from '@vitrum/shared-samplers';

type Complex = readonly [number, number];
const add = (a: Complex, b: Complex): Complex => [a[0] + b[0], a[1] + b[1]];
const sub = (a: Complex, b: Complex): Complex => [a[0] - b[0], a[1] - b[1]];
const mul = (a: Complex, b: Complex): Complex =>
  [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const scale = (a: Complex, s: number): Complex => [a[0] * s, a[1] * s];
const abs2 = (a: Complex): number => a[0] * a[0] + a[1] * a[1];
const div = (a: Complex, b: Complex): Complex => {
  const d = abs2(b);
  if (!(d > 1e-30) || !Number.isFinite(d)) return [0, 0];
  return [(a[0] * b[0] + a[1] * b[1]) / d,
    (a[1] * b[0] - a[0] * b[1]) / d];
};
const sqrtComplex = (z: Complex): Complex => {
  const radius = Math.hypot(z[0], z[1]);
  const real = Math.sqrt(Math.max(0, 0.5 * (radius + z[0])));
  const imag = Math.sqrt(Math.max(0, 0.5 * (radius - z[0])));
  return [real, z[1] < 0 ? -imag : imag];
};
const expI = (z: Complex): Complex => {
  const amplitude = Math.exp(Math.max(-80, Math.min(0, -z[1])));
  return [amplitude * Math.cos(z[0]), amplitude * Math.sin(z[0])];
};

interface Scatter {
  readonly rL: Complex; readonly tLR: Complex;
  readonly rR: Complex; readonly tRL: Complex;
}
const identity = (): Scatter => ({
  rL: [0, 0], tLR: [1, 0], rR: [0, 0], tRL: [1, 0],
});
/** Stable Redheffer star product; passive propagation factors never grow. */
function cascade(a: Scatter, b: Scatter): Scatter {
  const inv = div([1, 0], sub([1, 0], mul(a.rR, b.rL)));
  return {
    rL: add(a.rL, mul(mul(mul(a.tRL, b.rL), inv), a.tLR)),
    tLR: mul(mul(b.tLR, inv), a.tLR),
    rR: add(b.rR, mul(mul(mul(b.tLR, a.rR), inv), b.tRL)),
    tRL: mul(mul(a.tRL, inv), b.tRL),
  };
}

export interface ThinFilmStackOracleInput {
  readonly layers: readonly ThinFilmLayer[];
  readonly incidentIor: number;
  readonly substrateIor: number;
  readonly wavelengthNm: number;
  readonly cosTheta: number;
  readonly angleDependent: boolean;
  readonly reverse?: boolean;
}
export interface ThinFilmRtOracle {
  readonly reflectance: number;
  readonly transmittance: number;
  readonly absorption: number;
}
export interface ThinFilmRgbOracle {
  readonly reflectance: readonly [number, number, number];
  readonly transmittance: readonly [number, number, number];
  readonly reflectanceEnergy: number;
  readonly transmittanceEnergy: number;
  readonly absorptionEnergy: number;
}

export type ThinFilmNumericFailureReason =
  | 'non-finite-response'
  | 'non-passive-response';

/**
 * Structured production/preflight failure from the coherent TMM solver.
 *
 * RGB LUT construction runs before GPU publication, so callers can distinguish
 * an unsupported numerical stack from an ordinary scene/schema error and keep
 * the previous scene live. The old fallback silently converted either failure
 * into a perfect mirror, materially changing the authored transport.
 */
export class ThinFilmNumericError extends Error {
  readonly code = 'PT_WEBGPU_THIN_FILM_NUMERIC_FAILURE' as const;

  constructor(
    readonly reason: ThinFilmNumericFailureReason,
    readonly input: ThinFilmStackOracleInput,
    readonly response: {
      readonly reflectance: number;
      readonly transmittance: number;
    },
  ) {
    super(
      'pt-webgpu thin-film TMM numeric failure ' +
      `(${reason}; wavelength=${String(input.wavelengthNm)}nm, ` +
      `cosTheta=${String(input.cosTheta)}, ` +
      `reflectance=${String(response.reflectance)}, ` +
      `transmittance=${String(response.transmittance)})`,
    );
    this.name = 'ThinFilmNumericError';
  }
}

function physicalCosine(n: Complex, transverse: number): Complex {
  const ratio = div([transverse, 0], n);
  let cosine = sqrtComplex(sub([1, 0], mul(ratio, ratio)));
  const kz = mul(n, cosine);
  if (kz[1] < -1e-12 || (Math.abs(kz[1]) <= 1e-12 && kz[0] < 0)) {
    cosine = scale(cosine, -1);
  }
  return cosine;
}

function polarized(input: ThinFilmStackOracleInput, p: boolean): readonly [number, number] {
  const layers = input.reverse ? [...input.layers].reverse() : [...input.layers];
  const first: Complex = input.reverse ? [input.substrateIor, 0] : [input.incidentIor, 0];
  const last: Complex = input.reverse ? [input.incidentIor, 0] : [input.substrateIor, 0];
  const media: Complex[] = [
    first,
    ...layers.map((layer): Complex => [layer.ior, layer.extinctionCoefficient ?? 0]),
    last,
  ];
  const c0 = input.angleDependent ? Math.max(0, Math.min(1, input.cosTheta)) : 1;
  const transverse = first[0] * Math.sqrt(Math.max(0, 1 - c0 * c0));
  const cosines = media.map((n) => physicalCosine(n, transverse));
  const q = media.map((n, i) => p ? div(cosines[i]!, n) : mul(n, cosines[i]!));
  let network = identity();
  for (let i = 0; i < media.length - 1; i += 1) {
    const sum = add(q[i]!, q[i + 1]!);
    const rL = div(sub(q[i]!, q[i + 1]!), sum);
    network = cascade(network, {
      rL, tLR: div(scale(q[i]!, 2), sum),
      rR: scale(rL, -1), tRL: div(scale(q[i + 1]!, 2), sum),
    });
    if (i < layers.length) {
      const phase = scale(
        mul(media[i + 1]!, cosines[i + 1]!),
        2 * Math.PI * layers[i]!.thicknessNm / input.wavelengthNm,
      );
      const propagation = expI(phase);
      network = cascade(network, {
        rL: [0, 0], tLR: propagation, rR: [0, 0], tRL: propagation,
      });
    }
  }
  return [
    abs2(network.rL),
    Math.max(q.at(-1)![0], 0) / Math.max(q[0]![0], 1e-15) * abs2(network.tLR),
  ];
}

export function thinFilmRtAtWavelength(input: ThinFilmStackOracleInput): ThinFilmRtOracle {
  const [rs, ts] = polarized(input, false);
  const [rp, tp] = polarized(input, true);
  let reflectance = Math.max(0, 0.5 * (rs + rp));
  let transmittance = Math.max(0, 0.5 * (ts + tp));
  if (
    !Number.isFinite(reflectance) ||
    !Number.isFinite(transmittance) ||
    !Number.isFinite(reflectance + transmittance)
  ) {
    throw new ThinFilmNumericError(
      'non-finite-response',
      input,
      { reflectance, transmittance },
    );
  }
  const sum = reflectance + transmittance;
  if (sum > 1 + 1e-8) {
    throw new ThinFilmNumericError(
      'non-passive-response',
      input,
      { reflectance, transmittance },
    );
  }
  if (sum > 1) {
    reflectance /= sum;
    transmittance /= sum;
  }
  return { reflectance, transmittance,
    absorption: Math.max(0, 1 - reflectance - transmittance) };
}

/** 5 nm CIE 1931/D65 quadrature mirrored exactly by production WGSL. */
export function thinFilmRgb(
  input: Omit<ThinFilmStackOracleInput, 'wavelengthNm'>,
): ThinFilmRgbOracle {
  let normY = 0;
  for (let i = 0; i < CIE_TABLE_LENGTH; i += 1) {
    const endpoint = i === 0 || i === CIE_TABLE_LENGTH - 1 ? 0.5 : 1;
    normY += endpoint * CIE_D65_TABLE[i]! * CIE_Y_TABLE[i]! * CIE_LAMBDA_STEP;
  }
  let xr = 0; let yr = 0; let zr = 0; let xt = 0; let yt = 0; let zt = 0;
  for (let i = 0; i < CIE_TABLE_LENGTH; i += 1) {
    const rt = thinFilmRtAtWavelength({
      ...input, wavelengthNm: CIE_LAMBDA_MIN + i * CIE_LAMBDA_STEP,
    });
    const endpoint = i === 0 || i === CIE_TABLE_LENGTH - 1 ? 0.5 : 1;
    const w = endpoint * CIE_D65_TABLE[i]! * CIE_LAMBDA_STEP / normY;
    xr += rt.reflectance * w * CIE_X_TABLE[i]!;
    yr += rt.reflectance * w * CIE_Y_TABLE[i]!;
    zr += rt.reflectance * w * CIE_Z_TABLE[i]!;
    xt += rt.transmittance * w * CIE_X_TABLE[i]!;
    yt += rt.transmittance * w * CIE_Y_TABLE[i]!;
    zt += rt.transmittance * w * CIE_Z_TABLE[i]!;
  }
  const gamut = (v: readonly [number, number, number]) =>
    v.map((x) => Math.max(0, Math.min(1, x))) as unknown as readonly [number, number, number];
  return {
    reflectance: gamut(xyzToLinearSRGB(xr, yr, zr)),
    transmittance: gamut(xyzToLinearSRGB(xt, yt, zt)),
    reflectanceEnergy: yr, transmittanceEnergy: yt,
    absorptionEnergy: Math.max(0, 1 - yr - yt),
  };
}
