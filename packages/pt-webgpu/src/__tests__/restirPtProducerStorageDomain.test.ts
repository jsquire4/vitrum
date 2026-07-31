import { describe, expect, it } from 'vitest';
import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';

type Vec3 = readonly [number, number, number];

function canonicalizeStoredLo(input: Vec3): Vec3 {
  const lo = input.every(Number.isFinite)
    ? input.map((value) => Math.max(0, Math.fround(value)))
    : [0, 0, 0];
  const maxChannel = Math.max(...lo);
  if (!(maxChannel > 0)) return [0, 0, 0];

  const exponent = Math.max(-114, Math.min(128, Math.floor(Math.log2(maxChannel)) + 1));
  const encodeScale = 2 ** (12 - exponent);
  const decodeScale = 2 ** (exponent - 12);
  return lo.map((value) => {
    const quantized = Math.max(0, Math.min(4095, Math.round(value * encodeScale)));
    return Math.fround(quantized * decodeScale);
  }) as [number, number, number];
}

function targetForFixedDomain(lo: Vec3): number {
  const fCos: Vec3 = [0.9, 0.35, 0.7];
  return 0.2126 * fCos[0] * lo[0] + 0.7152 * fCos[1] * lo[1] + 0.0722 * fCos[2] * lo[2];
}

function interfaceEtaTOverI(frontFace: boolean, ior: number): number {
  return frontFace ? ior : 1 / ior;
}

describe('ReSTIR-PT producer stored-Lo target domain', () => {
  it('keeps one-candidate W exactly reciprocal to the source pdf after HDR packing', () => {
    const rawLo: Vec3 = [0.49, 0.49, 2048.4];
    const storedLo = canonicalizeStoredLo(rawLo);
    const pdfSrc = 0.25;
    const rawTarget = targetForFixedDomain(rawLo);
    const storedTarget = targetForFixedDomain(storedLo);

    expect(storedLo).not.toEqual(rawLo);
    expect(canonicalizeStoredLo(storedLo)).toEqual(storedLo);
    expect(storedTarget).toBeGreaterThan(0);

    const mismatchedW = rawTarget / pdfSrc / storedTarget;
    const storedDomainW = storedTarget / pdfSrc / storedTarget;
    expect(Math.abs(mismatchedW - 1 / pdfSrc)).toBeGreaterThan(0.005);
    expect(storedDomainW).toBeCloseTo(1 / pdfSrc, 12);
  });

  it('canonicalizes once before target, weight, update, and finalization', () => {
    const seedStart = RESTIR_PT_PRODUCER_WGSL.indexOf('// Candidate target (integrand-matching:');
    const canonicalize = RESTIR_PT_PRODUCER_WGSL.indexOf(
      'Lo = rptCanonicalizeStoredLo(Lo);',
      seedStart,
    );
    const target = RESTIR_PT_PRODUCER_WGSL.indexOf(
      'let pHat = restirPtTargetForDomainAtHero(r, heroLambda, woV, xs, Lo);',
      canonicalize,
    );
    const weight = RESTIR_PT_PRODUCER_WGSL.indexOf(
      'let logCandidateWeight = log(pHat) - log(pdfSrc);',
      target,
    );
    const update = RESTIR_PT_PRODUCER_WGSL.indexOf(
      '&r, xs, ns, Lo, heroLambda, pdfSrc, logCandidateWeight, &rng,',
      weight,
    );
    const finalise = RESTIR_PT_PRODUCER_WGSL.indexOf(
      'finaliseReservoirPTWGris(&r);',
      update,
    );

    expect(seedStart).toBeGreaterThanOrEqual(0);
    expect(canonicalize).toBeGreaterThan(seedStart);
    expect(target).toBeGreaterThan(canonicalize);
    expect(weight).toBeGreaterThan(target);
    expect(update).toBeGreaterThan(weight);
    expect(finalise).toBeGreaterThan(update);
    expect(
      RESTIR_PT_PRODUCER_WGSL.slice(seedStart, finalise).match(/rptCanonicalizeStoredLo\(Lo\)/g),
    ).toHaveLength(1);
  });
});

describe('ReSTIR-PT visible-vertex interface contract', () => {
  it('uses geometric winding rather than the interpolated shading normal for side', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'let vIsFront = vHit.frontFace;',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).not.toContain(
      'let vIsFront = dot(vHit.normal, primaryRay.direction) < 0.0;',
    );
  });

  it('maps the intrinsic IOR to the incident-side ratio used by sampling and PDF', () => {
    expect(interfaceEtaTOverI(true, 1.5)).toBeCloseTo(1.5, 15);
    expect(interfaceEtaTOverI(false, 1.5)).toBeCloseTo(2 / 3, 15);

    const etaStart = RESTIR_PT_PRODUCER_WGSL.indexOf(
      'let etaTOverIV = select(',
    );
    const pdfStart = RESTIR_PT_PRODUCER_WGSL.indexOf(
      'let pdfSrc = rptSourceDirectionalPdfFull(',
      etaStart,
    );
    const pdfEnd = RESTIR_PT_PRODUCER_WGSL.indexOf('\n  );', pdfStart);
    const pdfCall = RESTIR_PT_PRODUCER_WGSL.slice(pdfStart, pdfEnd);
    expect(etaStart).toBeGreaterThanOrEqual(0);
    expect(pdfStart).toBeGreaterThan(etaStart);
    expect(pdfCall).toContain(
      'baseColorV, roughnessV, metallicV, 0.0, etaTOverIV,',
    );
    expect(pdfCall).not.toContain(
      'baseColorV, roughnessV, metallicV, 0.0, iorV,',
    );
  });

  it('names the PDF helper contract etaTOverI and forwards it unchanged', () => {
    const helperStart = RESTIR_PT_PRODUCER_WGSL.indexOf(
      'fn rptSourceDirectionalPdfFull(',
    );
    const helperEnd = RESTIR_PT_PRODUCER_WGSL.indexOf(
      '\n}\n',
      helperStart,
    );
    const helper = RESTIR_PT_PRODUCER_WGSL.slice(
      helperStart,
      helperEnd + 3,
    );
    expect(helperStart).toBeGreaterThanOrEqual(0);
    expect(helper).toContain('etaTOverI: f32,');
    expect(helper).toContain(
      'baseColor, roughness, metallic, transmission, etaTOverI,',
    );
  });
});
