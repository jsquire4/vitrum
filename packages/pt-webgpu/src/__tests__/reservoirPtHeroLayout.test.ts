import { describe, expect, it } from 'vitest';
import {
  RESERVOIR_PT_HERO_WGSL,
  RESTIR_PT_PARAMS_WGSL,
} from '../wgsl/pathTrace/reservoirPtHero.wgsl.js';

function fnBody(src: string, fnName: string): string {
  const sig = src.indexOf(`fn ${fnName}(`);
  expect(sig, `fn ${fnName} present`).toBeGreaterThanOrEqual(0);
  const open = src.indexOf('{', sig);
  let depth = 0;
  let i = open;
  for (; i < src.length; i++) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}' && --depth === 0) break;
  }
  return src.slice(open + 1, i);
}

interface PackedLoMeta {
  readonly word0: number;
  readonly word1: number;
}

function packLoMetaCpu(
  loInput: readonly [number, number, number],
  heroLambdaInput: number,
  isFrontFace: boolean,
  mInput: number,
): PackedLoMeta {
  const finite = loInput.every((v) => Number.isFinite(v));
  const lo = finite
    ? loInput.map((v) => Math.max(0, v)) as [number, number, number]
    : [0, 0, 0] as [number, number, number];
  const maxChannel = Math.max(...lo);
  let exponentCode = 0;
  let q: [number, number, number] = [0, 0, 0];
  if (maxChannel > 0) {
    const exponent = Math.max(-114, Math.min(128, Math.floor(Math.log2(maxChannel)) + 1));
    exponentCode = exponent + 127;
    const scale = 2 ** (12 - exponent);
    q = lo.map((v) => Math.max(0, Math.min(4095, Math.round(v * scale)))) as [
      number, number, number,
    ];
  }
  const heroLambda = Number.isFinite(heroLambdaInput) ? heroLambdaInput : 550;
  const lambdaQ = Math.round(
    Math.max(0, Math.min(1, (heroLambda - 380) / 400)) * 32767,
  );
  const m = Math.max(0, Math.min(4095, Math.trunc(mInput)));
  const word0 = (
    (q[0] & 0xfff) |
    ((q[1] & 0xfff) << 12) |
    ((q[2] & 0xff) << 24)
  ) >>> 0;
  const word1 = (
    ((q[2] >>> 8) & 0xf) |
    ((exponentCode & 0xff) << 4) |
    ((lambdaQ & 0x7fff) << 12) |
    ((isFrontFace ? 1 : 0) << 27) |
    (((m >>> 8) & 0xf) << 28)
  ) >>> 0;
  return { word0, word1 };
}

function unpackLoMetaCpu(
  { word0, word1 }: PackedLoMeta,
  surfaceWord: number,
): {
  readonly lo: readonly [number, number, number];
  readonly heroLambda: number;
  readonly isFrontFace: boolean;
  readonly m: number;
} {
  const q: [number, number, number] = [
    word0 & 0xfff,
    (word0 >>> 12) & 0xfff,
    ((word0 >>> 24) & 0xff) | ((word1 & 0xf) << 8),
  ];
  const exponentCode = (word1 >>> 4) & 0xff;
  const scale = exponentCode === 0 ? 0 : 2 ** ((exponentCode - 127) - 12);
  const lambdaQ = (word1 >>> 12) & 0x7fff;
  return {
    lo: [q[0] * scale, q[1] * scale, q[2] * scale],
    heroLambda: 380 + 400 * lambdaQ / 32767,
    isFrontFace: ((word1 >>> 27) & 1) !== 0,
    m: (((word1 >>> 28) & 0xf) << 8) | ((surfaceWord >>> 24) & 0xff),
  };
}

function packMeshBaryCpu(v: number, w: number): number {
  const qv = Math.round(Math.max(0, Math.min(1, v)) * 4095);
  const qw = Math.round(Math.max(0, Math.min(1, w)) * 4095);
  return (qv & 0xfff) | ((qw & 0xfff) << 12);
}

function unpackMeshBaryCpu(packed: number): readonly [number, number] {
  let v = (packed & 0xfff) / 4095;
  let w = ((packed >>> 12) & 0xfff) / 4095;
  const sum = v + w;
  if (sum > 1) {
    v /= sum;
    w /= sum;
  }
  return [v, w];
}

describe('ReSTIR-PT compact hero reservoir ABI', () => {
  it('is exactly 16 u32 / 64 bytes and writes each word once', () => {
    expect(RESERVOIR_PT_HERO_WGSL).toContain(
      'const RESERVOIR_PT_HERO_STRIDE: u32 = 16u;',
    );
    const store = fnBody(RESERVOIR_PT_HERO_WGSL, 'storeReservoirPTHero_rw');
    const writes = [...store.matchAll(/buf\[b \+ (\d+)u\]\s*=/g)]
      .map((match) => Number(match[1]));
    expect(writes).toHaveLength(16);
    expect([...new Set(writes)].sort((a, b) => a - b)).toEqual(
      Array.from({ length: 16 }, (_, index) => index),
    );
  });

  it('both load paths read the same 16-word ABI', () => {
    for (const name of ['loadReservoirPTHero_ro', 'loadReservoirPTHero_rw']) {
      const body = fnBody(RESERVOIR_PT_HERO_WGSL, name);
      for (let index = 0; index < 16; index++) {
        expect(body, `${name} reads word ${index}`).toContain(`buf[b + ${index}u]`);
      }
    }
  });

  it('pins compression, rehydration, saturation, and corrupt-identity guards', () => {
    for (const token of [
      'pack2x16snorm(octEncode',
      'const RPT_MAX_STORED_M: u32 = 4095u;',
      'fn rptPackLoMeta(',
      'fn rptCanonicalizeStoredLo(',
      'fn rptVisibleMaterialAtSurface(',
      'fn rptHydrateVisibleDomain(',
      'fn rptVisibleIdentityIsValid(',
      'return emptyReservoirPTHero();',
      'r.pdfSrc = 1.0;',
    ]) {
      expect(RESERVOIR_PT_HERO_WGSL).toContain(token);
    }
  });

  it('keeps the reuse unit tunables in its 32-byte UBO', () => {
    expect(RESTIR_PT_PARAMS_WGSL).toContain('struct RestirPtParams {');
    expect(RESTIR_PT_PARAMS_WGSL).toContain('mClamp:   u32,');
    expect(RESTIR_PT_PARAMS_WGSL).toContain('wCap:     f32,');
  });
});

describe('ReSTIR-PT compact HDR/meta packing oracle', () => {
  it.each([
    ['zero', [0, 0, 0] as const],
    ['subnormal', [Number.MIN_VALUE, Number.MIN_VALUE, Number.MIN_VALUE] as const],
    ['very dark', [2 ** -120, 2 ** -121, 0] as const],
    ['ordinary HDR', [0.125, 12.5, 1000] as const],
    ['max finite f32', [3.402823466e38, 1.7014117e38, 0] as const],
    ['NaN rejected', [Number.NaN, 1, 2] as const],
  ])('%s stays finite and non-negative', (_label, lo) => {
    const packed = packLoMetaCpu(lo, 510, true, 4095);
    const decoded = unpackLoMetaCpu(packed, 0xff00_0000);
    for (const channel of decoded.lo) {
      expect(Number.isFinite(channel)).toBe(true);
      expect(channel).toBeGreaterThanOrEqual(0);
    }
    expect(decoded.isFrontFace).toBe(true);
    expect(decoded.m).toBe(4095);
  });

  it('round-trips representable HDR within one shared-exponent quantization step', () => {
    for (const lo of [
      [1e-20, 2e-20, 4e-20],
      [0.125, 12.5, 1000],
      [1e10, 2e10, 3e10],
      [3.402823466e38, 1.7014117e38, 0],
    ] as const) {
      const decoded = unpackLoMetaCpu(packLoMetaCpu(lo, 510, false, 20), 20 << 24);
      const maxChannel = Math.max(...lo);
      for (let channel = 0; channel < 3; channel++) {
        const normalizedAbsoluteError =
          Math.abs(decoded.lo[channel]! - lo[channel]!) / maxChannel;
        expect(normalizedAbsoluteError).toBeLessThanOrEqual(1 / 2047);
      }
    }
  });

  it('sanitizes nonfinite wavelength and accurately preserves finite wavelength', () => {
    const invalid = unpackLoMetaCpu(packLoMetaCpu([1, 1, 1], Number.NaN, false, 1), 1 << 24);
    expect(invalid.heroLambda).toBeCloseTo(550, 2);
    const finite = unpackLoMetaCpu(packLoMetaCpu([1, 1, 1], 612.345, false, 1), 1 << 24);
    expect(finite.heroLambda).toBeCloseTo(612.345, 2);
  });
});

describe('ReSTIR-PT compact barycentric packing oracle', () => {
  it.each([
    [0, 0],
    [1, 0],
    [0, 1],
    [0.5, 0.5],
    [1 / 3, 1 / 3],
    [0.9999, 0.0001],
  ])('keeps boundary barycentric (%s,%s) on the simplex', (v, w) => {
    const decoded = unpackMeshBaryCpu(packMeshBaryCpu(v, w));
    expect(decoded[0]).toBeGreaterThanOrEqual(0);
    expect(decoded[1]).toBeGreaterThanOrEqual(0);
    expect(decoded[0] + decoded[1]).toBeLessThanOrEqual(1);
  });

  it('keeps deterministic random valid barycentrics on the simplex', () => {
    let state = 0x12345678;
    const random = (): number => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 0x1_0000_0000;
    };
    for (let i = 0; i < 2048; i++) {
      let v = random();
      let w = random();
      if (v + w > 1) {
        v = 1 - v;
        w = 1 - w;
      }
      const decoded = unpackMeshBaryCpu(packMeshBaryCpu(v, w));
      expect(decoded[0] + decoded[1]).toBeLessThanOrEqual(1);
      expect(Math.abs(decoded[0] - v)).toBeLessThanOrEqual(1 / 2047);
      expect(Math.abs(decoded[1] - w)).toBeLessThanOrEqual(1 / 2047);
    }
  });
});

describe('ReSTIR-PT empty reservoir', () => {
  const empty = fnBody(RESERVOIR_PT_HERO_WGSL, 'emptyReservoirPTHero');

  it('initializes all persisted identity and metadata sentinels', () => {
    for (const token of [
      'r.W = 0.0;',
      'r.w_sum = 0.0;',
      'r.M = 0u;',
      'r.pdfSrc = 0.0;',
      'r.heroLambdaV = 550.0;',
      'r.isFrontFaceV = true;',
      'r.instanceIndexV = INVALID_TLAS_INSTANCE_INDEX;',
      'r.triangleIndexV = 0xffffffffu;',
    ]) {
      expect(empty).toContain(token);
    }
  });

  it('does not synthesize an unstored prefix gate and empties invalid one-edge paths', () => {
    expect(RESERVOIR_PT_HERO_WGSL).not.toContain('prefixVertexCount');
    const refresh = fnBody(RESERVOIR_PT_HERO_WGSL, 'refreshReconnectionStatePT');
    expect(refresh).toContain('!rptFinitePositive(dRecon) || dRecon <= 1e-6');
    expect(refresh).toContain('(*r).M = 0u;');
    expect(refresh).toContain('(*r).W = 0.0;');
    expect(refresh).toContain('(*r).w_sum = 0.0;');
  });
});
