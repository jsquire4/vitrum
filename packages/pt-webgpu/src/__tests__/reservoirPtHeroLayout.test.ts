/**
 * reservoirPtHeroLayout.test.ts — byte-identity guard for the hero ReSTIR-PT /
 * GRIS reservoir (ReservoirPTHero), mirroring walkaround-hybrid's
 * reservoirPtLayout.test.ts (the proven GI reservoir byte-golden style).
 *
 * The reservoir is serialised strided into `array<u32>` via bitcast<u32>(f32) /
 * raw-u32. The CONTRACT this test pins is that EACH field maps to a FROZEN GOLDEN
 * u32 index in the load/store helpers — so a future reorder of the fields (which
 * would silently corrupt every produced/reused reservoir, since the producer,
 * temporal and resolve passes all share this serialization) fails the test. The
 * golden is transcribed BY HAND from the layout comment, NOT read from the
 * current source, so a drift is caught.
 *
 * No GPU: pure source-string / index-mapping regression, same style as
 * walkaround-hybrid/reservoirPtLayout.test.ts and ddgiAtlasLayoutWgsl.test.ts.
 */

import { describe, it, expect } from 'vitest';
import {
  RESERVOIR_PT_HERO_WGSL,
  RESTIR_PT_PARAMS_WGSL,
} from '../wgsl/pathTrace/reservoirPtHero.wgsl.js';

// ── FROZEN GOLDEN — the ReservoirPTHero 192-byte (48 u32) layout ─────────────
// field → u32 index. Transcribed by hand from the layout comment. A reorder MUST
// fail. f32 fields go through bitcast<u32>(); u32 fields (M, prefixVertexCount,
// rngSeed, _padHybrid) are raw.
const GOLDEN_FIELD_INDEX: Record<string, number> = {
  'r.xv.x': 0, 'r.xv.y': 1, 'r.xv.z': 2,
  'r._pad0': 3,
  'r.nv.x': 4, 'r.nv.y': 5, 'r.nv.z': 6,
  'r.W': 7,
  'r.xs.x': 8, 'r.xs.y': 9, 'r.xs.z': 10,
  'r.w_sum': 11,
  'r.ns.x': 12, 'r.ns.y': 13, 'r.ns.z': 14,
  'r.M': 15,
  'r.Lo.x': 16, 'r.Lo.y': 17, 'r.Lo.z': 18,
  'r.pdfSrc': 19,
  'r.wi_recon.x': 20, 'r.wi_recon.y': 21, 'r.wi_recon.z': 22,
  'r.distRecon': 23,
  'r.cosReconOut': 24,
  'r.prefixVertexCount': 25,
  'r.roughnessV': 26,
  'r.metalV': 27,
  'r.albV.x': 28, 'r.albV.y': 29, 'r.albV.z': 30,
  'r.clearcoatV': 31,
  'r.clearcoatRoughnessV': 32,
  'r.sheenV': 33,
  'r.sheenRoughnessV': 34,
  'r.sheenColorV.x': 35, 'r.sheenColorV.y': 36, 'r.sheenColorV.z': 37,
  'r.iridescenceV': 38,
  'r.iridescenceIorV': 39,
  'r.iridescenceThicknessMinV': 40,
  'r.iridescenceThicknessMaxV': 41,
  'r.anisotropyV': 42,
  'r.anisotropyRotationV': 43,
  'r.hybridJacCache': 44,
  'r.hybridShiftPdf': 45,
  'r.rngSeed': 46,
  'r._padHybrid': 47,
};

const RAW_U32_FIELDS = new Set(['r.M', 'r.prefixVertexCount', 'r.rngSeed', 'r._padHybrid']);

/** Extract a named WGSL function body (text between the fn's `{` and its `}`). */
function fnBody(src: string, fnName: string): string {
  const sig = src.indexOf(`fn ${fnName}(`);
  expect(sig, `fn ${fnName} present`).toBeGreaterThanOrEqual(0);
  const open = src.indexOf('{', sig);
  let depth = 0;
  let i = open;
  for (; i < src.length; i++) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}') {
      depth--;
      if (depth === 0) break;
    }
  }
  return src.slice(open + 1, i);
}

describe('ReSTIR-PT hero reservoir — ReservoirPTHero stride = 48 u32 / 192 bytes', () => {
  it('declares RESERVOIR_PT_HERO_STRIDE = 48u', () => {
    expect(RESERVOIR_PT_HERO_WGSL).toContain('const RESERVOIR_PT_HERO_STRIDE: u32 = 48u;');
  });

  it('declares the ReservoirPTHero struct', () => {
    expect(RESERVOIR_PT_HERO_WGSL).toContain('struct ReservoirPTHero {');
  });

  it('declares the RestirPtParams UBO (the reuse unit owns its tunables)', () => {
    expect(RESTIR_PT_PARAMS_WGSL).toContain('struct RestirPtParams {');
    expect(RESTIR_PT_PARAMS_WGSL).toContain('wCap:     f32,');
    expect(RESTIR_PT_PARAMS_WGSL).toContain('mClamp:   u32,');
  });
});

describe('ReSTIR-PT hero reservoir — every field at its golden u32 index', () => {
  const store = fnBody(RESERVOIR_PT_HERO_WGSL, 'storeReservoirPTHero_rw');
  const loadRw = fnBody(RESERVOIR_PT_HERO_WGSL, 'loadReservoirPTHero_rw');
  const loadRo = fnBody(RESERVOIR_PT_HERO_WGSL, 'loadReservoirPTHero_ro');

  it('store helper writes every field at its golden u32 index', () => {
    for (const [field, idx] of Object.entries(GOLDEN_FIELD_INDEX)) {
      const isRawU32 = RAW_U32_FIELDS.has(field);
      const escaped = field.replace(/\./g, '\\.');
      const pat = isRawU32
        ? new RegExp(`buf\\[b \\+ ${idx}u\\]\\s*=\\s*${escaped};`)
        : new RegExp(`buf\\[b \\+ ${idx}u\\]\\s*=\\s*bitcast<u32>\\(${escaped}\\);`);
      expect(pat.test(store), `store ${field} at index ${idx}`).toBe(true);
    }
  });

  it('both load helpers read every field from its golden u32 index', () => {
    const checks: { lhs: string; indices: number[]; raw: boolean }[] = [
      { lhs: 'r.xv', indices: [0, 1, 2], raw: false },
      { lhs: 'r._pad0', indices: [3], raw: false },
      { lhs: 'r.nv', indices: [4, 5, 6], raw: false },
      { lhs: 'r.W', indices: [7], raw: false },
      { lhs: 'r.xs', indices: [8, 9, 10], raw: false },
      { lhs: 'r.w_sum', indices: [11], raw: false },
      { lhs: 'r.ns', indices: [12, 13, 14], raw: false },
      { lhs: 'r.M', indices: [15], raw: true },
      { lhs: 'r.Lo', indices: [16, 17, 18], raw: false },
      { lhs: 'r.pdfSrc', indices: [19], raw: false },
      { lhs: 'r.wi_recon', indices: [20, 21, 22], raw: false },
      { lhs: 'r.distRecon', indices: [23], raw: false },
      { lhs: 'r.cosReconOut', indices: [24], raw: false },
      { lhs: 'r.prefixVertexCount', indices: [25], raw: true },
      { lhs: 'r.roughnessV', indices: [26], raw: false },
      { lhs: 'r.metalV', indices: [27], raw: false },
      { lhs: 'r.albV', indices: [28, 29, 30], raw: false },
      { lhs: 'r.clearcoatV', indices: [31], raw: false },
      { lhs: 'r.clearcoatRoughnessV', indices: [32], raw: false },
      { lhs: 'r.sheenV', indices: [33], raw: false },
      { lhs: 'r.sheenRoughnessV', indices: [34], raw: false },
      { lhs: 'r.sheenColorV', indices: [35, 36, 37], raw: false },
      { lhs: 'r.iridescenceV', indices: [38], raw: false },
      { lhs: 'r.iridescenceIorV', indices: [39], raw: false },
      { lhs: 'r.iridescenceThicknessMinV', indices: [40], raw: false },
      { lhs: 'r.iridescenceThicknessMaxV', indices: [41], raw: false },
      { lhs: 'r.anisotropyV', indices: [42], raw: false },
      { lhs: 'r.anisotropyRotationV', indices: [43], raw: false },
      { lhs: 'r.hybridJacCache', indices: [44], raw: false },
      { lhs: 'r.hybridShiftPdf', indices: [45], raw: false },
      { lhs: 'r.rngSeed', indices: [46], raw: true },
      { lhs: 'r._padHybrid', indices: [47], raw: true },
    ];
    for (const body of [loadRw, loadRo]) {
      for (const { lhs, indices, raw } of checks) {
        for (const idx of indices) {
          const inner = raw ? `buf\\[b \\+ ${idx}u\\]` : `bitcast<f32>\\(buf\\[b \\+ ${idx}u\\]\\)`;
          expect(new RegExp(inner).test(body), `${lhs} reads index ${idx}`).toBe(true);
        }
        const esc = lhs.replace(/\./g, '\\.');
        expect(new RegExp(`${esc}\\s*=`).test(body), `${lhs} assigned`).toBe(true);
      }
    }
  });

  it('no write touches an index >= 48 (stride bound)', () => {
    const writes = [...store.matchAll(/buf\[b \+ (\d+)u\]/g)].map((m) => Number(m[1]));
    expect(writes.length).toBeGreaterThan(0);
    for (const idx of writes) expect(idx).toBeLessThan(48);
  });

  it('writes all 48 indices [0..47] exactly once', () => {
    const writes = [...store.matchAll(/buf\[b \+ (\d+)u\]/g)].map((m) => Number(m[1]));
    const seen = new Set(writes);
    expect(seen.size).toBe(48);
    for (let i = 0; i < 48; i++) expect(seen.has(i), `index ${i} written`).toBe(true);
  });
});

describe('ReSTIR-PT hero reservoir — empty constructor zeroes every field', () => {
  const empty = fnBody(RESERVOIR_PT_HERO_WGSL, 'emptyReservoirPTHero');

  it('zero/identity-initialises the sample fields', () => {
    expect(empty).toContain('r.xv = vec3f(0.0);');
    expect(empty).toContain('r.nv = vec3f(0,1,0);');
    expect(empty).toContain('r.xs = vec3f(0.0);');
    expect(empty).toContain('r.ns = vec3f(0,1,0);');
    expect(empty).toContain('r.Lo = vec3f(0.0);');
    expect(empty).toContain('r.W = 0.0;');
    expect(empty).toContain('r.w_sum = 0.0;');
    expect(empty).toContain('r.M = 0u;');
    expect(empty).toContain('r.pdfSrc = 0.0;');
  });

  it('zero-initialises the reconnection-shift cache + visible-vertex material', () => {
    expect(empty).toContain('r.wi_recon = vec3f(0.0);');
    expect(empty).toContain('r.distRecon = 0.0;');
    expect(empty).toContain('r.cosReconOut = 0.0;');
    expect(empty).toContain('r.prefixVertexCount = 0u;');
    expect(empty).toContain('r.roughnessV = 0.0;');
    expect(empty).toContain('r.metalV = 0.0;');
    expect(empty).toContain('r.albV = vec3f(0.0);');
    expect(empty).toContain('r.clearcoatV = 0.0;');
    expect(empty).toContain('r.sheenV = 0.0;');
    expect(empty).toContain('r.sheenColorV = vec3f(0.0);');
    expect(empty).toContain('r.iridescenceIorV = 1.3;');
    expect(empty).toContain('r.anisotropyV = 0.0;');
  });

  it('zero-initialises the Phase-0 (written-but-unread) hybrid + rngSeed headroom', () => {
    expect(empty).toContain('r.hybridJacCache = 0.0;');
    expect(empty).toContain('r.hybridShiftPdf = 0.0;');
    expect(empty).toContain('r.rngSeed = 0u;');
    expect(empty).toContain('r._padHybrid = 0u;');
  });
});
