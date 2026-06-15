/**
 * reservoirPtLayout.test.ts — GRIS Phase-0 bit-identity guard for the widened
 * ReSTIR-GI reservoir (ReservoirGI → ReservoirPT).
 *
 * Phase 0 widened the per-pixel GI reservoir from 20 u32 (80 bytes) to 30 u32
 * (120 bytes) by APPENDING the GRIS reconnection-shift cache at indices
 * [20..29]. The CONTRACT is that the existing [0..19] layout is byte-identical
 * to the pre-GRIS ReservoirGI, so every current temporal/spatial/shade read of
 * the reservoir is provably unaffected and rendered output stays bit-identical
 * in Phase 0 (the new fields are written-but-unread).
 *
 * These tests pin that contract by parsing the WGSL load/store helper bodies in
 * `reservoirGi.wgsl.ts` and asserting each field's `buf[b + N]` u32 index
 * matches a FROZEN GOLDEN of the original 80-byte layout for indices [0..19].
 * The golden is transcribed from the Sprint-16 layout comment, NOT from the
 * current source, so a future reorder of the shared fields fails the test.
 *
 * No GPU: this is a pure source-string / index-mapping regression, the same
 * style as ddgiAtlasLayoutWgsl.test.ts.
 */

import { describe, it, expect } from 'vitest';
import {
  buildReservoirGiWgsl,
  RESERVOIR_GI_WGSL,
} from '../src/shaders/reservoirGi.wgsl.js';

// ── FROZEN GOLDEN — the original Sprint-16 ReservoirGI 80-byte layout ─────────
// Field name → u32 index within the reservoir. Transcribed by hand from the
// pre-GRIS layout (xv 0..2, _pad0 3, nv 4..6, W 7, xs 8..10, w_sum 11, ns 12..14,
// M 15, Lo 16..18, lightId 19). The GRIS widening MUST NOT change any of these.
const GOLDEN_SHARED_FIELD_INDEX: Record<string, number> = {
  'r.xv.x': 0, 'r.xv.y': 1, 'r.xv.z': 2,
  'r._pad0': 3,
  'r.nv.x': 4, 'r.nv.y': 5, 'r.nv.z': 6,
  'r.W': 7,
  'r.xs.x': 8, 'r.xs.y': 9, 'r.xs.z': 10,
  'r.w_sum': 11,
  'r.ns.x': 12, 'r.ns.y': 13, 'r.ns.z': 14,
  'r.M': 15,
  'r.Lo.x': 16, 'r.Lo.y': 17, 'r.Lo.z': 18,
  'r.lightId': 19,
};

/**
 * Extract a named WGSL function body from the module source.
 * Returns the text between the function's opening `{` and its matching `}`.
 */
function fnBody(src: string, fnName: string): string {
  const sig = src.indexOf(`fn ${fnName}(`);
  expect(sig, `fn ${fnName} present`).toBeGreaterThanOrEqual(0);
  const open = src.indexOf('{', sig);
  // Brace-match to find the closing brace of this function.
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

describe('GRIS Phase-0 — ReservoirPT stride widened to 30 u32 / 120 bytes', () => {
  it('declares RESERVOIR_GI_STRIDE = 30u (was 20u)', () => {
    expect(RESERVOIR_GI_WGSL).toContain('const RESERVOIR_GI_STRIDE: u32 = 30u;');
  });

  it('renames the struct to ReservoirPT and keeps a ReservoirGI alias', () => {
    expect(RESERVOIR_GI_WGSL).toContain('struct ReservoirPT {');
    expect(RESERVOIR_GI_WGSL).toContain('alias ReservoirGI = ReservoirPT;');
  });
});

describe('H24 — default reservoir layout stays compact unless GRIS is structurally enabled', () => {
  const compact = buildReservoirGiWgsl({ grisCache: false });
  const full = buildReservoirGiWgsl({ grisCache: true });

  it('compact default declares the 20-u32 Sprint-16/17 stride', () => {
    expect(compact).toContain('const RESERVOIR_GI_STRIDE: u32 = 20u;');
    expect(full).toContain('const RESERVOIR_GI_STRIDE: u32 = 30u;');
  });

  it('compact store helper writes only the shared [0..19] prefix', () => {
    const store = fnBody(compact, 'storeReservoirGI_rw');
    const writes = [...store.matchAll(/buf\[b \+ (\d+)u\]/g)].map((m) => Number(m[1]));
    expect(writes.length).toBeGreaterThan(0);
    expect(Math.max(...writes)).toBe(19);
    expect(store).toContain('Compact default layout: no appended GRIS cache stores.');
  });

  it('compact load helpers zero appended GRIS fields instead of reading beyond index 19', () => {
    for (const body of [
      fnBody(compact, 'loadReservoirGI_rw'),
      fnBody(compact, 'loadReservoirGI_ro'),
    ]) {
      const reads = [...body.matchAll(/buf\[b \+ (\d+)u\]/g)].map((m) => Number(m[1]));
      expect(Math.max(...reads)).toBe(19);
      expect(body).toContain('r.wi_recon = vec3f(0.0);');
      expect(body).toContain('r.prefixVertexCount = 0u;');
    }
  });
});

describe('GRIS Phase-0 — shared fields [0..19] are byte-identical to old ReservoirGI', () => {
  const store = fnBody(RESERVOIR_GI_WGSL, 'storeReservoirGI_rw');
  const loadRw = fnBody(RESERVOIR_GI_WGSL, 'loadReservoirGI_rw');
  const loadRo = fnBody(RESERVOIR_GI_WGSL, 'loadReservoirGI_ro');

  it('store helper writes every shared field at its golden u32 index', () => {
    // f32 fields go through bitcast<u32>(...); u32 fields (M, lightId) are raw.
    for (const [field, idx] of Object.entries(GOLDEN_SHARED_FIELD_INDEX)) {
      const isRawU32 = field === 'r.M' || field === 'r.lightId';
      const escaped = field.replace(/\./g, '\\.');
      const pat = isRawU32
        ? new RegExp(`buf\\[b \\+ ${idx}u\\]\\s*=\\s*${escaped};`)
        : new RegExp(`buf\\[b \\+ ${idx}u\\]\\s*=\\s*bitcast<u32>\\(${escaped}\\);`);
      expect(pat.test(store), `store ${field} at index ${idx}`).toBe(true);
    }
  });

  it('both load helpers read every shared field from its golden u32 index', () => {
    for (const body of [loadRw, loadRo]) {
      // vec3 components: r.xv = vec3f(bitcast<f32>(buf[b + 0u]), bitcast<f32>(buf[b + 1u]), bitcast<f32>(buf[b + 2u]))
      // Build the expected component-index map per vec3 / scalar.
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
        { lhs: 'r.lightId', indices: [19], raw: true },
      ];
      for (const { lhs, indices, raw } of checks) {
        // Each index in `indices` must appear as a read for this field's RHS.
        for (const idx of indices) {
          const inner = raw ? `buf\\[b \\+ ${idx}u\\]` : `bitcast<f32>\\(buf\\[b \\+ ${idx}u\\]\\)`;
          expect(new RegExp(inner).test(body), `${lhs} reads index ${idx}`).toBe(true);
        }
        // And the assignment LHS exists.
        const esc = lhs.replace(/\./g, '\\.');
        expect(new RegExp(`${esc}\\s*=`).test(body), `${lhs} assigned`).toBe(true);
      }
    }
  });

  it('no shared field index collides with the appended GRIS range [20..29]', () => {
    for (const idx of Object.values(GOLDEN_SHARED_FIELD_INDEX)) {
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThanOrEqual(19);
    }
  });
});

describe('GRIS Phase-0 — appended fields live strictly after index 19', () => {
  const store = fnBody(RESERVOIR_GI_WGSL, 'storeReservoirGI_rw');

  it('the GRIS reconnection-shift cache is written at indices 20..29', () => {
    // wi_recon.xyz → 20,21,22 ; pdfReconBsdf → 23 ; distRecon → 24 ;
    // cosReconOut → 25 ; prefixVertexCount → 26 ; _padPT0..2 → 27,28,29.
    expect(store).toMatch(/buf\[b \+ 20u\]\s*=\s*bitcast<u32>\(r\.wi_recon\.x\);/);
    expect(store).toMatch(/buf\[b \+ 21u\]\s*=\s*bitcast<u32>\(r\.wi_recon\.y\);/);
    expect(store).toMatch(/buf\[b \+ 22u\]\s*=\s*bitcast<u32>\(r\.wi_recon\.z\);/);
    expect(store).toMatch(/buf\[b \+ 23u\]\s*=\s*bitcast<u32>\(r\.pdfReconBsdf\);/);
    expect(store).toMatch(/buf\[b \+ 24u\]\s*=\s*bitcast<u32>\(r\.distRecon\);/);
    expect(store).toMatch(/buf\[b \+ 25u\]\s*=\s*bitcast<u32>\(r\.cosReconOut\);/);
    expect(store).toMatch(/buf\[b \+ 26u\]\s*=\s*r\.prefixVertexCount;/);
    expect(store).toMatch(/buf\[b \+ 27u\]\s*=\s*r\._padPT0;/);
    expect(store).toMatch(/buf\[b \+ 28u\]\s*=\s*r\._padPT1;/);
    expect(store).toMatch(/buf\[b \+ 29u\]\s*=\s*r\._padPT2;/);
  });

  it('no write touches an index >= 30 (stride bound)', () => {
    const writes = [...store.matchAll(/buf\[b \+ (\d+)u\]/g)].map((m) => Number(m[1]));
    expect(writes.length).toBeGreaterThan(0);
    for (const idx of writes) expect(idx).toBeLessThan(30);
  });
});

describe('GRIS Phase-0 — emptyReservoirGI matches old empty on shared fields + zeroes GRIS fields', () => {
  const empty = fnBody(RESERVOIR_GI_WGSL, 'emptyReservoirGI');

  it('shared-field initialisers are unchanged from the original empty constructor', () => {
    // Original initialisers (Sprint-16): all positions/Lo zero, normals (0,1,0),
    // W/w_sum 0, M/lightId 0u, _pad0 0.
    expect(empty).toContain('r.xv = vec3f(0.0);');
    expect(empty).toContain('r.nv = vec3f(0,1,0);');
    expect(empty).toContain('r.xs = vec3f(0.0);');
    expect(empty).toContain('r.ns = vec3f(0,1,0);');
    expect(empty).toContain('r.Lo = vec3f(0.0);');
    expect(empty).toContain('r.W = 0.0;');
    expect(empty).toContain('r.w_sum = 0.0;');
    expect(empty).toContain('r.M = 0u;');
    expect(empty).toContain('r.lightId = 0u;');
    expect(empty).toContain('r._pad0 = 0.0;');
  });

  it('every appended GRIS field is zero-initialised', () => {
    expect(empty).toContain('r.wi_recon = vec3f(0.0);');
    expect(empty).toContain('r.pdfReconBsdf = 0.0;');
    expect(empty).toContain('r.distRecon = 0.0;');
    expect(empty).toContain('r.cosReconOut = 0.0;');
    expect(empty).toContain('r.prefixVertexCount = 0u;');
    expect(empty).toContain('r._padPT0 = 0u; r._padPT1 = 0u; r._padPT2 = 0u;');
  });
});
