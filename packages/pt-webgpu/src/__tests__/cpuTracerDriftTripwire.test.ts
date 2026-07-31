/**
 * cpuTracerDriftTripwire.test.ts — §H H55-b
 *
 * Lightweight drift-tripwire for the cpuTracer.ts mirrors.
 *
 * cpuTracer.ts is a test-only CPU reference tracer whose functions mirror
 * specific WGSL functions in the pt-webgpu shaders.  If a WGSL function is
 * edited without updating the mirror, the two can silently diverge, making
 * the cpuTracer tests no longer verify what they claim to.
 *
 * This test pins each WGSL function body by:
 *   1. Extracting the named `fn <name>(…)` body from the composed WGSL string
 *      (from the `{` to the matching closing `}`).
 *   2. Computing a SHA-256 of the normalized body (whitespace-collapsed, so
 *      trivial reformatting doesn't false-alarm).
 *   3. Asserting the hash matches the pinned value.
 *
 * When a WGSL function is intentionally changed, the test fails with a clear
 * message telling the editor to update the cpuTracer mirror AND repin the hash.
 *
 * SCOPE: this is a LIGHTWEIGHT tripwire, not a full correctness oracle.
 * The energy-conservation / numerics / adjoint tests provide the correctness
 * oracle.  A real-GPU reference comparison is out of scope (no GPU in CI).
 *
 * Mirrored functions and their source locations:
 *
 *   WGSL function            Source                                  cpuTracer mirror
 *   ──────────────────────   ──────────────────────────────────────  ────────────────────────────────────
 *   safe_normalize           common.wgsl.ts:42                       safeNormalize (cpuTracer.ts:116)
 *   intersectAabb            pathTrace/intersectionCore.wgsl.ts:22   intersectAabb (cpuTracer.ts:182)
 *   cosineHemisphereSample   pathTrace/bsdf.wgsl.ts:116              cosineHemisphereSample (cpuTracer.ts:232)
 *   sampleGgxVndfTangent     pathTrace/bsdf.wgsl.ts:140              sampleGgxVndfTangent (cpuTracer.ts:258)
 *   fresnelSchlick           pathTrace/material.wgsl.ts:499          fresnelSchlick (cpuTracer.ts:305)
 *   frDielectric             pathTrace/material.wgsl.ts:513          frDielectric (cpuTracer.ts:288)
 *   powerHeuristic           pathTrace/material.wgsl.ts:542          powerHeuristic (cpuTracer.ts:316)
 */

import { describe, it, expect } from 'vitest';
import { createHash } from 'node:crypto';
import { PT_WEBGPU_COMMON_WGSL } from '../wgsl/common.wgsl.js';
import { PT_WEBGPU_INTERSECTION_CORE_WGSL } from '../wgsl/pathTrace/intersectionCore.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_BSDF_WGSL } from '../wgsl/pathTrace/bsdf.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL } from '../wgsl/pathTrace/material.wgsl.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Extract the body of a named WGSL function from a WGSL string.
 *
 * Handles the pattern:
 *   fn <name>(<params…>) [-> <ret>] {
 *     …body…
 *   }
 *
 * Uses a simple brace-depth counter rather than a full WGSL parser — sufficient
 * for well-formed WGSL, which is what we generate.
 *
 * Returns the raw body text (from the opening `{` to the closing `}`, inclusive).
 */
function extractFnBody(wgsl: string, fnName: string): string {
  // Match the function header.  Allow multi-line params and optional return type.
  const headerRe = new RegExp(`fn\\s+${fnName}\\s*\\(`);
  const headerMatch = headerRe.exec(wgsl);
  if (headerMatch == null) {
    throw new Error(`[cpuTracerDriftTripwire] fn ${fnName} not found in the provided WGSL`);
  }
  // Scan forward from the header start to find the opening brace.
  let i = headerMatch.index + headerMatch[0].length;
  let depth = 0;
  let openBrace = -1;
  while (i < wgsl.length) {
    if (wgsl[i] === '{') {
      if (openBrace === -1) openBrace = i;
      depth++;
    } else if (wgsl[i] === '}') {
      depth--;
      if (depth === 0) {
        return wgsl.slice(openBrace, i + 1);
      }
    }
    i++;
  }
  throw new Error(`[cpuTracerDriftTripwire] unmatched brace for fn ${fnName}`);
}

/** Normalise a WGSL body for a stable hash: collapse all whitespace to single spaces. */
function normalise(body: string): string {
  return body.replace(/\s+/g, ' ').trim();
}

/** SHA-256 of the normalised body. */
function bodyHash(wgsl: string, fnName: string): string {
  const body = extractFnBody(wgsl, fnName);
  return createHash('sha256').update(normalise(body)).digest('hex').slice(0, 16);
}

// ── Frozen hashes ─────────────────────────────────────────────────────────────
//
// To repin after an intentional WGSL change:
//   1. Update the cpuTracer.ts mirror to match the new WGSL.
//   2. Run `npx vitest run src/__tests__/cpuTracerDriftTripwire.test.ts`
//      — the test will fail and print the new hash in the assertion message.
//   3. Copy the new hash into the constant below.
//   4. Commit both the WGSL change, the mirror update, and the hash update together.

/**
 * FROZEN hashes — hand-updated on each intentional WGSL change.
 *
 * These are the values that make this a real tripwire: if the WGSL changes
 * without a deliberate update here, the frozen hash won't match the live hash
 * and the test fails, prompting the editor to update the cpuTracer mirror.
 *
 * Current literals are seeded from the verified 2026-06-15 WGSL state.
 */
const FROZEN_HASHES: Record<string, string> = {
  // Repin by running the test with the new WGSL, capturing the failing assertion,
  // and updating the value here + updating cpuTracer.ts.
  safe_normalize:         '55763fc94df6cec4',
  intersectAabb:          'da9898b696fd3d16',
  cosineHemisphereSample: 'f93d14571d96f6e6',
  sampleGgxVndfTangent:   '867ece6ffbc4ae0b',
  fresnelSchlick:         'c5e709aecf383066',
  frDielectric:           'b37da4158d551392',
  powerHeuristic:         '1c52c6b8f659d42f',
};

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('cpuTracer drift tripwire — WGSL function body pins (H55-b)', () => {
  it('safe_normalize body is unchanged (cpuTracer.ts:116 safeNormalize)', () => {
    const live = bodyHash(PT_WEBGPU_COMMON_WGSL, 'safe_normalize');
    expect(live, [
      `fn safe_normalize body changed. Update cpuTracer.ts safeNormalize to match,`,
      `then update FROZEN_HASHES['safe_normalize'] = '${live}'.`,
    ].join('\n')).toBe(FROZEN_HASHES['safe_normalize']);
  });

  it('intersectAabb body is unchanged (cpuTracer.ts:182 intersectAabb)', () => {
    const live = bodyHash(PT_WEBGPU_INTERSECTION_CORE_WGSL, 'intersectAabb');
    expect(live, [
      `fn intersectAabb body changed. Update cpuTracer.ts intersectAabb to match,`,
      `then update FROZEN_HASHES['intersectAabb'] = '${live}'.`,
    ].join('\n')).toBe(FROZEN_HASHES['intersectAabb']);
  });

  it('cosineHemisphereSample body is unchanged (cpuTracer.ts:232 cosineHemisphereSample)', () => {
    const live = bodyHash(PT_WEBGPU_PATH_TRACE_BSDF_WGSL, 'cosineHemisphereSample');
    expect(live, [
      `fn cosineHemisphereSample body changed. Update cpuTracer.ts cosineHemisphereSample to match,`,
      `then update FROZEN_HASHES['cosineHemisphereSample'] = '${live}'.`,
    ].join('\n')).toBe(FROZEN_HASHES['cosineHemisphereSample']);
  });

  it('sampleGgxVndfTangent body is unchanged (cpuTracer.ts:258 sampleGgxVndfTangent)', () => {
    const live = bodyHash(PT_WEBGPU_PATH_TRACE_BSDF_WGSL, 'sampleGgxVndfTangent');
    expect(live, [
      `fn sampleGgxVndfTangent body changed. Update cpuTracer.ts sampleGgxVndfTangent to match,`,
      `then update FROZEN_HASHES['sampleGgxVndfTangent'] = '${live}'.`,
    ].join('\n')).toBe(FROZEN_HASHES['sampleGgxVndfTangent']);
  });

  it('fresnelSchlick body is unchanged (cpuTracer.ts:305 fresnelSchlick)', () => {
    const live = bodyHash(PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL, 'fresnelSchlick');
    expect(live, [
      `fn fresnelSchlick body changed. Update cpuTracer.ts fresnelSchlick to match,`,
      `then update FROZEN_HASHES['fresnelSchlick'] = '${live}'.`,
    ].join('\n')).toBe(FROZEN_HASHES['fresnelSchlick']);
  });

  it('frDielectric body is unchanged (cpuTracer.ts:288 frDielectric)', () => {
    const live = bodyHash(PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL, 'frDielectric');
    expect(live, [
      `fn frDielectric body changed. Update cpuTracer.ts frDielectric to match,`,
      `then update FROZEN_HASHES['frDielectric'] = '${live}'.`,
    ].join('\n')).toBe(FROZEN_HASHES['frDielectric']);
  });

  it('powerHeuristic body is unchanged (cpuTracer.ts:316 powerHeuristic)', () => {
    const live = bodyHash(PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL, 'powerHeuristic');
    expect(live, [
      `fn powerHeuristic body changed. Update cpuTracer.ts powerHeuristic to match,`,
      `then update FROZEN_HASHES['powerHeuristic'] = '${live}'.`,
    ].join('\n')).toBe(FROZEN_HASHES['powerHeuristic']);
  });

  it('extractFnBody correctly isolates a function with nested braces', () => {
    // Regression guard for the extractor itself.
    const wgsl = `fn outer(a: f32) -> f32 { let x = { 1.0 }; return x; } fn other() {}`;
    const body = extractFnBody(wgsl, 'outer');
    expect(body).toBe('{ let x = { 1.0 }; return x; }');
    // 'other' is not captured.
    expect(body).not.toContain('other');
  });
});
