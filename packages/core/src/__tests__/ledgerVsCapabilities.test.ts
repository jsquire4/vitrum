/**
 * Ledger vs capability-module consistency gate (H39/H40/H41).
 *
 * Asserts that the `BACKEND_PROMISE_LEDGER` entries agree with the exported
 * capability constants from each backend's capability module. This test can
 * only import SIDE-EFFECT-FREE symbols (no GPU construction). It imports the
 * exported `*_SUPPORT` objects and `buildCapabilities` functions, which are
 * pure data/pure functions — no WebGL/WebGPU context required.
 *
 * ── Importability note ──────────────────────────────────────────────────────
 * `@vitrum/pt-webgl2/src/capabilities.ts` and `@vitrum/pt-webgpu`'s support
 * consts are imported via their package paths. The walkaround-hybrid backend
 * does not export a standalone `*_SUPPORT` const (its support sets are inlined
 * at construction time); we assert against the ledger's own declared values for
 * structural self-consistency instead.
 *
 * ── What this test does NOT cover ───────────────────────────────────────────
 * Per-engine constructor compliance (does the built engine.capabilities match
 * the ledger row?) is tested in each backend package's own ledger test file
 * (e.g. `pt-webgpu/src/__tests__/promiseLedger.test.ts`). This test covers
 * the static consistency between the ledger and the exported support consts
 * only — no GPU needed.
 */

import { describe, expect, it } from 'vitest';
import { BACKEND_PROMISE_LEDGER } from '../engine/promiseLedger.js';
import { PT_WEBGL2_SUPPORT } from '@vitrum/pt-webgl2/src/capabilities.js';

function sorted(values: Iterable<string>): string[] {
  return Array.from(values).sort();
}

// ── pt-webgl2 ─────────────────────────────────────────────────────────────────

describe('BACKEND_PROMISE_LEDGER["pt-webgl2"] vs PT_WEBGL2_SUPPORT', () => {
  const ledger = BACKEND_PROMISE_LEDGER['pt-webgl2'];

  it('supportedPrimitiveKinds matches ledger (analytic excluded — empty runtime set)', () => {
    // PT_WEBGL2_SUPPORT does not include 'analytic'; ledger now matches.
    expect(sorted(PT_WEBGL2_SUPPORT.supportedPrimitiveKinds)).toEqual(
      sorted(ledger.supportedPrimitiveKinds),
    );
  });

  it('supportedEmitterKinds matches ledger', () => {
    expect(sorted(PT_WEBGL2_SUPPORT.supportedEmitterKinds)).toEqual(
      sorted(ledger.supportedEmitterKinds),
    );
  });

  it('supportedEnvironmentKinds matches ledger', () => {
    expect(sorted(PT_WEBGL2_SUPPORT.supportedEnvironmentKinds)).toEqual(
      sorted(ledger.supportedEnvironmentKinds),
    );
  });

  it('supportedAnalyticShapes: runtime set is empty; ledger says [] (unsupported)', () => {
    // The runtime set is an empty Set<AnalyticShape>; no analytic shape is
    // accepted. The ledger must also be empty.
    expect(PT_WEBGL2_SUPPORT.supportedAnalyticShapes.size).toBe(0);
    expect(ledger.supportedAnalyticShapes).toHaveLength(0);
  });

  it('ledger mutations grade: ALL must be "fallback-rebuild" or "unsupported" (buildCapabilities overrides)', () => {
    // buildCapabilities() in pt-webgl2/src/capabilities.ts overrides every
    // mutation kind to 'fallback-rebuild'. The ledger must reflect this.
    const { mutations } = ledger.supportDetails;
    const acceptableGrades = new Set(['fallback-rebuild', 'unsupported']);
    for (const [key, grade] of Object.entries(mutations) as [string, string][]) {
      expect(
        acceptableGrades.has(grade),
        `mutations.${key} should be fallback-rebuild or unsupported, got '${grade}'`,
      ).toBe(true);
    }
  });

  it('analytic primitiveKind not in supportedPrimitiveKinds (warn-and-skip)', () => {
    expect(ledger.supportedPrimitiveKinds).not.toContain('analytic');
  });

  it('supportDetails.primitives.analytic is "unsupported"', () => {
    expect(ledger.supportDetails.primitives.analytic).toBe('unsupported');
  });

  it('supportDetails.analyticShapes: all shapes are "unsupported"', () => {
    for (const grade of Object.values(ledger.supportDetails.analyticShapes)) {
      expect(grade).toBe('unsupported');
    }
  });
});

// ── walkaround-hybrid ─────────────────────────────────────────────────────────

describe('BACKEND_PROMISE_LEDGER["walkaround-hybrid"] structural self-consistency', () => {
  const ledger = BACKEND_PROMISE_LEDGER['walkaround-hybrid'];

  it('point and spot emitters are graded "native" (H41 additive analytic NEE in shade.wgsl)', () => {
    // H41 — promoted from 'approximate' (DDGI-only) to 'native' (additive
    // analytic NEE loop with inverse-square + spot cone falloff + shadow rays).
    expect(ledger.supportDetails.emitters.point).toBe('native');
    expect(ledger.supportDetails.emitters.spot).toBe('native');
  });

  it('rect-area, disc-area, directional, mesh-area emitters are "native"', () => {
    expect(ledger.supportDetails.emitters['rect-area']).toBe('native');
    expect(ledger.supportDetails.emitters['disc-area']).toBe('native');
    expect(ledger.supportDetails.emitters.directional).toBe('native');
    expect(ledger.supportDetails.emitters['mesh-area']).toBe('native');
  });

  it('supportedEmitterKinds includes point and spot (accepted, just approximate)', () => {
    expect(ledger.supportedEmitterKinds).toContain('point');
    expect(ledger.supportedEmitterKinds).toContain('spot');
  });

  it('analytic shapes are fallback-generated-mesh (converted to mesh before BVH ingestion)', () => {
    for (const grade of Object.values(ledger.supportDetails.analyticShapes)) {
      expect(grade).toBe('fallback-generated-mesh');
    }
  });

  it('accumulates is false (realtime GI, not converged PT)', () => {
    expect(ledger.accumulates).toBe(false);
  });
});

// ── pt-webgpu ─────────────────────────────────────────────────────────────────

describe('BACKEND_PROMISE_LEDGER["pt-webgpu"] structural self-consistency', () => {
  const ledger = BACKEND_PROMISE_LEDGER['pt-webgpu'];

  it('analytic shapes are all "native"', () => {
    for (const grade of Object.values(ledger.supportDetails.analyticShapes)) {
      expect(grade).toBe('native');
    }
  });

  it('supportedAnalyticShapes has all 5 shapes', () => {
    expect(ledger.supportedAnalyticShapes).toHaveLength(5);
    expect(ledger.supportedAnalyticShapes).toContain('sphere');
    expect(ledger.supportedAnalyticShapes).toContain('box');
    expect(ledger.supportedAnalyticShapes).toContain('capsule');
    expect(ledger.supportedAnalyticShapes).toContain('cylinder');
    expect(ledger.supportedAnalyticShapes).toContain('h-channel-came');
  });

  it('directional/rect-area/point/spot/mesh-area emitters are "native"', () => {
    const e = ledger.supportDetails.emitters;
    expect(e.directional).toBe('native');
    expect(e['rect-area']).toBe('native');
    expect(e.point).toBe('native');
    expect(e.spot).toBe('native');
    expect(e['mesh-area']).toBe('native');
  });

  it('disc-area emitter is "native" (analytic concentric-disc packing, 2026-06-10)', () => {
    // pt-webgpu now packs disc-area emitters natively into the rect stream with a shape
    // tag in emission.w (0=rect, 1=disc). Sampling uses the Shirley-Chiu concentric-disc
    // map; MIS-half uses circle containment + π·r² area. The 32-triangle fan is removed.
    // Promoted approximate → native 2026-06-10. A/B radiometric validation in R9-B.
    expect(ledger.supportDetails.emitters['disc-area']).toBe('native');
  });

  it('accumulates is true (converged PT)', () => {
    expect(ledger.accumulates).toBe(true);
  });
});
