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
import { BACKEND_PROMISE_LEDGER, MATERIAL_SPEC_FIELDS } from '../engine/promiseLedger.js';
import { buildCapabilities as buildPtWebgl2Capabilities, PT_WEBGL2_SUPPORT } from '@vitrum/pt-webgl2/src/capabilities.js';

function sorted(values: Iterable<string>): string[] {
  return Array.from(values).sort();
}

const SUPPORT_MODES = new Set([
  'native',
  'approximate',
  'fallback-generated-mesh',
  'fallback-rebuild',
  'unsupported',
]);

const DISPLACEMENT_FIELDS = [
  'displacementMap',
  'displacementScale',
  'displacementBias',
] as const;

const WALKAROUND_PERMANENT_UNSUPPORTED_OPTICAL_FIELDS = [
  'spectralAttenuation',
  'dispersionAbbeNumber',
  'thinFilmStack',
] as const;

const WALKAROUND_APPROXIMATE_VOLUME_LAYER_FIELDS = [
  'scatteringCoefficient',
  'scatteringAnisotropy',
  'scatteringCoefficientRGB',
  'frontLayer',
  'backLayer',
] as const;

const WALKAROUND_TRANSPARENT_TRANSPORT_BOUNDARY_FIELDS = [
  'alphaMode',
  'alphaCutoff',
  'opacity',
  'alphaMap',
  'transmission',
  'transmissionMap',
  'thickness',
  'thicknessMap',
] as const;

// ── material support rows ────────────────────────────────────────────────────

describe('BACKEND_PROMISE_LEDGER material support rows', () => {
  it('every backend material matrix exactly covers MATERIAL_SPEC_FIELDS', () => {
    const expected = sorted(MATERIAL_SPEC_FIELDS);

    for (const [backend, record] of Object.entries(BACKEND_PROMISE_LEDGER)) {
      expect(sorted(Object.keys(record.supportDetails.materials ?? {})), backend).toEqual(expected);
    }
  });

  it('every backend material matrix uses only declared BackendSupportMode values', () => {
    for (const [backend, record] of Object.entries(BACKEND_PROMISE_LEDGER)) {
      for (const [field, mode] of Object.entries(record.supportDetails.materials ?? {})) {
        expect(SUPPORT_MODES.has(mode), `${backend}.${field}=${mode}`).toBe(true);
      }
    }
  });
});

// ── Road-to-100 future-contract boundaries ───────────────────────────────────

describe('BACKEND_PROMISE_LEDGER Road-to-100 future-contract boundaries', () => {
  it('displacement rows stay approximate on every backend until microdisplacement is contracted', () => {
    for (const [backend, record] of Object.entries(BACKEND_PROMISE_LEDGER)) {
      for (const field of DISPLACEMENT_FIELDS) {
        expect(record.supportDetails.materials?.[field], `${backend}.${field}`).toBe('approximate');
      }
    }
  });

  it('walkaround specialty optical rows retain explicit unsupported/approximate boundaries', () => {
    const materials = BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportDetails.materials;

    for (const field of WALKAROUND_PERMANENT_UNSUPPORTED_OPTICAL_FIELDS) {
      expect(materials?.[field], field).toBe('unsupported');
    }
    for (const field of WALKAROUND_APPROXIMATE_VOLUME_LAYER_FIELDS) {
      expect(materials?.[field], field).toBe('approximate');
    }
  });

  it('walkaround transparent material rows do not claim true transparent GI transport', () => {
    const materials = BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportDetails.materials;

    for (const field of WALKAROUND_TRANSPARENT_TRANSPORT_BOUNDARY_FIELDS) {
      expect(materials?.[field], field).toBe('approximate');
    }
  });

  it('receiveShadow remains unsupported on every backend until the core contract changes', () => {
    for (const [backend, record] of Object.entries(BACKEND_PROMISE_LEDGER)) {
      expect(record.supportDetails.shadows.receiveShadow, backend).toBe('unsupported');
    }
  });
});

// ── pt-webgl2 ─────────────────────────────────────────────────────────────────

describe('BACKEND_PROMISE_LEDGER["pt-webgl2"] vs PT_WEBGL2_SUPPORT', () => {
  const ledger = BACKEND_PROMISE_LEDGER['pt-webgl2'];

  it('supportedPrimitiveKinds matches ledger', () => {
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

  it('supportedAnalyticShapes matches ledger', () => {
    expect(sorted(PT_WEBGL2_SUPPORT.supportedAnalyticShapes)).toEqual(
      sorted(ledger.supportedAnalyticShapes),
    );
  });

  it('ledger mutations grade: scalar material/emitter/env and same-topology geometry edits are native; topology rows stay fallback', () => {
    // pt-webgl2 updates scalar material slots, emitter textures (including the
    // mesh-area folded-material path), environment textures, and same-topology
    // transform/position/attribute payloads without rebuilding or reallocating
    // the merged BVH texture set. Topology/list edits still rebuild the texture
    // pack because there is no primitive-list splice yet.
    const { mutations } = ledger.supportDetails;
    expect(mutations.material).toBe('native');
    expect(mutations.environment).toBe('native');
    expect(mutations.resize).toBe('native');
    expect(mutations.transform).toBe('native');
    expect(mutations.positions).toBe('native');
    expect(mutations.emitter).toBe('native');
    expect(mutations.topology).toBe('fallback-rebuild');
    expect(mutations.addPrimitive).toBe('fallback-rebuild');
    expect(mutations.removePrimitive).toBe('fallback-rebuild');
    expect(mutations.lighting).toBe('unsupported');
  });

  it('debugSurface matches the method promise row', () => {
    const caps = buildPtWebgl2Capabilities('none', 8, Infinity, false);
    expect(caps.debugSurface).toBe(true);
    expect(ledger.methodPromises.debug).toBe(caps.debugSurface);
  });

  it('supportsAuxBuffers stays false because WebGL2 lacks variance and motion-vector outputs', () => {
    const fullCaps = buildPtWebgl2Capabilities('none', 8, Infinity, true);
    const liteCaps = buildPtWebgl2Capabilities('none', 8, Infinity, false);
    expect(fullCaps.supportsAuxBuffers).toBe(false);
    expect(ledger.supportsAuxBuffers).toBe(fullCaps.supportsAuxBuffers);
    expect(liteCaps.supportsAuxBuffers).toBe(false);
  });

  it('analytic primitiveKind is accepted through generated-mesh fallback', () => {
    expect(ledger.supportedPrimitiveKinds).toContain('analytic');
  });

  it('supportDetails.primitives.analytic is "fallback-generated-mesh"', () => {
    expect(ledger.supportDetails.primitives.analytic).toBe('fallback-generated-mesh');
  });

  it('supportDetails.analyticShapes: all shapes are "fallback-generated-mesh"', () => {
    for (const grade of Object.values(ledger.supportDetails.analyticShapes)) {
      expect(grade).toBe('fallback-generated-mesh');
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

  it('supportedEmitterKinds includes point and spot (native direct-light path)', () => {
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
