// Capability-drift pin for the hand-maintained per-backend tables in
// featureReport.ts (T1-4 / I4-4). These tables duplicate the remit of
// @vitrum/core's BackendSupportDetails; there is no compile-time link, so this
// test is the guard that catches silent drift between the adapter's tables and
// the backends' declared capabilities / promise-ledger rows.
//
// Non-breaking first step (per plan): this does NOT restructure the tables; it
// only asserts they stay consistent with core's declarations. If a future edit
// desyncs them, one of these assertions fails.
import { describe, expect, it } from 'vitest';
import {
  BACKEND_PROMISE_LEDGER,
  type BackendId,
  type MaterialSpec,
} from '@vitrum/core';
import {
  PT_WEBGPU_FULL_SUPPORT_MANIFEST,
  PT_WEBGPU_LITE_EXTRA_UNSUPPORTED_MATERIAL_FIELDS,
  PT_WEBGPU_LITE_SUPPORT_MANIFEST,
} from '@vitrum/pt-webgpu/support-profile';
import {
  analyzeGltfAsset,
  evaluateGltfBackendProfileCompatibility,
  VERTEX_COLOR_SUPPORT,
  PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS as ADAPTER_PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS,
} from './featureReport.js';

// The profile-id space the adapter models: one row per backend, plus the
// constrained pt-webgpu-lite profile. Kept here (not imported) so a drift in the
// adapter's profile set is caught against this independently-authored list.
const EXPECTED_PROFILE_IDS = [
  'pt-webgl2',
  'pt-webgpu',
  'pt-webgpu-lite',
  'walkaround-hybrid',
] as const;

const CORE_BACKENDS: readonly BackendId[] = ['pt-webgl2', 'pt-webgpu', 'walkaround-hybrid'];

describe('featureReport capability tables — drift pin against core BackendSupportDetails', () => {
  it('VERTEX_COLOR_SUPPORT keys exactly cover the modeled backend profile set', () => {
    expect(Object.keys(VERTEX_COLOR_SUPPORT).sort()).toEqual([...EXPECTED_PROFILE_IDS].sort());
  });

  it('every VERTEX_COLOR_SUPPORT profile maps to a real core backend (or the lite tier of one)', () => {
    for (const profileId of Object.keys(VERTEX_COLOR_SUPPORT)) {
      const backend = profileId === 'pt-webgpu-lite' ? 'pt-webgpu' : (profileId as BackendId);
      expect(BACKEND_PROMISE_LEDGER[backend]).toBeDefined();
    }
  });

  it('pt-webgpu-lite reports vertex colors as unsupported (it composes no group-3 bindings), unlike full pt-webgpu', () => {
    // Lite is strictly a restriction of full pt-webgpu; the vertex-color row must
    // reflect that the lite tier drops support the full backend has.
    expect(VERTEX_COLOR_SUPPORT['pt-webgpu-lite']).toBe('unsupported');
    expect(VERTEX_COLOR_SUPPORT['pt-webgpu']).not.toBe('unsupported');
  });

  it('every backend in core is represented by exactly one non-lite VERTEX_COLOR_SUPPORT profile', () => {
    const nonLiteProfiles = Object.keys(VERTEX_COLOR_SUPPORT).filter((p) => p !== 'pt-webgpu-lite');
    expect(nonLiteProfiles.sort()).toEqual([...CORE_BACKENDS].sort());
  });

  it('PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS are all valid keys of the pt-webgpu core materials matrix', () => {
    // Drift guard: if core renames/removes a MaterialSpec field the lite table
    // references, this fails (the field would no longer be an audited core row).
    const coreMaterials = BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.materials as
      Readonly<Partial<Record<keyof MaterialSpec, unknown>>>;
    const missing: string[] = [];
    for (const field of ADAPTER_PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS) {
      if (!(field in coreMaterials)) missing.push(String(field));
    }
    expect(missing).toEqual([]);
  });

  it('the lite-unsupported fields are NOT marked unsupported by the FULL pt-webgpu backend (lite is the restrictor)', () => {
    // The lite tier is what makes these fields unsupported; if the FULL backend
    // itself already declared one unsupported, the adapter table would be
    // double-encoding a core promise and would drift. Assert full-tier support.
    const coreMaterials = BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails.materials as
      Readonly<Partial<Record<keyof MaterialSpec, string>>>;
    const wronglyUnsupportedAtFull: string[] = [];
    for (const field of ADAPTER_PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS) {
      if (coreMaterials[field] === 'unsupported') wronglyUnsupportedAtFull.push(String(field));
    }
    expect(wronglyUnsupportedAtFull).toEqual([]);
  });

  it('the lite-unsupported list has no duplicate fields', () => {
    const set = new Set(ADAPTER_PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS as readonly string[]);
    expect(set.size).toBe(ADAPTER_PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS.length);
  });

  it('consumes the exact backend-owned lite-profile restriction tuple, including thicknessMap', () => {
    expect(ADAPTER_PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS)
      .toBe(PT_WEBGPU_LITE_EXTRA_UNSUPPORTED_MATERIAL_FIELDS);
    expect(ADAPTER_PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS).toContain('thicknessMap');
  });

  it('evaluates every pt-webgpu profile material row from the exact executable manifest', () => {
    const analyzed = analyzeGltfAsset({
      asset: { version: '2.0' },
      materials: [{}],
    });
    const report = {
      ...analyzed,
      materials: {
        ...analyzed.materials,
        materialFields: ['frontLayer'] as const,
        issuePaths: {
          ...analyzed.materials.issuePaths,
          'field:frontLayer': ['materials[0]'],
        },
      },
    };

    expect(PT_WEBGPU_FULL_SUPPORT_MANIFEST.materials.frontLayer).toBe('native');
    expect(PT_WEBGPU_LITE_SUPPORT_MANIFEST.materials.frontLayer).toBe('approximate');

    const full =
      evaluateGltfBackendProfileCompatibility(report, 'pt-webgpu');
    const lite =
      evaluateGltfBackendProfileCompatibility(report, 'pt-webgpu-lite');
    expect(full.issues.some((issue) => issue.name === 'frontLayer')).toBe(false);
    expect(lite.issues).toContainEqual({
      category: 'material',
      name: 'frontLayer',
      support: 'approximate',
      path: 'materials[0]',
      message:
        'Backend profile pt-webgpu-lite reports material field "frontLayer" as approximate.',
    });
  });

  it('makes the walkaround rough-transmission profile affect glTF compatibility', () => {
    const report = analyzeGltfAsset({
      asset: { version: '2.0' },
      materials: [{
        extensions: {
          KHR_materials_transmission: { transmissionFactor: 0.9 },
        },
      }],
    });

    expect(report.materials.materialProfiles).toEqual(['roughTransmission']);
    expect(report.materials.issuePaths['profile:roughTransmission']).toEqual([
      'materials[0].extensions.KHR_materials_transmission',
    ]);

    const walkaround =
      evaluateGltfBackendProfileCompatibility(report, 'walkaround-hybrid');
    expect(walkaround.issues).toContainEqual({
      category: 'material',
      name: 'profile:roughTransmission',
      support: 'approximate',
      path: 'materials[0].extensions.KHR_materials_transmission',
      message:
        'Backend profile walkaround-hybrid reports conditional material profile ' +
        '"roughTransmission" as approximate.',
    });
    expect(walkaround.approximateCount).toBeGreaterThan(0);

    const ptWebgpu =
      evaluateGltfBackendProfileCompatibility(report, 'pt-webgpu');
    expect(ptWebgpu.issues.some(
      (issue) => issue.name === 'profile:roughTransmission',
    )).toBe(false);
  });

  it('does not classify delta or disabled transmission as rough transmission', () => {
    const report = analyzeGltfAsset({
      asset: { version: '2.0' },
      materials: [
        {
          pbrMetallicRoughness: { roughnessFactor: 0 },
          extensions: {
            KHR_materials_transmission: { transmissionFactor: 0.9 },
          },
        },
        {
          pbrMetallicRoughness: { roughnessFactor: 0.35 },
          extensions: {
            KHR_materials_transmission: { transmissionFactor: 0 },
          },
        },
      ],
    });

    expect(report.materials.materialProfiles).toEqual([]);
  });
});
