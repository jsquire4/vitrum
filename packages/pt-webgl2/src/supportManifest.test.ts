import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { MATERIAL_SPEC_FIELDS } from '@vitrum/core';
import { buildCapabilities } from './capabilities.js';
import {
  PT_WEBGL2_MATERIAL_SUPPORT,
  PT_WEBGL2_SUPPORT,
  PT_WEBGL2_SUPPORT_MANIFEST,
} from './supportManifest.js';

function implementedEntries(
  record: Readonly<Record<string, string>>,
): string[] {
  return Object.entries(record)
    .filter(([, mode]) => mode !== 'unsupported')
    .map(([key]) => key)
    .sort();
}

describe('pt-webgl2 executable support manifest', () => {
  it('exhaustively classifies the core material contract', () => {
    expect(Object.keys(PT_WEBGL2_MATERIAL_SUPPORT).sort()).toEqual(
      [...MATERIAL_SPEC_FIELDS].sort(),
    );
    expect(
      Object.entries(PT_WEBGL2_MATERIAL_SUPPORT)
        .filter(([, mode]) => mode === 'unsupported')
        .map(([field]) => field),
    ).toEqual(['extensions']);
  });

  it('derives every coarse scene-acceptance set from detailed rows', () => {
    expect([...PT_WEBGL2_SUPPORT.supportedPrimitiveKinds].sort()).toEqual(
      implementedEntries(PT_WEBGL2_SUPPORT_MANIFEST.primitives),
    );
    expect([...PT_WEBGL2_SUPPORT.supportedEmitterKinds].sort()).toEqual(
      implementedEntries(PT_WEBGL2_SUPPORT_MANIFEST.emitters),
    );
    expect([...PT_WEBGL2_SUPPORT.supportedEnvironmentKinds].sort()).toEqual(
      implementedEntries(PT_WEBGL2_SUPPORT_MANIFEST.environments),
    );
    expect([...PT_WEBGL2_SUPPORT.supportedAnalyticShapes].sort()).toEqual(
      implementedEntries(PT_WEBGL2_SUPPORT_MANIFEST.analyticShapes),
    );
  });

  it.each([
    ['none', { bdpt: false, spectral: false, sampling: 'pcg' }],
    ['none', { bdpt: true, spectral: true, sampling: 'sobol' }],
    ['oidn-final', { bdpt: false, spectral: true, sampling: 'pcg' }],
    ['oidn-final', { bdpt: true, spectral: false, sampling: 'sobol' }],
  ] as const)(
    'publishes the executable manifest for denoiser=%s and every feature family',
    (denoiser, selected) => {
      const caps = buildCapabilities(denoiser, 8, 4096, selected);
      expect(caps.supportDetails).toBe(PT_WEBGL2_SUPPORT_MANIFEST);
      expect(caps.supportDetails?.mutations)
        .toBe(PT_WEBGL2_SUPPORT_MANIFEST.mutations);
      expect([...caps.supportedPrimitiveKinds!].sort()).toEqual(
        [...PT_WEBGL2_SUPPORT.supportedPrimitiveKinds].sort(),
      );
      expect([...caps.supportedEmitterKinds].sort()).toEqual(
        [...PT_WEBGL2_SUPPORT.supportedEmitterKinds].sort(),
      );
    },
  );

  it('derives mutation API flags and the coarse aux promise from the manifest', () => {
    const caps = buildCapabilities('none', 8, 4096);
    const mutations = PT_WEBGL2_SUPPORT_MANIFEST.mutations;
    expect(caps.incrementalPatchSupport).toEqual({
      transform: mutations.transform !== 'unsupported',
      positions: mutations.positions !== 'unsupported',
      material: mutations.material !== 'unsupported',
      emitter: mutations.emitter !== 'unsupported',
      topology: mutations.topology !== 'unsupported',
    });
    expect(caps.supportsAddRemovePrimitive).toBe(
      mutations.addPrimitive !== 'unsupported' &&
      mutations.removePrimitive !== 'unsupported',
    );
    expect(caps.supportsAuxBuffers)
      .toBe(PT_WEBGL2_SUPPORT_MANIFEST.motionVectors != null);
  });

  it('does not read the static promise ledger in runtime backend modules', () => {
    for (const file of ['capabilities.ts', 'index.ts', 'supportManifest.ts']) {
      const source = readFileSync(new URL(`./${file}`, import.meta.url), 'utf8');
      expect(source, file).not.toContain('BACKEND_PROMISE_LEDGER');
    }
  });
});
