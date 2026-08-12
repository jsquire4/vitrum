import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it, vi } from 'vitest';
import {
  BACKEND_PROMISE_LEDGER,
  ENGINE_DENOISER_MODES,
  MATERIAL_SPEC_FIELDS,
  supportSetsFromManifest,
} from '@vitrum/core';
import { SOBOL_DIMENSION_COUNT } from '@vitrum/shared-samplers';

import { createPTEngine_WebGPU } from '../index.js';
import {
  PT_WEBGPU_BDPT_SUPPORT,
  PT_WEBGPU_FULL_SUPPORT,
  PT_WEBGPU_FULL_SUPPORT_MANIFEST,
  PT_WEBGPU_LITE_SUPPORT,
  PT_WEBGPU_LITE_SUPPORT_MANIFEST,
} from '../supportManifest.js';
import { BDPT_MAX_LIGHT_BOUNCES } from '../ptWebgpuValidation.js';
import {
  PT_WEBGPU_LITE_EXTRA_UNSUPPORTED_MATERIAL_FIELDS,
} from '../supportDetails.js';
import { PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE } from '../webgpuLimits.js';

function makeDevice(tier: 'full' | 'lite'): GPUDevice {
  return {
    limits: {
      maxStorageBuffersPerShaderStage:
        tier === 'full'
          ? PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE
          : 8,
      maxStorageTexturesPerShaderStage: tier === 'full' ? 8 : 4,
    },
    createCommandEncoder: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function sorted(values: Iterable<string>): string[] {
  return Array.from(values).sort();
}

function productionTypeScriptFiles(directory: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = `${directory}/${entry.name}`;
    if (entry.isDirectory()) {
      if (entry.name !== '__tests__') {
        files.push(...productionTypeScriptFiles(path));
      }
    } else if (entry.isFile() && entry.name.endsWith('.ts')) {
      files.push(path);
    }
  }
  return files;
}

describe('pt-webgpu executable support manifests', () => {
  it('classifies every material and denoiser enum on both tiers', () => {
    const materialKeys = sorted(MATERIAL_SPEC_FIELDS);
    expect(sorted(Object.keys(PT_WEBGPU_FULL_SUPPORT_MANIFEST.materials)))
      .toEqual(materialKeys);
    expect(sorted(Object.keys(PT_WEBGPU_LITE_SUPPORT_MANIFEST.materials)))
      .toEqual(materialKeys);
    expect(sorted(Object.keys(PT_WEBGPU_FULL_SUPPORT_MANIFEST.denoisers)))
      .toEqual(sorted(ENGINE_DENOISER_MODES));
    expect(sorted(Object.keys(PT_WEBGPU_LITE_SUPPORT_MANIFEST.denoisers)))
      .toEqual(sorted(ENGINE_DENOISER_MODES));
  });

  it('derives both coarse acceptance sets from the selected manifest', () => {
    const full = supportSetsFromManifest(PT_WEBGPU_FULL_SUPPORT_MANIFEST);
    const lite = supportSetsFromManifest(PT_WEBGPU_LITE_SUPPORT_MANIFEST);
    for (const key of [
      'supportedPrimitiveKinds',
      'supportedEmitterKinds',
      'supportedEnvironmentKinds',
      'supportedAnalyticShapes',
    ] as const) {
      expect(sorted(PT_WEBGPU_FULL_SUPPORT[key])).toEqual(sorted(full[key]));
      expect(sorted(PT_WEBGPU_LITE_SUPPORT[key])).toEqual(sorted(lite[key]));
    }
    expect(sorted(PT_WEBGPU_LITE_SUPPORT.supportedPrimitiveKinds))
      .toEqual(['analytic', 'instanced-mesh', 'mesh', 'skinned-mesh']);
    expect(sorted(PT_WEBGPU_LITE_SUPPORT.supportedEmitterKinds))
      .toEqual(['directional', 'disc-area', 'point', 'rect-area', 'spot']);
    expect(sorted(PT_WEBGPU_LITE_SUPPORT.supportedAnalyticShapes))
      .toEqual(['box', 'capsule', 'cylinder', 'h-channel-came', 'sphere']);
  });

  it('publishes the selected manifest itself as live tier evidence', async () => {
    const full = await createPTEngine_WebGPU({ device: makeDevice('full') });
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const lite = await createPTEngine_WebGPU({ device: makeDevice('lite') });
    try {
      expect(full.capabilities.supportDetails)
        .toBe(PT_WEBGPU_FULL_SUPPORT_MANIFEST);
      expect(lite.capabilities.supportDetails)
        .toBe(PT_WEBGPU_LITE_SUPPORT_MANIFEST);
      expect(full.capabilities.supportDetails?.mutations)
        .toBe(PT_WEBGPU_FULL_SUPPORT_MANIFEST.mutations);
      expect(lite.capabilities.supportDetails?.mutations)
        .toBe(PT_WEBGPU_LITE_SUPPORT_MANIFEST.mutations);
      expect(full.capabilities.supportDetails?.samplingSequences)
        .toBe(PT_WEBGPU_FULL_SUPPORT_MANIFEST.samplingSequences);
      expect(lite.capabilities.supportDetails?.denoisers)
        .toBe(PT_WEBGPU_LITE_SUPPORT_MANIFEST.denoisers);
    } finally {
      full.dispose();
      lite.dispose();
      warn.mockRestore();
    }
  });

  it('uses the full manifest as the executable BDPT limit and tier gate', () => {
    expect(PT_WEBGPU_FULL_SUPPORT_MANIFEST.bidirectionalPathTracing)
      .toBe(PT_WEBGPU_BDPT_SUPPORT);
    expect(BDPT_MAX_LIGHT_BOUNCES)
      .toBe(PT_WEBGPU_BDPT_SUPPORT.maxLightVertices);
    expect(PT_WEBGPU_LITE_SUPPORT_MANIFEST.bidirectionalPathTracing)
      .toBeUndefined();
  });

  it('keeps the static full-tier ledger synchronized without consuming it at runtime', () => {
    expect(PT_WEBGPU_FULL_SUPPORT_MANIFEST)
      .toEqual(BACKEND_PROMISE_LEDGER['pt-webgpu'].supportDetails);
    expect(
      PT_WEBGPU_FULL_SUPPORT_MANIFEST.samplingSequences?.sobol
        ?.lowDiscrepancyDimensions,
    ).toBe(SOBOL_DIMENSION_COUNT);
  });

  it('derives the lite-only material rejection list from the tier delta', () => {
    const expected = Object.keys(PT_WEBGPU_LITE_SUPPORT_MANIFEST.materials)
      .filter((field) =>
        PT_WEBGPU_LITE_SUPPORT_MANIFEST.materials[
          field as keyof typeof PT_WEBGPU_LITE_SUPPORT_MANIFEST.materials
        ] === 'unsupported' &&
        PT_WEBGPU_FULL_SUPPORT_MANIFEST.materials[
          field as keyof typeof PT_WEBGPU_FULL_SUPPORT_MANIFEST.materials
        ] !== 'unsupported')
      .sort();
    expect(sorted(PT_WEBGPU_LITE_EXTRA_UNSUPPORTED_MATERIAL_FIELDS))
      .toEqual(expected);
  });

  it('keeps BACKEND_PROMISE_LEDGER out of every production runtime source', () => {
    const srcDirectory = fileURLToPath(new URL('../', import.meta.url))
      .replaceAll('\\', '/');
    const offenders = productionTypeScriptFiles(srcDirectory)
      .filter((path) => readFileSync(path, 'utf8').includes('BACKEND_PROMISE_LEDGER'));
    expect(offenders).toEqual([]);
  });
});
