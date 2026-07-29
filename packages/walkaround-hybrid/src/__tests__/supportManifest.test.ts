import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it, vi } from 'vitest';
import {
  BACKEND_PROMISE_LEDGER,
  ENGINE_DENOISER_MODES,
  MATERIAL_SPEC_FIELDS,
  supportSetsFromManifest,
  type EngineDenoiserMode,
} from '@vitrum/core';

import { HybridEngine, type HybridEngineOptions } from '../HybridEngine.js';
import { VALID_DENOISERS } from '../HybridEngineOptions.js';
import { WALKAROUND_NEURAL_DENOISER_SHAPE_REQUIREMENT } from '../neural/shapeContract.js';
import { CONSUMED_MATERIAL_FIELD_DOCS } from '../restir/consumedMaterialFields.js';
import {
  WALKAROUND_FULL_CERTIFIED_SUPPORT_MANIFEST,
  WALKAROUND_MATERIAL_SUPPORT,
  walkaroundSupportManifest,
  type WalkaroundNeuralCertification,
  type WalkaroundSupportTier,
} from '../supportManifest.js';

function sorted(values: Iterable<string>): string[] {
  return Array.from(values).sort();
}

function makeDeviceStub(): GPUDevice {
  return {
    limits: {
      maxBindGroups: 5,
      maxStorageBuffersPerShaderStage: 22,
      maxComputeWorkgroupStorageSize: 16_384,
    },
    createCommandEncoder: vi.fn(),
    createBuffer: vi.fn(),
    createTexture: vi.fn(),
    createBindGroup: vi.fn(),
    queue: { submit: vi.fn(), writeBuffer: vi.fn() },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    lost: new Promise<never>(() => {}),
  } as unknown as GPUDevice;
}

function makeOpts(overrides: Partial<HybridEngineOptions> = {}): HybridEngineOptions {
  return {
    device: makeDeviceStub(),
    width: 64,
    height: 64,
    primaryLightDir: [0, -1, 0],
    primaryLightIntensity: 1,
    skyTint: [1, 1, 1],
    skyIrradiance: 1,
    ...overrides,
  };
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

const TIERS = ['full', 'lite'] as const satisfies readonly WalkaroundSupportTier[];
const NEURAL_CERTIFICATIONS = [
  'absent',
  'uncertified',
  'certified',
] as const satisfies readonly WalkaroundNeuralCertification[];
const OIDN_MODEL_STATES = [false, true] as const;

describe('walkaround executable support manifests', () => {
  it('exhaustively classifies the material contract and matches real ingestion docs', () => {
    expect(sorted(Object.keys(WALKAROUND_MATERIAL_SUPPORT))).toEqual(sorted(MATERIAL_SPEC_FIELDS));
    expect(sorted(Object.keys(WALKAROUND_MATERIAL_SUPPORT))).toEqual(
      sorted(Object.keys(CONSUMED_MATERIAL_FIELD_DOCS)),
    );
  });

  it('enumerates every tier, neural-certification, and OIDN-model state', () => {
    for (const tier of TIERS) {
      for (const neuralCertification of NEURAL_CERTIFICATIONS) {
        for (const oidnModelAvailable of OIDN_MODEL_STATES) {
          const manifest = walkaroundSupportManifest({
            tier,
            neuralCertification,
            oidnModelAvailable,
          });
          expect(
            sorted(Object.keys(manifest.denoisers)),
            `${tier}/${neuralCertification}/oidn=${oidnModelAvailable}`,
          ).toEqual(sorted(ENGINE_DENOISER_MODES));
          expect(sorted(Object.keys(manifest.denoisers))).toEqual(sorted(VALID_DENOISERS));

          const expected: Readonly<Record<EngineDenoiserMode, string>> = {
            none: 'native',
            auto: 'native',
            atrous: 'native',
            'atrous-variance': 'native',
            'svgf-real': 'native',
            bmfr: tier === 'full' ? 'native' : 'unsupported',
            neural:
              tier === 'full' && neuralCertification === 'certified' ? 'native' : 'unsupported',
            'oidn-final': oidnModelAvailable ? 'native' : 'unsupported',
          };
          expect(manifest.denoisers).toEqual(expected);
        }
      }
    }
  });

  it('derives live coarse capabilities and strict scene acceptance from one manifest', () => {
    const profile = {
      tier: 'full',
      neuralCertification: 'absent',
      oidnModelAvailable: false,
    } as const;
    const manifest = walkaroundSupportManifest(profile);
    const sets = supportSetsFromManifest(manifest);
    const engine = new HybridEngine(makeOpts());
    try {
      expect(engine.capabilities.supportDetails).toBe(manifest);
      expect(engine.capabilities.supportedPrimitiveKinds).toEqual(sets.supportedPrimitiveKinds);
      expect(engine.capabilities.supportedEmitterKinds).toEqual(sets.supportedEmitterKinds);
      expect(engine.capabilities.supportedEnvironmentKinds).toEqual(sets.supportedEnvironmentKinds);
      expect(engine.capabilities.supportedAnalyticShapes).toEqual(sets.supportedAnalyticShapes);
      expect(engine.capabilities.supportDetails?.mutations).toBe(manifest.mutations);
      expect(engine.capabilities.incrementalPatchSupport).toEqual({
        transform: manifest.mutations.transform !== 'unsupported',
        positions: manifest.mutations.positions !== 'unsupported',
        material: manifest.mutations.material !== 'unsupported',
        emitter: manifest.mutations.emitter !== 'unsupported',
        topology: manifest.mutations.topology !== 'unsupported',
      });
      expect(engine.capabilities.supportsAuxBuffers)
        .toBe(manifest.motionVectors != null);
    } finally {
      engine.dispose();
    }
  });

  it('publishes exact lite and host-OIDN profiles on live instances', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const lite = new HybridEngine(makeOpts({ tier: 'lite' }));
    const oidn = new HybridEngine(
      makeOpts({
        extensions: {
          'walkaround-hybrid': {
            oidnModelUrl: '/models/oidn.onnx',
          },
        },
      }),
    );
    try {
      expect(lite.capabilities.supportDetails).toBe(
        walkaroundSupportManifest({
          tier: 'lite',
          neuralCertification: 'absent',
          oidnModelAvailable: false,
        }),
      );
      expect(oidn.capabilities.supportDetails).toBe(
        walkaroundSupportManifest({
          tier: 'full',
          neuralCertification: 'absent',
          oidnModelAvailable: true,
        }),
      );
    } finally {
      lite.dispose();
      oidn.dispose();
      warn.mockRestore();
    }
  });

  it('keeps the static fully provisioned ledger profile synchronized', () => {
    const expected = BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportDetails;
    expect(WALKAROUND_FULL_CERTIFIED_SUPPORT_MANIFEST).toEqual({
      ...expected,
      denoiserSpatialShapeRequirements: {
        ...expected.denoiserSpatialShapeRequirements,
        neural: WALKAROUND_NEURAL_DENOISER_SHAPE_REQUIREMENT,
      },
    });
  });

  it('keeps BACKEND_PROMISE_LEDGER out of every production runtime source', () => {
    const srcDirectory = fileURLToPath(new URL('../', import.meta.url)).replaceAll('\\', '/');
    const offenders = productionTypeScriptFiles(srcDirectory).filter((path) =>
      readFileSync(path, 'utf8').includes('BACKEND_PROMISE_LEDGER'),
    );
    expect(offenders).toEqual([]);
  });
});
