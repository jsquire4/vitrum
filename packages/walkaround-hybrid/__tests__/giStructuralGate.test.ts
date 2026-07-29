/**
 * giStructuralGate.test.ts — the regression guard that was MISSING when the
 * GRIS black-frame bug (f8df9a4) shipped.
 *
 * Root cause of f8df9a4: the first GRIS reconnection-shift implementation
 * changed the GI spatial AND temporal passes behind a runtime-only switch.
 * The supposedly compact default path inherited a pipeline structure and
 * shader branch it could not safely execute, regressing the default walkaround
 * render to an all-black frame on real GPUs (Mesa dzn) AND the lavapipe oracle.
 * Neither the unit tests nor the static wgslCompose check caught it: it was a
 * GPU-runtime / pipeline-structure failure.
 *
 * Generalized reuse is now the sole live path, so this test pins the stronger
 * invariant that the sole canonical path compiles the complete generalized
 * structure. It asserts at two layers:
 *
 *   1. SHADER STRUCTURE — composing the spatial + temporal GI WGSL with
 *      the canonical spatial + temporal GI modules includes the reconnection
 *      visibility, Jacobian, and full-technique MIS implementation.
 *
 *   2. PIPELINE LAYOUT — running the REAL `compilePipelines` against a recording
 *      mock device, the `temporalGi` + `spatialGi` compute pipelines use a
 *      two-group layout for both omitted and deprecated-true inputs because
 *      receiver-lobe p-hat recasts need the scene/material group.
 */

import { describe, expect, it } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
import { composeWgsl } from '../src/pipeline/wgslComposer.js';
import { SPATIAL_GI_MODULE } from '../src/shaders/spatialGi.wgsl.js';
import { TEMPORAL_GI_MODULE } from '../src/shaders/temporalGi.wgsl.js';
import type { RisGiNrcConfig } from '../src/shaders/risGiNrc.wgsl.js';
import { WGSL_MODULES } from '../src/pipeline/wgslModules.js';
import { compilePipelines } from '../src/pipeline/pipelineCompiler.js';
import type { BGLCache } from '../src/pipeline/bindGroupLayouts.js';

// compilePipelines → getLightTreeBindGroupLayout reads GPUShaderStage.COMPUTE;
// install the WebGPU constant globals before any of that runs.
installWebGPUPolyfills();

// Core generalized-reuse identifiers that must remain in the sole live path.
const SPATIAL_GRIS_IDENTS = [
  'grisProxyVisibilityAt',
  'grisDomainToCanonicalJacobian',
  'grisLogWeightedTransformedDensity',
] as const;

const TEMPORAL_GRIS_IDENTS = [
  'grisProxyVisibilityAt',   // the visibility-trace fn
  'grisDomainToCanonicalJacobian',
  'grisLogWeightedTransformedDensity',
] as const;

describe('GI pass shader structure — generalized reuse is the stable path', () => {
  it('spatialGi composes with the complete generalized estimator', () => {
    const source = composeWgsl(SPATIAL_GI_MODULE, WGSL_MODULES);
    expect(source).toContain('@group(1)');
    for (const ident of SPATIAL_GRIS_IDENTS) {
      expect(source, `GRIS spatialGi MUST contain '${ident}'`).toContain(ident);
    }
  });

  it('temporalGi composes with the complete generalized estimator', () => {
    const source = composeWgsl(TEMPORAL_GI_MODULE, WGSL_MODULES);
    expect(source).toContain('@group(1)');
    for (const ident of TEMPORAL_GRIS_IDENTS) {
      expect(source, `GRIS temporalGi MUST contain '${ident}'`).toContain(ident);
    }
  });
});

// ── Recording mock GPUDevice ─────────────────────────────────────────────────
// compilePipelines only touches createShaderModule / createBindGroupLayout /
// createPipelineLayout / create{Compute,Render}PipelineAsync. Each pipeline
// layout object carries `__bglCount` (the number of bind-group layouts it was
// built from); each compute pipeline records its label + the layout's
// `__bglCount`, so the test can read back exactly how many groups the
// temporalGi / spatialGi pipelines were given.
interface RecordedPipeline {
  label: string;
  bglCount: number;
}

interface RecordedShader {
  label: string;
  code: string;
}

function recordingDevice(recorded: RecordedPipeline[], shaders: RecordedShader[] = []): GPUDevice {
  const makeShaderModule = () => ({
    // No compile errors / warnings — exercises the real success path.
    getCompilationInfo: async () => ({ messages: [] as GPUCompilationMessage[] }),
  });
  const dev: Record<string, unknown> = {
    createShaderModule: (desc: { label?: string; code: string }) => {
      shaders.push({ label: desc.label ?? '<unlabeled>', code: desc.code });
      return makeShaderModule();
    },
    createBindGroupLayout: () => ({}),
    createPipelineLayout: (desc: { bindGroupLayouts: unknown[] }) =>
      ({ __bglCount: desc.bindGroupLayouts.length }),
    createComputePipelineAsync: async (desc: {
      label?: string;
      layout: { __bglCount?: number } | string;
    }) => {
      const bglCount =
        typeof desc.layout === 'object' && desc.layout && '__bglCount' in desc.layout
          ? (desc.layout.__bglCount ?? -1)
          : -1; // layout: 'auto' (PPG) — not under test
      recorded.push({ label: desc.label ?? '<unlabeled>', bglCount });
      return {};
    },
    createRenderPipelineAsync: async () => ({}),
  };
  return dev as unknown as GPUDevice;
}

async function compileAndFind(): Promise<Record<string, number>> {
  const recorded: RecordedPipeline[] = [];
  const device = recordingDevice(recorded);
  const bglCache: BGLCache = {} as BGLCache;
  await compilePipelines(device, bglCache, 'bgra8unorm');
  const byLabel: Record<string, number> = {};
  for (const r of recorded) byLabel[r.label] = r.bglCount;
  return byLabel;
}

async function compileAndRecordShaders(
  opts: { ppgEnabled?: boolean; nrcConfig?: RisGiNrcConfig } = {},
): Promise<RecordedShader[]> {
  const recorded: RecordedPipeline[] = [];
  const shaders: RecordedShader[] = [];
  const device = recordingDevice(recorded, shaders);
  const bglCache: BGLCache = {} as BGLCache;
  await compilePipelines(device, bglCache, 'bgra8unorm', opts);
  return shaders;
}

describe('GI pass PIPELINE LAYOUT — one generalized structural path', () => {
  it('default: temporalGi + spatialGi use the receiver-material two-group layout', async () => {
    const byLabel = await compileAndFind();
    expect(byLabel.temporalGi, 'default temporalGi pipeline must have 2 bind-group layouts').toBe(2);
    expect(byLabel.spatialGi, 'default spatialGi pipeline must have 2 bind-group layouts').toBe(2);
  });

});

describe('GI reservoir ABI — the live shader graph is always 28 u32', () => {
  it('default compile path uses the complete generalized reservoir module', async () => {
    const shaders = await compileAndRecordShaders();
    const risGi = shaders.find((s) => s.label === 'risGi')!.code;
    expect(risGi).toContain('const RESERVOIR_GI_STRIDE: u32 = 28u;');
    expect(risGi).toContain('words[27u] = r.historyEpoch;');
  });

  it('PPG update bakes the same reservoir stride as the GI reservoir module', async () => {
    const shaders = await compileAndRecordShaders({ ppgEnabled: true });
    const ppg = shaders.find((s) => s.label === 'ppg-update')!.code;
    expect(ppg).toMatch(/RESERVOIR_GI_STRIDE_LOCAL\s*:\s*u32\s*=\s*28u/);
  });
});

const NRC_CFG: RisGiNrcConfig = {
  levels: 8,
  featuresPerEntry: 2,
  oneBlobBins: 8,
  width: 64,
  outWidth: 3,
  hidden: 6,
};

describe('GI estimator combinations — no feature silently bypasses generalized reuse', () => {
  it.each([
    ['plain', {}],
    ['PPG', { ppgEnabled: true }],
    ['NRC', { nrcConfig: NRC_CFG }],
    ['PPG + NRC', { ppgEnabled: true, nrcConfig: NRC_CFG }],
  ] as const)('%s compiles the generalized temporal and spatial passes', async (_label, opts) => {
    const shaders = await compileAndRecordShaders(opts);
    const temporal = shaders.find((s) => s.label === 'temporalGi')!.code;
    const spatial = shaders.find((s) => s.label === 'spatialGi')!.code;
    expect(temporal).toContain('grisLogWeightedTransformedDensity');
    expect(spatial).toContain('grisLogWeightedTransformedDensity');
  });

  it('PPG + NRC keeps both proposal guidance and cache substitution in gi-ris', async () => {
    const shaders = await compileAndRecordShaders({
      ppgEnabled: true,
      nrcConfig: NRC_CFG,
    });
    const risGi = shaders.find((s) => s.label === 'risGi')!.code;
    expect(risGi).toContain('ppgGuidedOn = ubo.ppgEnabled == 1u');
    expect(risGi).toContain('nrcCanSubstitute');
    expect(risGi).not.toContain('nrcCanSubstitute && !grisOn');
    expect(shaders.some((s) => s.label === 'ppg-update')).toBe(true);
  });
});
