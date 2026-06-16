/**
 * giStructuralGate.test.ts — the regression guard that was MISSING when the
 * GRIS black-frame bug (f8df9a4) shipped.
 *
 * Root cause of f8df9a4: the GRIS reconnection-shift reuse (opt-in via
 * `restirPtReuse`) changed the GI spatial AND temporal passes UNCONDITIONALLY,
 * gated only by a runtime `ubo` check. The default path inherited a pipeline
 * structure and shader branch it could not safely execute, regressing the
 * default walkaround render to an all-black frame on real GPUs (Mesa dzn) AND
 * the lavapipe oracle. Neither the unit tests nor the static wgslCompose check
 * caught it: it was a GPU-runtime / pipeline-structure failure.
 *
 * This test pins the fix: an opt-in feature must NOT change the default pipeline
 * structure at all. It asserts at two layers:
 *
 *   1. SHADER STRUCTURE — composing the spatial + temporal GI WGSL with
 *      restirPtReuse OFF yields text with NONE of the GRIS identifiers
 *      (reconnection-visibility fn / grisReuse symbols). With ON, those ARE
 *      present. Both variants now bind `@group(1)` for receiver-lobe p-hat
 *      material recasts.
 *
 *   2. PIPELINE LAYOUT — running the REAL `compilePipelines` against a recording
 *      mock device, the `temporalGi` + `spatialGi` compute pipelines use a
 *      two-group layout for both OFF and ON, because receiver-lobe p-hat recasts
 *      need the scene/material group in the default path too.
 */

import { describe, expect, it } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
import { composeWgsl } from '../src/pipeline/wgslComposer.js';
import {
  SPATIAL_GI_MODULE,
  SPATIAL_GI_GRIS_MODULE,
} from '../src/shaders/spatialGi.wgsl.js';
import {
  TEMPORAL_GI_MODULE,
  TEMPORAL_GI_GRIS_MODULE,
} from '../src/shaders/temporalGi.wgsl.js';
import { WGSL_MODULES } from '../src/pipeline/wgslModules.js';
import { compilePipelines } from '../src/pipeline/pipelineCompiler.js';
import type { BGLCache } from '../src/pipeline/bindGroupLayouts.js';

// compilePipelines → getLightTreeBindGroupLayout reads GPUShaderStage.COMPUTE;
// install the WebGPU constant globals before any of that runs.
installWebGPUPolyfills();

// GRIS identifiers that MUST NOT appear in the default (OFF) GI passes — these
// are the branch/math symbols the f8df9a4 commit leaked onto the default path.
const SPATIAL_GRIS_IDENTS = [
  'grisReconnectionVisible',  // the visibility-trace fn
  'grisShiftJacobian',        // grisReuse shift symbol
  'grisPairwiseDenomNeighbor',// grisReuse pairwise-MIS symbol
] as const;

const TEMPORAL_GRIS_IDENTS = [
  'tgiReconnectionVisible',   // the visibility-trace fn
  'grisShiftJacobian',
  'grisPairwiseDenomNeighbor',
] as const;

describe('GI pass SHADER structure — default (restirPtReuse OFF) has no GRIS branch', () => {
  it('spatialGi OFF composes with receiver-lobe group(1) but NO GRIS identifiers', () => {
    const off = composeWgsl(SPATIAL_GI_MODULE, WGSL_MODULES);
    expect(off).toContain('@group(1)');
    expect(off).toContain('restir_gi_receiver_phat_from_surface_or_geometry(');
    for (const ident of SPATIAL_GRIS_IDENTS) {
      expect(off, `default spatialGi must NOT contain '${ident}'`).not.toContain(ident);
    }
    // It IS the legacy reuse (the known-good default).
    expect(off).toContain('jacobianReconnectionShift(');
    expect(off).toContain('fn spatialGiMain(');
  });

  it('temporalGi OFF composes with receiver-lobe group(1) but NO GRIS identifiers', () => {
    const off = composeWgsl(TEMPORAL_GI_MODULE, WGSL_MODULES);
    expect(off).toContain('@group(1)');
    expect(off).toContain('restir_gi_receiver_phat_from_surface_or_geometry(');
    for (const ident of TEMPORAL_GRIS_IDENTS) {
      expect(off, `default temporalGi must NOT contain '${ident}'`).not.toContain(ident);
    }
    expect(off).toContain('jacobianReconnectionShift(');
    expect(off).toContain('fn temporalGiMain(');
  });

  it('spatialGi ON composes WITH @group(1) and the GRIS identifiers', () => {
    const on = composeWgsl(SPATIAL_GI_GRIS_MODULE, WGSL_MODULES);
    expect(on).toContain('@group(1)');
    for (const ident of SPATIAL_GRIS_IDENTS) {
      expect(on, `GRIS spatialGi MUST contain '${ident}'`).toContain(ident);
    }
  });

  it('temporalGi ON composes WITH @group(1) and the GRIS identifiers', () => {
    const on = composeWgsl(TEMPORAL_GI_GRIS_MODULE, WGSL_MODULES);
    expect(on).toContain('@group(1)');
    for (const ident of TEMPORAL_GRIS_IDENTS) {
      expect(on, `GRIS temporalGi MUST contain '${ident}'`).toContain(ident);
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

async function compileAndFind(restirPtReuse: boolean): Promise<Record<string, number>> {
  const recorded: RecordedPipeline[] = [];
  const device = recordingDevice(recorded);
  const bglCache: BGLCache = {} as BGLCache;
  await compilePipelines(device, bglCache, 'bgra8unorm', { restirPtReuse });
  const byLabel: Record<string, number> = {};
  for (const r of recorded) byLabel[r.label] = r.bglCount;
  return byLabel;
}

async function compileAndRecordShaders(
  restirPtReuse: boolean,
  opts: { ppgEnabled?: boolean } = {},
): Promise<RecordedShader[]> {
  const recorded: RecordedPipeline[] = [];
  const shaders: RecordedShader[] = [];
  const device = recordingDevice(recorded, shaders);
  const bglCache: BGLCache = {} as BGLCache;
  await compilePipelines(device, bglCache, 'bgra8unorm', { restirPtReuse, ...opts });
  return shaders;
}

describe('GI pass PIPELINE LAYOUT — group count gated at compile time', () => {
  it('default (restirPtReuse OFF): temporalGi + spatialGi use the receiver-material two-group layout', async () => {
    const byLabel = await compileAndFind(false);
    expect(byLabel.temporalGi, 'default temporalGi pipeline must have 2 bind-group layouts').toBe(2);
    expect(byLabel.spatialGi, 'default spatialGi pipeline must have 2 bind-group layouts').toBe(2);
  });

  it('opt-in (restirPtReuse ON): temporalGi + spatialGi use a TWO-group layout', async () => {
    const byLabel = await compileAndFind(true);
    expect(byLabel.temporalGi, 'GRIS temporalGi pipeline must have 2 bind-group layouts').toBe(2);
    expect(byLabel.spatialGi, 'GRIS spatialGi pipeline must have 2 bind-group layouts').toBe(2);
  });
});

describe('H24 GI reservoir stride — shader source follows the structural gate', () => {
  it('default compile path uses the compact 20-u32 reservoir module', async () => {
    const shaders = await compileAndRecordShaders(false);
    const risGi = shaders.find((s) => s.label === 'risGi')!.code;
    expect(risGi).toContain('const RESERVOIR_GI_STRIDE: u32 = 20u;');
    expect(risGi).toContain('Compact default layout: no appended GRIS cache stores.');
    expect(risGi).not.toContain('buf[b + 29u] = r._padPT2;');
  });

  it('GRIS compile path uses the widened 30-u32 reservoir module', async () => {
    const shaders = await compileAndRecordShaders(true);
    const risGi = shaders.find((s) => s.label === 'risGi')!.code;
    expect(risGi).toContain('const RESERVOIR_GI_STRIDE: u32 = 30u;');
    expect(risGi).toContain('buf[b + 29u] = r._padPT2;');
  });

  it('PPG update bakes the same reservoir stride as the GI reservoir module', async () => {
    const offShaders = await compileAndRecordShaders(false, { ppgEnabled: true });
    const onShaders = await compileAndRecordShaders(true, { ppgEnabled: true });

    const offPpg = offShaders.find((s) => s.label === 'ppg-update')!.code;
    const onPpg = onShaders.find((s) => s.label === 'ppg-update')!.code;
    expect(offPpg).toMatch(/RESERVOIR_GI_STRIDE_LOCAL\s*:\s*u32\s*=\s*20u/);
    expect(onPpg).toMatch(/RESERVOIR_GI_STRIDE_LOCAL\s*:\s*u32\s*=\s*30u/);
  });
});
