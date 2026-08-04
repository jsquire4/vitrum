/**
 * nrcStructuralGate.test.ts — the GRIS-class regression guard for the NRC opt-in.
 *
 * The GRIS black-frame bug (f8df9a4) shipped because an opt-in feature
 * (grisReuse) added a `@group(1)` bind group + a two-group pipeline layout
 * to the GI passes UNCONDITIONALLY, gated only by a runtime UBO flag — which
 * STRUCTURALLY altered the DEFAULT pipeline and regressed the default render to
 * an all-black frame on real GPUs. Unit tests did not catch it because it was a
 * pipeline-STRUCTURE failure, not a numeric one. `giStructuralGate.test.ts`
 * pins the fix for GRIS; THIS test pins the SAME discipline for NRC.
 *
 * NRC (Müller et al. 2021) extends group(3), when ON, with two packed arenas
 * (immutable inference + mutable runtime) and a config UBO, plus the
 * inline-MLP-forward shader variant in the gi-ris pass. This test asserts
 * at two layers that turning NRC OFF (the default) is byte-for-byte the pre-NRC
 * pipeline:
 *
 *   1. SHADER STRUCTURE — composing the gi-ris WGSL with nrcConfig ABSENT yields
 *      text with NONE of the NRC identifiers (the packed NRC arena bindings,
 *      names, the inline-MLP-forward fn, the cache-query / record-write fns,
 *      the spread-termination predicate). With nrcConfig PRESENT those ARE
 *      present.
 *
 *   2. PIPELINE LAYOUT + REGISTERED-PASS SET — running the REAL `compilePipelines`
 *      against a recording mock device, the `risGi` compute pipeline uses a
 *      pipeline layout with exactly FOUR bind-group layouts both OFF and ON.
 *      The ON group(3) layout extends the ordinary bindings 0,1,3..6 with NRC
 *      bindings 7..9, and the SET of compiled pipeline labels is IDENTICAL —
 *      no NRC-only pipeline is added and no pass is removed.
 */

import { describe, expect, it } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
import { composeWgsl } from '../src/pipeline/wgslComposer.js';
import { RIS_GI_MODULE } from '../src/shaders/risGi.wgsl.js';
import { buildRisGiNrcModule, type RisGiNrcConfig } from '../src/shaders/risGiNrc.wgsl.js';
import { WGSL_MODULES } from '../src/pipeline/wgslModules.js';
import { compilePipelines } from '../src/pipeline/pipelineCompiler.js';
import type { BGLCache } from '../src/pipeline/bindGroupLayouts.js';
import {
  assertNrcDeviceCapable,
  NRC_REQUIRED_MAX_BIND_GROUPS,
  NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
  NRC_REQUIRED_WORKGROUP_STORAGE_BYTES,
} from '../src/pipeline/WalkaroundGPUPipeline.js';

installWebGPUPolyfills();

// A small, valid NRC config to compose the ON variant against.
const NRC_CFG: RisGiNrcConfig = {
  levels: 8,
  featuresPerEntry: 2,
  oneBlobBins: 8,
  width: 64,
  outWidth: 3,
  hidden: 6,
};

// NRC identifiers that MUST NOT appear in the default (OFF) gi-ris pass — the
// packed group(3) bindings + the inline-MLP/query/record/spread symbols that the
// NRC variant introduces.
const NRC_IDENTS = [
  '@group(3) @binding(7) var<storage, read> nrcInferenceArena',
  '@group(3) @binding(8) var<storage, read_write> nrcRuntimeArena',
  '@group(3) @binding(9) var<uniform> nrcCfg',
  'nrcInferenceArena',
  'nrcRuntimeArena',
  'nrcCfg',              // the encoding-config UBO
  'nrcMlpForward',       // the inline single-sample MLP forward
  'nrcQueryRadiance',    // the cache query
  'nrcWriteRecord',      // the self-training record write
  'nrcShouldTerminateIntoCache', // the spread-termination predicate
] as const;

describe('gi-ris SHADER structure — default (nrcEnabled OFF) is pre-NRC', () => {
  it('OFF gi-ris composes with NO NRC bindings or identifiers', () => {
    const off = composeWgsl(RIS_GI_MODULE, WGSL_MODULES);
    for (const ident of NRC_IDENTS) {
      expect(off, `default gi-ris must NOT contain '${ident}'`).not.toContain(ident);
    }
    // It IS the verbatim DDGI-estimate pass (the known-good default).
    expect(off).toContain('fn risGiMain(');
    expect(off).toContain('sampleDDGIAtPoint(');
  });

  it('ON gi-ris composes WITH packed group(3) NRC bindings and identifiers', () => {
    const on = composeWgsl(buildRisGiNrcModule(NRC_CFG), WGSL_MODULES);
    for (const ident of NRC_IDENTS) {
      expect(on, `NRC gi-ris MUST contain '${ident}'`).toContain(ident);
    }
    // It is STILL the gi-ris pass (same entry point) — a superset, not a fork.
    expect(on).toContain('fn risGiMain(');
    expect(on).toContain('sampleDDGIAtPoint(');
  });
});

// ── Recording mock GPUDevice (mirrors giStructuralGate.test.ts) ──────────────
interface RecordedPipeline {
  label: string;
  bglCount: number;
  bglBindings: number[][];
}

function recordingDevice(recorded: RecordedPipeline[]): GPUDevice {
  const makeShaderModule = () => ({
    getCompilationInfo: async () => ({ messages: [] as GPUCompilationMessage[] }),
  });
  const dev: Record<string, unknown> = {
    createShaderModule: () => makeShaderModule(),
    createBindGroupLayout: (desc: { entries?: Array<{ binding: number }> }) =>
      ({ __bindings: (desc.entries ?? []).map(({ binding }) => binding) }),
    createPipelineLayout: (desc: {
      bindGroupLayouts: Array<{ __bindings?: number[] }>;
    }) => ({
      __bglCount: desc.bindGroupLayouts.length,
      __bglBindings: desc.bindGroupLayouts.map((layout) => layout.__bindings ?? []),
    }),
    createComputePipelineAsync: async (desc: {
      label?: string;
      layout: { __bglCount?: number; __bglBindings?: number[][] } | string;
    }) => {
      const bglCount =
        typeof desc.layout === 'object' && desc.layout && '__bglCount' in desc.layout
          ? (desc.layout.__bglCount ?? -1)
          : -1; // layout: 'auto' (PPG) — not under test
      const bglBindings =
        typeof desc.layout === 'object' && desc.layout
          ? (desc.layout.__bglBindings ?? [])
          : [];
      recorded.push({ label: desc.label ?? '<unlabeled>', bglCount, bglBindings });
      return {};
    },
    createRenderPipelineAsync: async () => ({}),
  };
  return dev as unknown as GPUDevice;
}

async function compileAndCollect(nrcOn: boolean): Promise<RecordedPipeline[]> {
  const recorded: RecordedPipeline[] = [];
  const device = recordingDevice(recorded);
  const bglCache: BGLCache = {} as BGLCache;
  await compilePipelines(device, bglCache, 'bgra8unorm', {
    ...(nrcOn ? { nrcConfig: NRC_CFG } : {}),
  });
  return recorded;
}

describe('gi-ris PIPELINE LAYOUT + registered-pass set — gated at compile time', () => {
  it('default (nrcEnabled OFF): risGi uses a FOUR-group layout', async () => {
    const recorded = await compileAndCollect(false);
    const byLabel: Record<string, number> = {};
    for (const r of recorded) byLabel[r.label] = r.bglCount;
    expect(byLabel.risGi, 'default risGi pipeline must have 4 bind-group layouts').toBe(4);
  });

  it('opt-in (nrcEnabled ON): risGi stays within a FOUR-group layout', async () => {
    const recorded = await compileAndCollect(true);
    const byLabel: Record<string, number> = {};
    for (const r of recorded) byLabel[r.label] = r.bglCount;
    expect(byLabel.risGi, 'NRC risGi pipeline must remain at 4 bind-group layouts').toBe(4);
  });

  it('NRC extends only risGi group(3); DDGI state reuses the irradiance binding', async () => {
    const off = (await compileAndCollect(false)).find((r) => r.label === 'risGi');
    const on = (await compileAndCollect(true)).find((r) => r.label === 'risGi');
    expect(off?.bglBindings[3]).toEqual([0, 1, 3, 4, 5, 6]);
    expect(on?.bglBindings[3]).toEqual([0, 1, 3, 4, 5, 6, 7, 8, 9]);
  });

  it('the SET of compiled pipelines is IDENTICAL OFF vs ON (no pipeline/pass added or removed)', async () => {
    const off = (await compileAndCollect(false)).map((r) => r.label).sort();
    const on = (await compileAndCollect(true)).map((r) => r.label).sort();
    // The set of pipelines compiled does not change. NRC adds NO new
    // compute/render pipeline to the
    // walkaround pass list — the inline MLP forward runs inside the existing
    // gi-ris pass, and the trainer's pipelines live on the (off-graph)
    // NrcSubsystem, not the walkaround registry.
    expect(on).toEqual(off);
  });

  it('every NON-risGi pipeline has the SAME bind-group count OFF vs ON', async () => {
    const off = await compileAndCollect(false);
    const on = await compileAndCollect(true);
    const offByLabel: Record<string, number> = {};
    for (const r of off) offByLabel[r.label] = r.bglCount;
    for (const r of on) {
      if (r.label === 'risGi') continue; // the intended delta
      expect(r.bglCount, `pipeline '${r.label}' bind-group count must be unchanged by NRC`)
        .toBe(offByLabel[r.label]);
    }
  });
});

// NRC-ON capability gate — packed group(3) remains within four bind groups, while
// GI-RIS storage-buffer stage pressure and fused-MLP workgroup tiles are still
// validated explicitly.
describe('NRC-ON device capability gate', () => {
  it('passes when the device meets all NRC limits', () => {
    expect(() => assertNrcDeviceCapable(
      {
        maxBindGroups: NRC_REQUIRED_MAX_BIND_GROUPS,
        maxStorageBuffersPerShaderStage: NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
        maxComputeWorkgroupStorageSize: NRC_REQUIRED_WORKGROUP_STORAGE_BYTES,
      } as unknown as GPUSupportedLimits,
    )).not.toThrow();
    expect(() => assertNrcDeviceCapable(
      {
        maxBindGroups: 8,
        maxStorageBuffersPerShaderStage: NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
        maxComputeWorkgroupStorageSize: 32768,
      } as unknown as GPUSupportedLimits,
    )).not.toThrow();
  });

  it('accepts the portable four-bind-group floor', () => {
    expect(() => assertNrcDeviceCapable(
      {
        maxBindGroups: 4,
        maxStorageBuffersPerShaderStage: NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
        maxComputeWorkgroupStorageSize: 32768,
      } as unknown as GPUSupportedLimits,
    )).not.toThrow();
  });

  it('rejects a device below the four-bind-group requirement', () => {
    expect(() => assertNrcDeviceCapable(
      {
        maxBindGroups: NRC_REQUIRED_MAX_BIND_GROUPS - 1,
        maxStorageBuffersPerShaderStage: NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
        maxComputeWorkgroupStorageSize: 32768,
      } as unknown as GPUSupportedLimits,
    )).toThrow(/maxBindGroups/);
  });

  it('accepts the guaranteed 16 KiB workgroup floor used by the actual fused shader', () => {
    expect(() => assertNrcDeviceCapable(
      {
        maxBindGroups: 8,
        maxStorageBuffersPerShaderStage: NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
        maxComputeWorkgroupStorageSize: 16384,
      } as unknown as GPUSupportedLimits,
    )).not.toThrow();
  });

  it('throws when storage-buffer stage budget is below the GI-RIS NRC requirement', () => {
    expect(() => assertNrcDeviceCapable(
      {
        maxBindGroups: 8,
        maxStorageBuffersPerShaderStage: NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE - 1,
        maxComputeWorkgroupStorageSize: 32768,
      } as unknown as GPUSupportedLimits,
    )).toThrow(/maxStorageBuffersPerShaderStage/);
  });
});
