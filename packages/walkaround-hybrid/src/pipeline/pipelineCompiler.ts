/**
 * Pipeline compiler — creates all GPUComputePipeline and GPURenderPipeline
 * objects used by WalkaroundGPUPipeline EXCEPT the denoiser-specific ones
 * (those are owned by each {@link Denoiser} implementation in
 * `pipeline/denoisers/`).
 *
 * Called once from `initialize()`. Compiles all shader modules in parallel,
 * checks for compile errors, then creates pipeline layouts and dispatches
 * pipeline creation with `createComputePipelineAsync` / `createRenderPipelineAsync`.
 *
 * The shader source for every module is composed via the declarative WGSL
 * include-graph: each `*_MODULE` declares its `requires` array, and
 * `composeWgsl()` topo-sorts the closure and emits each dep exactly once.
 * Pre-R6 this file held nine hand-rolled `COMMON_WGSL + X_WGSL` concat
 * patterns plus an anti-duplication-by-comment explaining why
 * `ATROUS_VARIANCE_WGSL` was self-contained — both are gone (the
 * self-contained-ness is now structural: `ATROUS_VARIANCE_MODULE.requires`
 * is `[]`).
 *
 * W1-R3 (2026-05-17) — denoiser-conditional shader compiles + pipeline
 * creation (welford, atrous-variance, svgf-real reproj/moments/7×7/atrous)
 * moved into the corresponding {@link Denoiser.initialize} entries. This
 * module no longer branches on `denoiserMode` for anything — it compiles
 * the always-on RIS / temporal / spatial / shade / GTAO / GI / indirect /
 * adaptive-sampling / composite pipelines plus the shared `atrousPipeline`
 * the legacy à-trous denoiser AND the always-on indirect chain both
 * dispatch.
 */

import { composeWgsl } from './wgslComposer.js';
import {
  ATROUS_MODULE,
  COMPOSITE_FRAG_MODULE,
  COMPOSITE_VERT_MODULE,
  GTAO_MODULE,
  GTAO_UPSAMPLE_MODULE,
  INDIRECT_COMBINE_MODULE,
  INDIRECT_TEMPORAL_ACCUM_MODULE,
  PPG_GUIDE_MODULE,
  PPG_UPDATE_MODULE,
  RESOLVE_MODULE,
  RIS_GI_MODULE,
  RIS_MODULE,
  SAMPLE_BUDGET_MODULE,
  SHADE_MODULE,
  SPATIAL_GI_MODULE,
  SPATIAL_MODULE,
  TEMPORAL_ACCUM_MODULE,
  TEMPORAL_GI_MODULE,
  TEMPORAL_MODULE,
  WGSL_MODULES,
} from './wgslModules.js';
import {
  getFrameBindGroupLayout,
  getSceneBindGroupLayout,
  getUboBindGroupLayout,
  getAtrousBindGroupLayout,
  getAccumBindGroupLayout,
  getCompositeBindGroupLayout,
  getHybridLayersBindGroupLayout,
  getSampleBudgetBindGroupLayout,
  getResolveBindGroupLayout,
  getGTAOBindGroupLayout,
  getGTAOUpsampleBindGroupLayout,
  getTemporalGiBindGroupLayout,
  getSpatialGiBindGroupLayout,
  getIndirectCombineBindGroupLayout,
  getIndirectTemporalAccumBindGroupLayout,
  type BGLCache,
} from './bindGroupLayouts.js';

export interface CompiledPipelines {
  risPipeline: GPUComputePipeline;
  /** T2.H3 — PPG update kernel (training on L_i, Müller §3.3). */
  ppgUpdatePipeline?: GPUComputePipeline;
  /** T2.H3 — PPG guide kernel (dTree direction sampling + MIS PDF, Müller §3.2, §3.4). */
  ppgGuidePipeline?: GPUComputePipeline;
  temporalPipeline: GPUComputePipeline;
  spatialPipeline: GPUComputePipeline;
  shadePipeline: GPUComputePipeline;
  /** Shared à-trous pipeline — used by the legacy `AtrousDenoiser` AND by
   *  the always-on indirect-channel chain
   *  (`WalkaroundGPUPipeline._dispatchAtrousIndirect`). */
  atrousPipeline: GPUComputePipeline;
  accumPipeline: GPUComputePipeline;
  compositePipeline: GPURenderPipeline;
  /** Sprint 9 — adaptive sampling tier classifier (runs before RIS). */
  sampleBudgetPipeline: GPUComputePipeline;
  /** Sprint 9 — resolve pass (runs between temporalAccum and composite). */
  resolvePipeline: GPUComputePipeline;
  /** Sprint 15 — half-res GTAO compute pass. */
  gtaoPipeline: GPUComputePipeline;
  /** Sprint 15 — bilateral upsample from half-res AO to full-res. */
  gtaoUpsamplePipeline: GPUComputePipeline;
  /** Sprint 16 — ReSTIR-GI RIS pass (half-res). */
  risGiPipeline: GPUComputePipeline;
  /** Sprint 17 — GI temporal-reuse pass. */
  temporalGiPipeline: GPUComputePipeline;
  /** Sprint 17 — GI spatial-reuse pass (run twice with ping-pong). */
  spatialGiPipeline: GPUComputePipeline;
  /** Sprint 18 — indirect-blur + combine pass. */
  indirectCombinePipeline: GPUComputePipeline;
  /** Sprint 18 follow-up — pre-atrous temporal accumulator on indirect. */
  indirectTemporalAccumPipeline: GPUComputePipeline;
}

export async function compilePipelines(
  device: GPUDevice,
  bglCache: BGLCache,
  swapChainFormat: GPUTextureFormat,
  opts?: { verbose?: boolean; ppgEnabled?: boolean },
): Promise<CompiledPipelines> {
  // Compile all shader modules. The include-graph (composeWgsl + WGSL_MODULES)
  // resolves each module's dependency closure exactly once — no hand-rolled
  // `COMMON_WGSL + X_WGSL` concat patterns remain.
  const risSM = device.createShaderModule({
    label: 'ris',
    code: composeWgsl(RIS_MODULE, WGSL_MODULES),
  });
  const temporalSM = device.createShaderModule({
    label: 'temporal',
    code: composeWgsl(TEMPORAL_MODULE, WGSL_MODULES),
  });
  const spatialSM = device.createShaderModule({
    label: 'spatial',
    code: composeWgsl(SPATIAL_MODULE, WGSL_MODULES),
  });
  const shadeSM = device.createShaderModule({
    label: 'shade',
    code: composeWgsl(SHADE_MODULE, WGSL_MODULES),
  });
  const atrousSM = device.createShaderModule({
    label: 'atrous',
    code: composeWgsl(ATROUS_MODULE, WGSL_MODULES),
  });
  const compVertSM = device.createShaderModule({
    label: 'comp-vert',
    code: composeWgsl(COMPOSITE_VERT_MODULE, WGSL_MODULES),
  });
  const compFragSM = device.createShaderModule({
    label: 'comp-frag',
    code: composeWgsl(COMPOSITE_FRAG_MODULE, WGSL_MODULES),
  });

  // Sprint 9 — sample-budget and resolve are standalone compute shaders.
  // sampleBudget.wgsl template-interpolates WELFORD_VARIANCE_WGSL from
  // @vitrum/shared-denoisers into its own source; resolve.wgsl is
  // self-contained. Both modules declare `requires: []`.
  const sampleBudgetSM = device.createShaderModule({
    label: 'sample-budget',
    code: composeWgsl(SAMPLE_BUDGET_MODULE, WGSL_MODULES),
  });
  const resolveSM = device.createShaderModule({
    label: 'resolve',
    code: composeWgsl(RESOLVE_MODULE, WGSL_MODULES),
  });

  // Check for compile errors on every shader module before proceeding.
  const modules: [string, GPUShaderModule][] = [
    ['ris', risSM],
    ['temporal', temporalSM],
    ['spatial', spatialSM],
    ['shade', shadeSM],
    ['atrous', atrousSM],
    ['comp-vert', compVertSM],
    ['comp-frag', compFragSM],
    ['sample-budget', sampleBudgetSM],
    ['resolve', resolveSM],
  ];
  for (const [label, sm] of modules) {
    const info = await sm.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length > 0) {
      console.error(
        `[ReSTIR] Shader compile errors in '${label}':`,
        errors.map((e) => `line ${e.lineNum}: ${e.message}`),
      );
      throw new Error(
        `[ReSTIR] Shader compile error in '${label}': ${errors[0]!.message} (line ${errors[0]!.lineNum})`,
      );
    }
    const warns = info.messages.filter((m) => m.type === 'warning');
    if (warns.length > 0) {
      console.warn(
        `[ReSTIR] Shader warnings in '${label}':`,
        warns.map((w) => w.message),
      );
    }
  }

  // Pipeline layouts.
  // - computeLayout: shared by RIS, temporal, spatial. 3 bind groups.
  // - shadeLayout: adds DDGI as 4th bind group. Used only by the shade
  //   pipeline. RIS/temporal/spatial don't need DDGI inputs.
  const computeLayout = device.createPipelineLayout({
    bindGroupLayouts: [
      getFrameBindGroupLayout(device, bglCache),
      getSceneBindGroupLayout(device, bglCache),
      getUboBindGroupLayout(device, bglCache),
    ],
  });
  const shadeLayout = device.createPipelineLayout({
    bindGroupLayouts: [
      getFrameBindGroupLayout(device, bglCache),
      getSceneBindGroupLayout(device, bglCache),
      getUboBindGroupLayout(device, bglCache),
      getHybridLayersBindGroupLayout(device, bglCache),
    ],
  });
  const atrousLayout = device.createPipelineLayout({
    bindGroupLayouts: [getAtrousBindGroupLayout(device, bglCache)],
  });
  const accumLayout = device.createPipelineLayout({
    bindGroupLayouts: [getAccumBindGroupLayout(device, bglCache)],
  });
  const compositeLayout = device.createPipelineLayout({
    bindGroupLayouts: [getCompositeBindGroupLayout(device, bglCache)],
  });
  const sampleBudgetLayout = device.createPipelineLayout({
    bindGroupLayouts: [getSampleBudgetBindGroupLayout(device, bglCache)],
  });
  const resolveLayout = device.createPipelineLayout({
    bindGroupLayouts: [getResolveBindGroupLayout(device, bglCache)],
  });
  const gtaoLayout = device.createPipelineLayout({
    bindGroupLayouts: [getGTAOBindGroupLayout(device, bglCache)],
  });
  const gtaoUpsampleLayout = device.createPipelineLayout({
    bindGroupLayouts: [getGTAOUpsampleBindGroupLayout(device, bglCache)],
  });
  // Sprint 17 — GI temporal + spatial passes each use a single dedicated
  // bind group at group(0). No frame/scene/ubo groups — these passes are
  // pure reservoir-buffer ops + a small uniform read.
  const temporalGiLayout = device.createPipelineLayout({
    bindGroupLayouts: [getTemporalGiBindGroupLayout(device, bglCache)],
  });
  const spatialGiLayout = device.createPipelineLayout({
    bindGroupLayouts: [getSpatialGiBindGroupLayout(device, bglCache)],
  });
  // Sprint 18 — indirect-combine pass uses a single dedicated bind group.
  const indirectCombineLayout = device.createPipelineLayout({
    bindGroupLayouts: [getIndirectCombineBindGroupLayout(device, bglCache)],
  });
  const indirectTemporalAccumLayout = device.createPipelineLayout({
    bindGroupLayouts: [getIndirectTemporalAccumBindGroupLayout(device, bglCache)],
  });

  // Compile compute pipelines in parallel.
  const [risPipeline, temporalPipeline, spatialPipeline, shadePipeline] = await Promise.all([
    device.createComputePipelineAsync({
      label: 'ris',
      layout: computeLayout,
      compute: { module: risSM, entryPoint: 'risMain' },
    }),
    device.createComputePipelineAsync({
      label: 'temporal',
      layout: computeLayout,
      compute: { module: temporalSM, entryPoint: 'temporalMain' },
    }),
    device.createComputePipelineAsync({
      label: 'spatial',
      layout: computeLayout,
      compute: { module: spatialSM, entryPoint: 'spatialMain' },
    }),
    device.createComputePipelineAsync({
      label: 'shade',
      layout: shadeLayout,
      compute: { module: shadeSM, entryPoint: 'shadeMain' },
    }),
  ]);

  const atrousPipeline = await device.createComputePipelineAsync({
    label: 'atrous',
    layout: atrousLayout,
    compute: { module: atrousSM, entryPoint: 'atrousMain' },
  });

  // Sprint 9 — adaptive sampling pipelines.
  const [sampleBudgetPipeline, resolvePipeline] = await Promise.all([
    device.createComputePipelineAsync({
      label: 'sample-budget',
      layout: sampleBudgetLayout,
      compute: { module: sampleBudgetSM, entryPoint: 'sampleBudgetKernel' },
    }),
    device.createComputePipelineAsync({
      label: 'resolve',
      layout: resolveLayout,
      compute: { module: resolveSM, entryPoint: 'resolveKernel' },
    }),
  ]);

  // Sprint 15 — GTAO pipelines.
  const gtaoSM = device.createShaderModule({
    label: 'gtao',
    code: composeWgsl(GTAO_MODULE, WGSL_MODULES),
  });
  const gtaoUpsampleSM = device.createShaderModule({
    label: 'gtao-upsample',
    code: composeWgsl(GTAO_UPSAMPLE_MODULE, WGSL_MODULES),
  });
  const [gtaoPipeline, gtaoUpsamplePipeline] = await Promise.all([
    device.createComputePipelineAsync({
      label: 'gtao',
      layout: gtaoLayout,
      compute: { module: gtaoSM, entryPoint: 'gtaoMain' },
    }),
    device.createComputePipelineAsync({
      label: 'gtao-upsample',
      layout: gtaoUpsampleLayout,
      compute: { module: gtaoUpsampleSM, entryPoint: 'gtaoUpsampleMain' },
    }),
  ]);

  // Sprint 16 — ReSTIR-GI RIS pipeline. Reuses the existing shadeLayout
  // (frame + scene + ubo + hybrid-layers) so it can re-cast the primary ray
  // (gNormalDepth on group 0), traverse the BVH (group 1), read UBO + AO
  // (group 2), and sample the DDGI atlas (group 3). The reservoir-gi-current
  // storage buffer rides on the frame BGL at binding 11.
  const risGiSM = device.createShaderModule({
    label: 'risGi',
    code: composeWgsl(RIS_GI_MODULE, WGSL_MODULES),
  });
  const risGiPipeline = await device.createComputePipelineAsync({
    label: 'risGi',
    layout: shadeLayout,
    compute: { module: risGiSM, entryPoint: 'risGiMain' },
  });

  // Sprint 17 — GI temporal + spatial reuse pipelines.
  const temporalGiSM = device.createShaderModule({
    label: 'temporalGi',
    code: composeWgsl(TEMPORAL_GI_MODULE, WGSL_MODULES),
  });
  const spatialGiSM = device.createShaderModule({
    label: 'spatialGi',
    code: composeWgsl(SPATIAL_GI_MODULE, WGSL_MODULES),
  });
  const [temporalGiPipeline, spatialGiPipeline] = await Promise.all([
    device.createComputePipelineAsync({
      label: 'temporalGi',
      layout: temporalGiLayout,
      compute: { module: temporalGiSM, entryPoint: 'temporalGiMain' },
    }),
    device.createComputePipelineAsync({
      label: 'spatialGi',
      layout: spatialGiLayout,
      compute: { module: spatialGiSM, entryPoint: 'spatialGiMain' },
    }),
  ]);

  // Sprint 18 — indirect-combine pipeline.
  const indirectCombineSM = device.createShaderModule({
    label: 'indirectCombine',
    code: composeWgsl(INDIRECT_COMBINE_MODULE, WGSL_MODULES),
  });
  const indirectCombinePipeline = await device.createComputePipelineAsync({
    label: 'indirectCombine',
    layout: indirectCombineLayout,
    compute: { module: indirectCombineSM, entryPoint: 'indirectCombineMain' },
  });

  // Sprint 18 follow-up — indirect pre-atrous temporal accumulator.
  const indirectTemporalAccumSM = device.createShaderModule({
    label: 'indirectTemporalAccum',
    code: composeWgsl(INDIRECT_TEMPORAL_ACCUM_MODULE, WGSL_MODULES),
  });
  const indirectTemporalAccumPipeline = await device.createComputePipelineAsync({
    label: 'indirectTemporalAccum',
    layout: indirectTemporalAccumLayout,
    compute: { module: indirectTemporalAccumSM, entryPoint: 'indirectTemporalAccumMain' },
  });

  const accumSM = device.createShaderModule({
    label: 'accum',
    code: composeWgsl(TEMPORAL_ACCUM_MODULE, WGSL_MODULES),
  });
  const accumPipeline = await device.createComputePipelineAsync({
    label: 'temporalAccum',
    layout: accumLayout,
    compute: { module: accumSM, entryPoint: 'temporalAccumMain' },
  });

  // Composite render pipeline.
  const compositePipeline = await device.createRenderPipelineAsync({
    label: 'composite',
    layout: compositeLayout,
    vertex: { module: compVertSM, entryPoint: 'vertMain' },
    fragment: {
      module: compFragSM,
      entryPoint: 'fragMain',
      targets: [{ format: swapChainFormat }],
    },
    primitive: { topology: 'triangle-list' },
  });

  // T2.H3 — PPG pipelines (Müller 2017 §3.2–3.4): opt-in via opts.ppgEnabled.
  // Both kernels use `layout: 'auto'` — their bind group layouts are simple
  // enough that WebGPU can derive them from the shader without a manual
  // PipelineLayout. pipelineCompiler only compiles; bind-group creation and
  // dispatch live in WalkaroundGPUPipeline.renderFrame.
  let ppgUpdatePipeline: GPUComputePipeline | undefined;
  let ppgGuidePipeline: GPUComputePipeline | undefined;
  if (opts?.ppgEnabled) {
    const ppgUpdateSM = device.createShaderModule({
      label: 'ppg-update',
      code: composeWgsl(PPG_UPDATE_MODULE, WGSL_MODULES),
    });
    const ppgGuideSM = device.createShaderModule({
      label: 'ppg-guide',
      code: composeWgsl(PPG_GUIDE_MODULE, WGSL_MODULES),
    });
    for (const [label, sm] of [
      ['ppg-update', ppgUpdateSM],
      ['ppg-guide', ppgGuideSM],
    ] as [string, GPUShaderModule][]) {
      const info = await sm.getCompilationInfo();
      const errs = info.messages.filter((m) => m.type === 'error');
      if (errs.length > 0) {
        console.error(
          `[ReSTIR] PPG shader compile errors in '${label}':`,
          errs.map((e) => `line ${e.lineNum}: ${e.message}`),
        );
        throw new Error(`[ReSTIR] PPG shader compile error in '${label}': ${errs[0]!.message}`);
      }
    }
    [ppgUpdatePipeline, ppgGuidePipeline] = await Promise.all([
      device.createComputePipelineAsync({
        label: 'ppg-update',
        layout: 'auto',
        compute: { module: ppgUpdateSM, entryPoint: 'ppgUpdateMain' },
      }),
      device.createComputePipelineAsync({
        label: 'ppg-guide',
        layout: 'auto',
        compute: { module: ppgGuideSM, entryPoint: 'ppgGuideMain' },
      }),
    ]);
    if (opts?.verbose) {
      console.log('[ReSTIR] PPG pipelines compiled (Müller 2017 — sTree + dTree + MIS)');
    }
  }

  if (opts?.verbose) {
    console.log('[ReSTIR] All pipelines compiled successfully');
  }

  return {
    risPipeline,
    temporalPipeline,
    spatialPipeline,
    shadePipeline,
    atrousPipeline,
    accumPipeline,
    compositePipeline,
    sampleBudgetPipeline,
    resolvePipeline,
    gtaoPipeline,
    gtaoUpsamplePipeline,
    risGiPipeline,
    temporalGiPipeline,
    spatialGiPipeline,
    indirectCombinePipeline,
    indirectTemporalAccumPipeline,
    // T2.H3 — PPG pipelines (Müller 2017): only present when ppgEnabled.
    ...(ppgUpdatePipeline !== undefined && ppgGuidePipeline !== undefined
      ? { ppgUpdatePipeline, ppgGuidePipeline }
      : {}),
  };
}
