/**
 * Pipeline compiler — creates all GPUComputePipeline and GPURenderPipeline
 * objects used by WalkaroundGPUPipeline.
 *
 * Called once from `initialize()`. Compiles all shader modules in parallel,
 * checks for compile errors, then creates pipeline layouts and dispatches
 * pipeline creation with `createComputePipelineAsync` / `createRenderPipelineAsync`.
 *
 * The temporalAccum shader is NOT concatenated with COMMON_WGSL — it is a
 * standalone compute shader with no dependency on the common library.
 */

import { COMMON_WGSL } from '../shaders/common.wgsl.js';
import { RIS_WGSL } from '../shaders/ris.wgsl.js';
import { TEMPORAL_WGSL } from '../shaders/temporal.wgsl.js';
import { SPATIAL_WGSL } from '../shaders/spatial.wgsl.js';
import { SHADE_WGSL } from '../shaders/shade.wgsl.js';
import { SAMPLE_BUDGET_WGSL } from '../shaders/sampleBudget.wgsl.js';
import { RESOLVE_WGSL } from '../shaders/resolve.wgsl.js';
import { GTAO_WGSL } from '../shaders/gtao.wgsl.js';
import { GTAO_UPSAMPLE_WGSL } from '../shaders/gtaoUpsample.wgsl.js';
import { RIS_GI_WGSL } from '../shaders/risGi.wgsl.js';
import { TEMPORAL_GI_WGSL } from '../shaders/temporalGi.wgsl.js';
import { SPATIAL_GI_WGSL } from '../shaders/spatialGi.wgsl.js';
import { INDIRECT_COMBINE_WGSL } from '../shaders/indirectCombine.wgsl.js';
import { INDIRECT_TEMPORAL_ACCUM_WGSL } from '../shaders/indirectTemporalAccum.wgsl.js';
import { SURFACE_TEXTURES_WGSL } from '../shaders/surfaceTextures.wgsl.js';
import { DDGI_SAMPLE_WGSL } from '../ddgi/ddgiSampleWgsl.js';
import {
  ATROUS_WGSL,
  ATROUS_VARIANCE_WGSL,
  TEMPORAL_ACCUM_WGSL,
  SVGF_REPROJECTION_WGSL,
  SVGF_VARIANCE_FROM_MOMENTS_WGSL,
  SVGF_7X7_SPATIAL_FALLBACK_WGSL,
} from '@vitrum/shared-denoisers';
import { WELFORD_TEMPORAL_WGSL } from '../shaders/welfordTemporal.wgsl.js';
import { COMPOSITE_VERT_WGSL, COMPOSITE_FRAG_WGSL } from '../shaders/composite.wgsl.js';
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
  temporalPipeline: GPUComputePipeline;
  spatialPipeline: GPUComputePipeline;
  shadePipeline: GPUComputePipeline;
  atrousPipeline: GPUComputePipeline;
  accumPipeline: GPUComputePipeline;
  compositePipeline: GPURenderPipeline;
  denoiserMode: 'atrous' | 'atrous-variance' | 'svgf-real';
  welfordPipeline?: GPUComputePipeline;
  atrousVarianceVariancePipeline?: GPUComputePipeline;
  atrousVarianceAtrousPipeline?: GPUComputePipeline;
  /** T2.H1 — svgf-real: bilinear reprojection + EMA history (Schied 2017 Stage 1). */
  svgfReprojPipeline?: GPUComputePipeline;
  /** T2.H1 — svgf-real: variance from moments Eq. 5. */
  svgfMomentsPipeline?: GPUComputePipeline;
  /** T2.H1 — svgf-real: 7×7 spatial fallback §4.3. */
  svgfFallbackPipeline?: GPUComputePipeline;
  /** T2.H1 — svgf-real: à-trous pass (reuses atrous-variance shader). */
  svgfRealAtrousPipeline?: GPUComputePipeline;
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
  opts?: { verbose?: boolean; denoiser?: 'atrous' | 'atrous-variance' | 'svgf-real' },
): Promise<CompiledPipelines> {
  const denoiserMode = opts?.denoiser ?? 'atrous-variance';
  // Compile all shader modules (common WGSL is prepended to each ReSTIR pass).
  const risSM      = device.createShaderModule({ label: 'ris',      code: COMMON_WGSL + RIS_WGSL });
  const temporalSM = device.createShaderModule({ label: 'temporal', code: COMMON_WGSL + TEMPORAL_WGSL });
  const spatialSM  = device.createShaderModule({ label: 'spatial',  code: COMMON_WGSL + SPATIAL_WGSL });
  const shadeSM    = device.createShaderModule({ label: 'shade',    code: COMMON_WGSL + SURFACE_TEXTURES_WGSL + DDGI_SAMPLE_WGSL + SHADE_WGSL });
  const atrousSM   = device.createShaderModule({ label: 'atrous',   code: COMMON_WGSL + ATROUS_WGSL });
  const compVertSM = device.createShaderModule({ label: 'comp-vert', code: COMPOSITE_VERT_WGSL });
  const compFragSM = device.createShaderModule({ label: 'comp-frag', code: COMPOSITE_FRAG_WGSL });

  // Sprint 9 — sample-budget and resolve are standalone compute shaders.
  // sampleBudget.wgsl imports WELFORD_VARIANCE_WGSL from @vitrum/shared-denoisers
  // directly; resolve.wgsl is self-contained.
  const sampleBudgetSM = device.createShaderModule({ label: 'sample-budget', code: SAMPLE_BUDGET_WGSL });
  const resolveSM      = device.createShaderModule({ label: 'resolve',       code: RESOLVE_WGSL });

  // Check for compile errors on every shader module before proceeding.
  const welfordSM =
    denoiserMode === 'atrous-variance'
      ? device.createShaderModule({ label: 'welford-temporal', code: COMMON_WGSL + WELFORD_TEMPORAL_WGSL })
      : null;
  // ATROUS_VARIANCE_WGSL is self-contained: it declares its own PI, INV_PI, LUM_W, and
  // WelfordVariance struct (via WELFORD_VARIANCE_WGSL). Do NOT prepend
  // COMMON_WGSL here — it would cause WGSL redeclaration errors on those
  // names. The welford-temporal pass DOES need COMMON_WGSL (for the BVH /
  // shared math helpers), which is why those two diverge.
  const atrousVarianceSM =
    denoiserMode === 'atrous-variance'
      ? device.createShaderModule({ label: 'atrous-variance', code: ATROUS_VARIANCE_WGSL })
      : null;

  // T2.H1 — svgf-real shader modules.
  const svgfReprojSM = denoiserMode === 'svgf-real'
    ? device.createShaderModule({ label: 'svgf-reproj', code: SVGF_REPROJECTION_WGSL })
    : null;
  const svgfMomentsSM = denoiserMode === 'svgf-real'
    ? device.createShaderModule({ label: 'svgf-moments', code: SVGF_VARIANCE_FROM_MOMENTS_WGSL })
    : null;
  const svgfFallbackSM = denoiserMode === 'svgf-real'
    ? device.createShaderModule({ label: 'svgf-7x7', code: SVGF_7X7_SPATIAL_FALLBACK_WGSL })
    : null;
  // svgf-real reuses the atrous-variance à-trous chain for spatial filtering.
  const svgfRealAtrousVarianceSM = denoiserMode === 'svgf-real'
    ? device.createShaderModule({ label: 'svgf-real-atrous-variance', code: ATROUS_VARIANCE_WGSL })
    : null;

  const modules: [string, GPUShaderModule][] = [
    ['ris', risSM], ['temporal', temporalSM], ['spatial', spatialSM],
    ['shade', shadeSM], ['atrous', atrousSM],
    ['comp-vert', compVertSM], ['comp-frag', compFragSM],
    ['sample-budget', sampleBudgetSM], ['resolve', resolveSM],
    ...(welfordSM ? [['welford', welfordSM] as [string, GPUShaderModule]] : []),
    ...(atrousVarianceSM ? [['atrous-variance', atrousVarianceSM] as [string, GPUShaderModule]] : []),
    ...(svgfReprojSM           ? [['svgf-reproj',   svgfReprojSM]   as [string, GPUShaderModule]] : []),
    ...(svgfMomentsSM          ? [['svgf-moments',  svgfMomentsSM]  as [string, GPUShaderModule]] : []),
    ...(svgfFallbackSM         ? [['svgf-7x7',      svgfFallbackSM] as [string, GPUShaderModule]] : []),
    ...(svgfRealAtrousVarianceSM ? [['svgf-real-atrous-variance', svgfRealAtrousVarianceSM] as [string, GPUShaderModule]] : []),
  ];
  for (const [label, sm] of modules) {
    const info = await sm.getCompilationInfo();
    const errors = info.messages.filter(m => m.type === 'error');
    if (errors.length > 0) {
      console.error(`[ReSTIR] Shader compile errors in '${label}':`, errors.map(e => `line ${e.lineNum}: ${e.message}`));
      throw new Error(`[ReSTIR] Shader compile error in '${label}': ${errors[0]!.message} (line ${errors[0]!.lineNum})`);
    }
    const warns = info.messages.filter(m => m.type === 'warning');
    if (warns.length > 0) {
      console.warn(`[ReSTIR] Shader warnings in '${label}':`, warns.map(w => w.message));
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
  const [risPipeline, temporalPipeline, spatialPipeline, shadePipeline] =
    await Promise.all([
      device.createComputePipelineAsync({ label: 'ris',      layout: computeLayout, compute: { module: risSM,      entryPoint: 'risMain'      } }),
      device.createComputePipelineAsync({ label: 'temporal', layout: computeLayout, compute: { module: temporalSM, entryPoint: 'temporalMain' } }),
      device.createComputePipelineAsync({ label: 'spatial',  layout: computeLayout, compute: { module: spatialSM,  entryPoint: 'spatialMain'  } }),
      device.createComputePipelineAsync({ label: 'shade',    layout: shadeLayout,   compute: { module: shadeSM,    entryPoint: 'shadeMain'    } }),
    ]);

  const atrousPipeline = await device.createComputePipelineAsync({
    label: 'atrous', layout: atrousLayout,
    compute: { module: atrousSM, entryPoint: 'atrousMain' },
  });

  // Sprint 9 — adaptive sampling pipelines.
  const [sampleBudgetPipeline, resolvePipeline] = await Promise.all([
    device.createComputePipelineAsync({
      label: 'sample-budget', layout: sampleBudgetLayout,
      compute: { module: sampleBudgetSM, entryPoint: 'sampleBudgetKernel' },
    }),
    device.createComputePipelineAsync({
      label: 'resolve', layout: resolveLayout,
      compute: { module: resolveSM, entryPoint: 'resolveKernel' },
    }),
  ]);

  // Sprint 15 — GTAO pipelines.
  const gtaoSM = device.createShaderModule({ label: 'gtao', code: GTAO_WGSL });
  const gtaoUpsampleSM = device.createShaderModule({ label: 'gtao-upsample', code: GTAO_UPSAMPLE_WGSL });
  const [gtaoPipeline, gtaoUpsamplePipeline] = await Promise.all([
    device.createComputePipelineAsync({
      label: 'gtao', layout: gtaoLayout,
      compute: { module: gtaoSM, entryPoint: 'gtaoMain' },
    }),
    device.createComputePipelineAsync({
      label: 'gtao-upsample', layout: gtaoUpsampleLayout,
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
    code: COMMON_WGSL + DDGI_SAMPLE_WGSL + RIS_GI_WGSL,
  });
  const risGiPipeline = await device.createComputePipelineAsync({
    label: 'risGi', layout: shadeLayout,
    compute: { module: risGiSM, entryPoint: 'risGiMain' },
  });

  // Sprint 17 — GI temporal + spatial reuse pipelines.
  const temporalGiSM = device.createShaderModule({
    label: 'temporalGi',
    code: COMMON_WGSL + TEMPORAL_GI_WGSL,
  });
  const spatialGiSM = device.createShaderModule({
    label: 'spatialGi',
    code: COMMON_WGSL + SPATIAL_GI_WGSL,
  });
  const [temporalGiPipeline, spatialGiPipeline] = await Promise.all([
    device.createComputePipelineAsync({
      label: 'temporalGi', layout: temporalGiLayout,
      compute: { module: temporalGiSM, entryPoint: 'temporalGiMain' },
    }),
    device.createComputePipelineAsync({
      label: 'spatialGi', layout: spatialGiLayout,
      compute: { module: spatialGiSM, entryPoint: 'spatialGiMain' },
    }),
  ]);

  // Sprint 18 — indirect-combine pipeline.
  const indirectCombineSM = device.createShaderModule({
    label: 'indirectCombine',
    code: INDIRECT_COMBINE_WGSL,
  });
  const indirectCombinePipeline = await device.createComputePipelineAsync({
    label: 'indirectCombine',
    layout: indirectCombineLayout,
    compute: { module: indirectCombineSM, entryPoint: 'indirectCombineMain' },
  });

  // Sprint 18 follow-up — indirect pre-atrous temporal accumulator.
  const indirectTemporalAccumSM = device.createShaderModule({
    label: 'indirectTemporalAccum',
    code: INDIRECT_TEMPORAL_ACCUM_WGSL,
  });
  const indirectTemporalAccumPipeline = await device.createComputePipelineAsync({
    label: 'indirectTemporalAccum',
    layout: indirectTemporalAccumLayout,
    compute: { module: indirectTemporalAccumSM, entryPoint: 'indirectTemporalAccumMain' },
  });

  // T2.H1 — svgf-real pipelines (compile only when denoiserMode === 'svgf-real').
  let svgfReprojPipeline:      GPUComputePipeline | undefined;
  let svgfMomentsPipeline:     GPUComputePipeline | undefined;
  let svgfFallbackPipeline:    GPUComputePipeline | undefined;
  let svgfRealAtrousPipeline:  GPUComputePipeline | undefined;
  if (denoiserMode === 'svgf-real' && svgfReprojSM && svgfMomentsSM && svgfFallbackSM && svgfRealAtrousVarianceSM) {
    [svgfReprojPipeline, svgfMomentsPipeline, svgfFallbackPipeline] = await Promise.all([
      device.createComputePipelineAsync({
        label: 'svgf-real-reproj',
        layout: 'auto',
        compute: { module: svgfReprojSM, entryPoint: 'svgfReprojMain' },
      }),
      device.createComputePipelineAsync({
        label: 'svgf-real-moments',
        layout: 'auto',
        compute: { module: svgfMomentsSM, entryPoint: 'svgfVarianceFromMomentsMain' },
      }),
      device.createComputePipelineAsync({
        label: 'svgf-real-7x7',
        layout: 'auto',
        compute: { module: svgfFallbackSM, entryPoint: 'svgf7x7FallbackMain' },
      }),
    ]);
    svgfRealAtrousPipeline = await device.createComputePipelineAsync({
      label: 'svgf-real-atrous',
      layout: 'auto',
      compute: { module: svgfRealAtrousVarianceSM, entryPoint: 'svgfAtrousMain' },
    });
  }

  let welfordPipeline: GPUComputePipeline | undefined;
  let atrousVarianceVariancePipeline: GPUComputePipeline | undefined;
  let atrousVarianceAtrousPipeline: GPUComputePipeline | undefined;
  if (denoiserMode === 'atrous-variance' && welfordSM && atrousVarianceSM) {
    welfordPipeline = await device.createComputePipelineAsync({
      label: 'welford-temporal',
      layout: 'auto',
      compute: { module: welfordSM, entryPoint: 'welfordTemporalMain' },
    });
    atrousVarianceVariancePipeline = await device.createComputePipelineAsync({
      label: 'atrous-variance-variance',
      layout: 'auto',
      compute: { module: atrousVarianceSM, entryPoint: 'svgfVarianceMain' },
    });
    atrousVarianceAtrousPipeline = await device.createComputePipelineAsync({
      label: 'atrous-variance-atrous',
      layout: 'auto',
      compute: { module: atrousVarianceSM, entryPoint: 'svgfAtrousMain' },
    });
  }

  const accumSM = device.createShaderModule({ label: 'accum', code: TEMPORAL_ACCUM_WGSL });
  const accumPipeline = await device.createComputePipelineAsync({
    label: 'temporalAccum', layout: accumLayout,
    compute: { module: accumSM, entryPoint: 'temporalAccumMain' },
  });

  // Composite render pipeline.
  const compositePipeline = await device.createRenderPipelineAsync({
    label: 'composite',
    layout: compositeLayout,
    vertex:   { module: compVertSM, entryPoint: 'vertMain' },
    fragment: {
      module: compFragSM,
      entryPoint: 'fragMain',
      targets: [{ format: swapChainFormat }],
    },
    primitive: { topology: 'triangle-list' },
  });

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
    denoiserMode,
    ...(welfordPipeline !== undefined &&
    atrousVarianceVariancePipeline !== undefined &&
    atrousVarianceAtrousPipeline !== undefined
      ? { welfordPipeline, atrousVarianceVariancePipeline, atrousVarianceAtrousPipeline }
      : {}),
    ...(svgfReprojPipeline !== undefined &&
    svgfMomentsPipeline !== undefined &&
    svgfFallbackPipeline !== undefined &&
    svgfRealAtrousPipeline !== undefined
      ? { svgfReprojPipeline, svgfMomentsPipeline, svgfFallbackPipeline, svgfRealAtrousPipeline }
      : {}),
  };
}
