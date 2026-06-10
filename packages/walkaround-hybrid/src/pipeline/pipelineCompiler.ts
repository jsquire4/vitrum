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
  CB_PREFILL_MODULE,
  COMPOSITE_FRAG_MODULE,
  COMPOSITE_VERT_MODULE,
  GTAO_MODULE,
  GTAO_UPSAMPLE_MODULE,
  INDIRECT_COMBINE_MODULE,
  INDIRECT_TEMPORAL_ACCUM_MODULE,
  MOTION_VECTORS_MODULE,
  PPG_UPDATE_MODULE,
  RESOLVE_MODULE,
  REGIR_BUILD_MODULE,
  RIS_GI_MODULE,
  RIS_MODULE,
  SAMPLE_BUDGET_MODULE,
  SHADE_MODULE,
  SPATIAL_GI_MODULE,
  SPATIAL_GI_GRIS_MODULE,
  SPATIAL_MODULE,
  TEMPORAL_ACCUM_MODULE,
  TEMPORAL_GI_MODULE,
  TEMPORAL_GI_GRIS_MODULE,
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
  getCbPrefillBindGroupLayout,
  getMotionVectorsBindGroupLayout,
  getGTAOBindGroupLayout,
  getGTAOUpsampleBindGroupLayout,
  getTemporalGiBindGroupLayout,
  getSpatialGiBindGroupLayout,
  getIndirectCombineBindGroupLayout,
  getIndirectTemporalAccumBindGroupLayout,
  getLightTreeBindGroupLayout,
  getRegirBuildBindGroupLayout,
  getNrcBindGroupLayout,
  type BGLCache,
} from './bindGroupLayouts.js';
import { buildRisGiNrcModule, type RisGiNrcConfig } from '../shaders/risGiNrc.wgsl.js';
import { buildPpgUpdateWgsl } from '../ppg/ppgUpdate.wgsl.js';

interface CompiledPipelines {
  risPipeline: GPUComputePipeline;
  /** T2.H3 — PPG update kernel (training on L_i, Müller §3.3). */
  ppgUpdatePipeline?: GPUComputePipeline;
  temporalPipeline: GPUComputePipeline;
  spatialPipeline: GPUComputePipeline;
  shadePipeline: GPUComputePipeline;
  motionVectorsPipeline: GPUComputePipeline;
  /** Shared à-trous pipeline — used by the legacy `AtrousDenoiser` AND by
   *  the always-on indirect-channel chain (`AtrousIndirectPass`). */
  atrousPipeline: GPUComputePipeline;
  accumPipeline: GPUComputePipeline;
  compositePipeline: GPURenderPipeline;
  /** Sprint 9 — adaptive sampling tier classifier (runs before RIS). */
  sampleBudgetPipeline: GPUComputePipeline;
  /** Sprint 9 — resolve pass (runs between temporalAccum and composite). */
  resolvePipeline: GPUComputePipeline;
  /** Checkerboard pre-denoiser gap-fill (runs before denoiser-adapter when
   *  checkerboard is on AND a real denoiser is active). */
  cbPrefillPipeline: GPUComputePipeline;
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
  /** ReGIR grid-build kernel (Boksansky 2021). Opt-in via `opts.regirEnabled`;
   *  undefined when ReGIR is off (RIS then uses the light-tree path). */
  regirBuildPipeline?: GPUComputePipeline;
}

export async function compilePipelines(
  device: GPUDevice,
  bglCache: BGLCache,
  swapChainFormat: GPUTextureFormat,
  opts?: {
    verbose?: boolean;
    ppgEnabled?: boolean;
    regirEnabled?: boolean;
    restirPtReuse?: boolean;
    /** NRC (Müller et al. 2021) — COMPILE-TIME structural gate. When set, the
     *  gi-ris pipeline is built with a 5th `@group(4)` NRC bind group + the
     *  inline-MLP-forward shader variant; when absent (default) gi-ris is the
     *  verbatim 4-group DDGI-estimate pass. MUST be a compile-time decision —
     *  a runtime UBO flag that bound an extra group on the default path is the
     *  GRIS-class regression (f8df9a4). The value carries the encoding/MLP
     *  config the WGSL sizes are baked from (must match the host NrcSubsystem). */
    nrcConfig?: RisGiNrcConfig;
    /**
     * H29 — per-cell dTree node cap baked into the PPG flux buffer
     * (= `fluxAtomicsBuf.size / 4 / maxSpatialCells` from allocatePPGResources).
     * The PPG update shader's `MAX_DTREE_NODES_PER_CELL` is template-interpolated
     * from this value so the shader and the host agree on the stride.
     * Default 341 (depth-4 full quadtree). Only used when ppgEnabled is true.
     */
    ppgMaxDTreeNodesPerCell?: number;
  },
): Promise<CompiledPipelines> {
  // GRIS / ReSTIR-PT reconnection-shift reuse is opt-in via the host flag
  // `HybridEngineOptions.restirPtReuse`. The gate is COMPILE-TIME (the flag is
  // fixed at engine creation) because turning it on STRUCTURALLY changes the GI
  // spatial + temporal passes — they gain a `@group(1)` scene BVH/TLAS group for
  // the reconnection-visibility ray. Binding a second group / changing the
  // pipeline layout on the DEFAULT path is what regressed the default render to
  // an all-black frame (f8df9a4), so when OFF we compose the verbatim Sprint-17
  // single-group GI passes and build the single-group layout; when ON we compose
  // the GRIS variants and the two-group layout. See spatialGi.wgsl.ts /
  // temporalGi.wgsl.ts headers.
  const grisOn = opts?.restirPtReuse === true;
  // NRC (Müller et al. 2021) — COMPILE-TIME structural gate, same discipline as
  // GRIS above. When ON the gi-ris pass gains a `@group(4)` NRC group + the
  // inline-MLP-forward variant; when OFF (default) gi-ris is the verbatim
  // 4-group DDGI-estimate pass. Binding a 5th group on the default path would
  // alter the default pipeline structure (the GRIS-class regression), so the
  // structure is gated at compile time, not by a runtime UBO flag.
  const nrcOn = opts?.nrcConfig !== undefined;
  // Compile all shader modules. The include-graph (composeWgsl + WGSL_MODULES)
  // resolves each module's dependency closure exactly once — no hand-rolled
  // `COMMON_WGSL + X_WGSL` concat patterns remain.
  const risSM      = device.createShaderModule({ label: 'ris',       code: composeWgsl(RIS_MODULE,      WGSL_MODULES) });
  const temporalSM = device.createShaderModule({ label: 'temporal',  code: composeWgsl(TEMPORAL_MODULE, WGSL_MODULES) });
  const spatialSM  = device.createShaderModule({ label: 'spatial',   code: composeWgsl(SPATIAL_MODULE,  WGSL_MODULES) });
  const shadeSM    = device.createShaderModule({ label: 'shade',     code: composeWgsl(SHADE_MODULE,    WGSL_MODULES) });
  const atrousSM   = device.createShaderModule({ label: 'atrous',    code: composeWgsl(ATROUS_MODULE,   WGSL_MODULES) });
  const compVertSM = device.createShaderModule({ label: 'comp-vert', code: composeWgsl(COMPOSITE_VERT_MODULE, WGSL_MODULES) });
  const compFragSM = device.createShaderModule({ label: 'comp-frag', code: composeWgsl(COMPOSITE_FRAG_MODULE, WGSL_MODULES) });

  // Sprint 9 — sample-budget and resolve are standalone compute shaders.
  // sampleBudget.wgsl template-interpolates WELFORD_VARIANCE_WGSL from
  // @vitrum/shared-denoisers into its own source; resolve.wgsl is
  // self-contained. Both modules declare `requires: []`.
  const sampleBudgetSM  = device.createShaderModule({ label: 'sample-budget', code: composeWgsl(SAMPLE_BUDGET_MODULE,  WGSL_MODULES) });
  const resolveSM       = device.createShaderModule({ label: 'resolve',       code: composeWgsl(RESOLVE_MODULE,        WGSL_MODULES) });
  const cbPrefillSM     = device.createShaderModule({ label: 'cb-prefill',    code: composeWgsl(CB_PREFILL_MODULE,     WGSL_MODULES) });
  const motionVectorsSM = device.createShaderModule({ label: 'motion-vectors', code: composeWgsl(MOTION_VECTORS_MODULE, WGSL_MODULES) });

  // Check for compile errors on every shader module before proceeding.
  const modules: [string, GPUShaderModule][] = [
    ['ris', risSM], ['temporal', temporalSM], ['spatial', spatialSM],
    ['shade', shadeSM], ['atrous', atrousSM],
    ['comp-vert', compVertSM], ['comp-frag', compFragSM],
    ['sample-budget', sampleBudgetSM], ['resolve', resolveSM], ['cb-prefill', cbPrefillSM], ['motion-vectors', motionVectorsSM],
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
  // risLayout: computeLayout + a RIS-ONLY 4th group (light-tree storage buffer)
  // for spatially-aware DI light selection. Kept separate from computeLayout so
  // the extra storage buffer lands on the RIS pipeline only (16 storage buffers,
  // at the full-tier floor) — temporal/spatial/shade are unaffected. RIS uses 4
  // bind groups (frame/scene/ubo/lightTree), within the Lovelace maxBindGroups
  // cap of 4.
  const risLayout = device.createPipelineLayout({
    bindGroupLayouts: [
      getFrameBindGroupLayout(device, bglCache),
      getSceneBindGroupLayout(device, bglCache),
      getUboBindGroupLayout(device, bglCache),
      getLightTreeBindGroupLayout(device, bglCache),
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
  const cbPrefillLayout = device.createPipelineLayout({
    bindGroupLayouts: [getCbPrefillBindGroupLayout(device, bglCache)],
  });
  const motionVectorsLayout = device.createPipelineLayout({
    bindGroupLayouts: [getMotionVectorsBindGroupLayout(device, bglCache)],
  });
  const gtaoLayout = device.createPipelineLayout({
    bindGroupLayouts: [getGTAOBindGroupLayout(device, bglCache)],
  });
  const gtaoUpsampleLayout = device.createPipelineLayout({
    bindGroupLayouts: [getGTAOUpsampleBindGroupLayout(device, bglCache)],
  });
  // Sprint 17 — GI temporal + spatial passes. group(0) is their dedicated
  // reservoir-buffer + uniform group. When GRIS reuse is ON (restirPtReuse),
  // group(1) is the SHARED scene BVH/TLAS group so the reconnection-visibility
  // ray can traverse the scene; the GI shaders then declare those `@group(1)`
  // bindings. When OFF (default) the layout is single-group — byte-for-byte the
  // pre-GRIS Sprint-17 pipeline, the known-good default. Storage-buffer budget
  // (GRIS ON only): group(0) carries 2 reservoir storage buffers, group(1)
  // carries 11 scene storage buffers → 13 total, well under the
  // `HYBRID_WEBGPU_REQUIRED_LIMITS.maxStorageBuffersPerShaderStage = 16` floor.
  const temporalGiLayout = device.createPipelineLayout({
    bindGroupLayouts: grisOn
      ? [
          getTemporalGiBindGroupLayout(device, bglCache),
          getSceneBindGroupLayout(device, bglCache),
        ]
      : [getTemporalGiBindGroupLayout(device, bglCache)],
  });
  const spatialGiLayout = device.createPipelineLayout({
    bindGroupLayouts: grisOn
      ? [
          getSpatialGiBindGroupLayout(device, bglCache),
          getSceneBindGroupLayout(device, bglCache),
        ]
      : [getSpatialGiBindGroupLayout(device, bglCache)],
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
      device.createComputePipelineAsync({ label: 'ris',      layout: risLayout,     compute: { module: risSM,      entryPoint: 'risMain'      } }),
      device.createComputePipelineAsync({ label: 'temporal', layout: computeLayout, compute: { module: temporalSM, entryPoint: 'temporalMain' } }),
      device.createComputePipelineAsync({ label: 'spatial',  layout: computeLayout, compute: { module: spatialSM,  entryPoint: 'spatialMain'  } }),
      device.createComputePipelineAsync({ label: 'shade',    layout: shadeLayout,   compute: { module: shadeSM,    entryPoint: 'shadeMain'    } }),
    ]);
  const motionVectorsPipeline = await device.createComputePipelineAsync({
    label: 'motion-vectors',
    layout: motionVectorsLayout,
    compute: { module: motionVectorsSM, entryPoint: 'motionVectorsMain' },
  });

  const atrousPipeline = await device.createComputePipelineAsync({
    label: 'atrous', layout: atrousLayout,
    compute: { module: atrousSM, entryPoint: 'atrousMain' },
  });

  // Sprint 9 — adaptive sampling pipelines + checkerboard pre-denoiser fill.
  const [sampleBudgetPipeline, resolvePipeline, cbPrefillPipeline] = await Promise.all([
    device.createComputePipelineAsync({
      label: 'sample-budget', layout: sampleBudgetLayout,
      compute: { module: sampleBudgetSM, entryPoint: 'sampleBudgetKernel' },
    }),
    device.createComputePipelineAsync({
      label: 'resolve', layout: resolveLayout,
      compute: { module: resolveSM, entryPoint: 'resolveKernel' },
    }),
    device.createComputePipelineAsync({
      label: 'cb-prefill', layout: cbPrefillLayout,
      compute: { module: cbPrefillSM, entryPoint: 'cbPrefillKernel' },
    }),
  ]);

  // Sprint 15 — GTAO pipelines.
  const gtaoSM = device.createShaderModule({ label: 'gtao', code: composeWgsl(GTAO_MODULE, WGSL_MODULES) });
  const gtaoUpsampleSM = device.createShaderModule({ label: 'gtao-upsample', code: composeWgsl(GTAO_UPSAMPLE_MODULE, WGSL_MODULES) });
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
  // NRC ON: compose the 5-group inline-MLP-forward variant + build a 5-group
  // layout (shade layout + the NRC @group(4)). NRC OFF (default): the verbatim
  // 4-group DDGI-estimate pass on the shade layout — byte-for-byte pre-NRC.
  const risGiSM = device.createShaderModule({
    label: 'risGi',
    code: nrcOn
      ? composeWgsl(buildRisGiNrcModule(opts!.nrcConfig!), WGSL_MODULES)
      : composeWgsl(RIS_GI_MODULE, WGSL_MODULES),
  });
  const risGiLayout = nrcOn
    ? device.createPipelineLayout({
        bindGroupLayouts: [
          getFrameBindGroupLayout(device, bglCache),
          getSceneBindGroupLayout(device, bglCache),
          getUboBindGroupLayout(device, bglCache),
          getHybridLayersBindGroupLayout(device, bglCache),
          getNrcBindGroupLayout(device, bglCache),
        ],
      })
    : shadeLayout;
  const risGiPipeline = await device.createComputePipelineAsync({
    label: 'risGi', layout: risGiLayout,
    compute: { module: risGiSM, entryPoint: 'risGiMain' },
  });

  // Sprint 17 — GI temporal + spatial reuse pipelines. Compose the GRIS variant
  // (adds @group(1) scene bindings + the reconnection-shift branch) only when
  // restirPtReuse is ON; otherwise compose the verbatim Sprint-17 single-group
  // pass. The chosen module's bindings MUST match the layout selected above.
  const temporalGiSM = device.createShaderModule({
    label: 'temporalGi',
    code: composeWgsl(grisOn ? TEMPORAL_GI_GRIS_MODULE : TEMPORAL_GI_MODULE, WGSL_MODULES),
  });
  const spatialGiSM = device.createShaderModule({
    label: 'spatialGi',
    code: composeWgsl(grisOn ? SPATIAL_GI_GRIS_MODULE : SPATIAL_GI_MODULE, WGSL_MODULES),
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

  const accumSM = device.createShaderModule({ label: 'accum', code: composeWgsl(TEMPORAL_ACCUM_MODULE, WGSL_MODULES) });
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

  // T2.H3 — PPG update pipeline (Müller 2017 §3.3): opt-in via opts.ppgEnabled.
  // Guided sampling itself is inlined in gi-ris via ppgPdf.wgsl; the update
  // kernel is the only standalone PPG training pass.
  // H29: the WGSL MAX_DTREE_NODES_PER_CELL is built from the live allocation
  // value (opts.ppgMaxDTreeNodesPerCell) so the shader and host stride agree.
  let ppgUpdatePipeline: GPUComputePipeline | undefined;
  if (opts?.ppgEnabled) {
    const ppgMaxDTreeNodesPerCell = opts?.ppgMaxDTreeNodesPerCell ?? 341;
    const ppgUpdateModule = {
      name: 'ppgUpdate' as const,
      source: buildPpgUpdateWgsl(ppgMaxDTreeNodesPerCell),
      requires: ['luminance', 'ppgTreeLayout'] as const,
    };
    const ppgUpdateSM = device.createShaderModule({ label: 'ppg-update', code: composeWgsl(ppgUpdateModule, WGSL_MODULES) });
    const info = await ppgUpdateSM.getCompilationInfo();
    const errs = info.messages.filter(m => m.type === 'error');
    if (errs.length > 0) {
      console.error('[ReSTIR] PPG shader compile errors in \'ppg-update\':', errs.map(e => `line ${e.lineNum}: ${e.message}`));
      throw new Error(`[ReSTIR] PPG shader compile error in 'ppg-update': ${errs[0]!.message}`);
    }
    ppgUpdatePipeline = await device.createComputePipelineAsync({
      label: 'ppg-update', layout: 'auto',
      compute: { module: ppgUpdateSM, entryPoint: 'ppgUpdateMain' },
    });
    if (opts?.verbose) {
      console.log('[ReSTIR] PPG update pipeline compiled (Muller 2017 - sTree + dTree training)');
    }
  }

  // ReGIR grid-build pipeline (Boksansky 2021) — opt-in via opts.regirEnabled.
  // Its own single bind group (combined light-tree + grid buffer read_write,
  // emitters, ubo). Compiled only when requested so non-ReGIR engines pay no
  // boot cost and the buffer stays read-only on every other layout.
  let regirBuildPipeline: GPUComputePipeline | undefined;
  if (opts?.regirEnabled) {
    const regirBuildSM = device.createShaderModule({
      label: 'regir-build',
      code: composeWgsl(REGIR_BUILD_MODULE, WGSL_MODULES),
    });
    const info = await regirBuildSM.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === 'error');
    if (errs.length > 0) {
      console.error('[ReSTIR] ReGIR shader compile errors in \'regir-build\':',
        errs.map((e) => `line ${e.lineNum}: ${e.message}`));
      throw new Error(`[ReSTIR] ReGIR shader compile error in 'regir-build': ${errs[0]!.message}`);
    }
    const regirBuildLayout = device.createPipelineLayout({
      bindGroupLayouts: [getRegirBuildBindGroupLayout(device, bglCache)],
    });
    regirBuildPipeline = await device.createComputePipelineAsync({
      label: 'regir-build', layout: regirBuildLayout,
      compute: { module: regirBuildSM, entryPoint: 'regirBuildMain' },
    });
    if (opts?.verbose) {
      console.log('[ReSTIR] ReGIR grid-build pipeline compiled (Boksansky 2021)');
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
    motionVectorsPipeline,
    atrousPipeline,
    accumPipeline,
    compositePipeline,
    sampleBudgetPipeline,
    resolvePipeline,
    cbPrefillPipeline,
    gtaoPipeline,
    gtaoUpsamplePipeline,
    risGiPipeline,
    temporalGiPipeline,
    spatialGiPipeline,
    indirectCombinePipeline,
    indirectTemporalAccumPipeline,
    // T2.H3 — PPG update pipeline (Müller 2017): only present when ppgEnabled.
    ...(ppgUpdatePipeline !== undefined ? { ppgUpdatePipeline } : {}),
    // ReGIR grid-build (Boksansky 2021): only present when regirEnabled.
    ...(regirBuildPipeline !== undefined ? { regirBuildPipeline } : {}),
  };
}
