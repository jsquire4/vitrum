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

import type { EngineWarning } from '@vitrum/core';
import { composeWgsl } from './wgslComposer.js';
import { checkShaderCompile } from './shaderUtils.js';
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
  RESOLVE_MODULE,
  REGIR_BUILD_MODULE,
  RIS_GI_MODULE,
  RIS_MODULE,
  SAMPLE_BUDGET_MODULE,
  SHADE_MODULE,
  SPATIAL_GI_MODULE,
  SPATIAL_MODULE,
  TEMPORAL_ACCUM_MODULE,
  TEMPORAL_GI_MODULE,
  TEMPORAL_MODULE,
  TRANSPARENT_OIT_MODULE,
  WGSL_MODULES,
} from './wgslModules.js';
import {
  getFrameBindGroupLayout,
  getRisGiFrameBindGroupLayout,
  getSceneBindGroupLayout,
  getUboBindGroupLayout,
  getAtrousBindGroupLayout,
  getAccumBindGroupLayout,
  getCompositeBindGroupLayout,
  getHybridLayersBindGroupLayout,
  getShadeHybridLayersBindGroupLayout,
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
  getTransparentOitBindGroupLayout,
  getLightTreeBindGroupLayout,
  getRegirBuildBindGroupLayout,
  getNrcHybridLayersBindGroupLayout,
  type BGLCache,
} from './bindGroupLayouts.js';
import { buildRisGiNrcModule, type RisGiNrcConfig } from '../shaders/risGiNrc.wgsl.js';
import { buildReservoirGiModule } from '../shaders/reservoirGi.wgsl.js';
import { RESERVOIR_GI_STRIDE_U32 } from '../gi/giLayout.js';
import {
  buildPpgUpdateWgsl,
  PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL,
} from '../ppg/ppgUpdate.wgsl.js';
import {
  assertHybridSwapChainFormat,
  hybridCompositeFragmentConstants,
} from '../presentationTarget.js';

/**
 * Keys in `CompiledPipelines` whose value type is `GPUComputePipeline` (not
 * `GPURenderPipeline`, not optional). Derived from the interface rather than
 * hand-maintained so it stays in sync automatically (D3.15).
 */
type ComputePipelineKey = {
  [K in keyof Required<CompiledPipelines>]: Required<CompiledPipelines>[K] extends GPUComputePipeline ? K : never;
}[keyof Required<CompiledPipelines>];

interface CompiledPipelines {
  risPipeline: GPUComputePipeline;
  /** T2.H3 — PPG update kernel (training on L_i, Müller §3.3). */
  ppgUpdatePipeline?: GPUComputePipeline;
  temporalPipeline: GPUComputePipeline;
  spatialPipeline: GPUComputePipeline;
  /** ReSTIR-DI spatial round two, specialized with an independent RNG salt. */
  spatialPipelineRoundTwo: GPUComputePipeline;
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
  /** Camera-visible transparent layer composition over combined radiance. */
  transparentOitPipeline: GPUComputePipeline;
  /** Sprint 18 follow-up — pre-atrous temporal accumulator on indirect. */
  indirectTemporalAccumPipeline: GPUComputePipeline;
  /** Reset variant: same bindings, but publishes current indirect only. */
  indirectTemporalAccumResetPipeline: GPUComputePipeline;
  /** ReGIR grid-build kernel (Boksansky 2021). Opt-in via `opts.regirEnabled`;
   *  undefined when ReGIR is off (RIS then uses the light-tree path). */
  regirBuildPipeline?: GPUComputePipeline;
}

async function checkPipelineShaderModules(
  modules: readonly (readonly [label: string, module: GPUShaderModule])[],
  onWarning?: (warning: EngineWarning) => void,
): Promise<void> {
  await Promise.all(modules.map(async ([label, module]) => {
    await checkShaderCompile(module, label, {
      prefix: '[ReSTIR]',
      onWarnings: (warnings) => emitShaderCompilationWarnings(label, warnings, {
        ...(onWarning !== undefined ? { onWarning } : {}),
      }),
    });
  }));
}

interface CompositeShaderModules {
  readonly vertex: GPUShaderModule;
  readonly fragment: GPUShaderModule;
}

const compositeShaderModulesByDevice =
  new WeakMap<GPUDevice, CompositeShaderModules>();

/**
 * Compile the format-specialized composite pipeline synchronously.
 *
 * WebGPU render-pipeline color targets bake their format, while the host may
 * legally replace its canvas configuration between frames. Shader modules do
 * not bake that target format, so this bounded render-frame rebuild reuses the
 * modules whose compilation info was checked by a successful `compilePipelines`
 * call instead of creating unchecked modules on the synchronous frame path.
 */
export function createCompositePipeline(
  device: GPUDevice,
  bglCache: BGLCache,
  swapChainFormat: GPUTextureFormat,
): GPURenderPipeline {
  assertHybridSwapChainFormat(
    swapChainFormat,
    'createCompositePipeline.swapChainFormat',
  );
  const modules = compositeShaderModulesByDevice.get(device);
  if (modules == null) {
    throw new Error(
      'createCompositePipeline requires a successful compilePipelines call for this device.',
    );
  }
  const layout = device.createPipelineLayout({
    bindGroupLayouts: [getCompositeBindGroupLayout(device, bglCache)],
  });
  return device.createRenderPipeline({
    label: 'composite',
    layout,
    vertex: { module: modules.vertex, entryPoint: 'vertMain' },
    fragment: {
      module: modules.fragment,
      entryPoint: 'fragMain',
      targets: [{ format: swapChainFormat }],
      constants: hybridCompositeFragmentConstants(swapChainFormat),
    },
    primitive: { topology: 'triangle-list' },
  });
}

export async function compilePipelines(
  device: GPUDevice,
  bglCache: BGLCache,
  swapChainFormat: GPUTextureFormat,
  opts?: {
    verbose?: boolean;
    onWarning?: (warning: EngineWarning) => void;
    ppgEnabled?: boolean;
    regirEnabled?: boolean;
    /** NRC (Müller et al. 2021) — COMPILE-TIME structural gate. When set, the
     *  gi-ris pipeline is built with a 5th `@group(4)` NRC bind group + the
     *  inline-MLP-forward shader variant; when absent (default) gi-ris is the
     *  verbatim 4-group DDGI-estimate pass. MUST be a compile-time decision —
     *  a runtime UBO flag that bound an extra group on the NRC-off path caused
     *  the prior structural-layout regression (f8df9a4). The value carries the
     *  encoding/MLP config the WGSL sizes are baked from (must match the host
     *  NrcSubsystem). */
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
  assertHybridSwapChainFormat(
    swapChainFormat,
    'compilePipelines.swapChainFormat',
  );
  // Generalized reconnection-shift reuse is the sole structural GI path.
  const reservoirGiStrideU32 = RESERVOIR_GI_STRIDE_U32;
  const wgslModules = new Map(WGSL_MODULES);
  wgslModules.set('reservoirGi', buildReservoirGiModule());
  // NRC (Müller et al. 2021) — COMPILE-TIME structural gate. When ON the gi-ris
  // pass gains a `@group(4)` NRC group + the
  // inline-MLP-forward variant; when OFF (default) gi-ris is the verbatim
  // 4-group DDGI-estimate pass. Binding a 5th group on the NRC-off path would
  // alter that pipeline structure, so the structure is gated at compile time,
  // not by a runtime UBO flag.
  const nrcOn = opts?.nrcConfig !== undefined;
  // Compile all shader modules. The include-graph (composeWgsl + WGSL_MODULES)
  // resolves each module's dependency closure exactly once — no hand-rolled
  // `COMMON_WGSL + X_WGSL` concat patterns remain.
  const risSM      = device.createShaderModule({ label: 'ris',       code: composeWgsl(RIS_MODULE,      wgslModules) });
  const temporalSM = device.createShaderModule({ label: 'temporal',  code: composeWgsl(TEMPORAL_MODULE, wgslModules) });
  const spatialSM  = device.createShaderModule({ label: 'spatial',   code: composeWgsl(SPATIAL_MODULE,  wgslModules) });
  const shadeSM    = device.createShaderModule({ label: 'shade',     code: composeWgsl(SHADE_MODULE,    wgslModules) });
  const atrousSM   = device.createShaderModule({ label: 'atrous',    code: composeWgsl(ATROUS_MODULE,   wgslModules) });
  const compVertSM = device.createShaderModule({ label: 'comp-vert', code: composeWgsl(COMPOSITE_VERT_MODULE, wgslModules) });
  const compFragSM = device.createShaderModule({ label: 'comp-frag', code: composeWgsl(COMPOSITE_FRAG_MODULE, wgslModules) });

  // Sprint 9 — sample-budget and resolve are standalone compute shaders.
  // sampleBudget.wgsl template-interpolates WELFORD_VARIANCE_WGSL from
  // @vitrum/shared-denoisers into its own source; resolve.wgsl is
  // self-contained. Both modules declare `requires: []`.
  const sampleBudgetSM  = device.createShaderModule({ label: 'sample-budget', code: composeWgsl(SAMPLE_BUDGET_MODULE,  wgslModules) });
  const resolveSM       = device.createShaderModule({ label: 'resolve',       code: composeWgsl(RESOLVE_MODULE,        wgslModules) });
  const cbPrefillSM     = device.createShaderModule({ label: 'cb-prefill',    code: composeWgsl(CB_PREFILL_MODULE,     wgslModules) });
  const motionVectorsSM = device.createShaderModule({ label: 'motion-vectors', code: composeWgsl(MOTION_VECTORS_MODULE, wgslModules) });
  const transparentOitSM = device.createShaderModule({ label: 'transparent-oit', code: composeWgsl(TRANSPARENT_OIT_MODULE, wgslModules) });

  // Check for compile errors on every shader module before proceeding.
  const modules: readonly (readonly [string, GPUShaderModule])[] = [
    ['ris', risSM], ['temporal', temporalSM], ['spatial', spatialSM],
    ['shade', shadeSM], ['atrous', atrousSM],
    ['comp-vert', compVertSM], ['comp-frag', compFragSM],
    ['sample-budget', sampleBudgetSM], ['resolve', resolveSM], ['cb-prefill', cbPrefillSM], ['motion-vectors', motionVectorsSM],
    ['transparent-oit', transparentOitSM],
  ];
  await checkPipelineShaderModules(modules, opts?.onWarning);

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
  // the extra storage buffer lands on the RIS pipeline only (at the derived
  // full-tier floor) — temporal/spatial/shade are unaffected. RIS uses 4
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
      getShadeHybridLayersBindGroupLayout(device, bglCache),
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
  // Sprint 17 + rich-material receiver target — GI temporal + spatial passes.
  // group(0) is their dedicated reservoir-buffer + uniform group. group(1) is
  // always the shared scene BVH/TLAS/material-atlas group: the canonical
  // generalized-reuse path recasts receiver material payloads for the GI p-hat
  // and uses the same group for reconnection visibility. Storage-buffer budget:
  // group(0) carries 2 reservoir storage buffers, group(1) carries the scene
  // storage buffers, staying under the full-tier floor.
  const temporalGiLayout = device.createPipelineLayout({
    bindGroupLayouts: [
      getTemporalGiBindGroupLayout(device, bglCache),
      getSceneBindGroupLayout(device, bglCache),
    ],
  });
  const spatialGiLayout = device.createPipelineLayout({
    bindGroupLayouts: [
      getSpatialGiBindGroupLayout(device, bglCache),
      getSceneBindGroupLayout(device, bglCache),
    ],
  });
  // Sprint 18 — indirect-combine pass uses a single dedicated bind group.
  const indirectCombineLayout = device.createPipelineLayout({
    bindGroupLayouts: [getIndirectCombineBindGroupLayout(device, bglCache)],
  });
  const indirectTemporalAccumLayout = device.createPipelineLayout({
    bindGroupLayouts: [getIndirectTemporalAccumBindGroupLayout(device, bglCache)],
  });
  const transparentOitLayout = device.createPipelineLayout({
    bindGroupLayouts: [
      getFrameBindGroupLayout(device, bglCache),
      getSceneBindGroupLayout(device, bglCache),
      getUboBindGroupLayout(device, bglCache),
      getTransparentOitBindGroupLayout(device, bglCache),
    ],
  });

  // Sprint 15 — GTAO shader modules (needed by PIPELINE_SPECS table below).
  const gtaoSM = device.createShaderModule({ label: 'gtao', code: composeWgsl(GTAO_MODULE, wgslModules) });
  const gtaoUpsampleSM = device.createShaderModule({ label: 'gtao-upsample', code: composeWgsl(GTAO_UPSAMPLE_MODULE, wgslModules) });

  // Sprint 18 — indirect-combine + indirect-temporal-accum modules.
  const indirectCombineSM = device.createShaderModule({
    label: 'indirectCombine',
    code: composeWgsl(INDIRECT_COMBINE_MODULE, wgslModules),
  });
  const indirectTemporalAccumSM = device.createShaderModule({
    label: 'indirectTemporalAccum',
    code: composeWgsl(INDIRECT_TEMPORAL_ACCUM_MODULE, wgslModules),
  });
  const accumSM = device.createShaderModule({ label: 'accum', code: composeWgsl(TEMPORAL_ACCUM_MODULE, wgslModules) });
  await checkPipelineShaderModules([
    ['gtao', gtaoSM],
    ['gtao-upsample', gtaoUpsampleSM],
    ['indirectCombine', indirectCombineSM],
    ['indirectTemporalAccum', indirectTemporalAccumSM],
    ['accum', accumSM],
  ], opts?.onWarning);

  /**
   * Always-on compute pipelines (D3.15 extensibility table). Adding an
   * always-on pipeline = one row here + the corresponding field in
   * `CompiledPipelines`. Conditional / opt-in pipelines (risGi/temporalGi/
   * spatialGi/ppgUpdate/regirBuild) and the composite render pipeline stay
   * below because they require non-trivial per-pipeline conditional logic.
   */
  const PIPELINE_SPECS: ReadonlyArray<{
    readonly key: ComputePipelineKey;
    readonly module: GPUShaderModule;
    readonly layout: GPUPipelineLayout;
    readonly entryPoint: string;
    readonly constants?: Record<string, number>;
  }> = [
    { key: 'risPipeline',                  module: risSM,                layout: risLayout,                  entryPoint: 'risMain'                  },
    { key: 'temporalPipeline',             module: temporalSM,           layout: computeLayout,              entryPoint: 'temporalMain'             },
    { key: 'spatialPipeline',              module: spatialSM,            layout: computeLayout,              entryPoint: 'spatialMain', constants: { SPATIAL_ROUND_INDEX: 0 } },
    { key: 'spatialPipelineRoundTwo',      module: spatialSM,            layout: computeLayout,              entryPoint: 'spatialMain', constants: { SPATIAL_ROUND_INDEX: 1 } },
    { key: 'shadePipeline',                module: shadeSM,              layout: shadeLayout,                entryPoint: 'shadeMain'                },
    { key: 'motionVectorsPipeline',        module: motionVectorsSM,      layout: motionVectorsLayout,        entryPoint: 'motionVectorsMain'        },
    { key: 'atrousPipeline',               module: atrousSM,             layout: atrousLayout,               entryPoint: 'atrousMain'               },
    { key: 'sampleBudgetPipeline',         module: sampleBudgetSM,       layout: sampleBudgetLayout,         entryPoint: 'sampleBudgetKernel'       },
    { key: 'resolvePipeline',              module: resolveSM,            layout: resolveLayout,              entryPoint: 'resolveKernel'            },
    { key: 'cbPrefillPipeline',            module: cbPrefillSM,          layout: cbPrefillLayout,            entryPoint: 'cbPrefillKernel'          },
    { key: 'gtaoPipeline',                 module: gtaoSM,               layout: gtaoLayout,                 entryPoint: 'gtaoMain'                 },
    { key: 'gtaoUpsamplePipeline',         module: gtaoUpsampleSM,       layout: gtaoUpsampleLayout,         entryPoint: 'gtaoUpsampleMain'         },
    { key: 'indirectCombinePipeline',      module: indirectCombineSM,    layout: indirectCombineLayout,      entryPoint: 'indirectCombineMain'      },
    { key: 'transparentOitPipeline',       module: transparentOitSM,     layout: transparentOitLayout,       entryPoint: 'transparentOitMain'       },
    { key: 'indirectTemporalAccumPipeline',module: indirectTemporalAccumSM, layout: indirectTemporalAccumLayout, entryPoint: 'indirectTemporalAccumMain' },
    { key: 'indirectTemporalAccumResetPipeline', module: indirectTemporalAccumSM, layout: indirectTemporalAccumLayout, entryPoint: 'indirectTemporalAccumResetMain' },
    { key: 'accumPipeline',                module: accumSM,              layout: accumLayout,                entryPoint: 'temporalAccumMain'        },
  ] as const;

  // Compile all always-on compute pipelines in parallel via the spec table.
  const pipelineDraft: Partial<CompiledPipelines> = {};
  await Promise.all(
    PIPELINE_SPECS.map(async ({ key, module, layout, entryPoint, constants }) => {
      pipelineDraft[key] = await device.createComputePipelineAsync({
        label: key.replace(/Pipeline$/, ''),
        layout,
        compute: {
          module,
          entryPoint,
          ...(constants !== undefined ? { constants } : {}),
        },
      });
    }),
  );

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
      ? composeWgsl(buildRisGiNrcModule(opts.nrcConfig!), wgslModules)
      : composeWgsl(RIS_GI_MODULE, wgslModules),
  });
  await checkPipelineShaderModules([['risGi', risGiSM]], opts?.onWarning);
  const risGiLayout = device.createPipelineLayout({
    bindGroupLayouts: [
      getRisGiFrameBindGroupLayout(device, bglCache),
      getSceneBindGroupLayout(device, bglCache),
      getUboBindGroupLayout(device, bglCache),
      nrcOn
        ? getNrcHybridLayersBindGroupLayout(device, bglCache)
        : getHybridLayersBindGroupLayout(device, bglCache),
    ],
  });
  pipelineDraft['risGiPipeline'] = await device.createComputePipelineAsync({
    label: 'risGi', layout: risGiLayout,
    compute: { module: risGiSM, entryPoint: 'risGiMain' },
  });

  // GI temporal + spatial reuse always compile the generalized
  // reconnection-shift estimator. Both bind group(1) for material/visibility.
  const temporalGiSM = device.createShaderModule({
    label: 'temporalGi',
    code: composeWgsl(TEMPORAL_GI_MODULE, wgslModules),
  });
  const spatialGiSM = device.createShaderModule({
    label: 'spatialGi',
    code: composeWgsl(SPATIAL_GI_MODULE, wgslModules),
  });
  await checkPipelineShaderModules([
    ['temporalGi', temporalGiSM],
    ['spatialGi', spatialGiSM],
  ], opts?.onWarning);
  [pipelineDraft['temporalGiPipeline'], pipelineDraft['spatialGiPipeline']] = await Promise.all([
    device.createComputePipelineAsync({
      label: 'temporalGi', layout: temporalGiLayout,
      compute: { module: temporalGiSM, entryPoint: 'temporalGiMain' },
    }),
    device.createComputePipelineAsync({
      label: 'spatialGi', layout: spatialGiLayout,
      compute: { module: spatialGiSM, entryPoint: 'spatialGiMain' },
    }),
  ]);

  // Composite render pipeline (not in PIPELINE_SPECS — GPURenderPipeline, not compute).
  pipelineDraft['compositePipeline'] = await device.createRenderPipelineAsync({
    label: 'composite',
    layout: compositeLayout,
    vertex:   { module: compVertSM, entryPoint: 'vertMain' },
    fragment: {
      module: compFragSM,
      entryPoint: 'fragMain',
      targets: [{ format: swapChainFormat }],
      constants: hybridCompositeFragmentConstants(swapChainFormat),
    },
    primitive: { topology: 'triangle-list' },
  });

  // T2.H3 — PPG update pipeline (Müller 2017 §3.3): opt-in via opts.ppgEnabled.
  // Guided sampling itself is inlined in gi-ris via ppgPdf.wgsl; the update
  // kernel is the only standalone PPG training pass.
  // H29: the WGSL MAX_DTREE_NODES_PER_CELL is built from the live allocation
  // value (opts.ppgMaxDTreeNodesPerCell) so the shader and host stride agree.
  if (opts?.ppgEnabled) {
    const ppgMaxDTreeNodesPerCell = opts?.ppgMaxDTreeNodesPerCell
      ?? PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL;
    const ppgUpdateModule = {
      name: 'ppgUpdate' as const,
      source: buildPpgUpdateWgsl(ppgMaxDTreeNodesPerCell, reservoirGiStrideU32),
      requires: ['ppgTreeLayout'] as const,
    };
    const ppgUpdateSM = device.createShaderModule({ label: 'ppg-update', code: composeWgsl(ppgUpdateModule, wgslModules) });
    await checkPipelineShaderModules([['ppg-update', ppgUpdateSM]], opts?.onWarning);
    pipelineDraft['ppgUpdatePipeline'] = await device.createComputePipelineAsync({
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
  if (opts?.regirEnabled) {
    const regirBuildSM = device.createShaderModule({
      label: 'regir-build',
      code: composeWgsl(REGIR_BUILD_MODULE, wgslModules),
    });
    await checkPipelineShaderModules([['regir-build', regirBuildSM]], opts?.onWarning);
    const regirBuildLayout = device.createPipelineLayout({
      bindGroupLayouts: [getRegirBuildBindGroupLayout(device, bglCache)],
    });
    pipelineDraft['regirBuildPipeline'] = await device.createComputePipelineAsync({
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

  compositeShaderModulesByDevice.set(device, {
    vertex: compVertSM,
    fragment: compFragSM,
  });

  // All pipelines accumulated into pipelineDraft — assert completeness.
  return pipelineDraft as CompiledPipelines;
}

export function emitShaderCompilationWarnings(
  label: string,
  warnings: readonly GPUCompilationMessage[],
  options: {
    readonly onWarning?: (warning: EngineWarning) => void;
  } = {},
): void {
  const warning: EngineWarning = {
    code: 'walkaround-hybrid.shader-compilation-warning',
    backend: 'walkaround-hybrid',
    phase: 'construction',
    method: 'initialize',
    message: `[ReSTIR] Shader warnings in '${label}': ${warnings.map((w) => w.message).join('; ')}`,
    details: {
      shaderLabel: label,
      warnings: warnings.map((w) => ({
        type: w.type,
        message: w.message,
        lineNum: w.lineNum,
        linePos: w.linePos,
        offset: w.offset,
        length: w.length,
      })),
    },
  };
  if (options.onWarning !== undefined) {
    try {
      options.onWarning(warning);
    } catch {
      // Host warning callbacks must not break pipeline initialization.
    }
    return;
  }
  console.warn(`[ReSTIR] Shader warnings in '${label}':`, warnings.map((w) => w.message));
}
