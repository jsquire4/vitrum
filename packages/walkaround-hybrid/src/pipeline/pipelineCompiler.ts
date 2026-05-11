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
import { SURFACE_TEXTURES_WGSL } from '../shaders/surfaceTextures.wgsl.js';
import { DDGI_SAMPLE_WGSL } from '../ddgi/ddgiSampleWgsl.js';
import {
  injectPpgBindingsIntoShadeWgsl,
  injectPpgRecordBeforeHdrStore,
} from '../shaders/shadePpgTrain.wgsl.js';
import {
  injectPpgGuideBounceIntoShadeWgsl,
  injectPpgGuideDeclsIntoShadeWgsl,
} from '../shaders/shadePpgGuide.wgsl.js';
import { PPG_UPDATE_WGSL } from '../ppg/wgsl/ppgUpdate.wgsl.js';
import { ATROUS_WGSL, SVGF_WGSL, TEMPORAL_ACCUM_WGSL } from '@vitrum/shared-denoisers';
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
  getHybridLayersBindGroupLayoutWithPpg,
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
  denoiserMode: 'atrous' | 'svgf';
  welfordPipeline?: GPUComputePipeline;
  svgfVariancePipeline?: GPUComputePipeline;
  svgfAtrousPipeline?: GPUComputePipeline;
  /** Sprint 11 — path guiding training (shade writes samples; dispatch follows). */
  ppgEnabled: boolean;
  ppgUpdatePipeline?: GPUComputePipeline;
}

export async function compilePipelines(
  device: GPUDevice,
  bglCache: BGLCache,
  swapChainFormat: GPUTextureFormat,
  opts?: { verbose?: boolean; denoiser?: 'atrous' | 'svgf'; ppgEnabled?: boolean },
): Promise<CompiledPipelines> {
  const denoiserMode = opts?.denoiser ?? 'svgf';
  const ppgOn = opts?.ppgEnabled === true;
  // Compile all shader modules (common WGSL is prepended to each ReSTIR pass).
  const risSM      = device.createShaderModule({ label: 'ris',      code: COMMON_WGSL + RIS_WGSL });
  const temporalSM = device.createShaderModule({ label: 'temporal', code: COMMON_WGSL + TEMPORAL_WGSL });
  const spatialSM  = device.createShaderModule({ label: 'spatial',  code: COMMON_WGSL + SPATIAL_WGSL });
  const shadeWgslBody = ppgOn
    ? injectPpgRecordBeforeHdrStore(
        injectPpgGuideBounceIntoShadeWgsl(
          injectPpgGuideDeclsIntoShadeWgsl(
            injectPpgBindingsIntoShadeWgsl(SHADE_WGSL),
          ),
        ),
      )
    : SHADE_WGSL;
  const shadeSM    = device.createShaderModule({ label: 'shade',    code: COMMON_WGSL + SURFACE_TEXTURES_WGSL + DDGI_SAMPLE_WGSL + shadeWgslBody });
  const atrousSM   = device.createShaderModule({ label: 'atrous',   code: COMMON_WGSL + ATROUS_WGSL });
  const compVertSM = device.createShaderModule({ label: 'comp-vert', code: COMPOSITE_VERT_WGSL });
  const compFragSM = device.createShaderModule({ label: 'comp-frag', code: COMPOSITE_FRAG_WGSL });

  const ppgUpdateSM = ppgOn
    ? device.createShaderModule({ label: 'ppg-update', code: COMMON_WGSL + PPG_UPDATE_WGSL })
    : null;

  // Check for compile errors on every shader module before proceeding.
  const welfordSM =
    denoiserMode === 'svgf'
      ? device.createShaderModule({ label: 'welford-temporal', code: COMMON_WGSL + WELFORD_TEMPORAL_WGSL })
      : null;
  // SVGF_WGSL is self-contained: it declares its own PI, INV_PI, LUM_W, and
  // WelfordVariance struct (via WELFORD_VARIANCE_WGSL). Do NOT prepend
  // COMMON_WGSL here — it would cause WGSL redeclaration errors on those
  // names. The welford-temporal pass DOES need COMMON_WGSL (for the BVH /
  // shared math helpers), which is why those two diverge.
  const svgfSM =
    denoiserMode === 'svgf'
      ? device.createShaderModule({ label: 'svgf', code: SVGF_WGSL })
      : null;

  const modules: [string, GPUShaderModule][] = [
    ['ris', risSM], ['temporal', temporalSM], ['spatial', spatialSM],
    ['shade', shadeSM], ['atrous', atrousSM],
    ['comp-vert', compVertSM], ['comp-frag', compFragSM],
    ...(welfordSM ? [['welford', welfordSM] as [string, GPUShaderModule]] : []),
    ...(svgfSM ? [['svgf', svgfSM] as [string, GPUShaderModule]] : []),
    ...(ppgUpdateSM ? [['ppg-update', ppgUpdateSM] as [string, GPUShaderModule]] : []),
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
      ppgOn
        ? getHybridLayersBindGroupLayoutWithPpg(device, bglCache)
        : getHybridLayersBindGroupLayout(device, bglCache),
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

  let welfordPipeline: GPUComputePipeline | undefined;
  let svgfVariancePipeline: GPUComputePipeline | undefined;
  let svgfAtrousPipeline: GPUComputePipeline | undefined;
  if (denoiserMode === 'svgf' && welfordSM && svgfSM) {
    welfordPipeline = await device.createComputePipelineAsync({
      label: 'welford-temporal',
      layout: 'auto',
      compute: { module: welfordSM, entryPoint: 'welfordTemporalMain' },
    });
    svgfVariancePipeline = await device.createComputePipelineAsync({
      label: 'svgf-variance',
      layout: 'auto',
      compute: { module: svgfSM, entryPoint: 'svgfVarianceMain' },
    });
    svgfAtrousPipeline = await device.createComputePipelineAsync({
      label: 'svgf-atrous',
      layout: 'auto',
      compute: { module: svgfSM, entryPoint: 'svgfAtrousMain' },
    });
  }

  let ppgUpdatePipeline: GPUComputePipeline | undefined;
  if (ppgOn && ppgUpdateSM) {
    ppgUpdatePipeline = await device.createComputePipelineAsync({
      label: 'ppg-update',
      layout: 'auto',
      compute: { module: ppgUpdateSM, entryPoint: 'ppgUpdateKernel' },
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
    denoiserMode,
    ppgEnabled: ppgOn,
    ...(welfordPipeline !== undefined &&
    svgfVariancePipeline !== undefined &&
    svgfAtrousPipeline !== undefined
      ? { welfordPipeline, svgfVariancePipeline, svgfAtrousPipeline }
      : {}),
    ...(ppgUpdatePipeline !== undefined ? { ppgUpdatePipeline } : {}),
  };
}
