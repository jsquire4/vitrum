// pt-webgpu factory option validation + warning sequence (T3-B god-file split, 2026-07-20).
//
// Extracted verbatim from `createPTEngine_WebGPU` in `index.ts` (~240 lines of
// throw/warn validation). Behaviour is byte-identical: the factory now calls
// `validatePtWebgpuOptions(opts)` which performs the same ordered sequence of
// structural throws + structured `onWarning` emissions and returns the resolved
// `{ traceTier, effectiveOpts, resolvedBdptMaxLightBounces }` the factory then
// hands to the engine constructor.
//
// Shared numeric caps and the small pure resolvers (`emitPteWarning`,
// `resolveBdptMaxLightBounces`, etc.) also live here now — the engine class in
// `index.ts` re-imports them, so there is a single source of truth.

import type { EngineWarning } from '@vitrum/core';
import type { PTEngineWebGPUOptions } from './index.js';
import { resolvePtWebgpuTraceTier, type PtWebgpuTraceTier } from './traceTier.js';
import {
  PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_CWBVH_CLOSEST_RESTIR_PT_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
} from './webgpuLimits.js';

export const EXPERIMENTAL_MAX_BOUNCES = 8;
export const BDPT_MAX_LIGHT_BOUNCES = 8;
// D2 (2026-07-20): raised 1 → 2 unconditionally. With maxLv=2 the kernel
// connection loop `for lvi=1u; lvi<maxLv` executes lvi=1, so BDPT does real
// light-path connections out of the box instead of being silently inert at the
// old default of 1 (which performed zero connections). BDPT remains opt-in
// (`bdpt:true`); this only fixes its default light-bounce count.
export const BDPT_SAFE_DEFAULT_LIGHT_BOUNCES = 2;

export function emitPteWarning(
  opts: Pick<PTEngineWebGPUOptions, 'onWarning'>,
  warning: EngineWarning,
  ...consoleArgs: readonly unknown[]
): void {
  console.warn(...(consoleArgs.length > 0 ? consoleArgs : [warning.message]));
  try {
    opts.onWarning?.(warning);
  } catch {
    // Host warning callbacks must not break engine construction.
  }
}

function hasOidnModelUrl(opts: Pick<PTEngineWebGPUOptions, 'oidn'>): boolean {
  return typeof opts.oidn?.modelUrl === 'string' && opts.oidn.modelUrl.length > 0;
}

function resolvePtWebgpuAutoDenoiser(opts: PTEngineWebGPUOptions): PTEngineWebGPUOptions {
  if (opts.denoiser !== 'auto') return opts;
  const resolved = hasOidnModelUrl(opts) ? 'oidn-final' : 'none';
  const reason = resolved === 'oidn-final' ? 'host-oidn-model-url' : 'no-host-model-assets';
  emitPteWarning(opts, {
    code: 'pt-webgpu.denoiser-auto-resolved',
    backend: 'pt-webgpu',
    phase: 'construction',
    method: 'createPTEngine_WebGPU',
    message:
      `[vitrum/pt-webgpu] denoiser:'auto' resolved to '${resolved}' (${reason}). ` +
      `pt-webgpu ships no OIDN model; provide oidn.modelUrl to enable the async final-pass OIDN denoiser.`,
    details: {
      requested: 'auto',
      resolved,
      reason,
      packageProvidesProductionWeights: false,
    },
  });
  return { ...opts, denoiser: resolved };
}

export function resolveBdptMaxLightBounces(requested: number | undefined): number {
  if (requested === undefined) return BDPT_SAFE_DEFAULT_LIGHT_BOUNCES;
  return Math.min(BDPT_MAX_LIGHT_BOUNCES, Math.floor(requested));
}

function assertRestirPtReuseSupported(device: GPUDevice, traceTier: PtWebgpuTraceTier): void {
  if (traceTier !== 'full') {
    throw new Error(
      'createPTEngine_WebGPU: restirPtReuse requires traceTier "full"; the selected lite tier cannot bind the ReSTIR-PT reuse reservoirs.',
    );
  }
  const maxBuffers = device.limits.maxStorageBuffersPerShaderStage;
  if (maxBuffers < PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE) {
    throw new Error(
      `createPTEngine_WebGPU: restirPtReuse requires maxStorageBuffersPerShaderStage >= ` +
      `${PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE}; device exposes ${maxBuffers}. ` +
      'Request the ReSTIR-PT reuse limit floor when acquiring the GPUDevice.',
    );
  }
}

function assertCwbvhClosestSupported(
  device: GPUDevice,
  traceTier: PtWebgpuTraceTier,
  restirPtReuse: boolean,
): void {
  if (traceTier !== 'full') {
    throw new Error(
      "createPTEngine_WebGPU: bvhTraversal:'cwbvh-closest-experimental' requires traceTier \"full\"; the selected lite tier does not bind full-tier TLAS/material/CWBVH groups.",
    );
  }
  const required = restirPtReuse
    ? PT_WEBGPU_CWBVH_CLOSEST_RESTIR_PT_REQUIRED_STORAGE_BUFFERS_PER_STAGE
    : PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE;
  const maxBuffers = device.limits.maxStorageBuffersPerShaderStage;
  if (maxBuffers < required) {
    throw new Error(
      "createPTEngine_WebGPU: bvhTraversal:'cwbvh-closest-experimental' requires " +
      `maxStorageBuffersPerShaderStage >= ${required}; device exposes ${maxBuffers}. ` +
      'Request the CWBVH traversal limit floor when acquiring the GPUDevice, or omit bvhTraversal to use the binary BVH.',
    );
  }
}

/** The resolved values `createPTEngine_WebGPU` needs after validation. */
export interface ValidatedPtWebgpuOptions {
  readonly traceTier: PtWebgpuTraceTier;
  readonly effectiveOpts: PTEngineWebGPUOptions;
  readonly resolvedBdptMaxLightBounces: number;
}

/**
 * Validate + normalize `createPTEngine_WebGPU` options: perform every structural
 * throw and emit every construction-time `onWarning`, then return the resolved
 * trace tier, denoiser-resolved effective options, and BDPT light-bounce count.
 * Byte-identical to the former inline factory body (T3-B extraction).
 */
export function validatePtWebgpuOptions(opts: PTEngineWebGPUOptions): ValidatedPtWebgpuOptions {
  if (opts.device == null || typeof (opts.device).createCommandEncoder !== 'function') {
    throw new TypeError(
      'createPTEngine_WebGPU: device must be a GPUDevice instance',
    );
  }
  const maxBounces = opts.maxBounces;
  if (maxBounces !== undefined && maxBounces < 1) {
    throw new RangeError(
      `createPTEngine_WebGPU: maxBounces structural cap must be >= 1 (got ${maxBounces})`,
    );
  }
  const maxSpp = opts.maxSamplesPerPixel;
  if (maxSpp !== undefined && maxSpp < 1) {
    throw new RangeError(
      `createPTEngine_WebGPU: maxSamplesPerPixel structural cap must be >= 1 (got ${maxSpp})`,
    );
  }
  if (maxBounces !== undefined && maxBounces > EXPERIMENTAL_MAX_BOUNCES) {
    emitPteWarning(opts, {
      code: 'pt-webgpu.max-bounces-clamped',
      backend: 'pt-webgpu',
      phase: 'construction',
      method: 'createPTEngine_WebGPU',
      message: `[vitrum/pt-webgpu] maxBounces=${maxBounces} requested, clamping to experimental limit ${EXPERIMENTAL_MAX_BOUNCES}.`,
      details: { requested: maxBounces, clampedTo: EXPERIMENTAL_MAX_BOUNCES },
    });
  }
  const bdptMaxLightBounces = opts.bdptOptions?.maxLightBounces;
  if (
    bdptMaxLightBounces !== undefined &&
    (!Number.isFinite(bdptMaxLightBounces) || bdptMaxLightBounces < 1)
  ) {
    throw new RangeError(
      `createPTEngine_WebGPU: bdptOptions.maxLightBounces must be a finite number >= 1 (got ${bdptMaxLightBounces})`,
    );
  }
  if (bdptMaxLightBounces !== undefined && bdptMaxLightBounces > BDPT_MAX_LIGHT_BOUNCES) {
    emitPteWarning(opts, {
      code: 'pt-webgpu.bdpt-max-light-bounces-clamped',
      backend: 'pt-webgpu',
      phase: 'construction',
      method: 'createPTEngine_WebGPU',
      message:
        `[vitrum/pt-webgpu] bdptOptions.maxLightBounces=${bdptMaxLightBounces} requested, ` +
        `clamping to supported BDPT light-subpath limit ${BDPT_MAX_LIGHT_BOUNCES}.`,
      details: { requested: bdptMaxLightBounces, clampedTo: BDPT_MAX_LIGHT_BOUNCES },
    });
  }
  if (
    bdptMaxLightBounces !== undefined &&
    bdptMaxLightBounces <= BDPT_MAX_LIGHT_BOUNCES &&
    !Number.isInteger(bdptMaxLightBounces)
  ) {
    emitPteWarning(opts, {
      code: 'pt-webgpu.bdpt-max-light-bounces-rounded',
      backend: 'pt-webgpu',
      phase: 'construction',
      method: 'createPTEngine_WebGPU',
      message:
        `[vitrum/pt-webgpu] bdptOptions.maxLightBounces=${bdptMaxLightBounces} requested, ` +
        `rounding down to integer ${resolveBdptMaxLightBounces(bdptMaxLightBounces)}.`,
      details: {
        requested: bdptMaxLightBounces,
        roundedTo: resolveBdptMaxLightBounces(bdptMaxLightBounces),
      },
    });
  }
  const resolvedBdptMaxLightBounces = resolveBdptMaxLightBounces(bdptMaxLightBounces);
  const traceTier = resolvePtWebgpuTraceTier(opts.device, opts.traceTier);
  const effectiveOpts = resolvePtWebgpuAutoDenoiser(opts);
  if (
    opts.bdpt === true &&
    traceTier === 'full' &&
    resolvedBdptMaxLightBounces > BDPT_SAFE_DEFAULT_LIGHT_BOUNCES
  ) {
    if (opts.bdptOptions?.experimentalMultiVertex !== true) {
      throw new RangeError(
        `createPTEngine_WebGPU: bdptOptions.maxLightBounces > ${BDPT_SAFE_DEFAULT_LIGHT_BOUNCES} activates the multi-vertex BDPT research path; ` +
        `set bdptOptions.experimentalMultiVertex=true to opt in, or omit maxLightBounces for the safe default (${BDPT_SAFE_DEFAULT_LIGHT_BOUNCES}).`,
      );
    }
    emitPteWarning(opts, {
      code: 'pt-webgpu.bdpt-multivertex-research-mode',
      backend: 'pt-webgpu',
      phase: 'construction',
      method: 'createPTEngine_WebGPU',
      message:
        `[vitrum/pt-webgpu] bdptOptions.maxLightBounces=${resolvedBdptMaxLightBounces} enables the ` +
        'multi-vertex BDPT research path. Current radiometric evidence shows this path is not yet ' +
        'weighted against the regular eye-path strategy, so it requires experimentalMultiVertex:true; omit ' +
        'bdptOptions.maxLightBounces for the endpoint-only radiometrically neutral default.',
      details: {
        requested: bdptMaxLightBounces,
        resolved: resolvedBdptMaxLightBounces,
        safeDefault: BDPT_SAFE_DEFAULT_LIGHT_BOUNCES,
        experimentalMultiVertex: true,
        promotionReady: false,
        currentEstimator: 'additive-sidecar-not-weighted-against-eye-path',
        blocker: 'not-weighted-against-regular-eye-path-strategy',
        requiredEstimator: 'multi-vertex-light-subpath-strategies-weighted-against-regular-eye-path-strategy',
        safeAlternative: `omit bdptOptions.maxLightBounces or set maxLightBounces:${BDPT_SAFE_DEFAULT_LIGHT_BOUNCES}`,
        evidencePath: 'tools/radiometric-ab/results-bdpt.json',
      },
    });
  }
  if (
    effectiveOpts.denoiser != null &&
    effectiveOpts.denoiser !== 'none' &&
    effectiveOpts.denoiser !== 'oidn-final'
  ) {
    emitPteWarning(effectiveOpts, {
      code: 'pt-webgpu.unsupported-denoiser',
      backend: 'pt-webgpu',
      phase: 'construction',
      method: 'createPTEngine_WebGPU',
      message:
        `[vitrum/pt-webgpu] denoiser="${effectiveOpts.denoiser}" requested, but only 'none' and 'oidn-final' are wired. ` +
        "pt-webgpu is a converged progressive path tracer; 'svgf-real' is a real-time 1-spp filter and is unsupported here — " +
        "use 'oidn-final' for converged denoising. Degrading to no-denoise.",
      details: { requested: effectiveOpts.denoiser },
    });
  }
  const requestedSampling = (opts as { readonly sampling?: unknown }).sampling;
  if (
    requestedSampling != null &&
    requestedSampling !== 'pcg' &&
    requestedSampling !== 'sobol'
  ) {
    emitPteWarning(opts, {
      code: 'pt-webgpu.unsupported-sampling',
      backend: 'pt-webgpu',
      phase: 'construction',
      method: 'createPTEngine_WebGPU',
      message:
        `[vitrum/pt-webgpu] sampling="${String(requestedSampling)}" requested, but only ` +
        "'pcg' and 'sobol' are wired. Degrading to the default PCG stream.",
      details: { requested: requestedSampling },
    });
  }
  if (requestedSampling === 'sobol') {
    emitPteWarning(opts, {
      code: 'pt-webgpu.sobol-sampling-experimental',
      backend: 'pt-webgpu',
      phase: 'construction',
      method: 'createPTEngine_WebGPU',
      message:
        "[vitrum/pt-webgpu] sampling:'sobol' enables an opt-in hash-based " +
        'Owen-scrambled Sobol RNG across the pt-webgpu megakernel and auxiliary ' +
        'SPPM/ReSTIR-PT/BDPT pipelines with a tiled ranked rotation. The ' +
        'dimension-assignment audit is pinned; WSL-lite RMSE evidence is bounded, ' +
        'while full-tier default-promotion evidence remains a Road-to-100 tail.',
      details: {
        sampling: 'sobol',
        fallback: 'none',
        rotation: 'ranked-8x8',
        promotionTails: ['equal-time-rmse-ab'],
      },
    });
  }
  const requestedBvhTraversal = (opts as { readonly bvhTraversal?: unknown }).bvhTraversal;
  const cwbvhClosestRequested = requestedBvhTraversal === 'cwbvh-closest-experimental';
  if (
    requestedBvhTraversal != null &&
    requestedBvhTraversal !== 'binary' &&
    requestedBvhTraversal !== 'cwbvh-closest-experimental'
  ) {
    emitPteWarning(opts, {
      code: 'pt-webgpu.unsupported-bvh-traversal',
      backend: 'pt-webgpu',
      phase: 'construction',
      method: 'createPTEngine_WebGPU',
      message:
        `[vitrum/pt-webgpu] bvhTraversal="${String(requestedBvhTraversal)}" requested, but only ` +
        "'binary' and 'cwbvh-closest-experimental' are wired. Degrading to binary traversal.",
      details: { requested: requestedBvhTraversal, fallback: 'binary' },
    });
  }
  // H51-C: warn once listing any extensions keys the host supplied that are
  // either (a) graduated legacy keys that no longer do anything, or (b) truly
  // unknown keys. In both cases the key is silently ignored at runtime; the
  // warn ensures the host is aware that migration is needed.
  //
  // pt-webgpu graduated its formerly-experimental extensions in 2025:
  //   vitrum.ptWebgpu.spectralHeroWavelength.*  →  opts.spectral (boolean)
  //   vitrum.ptWebgpu.bdpt.*                    →  opts.bdpt (boolean) + opts.bdptOptions
  //   vitrum.ptWebgpu.oidn.*                    →  opts.denoiser:'oidn-final' + opts.oidn
  if (opts.extensions != null) {
    const GRADUATED_KEY_MIGRATION: Record<string, string> = {
      'vitrum.ptWebgpu.spectralHeroWavelength':
        "opts.spectral (boolean) — set spectral: true to enable hero-wavelength spectral transport",
      'vitrum.ptWebgpu.bdpt':
        "opts.bdpt (boolean) + opts.bdptOptions — set bdpt: true to enable bidirectional path tracing",
      'vitrum.ptWebgpu.oidn':
        "opts.denoiser: 'oidn-final' + opts.oidn: { modelUrl } — pass denoiser:'oidn-final' with an OIDN model URL",
    };
    const allKeys = Object.keys(opts.extensions);
    for (const [prefix, migration] of Object.entries(GRADUATED_KEY_MIGRATION)) {
      const matchingKeys = allKeys.filter((k) => k.startsWith(prefix));
      if (matchingKeys.length > 0) {
        emitPteWarning(opts, {
          code: 'pt-webgpu.graduated-extension-key',
          backend: 'pt-webgpu',
          phase: 'construction',
          method: 'createPTEngine_WebGPU',
          message:
            `[vitrum/pt-webgpu] Extension key(s) ${matchingKeys.map((k) => JSON.stringify(k)).join(', ')} ` +
            `are no longer consumed — this key graduated to a first-class option. ` +
            `Replace with: ${migration}.`,
          details: { keys: matchingKeys, migration },
        });
      }
    }
    const unknownKeys = allKeys.filter(
      (k) => !Object.keys(GRADUATED_KEY_MIGRATION).some((prefix) => k.startsWith(prefix)),
    );
    if (unknownKeys.length > 0) {
      emitPteWarning(opts, {
        code: 'pt-webgpu.unknown-extension-key',
        backend: 'pt-webgpu',
        phase: 'construction',
        method: 'createPTEngine_WebGPU',
        message:
          `[vitrum/pt-webgpu] Unknown extensions keys will be ignored: ${unknownKeys.map((k) => JSON.stringify(k)).join(', ')}. ` +
          "pt-webgpu's stable extensions (spectral, bdpt, oidn, restirPtReuse) are now first-class " +
          'named options. Check the PTEngineWebGPUOptions interface for the current option set.',
        details: { keys: unknownKeys },
      });
    }
  }

  if (opts.restirPtReuse === true) {
    assertRestirPtReuseSupported(opts.device, traceTier);
    if (opts.restirPtReuseOptions?.experimentalGlossyReuse === true) {
      emitPteWarning(opts, {
        code: 'pt-webgpu.restir-pt-glossy-reuse-research-mode',
        backend: 'pt-webgpu',
        phase: 'construction',
        method: 'createPTEngine_WebGPU',
        message:
          '[vitrum/pt-webgpu] restirPtReuseOptions.experimentalGlossyReuse=true admits glossy/metallic visible vertices into the ' +
          'ReSTIR-PT temporal/spatial feedback loop. Current radiometric evidence marks this branch as a non-promotable research finding; ' +
          'omit experimentalGlossyReuse for the diffuse-safe validated default.',
        details: {
          restirPtReuse: true,
          experimentalGlossyReuse: true,
          promotionReady: false,
          blocker: 'glossy-visible-vertex-reuse-outside-diffuse-safe-validation-envelope',
          evidencePath: 'tools/radiometric-ab/results-restir-pt-glossy-research.json',
        },
      });
    }
  }
  if (cwbvhClosestRequested) {
    assertCwbvhClosestSupported(opts.device, traceTier, opts.restirPtReuse === true);
    emitPteWarning(opts, {
      code: 'pt-webgpu.cwbvh-closest-experimental',
      backend: 'pt-webgpu',
      phase: 'construction',
      method: 'createPTEngine_WebGPU',
      message:
        "[vitrum/pt-webgpu] bvhTraversal:'cwbvh-closest-experimental' routes full-tier closest-hit and any-hit mesh traversal through the uploaded CWBVH forest. " +
        'The any-hit wrapper preserves castShadow:false predicate parity; renderer parity/performance A/B is still required before default promotion.',
      details: {
        traversal: 'cwbvh-closest-experimental',
        closestHit: 'cwbvh',
        anyHit: 'cwbvh-cast-shadow-aware',
        requiredStorageBuffersPerStage: opts.restirPtReuse === true
          ? PT_WEBGPU_CWBVH_CLOSEST_RESTIR_PT_REQUIRED_STORAGE_BUFFERS_PER_STAGE
          : PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
      },
    });
  }
  if (traceTier === 'full') {
    console.info(
      '[vitrum/pt-webgpu] Full trace tier: TLAS, analytic shapes, HDRI, area lights, motion/variance aux, caustics.',
    );
  } else {
    emitPteWarning(opts, {
      code: 'pt-webgpu.lite-tier',
      backend: 'pt-webgpu',
      phase: 'construction',
      method: 'createPTEngine_WebGPU',
      message:
        '[vitrum/pt-webgpu] Lite trace tier (software-adapter fallback): merged mesh-like BVH, directional/point/spot/rect-area/disc-area emitters, HDRI and procedural-sky environments. ' +
        'Disabled on lite: analytic shapes, TLAS, mesh-area emitters, caustics, BDPT, and motion/variance aux buffers. ' +
        `On a discrete GPU host, request a device with maxStorageBuffersPerShaderStage >= ${PT_WEBGPU_FULL_REQUIRED_STORAGE_BUFFERS_PER_STAGE} and maxStorageTexturesPerShaderStage >= 5, or pass traceTier: "full" after verifying limits.`,
      details: { traceTier },
    });
  }
  return { traceTier, effectiveOpts, resolvedBdptMaxLightBounces };
}
