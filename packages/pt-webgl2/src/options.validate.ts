// options.validate — factory option validation + construction-time warning emission
// for createPTEngine_WebGL2 (extracted from index.ts, T3-D / D11-1).
//
// BEHAVIOR-PRESERVING: the throw messages, the emitted EngineWarning
// [code, message, details] sequences, and the resolved `effectiveOpts` are
// byte-identical to the pre-extraction inline factory body. Pinned by
// factoryWarningsPin.test.ts + engineContract.test.ts.

import type { EngineWarning } from '@vitrum/core';
import type { PTEngineWebGL2Options } from './options.js';

// A5 — light-subpath ping-pong width (one column per light bounce). MUST match the
// `BDPT_MAX_LIGHT_BOUNCES=3` layout the GLSL light-subpath/connection kernels assume
// (bdpt_light_subpath.glsl.js header; the connection sweep caps the merged path at
// BDPT_MAX_MERGED=19 with n = c + e + 3, BDPT_MAX_EYE_DEPTH=8).
export const BDPT_SAFE_DEFAULT_LIGHT_BOUNCES = 1;
export const BDPT_MAX_LIGHT_BOUNCES = 3;

export function resolveBdptMaxLightBounces(value: number | undefined): number {
  if (value === undefined) return BDPT_SAFE_DEFAULT_LIGHT_BOUNCES;
  return Math.max(1, Math.min(BDPT_MAX_LIGHT_BOUNCES, Math.floor(value)));
}

export function emitWebgl2Warning(
  opts: Pick<PTEngineWebGL2Options, 'onWarning'>,
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

function hasOidnModelUrl(opts: Pick<PTEngineWebGL2Options, 'oidn'>): boolean {
  return typeof opts.oidn?.modelUrl === 'string' && opts.oidn.modelUrl.length > 0;
}

export function resolveWebgl2AutoDenoiser(opts: PTEngineWebGL2Options): PTEngineWebGL2Options {
  if (opts.denoiser !== 'auto') return opts;
  const resolved = hasOidnModelUrl(opts) ? 'oidn-final' : 'none';
  const reason = resolved === 'oidn-final' ? 'host-oidn-model-url' : 'no-host-model-assets';
  emitWebgl2Warning(opts, {
    code: 'pt-webgl2.denoiser-auto-resolved',
    backend: 'pt-webgl2',
    phase: 'construction',
    method: 'createPTEngine_WebGL2',
    message:
      `[vitrum/pt-webgl2] denoiser:'auto' resolved to '${resolved}' (${reason}). ` +
      `pt-webgl2 ships no OIDN model; provide oidn.modelUrl to enable the async final-pass OIDN denoiser.`,
    details: {
      requested: 'auto',
      resolved,
      reason,
      packageProvidesProductionWeights: false,
    },
  });
  return { ...opts, denoiser: resolved };
}

/**
 * Validate the factory options (throwing on hard errors), emit all construction
 * warnings, and resolve `denoiser:'auto'`. Returns the effective options the
 * engine constructor consumes. Byte-identical to the pre-extraction inline factory
 * validation body in createPTEngine_WebGL2.
 */
export function validateAndResolveWebgl2Options(
  opts: PTEngineWebGL2Options,
): PTEngineWebGL2Options {
  const gl = opts.device;
  if (gl == null || typeof gl.createFramebuffer !== 'function') {
    throw new TypeError('createPTEngine_WebGL2: device must be a WebGL2RenderingContext');
  }
  if (opts.maxBounces !== undefined && opts.maxBounces < 1) {
    throw new RangeError(`createPTEngine_WebGL2: maxBounces must be >= 1 (got ${opts.maxBounces})`);
  }
  if (opts.maxSamplesPerPixel !== undefined && opts.maxSamplesPerPixel < 1) {
    throw new RangeError(
      `createPTEngine_WebGL2: maxSamplesPerPixel must be >= 1 (got ${opts.maxSamplesPerPixel})`,
    );
  }
  if (
    opts.materialLodDepth !== undefined &&
    (!Number.isFinite(opts.materialLodDepth) || opts.materialLodDepth < 0)
  ) {
    throw new RangeError(
      `createPTEngine_WebGL2: materialLodDepth must be a finite number >= 0 (got ${opts.materialLodDepth})`,
    );
  }
  if (opts.causticStrategy === 'manifold-nee' || opts.causticStrategy === 'photon-map') {
    const approximation = opts.causticStrategy === 'manifold-nee'
      ? 'deterministic refraction-walk heuristic'
      : 'deterministic cone-traced photon estimate';
    emitWebgl2Warning(opts, {
      code: 'pt-webgl2.caustic-strategy-approximation',
      backend: 'pt-webgl2',
      phase: 'construction',
      method: 'createPTEngine_WebGL2',
      message:
        `[vitrum/pt-webgl2] causticStrategy="${opts.causticStrategy}" is an approximate WebGL2 ` +
        `${approximation}, not the promoted pt-webgpu MNEE/BDPT reference path. ` +
        'Use it for compatibility captures or approximate caustic hints; use pt-webgpu full tier for promoted caustic evidence.',
      details: {
        requested: opts.causticStrategy,
        approximation,
        fallback: 'approximate caustic contribution',
      },
    });
  }
  const bdptMaxLightBounces = opts.bdptOptions?.maxLightBounces;
  if (
    bdptMaxLightBounces !== undefined &&
    (!Number.isFinite(bdptMaxLightBounces) || bdptMaxLightBounces < 1)
  ) {
    throw new RangeError(
      `createPTEngine_WebGL2: bdptOptions.maxLightBounces must be a finite number >= 1 (got ${bdptMaxLightBounces})`,
    );
  }
  if (bdptMaxLightBounces !== undefined && bdptMaxLightBounces > BDPT_MAX_LIGHT_BOUNCES) {
    emitWebgl2Warning(opts, {
      code: 'pt-webgl2.bdpt-max-light-bounces-clamped',
      backend: 'pt-webgl2',
      phase: 'construction',
      method: 'createPTEngine_WebGL2',
      message:
        `[vitrum/pt-webgl2] bdptOptions.maxLightBounces=${bdptMaxLightBounces} requested, ` +
        `clamping to supported WebGL2 BDPT light-subpath limit ${BDPT_MAX_LIGHT_BOUNCES}.`,
      details: { requested: bdptMaxLightBounces, clampedTo: BDPT_MAX_LIGHT_BOUNCES },
    });
  }
  if (
    bdptMaxLightBounces !== undefined &&
    bdptMaxLightBounces <= BDPT_MAX_LIGHT_BOUNCES &&
    !Number.isInteger(bdptMaxLightBounces)
  ) {
    emitWebgl2Warning(opts, {
      code: 'pt-webgl2.bdpt-max-light-bounces-rounded',
      backend: 'pt-webgl2',
      phase: 'construction',
      method: 'createPTEngine_WebGL2',
      message:
        `[vitrum/pt-webgl2] bdptOptions.maxLightBounces=${bdptMaxLightBounces} requested, ` +
        `rounding down to integer ${resolveBdptMaxLightBounces(bdptMaxLightBounces)}.`,
      details: {
        requested: bdptMaxLightBounces,
        roundedTo: resolveBdptMaxLightBounces(bdptMaxLightBounces),
      },
    });
  }
  const resolvedBdptMaxLightBounces = resolveBdptMaxLightBounces(bdptMaxLightBounces);
  if (opts.bdpt === true && resolvedBdptMaxLightBounces > BDPT_SAFE_DEFAULT_LIGHT_BOUNCES) {
    if (opts.bdptOptions?.experimentalMultiVertex !== true) {
      throw new RangeError(
        'createPTEngine_WebGL2: bdptOptions.maxLightBounces > 1 activates the multi-vertex BDPT research path; ' +
        'set bdptOptions.experimentalMultiVertex=true to opt in, or omit maxLightBounces for the endpoint-only safe default.',
      );
    }
    emitWebgl2Warning(opts, {
      code: 'pt-webgl2.bdpt-multivertex-research-mode',
      backend: 'pt-webgl2',
      phase: 'construction',
      method: 'createPTEngine_WebGL2',
      message:
        `[vitrum/pt-webgl2] bdptOptions.maxLightBounces=${resolvedBdptMaxLightBounces} enables the ` +
        'multi-vertex BDPT research path. Current radiometric promotion evidence is endpoint-only, ' +
        'so this WebGL2 path requires experimentalMultiVertex:true; omit bdptOptions.maxLightBounces for the ' +
        'endpoint-only safe default.',
      details: {
        requested: bdptMaxLightBounces,
        resolved: resolvedBdptMaxLightBounces,
        safeDefault: BDPT_SAFE_DEFAULT_LIGHT_BOUNCES,
        experimentalMultiVertex: true,
      },
    });
  }
  if (
    opts.backgroundBlur !== undefined &&
    (!Number.isFinite(opts.backgroundBlur) || opts.backgroundBlur < 0)
  ) {
    throw new RangeError(
      `createPTEngine_WebGL2: backgroundBlur must be a finite number >= 0 (got ${opts.backgroundBlur})`,
    );
  }
  const effectiveOpts = resolveWebgl2AutoDenoiser(opts);
  // Unsupported realtime denoisers still degrade to no-denoise with a clear
  // warning. `oidn-final` is handled by the engine constructor because it is a
  // real asynchronous final-pass path that requires host-provided model config.
  if (effectiveOpts.denoiser != null && effectiveOpts.denoiser !== 'none' && effectiveOpts.denoiser !== 'oidn-final') {
    emitWebgl2Warning(effectiveOpts, {
      code: 'pt-webgl2.unsupported-denoiser',
      backend: 'pt-webgl2',
      phase: 'construction',
      method: 'createPTEngine_WebGL2',
      message:
        `[vitrum/pt-webgl2] denoiser="${effectiveOpts.denoiser}" requested, but pt-webgl2 only supports ` +
        '`none` and the asynchronous final-pass `oidn-final` denoiser. ' +
        'Degrading to no-denoise (denoiserState will report "disabled").',
      details: { requested: effectiveOpts.denoiser },
    });
  }
  // Item 22 — DOF × equirectangular regime guard (trust-remediation-plan §22).
  // Thin-lens DOF applied to equirectangular projection is physically undefined:
  // the blur direction has no meaning per sphere region (the GLSL DOF block
  // translates ray.origin by an aperture sample in camera space, but the equirect
  // ray directions span the full sphere — there is no consistent focal plane).
  // Silently generating blurry/incorrect output is worse than ignoring the option,
  // so we force DOF off for equirect and warn once so the host is not surprised.
  //
  // Orthographic + DOF is left as-is: the GLSL produces tilt-shift-style focus
  // (focalPoint = fixed point on the -Z frustum; new ray.direction = focalPoint -
  // shifted_origin) which is physically coherent — parallel projections plus an
  // aperture offset is the standard orthographic-camera DOF model.
  if (opts.cameraType === 'equirectangular' && opts.dof != null) {
    emitWebgl2Warning(opts, {
      code: 'pt-webgl2.equirectangular-dof-ignored',
      backend: 'pt-webgl2',
      phase: 'construction',
      method: 'createPTEngine_WebGL2',
      message:
        '[vitrum/pt-webgl2] dof is ignored when cameraType is "equirectangular". ' +
        'Thin-lens depth of field is physically undefined for full-sphere equirectangular ' +
        'projection (blur direction has no meaning per sphere region). ' +
        'The engine will render without DOF. Remove the dof option to suppress this warning.',
      details: { cameraType: opts.cameraType },
    });
  }
  return effectiveOpts;
}
