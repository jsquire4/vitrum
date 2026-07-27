// @vitrum/engine — raw WebGPU device negotiation helper.
//
// `negotiateWebGPUDevice()` is the lifecycle-layer peer of `attachVitrum`: a
// convenience that ACQUIRES a WebGPU adapter + device the host can hand to a
// backend factory (`createPTEngine_WebGPU`, `createWalkaroundEngine_Hybrid`),
// together with the preferred canvas format and the graceful-degradation
// {@link AdapterProfile}.
//
// HOST-OWNS-LIFECYCLE — the load-bearing rule of the whole library. This helper
// is a convenience over `navigator.gpu.requestAdapter` + `adapter.requestDevice`
// + `probeAdapterProfile`; it returns the device and then forgets it. The HOST
// owns the returned `device` and MUST `device.destroy()` it when done (after
// disposing any engine built on it). The helper:
//   • tracks NOTHING — it holds no reference, registers no dispose hook, and
//     wires no `device.lost` handler. There is no hidden ownership.
//   • is OPTIONAL — engines still accept a host-acquired device directly. A host
//     that already owns a device (e.g. shares it across subsystems) skips this
//     helper entirely and passes its own `GPUDevice` to the factory.
//   • does NOT bind a canvas. Swap-chain configuration is `configureWebGpuCanvas`
//     / `attachVitrum`'s job; `format` is returned for the host to configure its
//     own context if it presents directly.
//
// Limit selection reuses the SAME thresholds the backend factories use — no
// magic numbers are re-typed here (the `target` switch delegates to
// `HYBRID_WEBGPU_REQUIRED_LIMITS` / `HYBRID_LITE_LIMITS`,
// `ptWebgpuRequiredLimitsForAdapter`, and `computeProgressiveLimitUnion`).

import type { AdapterProfile } from '@vitrum/core';
import {
  HYBRID_WEBGPU_REQUIRED_LIMITS,
  HYBRID_LITE_LIMITS,
  nrcWebGpuRequiredFeaturesForConfig,
  nrcWebGpuRequiredLimitsForConfig,
  type NrcConfig,
} from '@vitrum/walkaround-hybrid';
import {
  ptWebgpuRequiredLimitsForAdapter,
  PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_CWBVH_CLOSEST_RESTIR_PT_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
  PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
  PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE,
  PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE,
} from '@vitrum/pt-webgpu';

import { probeAdapterProfile } from './adapterProfile.js';
import { checkProgressiveLimitUnion, computeProgressiveLimitUnion } from './createProgressiveEngine.js';

/** Which backend the negotiated device is intended for. Selects the
 *  `requiredLimits` requested at `adapter.requestDevice` so the device is built
 *  to the backend's floor — reusing the same thresholds the backend factories
 *  apply, never re-typed numbers.
 *
 *   • `'walkaround-hybrid'` — the realtime GI backend. Requests the FULL hybrid
 *     floor when the adapter is `hybridCapable`, else the LITE floor (matching
 *     `createEngine`'s walkaround constructor). Throws when the adapter is below
 *     the lite floor (`recommendedRealtimeTier === 'unavailable'`).
 *   • `'pt-webgpu'` — the converged WebGPU path tracer. Requests the highest
 *     tier the adapter can satisfy (`ptWebgpuRequiredLimitsForAdapter`).
 *   • `'progressive'` — BOTH backends on one shared device. Requests the limit
 *     UNION (`computeProgressiveLimitUnion`); throws if the adapter can't meet it.
 *   • `'none'` — request a device with NO `requiredLimits` (adapter defaults).
 *     For a host that only needs a bare device + profile and will validate
 *     limits itself. */
export type NegotiateTarget =
  | 'walkaround-hybrid'
  | 'pt-webgpu'
  | 'progressive'
  | 'none';

export interface NegotiateWebGPUDeviceOptions {
  /** A pre-acquired, HOST-OWNED adapter to reuse instead of requesting a fresh
   *  one. When given, this helper does NOT request a new adapter — it only
   *  requests a device against the one supplied (still host-owned). When absent,
   *  the helper runs `navigator.gpu.requestAdapter(...)` for convenience. */
  readonly adapter?: GPUAdapter;

  /** Power-preference hint for the adapter request (ignored when `adapter` is
   *  supplied). Default `'high-performance'` — vitrum's renderers are GPU-heavy. */
  readonly powerPreference?: GPUPowerPreference;

  /** Which backend the device targets — selects the `requiredLimits` (see
   *  {@link NegotiateTarget}). Default `'pt-webgpu'`. */
  readonly target?: NegotiateTarget;

  /** Explicit `requiredLimits` override. When provided, REPLACES the
   *  target-derived limits. Every entry is validated as a finite, non-negative
   *  safe integer and preflighted against the adapter; an unsatisfied or unknown
   *  limit rejects before device allocation. Most hosts leave this unset and
   *  rely on `target`. */
  readonly requiredLimits?: Readonly<Record<string, number>>;

  /** Additional features required from the adapter. Unsupported entries reject
   *  before device allocation. NRC adds `shader-f16` automatically when the
   *  resolved `nrcConfig.useF16` contract requires it. */
  readonly requiredFeatures?: readonly GPUFeatureName[];

  /** Optional `label` forwarded to `adapter.requestDevice` for debugging. */
  readonly label?: string;

  /**
   * For `target: 'pt-webgpu'` or `'progressive'` — include the ReSTIR-PT reuse
   * reservoir buffers in the device-limit union (raises the
   * `maxStorageBuffersPerShaderStage` floor to the ReSTIR-PT reuse tier).
   * Matches the `restirPtReuse` option on {@link CreateProgressiveEngineOptions}
   * so a host can build the device with `negotiateWebGPUDevice` and then pass
   * it to `createProgressiveEngine` with the same flag without a limit
   * mismatch. Ignored for all other targets.
   */
  readonly restirPtReuse?: boolean;

  /**
   * For `target: 'pt-webgpu'` or `'progressive'` — include the opt-in CWBVH
   * closest-hit traversal buffers in the required device floor. Must match the
   * converged backend's `bvhTraversal: 'cwbvh-closest'` option.
   * Ignored for all other targets.
   */
  readonly cwbvhClosest?: boolean;

  /**
   * For `target: 'walkaround-hybrid'` or `'progressive'` — include the
   * realtime engine's opt-in Neural Radiance Cache bind-group, buffer, and
   * workgroup-storage requirements. Must match `realtimeOptions.nrcEnabled`
   * when the negotiated device is passed to `createProgressiveEngine`.
   */
  readonly nrcEnabled?: boolean;
  /** NRC shape/precision used for dynamic limits and feature negotiation. */
  readonly nrcConfig?: Partial<NrcConfig>;
}

/** The negotiated, HOST-OWNED WebGPU handles. The caller is responsible for
 *  `device.destroy()` (after disposing any engine built on it). */
export interface NegotiatedWebGPUDevice {
  /** The adapter the device was requested from (the one passed in, or the one
   *  this helper acquired). Host-owned; nothing to dispose on an adapter. */
  readonly adapter: GPUAdapter;
  /** The newly-acquired device. HOST-OWNED — the host MUST `device.destroy()`
   *  it when finished. This helper holds no reference to it. */
  readonly device: GPUDevice;
  /** The browser's preferred canvas texture format
   *  (`navigator.gpu.getPreferredCanvasFormat()`), for the host to configure its
   *  own presentation context. `'bgra8unorm'` when the API is unavailable. */
  readonly format: GPUTextureFormat;
  /** The graceful-degradation {@link AdapterProfile} computed from the adapter
   *  (same data `createEngine`'s `onAdapterProfile` callback delivers). Lets the
   *  host pick a backend / tier from data. */
  readonly profile: AdapterProfile;
}

/**
 * Acquire a HOST-OWNED WebGPU adapter + device + preferred-format + capability
 * {@link AdapterProfile} as a convenience for hosts that drive the backend
 * factories directly (without `createEngine`). The returned device is owned by
 * the CALLER — see the module header / {@link NegotiatedWebGPUDevice.device}.
 *
 * @throws if WebGPU is unavailable (`navigator.gpu` undefined), if no adapter is
 *   available (and none was supplied), or if the chosen `target`'s required
 *   limits exceed the adapter's capability (a clear, gap-naming error rather
 *   than an opaque `requestDevice` rejection).
 */
export async function negotiateWebGPUDevice(
  options: NegotiateWebGPUDeviceOptions = {},
): Promise<NegotiatedWebGPUDevice> {
  const target = assertNegotiateTarget(options.target ?? 'pt-webgpu');

  if (typeof navigator === 'undefined' || navigator.gpu == null) {
    throw new Error(
      'negotiateWebGPUDevice: WebGPU is unavailable (navigator.gpu is undefined). ' +
        'Use a WebGL2 backend (createEngine({ prefer: "quality" })) on this host.',
    );
  }

  const adapter =
    options.adapter ??
    (await navigator.gpu.requestAdapter({
      powerPreference: options.powerPreference ?? 'high-performance',
    }));
  if (adapter == null) {
    throw new Error(
      'negotiateWebGPUDevice: navigator.gpu.requestAdapter() returned null (no WebGPU adapter available).',
    );
  }

  // Capability profile FIRST — it is pure data, never throws, and the
  // walkaround target gates on it. Reuses the single-source threshold logic.
  const profile = await probeAdapterProfile(adapter);

  const requiredLimits = resolveRequiredLimits(adapter, profile, options, target);
  const requiredFeatures = resolveRequiredFeatures(adapter, options, target);

  // Resolve presentation metadata before device allocation. A browser bug or
  // host shim that throws here must not strand a newly-created GPUDevice.
  const format =
    navigator.gpu.getPreferredCanvasFormat?.() ?? ('bgra8unorm');

  const device = await adapter.requestDevice({
    ...(requiredLimits != null ? { requiredLimits } : {}),
    ...(requiredFeatures.length > 0
      ? { requiredFeatures }
      : {}),
    ...(options.label != null ? { label: options.label } : {}),
  });

  return { adapter, device, format, profile };
}

/** Resolve and preflight the `requiredLimits` for the chosen target.
 *  Returns `undefined` for `target: 'none'` with no explicit override (request
 *  the device with adapter defaults). */
function resolveRequiredLimits(
  adapter: GPUAdapter,
  profile: AdapterProfile,
  options: NegotiateWebGPUDeviceOptions,
  target: NegotiateTarget,
): Record<string, number> | undefined {
  // An explicit override always wins, but it remains a guarantee: never weaken
  // a host's declared requirement to make requestDevice appear to succeed.
  if (options.requiredLimits != null) {
    return assertRequiredLimits(adapter, options.requiredLimits, 'explicit requiredLimits');
  }
  switch (target) {
    case 'none':
      return undefined;

    case 'pt-webgpu': {
      const maxBuffers = adapter.limits.maxStorageBuffersPerShaderStage;
      const maxTextures = adapter.limits.maxStorageTexturesPerShaderStage;
      if (
        maxBuffers < PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE ||
        maxTextures < PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE
      ) {
        throw new Error(
          'negotiateWebGPUDevice: target "pt-webgpu" — this adapter is below the ' +
            `lite-tier floor (${PT_WEBGPU_LITE_REQUIRED_STORAGE_BUFFERS_PER_STAGE} ` +
            'storage buffers/stage and ' +
            `${PT_WEBGPU_LITE_REQUIRED_STORAGE_TEXTURES_PER_STAGE} storage textures/stage); ` +
            `adapter exposes ${maxBuffers} and ${maxTextures}.`,
        );
      }

      const restirPtReuse = options.restirPtReuse === true;
      const cwbvhClosest = options.cwbvhClosest === true;
      const optionalBufferFloor = cwbvhClosest
        ? (restirPtReuse
            ? PT_WEBGPU_CWBVH_CLOSEST_RESTIR_PT_REQUIRED_STORAGE_BUFFERS_PER_STAGE
            : PT_WEBGPU_CWBVH_CLOSEST_REQUIRED_STORAGE_BUFFERS_PER_STAGE)
        : (restirPtReuse
            ? PT_WEBGPU_RESTIR_PT_REUSE_REQUIRED_STORAGE_BUFFERS_PER_STAGE
            : undefined);
      if (
        optionalBufferFloor != null &&
        (maxBuffers < optionalBufferFloor ||
          maxTextures < PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE)
      ) {
        throw new Error(
          'negotiateWebGPUDevice: target "pt-webgpu" cannot enable the requested ' +
            `${cwbvhClosest ? 'CWBVH closest-hit' : ''}` +
            `${cwbvhClosest && restirPtReuse ? ' + ' : ''}` +
            `${restirPtReuse ? 'ReSTIR-PT reuse' : ''} layout. ` +
            `Need at least ${optionalBufferFloor} storage buffers/stage and ` +
            `${PT_WEBGPU_FULL_REQUIRED_STORAGE_TEXTURES_PER_STAGE} storage textures/stage; ` +
            `adapter exposes ${maxBuffers} and ${maxTextures}.`,
        );
      }

      const required = ptWebgpuRequiredLimitsForAdapter(adapter, { restirPtReuse, cwbvhClosest });
      return assertRequiredLimits(adapter, required, 'target "pt-webgpu"');
    }

    case 'walkaround-hybrid': {
      // Mirror createEngine's walkaround constructor: full when hybridCapable,
      // else lite — and throw below the lite floor instead of letting the
      // device request fail opaquely.
      if (profile.recommendedRealtimeTier === 'unavailable') {
        throw new Error(
          'negotiateWebGPUDevice: target "walkaround-hybrid" — this adapter cannot run the ' +
            `realtime engine (recommendedRealtimeTier='unavailable', ` +
            `maxStorageBuffersPerStage=${profile.maxStorageBuffersPerStage}, ` +
            `maxStorageTexturesPerStage=${profile.maxStorageTexturesPerStage}, ` +
            `isSoftwareAdapter=${profile.isSoftwareAdapter}). ` +
            'Negotiate a "pt-webgpu" device or use a WebGL2 backend on this hardware.',
        );
      }
      const required = options.nrcEnabled === true
        ? nrcWebGpuRequiredLimitsForConfig(options.nrcConfig ?? {})
        : (profile.hybridCapable
            ? HYBRID_WEBGPU_REQUIRED_LIMITS
            : HYBRID_LITE_LIMITS);
      return assertRequiredLimits(adapter, required, 'target "walkaround-hybrid"');
    }

    case 'progressive': {
      // Both backends on one shared device — the limit UNION. Preflight the
      // adapter so the gap is named (matches createProgressiveEngine).
      // Forward restirPtReuse so a host negotiating a progressive device for
      // ReSTIR-PT reuse gets the higher buffer floor (matching the
      // createProgressiveEngine limit-union preflight).
      const restirPtReuse = options.restirPtReuse === true;
      const cwbvhClosest = options.cwbvhClosest === true;
      const nrcEnabled = options.nrcEnabled === true;
      const unionOptions = {
        restirPtReuse,
        cwbvhClosest,
        nrcEnabled,
        ...(options.nrcConfig !== undefined ? { nrcConfig: options.nrcConfig } : {}),
      };
      const union = computeProgressiveLimitUnion(unionOptions);
      const unmet = checkProgressiveLimitUnion(adapter, unionOptions);
      if (unmet.length > 0) {
        throw new Error(
          'negotiateWebGPUDevice: target "progressive" — this adapter cannot satisfy the ' +
            'device-limit UNION of the walkaround-hybrid + pt-webgpu backends required to run ' +
            'both on one shared device. Unmet limits: ' +
            unmet.join('; ') +
            '. Negotiate a single-backend device ("walkaround-hybrid" or "pt-webgpu") instead.',
        );
      }
      return assertRequiredLimits(adapter, union, 'target "progressive"');
    }
  }
}

const NEGOTIATE_TARGETS: readonly NegotiateTarget[] = [
  'walkaround-hybrid',
  'pt-webgpu',
  'progressive',
  'none',
];

function assertNegotiateTarget(value: unknown): NegotiateTarget {
  if (!NEGOTIATE_TARGETS.includes(value as NegotiateTarget)) {
    throw new TypeError(
      `negotiateWebGPUDevice: target must be one of ${NEGOTIATE_TARGETS.join(', ')}; ` +
        `received ${String(value)}.`,
    );
  }
  return value as NegotiateTarget;
}

function assertRequiredFeatures(
  adapter: GPUAdapter,
  requiredFeatures: readonly GPUFeatureName[] | undefined,
): void {
  if (requiredFeatures == null) return;
  for (const feature of requiredFeatures) {
    if (!adapter.features.has(feature)) {
      throw new Error(
        `negotiateWebGPUDevice: required feature "${String(feature)}" is not supported ` +
          'by this adapter.',
      );
    }
  }
}

function resolveRequiredFeatures(
  adapter: GPUAdapter,
  options: NegotiateWebGPUDeviceOptions,
  target: NegotiateTarget,
): GPUFeatureName[] {
  const required = new Set<GPUFeatureName>(options.requiredFeatures ?? []);
  if (
    options.nrcEnabled === true &&
    (target === 'walkaround-hybrid' || target === 'progressive')
  ) {
    for (const feature of nrcWebGpuRequiredFeaturesForConfig(options.nrcConfig ?? {})) {
      required.add(feature);
    }
  }
  const resolved = [...required];
  assertRequiredFeatures(adapter, resolved);
  return resolved;
}

// These are the two WebGPU limits whose smaller value is the stronger
// capability. Every other currently specified GPUSupportedLimits field is a
// maximum, where larger is stronger.
const MINIMUM_DIRECTION_LIMITS = new Set<string>([
  'minUniformBufferOffsetAlignment',
  'minStorageBufferOffsetAlignment',
]);

/** Validate a required-limit map without silently weakening the requirement. */
function assertRequiredLimits(
  adapter: GPUAdapter,
  required: Readonly<Record<string, number>>,
  context: string,
): Record<string, number> {
  const adapterLimits = adapter.limits as unknown as Record<string, unknown>;
  const validated: Record<string, number> = {};
  for (const [key, wanted] of Object.entries(required)) {
    if (!Number.isSafeInteger(wanted) || wanted < 0) {
      throw new TypeError(
        `negotiateWebGPUDevice: ${context}.${key} must be a finite, non-negative ` +
          `safe integer; received ${String(wanted)}.`,
      );
    }

    const supported = adapterLimits[key];
    if (typeof supported !== 'number' || !Number.isFinite(supported)) {
      throw new Error(
        `negotiateWebGPUDevice: ${context} names unknown or unreported adapter limit ` +
          `"${key}".`,
      );
    }

    const smallerIsStronger = MINIMUM_DIRECTION_LIMITS.has(key);
    const satisfied = smallerIsStronger ? supported <= wanted : supported >= wanted;
    if (!satisfied) {
      const relation = smallerIsStronger ? '<=' : '>=';
      throw new Error(
        `negotiateWebGPUDevice: ${context} requires ${key} ${relation} ${wanted}, ` +
          `but this adapter exposes ${supported}.`,
      );
    }
    validated[key] = wanted;
  }
  return validated;
}
