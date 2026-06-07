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
} from '@vitrum/walkaround-hybrid';
import {
  ptWebgpuRequiredLimitsForAdapter,
  mergeAdapterRequiredLimits,
} from '@vitrum/pt-webgpu';

import { probeAdapterProfile } from './adapterProfile.js';
import { computeProgressiveLimitUnion } from './createProgressiveEngine.js';

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
   *  target-derived limits (each entry is still clamped to the adapter's actual
   *  capability via `mergeAdapterRequiredLimits`, so an over-ask never makes the
   *  device request reject opaquely). Most hosts leave this unset and rely on
   *  `target`. */
  readonly requiredLimits?: Readonly<Record<string, number>>;

  /** `requiredFeatures` forwarded verbatim to `adapter.requestDevice`. The
   *  helper does not add any features itself. */
  readonly requiredFeatures?: readonly GPUFeatureName[];

  /** Optional `label` forwarded to `adapter.requestDevice` for debugging. */
  readonly label?: string;
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

  const requiredLimits = resolveRequiredLimits(adapter, profile, options);

  const device = await adapter.requestDevice({
    ...(requiredLimits != null ? { requiredLimits } : {}),
    ...(options.requiredFeatures != null
      ? { requiredFeatures: [...options.requiredFeatures] }
      : {}),
    ...(options.label != null ? { label: options.label } : {}),
  });

  const format =
    navigator.gpu.getPreferredCanvasFormat?.() ?? ('bgra8unorm' as GPUTextureFormat);

  return { adapter, device, format, profile };
}

/** Resolve the `requiredLimits` for the chosen target, clamped to the adapter.
 *  Returns `undefined` for `target: 'none'` with no explicit override (request
 *  the device with adapter defaults). */
function resolveRequiredLimits(
  adapter: GPUAdapter,
  profile: AdapterProfile,
  options: NegotiateWebGPUDeviceOptions,
): Record<string, number> | undefined {
  // An explicit override always wins, clamped to the adapter so an over-ask
  // can't make requestDevice reject opaquely.
  if (options.requiredLimits != null) {
    return mergeAdapterRequiredLimits(adapter, { ...options.requiredLimits });
  }

  const target = options.target ?? 'pt-webgpu';
  switch (target) {
    case 'none':
      return undefined;

    case 'pt-webgpu':
      // The factory's own adapter-aware tier resolver (full vs lite).
      return ptWebgpuRequiredLimitsForAdapter(adapter);

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
      return mergeAdapterRequiredLimits(
        adapter,
        profile.hybridCapable
          ? { ...HYBRID_WEBGPU_REQUIRED_LIMITS }
          : { ...HYBRID_LITE_LIMITS },
      );
    }

    case 'progressive': {
      // Both backends on one shared device — the limit UNION. Preflight the
      // adapter so the gap is named (matches createProgressiveEngine).
      const union = computeProgressiveLimitUnion();
      const unmet: string[] = [];
      for (const [key, wanted] of Object.entries(union)) {
        const cap = (adapter.limits as unknown as Record<string, number | undefined>)[key];
        if (typeof cap !== 'number' || cap < wanted) {
          unmet.push(`${key}: need ≥${wanted}, adapter has ${cap ?? 'undefined'}`);
        }
      }
      if (unmet.length > 0) {
        throw new Error(
          'negotiateWebGPUDevice: target "progressive" — this adapter cannot satisfy the ' +
            'device-limit UNION of the walkaround-hybrid + pt-webgpu backends required to run ' +
            'both on one shared device. Unmet limits: ' +
            unmet.join('; ') +
            '. Negotiate a single-backend device ("walkaround-hybrid" or "pt-webgpu") instead.',
        );
      }
      return union;
    }
  }
}
