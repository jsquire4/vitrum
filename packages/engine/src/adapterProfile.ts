// @vitrum/engine — graceful-degradation adapter probe.
//
// `probeAdapterProfile()` produces the `@vitrum/core` `AdapterProfile` data
// shape from a live `GPUDevice`, a `GPUAdapter`, or (with no argument) a fresh
// `navigator.gpu.requestAdapter()` probe. It computes every capability verdict
// from the REAL backend limit thresholds — `HYBRID_WEBGPU_REQUIRED_LIMITS` /
// `HYBRID_LITE_LIMITS` (@vitrum/walkaround-hybrid) and pt-webgpu's own
// `selectPtWebgpuTraceTier` verdict — so there is exactly one source of truth
// for each threshold. No magic numbers are re-typed here (see the
// threshold-coupling guard test).
//
// Why this lives in @vitrum/engine and not @vitrum/core: @vitrum/core has zero
// package dependencies, but the thresholds live in walkaround-hybrid and
// pt-webgpu. @vitrum/engine already depends on all of them. The *type*
// (`AdapterProfile`) lives in @vitrum/core for host type-annotation ergonomics.

import {
  probeWebGPU,
  isSwiftShaderAdapter,
  extractGpuLimits,
  type AdapterProfile,
  type RealtimeTier,
  type HeroBackendRec,
  type PtWebgpuTierRec,
  type WgpuAdapterKind,
} from '@vitrum/core';
import {
  HYBRID_WEBGPU_REQUIRED_LIMITS,
  HYBRID_LITE_LIMITS,
} from '@vitrum/walkaround-hybrid';
import { selectPtWebgpuTraceTier } from '@vitrum/pt-webgpu';

/** Resolved limit-bag + adapter-info needed to compute a profile. Extracted so
 *  the three input shapes (no source / GPUDevice / GPUAdapter) converge on one
 *  verdict path. */
interface ResolvedAdapterFacts {
  readonly hasWebGPU: boolean;
  readonly limits: Record<string, number>;
  readonly info: { vendor?: string | null; architecture?: string | null } | null;
  readonly adapterKind: WgpuAdapterKind;
}

function isGpuDevice(source: GPUDevice | GPUAdapter): source is GPUDevice {
  // A GPUDevice has a `queue`; a GPUAdapter does not. Both expose `limits`.
  return typeof (source as { queue?: unknown }).queue !== 'undefined';
}

/** SwiftShader / unknown / hardware classification reused from the core
 *  heuristic, with the same empty-info → 'unknown' fallback as `probeWebGPU`. */
function classify(info: { vendor?: string | null; architecture?: string | null } | null): WgpuAdapterKind {
  if (info == null) return 'unknown';
  if (isSwiftShaderAdapter(info)) return 'swiftshader';
  const vendor = (info.vendor ?? '').length;
  const arch = (info.architecture ?? '').length;
  if (vendor === 0 && arch === 0) return 'unknown';
  return 'hardware';
}

async function resolveFacts(
  source?: GPUDevice | GPUAdapter,
): Promise<ResolvedAdapterFacts> {
  if (source == null) {
    // No source — run the canonical navigator probe. probeWebGPU already
    // extracts the limit bag + classifies the adapter, so reuse it whole.
    const probe = await probeWebGPU();
    if (!probe.supported) {
      return { hasWebGPU: false, limits: {}, info: null, adapterKind: 'unknown' };
    }
    return {
      hasWebGPU: true,
      limits: probe.limits ?? {},
      info: {
        ...(probe.vendor != null ? { vendor: probe.vendor } : {}),
        ...(probe.architecture != null ? { architecture: probe.architecture } : {}),
      },
      adapterKind: probe.adapterKind ?? 'unknown',
    };
  }

  // GPUDevice or GPUAdapter — read `.limits` directly. Adapter info is
  // best-effort: GPUAdapter exposes `.info`; GPUDevice exposes `.adapterInfo`
  // on some implementations. Absent info → 'unknown' (never falsely software).
  // Read by canonical name (portable across browsers' enumerable own-keys AND
  // native-wgpu non-enumerable getters — Deno/wgpu-py). See `extractGpuLimits`.
  const limits = extractGpuLimits((source as { limits?: unknown }).limits);
  let info: { vendor?: string | null; architecture?: string | null } | null = null;
  const rawInfo = isGpuDevice(source)
    ? (source as { adapterInfo?: { vendor?: string | null; architecture?: string | null } }).adapterInfo
    : (source as { info?: { vendor?: string | null; architecture?: string | null } }).info;
  if (rawInfo != null) {
    info = {
      ...(rawInfo.vendor != null ? { vendor: rawInfo.vendor } : {}),
      ...(rawInfo.architecture != null ? { architecture: rawInfo.architecture } : {}),
    };
  }
  return { hasWebGPU: true, limits, info, adapterKind: classify(info) };
}

/** Reuse pt-webgpu's OWN full/lite/none verdict by feeding it a device-shaped
 *  `{ limits }` stub. `selectPtWebgpuTraceTier` reads only the two storage
 *  limits and THROWS below the lite floor — we map that throw to `'none'`.
 *  Reusing the function (rather than re-deriving from re-typed thresholds) is
 *  what keeps the pt-webgpu thresholds single-sourced. */
function ptWebgpuTierFromLimits(buf: number, tex: number): PtWebgpuTierRec {
  const deviceStub = {
    limits: {
      maxStorageBuffersPerShaderStage: buf,
      maxStorageTexturesPerShaderStage: tex,
    },
  } as unknown as GPUDevice;
  try {
    return selectPtWebgpuTraceTier(deviceStub);
  } catch {
    return 'none';
  }
}

/**
 * Probe an adapter's graceful-degradation capabilities.
 *
 * @param source Optional live `GPUDevice` or `GPUAdapter`. When omitted, a
 *   fresh high-performance `navigator.gpu.requestAdapter()` probe runs.
 * @returns A pure-data {@link AdapterProfile}. Never throws — an unavailable /
 *   software adapter resolves to a profile with `recommendedRealtimeTier:
 *   'unavailable'` rather than an error, so hosts branch on data, not catches.
 */
export async function probeAdapterProfile(
  source?: GPUDevice | GPUAdapter,
): Promise<AdapterProfile> {
  const facts = await resolveFacts(source);

  // WebGL2 fallback presence — drives the hero-backend recommendation when
  // WebGPU is absent. A synchronous canvas probe (no async adapter request);
  // returns true in headless/no-DOM environments so a missing `document` does
  // not falsely downgrade the recommendation to 'none'.
  const hasWebGL2 = detectWebGL2Sync();

  if (!facts.hasWebGPU) {
    return Object.freeze<AdapterProfile>({
      hasWebGPU: false,
      hybridCapable: false,
      hybridLiteCapable: false,
      ptWebgpuTier: 'none',
      maxStorageBuffersPerStage: 0,
      maxStorageTexturesPerStage: 0,
      isSoftwareAdapter: false,
      adapterKind: facts.adapterKind,
      hasWebGL2,
      recommendedRealtimeTier: 'unavailable',
      recommendedHeroBackend: hasWebGL2 ? 'pt-webgl2' : 'none',
      limits: {},
    });
  }

  const buf = facts.limits['maxStorageBuffersPerShaderStage'] ?? 8;
  const tex = facts.limits['maxStorageTexturesPerShaderStage'] ?? 4;

  // Capability booleans — every threshold imported, never inlined.
  const hybridCapable =
    buf >= HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageBuffersPerShaderStage']! &&
    tex >= HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageTexturesPerShaderStage']!;
  const hybridLiteCapable =
    buf >= HYBRID_LITE_LIMITS['maxStorageBuffersPerShaderStage']! &&
    tex >= HYBRID_LITE_LIMITS['maxStorageTexturesPerShaderStage']!;
  const ptWebgpuTier = ptWebgpuTierFromLimits(buf, tex);

  const isSoftwareAdapter = isSwiftShaderAdapter(facts.info);

  const recommendedRealtimeTier = recommendRealtimeTier({
    isSoftwareAdapter,
    hybridCapable,
    hybridLiteCapable,
  });
  const recommendedHeroBackend = recommendHeroBackend(ptWebgpuTier, hasWebGL2);

  return Object.freeze<AdapterProfile>({
    hasWebGPU: true,
    hybridCapable,
    hybridLiteCapable,
    ptWebgpuTier,
    maxStorageBuffersPerStage: buf,
    maxStorageTexturesPerStage: tex,
    isSoftwareAdapter,
    adapterKind: facts.adapterKind,
    hasWebGL2,
    recommendedRealtimeTier,
    recommendedHeroBackend,
    limits: Object.freeze({ ...facts.limits }),
  });
}

/** Class A–E → preset ceiling. Pure; testable without a GPU. */
function recommendRealtimeTier(args: {
  isSoftwareAdapter: boolean;
  hybridCapable: boolean;
  hybridLiteCapable: boolean;
}): RealtimeTier {
  // Class D — never initialize hybrid on a software rasterizer (§4.4/§10.4),
  // even if its reported limits would pass.
  if (args.isSoftwareAdapter) return 'unavailable';
  // Class D — below the lite floor ⇒ hybrid unavailable.
  if (!args.hybridLiteCapable) return 'unavailable';
  // Class A — full limits, real hardware ⇒ ultra ceiling.
  if (args.hybridCapable) return 'ultra';
  // Class B/C — lite-only ⇒ medium ceiling (the lite path).
  return 'medium';
}

/** pt-webgpu tier + WebGL2 presence → hero-render backend recommendation. */
function recommendHeroBackend(
  ptWebgpuTier: PtWebgpuTierRec,
  hasWebGL2: boolean,
): HeroBackendRec {
  if (ptWebgpuTier === 'full') return 'pt-webgpu-full';
  if (ptWebgpuTier === 'lite') return 'pt-webgpu-lite';
  if (hasWebGL2) return 'pt-webgl2';
  return 'none';
}

/** Synchronous WebGL2 presence check (no async adapter request). Returns true
 *  in non-DOM environments (headless tests) so a missing `document` does not
 *  falsely report 'none' — the realtime/hero recommendation already gates on
 *  the WebGPU verdicts.
 *
 *  H31-e — the catch branch used to return `true`, which caused a false-positive
 *  `recommendedHeroBackend` on any exception thrown by `getContext('webgl2')` (e.g.
 *  a CSP violation or a WebGL-disabled worker). Return `false` on exception so the
 *  WebGL2 recommendation is conservative on real failures. The `document` undefined
 *  guard stays `true` — headless test environments without a DOM are intentionally
 *  assumed to support WebGL2 (test stubs handle that). */
function detectWebGL2Sync(): boolean {
  if (typeof document === 'undefined') return true;
  try {
    const canvas = document.createElement('canvas');
    return canvas.getContext('webgl2') != null;
  } catch {
    return false;
  }
}
