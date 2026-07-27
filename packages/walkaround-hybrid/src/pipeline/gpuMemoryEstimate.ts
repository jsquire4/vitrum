/**
 * GPU memory budget instrumentation for the walkaround-hybrid pipeline.
 *
 * Iterates the per-algorithm FrameResources sub-structs created by
 * `createFrameResources` and produces an approximate byte breakdown by
 * category, texture format, and buffer-usage class.  Surfaced via
 * `HybridEngine.debug.estimatedGpuMemoryBytes()` and (when present) on
 * `FrameStats.gpuMemoryBytes`.
 *
 * Estimation, not exact: WebGPU does not report driver-allocated size.
 * Texture bytes are computed from every exposed mip level, array layer, and
 * depth slice. We do not account for implementation-driven row-alignment
 * padding or BC/ASTC block size. If a backend later adopts block-compressed formats
 * this helper would *under-report*, so it intentionally throws on
 * unrecognised formats rather than silently returning zero — that fails
 * loud on stale lookup tables.
 *
 * The breakdown matches `@vitrum/core`'s `GpuMemoryBreakdown` shape:
 *
 *   {
 *     total: 247_000_000,
 *     byCategory: {
 *       common: 50_000_000,
 *       restirDI: 30_000_000,
 *       restirGI: 30_000_000,
 *       svgf: 60_000_000,
 *       gtao: 8_000_000,
 *       ddgi: 25_000_000,
 *       ppg: 0,
 *       neural: 0,         // InferenceGraph owns these; supplied separately when available
 *     },
 *     byTextureFormat: { rgba16float: ..., r32uint: ..., ... },
 *     byBufferUsage:   { storage: ..., uniform: ..., vertex: ... },
 *   }
 */

import type { GpuMemoryBreakdown } from '@vitrum/core';
import type { FrameResources } from './resourceManager.js';

export type GpuMemoryResourceSection = Readonly<Record<string, unknown>>;
export type GpuMemoryExternalSections = Readonly<Record<string, GpuMemoryResourceSection>>;

/**
 * Bytes per texel for every WebGPU `GPUTextureFormat` the walkaround
 * pipeline currently uses. Extending this table is the supported path
 * for adopting a new format — adding the format string without updating
 * this map throws at `estimateFrameResourcesMemory` time, which is the
 * intended failure mode.
 *
 * Block-compressed formats are intentionally absent: the walkaround
 * path does not allocate them, and supporting them properly requires a
 * (width, height) → block-count calculation rather than a flat per-texel
 * multiplier.
 */
const TEXEL_BYTES: Readonly<Record<string, number>> = Object.freeze({
  // 8-bit channels
  'r8unorm':    1,
  'r8snorm':    1,
  'r8uint':     1,
  'r8sint':     1,
  'rg8unorm':   2,
  'rg8snorm':   2,
  'rg8uint':    2,
  'rg8sint':    2,
  'rgba8unorm': 4,
  'rgba8unorm-srgb': 4,
  'rgba8snorm': 4,
  'rgba8uint':  4,
  'rgba8sint':  4,
  'bgra8unorm': 4,
  'bgra8unorm-srgb': 4,

  // 16-bit channels
  'r16uint':    2,
  'r16sint':    2,
  'r16float':   2,
  'rg16uint':   4,
  'rg16sint':   4,
  'rg16float':  4,
  'rgba16uint': 8,
  'rgba16sint': 8,
  'rgba16float': 8,

  // 32-bit channels
  'r32uint':    4,
  'r32sint':    4,
  'r32float':   4,
  'rg32uint':   8,
  'rg32sint':   8,
  'rg32float':  8,
  'rgba32uint': 16,
  'rgba32sint': 16,
  'rgba32float': 16,

  // Packed
  'rgb10a2unorm': 4,
  'rgb10a2uint':  4,
  'rg11b10ufloat': 4,
  'rgb9e5ufloat':  4,

  // Depth/stencil — counted as worst-case per-texel size; walkaround
  // does not bind these directly but the estimator should still produce
  // a defensible upper bound when an extension package allocates one.
  'stencil8':       1,
  'depth16unorm':   2,
  'depth24plus':    4,
  'depth24plus-stencil8': 4,
  'depth32float':   4,
  'depth32float-stencil8': 5,
});

/**
 * Return bytes-per-texel for a known WebGPU format. Throws on unknown
 * formats — the failure is preferable to silent under-reporting.
 *
 * Exported for unit-test cross-checks; callers in the engine path should
 * use {@link estimateFrameResourcesMemory} instead of touching this
 * directly.
 */
export function bytesPerTexel(format: GPUTextureFormat): number {
  const bytes = TEXEL_BYTES[format];
  if (bytes === undefined) {
    throw new Error(
      `[gpuMemoryEstimate] unknown texture format '${format}'. ` +
      `Add it to TEXEL_BYTES in pipeline/gpuMemoryEstimate.ts.`,
    );
  }
  return bytes;
}

/**
 * Classify a buffer's dominant usage. WebGPU lets one buffer carry
 * multiple usage flags (STORAGE | COPY_DST | COPY_SRC, etc.); the
 * estimator attributes bytes to the highest-priority class so the
 * resulting budget reflects path-tracer allocator pressure rather than
 * incidental copy bits.
 *
 * Priority (high → low): STORAGE > UNIFORM > VERTEX > INDEX > other.
 */
export function classifyBufferUsage(usage: number): 'storage' | 'uniform' | 'vertex' | 'index' | 'other' {
  // Guard against environments where `GPUBufferUsage` isn't a runtime
  // global (e.g. Vitest workers without the WebGPU shim). The numeric
  // constants are part of the WebGPU spec; if missing we fall through
  // to 'other' rather than blowing up.
  const G = (globalThis as unknown as { GPUBufferUsage?: Record<string, number | undefined> }).GPUBufferUsage;
  if (!G) return 'other';
  const STORAGE = G.STORAGE ?? 0;
  const UNIFORM = G.UNIFORM ?? 0;
  const VERTEX  = G.VERTEX  ?? 0;
  const INDEX   = G.INDEX   ?? 0;
  if (STORAGE !== 0 && (usage & STORAGE) !== 0) return 'storage';
  if (UNIFORM !== 0 && (usage & UNIFORM) !== 0) return 'uniform';
  if (VERTEX  !== 0 && (usage & VERTEX)  !== 0) return 'vertex';
  if (INDEX   !== 0 && (usage & INDEX)   !== 0) return 'index';
  return 'other';
}

/** A texture with the size + format inputs we need. The GPUTexture spec
 *  exposes these as plain numeric / string properties at runtime; we
 *  type the helper this way so the test harness can substitute a plain
 *  object without a real WebGPU implementation. */
interface MeasurableTexture {
  readonly width: number;
  readonly height: number;
  readonly format: GPUTextureFormat;
  readonly depthOrArrayLayers?: number;
  readonly mipLevelCount?: number;
  readonly dimension?: GPUTextureDimension;
}

/** A buffer with the fields we need. Same rationale as MeasurableTexture. */
interface MeasurableBuffer {
  readonly size: number;
  readonly usage: number;
}

/** Compute one texture's byte footprint, including every explicit mip level. */
function textureBytes(t: MeasurableTexture): number {
  const mipLevelCount = Math.max(1, Math.floor(t.mipLevelCount ?? 1));
  const depthOrLayers = Math.max(1, Math.floor(t.depthOrArrayLayers ?? 1));
  const bytesPerPixel = bytesPerTexel(t.format);
  let bytes = 0;
  for (let level = 0; level < mipLevelCount; level += 1) {
    const width = Math.max(1, Math.floor(t.width / (2 ** level)));
    const height = Math.max(1, Math.floor(t.height / (2 ** level)));
    // Array/cube layers persist at every level. Only a genuine 3D texture
    // shrinks its depth as the mip level increases.
    const layers = t.dimension === '3d'
      ? Math.max(1, Math.floor(depthOrLayers / (2 ** level)))
      : depthOrLayers;
    bytes += width * height * layers * bytesPerPixel;
  }
  return bytes;
}

/**
 * Iterate the per-algorithm sub-structs of {@link FrameResources} and
 * accumulate texture + buffer bytes into a {@link GpuMemoryBreakdown}.
 *
 * `byCategory` keys match the sub-struct names verbatim (the contract
 * surface — see `EngineDebugSurface.estimatedGpuMemoryBytes`); algorithms
 * with no resources allocated report 0. Graph-owned neural memory is an
 * external category rather than a fictitious FrameResources section.
 *
 * The implementation is deliberately structural: we walk each sub-struct's
 * own enumerable properties and dispatch on the GPU-object brand
 * (`typeof .format === 'string'` → texture; `typeof .size === 'number'`
 * → buffer; samplers and anything else skipped). Adding a field to a
 * sub-struct in `resourceManager.ts` does not require touching this code,
 * which is the point — the estimator should not silently rot when a
 * future sprint adds a texture.
 */
export function estimateFrameResourcesMemory(
  res: FrameResources,
  externalSections: GpuMemoryExternalSections = {},
): GpuMemoryBreakdown {
  // Mutable accumulators; frozen on return.
  const byCategory: Record<string, number> = {};
  const byTextureFormat: Record<string, number> = {};
  const byBufferUsage: Record<string, number> = {};
  let total = 0;

  const addSection = (cat: string, section: GpuMemoryResourceSection): void => {
    let catBytes = 0;
    for (const fieldName of Object.keys(section)) {
      const obj = section[fieldName] as Record<string, unknown> | null | undefined;
      if (obj == null) continue;

      if (typeof obj.format === 'string' &&
          typeof obj.width === 'number' &&
          typeof obj.height === 'number') {
        const tex = obj as unknown as MeasurableTexture;
        const bytes = textureBytes(tex);
        catBytes += bytes;
        byTextureFormat[tex.format] = (byTextureFormat[tex.format] ?? 0) + bytes;
        continue;
      }

      if (typeof obj.size === 'number' &&
          typeof obj.usage === 'number') {
        const buf = obj as unknown as MeasurableBuffer;
        const bytes = buf.size;
        catBytes += bytes;
        const usageClass = classifyBufferUsage(buf.usage);
        byBufferUsage[usageClass] = (byBufferUsage[usageClass] ?? 0) + bytes;
      }
    }
    byCategory[cat] = (byCategory[cat] ?? 0) + catBytes;
    total += catBytes;
  };

  // ── Walk each FrameResources sub-struct ────────────────────────────────
  // Cast to a record-of-records so we can iterate the public sub-struct
  // names declared in resourceManager.ts.
  // NOTE on the `?? {}` + `byCategory[cat] = catBytes` guarantee below:
  // `addSection` uses `(byCategory[cat] ?? 0) + catBytes`, and each KNOWN
  // category is visited exactly once here, so the first (and only) write per
  // known category is `0 + catBytes` — identical to the prior direct
  // assignment. Empty/absent sub-structs still emit a `byCategory[cat] = 0`
  // key (the contract surface expects all known categories present), because
  // `addSection` always writes the key even for a zero-byte section.
  //
  // Samplers / frozen empty placeholders contribute 0: `addSection` skips any
  // field lacking the texture (`format`+`width`+`height`) or buffer
  // (`size`+`usage`) brand. WebGPU drivers allocate a per-sampler descriptor
  // block (~64 bytes) but the spec doesn't expose it and the rounding error
  // against the 100+ MB texture budget is entirely in the noise.
  const sections = res as unknown as Record<string, Record<string, unknown>>;
  const KNOWN_CATEGORIES = ['common', 'restirDI', 'restirGI', 'ddgi', 'gtao', 'svgf', 'ppg', 'neural'] as const;

  for (const cat of KNOWN_CATEGORIES) {
    addSection(cat, sections[cat] ?? {});
  }

  for (const [cat, section] of Object.entries(externalSections)) {
    addSection(cat, section);
  }

  return Object.freeze({
    total,
    byCategory:      Object.freeze(byCategory),
    byTextureFormat: Object.freeze(byTextureFormat),
    byBufferUsage:   Object.freeze(byBufferUsage),
  });
}
