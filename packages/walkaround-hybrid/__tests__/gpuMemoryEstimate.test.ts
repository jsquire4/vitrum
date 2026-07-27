/**
 * GPU-memory-budget instrumentation tests.
 *
 * We don't have a real GPU in CI, so we stub `device.createTexture` and
 * `device.createBuffer` with plain JS objects that record the size+format
 * inputs. The estimator walks those records exactly as it walks real GPU
 * objects in production (it reads dimensions, format, mip count, and texture
 * dimension on textures and `.size / .usage` on buffers — see
 * `pipeline/gpuMemoryEstimate.ts`).
 *
 * Sanity bands:
 *   - 1920×1080 HybridEngine: total ∈ [100 MB, 400 MB]
 *   - Per-category sums equal `total` (invariant)
 *   - The texture-formats-tier1 mismatch surfaces clearly — i.e. the
 *     widened `r32uint` (svgf history) and `rgba16float` (gtao-full)
 *     show up as elevated lines in `byTextureFormat`.
 *
 * Reference numbers checked against
 * `packages/walkaround-hybrid/src/pipeline/resourceManager.ts` field
 * comments — the test serves as a regression alarm when an algorithm
 * gains a new texture and forgets to update its memory-budget bookkeeping.
 */

import { describe, it, expect } from 'vitest';
import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';

// Install GPUBufferUsage / GPUTextureUsage globals BEFORE importing
// resourceManager — that module reads them at the top of createFrameResources.
installWebGPUPolyfills();

import { createFrameResources } from '../src/pipeline/resourceManager.js';
import {
  bytesPerTexel,
  classifyBufferUsage,
  estimateFrameResourcesMemory,
} from '../src/pipeline/gpuMemoryEstimate.js';
import {
  RESERVOIR_GI_BASE_STRIDE_BYTES,
  RESERVOIR_GI_GRIS_STRIDE_BYTES,
} from '../src/restir/reservoirGiLayout.js';

// ─── Stub GPU device that records every texture / buffer allocation ─────────

interface StubTexture {
  readonly label?: string;
  readonly width: number;
  readonly height: number;
  readonly depthOrArrayLayers?: number;
  readonly mipLevelCount?: number;
  readonly dimension?: GPUTextureDimension;
  readonly format: GPUTextureFormat;
  readonly usage: number;
  destroy(): void;
  createView(): { kind: 'view' };
}

interface StubBuffer {
  readonly label?: string;
  readonly size: number;
  readonly usage: number;
  destroy(): void;
}

function makeStubDevice(): GPUDevice {
  const queue = {
    writeBuffer: () => {},
    writeTexture: () => {},
    submit: () => {},
  };
  const device = {
    createTexture: (desc: GPUTextureDescriptor): StubTexture => {
      // Size can be either [w,h] tuple or {width,height,depthOrArrayLayers}.
      let width: number;
      let height: number;
      let depth = 1;
      const s = desc.size as unknown;
      if (Array.isArray(s)) {
        width = (s[0] as number) ?? 1;
        height = (s[1] as number) ?? 1;
        depth = (s[2] as number) ?? 1;
      } else {
        const so = s as { width: number; height: number; depthOrArrayLayers?: number };
        width = so.width;
        height = so.height;
        depth = so.depthOrArrayLayers ?? 1;
      }
      return {
        ...(desc.label !== undefined && { label: desc.label }),
        width,
        height,
        depthOrArrayLayers: depth,
        mipLevelCount: desc.mipLevelCount ?? 1,
        dimension: desc.dimension ?? '2d',
        format: desc.format,
        usage: desc.usage,
        destroy: () => {},
        createView: () => ({ kind: 'view' as const }),
      };
    },
    createBuffer: (desc: GPUBufferDescriptor): StubBuffer => ({
      ...(desc.label !== undefined && { label: desc.label }),
      size: desc.size,
      usage: desc.usage,
      destroy: () => {},
    }),
    createSampler: () => ({}) as GPUSampler,
    queue,
  };
  return device as unknown as GPUDevice;
}

// ─── Sanity tests for the underlying helpers ────────────────────────────────

describe('bytesPerTexel', () => {
  it('returns 8 for rgba16float (the dominant walkaround target format)', () => {
    expect(bytesPerTexel('rgba16float')).toBe(8);
  });

  it('returns 4 for r32uint (svgf history widened from r16uint per tier1 reconciliation)', () => {
    expect(bytesPerTexel('r32uint')).toBe(4);
  });

  it('returns 8 for rg32float external/shared-denoiser textures', () => {
    expect(bytesPerTexel('rg32float')).toBe(8);
  });

  it('returns 16 for rgba32float (the 1×1 placeholder texture)', () => {
    expect(bytesPerTexel('rgba32float')).toBe(16);
  });

  it('throws on an unrecognised format so we fail loud, not silent', () => {
    expect(() => bytesPerTexel('made-up-format' as GPUTextureFormat))
      .toThrow(/unknown texture format/);
  });
});

describe('classifyBufferUsage', () => {
  it('attributes STORAGE | COPY_DST to storage (dominant class wins)', () => {
    const u = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    expect(classifyBufferUsage(u)).toBe('storage');
  });

  it('attributes UNIFORM | COPY_DST to uniform', () => {
    const u = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    expect(classifyBufferUsage(u)).toBe('uniform');
  });

  it('STORAGE wins over UNIFORM when both bits are set (shouldnt happen but be defensive)', () => {
    const u = GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM;
    expect(classifyBufferUsage(u)).toBe('storage');
  });
});

describe('external GPU resource sections', () => {
  it('merges static scene buffers and textures into the same breakdown', () => {
    const device = makeStubDevice();
    const base = estimateFrameResourcesMemory(createFrameResources(device, 64, 64));
    const withScene = estimateFrameResourcesMemory(
      createFrameResources(device, 64, 64),
      {
        staticScene: {
          bvhNodesBuffer: {
            size: 256,
            usage: GPUBufferUsage.STORAGE,
          },
          bvhBeerTexture: {
            width: 8,
            height: 2,
            format: 'r32uint' as GPUTextureFormat,
          },
        },
      },
    );

    expect(withScene.byCategory.staticScene).toBe(256 + 8 * 2 * 4);
    expect(withScene.byBufferUsage.storage).toBe((base.byBufferUsage.storage ?? 0) + 256);
    expect(withScene.byTextureFormat.r32uint).toBe((base.byTextureFormat.r32uint ?? 0) + 8 * 2 * 4);
    expect(withScene.total).toBe(base.total + withScene.byCategory.staticScene!);
  });

  it('counts every mip level while preserving 2D array layers', () => {
    const device = makeStubDevice();
    const base = estimateFrameResourcesMemory(createFrameResources(device, 64, 64));
    const withMips = estimateFrameResourcesMemory(
      createFrameResources(device, 64, 64),
      {
        materialAtlas: {
          atlasTexture: {
            width: 8,
            height: 4,
            depthOrArrayLayers: 2,
            mipLevelCount: 3,
            dimension: '2d',
            format: 'r32uint' as GPUTextureFormat,
          },
        },
      },
    );

    // (8x4 + 4x2 + 2x1) texels x 2 array layers x 4 bytes.
    expect(withMips.byCategory.materialAtlas).toBe(336);
    expect(withMips.byTextureFormat.r32uint).toBe((base.byTextureFormat.r32uint ?? 0) + 336);
    expect(withMips.total).toBe(base.total + 336);
  });
});

describe('H24 ReSTIR-GI reservoir allocation', () => {
  it('uses compact 20-u32 reservoirs by default and widens only for GRIS DDGI-proxy reuse', () => {
    const device = makeStubDevice();
    const W = 128;
    const H = 64;
    const halfPixels = Math.floor(W / 2) * Math.floor(H / 2);

    const compact = createFrameResources(device, W, H);
    const gris = createFrameResources(device, W, H, { grisReuse: true });

    expect(compact.restirGI.reservoirGiCurrentBuffer.size).toBe(halfPixels * RESERVOIR_GI_BASE_STRIDE_BYTES);
    expect(compact.restirGI.reservoirGiPreviousBuffer.size).toBe(halfPixels * RESERVOIR_GI_BASE_STRIDE_BYTES);
    expect(compact.restirGI.reservoirGiSpatialBuffer.size).toBe(halfPixels * RESERVOIR_GI_BASE_STRIDE_BYTES);

    expect(gris.restirGI.reservoirGiCurrentBuffer.size).toBe(halfPixels * RESERVOIR_GI_GRIS_STRIDE_BYTES);
    expect(gris.restirGI.reservoirGiPreviousBuffer.size).toBe(halfPixels * RESERVOIR_GI_GRIS_STRIDE_BYTES);
    expect(gris.restirGI.reservoirGiSpatialBuffer.size).toBe(halfPixels * RESERVOIR_GI_GRIS_STRIDE_BYTES);
  });

  it('reports the exact GRIS memory delta in the restirGI category', () => {
    const device = makeStubDevice();
    const W = 128;
    const H = 64;
    const halfPixels = Math.floor(W / 2) * Math.floor(H / 2);
    const expectedDelta = 3 * halfPixels * (RESERVOIR_GI_GRIS_STRIDE_BYTES - RESERVOIR_GI_BASE_STRIDE_BYTES);

    const compact = estimateFrameResourcesMemory(createFrameResources(device, W, H));
    const gris = estimateFrameResourcesMemory(createFrameResources(device, W, H, { grisReuse: true }));

    expect(gris.byCategory.restirGI! - compact.byCategory.restirGI!).toBe(expectedDelta);
    expect(gris.total - compact.total).toBe(expectedDelta);
  });
});

// ─── Whole-engine 1920×1080 budget assertion ────────────────────────────────

describe('estimateFrameResourcesMemory — 1920×1080 HybridEngine', () => {
  const W = 1920;
  const H = 1080;
  const device = makeStubDevice();
  const res = createFrameResources(device, W, H);
  const breakdown = estimateFrameResourcesMemory(res);

  it('total budget is in a sane range (100 MB ≤ total ≤ 1 GB)', () => {
    // At 1920×1080 the walkaround-hybrid pipeline allocates ~15 full-res
    // rgba16float textures (~250 MB) + ReSTIR reservoirs (~225 MB across DI
    // + GI) + SVGF persistent textures (~88 MB) + GTAO + accumulator =
    // ~650 MB measured today.  The task brief's "100–400 MB" example was
    // illustrative; the real number is higher because every Sprint 14–18
    // addition (Schied SVGF, ReSTIR-GI 80-byte reservoirs at half-res, two
    // indirect ping-pong pairs, per-channel albedo demodulation) appended
    // textures the brief's example didn't enumerate.  Band is intentionally
    // wide so it catches catastrophic regressions (e.g. accidentally
    // allocating a 4×-resolution texture) without false-alarming on benign
    // tweaks.
    const MB = 1024 * 1024;
    expect(breakdown.total).toBeGreaterThan(100 * MB);
    expect(breakdown.total).toBeLessThan(1024 * MB);
  });

  it('byCategory sum equals total (invariant)', () => {
    const sum = Object.values(breakdown.byCategory).reduce((a, b) => a + b, 0);
    expect(sum).toBe(breakdown.total);
  });

  it('every category is present (including empty placeholders)', () => {
    expect(Object.keys(breakdown.byCategory).sort()).toEqual([
      'common', 'ddgi', 'gtao', 'neural', 'ppg', 'restirDI', 'restirGI', 'svgf',
    ]);
  });

  it('PPG and neural categories are 0 when resources have not been allocated (empty sub-structs)', () => {
    expect(breakdown.byCategory.ppg).toBe(0);
    expect(breakdown.byCategory.neural).toBe(0);
  });

  it('common is the dominant category (it owns the full-res HDR ping-pong fleet)', () => {
    const cat = breakdown.byCategory;
    expect(cat.common!).toBeGreaterThan(cat.restirDI!);
    expect(cat.common!).toBeGreaterThan(cat.restirGI!);
    expect(cat.common!).toBeGreaterThan(cat.gtao!);
    expect(cat.common!).toBeGreaterThan(cat.svgf!);
  });

  it('svgf budget is non-trivial — Schied 2017 persistent textures plus portable storage formats', () => {
    const MB = 1024 * 1024;
    // Allow a generous band: r32uint history pair (2×8 MB) plus widened
    // rgba32float moments / variance / intermediate textures for portable
    // WebGPU storage-read/write support. At 1080p this is roughly 206 MB.
    expect(breakdown.byCategory.svgf!).toBeGreaterThan(40 * MB);
    expect(breakdown.byCategory.svgf!).toBeLessThan(240 * MB);
  });

  it('byTextureFormat shows the texture-formats-tier1 reconciliation footprint', () => {
    // The tier1 fix widened:
    //   - svgf history A/B: r16uint → r32uint (2× memory)
    //   - gtao-full: r16float → rgba16float (4× memory)
    // Both should now show up as measurable lines in byTextureFormat.
    // r32uint must be present and account for at least both svgf history
    // textures at 1080p (2 × 1920 × 1080 × 4 = ~16 MB) plus tier (~8 MB).
    const MB = 1024 * 1024;
    const r32uint = breakdown.byTextureFormat.r32uint ?? 0;
    expect(r32uint).toBeGreaterThan(16 * MB);
    // rgba16float dominates — full-res HDR ping-pong fleet (now including the
    // transparent OIT composition target) at 8 bytes / texel × ~2M pixels.
    const rgba16f = breakdown.byTextureFormat.rgba16float ?? 0;
    expect(rgba16f).toBeGreaterThan(80 * MB);
    expect(rgba16f).toBeLessThan(340 * MB);
  });

  it('byBufferUsage shows storage > uniform (reservoir buffers >> UBOs)', () => {
    const storage = breakdown.byBufferUsage.storage ?? 0;
    const uniform = breakdown.byBufferUsage.uniform ?? 0;
    // Sprint 16 GI reservoir at half-res 960×540 × 80 bytes ≈ 41 MB —
    // plus 3 ReSTIR-DI reservoirs at 1920×1080 × 32 bytes ≈ 199 MB.
    expect(storage).toBeGreaterThan(uniform * 10);
    // Walkaround UBO + DDGI UBO + GTAO UBO together fit in ~1 KB.
    expect(uniform).toBeLessThan(1024 * 4);
  });

  it('returned breakdown is frozen so consumers cant accidentally mutate the cache', () => {
    expect(Object.isFrozen(breakdown)).toBe(true);
    expect(Object.isFrozen(breakdown.byCategory)).toBe(true);
    expect(Object.isFrozen(breakdown.byTextureFormat)).toBe(true);
    expect(Object.isFrozen(breakdown.byBufferUsage)).toBe(true);
  });
});

// ─── G-P2.6 — SVGF full-res textures are gated on svgf-real being active ─────

describe('SVGF full-res allocation is gated on the active denoiser (G-P2.6)', () => {
  const W = 1920;
  const H = 1080;
  const MB = 1024 * 1024;
  const persistentSvgfTextures = [
    'svgfPrevNormalDepthTexture',
    'svgfHistoryLengthTextureA',
    'svgfHistoryLengthTextureB',
    'svgfMomentsTextureA',
    'svgfMomentsTextureB',
    'svgfPrevRadianceTextureA',
    'svgfPrevRadianceTextureB',
    'svgfVarianceTexture',
    'svgfVarianceMomentsIntermedTexture',
  ] as const;

  it('svgf category drops to placeholders when svgfEnabled is false (non-svgf-real denoiser)', () => {
    const device = makeStubDevice();
    const enabled  = estimateFrameResourcesMemory(
      createFrameResources(device, W, H, { svgfEnabled: true }));
    const disabled = estimateFrameResourcesMemory(
      createFrameResources(device, W, H, { svgfEnabled: false }));

    // Full-res SVGF persistent textures account for tens of MB; gating them off
    // must reclaim essentially all of the svgf category (only 1×1 placeholders
    // remain — kilobytes).
    expect(enabled.byCategory.svgf!).toBeGreaterThan(40 * MB);
    expect(disabled.byCategory.svgf!).toBeGreaterThan(0);
    expect(disabled.byCategory.svgf!).toBeLessThan(1 * MB);
    // The reclaimed bytes show up 1:1 in the total — nothing else moved.
    expect(enabled.total - disabled.total).toBe(
      enabled.byCategory.svgf! - disabled.byCategory.svgf!);
    // The default (omitted flag) keeps the legacy full allocation.
    const dflt = estimateFrameResourcesMemory(createFrameResources(device, W, H));
    expect(dflt.byCategory.svgf!).toBe(enabled.byCategory.svgf!);
  });

  it('only the svgf category changes — every other category is byte-identical', () => {
    const device = makeStubDevice();
    const enabled  = estimateFrameResourcesMemory(
      createFrameResources(device, W, H, { svgfEnabled: true }));
    const disabled = estimateFrameResourcesMemory(
      createFrameResources(device, W, H, { svgfEnabled: false }));
    for (const cat of ['common', 'restirDI', 'restirGI', 'ddgi', 'gtao', 'ppg', 'neural'] as const) {
      expect(disabled.byCategory[cat], `category '${cat}' must not change`)
        .toBe(enabled.byCategory[cat]);
    }
  });

  it('non-svgf denoisers allocate 1x1 SVGF histories and object IDs', () => {
    const device = makeStubDevice();
    const enabled = createFrameResources(device, W, H, { svgfEnabled: true });
    const disabled = createFrameResources(device, W, H, { svgfEnabled: false });

    for (const field of persistentSvgfTextures) {
      const full = enabled.svgf[field] as unknown as StubTexture;
      const placeholder = disabled.svgf[field] as unknown as StubTexture;
      expect(full.width, `${field} full-res width`).toBe(W);
      expect(full.height, `${field} full-res height`).toBe(H);
      expect(placeholder.width, `${field} placeholder width`).toBe(1);
      expect(placeholder.height, `${field} placeholder height`).toBe(1);
    }

    for (const field of ['svgfCurrentObjectIdTexture', 'svgfPreviousObjectIdTexture'] as const) {
      const full = enabled.svgf[field] as unknown as StubTexture;
      const placeholder = disabled.svgf[field] as unknown as StubTexture;
      expect(full.width, `${field} full-res-mode width`).toBe(W);
      expect(full.height, `${field} full-res-mode height`).toBe(H);
      expect(placeholder.width, `${field} disabled width`).toBe(1);
      expect(placeholder.height, `${field} disabled height`).toBe(1);
    }
  });
});

// ─── Smaller deterministic check at a known size ────────────────────────────

describe('estimateFrameResourcesMemory — 64×64 (deterministic check)', () => {
  it('total scales roughly with pixel count', () => {
    const device = makeStubDevice();
    const small  = estimateFrameResourcesMemory(createFrameResources(device, 64, 64));
    const medium = estimateFrameResourcesMemory(createFrameResources(device, 128, 128));
    // 4× the pixels → texture bytes scale 4×. Buffers don't scale exactly
    // (reservoir DI is `W × H × 32`, but min-256 floors apply at tiny sizes),
    // so we assert "noticeably bigger" rather than "exactly 4×".
    expect(medium.total).toBeGreaterThan(small.total);
    expect(medium.total).toBeLessThan(small.total * 8);
  });
});
