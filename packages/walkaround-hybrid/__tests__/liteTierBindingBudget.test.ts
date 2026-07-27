/**
 * liteTierBindingBudget.test.ts — §H H56-b
 *
 * Lite executes less work but compiles the same explicit bind-group layouts as
 * full. WebGPU validates all declared layout entries, including dummy-bound or
 * runtime-inactive TLAS/RC slots, so both tiers require the same structural
 * device floor. The pass-layout recording test covers the global peaks; this
 * legacy static-WGSL test pins the shade-specific declaration breakdown.
 */

import { describe, it, expect } from 'vitest';
import {
  HYBRID_LITE_LIMITS,
  HYBRID_WEBGPU_REQUIRED_LIMITS,
} from '../src/pipeline/WalkaroundGPUPipeline.js';
import { WGSL_MODULES } from '../src/pipeline/wgslModules.js';
import { composeWgsl } from '../src/pipeline/wgslComposer.js';
import { SHADE_MODULE, SHADE_WGSL } from '../src/shaders/shade.wgsl.js';

// ── Static WGSL binding counter ───────────────────────────────────────────────

/**
 * Count `var<storage, …>` declarations in a WGSL shader fragment.
 * These count against maxStorageBuffersPerShaderStage on strict backends.
 */
function countStorageBuffers(wgsl: string): number {
  return (wgsl.match(/var<storage,\s*(?:read|read_write|write)>/g) ?? []).length;
}

/**
 * Count `texture_storage_2d<…, write>` declarations in a WGSL shader.
 * These count against maxStorageTexturesPerShaderStage.
 */
function countStorageTextures(wgsl: string): number {
  return (wgsl.match(/texture_storage_2d<[^>]+,\s*write>/g) ?? []).length;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('lite-tier binding budget — static WGSL analysis (H56-b)', () => {
  const composedShadeWgsl = composeWgsl(SHADE_MODULE, WGSL_MODULES);

  it('HYBRID_LITE_LIMITS has both axes defined', () => {
    expect(HYBRID_LITE_LIMITS['maxStorageBuffersPerShaderStage']).toBeTypeOf('number');
    expect(HYBRID_LITE_LIMITS['maxStorageTexturesPerShaderStage']).toBeTypeOf('number');
  });

  it('uses the same validated layout floor for lite and full', () => {
    expect(HYBRID_LITE_LIMITS).toEqual(HYBRID_WEBGPU_REQUIRED_LIMITS);
  });

  it('shade.wgsl declares six storage textures within the seven-texture global peak', () => {
    // The 6 storage textures (hdrColorOut, gNormalDepthOut, hdrIndirectOut,
    // hdrTotalOut, hdrAlbedoOut, svgfObjectIdOut) are the structural minimum.
    // shade writes all 6 simultaneously; this cannot be reduced without a WGSL
    // fork — the lite decision explicitly avoids forking shade.wgsl.
    const texCount = countStorageTextures(SHADE_WGSL);
    expect(texCount).toBe(6);
    expect(texCount).toBeLessThanOrEqual(HYBRID_LITE_LIMITS['maxStorageTexturesPerShaderStage']!);
  });

  it('composed shade declares eight storage buffers (4 frame + 3 scene arenas + 1 RC)', () => {
    // The root shade module owns only the four frame-group reservoir buffers.
    // All logical scene arrays are loaded through the dependency-owned arenas.
    expect(countStorageBuffers(SHADE_WGSL)).toBe(4);

    // The composed shader sits exactly on WebGPU's guaranteed floor: four frame
    // buffers + geometry/TLAS/lighting arenas + one RC cascade buffer.
    const bufCount = countStorageBuffers(composedShadeWgsl);
    expect(bufCount).toBe(8);
    expect(bufCount).toBeLessThanOrEqual(HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageBuffersPerShaderStage']!);
  });

  it('uses three scene arena slots regardless of merged-BVH or TLAS traversal mode', () => {
    const arenaDecls = composedShadeWgsl.match(
      /@group\(1\)\s*@binding\([0-2]\)\s*var<storage,\s*read>\s*scene(?:Geometry|Tlas|Lighting)Arena/g,
    ) ?? [];
    expect(arenaDecls).toHaveLength(3);
    expect(new Set(arenaDecls).size).toBe(3);
    expect(countStorageBuffers(composedShadeWgsl)).toBe(
      4 /* frame */ + 3 /* scene */ + 1 /* RC */,
    );
  });

  it('the TLAS loader surface is present and reads the versioned TLAS arena', () => {
    const tlasLoaders = [
      'tlasLoadNode',
      'tlasLoadInstanceIndex',
      'tlasLoadBlasRoot',
      'tlasLoadWorldToLocalColumn',
      'tlasLoadLocalToWorldColumn',
    ];
    for (const name of tlasLoaders) {
      expect(composedShadeWgsl, `${name} must be defined in composed shade`).toMatch(
        new RegExp(`fn\\s+${name}\\s*\\(`),
      );
    }
    expect(composedShadeWgsl).toContain('@group(1) @binding(1) var<storage, read> sceneTlasArena: array<u32>;');
    expect(composedShadeWgsl).not.toMatch(/var<storage,\s*read>\s+tlas(?:Nodes|Instance)/);
  });

  it('PPG tree bindings are NOT referenced in shade.wgsl (lite-forbidden, risGi-only)', () => {
    // PPG is forbidden on the lite tier.  shade.wgsl reads hybrid-layers group-3
    // but only bindings 0-5 (DDGI + RC params).  Bindings 6-8 (PPG sTree/dTree/
    // dTreeOffsets) are read-only by risGi.wgsl, not shade.  Verifying this means
    // the shade pipeline never requires PPG resources.
    expect(SHADE_WGSL).not.toMatch(/ppgEnabled\b/);
    // shade.wgsl should not declare the PPG tree buffer variables.
    expect(SHADE_WGSL).not.toMatch(/var<storage[^>]*>\s+ppg/);
    expect(SHADE_WGSL).not.toMatch(/sTreeBuffer|dTreeBuffer|dTreeOffsetsBuffer/);
  });

  it('shade.wgsl storage texture declarations match expected names', () => {
    // Pin the specific storage-texture names so a future rename triggers this test.
    const expectedTextures = [
      'hdrColorOut',
      'gNormalDepthOut',
      'hdrIndirectOut',
      'hdrTotalOut',
      'hdrAlbedoOut',
      'svgfObjectIdOut',
    ];
    for (const name of expectedTextures) {
      expect(SHADE_WGSL, `Storage texture '${name}' must be declared in shade.wgsl`).toContain(name);
    }
    // Exact count remains 6.
    expect(countStorageTextures(SHADE_WGSL)).toBe(expectedTextures.length);
  });
});
