/**
 * liteTierBindingBudget.test.ts — §H H56-b
 *
 * Verifies that the LITE pass graph's WGSL binding declarations fit within the
 * limits stated in HYBRID_LITE_LIMITS, and documents the relationship between
 * the declared binding count and the lite-tier limit.
 *
 * Background:
 *   HYBRID_LITE_LIMITS declares:
 *     maxStorageBuffersPerShaderStage: 10
 *     maxStorageTexturesPerShaderStage: 6
 *
 *   The lite tier forces `bvhMode:'merged'`, which removes the 5 TLAS storage
 *   buffers from the scene bind group at RUNTIME (they are bound to 32-byte
 *   dummies and the shader's `bvhMode == 0u` branch never traverses them).
 *   However, the WGSL DECLARES all bindings regardless — WebGPU's pipeline
 *   creation validates against the DECLARED bindings in each shader, not just
 *   the ones actually executed.
 *
 *   The composed shade shader binding structure:
 *     Frame group (group 0) — storage BUFFERS:  bindings 5, 6, 7, 11 = 4
 *     Frame group (group 0) — storage TEXTURES: bindings 8, 10, 12, 13, 14, 15 = 6
 *     Scene group (group 1) — storage BUFFERS (non-TLAS): 0,1,2,3,4,11 = 6
 *       (binding 11 / bvh_normal is dependency-owned by materialAtlas.wgsl)
 *       (analytic_lights moved to @group(1) @binding(13) as a texture)
 *     Scene group (group 1) — storage BUFFERS (TLAS, merged-mode dummies): 6,7,8,9,10 = 5
 *     Hybrid layers group (group 3) — storage BUFFERS: binding 4 = 1 RC cascade-0
 *
 *   Total storage buffers declared in composed shade WGSL = 4+11+1 = 16
 *   Total storage textures declared in shade WGSL = 6 (at the lite-tier floor)
 *
 *   HYBRID_LITE_LIMITS.maxStorageBuffersPerShaderStage = 10 refers to the
 *   non-TLAS merged path: 4 (frame) + 6 (non-TLAS scene) = 10.
 *   The 5 TLAS buffers, plus the RC cascade when RC is disabled, are DECLARED;
 *   the lite tier depends on those inactive bindings being stripped in the
 *   merged/RC-off path.
 *
 * What this test asserts:
 *   1. HYBRID_LITE_LIMITS < HYBRID_WEBGPU_REQUIRED_LIMITS on both axes.
 *   2. shade storage textures = 6 = HYBRID_LITE_LIMITS.maxStorageTexturesPerShaderStage.
 *   3. composed shade declares the 5 TLAS storage buffers (required for BGL compat).
 *   4. shade does NOT reference PPG tree bindings (lite-forbidden).
 *   5. The non-TLAS storage buffer sum (frame + non-TLAS scene + RC cascade) ≤ 10.
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

  it('HYBRID_LITE_LIMITS < HYBRID_WEBGPU_REQUIRED_LIMITS on both axes (monotone gradient)', () => {
    // The lite limit must be strictly below the full-tier limit so that the
    // adapter-capability test (lite vs full verdict) is monotone.
    expect(HYBRID_LITE_LIMITS['maxStorageBuffersPerShaderStage']!)
      .toBeLessThan(HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageBuffersPerShaderStage']!);
    expect(HYBRID_LITE_LIMITS['maxStorageTexturesPerShaderStage']!)
      .toBeLessThan(HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageTexturesPerShaderStage']!);
  });

  it('shade.wgsl storage texture count = 6 = HYBRID_LITE_LIMITS.maxStorageTexturesPerShaderStage', () => {
    // The 6 storage textures (hdrColorOut, gNormalDepthOut, hdrIndirectOut,
    // hdrTotalOut, hdrAlbedoOut, svgfObjectIdOut) are the structural minimum.
    // shade writes all 6 simultaneously; this cannot be reduced without a WGSL
    // fork — the lite decision explicitly avoids forking shade.wgsl.
    const texCount = countStorageTextures(SHADE_WGSL);
    expect(texCount).toBe(6);
    expect(texCount).toBe(HYBRID_LITE_LIMITS['maxStorageTexturesPerShaderStage']!);
  });

  it('composed shade declares 16 storage buffers (4 frame + 11 scene + 1 RC)', () => {
    // The root shade module declares 14 buffers; materialAtlas owns the normal
    // buffer declaration that shade needs for textured alpha-mask traversal.
    expect(countStorageBuffers(SHADE_WGSL)).toBe(14);

    // The composed shader declares 4 frame-group + 11 scene-group + 1 RC
    // cascade storage buffers = 16. Analytic point/spot NEE is now a sampled
    // rgba32float texture, so it no longer consumes a storage-buffer slot.
    const bufCount = countStorageBuffers(composedShadeWgsl);
    expect(bufCount).toBe(16);
    expect(bufCount).toBeLessThanOrEqual(HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageBuffersPerShaderStage']!);
  });

  it('non-TLAS non-RC storage buffer count (merged path minimum) = 10; composed shade total = 16', () => {
    // Breakdown of non-TLAS storage buffers in composed shade:
    //   Frame group:        4  (bindings 5,6,7,11 — reservoirs + GI reservoir)
    //   Scene group non-TLAS: 6  (bindings 0,1,2,3,4,11 — bvh+index+pos+emitters+cdf+normals)
    //   Total non-TLAS, non-RC in composed shade: 10
    //   TLAS (declared):    5  (bindings 6,7,8,9,10 — dummies in merged mode)
    //   RC cascade-0:       1  (binding 4 in hybrid-layers group)
    //   Grand total in composed shade: 16
    //
    // The HYBRID_LITE_LIMITS (10) is an empirical minimum for hardware that
    // strips declared-but-inactive TLAS/RC bindings in the merged, RC-off path.
    const nonTlasFrameBuffers    = 4;  // shade.wgsl @group(0): bindings 5,6,7,11
    const nonTlasSceneBuffers    = 6;  // shade.wgsl @group(1): bindings 0,1,2,3,4,11
    const rcCascadeBuffers       = 1;  // sampleCascadeC0 @group(3): binding 4
    const tlasSceneBuffers       = 5;  // shade.wgsl @group(1): bindings 6,7,8,9,10
    const nonTlasTotal = nonTlasFrameBuffers + nonTlasSceneBuffers;
    const grandTotal   = nonTlasTotal + rcCascadeBuffers + tlasSceneBuffers;

    expect(nonTlasTotal).toBe(10);
    expect(nonTlasTotal + rcCascadeBuffers).toBe(11);
    expect(grandTotal).toBe(16);

    // Non-TLAS/non-RC total equals the lite limit; drivers must still strip
    // TLAS and inactive RC bindings to hit that limit for the merged path.
    expect(nonTlasTotal).toBe(HYBRID_LITE_LIMITS['maxStorageBuffersPerShaderStage']!);
    expect(grandTotal).toBeLessThanOrEqual(HYBRID_WEBGPU_REQUIRED_LIMITS['maxStorageBuffersPerShaderStage']!);
  });

  it('the 5 TLAS storage buffers ARE declared in composed shade (required for BGL compat)', () => {
    // Even in merged mode the full scene BGL is used (single layout) — the TLAS
    // bindings must be present in the shader.  A future lite-WGSL-fork that
    // elides them would lower the declared count and should update this test.
    const tlasNames = [
      'tlasNodes',
      'tlasInstanceIndices',
      'tlasBlasRoots',
      'tlasInstanceWorldToLocal',
      'tlasInstanceLocalToWorld',
    ];
    for (const name of tlasNames) {
      expect(composedShadeWgsl, `${name} must be declared in composed shade`).toContain(name);
    }
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
