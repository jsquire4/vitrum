/**
 * W9 Phase 2 — per-pixel position wiring for PPG guide (sweep finding #5).
 *
 * Pre-W9-Phase-2 the PPG guide kernel mapped every pixel to a single sTree
 * cell because it used `sceneCentre` (scene-AABB midpoint) for the cell
 * lookup. Phase 2 wires a binding to the half-res ReSTIR-GI reservoir buffer
 * and reads its `xv` field — the per-pixel primary-hit world position.
 *
 * This file is the structural pin: any future edit that drops binding(4),
 * changes the reservoir-GI stride, or moves `xv` away from u32 offsets 0..2
 * fails here loudly instead of silently re-introducing the single-cell bug.
 */

import { describe, it, expect } from 'vitest';
import { PPG_GUIDE_WGSL } from '../src/ppg/ppgGuide.wgsl.js';

describe('PPG guide — per-pixel position binding (W9 Phase 2 / sweep #5)', () => {
  it('declares group(0) binding(4) for the ReservoirGI buffer', () => {
    expect(PPG_GUIDE_WGSL).toMatch(
      /@group\(0\)\s*@binding\(4\)\s+var<storage,\s*read>\s+ppgReservoirGiBuf/,
    );
  });

  it('reads ReservoirGI xv at u32 offsets 0..2 (matches common.wgsl layout)', () => {
    // xv lives at u32 offsets 0,1,2 in the 80-byte reservoir stride.
    expect(PPG_GUIDE_WGSL).toContain('ppgReservoirGiBuf[b + 0u]');
    expect(PPG_GUIDE_WGSL).toContain('ppgReservoirGiBuf[b + 1u]');
    expect(PPG_GUIDE_WGSL).toContain('ppgReservoirGiBuf[b + 2u]');
  });

  it('reads ReservoirGI M (sample count) at u32 offset 15 for degenerate-fallback gate', () => {
    expect(PPG_GUIDE_WGSL).toContain('ppgReservoirGiBuf[b + 15u]');
  });

  it('uses local RESERVOIR_GI_STRIDE = 20u (must match common.wgsl)', () => {
    expect(PPG_GUIDE_WGSL).toContain('RESERVOIR_GI_STRIDE_LOCAL : u32 = 20u');
  });

  it('maps full-res pixel to half-res reservoir index (W/2-stride, x>>1, y>>1)', () => {
    expect(PPG_GUIDE_WGSL).toMatch(/halfWidth\s*=\s*max\(1u,\s*ppgGuideUBO\.imgWidth\s*>>\s*1u\)/);
    expect(PPG_GUIDE_WGSL).toMatch(/halfX\s*=\s*fullResX\s*>>\s*1u/);
    expect(PPG_GUIDE_WGSL).toMatch(/halfY\s*=\s*fullResY\s*>>\s*1u/);
  });

  it('keeps scene-centre as fallback when reservoir is degenerate (M==0)', () => {
    // Both branches of the fetchPrimaryHitPos helper exist:
    //   - degenerate M=0 → scene-centre midpoint
    //   - valid M>0     → bitcast<f32> from u32 buffer
    expect(PPG_GUIDE_WGSL).toContain('if (M == 0u)');
    expect(PPG_GUIDE_WGSL).toMatch(/0\.5\s*\*\s*\(ppgGuideUBO\.sceneMinX\s*\+\s*ppgGuideUBO\.sceneMaxX\)/);
  });

  it('the placeholder "sceneCentre" sTree lookup is gone from ppgGuideMain', () => {
    // The pre-Phase-2 code computed `let sceneCentre = vec3<f32>(...)` and
    // passed it to `sTreeFindLeafBase`. Phase 2 routes through
    // `fetchPrimaryHitPos` instead. Asserting the literal absence prevents a
    // regression that re-introduces the single-cell bug by reverting just
    // the call site while leaving the new binding in place.
    expect(PPG_GUIDE_WGSL).not.toMatch(/let sceneCentre = vec3<f32>\(\s*0\.5/);
    expect(PPG_GUIDE_WGSL).toMatch(/let hitPos = fetchPrimaryHitPos\(/);
    expect(PPG_GUIDE_WGSL).toMatch(/let sBase = sTreeFindLeafBase\(hitPos\)/);
  });
});
