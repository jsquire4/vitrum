import { describe, expect, it } from 'vitest';
import { RIS_WGSL } from '../ris.wgsl.js';
import {
  RIS_GI_GLASS_RESERVOIR_LOOP_WGSL,
  RIS_GI_GLASS_VISIBILITY_TAIL_WGSL,
} from '../risGiGlassWalk.wgsl.js';
import { SHADING_TERMS_WGSL } from '../shadingTerms.wgsl.js';
import { NRC_INDEPENDENT_SUFFIX_WGSL } from '../nrcIndependentSuffix.wgsl.js';
import { RESERVOIR_DI_WGSL } from '../reservoirDi.wgsl.js';
import { SAMPLE_CASCADE_C0_WGSL } from '../sampleCascadeC0.wgsl.js';
import { TEMPORAL_GI_WGSL, TEMPORAL_GI_GRIS_WGSL } from '../temporalGi.wgsl.js';
import { SPATIAL_GI_WGSL, SPATIAL_GI_GRIS_WGSL } from '../spatialGi.wgsl.js';

describe('exact represented radiometric support', () => {
  it('retains tiny positive DI source PDFs and rejects only non-finite weights', () => {
    expect(RIS_WGSL).toContain('let pX = emitterSelPmf * ls.pdfArea;');
    expect(RIS_WGSL).toContain('if (pHat > 0.0 && pX > 0.0) { w = pHat / pX; }');
    expect(RIS_WGSL).not.toContain('max(1e-15, emitterSelPmf * ls.pdfArea)');
    expect(RESERVOIR_DI_WGSL).toContain('if (!reservoirDiFinite(w) || !(w > 0.0)) { return; }');

    const tinyPdf = 1e-20;
    const tinyTarget = 3e-20;
    const representedWeight = tinyTarget / tinyPdf;
    expect(Number.isFinite(representedWeight)).toBe(true);
    expect(representedWeight).toBeCloseTo(3, 15);
  });

  it('does not erase tiny positive GI targets, visibility, or mixture PDFs', () => {
    const glass = RIS_GI_GLASS_RESERVOIR_LOOP_WGSL + RIS_GI_GLASS_VISIBILITY_TAIL_WGSL;
    expect(glass).toContain('!reservoirGiFinite(pHat_g) || !(pHat_g > 0.0)');
    expect(glass).toContain('candidateVisibility_g <= 0.0');
    expect(glass).toContain('if (pSrc_g > 0.0) { w_g = pHat_g / pSrc_g; }');
    expect(glass).toContain('if (!reservoirGiFinite(w_g) || !(w_g > 0.0))');
    expect(glass).not.toMatch(/pHat_g\s*[<=>]+\s*1e-/);
    expect(glass).not.toMatch(/candidateVisibility_g\s*[<=>]+\s*0\.001/);

    for (const src of [TEMPORAL_GI_WGSL, TEMPORAL_GI_GRIS_WGSL, SPATIAL_GI_WGSL, SPATIAL_GI_GRIS_WGSL]) {
      expect(src).not.toMatch(/pHat[^\n]*1e-9/);
      expect(src).not.toMatch(/denominator[^\n]*1e-12/);
    }
  });

  it('preserves tiny positive cosine, shadow, RC, and NRC support', () => {
    expect(SHADING_TERMS_WGSL).toContain('let rcHasEnergy = max(Lo_rc.r, max(Lo_rc.g, Lo_rc.b)) > 0.0;');
    expect(SHADING_TERMS_WGSL).not.toMatch(/(?:nDotL|nDotSun|sunShadowT|rcHasEnergy)[^\n]*(?:1e-|0\.001)/);
    expect(SAMPLE_CASCADE_C0_WGSL).toContain('if (cosTheta <= 0.0) { continue; }');
    expect(SAMPLE_CASCADE_C0_WGSL).toContain('if (Wsum > 0.0)');
    expect(NRC_INDEPENDENT_SUFFIX_WGSL).toContain('if (!(proposalPdf > 0.0)) { break; }');
    expect(NRC_INDEPENDENT_SUFFIX_WGSL).not.toMatch(/(?:selectionPmf|pdfArea|proposalPdf|cosLight)[^\n]*1e-/);
  });
});
