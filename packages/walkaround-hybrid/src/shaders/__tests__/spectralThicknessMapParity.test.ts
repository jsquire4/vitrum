import { describe, expect, it } from 'vitest';
import { MANIFOLD_CAUSTICS_WGSL } from '../manifoldCaustics.wgsl.js';
import { REFRACTIVE_CAUSTICS_WGSL } from '../refractiveCaustics.wgsl.js';
import { SHADING_TERMS_WGSL } from '../shadingTerms.wgsl.js';

function mappedSegmentTransfer(
  baseTint: number,
  baseThickness: number,
  thicknessMapScale: number,
  scatteringCoefficient: number,
  segmentDistance: number,
): number {
  const transportDistance = Math.min(
    segmentDistance,
    baseThickness * thicknessMapScale,
  );
  const mappedAbsorption = baseTint ** (transportDistance / baseThickness);
  return mappedAbsorption * Math.exp(
    -scatteringCoefficient * transportDistance,
  );
}

describe('hybrid spectral thickness-map transport parity', () => {
  it('caps absorption and scattering distance by the mapped authored thickness', () => {
    const baseTint = 0.36;
    const baseThickness = 2;
    const scatteringCoefficient = 0.15;
    const segmentDistance = 3;

    expect(mappedSegmentTransfer(
      baseTint,
      baseThickness,
      0.25,
      scatteringCoefficient,
      segmentDistance,
    )).toBeCloseTo(
      baseTint ** 0.25 *
        Math.exp(-scatteringCoefficient * baseThickness * 0.25),
      14,
    );
    expect(mappedSegmentTransfer(
      baseTint,
      baseThickness,
      0,
      scatteringCoefficient,
      segmentDistance,
    )).toBeCloseTo(1, 14);
    expect(mappedSegmentTransfer(baseTint, baseThickness, 1, 0, baseThickness))
      .toBeCloseTo(baseTint, 14);
  });

  it('scales camera-visible spectral glass emission without changing topology', () => {
    expect(SHADING_TERMS_WGSL).toContain(
      'materialOpticalThickness(triIndex) * thicknessMapScale',
    );
    expect(SHADING_TERMS_WGSL).toContain(
      'MATERIAL_MAP_THICKNESS_TEXEL_OFFSET',
    );
    expect(SHADING_TERMS_WGSL).not.toContain(
      'materialOpticalThickness(triIndex) *= thicknessMapScale',
    );
  });

  it('uses the actual exit payload and canonical cap in refractive caustics', () => {
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'var mediumThicknessMapScale: array<f32, 4>;',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'var mediumScattering: array<vec4f, 4>;',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'segmentThicknessMapScale = materialShadowThicknessMapScale(hit);',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'homogeneousBeerTransmittanceRgb(',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'referenceThickness * clamp(segmentThicknessMapScale, 0.0, 1.0)',
    );
    expect(REFRACTIVE_CAUSTICS_WGSL).toContain(
      'segmentScattering = sampleVolumeScatteringControls(hit.indices.w);',
    );
  });

  it('routes manifold exits and live endpoint media through the canonical cap', () => {
    expect(MANIFOLD_CAUSTICS_WGSL).toContain(
      'materialShadowBeerForSegment(',
    );
    expect(MANIFOLD_CAUSTICS_WGSL).toContain(
      'receiverContaining.state.thicknessMapScale[live]',
    );
    expect(MANIFOLD_CAUSTICS_WGSL).toContain(
      'smsThicknessMapScale(optics),',
    );
    expect(MANIFOLD_CAUSTICS_WGSL).not.toContain(
      'applyThicknessMapToBeerTint(',
    );
  });
});
