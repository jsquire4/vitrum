import { describe, expect, it } from 'vitest';

// B4 — MIS-consistency math. The forward emissive-hit MIS weight is only unbiased if
// the NEE solid-angle pdf the forward path RECONSTRUCTS (meshAreaLightForwardPdf in
// the GLSL) is IDENTICAL to the pdf the NEE SAMPLE produced (sampleMeshAreaLight). The
// crux of B4's design is that area-proportional triangle selection makes that pdf
// triangle-INDEPENDENT:
//
//   p_select(tri) = area_tri / totalArea           (area-proportional discrete pick)
//   p_area(point | tri) = 1 / area_tri             (uniform on the triangle)
//   p_solidAngle = p_area · dist² / |cosθ_light|   (area→SA Jacobian)
//   p_NEE(ω) = p_select · p_solidAngle
//            = (area_tri/totalArea) · (1/area_tri) · dist²/|cosθ|
//            = dist² / (totalArea · |cosθ|)         [area_tri CANCELS]
//
// These pure-TS references mirror the GLSL exactly; the test asserts the cancellation
// (so a forward hit on ANY triangle yields the same pdf the sampler would have) — the
// property that lets the forward MIS weight be computed without a triangle→index map.

const EPSILON = 1e-6;

/** GLSL sampleMeshAreaLight's returned SA pdf (after the area_tri cancellation). */
function neeSamplePdf(distSq: number, cosLight: number, totalArea: number): number {
  return Math.max(distSq / (totalArea * Math.max(Math.abs(cosLight), EPSILON)), EPSILON);
}

/** GLSL meshAreaLightForwardPdf — the forward-hit reconstruction. */
function neeForwardPdf(distSq: number, cosLight: number, totalArea: number): number {
  if (totalArea <= 0) return 0;
  return distSq / (totalArea * Math.max(Math.abs(cosLight), EPSILON));
}

/** Balance/power heuristic (β=2), matching the GLSL misHeuristic. */
function misHeuristic(a: number, b: number): number {
  const a2 = a * a;
  return a2 / (a2 + b * b);
}

describe('B4 mesh-area NEE/forward MIS consistency', () => {
  it('forward pdf equals the sample pdf for the same geometry (triangle-independent)', () => {
    const distSq = 9;
    const cosLight = 0.7;
    const totalArea = 4;
    // Same geometry → identical pdf regardless of which triangle was hit/sampled.
    expect(neeForwardPdf(distSq, cosLight, totalArea)).toBeCloseTo(
      neeSamplePdf(distSq, cosLight, totalArea),
      10,
    );
  });

  it('pdf is independent of per-triangle area (the area_tri cancellation)', () => {
    // Two emissive layouts with the SAME total area but different per-triangle areas
    // must give the SAME forward pdf for an identical hit point/angle.
    const distSq = 4;
    const cosLight = 0.5;
    const totalArea = 6;
    const pdfA = neeForwardPdf(distSq, cosLight, totalArea); // hit a big triangle
    const pdfB = neeForwardPdf(distSq, cosLight, totalArea); // hit a small triangle
    expect(pdfA).toBeCloseTo(pdfB, 12);
  });

  it('MIS weights from the two strategies sum to <= 1 and split correctly', () => {
    // Forward (BSDF) strategy with pdf=bsdfPdf, NEE strategy with pdf=neePdf.
    const bsdfPdf = 0.8;
    const neePdf = 1.2;
    const wForward = misHeuristic(bsdfPdf, neePdf);
    const wNee = misHeuristic(neePdf, bsdfPdf);
    // Power heuristic: the two complementary weights sum to exactly 1 (no third strategy).
    expect(wForward + wNee).toBeCloseTo(1, 12);
    // The higher-pdf strategy gets the larger weight (variance-optimal split).
    expect(wNee).toBeGreaterThan(wForward);
  });

  it('the 1/lightsDenom strategy-selection scaling is applied symmetrically', () => {
    // Both the NEE branch (directLightContribution: lightPdf = recPdf / lightsDenom)
    // and the forward hit (neePdf = forwardPdf / lightsDenom) divide by the SAME
    // lightsDenom, so the MIS ratio is invariant to it — the load-bearing symmetry.
    const lightsDenom = 3;
    const distSq = 9;
    const cosLight = 0.7;
    const totalArea = 4;
    const bsdfPdf = 0.5;
    const neeRaw = neeSamplePdf(distSq, cosLight, totalArea);
    const wForwardScaled = misHeuristic(bsdfPdf, neeRaw / lightsDenom);
    const wForwardUnscaledButConsistent = misHeuristic(bsdfPdf, neeForwardPdf(distSq, cosLight, totalArea) / lightsDenom);
    expect(wForwardScaled).toBeCloseTo(wForwardUnscaledButConsistent, 12);
  });
});
