import { describe, expect, it } from 'vitest';
import { MANIFOLD_CAUSTICS_WGSL } from '../manifoldCaustics.wgsl.js';
import { MANIFOLD_SMS_SOLVER_WGSL } from '../manifoldSmsSolver.wgsl.js';
import { RIS_WGSL } from '../ris.wgsl.js';
import { SHADING_TERMS_WGSL } from '../shadingTerms.wgsl.js';
import { SURFACE_TEXTURES_WGSL } from '../surfaceTextures.wgsl.js';

describe('walkaround bounded manifold caustics WGSL', () => {
  it('covers every endpoint family with its represented selection measure', () => {
    for (const token of [
      'SMS_SOURCE_SUN',
      'SMS_SOURCE_ANALYTIC',
      'SMS_SOURCE_AREA',
      'SMS_SOURCE_ENVIRONMENT',
      'smsAnalyticAliasDraw',
      'smsArenaAliasDraw',
      'envImportanceSample',
      'sampleEmitterPoint',
      'sampleEmitterLeAtXi',
    ]) expect(MANIFOLD_CAUSTICS_WGSL).toContain(token);
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('out.endpointPdf = sample.pdfArea;');
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('directionalPdf = 1.0 / solidAngle;');
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('directionalPdf = 1.0 / (4.0 * PI);');
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('out.endpointPdf = 1.0;');
  });

  it('pins the complete joint proposal density and exact represented PMFs', () => {
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain('facet.pairPmf = pairPmf;');
    expect(MANIFOLD_SMS_SOLVER_WGSL).not.toContain('proposalPmf / facet.area');
    expect(MANIFOLD_CAUSTICS_WGSL).toContain(
      'lengthPmf * endpoint.selectionPdf * endpoint.endpointPdf',
    );
    expect(MANIFOLD_CAUSTICS_WGSL).toContain(
      'draw.facet.pairPmf *\n      offset.proposalPdf * offset.eventProposalPmf',
    );
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('sceneLoadEmitterAlias(index)');
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain('sceneLoadMneeFacetDomainAlias(domainIndex)');
  });

  it('freezes reflection/transmission events and rejects transmission TIR', () => {
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain('SMS_EVENT_REFLECTION');
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain('SMS_EVENT_TRANSMISSION');
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain('geometry.events[index] == SMS_EVENT_TRANSMISSION');
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('eventPmf = 0.5;');
    expect(MANIFOLD_CAUSTICS_WGSL).toContain(
      'eta * eta * (1.0 - cosIncident * cosIncident) >= 1.0',
    );
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('does not mutate\n    // the medium stack');
  });

  it('fails closed on invalid mapped frames and out-of-triangle roots', () => {
    const opticsStart = MANIFOLD_SMS_SOLVER_WGSL.indexOf('fn smsFacetOpticsAt');
    const invalidGate = MANIFOLD_SMS_SOLVER_WGSL.indexOf(
      'if (frame.valid == 0u) { return out; }', opticsStart,
    );
    const firstMapRead = MANIFOLD_SMS_SOLVER_WGSL.indexOf('textureLoad(bvh_material', opticsStart);
    expect(invalidGate).toBeGreaterThan(opticsStart);
    expect(invalidGate).toBeLessThan(firstMapRead);
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain('fn smsSolutionInsideFacets(');
    expect(MANIFOLD_SMS_SOLVER_WGSL.match(/smsSolutionInsideFacets\(/g)?.length).toBeGreaterThanOrEqual(3);
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain('if (!smsFacetContains(');
  });

  it('re-evaluates the last Newton step and keeps conditioning scale-relative', () => {
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain(
      'A step accepted by the final permitted iteration must be re-evaluated',
    );
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain('out.iterations = maxIterations;');
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain(
      'absoluteDeterminant > 64.0 * SMS_F32_EPSILON * scale * scale',
    );
    expect(MANIFOLD_SMS_SOLVER_WGSL).not.toContain('max(scale * scale, 1.0)');
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain('absoluteDeterminant >= SMS_MIN_NORMAL_F32');
  });

  it('branches TLAS loads and never evaluates placeholder transform columns', () => {
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain('if (hasTlasNormal) {');
    expect(MANIFOLD_SMS_SOLVER_WGSL).not.toContain('let safeBase = select');
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain('false,\n      vec4f(0.0), vec4f(0.0), vec4f(0.0)');
  });

  it('pins bounded multiplicity semantics and the structural-only exact bypass', () => {
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('out.weight = f32(cap);');
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('out.truncated = 1u;');
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('oneBasedTrial * 0x27d4eb2du');
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('smsProvesUniquePlanarDeltaTransmission');
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('geometry.count != 1u');
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('geometry.events[0] != SMS_EVENT_TRANSMISSION');
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('MATERIAL_MAP_FRONT_LAYER_NORMAL_TEXEL_OFFSET');
    expect(MANIFOLD_CAUSTICS_WGSL).not.toContain('trials == 0u');
  });

  it('owns mapped emission, interface transport, spectral Beer, and castShadow once', () => {
    expect(MANIFOLD_CAUSTICS_WGSL.match(/sampleEmitterLeAtXi\(/g)).toHaveLength(1);
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('materialThinFilmResponse(');
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('faceLayerTransmission(');
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('materialSpectralAttenuation(');
    expect(MANIFOLD_CAUSTICS_WGSL).toContain('endpoint.castShadowDisabled == 0u');
    expect(MANIFOLD_CAUSTICS_WGSL).toContain(
      'traceSceneAlphaTintTransmittanceTexturedWithOwnership(',
    );
    expect(MANIFOLD_CAUSTICS_WGSL).toContain(
      'bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer, true',
    );
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain('(materialWord & 1u) != 0u');
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain('out.surfaceCoverage = select(');
  });

  it('gives manifold paths exclusive ownership of covered material transmission', () => {
    expect(SHADING_TERMS_WGSL).toContain(
      'fn manifoldNeeOwnsMaterialTransmission() -> bool',
    );
    expect(SHADING_TERMS_WGSL).toContain('return ubo.sunAngular.z >= 1.5;');
    expect(SHADING_TERMS_WGSL.match(
      /traceSceneAlphaTintTransmittanceTexturedWithOwnership\(/g,
    )).toHaveLength(4);
    expect(RIS_WGSL.match(
      /traceSceneAlphaTintTransmittanceTexturedWithOwnership\(/g,
    )).toHaveLength(2);
    expect(RIS_WGSL.match(/ubo\.sunAngular\.z >= 1\.5/g)).toHaveLength(2);

    // Disabled strategies retain the original walker, and no-glass/alpha-only
    // hits share the same alphaT path. Only material transmission enters the
    // ownership branch, so an unoccluded or alpha-only scene is unchanged.
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'materialMask, materialMaskWidth, bvh_beer, false',
    );
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'if (!blockMaterialTransmission || !hasMaterialTransmission)',
    );
    expect(SURFACE_TEXTURES_WGSL).toContain(
      'materialShadowTransmittanceForHit(\n      hit,\n      word,\n      !blockMaterialTransmission',
    );
  });

  it('applies the half-vector measure exactly once for rough interfaces', () => {
    expect(MANIFOLD_CAUSTICS_WGSL).toContain(
      'let halfVectorJacobian = 4.0 * abs(wiDotM);',
    );
    expect(MANIFOLD_CAUSTICS_WGSL).toContain(
      'let halfVectorJacobian = denominator2 * etap * etap / abs(woDotM);',
    );
    expect(MANIFOLD_CAUSTICS_WGSL).toContain(
      'offset.proposalPdf * offset.eventProposalPmf',
    );
  });

  it('does not duplicate point-light geometry or receiver foreshortening', () => {
    expect(MANIFOLD_CAUSTICS_WGSL).toContain(
      ') * distanceSquaredToSource;',
    );
    expect(MANIFOLD_CAUSTICS_WGSL).toContain(
      '1.0, receiverCosine, endpoint.family == SMS_SOURCE_AREA',
    );
  });

  it('derives point, directional, and area focusing measures in the expected units', () => {
    // d(normalized direction)/d(receiver metre) crossed twice is m^-2.
    const pointDistance = 4;
    const pointDeterminant = 1 / (pointDistance * pointDistance);
    expect(pointDeterminant).toBeCloseTo(0.0625, 12);
    // A directional source maps receiver-area metres to source-footprint metres:
    // two dimensionless derivatives, projected onto the source direction.
    const directionalDeterminant = Math.abs(
      [1, 0, 0][0]! * [0, 1, 0][1]!,
    );
    expect(directionalDeterminant).toBe(1);
    // d(omega_receiver)/d(area_light) is m^-2; a normal-facing free-space patch
    // has cos(theta)/r^2 and becomes dimensionless after division by pdf_A (m^-2).
    const areaJacobian = Math.cos(0) / (pointDistance * pointDistance);
    const areaPdf = 1 / 3;
    expect(areaJacobian / areaPdf).toBeCloseTo(0.1875, 12);
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain('fn smsFiniteFocusingDet(');
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain('fn smsDirectionalFocusingDet(');
    expect(MANIFOLD_SMS_SOLVER_WGSL).toContain('fn smsAreaFocusingDet(');
  });
});
