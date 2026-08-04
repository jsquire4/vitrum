import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { PT_WEBGPU_TRACE_LITE_WGSL } from '../wgsl/pathTraceBruteforceLite.wgsl.js';
import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { PT_WEBGPU_ADJOINT_PASS_WGSL } from '../wgsl/pathTrace/adjointPass.wgsl.js';
import { RESERVOIR_PT_HERO_WGSL } from '../wgsl/pathTrace/reservoirPtHero.wgsl.js';
import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';

function functionSignature(source: string, name: string): string {
  const marker = `fn ${name}(`;
  const start = source.indexOf(marker);
  if (start < 0) throw new Error(`missing ${name}`);
  let depth = 0;
  for (let index = start + marker.length - 1; index < source.length; index += 1) {
    const char = source[index];
    if (char === '(') depth += 1;
    if (char === ')') {
      depth -= 1;
      if (depth === 0) {
        const returnEnd = source.indexOf('{', index);
        return source.slice(start, returnEnd).replace(/\s+/g, ' ').trim();
      }
    }
  }
  throw new Error(`unterminated ${name}`);
}

describe('pt-webgpu material texture validity closure', () => {
  it('propagates validity through fetch, every bilinear tap, and mip filtering', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('struct MaterialTextureSample');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'if (!(c00.valid && c10.valid && c01.valid && c11.valid))',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'if (!(c0.valid && c1.valid)) { return materialTextureInvalidSample(); }',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('textureNumLayers(materialTexturesLinear)');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('textureNumLevels(materialTexturesLinear)');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('!materialTextureFiniteVec4(value)');
  });

  it('uses checked descriptor conversions and overflow-safe UV-plane arithmetic', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn materialTextureDescriptorBase');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn materialTextureLayerIndex');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn materialTextureExactU32');
    expect(PT_WEBGPU_TRACE_WGSL).not.toMatch(/(?:u32|i32)\(materialTexDescriptors/);
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'tailPlane > (uvCount - 1u - vertexIndex) / vertexCount',
    );
    expect(PT_WEBGPU_TRACE_WGSL).not.toContain(
      '(gpuUvSlot - 2u) * vertexCount + vertexIndex',
    );
  });

  it('pins exact invalid-role defaults and scale-safe perturbation', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'if (!sample.valid) { return vec3f(0.0); }',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'if (!sample.valid) { return -1.0; }',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn materialTextureApplyTangentNormal');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let coefficientScale = max(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'if (!(sampleC.valid && sampleU.valid && sampleV.valid))',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'let scaledGradient = vec2f(dhdu, dhdv) * bumpScale;',
    );
  });

  it('keeps lite/full specialty estimator interfaces in lockstep', () => {
    expect(functionSignature(PT_WEBGPU_TRACE_LITE_WGSL, 'manifoldNeeContribution'))
      .toBe(functionSignature(PT_WEBGPU_TRACE_WGSL, 'manifoldNeeContribution'));
    expect(functionSignature(PT_WEBGPU_TRACE_LITE_WGSL, 'photonMapUpdateProgressive'))
      .toBe(functionSignature(PT_WEBGPU_TRACE_WGSL, 'photonMapUpdateProgressive'));
    expect(PT_WEBGPU_TRACE_LITE_WGSL).not.toContain('fn photonMapContribution(');
  });

  it('routes MNEE alpha coverage through the same fail-closed descriptor decode', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn mneeFacetCoverage(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'let base = materialTextureDescriptorBase(matId);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'if (alphaMode == 0xffffffffu) { return 0.0; }',
    );
  });

  it('checks material-record, thin-film LUT, adjoint, and intersection arithmetic before access', () => {
    for (const helper of [
      'materialCheckedAddU32',
      'materialCheckedMulU32',
      'materialSpanValid',
      'materialRecordExactU32',
      'materialRecordBase',
      'materialRecordIndex',
    ]) {
      expect(PT_WEBGPU_TRACE_WGSL).toContain(`fn ${helper}(`);
    }
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'let lutBaseScalar = materialCheckedMulU32(lutBaseVec4, 4u);',
    );
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(
      'if (gradientOffset > 0xfffffffdu) { continue; }',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn materialShadowCastDisabled(');
  });

  it('fails MNEE closed for mapped interfaces and unsupported outer layers', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn mneeFacetHasMappedInterface(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('clearcoatNormalIdx >= 0');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('frontLayerNormalIdx >= 0');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('backLayerNormalIdx >= 0');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn mneeFacetHasUnsupportedOuterLayer(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'return mat.clearcoat > 0.0 || mat.sheen > 0.0;',
    );
  });

  it('derives both thin-sheet faces independently and applies exact entry outer-layer attenuation', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'oppositeNormal = applyNormalMap(matId, hit.triIndex, hit.baryVW, oppositeNormal, hit.instanceIndex, !isFrontFace);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'exitNormal = applyNormalMap(\n    matId, hit.triIndex, hit.baryVW, exitNormal, hit.instanceIndex, !frontFace,',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('let outerLayerAttenuation =');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'max(exitLayerWeight, vec3f(0.0)) * outerLayerAttenuation;',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn bdptSampleMaterialAtPayload(');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('fn applyClearcoatNormalMap(');
    expect(PT_WEBGPU_TRACE_LITE_WGSL).toContain('fn sampleSheenColorTexture(');
  });

  it('starts every clearcoat-normal map from the independently oriented smooth interface', () => {
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'var clearcoatNormal = interfaceBaseNormal;',
    );
    expect(SPPM_PHOTON_PASS_WGSL).toContain(
      'matId, hit.triIndex, hit.baryVW, interfaceBaseNormal, hit.instanceIndex,',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'triIndex, baryVW, instanceIndex, isFrontFace, shadingNormal,',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'matId, triIndex, baryVW, interfaceBaseNormal, instanceIndex,',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'out.normal = interfaceBaseNormal;',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'vMatId, vHit.triIndex, vHit.baryVW, interfaceBaseNormalV, vHit.instanceIndex,',
    );
    expect(RESERVOIR_PT_HERO_WGSL).toContain(
      'triIndex, baryVW, instanceIndex, isFrontFace, shadingNormal,',
    );
    expect(RESERVOIR_PT_HERO_WGSL).toContain(
      'matId, triIndex, baryVW, interfaceBaseNormal, instanceIndex,',
    );
  });
});
