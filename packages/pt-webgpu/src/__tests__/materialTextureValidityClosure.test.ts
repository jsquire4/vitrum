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
    // The four bilinear taps and the two mip levels are fetched in loops now —
    // one inlined call site each instead of four/two (see the INLINE-BUDGET notes
    // in material.wgsl.ts). The closure being pinned here is unchanged: validity
    // is ANDed across EVERY tap and EVERY mip before a sample is accepted, and
    // the loop bounds keep "every" honest.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('tapIndex < 4u');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('tapsValid = tapsValid && tap.valid;');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('if (!tapsValid) {');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('mipIndex < 2u');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'mipsValid = mipsValid && mipSample.valid;',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'if (!mipsValid) { return materialTextureInvalidSample(); }',
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
    // The three bump height taps (centre, +u, +v) are fetched in a loop now — one
    // inlined call site instead of three. Every tap's validity is still required
    // before the gradient is used, and the loop bound keeps "all three" honest.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('tapIndex < 3u');
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'heightsValid = heightsValid && tap.valid;',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'if (!heightsValid) { return shadingNormal; }',
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
    // The entry and exit interface normals are derived in a two-iteration loop —
    // one inlined call site for applyNormalMap/applyBumpMap instead of two each
    // (see the INLINE-BUDGET note in bsdf.wgsl.ts). The independence being pinned
    // here is unchanged: the exit face starts from the negated base normal and is
    // mapped with the OPPOSITE face flag, and exitNormal is still taken from the
    // second iteration rather than reusing the entry result.
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'var mapped = select(interfaceBaseNormal, -interfaceBaseNormal, isExit);',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain(
      'select(frontFace, !frontFace, isExit),',
    );
    expect(PT_WEBGPU_TRACE_WGSL).toContain('var exitNormal = interfaceNormals[1];');
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
