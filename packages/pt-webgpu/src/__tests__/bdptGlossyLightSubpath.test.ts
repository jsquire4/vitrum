/**
 * A9 — BDPT production-quality structure tests: the REAL glossy/specular light
 * subpath, the 5-row light-path vertex carrying the light-vertex BSDF plus
 * hit-local material payload for the §10.3 connection, the raised bounce cap,
 * and the isotropic point emitter.
 *
 * These pin the WGSL structure + the host-side caps + the CPU oracle parity. The
 * GPU radiometric A/Bs (equal-spp variance vs the megakernel on a glass Cornell;
 * the BDPT caustic scene) are V28 queue entries — see road-to-100 A9.
 */
import { describe, expect, it } from 'vitest';

import { PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL } from '../wgsl/bdpt/bdptLightSubpath.wgsl.js';
import { PT_WEBGPU_BDPT_CONNECTION_WGSL } from '../wgsl/bdpt/bdptConnection.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL } from '../wgsl/pathTrace/material.wgsl.js';
import { BdptLightPathBufferWebGPU } from '../bdpt/bdptLightPathBufferWebGPU.js';

describe('A9 — glossy/specular BDPT light subpath', () => {
  it('samples the REAL BSDF (glossy partition + cosine diffuse), not Lambertian-only', () => {
    // BDPT light-subpath estimator coherence (2026-06-10): the scatter direction
    // is sampled at the PREVIOUS vertex (prevPos) using its stored outgoing direction
    // (woAtPrev), and that SAME direction is used to extend the path (trace) AND
    // to compute the stored throughput / pdfFwd. The old two-step
    // (cosine-hemisphere trace + discard + real-BSDF sample at newPos) is gone.
    // The BSDF is sampled at prevPos through the shared main-path sampler so the
    // scalar clearcoat/sheen source-lobe mixture and sampled PDF stay coherent.
    //
    // PTWG-BDPT-01 (2026-06-15): finite area emitters need the cos/pdfΩ = π
    // factor after the first traced hit; legacy pseudo emitters keep the old
    // INV_PI branch because their bounce-0 normalization already includes the
    // direction-density term.
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let bsPrev = sampleNextBounceDirectionWithClearcoatNormal(');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('prevMat.clearcoat,');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('prevMat.sheenRoughness,');
    // f and throughput computed at prevPos (prevMat/prevNormal/woAtPrev/scatterDir).
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('fPrev = evaluateBrdfFullWithClearcoatNormal(');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('prevMat.specularColor, prevMat.specularIntensity,');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let newThroughput = prevThroughput * fPrev * cosPrev / pdfFwd;');
    // pdfFwd = scatter pdf at prevPos (SA, no baked-in geometry term).
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('pdfScatter = brdfDirectionalPdfFullSampledWithClearcoatNormal(prevBc, prevRough, prevMetal, 0.0, prevMat.ior,');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let pdfFwd = pdfScatter;');
    // pdfRev(prevCol) is patched to the TRUE reverse density (Item-3 fix 2026-06-10):
    // for surface vertices, brdfDirectionalPdf(prevNormal, scatterDir, woAtPrev) —
    // NOT pdfFwd, which was the forward pdf and only equal for symmetric BSDFs.
    // For emitter vertices (prevMatId < 0), Lambertian cosine hemisphere IS symmetric
    // so pdfFwd == pdfRev; the emitter branch correctly falls back to pdfFwd.
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('bdptLightPath[bdptLightPathIndex(prevCol, 2u)] = vec4f(old_r2prev.xyz, pdfRevAtPrev);');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('pdfRevAtPrev = brdfDirectionalPdfFullSampledWithClearcoatNormal(prevBcRev, prevRoughRev, prevMetalRev, 0.0,');
  });

  it('finite-area emitter extension keeps the needed π factor, legacy pseudo emitters do not double-apply it', () => {
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'fPrev = select(vec3f(INV_PI), vec3f(1.0), prevMatId == BDPT_LV_AREA_EMITTER_MATID);',
    );
    // Surface vertices still use the real extension-aware BRDF.
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('fPrev = evaluateBrdfFullWithClearcoatNormal(');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('prevMat.specularColor, prevMat.specularIntensity,');
  });

  it('directional bounce-0 uses packed RGB records and scene-scaled pseudo distance', () => {
    const code = PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL
      .split('\n')
      .filter((line) => !line.trim().startsWith('//'))
      .join('\n');

    expect(code).toContain('var n = params.directionalLightCount;');
    expect(code).toContain('for (var di = 0u; di < params.directionalLightCount; di = di + 1u)');
    expect(code).toContain('let dDirAD = directionalLights[dBase];');
    expect(code).toContain('let dIrrMean = directionalLights[dBase + 1u];');
    expect(code).toContain('bdptDistantEmitterPosition(lightDir)');
    expect(code).toContain('bdptFinishBounce0(col, emitPos, lightDir, dIrrMean.rgb, discretePdf, rng);');
    expect(code).not.toContain('params.lightDir.w');
    expect(code).not.toContain('* 50.0');
  });

  it('records the light-vertex matId + wo-toward-prev so the connection can evaluate the real BSDF', () => {
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('bdptWriteLvBsdf(col, f32(matIdx), woLp);');
    // The emitter vertex is marked with the sentinel matId < 0 (Lambertian/emission).
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('const BDPT_LV_EMITTER_MATID: f32 = -1.0;');
  });

  it('stores hit-local material payload for texture-mapped light-path vertices', () => {
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'bdptLightPath[bdptLightPathIndex(col, 4u)] = vec4f(bitcast<f32>(triIndex), baryVW.x, baryVW.y, bitcast<f32>(instanceIndex));',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'bdptWriteLvMaterialPayload(col, hit.triIndex, hit.baryVW, hit.instanceIndex);',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'let prevPayload = bdptLightPath[bdptLightPathIndex(prevCol, 4u)];',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'let prevMat = bdptSampleMaterialAtPayload(u32(prevMatId), prevPayload, prevNormal);',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let triIndex = bitcast<u32>(payload.x);');
  });

  it('samples texture-map material payloads before BDPT light-subpath scatter', () => {
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'out.baseColor = mat.baseColor * sampleVertexColor(triIndex, baryVW).rgb * sampleBaseColorTexture(matId, triIndex, baryVW).rgb;',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let orm = sampleOrmTexture(matId, triIndex, baryVW);');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'out.clearcoat = clamp(mat.clearcoat * sampleClearcoatTexture(matId, triIndex, baryVW), 0.0, 1.0);',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'out.clearcoatNormal = applyClearcoatNormalMap(matId, triIndex, baryVW, shadingNormal, instanceIndex);',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('prevNormal,');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('prevMat.clearcoatNormal,');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'out.specularColor = clamp(mat.specularColor * sampleSpecularColorTexture(matId, triIndex, baryVW), vec3f(0.0), vec3f(1.0));',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'nsFront = applyNormalMap(matIdx, hit.triIndex, hit.baryVW, nsFront, hit.instanceIndex);',
    );
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain(
      'prevMat.anisotropy, prevMat.anisotropyRotation',
    );
  });

  it('the §10.3 connection evaluates the REAL light-vertex BSDF + pdfs for a surface vertex', () => {
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('eyeClearcoatNormal: vec3f,');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('specularColor: vec3f,');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('specularIntensity: f32,');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('let eyeBrdf = evaluateBrdfFullWithClearcoatNormal(');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'baseColor, roughness, metallic, eyeNormal, eyeClearcoatNormal, eyeWo, connDir,',
    );
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('specularColor, specularIntensity,');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('let revLc = brdfDirectionalPdfFullSampledWithClearcoatNormal(');
    // lightBsdfCosTheta uses the real BSDF when matId >= 0 (was always cosθ/π),
    // without re-multiplying cosLight (the geometry term owns edge cosines).
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('let lv4 = bdptLightPath[bdptLightPathIndex(lightVtxIdx, 4u)];');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('if (lvMatId >= 0.0) {');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('let lvMat = bdptSampleMaterialAtPayload(u32(lvMatId), lv4, lightNormal);');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('let lvBrdf = evaluateBrdfFullWithClearcoatNormal(');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('lightNormal, lvMat.clearcoatNormal, -connDir, lvWoPrev,');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('lightBsdfCosTheta = lvBrdf;');
    // The MIS pdf bookkeeping (fwdEe + revLcMinus) also uses the real BSDF pdf.
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('fwdEe = brdfDirectionalPdfFullSampledWithClearcoatNormal(');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('let revLc = brdfDirectionalPdfFullSampledWithClearcoatNormal(');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('lightNormal, lvMatF.clearcoatNormal, lvWoPrev, lcToE,');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain(
      'lvMatF.clearcoat, lvMatF.clearcoatRoughness, lvMatF.sheen, lvMatF.sheenRoughness,',
    );
    // Legacy pseudo emitters keep the Lambertian emission profile; finite area
    // emitters use a distinct sentinel and contribute no extra endpoint factor.
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('if (lvMatId == -1.0) {');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('lightBsdfCosTheta = vec3f(cosLight / PI);');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('const BDPT_LV_AREA_EMITTER_MATID: f32 = -2.0;');
  });

  it('the light-path vertex is 5 rows (row 4 = tri/bary/instance material payload)', () => {
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL).toContain('const BDPT_LIGHT_PATH_ROWS = 5u;');
  });
});

describe('A9 — raised bounce cap + isotropic point emitter', () => {
  it('the light-path buffer accepts maxLightBounces up to 8 (was capped at 3)', () => {
    const sized: { size: number }[] = [];
    const fakeDevice = {
      createBuffer(desc: { size: number }) {
        sized.push({ size: desc.size });
        return { destroy() {} } as unknown as GPUBuffer;
      },
    } as unknown as GPUDevice;
    const buf = new BdptLightPathBufferWebGPU(fakeDevice, { maxLightBounces: 8 });
    expect(buf.maxLightBounces).toBe(8);
    // 8 columns × 5 rows × 16 B.
    expect(sized[0]!.size).toBe(8 * 5 * 16);
    expect(() => new BdptLightPathBufferWebGPU(fakeDevice, { maxLightBounces: 9 })).toThrow(
      /maxLightBounces must be 1..8/,
    );
  });

  it('the point emitter is ISOTROPIC (uniform sphere, 1/4π), not cosine-up', () => {
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('fn bdptFinishBounce0Isotropic(');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let pdfDir = 0.25 * INV_PI;');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('bdptFinishBounce0Isotropic(col, pos, rad, discretePdf, rng);');
  });
});
