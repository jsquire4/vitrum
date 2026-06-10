/**
 * A9 — BDPT production-quality structure tests: the REAL glossy/specular light
 * subpath, the 4-row light-path vertex carrying the light-vertex BSDF for the
 * §10.3 connection, the raised bounce cap, and the isotropic point emitter.
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
    // The extension must use glossyReflectionSample on the spec lobe + cosine on
    // the diffuse lobe, and the throughput is f·cos/pdf with the REAL BSDF.
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('glossyReflectionSample(&rng, woLp, nsFront, tanT, tanB, rough)');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let fLp = evaluateBrdf(bc, rough, metal, nsFront, woLp, nextDir);');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let newThroughput = prevThroughput * fLp * cosNext / pdfFwd;');
    // pdfFwd/pdfRev are the REAL brdfDirectionalPdf (SA), not the cosine cosθ/π.
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let pdfFwd = brdfDirectionalPdf(bc, rough, metal, 0.0, mat.ior, nsFront, woLp, nextDir);');
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('let pdfRev = brdfDirectionalPdf(bc, rough, metal, 0.0, mat.ior, nsFront, nextDir, toPrev);');
  });

  it('records the light-vertex matId + wo-toward-prev so the connection can evaluate the real BSDF', () => {
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('bdptWriteLvBsdf(col, f32(matIdx), woLp);');
    // The emitter vertex is marked with the sentinel matId < 0 (Lambertian/emission).
    expect(PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL).toContain('const BDPT_LV_EMITTER_MATID: f32 = -1.0;');
  });

  it('the §10.3 connection evaluates the REAL light-vertex BSDF + pdfs for a surface vertex', () => {
    // lightBsdfCosTheta uses the real BSDF when matId >= 0 (was always cosθ/π).
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('if (lvMatId >= 0.0) {');
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('lightBsdfCosTheta = lvBrdf * cosLight;');
    // The MIS pdf bookkeeping (fwdEe + revLcMinus) also uses the real BSDF pdf.
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('lightNormal, lvWoPrev, lcToE)');
    // The emitter (matId < 0) keeps the Lambertian emission profile.
    expect(PT_WEBGPU_BDPT_CONNECTION_WGSL).toContain('var lightBsdfCosTheta = vec3f(cosLight / PI);');
  });

  it('the light-path vertex is 4 rows (row 3 = matId + wo-toward-prev)', () => {
    expect(PT_WEBGPU_PATH_TRACE_MATERIAL_WGSL).toContain('const BDPT_LIGHT_PATH_ROWS = 4u;');
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
    // 8 columns × 4 rows × 16 B.
    expect(sized[0]!.size).toBe(8 * 4 * 16);
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
