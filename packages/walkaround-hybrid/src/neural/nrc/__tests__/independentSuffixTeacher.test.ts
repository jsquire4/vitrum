import { describe, expect, it } from 'vitest';
import { NRC_INDEPENDENT_SUFFIX_WGSL } from '../../../shaders/nrcIndependentSuffix.wgsl.ts';
import { RIS_GI_NRC_BODY, buildRisGiNrcModule } from '../../../shaders/risGiNrc.wgsl.ts';
import { composeWgsl } from '../../../pipeline/wgslComposer.ts';
import { WGSL_MODULES } from '../../../pipeline/wgslModules.ts';

const CFG = {
  levels: 8,
  featuresPerEntry: 2,
  oneBlobBins: 8,
  width: 64,
  outWidth: 3,
  hidden: 6,
} as const;

describe('NRC independent suffix teacher', () => {
  it('uses an independently traced target rather than DDGI/cache distillation', () => {
    expect(NRC_INDEPENDENT_SUFFIX_WGSL).toContain('fn nrcTraceIndependentSuffix(');
    expect(NRC_INDEPENDENT_SUFFIX_WGSL).not.toContain('sampleDDGIAtPoint(');
    expect(NRC_INDEPENDENT_SUFFIX_WGSL).not.toContain('nrcQueryRadiance(');
    expect(RIS_GI_NRC_BODY).toContain('nrcTrackTarget = nrcTraceIndependentSuffix(');
    expect(RIS_GI_NRC_BODY).not.toContain('nrcTrackTarget = ddgiLo');
  });

  it('pins bounded depth, Russian roulette, and the matching defensive mixture PDF', () => {
    const src = NRC_INDEPENDENT_SUFFIX_WGSL;
    expect(src).toContain('const NRC_TEACHER_MAX_VERTICES: u32 = 4u;');
    expect(src).toContain('const NRC_TEACHER_RR_START: u32 = 2u;');
    expect(src).toContain('NRC_TEACHER_SPECULAR_MIX * pdfSpec +');
    expect(src).toContain('(1.0 - NRC_TEACHER_SPECULAR_MIX) * pdfCos;');
    expect(src).toContain('nrcTeacherMaterialResponse(payload, normal, wo, nextDir) / proposalPdf');
    expect(src).toContain('nextThroughput = nextThroughput / survive;');
  });

  it('covers authored light families and mapped surface radiance', () => {
    const src = NRC_INDEPENDENT_SUFFIX_WGSL;
    expect(src).toContain('sampleEmitterIdx(count, rand_f32(rng))');
    expect(src).toContain('emitterCdfPmf(count, lid)');
    expect(src).toContain('sampleEmitterLeAtXi(emitter, xi)');
    expect(src).toContain('nrcTeacherAnalyticNee(');
    expect(src).toContain('nrcTeacherSunNee(');
    expect(src).toContain('restir_gi_surface_emission_for_hit(currentHit)');
    // Light maps are receiver-local baked irradiance and must never train the
    // independent transport suffix as if they were scene emission.
    expect(src).not.toContain('sampleLightMap(');
    expect(src).toContain('envRadiance(nextDir)');
  });

  it('evaluates direct lighting at the terminal opaque vertex before truncation', () => {
    const trace = NRC_INDEPENDENT_SUFFIX_WGSL.slice(
      NRC_INDEPENDENT_SUFFIX_WGSL.indexOf('fn nrcTraceIndependentSuffix('),
    );
    const direct = trace.indexOf('nrcTeacherAreaNee(currentPos');
    const terminal = trace.indexOf(
      'if (depth + 1u >= NRC_TEACHER_MAX_VERTICES) { break; }',
      direct,
    );
    expect(direct).toBeGreaterThan(0);
    expect(terminal).toBeGreaterThan(direct);
  });

  it('caps independent suffix traces to one collision-free candidate per record slot', () => {
    expect(RIS_GI_NRC_BODY).toContain('let nrcTeacherStride = max(1u, 1u +');
    expect(RIS_GI_NRC_BODY).toContain('let nrcTeacherSlot = pixelIdxGi / nrcTeacherStride;');
    expect(RIS_GI_NRC_BODY).toContain('if (!nrcFired && nrcTeacherEligible)');
    expect(RIS_GI_NRC_BODY).toContain('nrcWriteRecord(\n      nrcTeacherSlot,');
  });

  it('keeps glass out of a key that has no IOR/transmission coordinate', () => {
    expect(RIS_GI_NRC_BODY).toContain('xsTransmission <= 0.3');
    expect(RIS_GI_NRC_BODY).toContain('var teacherRng = pcgInit(');
  });

  it('composes all teacher bindings and helpers into the production NRC shader', () => {
    const src = composeWgsl(buildRisGiNrcModule(CFG), WGSL_MODULES);
    expect(src).toContain('@group(1) @binding(2) var<storage, read> sceneLightingArena');
    expect(src).toContain('fn sceneLoadEmitter(');
    expect(src).toContain('fn sceneLoadEmitterCdf(');
    expect(src).not.toMatch(/var<storage,\s*read>\s+emitters\b/);
    expect(src).not.toMatch(/var<storage,\s*read>\s+emitterCdf\b/);
    expect(src).toContain('@group(1) @binding(13) var analytic_lights');
    expect(src).toContain('fn sampleEmitterLeAtXi(');
    expect(src).toContain('fn nrc_teacherPointSpotAttenuation(');
  });
});

describe('defensive-mixture estimator oracle', () => {
  it('recovers Lambertian hemispherical reflectance when every sample divides by the mixture PDF', () => {
    // An abstract two-proposal analogue of the shader's cosine + VNDF mixture.
    // Proposal A is cosine-weighted, proposal B uniform-hemisphere; both have
    // full support. The target integral of rho/pi * cos(theta) is exactly rho.
    const rho = 0.73;
    const alpha = 0.5;
    const count = 200_000;
    let state = 0x12345678;
    const random = (): number => {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      return state / 0x1_0000_0000;
    };
    let sum = 0;
    for (let i = 0; i < count; i++) {
      const chooseA = random() < alpha;
      const u = random();
      const cosTheta = chooseA ? Math.sqrt(1 - u) : u;
      const pCos = cosTheta / Math.PI;
      const pUniform = 1 / (2 * Math.PI);
      const pMix = alpha * pCos + (1 - alpha) * pUniform;
      sum += (rho / Math.PI) * cosTheta / pMix;
    }
    expect(sum / count).toBeCloseTo(rho, 2);
  });
});
