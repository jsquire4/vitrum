// adjointHarness.test.ts — the V24 GPU-adjoint harness kernel composes the
// path-replay BRDF partials + their primitives into a dispatchable shader. The
// EXECUTED GPU == CPU-oracle A/B runs on real hardware (wsl-gpu
// scripts/adjoint-validate.ts, lavapipe: max relative err ~2e-7 = f32 precision);
// here we pin the host-side packing + that the kernel bundles the real partials.
import { describe, it, expect } from 'vitest';
import {
  ADJOINT_HARNESS_WGSL,
  packAdjointHarnessInput,
  ADJOINT_HARNESS_INPUT_FLOATS,
  ADJOINT_SHADING_FD_WGSL,
  packShadingAdjointInput,
  ADJOINT_SHADING_INPUT_FLOATS,
} from '../inverse/adjointHarness.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL } from '../wgsl/pathTrace/pathTraceAdjoint.wgsl.js';
import {
  PT_WEBGPU_ADJOINT_PASS_WGSL,
  ADJOINT_PARAMS_UBO_BYTES,
  ADJOINT_FIELD_BASECOLOR,
  ADJOINT_FIELD_ROUGHNESS,
  ADJOINT_FIELD_EMISSIVE,
  ADJOINT_FIELD_SPECULAR_COLOR,
	  ADJOINT_FIELD_SPECULAR_INTENSITY,
	  ADJOINT_FIELD_METALLIC,
	  ADJOINT_FIELD_EMISSIVE_INTENSITY,
	  ADJOINT_FIELD_CLEARCOAT,
	  ADJOINT_FIELD_CLEARCOAT_ROUGHNESS,
	} from '../wgsl/pathTrace/adjointPass.wgsl.js';

describe('adjoint harness (V24 GPU partials A/B)', () => {
  it('packs an input into the 16-float vec4-aligned AdjIn record', () => {
    const r = packAdjointHarnessInput([0.8, 0.2, 0.1], 0.5, 1.0, [0, 0, 1], [0.2, 0.1, 1], [-0.3, 0.2, 1]);
    expect(r).toHaveLength(ADJOINT_HARNESS_INPUT_FLOATS);
    expect(r.slice(0, 4)).toEqual([0.8, 0.2, 0.1, 0.5]);   // baseColor.xyz, roughness
    expect(r.slice(4, 8)).toEqual([0, 0, 1, 1.0]);          // normal.xyz, metallic
    expect(r.slice(8, 12)).toEqual([0.2, 0.1, 1, 0]);       // wo.xyz, pad
    expect(r.slice(12, 16)).toEqual([-0.3, 0.2, 1, 0]);     // wi.xyz, pad
  });

  it('bundles the REAL path-replay adjoint partials + a dispatch entry', () => {
    // The partials under test must be the byte-identical production WGSL.
    expect(ADJOINT_HARNESS_WGSL).toContain(PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL);
    expect(ADJOINT_HARNESS_WGSL).toContain('fn dBrdf_dBaseColor(');
    expect(ADJOINT_HARNESS_WGSL).toContain('fn dBrdf_dRoughness(');
    expect(ADJOINT_HARNESS_WGSL).toContain('@compute @workgroup_size(64)');
    expect(ADJOINT_HARNESS_WGSL).toContain('hOutBC[i] = vec4f(dBrdf_dBaseColor(');
    expect(ADJOINT_HARNESS_WGSL).toContain('hOutR[i]  = vec4f(dBrdf_dRoughness(');
    // gradAccum is declared so the bundled adjointScatter compiles.
    expect(ADJOINT_HARNESS_WGSL).toContain('gradAccum: array<atomic<i32>>');
  });

  it('packs a shading-adjoint input into the 24-float ShIn record', () => {
    const r = packShadingAdjointInput([0.8, 0.2, 0.1], 0.5, 0.0, [0, 0, 1], [0.2, 0.1, 1], [-0.3, 0.2, 1], [3, 3, 3], [0.1, 0.1, 0.1]);
    expect(r).toHaveLength(ADJOINT_SHADING_INPUT_FLOATS);
    expect(r.slice(0, 4)).toEqual([0.8, 0.2, 0.1, 0.5]);  // baseColor.xyz, roughness
    expect(r.slice(16, 20)).toEqual([3, 3, 3, 0]);          // Li.xyz, pad
    expect(r.slice(20, 24)).toEqual([0.1, 0.1, 0.1, 0]);    // tgt.xyz, pad
  });

  it('shading-adjoint kernel bundles the forward + real partials + adjoint-vs-FD', () => {
    // The chain-rule/accumulation A/B runs on hardware (adjoint-fd-validate.ts,
    // lavapipe: analytic == central-FD, max rel 7.9e-4). Here we pin the structure.
    expect(ADJOINT_SHADING_FD_WGSL).toContain(PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL); // byte-identical partials
    expect(ADJOINT_SHADING_FD_WGSL).toContain('fn evaluateBrdf(');               // forward
    expect(ADJOINT_SHADING_FD_WGSL).toContain('gradAdj: array<atomic<i32>>');     // analytic accumulator
    expect(ADJOINT_SHADING_FD_WGSL).toContain('gradFd:  array<atomic<i32>>');     // FD accumulator
    expect(ADJOINT_SHADING_FD_WGSL).toContain('let dLoss_dR = 2.0 * (rendered - s.tgt)'); // ∂loss/∂rendered
    expect(ADJOINT_SHADING_FD_WGSL).not.toContain('target:'); // 'target' is a reserved WGSL keyword
  });

  it('engine adjoint PASS bundles the real partials + re-trace + faceforward + scatter', () => {
    // GPU-validated end-to-end via wsl-gpu v24-inverse-fit --method=path-replay
    // (baseColor/roughness) + v24-emissive-fit.mjs (emissive) — each sign-matches the
    // FD gradient + drives a converging fit, lavapipe 2026-06-03.
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(PT_WEBGPU_PATH_TRACE_ADJOINT_WGSL); // byte-identical partials
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn generatePrimaryRay');           // re-trace
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('sampleCount: u32');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('ADJOINT_FROZEN_SEED_BASE + sampleIdx');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let invReplaySamples = 1.0 / f32(replaySamples)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('generatePrimaryRay(gid.x, gid.y, jitter)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn closestHit');                   // brute-force intersect
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn anyHit');                        // shadow rays
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('select(-nGeo, nGeo, dot(nGeo, ray.direction) < 0.0)'); // faceforward
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let m27 = materials[matId * MATERIAL_VEC4_STRIDE + 27u]');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let m26 = materials[matId * MATERIAL_VEC4_STRIDE + 26u]');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let isUnlit = (u32(max(m26.w, 0.0)) & 2u) != 0u');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dBrdf_dBaseColorWithSpecular(');
	    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dBrdf_dMetallic(');
	    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dBrdf_dSpecularColor(');
	    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dBrdf_dSpecularIntensity(');
	    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let m23 = materials[matId * MATERIAL_VEC4_STRIDE + 23u]');
	    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dBrdf_dClearcoat(');
	    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dBrdf_dClearcoatRoughness(');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('directionalLights');              // delta directional NEE
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('for (var di = 0u; di < params.directionalLightCount; di = di + 1u)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let directionalShadowDisabled = dDirAD.w < 0.0');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('spotLights');                     // spot NEE
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('for (var si = 0u; si < params.spotLightCount; si = si + 1u)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let softness = smoothstep(cosOuter, max(cosInner, cosOuter + 1e-6), coneCos)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('rectAreaLights');                  // rect-area NEE
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let isDisc = abs(rshape.w - 1.0) < 0.5');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('max(PI * dot(ru, ru), 1e-6)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('rectAreaLights[rb].w <= 0.5 && anyHit');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('cosLight * area / dist2');         // area geometric term
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('meshAreaLights');                  // mesh-area NEE
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let center = (a + b + c) * (1.0 / 3.0)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('mr.w <= 0.5 && anyHit');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let gUnlitBaseColor = dLoss_dR');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let gBase = select(gBaseColor, gUnlitBaseColor, isUnlit)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gBase.x * invReplaySamples)'); // per-param scatter
	    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gMetallic * invReplaySamples)');
	    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gClearcoat * invReplaySamples)');
	    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gClearcoatRoughness * invReplaySamples)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gSpecularColor.x * invReplaySamples)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gSpecularIntensity * invReplaySamples)');
    // Emissive is the camera-DIRECT primary-hit partial (NOT a NEE term): the fixed
    // emissiveIntensity rides in the descriptor `.w` (bitcast f32) and folds in.
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let emissiveIntensity = bitcast<f32>(d.w)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gEmissive.x * invReplaySamples)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let descBase = k * 2u');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(`d.y == ${ADJOINT_FIELD_EMISSIVE_INTENSITY}u`);
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dContribution_dEmissiveIntensity(vec3f(1.0), emissiveRgb)');
    // The UBO is mat4 + vec4 + 3×uvec4 = 128 bytes; the field codes are stable.
    expect(ADJOINT_PARAMS_UBO_BYTES).toBe(128);
    expect(ADJOINT_FIELD_BASECOLOR).toBe(0);
    expect(ADJOINT_FIELD_ROUGHNESS).toBe(1);
    expect(ADJOINT_FIELD_EMISSIVE).toBe(2);
    expect(ADJOINT_FIELD_SPECULAR_COLOR).toBe(3);
    expect(ADJOINT_FIELD_SPECULAR_INTENSITY).toBe(4);
	    expect(ADJOINT_FIELD_METALLIC).toBe(5);
	    expect(ADJOINT_FIELD_EMISSIVE_INTENSITY).toBe(6);
	    expect(ADJOINT_FIELD_CLEARCOAT).toBe(7);
	    expect(ADJOINT_FIELD_CLEARCOAT_ROUGHNESS).toBe(8);
	  });
});
