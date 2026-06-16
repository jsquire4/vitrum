// adjointHarness.test.ts — the V24 GPU-adjoint harness kernel composes the
// path-replay BRDF partials + their primitives into a dispatchable shader. The
// EXECUTED GPU == CPU-oracle A/B runs on real hardware (wsl-gpu
// scripts/adjoint-validate.ts, lavapipe: max relative err ~2e-7 = f32 precision);
// here we pin the host-side packing + that the kernel bundles the real partials.
import { readFileSync } from 'node:fs';
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
  ADJOINT_FIELD_SHEEN,
  ADJOINT_FIELD_SHEEN_ROUGHNESS,
  ADJOINT_FIELD_SHEEN_COLOR,
  ADJOINT_FIELD_IRIDESCENCE,
  ADJOINT_FIELD_IRIDESCENCE_IOR,
  ADJOINT_FIELD_IRIDESCENCE_THICKNESS_RANGE,
  ADJOINT_FIELD_ANISOTROPY,
  ADJOINT_FIELD_ANISOTROPY_ROTATION,
  ADJOINT_FIELD_EMITTER_COLOR,
  ADJOINT_FIELD_EMITTER_INTENSITY,
  ADJOINT_EMITTER_TARGET_DIRECTIONAL,
  ADJOINT_EMITTER_TARGET_POINT,
  ADJOINT_EMITTER_TARGET_SPOT,
  ADJOINT_EMITTER_TARGET_RECT,
  ADJOINT_EMITTER_TARGET_MESH,
} from '../wgsl/pathTrace/adjointPass.wgsl.js';

const ADJOINT_PASS_TS = readFileSync(new URL('../adjointPass.ts', import.meta.url), 'utf8');

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
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let m24 = materials[matId * MATERIAL_VEC4_STRIDE + 24u]');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let m25 = materials[matId * MATERIAL_VEC4_STRIDE + 25u]');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let sheen = clamp(m23.z, 0.0, 1.0)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let sheenRoughness = clamp(m23.w, 0.0, 1.0)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let sheenColor = clamp(m24.rgb, vec3f(0.0), vec3f(1.0))');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let iridescence = clamp(m24.w, 0.0, 1.0)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let iridescenceIor = max(m25.x, 1.0)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let materialTexBase = matId * ADJOINT_MATERIAL_TEX_VEC4_STRIDE');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let anisoDesc = materialTexDescriptors[materialTexBase + 5u]');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('anisotropy = clamp(anisoDesc.x, 0.0, 1.0)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dBrdf_dSheen(');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dBrdf_dSheenColor(');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dBrdf_dSheenRoughness(');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dBrdf_dIridescence(');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dBrdf_dIridescenceIor(');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dBrdf_dAnisotropy(');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dBrdf_dAnisotropyRotation(');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('directionalLights');              // delta directional NEE
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('for (var di = 0u; di < params.directionalLightCount; di = di + 1u)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let directionalShadowDisabled = dDirAD.w < 0.0');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('spotLights');                     // spot NEE
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('for (var si = 0u; si < params.spotLightCount; si = si + 1u)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let softness = smoothstep(cosOuter, max(cosInner, cosOuter + 1e-6), coneCos)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('rectAreaLights');                  // rect-area NEE
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let isDisc = abs(rshape.w - 1.0) < 0.5');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn adjointConcentricDiscSample');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let xi1 = rand_f32(&rng)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('area = max(PI * r * r, 1e-6)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('rectAreaLights[rb].w <= 0.5 && anyHit');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let lightPdf = dist2 / max(cosLight * area, 1e-6)'); // area PDF
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('meshAreaLights');                  // mesh-area NEE
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let su = sqrt(r1)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).not.toContain('let center = (a + b + c) * (1.0 / 3.0)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('mr.w <= 0.5 && anyHit');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let descKind = d.w & 255u');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('targetSlot >= d.x + descCount');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(`${ADJOINT_EMITTER_TARGET_MESH}u`);
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn sampleAdjointBaseColorTexture');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn sampleAdjointAoFactor');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn sampleAdjointVertexColor');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn sampleAdjointOrmTexture');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn sampleAdjointClearcoatTexture');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn sampleAdjointClearcoatRoughnessTexture');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn sampleAdjointSheenColorTexture');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn sampleAdjointSheenRoughnessTexture');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn sampleAdjointIridescenceTexture');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn sampleAdjointIridescenceThicknessTexture');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn sampleAdjointAnisotropyTexture');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn sampleAdjointSpecularColorTexture');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn sampleAdjointSpecularIntensityTexture');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let baseColorFactor = sampleAdjointVertexColor(hit.tri, vec2f(hit.bary.y, hit.bary.z)).rgb *');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('sampleAdjointBaseColorTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z)).rgb');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('sampleAdjointAoFactor(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z))');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let effectiveBaseColor = baseColor * baseColorFactor');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let ormFactor = sampleAdjointOrmTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z))');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let effectiveRoughness = clamp(roughness * ormFactor.g, 0.02, 1.0)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let effectiveMetallic = clamp(metallic * ormFactor.b, 0.0, 1.0)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let specularColorFactor = sampleAdjointSpecularColorTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z))');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let specularIntensityFactor = sampleAdjointSpecularIntensityTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z))');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let effectiveSpecularColor = clamp(specularColor * specularColorFactor, vec3f(0.0), vec3f(1.0))');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let effectiveSpecularIntensity = clamp(specularIntensity * specularIntensityFactor, 0.0, 1.0)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let clearcoatFactor = sampleAdjointClearcoatTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z))');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let clearcoatRoughnessFactor = sampleAdjointClearcoatRoughnessTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z))');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let sheenColorFactor = sampleAdjointSheenColorTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z))');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let sheenRoughnessFactor = sampleAdjointSheenRoughnessTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z))');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let effectiveClearcoat = clamp(clearcoat * clearcoatFactor, 0.0, 1.0)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let effectiveClearcoatRoughness = clamp(clearcoatRoughness * clearcoatRoughnessFactor, 0.0, 1.0)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let effectiveSheenColor = clamp(sheenColor * sheenColorFactor, vec3f(0.0), vec3f(1.0))');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let effectiveSheenRoughness = clamp(sheenRoughness * sheenRoughnessFactor, 0.0, 1.0)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let iridescenceFactor = sampleAdjointIridescenceTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z))');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let iridescenceThicknessSample = sampleAdjointIridescenceThicknessTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z))');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('var effectiveIridescence = clamp(iridescence * iridescenceFactor, 0.0, 1.0)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let iridescenceThickness = mix(iridescenceThicknessMin, iridescenceThicknessMax, iridescenceThicknessSample)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('authoredIridescenceThicknessMin: f32');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('authoredIridescenceThicknessMax: f32');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('iridescenceThicknessTexel: f32');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('iridescenceThicknessMin, iridescenceThicknessMax, iridescenceThicknessSample');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let anisotropyMapSample = sampleAdjointAnisotropyTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z))');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let effectiveAnisotropy = clamp(anisotropy * anisotropyMapSample.strength, 0.0, 1.0)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let effectiveAnisotropyRotation = anisotropyRotation + anisotropyMapSample.rotationOffset');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let gUnlitBaseColor = dLoss_dR * baseColorFactor');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let gBase = select(gBaseColor * baseColorFactor, gUnlitBaseColor, isUnlit)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gBase.x * invReplaySamples)'); // per-param scatter
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gRough * ormFactor.g * invReplaySamples)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gMetallic * ormFactor.b * invReplaySamples)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gClearcoat * clearcoatFactor * invReplaySamples)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gClearcoatRoughness * clearcoatRoughnessFactor * invReplaySamples)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gSheen * invReplaySamples)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gSheenRoughness * sheenRoughnessFactor * invReplaySamples)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gSheenColor.x * sheenColorFactor.x * invReplaySamples)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gIridescence * iridescenceGradientFactor * invReplaySamples)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gIridescenceIor * invReplaySamples)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gAnisotropy * anisotropyMapSample.strength * invReplaySamples)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gAnisotropyRotation * invReplaySamples)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gSpecularColor.x * specularColorFactor.x * invReplaySamples)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gSpecularIntensity * specularIntensityFactor * invReplaySamples)');
    // Emissive is the camera-DIRECT primary-hit partial (NOT a NEE term): the fixed
    // emissiveIntensity rides in the descriptor `.w` (bitcast f32) and folds in;
    // camera-direct emissive maps are sampled with the forward sRGB UV metadata.
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('@group(0) @binding(14) var<storage, read>      meshUvs');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('@group(0) @binding(15) var<storage, read>      materialTexDescriptors');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('@group(0) @binding(16) var                      materialTextures');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('@group(0) @binding(17) var                      materialTexSampler');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('@group(0) @binding(18) var<storage, read>       meshVertexColors');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('@group(0) @binding(19) var                      materialTexturesLinear');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('const ADJOINT_MATERIAL_TEX_VEC4_STRIDE = 82u;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('const ADJOINT_MATERIAL_TEX_UV_BASE_COLOR = 19u;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('const ADJOINT_MATERIAL_TEX_UV_EMISSIVE = 21u;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('const ADJOINT_MATERIAL_TEX_UV_ROUGHNESS = 25u;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('const ADJOINT_MATERIAL_TEX_UV_METALLIC = 27u;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('const ADJOINT_MATERIAL_TEX_UV_AO = 29u;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('const ADJOINT_MATERIAL_TEX_UV_CLEARCOAT = 51u;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('const ADJOINT_MATERIAL_TEX_UV_CLEARCOAT_ROUGHNESS = 53u;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('const ADJOINT_MATERIAL_TEX_UV_SHEEN_COLOR = 55u;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('const ADJOINT_MATERIAL_TEX_UV_SHEEN_ROUGHNESS = 57u;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('const ADJOINT_MATERIAL_TEX_UV_IRIDESCENCE = 59u;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('const ADJOINT_MATERIAL_TEX_UV_IRIDESCENCE_THICKNESS = 61u;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('const ADJOINT_MATERIAL_TEX_UV_SPECULAR_COLOR = 63u;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('const ADJOINT_MATERIAL_TEX_UV_SPECULAR_INTENSITY = 65u;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('const ADJOINT_MATERIAL_TEX_UV_ANISOTROPY = 35u;');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn sampleAdjointEmissiveTexture');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let emissiveTexel = sampleAdjointEmissiveTexture(matId, hit.tri, vec2f(hit.bary.y, hit.bary.z)).rgb');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let emissiveIntensity = bitcast<f32>(d.w)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dRendered_dEmissivePerUnitIntensity * emissiveTexel * emissiveIntensity');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('adjointScatter(gradOffset, gEmissive.x * invReplaySamples)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('let descBase = k * 2u');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(`d.y == ${ADJOINT_FIELD_EMISSIVE_INTENSITY}u`);
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dContribution_dEmissiveIntensity(vec3f(1.0), emissiveRgb * emissiveTexel)');
    // Emitter color/intensity target gradients are separate from material slots:
    // the descriptor matches kind-local light slots and scatters through the
    // direct-light BRDF value plus the same attenuation/geometric factors used
    // for material direct-light adjoints.
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn directLightBrdfValue');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('fn scatterEmitterRadianceGradient');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(`d.y != ${ADJOINT_FIELD_EMITTER_COLOR}u && d.y != ${ADJOINT_FIELD_EMITTER_INTENSITY}u`);
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(`d.y == ${ADJOINT_FIELD_EMITTER_COLOR}u || d.y == ${ADJOINT_FIELD_EMITTER_INTENSITY}u`);
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(`${ADJOINT_EMITTER_TARGET_DIRECTIONAL}u`);
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(`${ADJOINT_EMITTER_TARGET_POINT}u`);
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(`${ADJOINT_EMITTER_TARGET_SPOT}u`);
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain(`${ADJOINT_EMITTER_TARGET_RECT}u`);
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dLoss_dR * brdfValue * (nDotL * attenuation)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dLoss_dR * brdfValue * (nDotL * softness * attenuation)');
    expect(PT_WEBGPU_ADJOINT_PASS_WGSL).toContain('dLoss_dR * brdfValue * (nDotL * areaFactor)');
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
    expect(ADJOINT_FIELD_SHEEN).toBe(9);
    expect(ADJOINT_FIELD_SHEEN_ROUGHNESS).toBe(10);
    expect(ADJOINT_FIELD_SHEEN_COLOR).toBe(11);
    expect(ADJOINT_FIELD_IRIDESCENCE).toBe(12);
    expect(ADJOINT_FIELD_IRIDESCENCE_IOR).toBe(13);
    expect(ADJOINT_FIELD_ANISOTROPY).toBe(14);
    expect(ADJOINT_FIELD_ANISOTROPY_ROTATION).toBe(15);
    expect(ADJOINT_FIELD_EMITTER_COLOR).toBe(16);
    expect(ADJOINT_FIELD_EMITTER_INTENSITY).toBe(17);
    expect(ADJOINT_FIELD_IRIDESCENCE_THICKNESS_RANGE).toBe(18);
    expect(ADJOINT_EMITTER_TARGET_DIRECTIONAL).toBe(1);
    expect(ADJOINT_EMITTER_TARGET_POINT).toBe(2);
    expect(ADJOINT_EMITTER_TARGET_SPOT).toBe(3);
    expect(ADJOINT_EMITTER_TARGET_RECT).toBe(4);
    expect(ADJOINT_EMITTER_TARGET_MESH).toBe(5);
  });

  it('engine adjoint PASS binds the material texture replay resources', () => {
    expect(ADJOINT_PASS_TS).toContain('binding: 14, resource: { buffer: sb.uvsBuffer }');
    expect(ADJOINT_PASS_TS).toContain('binding: 15, resource: { buffer: sb.materialTexDescriptorsBuffer }');
    expect(ADJOINT_PASS_TS).toContain('binding: 16, resource: sb.materialTextureView');
    expect(ADJOINT_PASS_TS).toContain('binding: 17, resource: sb.materialTextureSampler');
    expect(ADJOINT_PASS_TS).toContain('binding: 18, resource: { buffer: sb.colorsBuffer }');
    expect(ADJOINT_PASS_TS).toContain('binding: 19, resource: sb.materialLinearTextureView');
    expect(ADJOINT_PASS_TS).toContain("case 'iridescenceIor':");
    expect(ADJOINT_PASS_TS).toContain('fieldCode = ADJOINT_FIELD_IRIDESCENCE_IOR;');
    expect(ADJOINT_PASS_TS).toContain("case 'iridescenceThicknessRange':");
    expect(ADJOINT_PASS_TS).toContain('fieldCode = ADJOINT_FIELD_IRIDESCENCE_THICKNESS_RANGE;');
    expect(ADJOINT_PASS_TS).toContain("case 'anisotropy':");
    expect(ADJOINT_PASS_TS).toContain('fieldCode = ADJOINT_FIELD_ANISOTROPY;');
    expect(ADJOINT_PASS_TS).toContain("case 'anisotropyRotation':");
    expect(ADJOINT_PASS_TS).toContain('fieldCode = ADJOINT_FIELD_ANISOTROPY_ROTATION;');
    expect(ADJOINT_PASS_TS).toContain("if (p.domain === 'emitters')");
    expect(ADJOINT_PASS_TS).toContain('fieldCode = ADJOINT_FIELD_EMITTER_COLOR;');
    expect(ADJOINT_PASS_TS).toContain('adjointEmitterTargetForScene(scene, p.id)');
    expect(ADJOINT_PASS_TS).toContain('kind: ADJOINT_EMITTER_TARGET_POINT');
    expect(ADJOINT_PASS_TS).toContain('kind: ADJOINT_EMITTER_TARGET_RECT');
  });
});
