// mneeReflectionCaustic.test.ts — Phase I.1 step 3: the REAL MNEE point-light
// specular-REFLECTION caustic wired into the beauty pass (caustic strategy mode 1).
//
// The EXECUTED radiometric correctness check runs on GPU: a {point light + flat
// mirror + diffuse floor} scene rendered through pt-webgpu with
// causticStrategy:'manifold-nee', A/B'd against the DETERMINISTIC analytic
// mirror-image floor irradiance (wsl-gpu scripts/mnee-reflection-caustic-ab.mjs).
// The mneeReflectionIrradiance core is ALREADY GPU-validated against the analytic
// mirror-image irradiance (mnee-reflection-validate.ts). Here we pin the HOST-side
// WGSL composition + the kernel wiring so a refactor can't silently unwire it.
import { describe, it, expect } from 'vitest';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL } from '../wgsl/pathTrace/caustic.wgsl.js';
import {
  PT_WEBGPU_TRACE_WGSL,
  composePtWebgpuTraceWgsl,
} from '../wgsl/pathTraceBruteforce.wgsl.js';
import {
  MNEE_NEWTON_WGSL,
  MNEE_CONNECTION_WGSL,
} from '../wgsl/pathTrace/mneeNewton.wgsl.js';

describe('MNEE reflection caustic — kernel wiring (Phase I.1)', () => {
  it('caustic module defines the real reflection-caustic fn and calls the validated core', () => {
    // The new contribution fn + the GPU-validated irradiance core call.
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('fn pointLightReflectionCaustic(');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'let e = mneeReflectionIrradiance(hitPos, normal, mirrorP, mirrorN, mTu, mTv, lightPos, lightI);',
    );
    // The DELTA connection: throughput · f_r · E (E carries cosθ_recv) — no MIS,
    // no pdf division (no other technique reaches a point-light mirror caustic).
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'contribution = contribution + throughput * fr * e;',
    );
  });

  it('wires finite rect/disc and mesh-area reflection MNEE through the area determinant', () => {
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('fn finiteAreaReflectionCaustic(');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('fn areaLightReflectionCausticSample(');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let rectCount = params.rectAreaLightCount;');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let meshCount = params.meshAreaLightCount;');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let disc = concentricDiscSample(vec2f(xi1 * 2.0 - 1.0, xi2 * 2.0 - 1.0));');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('lightArea = max(0.5 * length(cross(lightU, lightV)), 1e-6);');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let lightToVertex = safe_normalize(v - lightPos);');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let det = mneePdfJacobianDetAxes(v, hitPos, jac.dadL, jac.dbdL, mirrorTu, mirrorTv, lightU, lightV);');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let lightPdf = (1.0 / max(lightArea, 1e-8)) / max(det, 1e-12);');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('baseColor, roughness, metallic, transmission, ior, normal, wo, wi,');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let misWeight = powerHeuristic(lightPdf, brdfPdf);');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('contribution = contribution + sample * f32(finiteCount);');
  });

  it('iterates point lights with the correct stride via POINT_LIGHT_VEC4_STRIDE (H51-D / H1-class fix)', () => {
    // H51-D bumped the point stride to 3 vec4f (12 floats): [pos, radiance, dist+decay].
    // caustic.wgsl.ts previously used the stale stride-2 literal (`li * 2u`), which
    // read position/radiance from the wrong vec4f slots for lights 1+. The fix
    // uses POINT_LIGHT_VEC4_STRIDE (=3u, declared in material.wgsl.ts, composed
    // before caustic) — a shared constant that is parity-tested in emitterStride.test.ts.
    // caustic point/spot stride fix (H1-class), 2026-06-10.
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).not.toContain('let lbase = li * 2u;');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let lbase = li * POINT_LIGHT_VEC4_STRIDE;');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let lightPos = pointLights[lbase].xyz;');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let lightI = pointLights[lbase + 1u].rgb;');
  });

  it('seeds the mirror via hemisphere ray-casts + accepts only smooth metallic hits', () => {
    // (a) SEED: cosine-hemisphere seed rays from the receiver, traceClosest each.
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let seedHit = traceClosest(seedRay, 1e-4, INFINITY);');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('buildOnb(normal, &st, &sb);');
    // A "mirror" is smooth + metallic (rejects the diffuse floor/walls as reflectors).
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('const REFLECT_ROUGH_MAX');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('const REFLECT_METAL_MIN');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'if (mMat.roughness > REFLECT_ROUGH_MAX || mMat.metallic < REFLECT_METAL_MIN) {',
    );
  });

  it('visibility-tests BOTH connection legs (receiver→vertex and vertex→light)', () => {
    // leg A: receiver → v (bounded just short of the mirror vertex).
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let rayA = Ray(hitPos + normal * 1e-3, wi);');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('if (traceAny(rayA, 1e-4, max(distA - 2e-3, 1e-3))) { continue; }');
    // leg B: v → light, stepping THROUGH the mirror's own facets (a thin mirror
    // SOLID has a second facet between the reflection vertex and the light; a naive
    // shadow ray self-occludes on it — the bug the render A/B caught). Only a
    // non-mirror hit blocks the connection.
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let toLight = lightPos - v;');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let segRay = Ray(legBOrigin, dirB);');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'let isMirror = segMat.roughness <= REFLECT_ROUGH_MAX && segMat.metallic >= REFLECT_METAL_MIN;',
    );
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('if (!isMirror) { legBBlocked = true; break; }');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('if (legBBlocked) { continue; }');
  });

  it('manifoldNeeContribution dispatches the reflection caustic for ANY receiver', () => {
    // The reflection caustic runs BEFORE (and independent of) the legacy
    // transmissive cone-search gate — so a diffuse floor catches a mirror caustic
    // with no glass present. The transmissive branch sums on top.
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('var total = pointLightReflectionCaustic(');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('total = total + finiteAreaReflectionCaustic(');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,',
    );
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('anisotropy, anisotropyRotation,');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('return total + transmissiveContribution;');
  });
});

describe('MNEE reflection caustic — trace-kernel composition', () => {
  it('composes the MNEE Newton + connection cores AHEAD of caustic (struct-before-use)', () => {
    // pointLightReflectionCaustic calls mneeReflectionIrradiance (MNEE_CONNECTION_WGSL),
    // which calls mneeNewtonSolve + mnee_safe_normalize (MNEE_NEWTON_WGSL); both must
    // be concatenated before the caustic module in the composed shader.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn mneeNewtonSolve(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn mneeReflectionIrradiance(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn pointLightReflectionCaustic(');
    // Ordering: the connection core (mneeReflectionIrradiance) must appear BEFORE
    // its only call site (inside pointLightReflectionCaustic).
    const idxCore = PT_WEBGPU_TRACE_WGSL.indexOf('fn mneeReflectionIrradiance(');
    const idxCall = PT_WEBGPU_TRACE_WGSL.indexOf('fn pointLightReflectionCaustic(');
    expect(idxCore).toBeGreaterThan(0);
    expect(idxCall).toBeGreaterThan(idxCore);
    // And the Newton solve must precede the connection core that calls it.
    const idxNewton = PT_WEBGPU_TRACE_WGSL.indexOf('fn mneeNewtonSolve(');
    expect(idxNewton).toBeGreaterThan(0);
    expect(idxCore).toBeGreaterThan(idxNewton);
  });

  it('the BDPT-on variant ALSO carries the MNEE reflection core (caustic runs under BDPT)', () => {
    const bdptOn = composePtWebgpuTraceWgsl(true);
    expect(bdptOn).toContain('fn mneeReflectionIrradiance(');
    expect(bdptOn).toContain('fn pointLightReflectionCaustic(');
  });

  it('the composed MNEE modules are byte-identical to the source exports', () => {
    // No accidental whitespace/edit drift between the standalone validated core
    // exports and what is concatenated into the trace kernel.
    expect(PT_WEBGPU_TRACE_WGSL).toContain(MNEE_NEWTON_WGSL);
    expect(PT_WEBGPU_TRACE_WGSL).toContain(MNEE_CONNECTION_WGSL);
  });
});
