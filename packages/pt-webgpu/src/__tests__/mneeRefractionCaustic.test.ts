// mneeRefractionCaustic.test.ts — Phase I.1 sibling: the REAL MNEE point-light
// specular-REFRACTION caustic (the "water surface") wired into the beauty pass
// alongside the reflection caustic (caustic strategy mode 1).
//
// The EXECUTED radiometric correctness check runs on GPU: a {point light + flat
// refractive interface + diffuse floor below} scene rendered through pt-webgpu with
// causticStrategy:'manifold-nee', A/B'd against a DETERMINISTIC FORWARD-TRACED grid
// reference (light → interface → Snell → floor splat) — there is no analytic image
// for refraction. That A/B lives in wsl-gpu scripts/mnee-refraction-caustic-ab.ts
// and passes on lavapipe (integral ratio 0.986, least-squares slope 0.984, 99% of
// the lit floor firing). Here we pin the HOST-side WGSL so a refactor can't silently
// unwire it, and we pin the physics that makes refraction DIFFERENT from reflection
// (Fresnel transmittance + the focusing Jacobian — copying the reflection formula
// would be wrong).
import { describe, it, expect } from 'vitest';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL } from '../wgsl/pathTrace/caustic.wgsl.js';
import {
  PT_WEBGPU_TRACE_WGSL,
  composePtWebgpuTraceWgsl,
} from '../wgsl/pathTraceBruteforce.wgsl.js';

describe('MNEE refraction caustic — kernel wiring (Phase I.1 sibling)', () => {
  it('caustic module defines the refraction-caustic fn + the focusing-Jacobian helper', () => {
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('fn pointLightRefractionCaustic(');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('fn refractionFocusingDet(');
    // The DELTA connection: throughput · f_r · E (E = I·T·|dω_L/dA_recv|) — no MIS,
    // no pdf division (a point-light specular refraction caustic is unreachable by
    // any other technique, exactly like the reflection case).
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('contribution = contribution + throughput * fr * e;');
  });

  it('accumulates E = I · T · |dω_L/dA_recv| — Fresnel TRANSMITTANCE × the focusing Jacobian', () => {
    // (1) Fresnel TRANSMITTANCE 1 − Fr (NOT reflectance) via the dielectric Fresnel.
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let T = 1.0 - frDielectric(cosI, etaI / etaT);');
    // (2) the refraction FOCUSING Jacobian (NOT 1 like a flat mirror).
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'let focDet = refractionFocusingDet(v, ifaceN, ifaceTu, ifaceTv, hitPos, recvTu, recvTv, lightPos, etaI, etaT);',
    );
    // E = I · T · focDet — no separate cosθ_recv (it is inside the Jacobian).
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let e = lightI * T * focDet;');
  });

  it('the focusing Jacobian is |∂ω_L/∂u × ∂ω_L/∂v| by FD through the Newton solve', () => {
    // Perturb the receiver, re-solve the vertex warm-started at the converged v,
    // measure how ω_L = normalize(v − light) moves; the cross-product magnitude is
    // the focusing factor.
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let w0 = mnee_safe_normalize(vSolved - light);');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let dwu = (mnee_safe_normalize(ru.vertex - light) - w0) / eps;');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('return length(cross(dwu, dwv));');
  });

  it('seeds a TRANSMISSIVE interface (not metallic) + multi-seed Newton from the plane bracket', () => {
    // SEED: accept a transmissive interface (transmission ≥ REFRACT_TRANSMIT_MIN),
    // reject metallic (that is the reflection caustic's mirror).
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('const REFRACT_TRANSMIT_MIN');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'if (iMat.transmission < REFRACT_TRANSMIT_MIN || iMat.metallic > 0.5) {',
    );
    // The Newton is seeded from the PLANE geometry (receiver projection ↔ light-line
    // crossing), NOT the random seed-ray hit — the fix that took the firing fraction
    // from <1% to ~100% of the lit floor.
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let recvProj = hitPos - planeD0 * ifaceN;');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let seedCenter = (recvProj + lineCross) * 0.5;');
    // The branch / forward-consistency test: light→v refracted ray must aim at recv,
    // with the interface normal oriented AGAINST the incident travel for refract().
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'let nForRefr = select(ifaceN, -ifaceN, dot(wiTravel, ifaceN) > 0.0);',
    );
  });

  it('visibility-tests BOTH connection legs, stepping through transmissive/mirror facets', () => {
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('fn causticTransmissiveLegBlocked(');
    // leg A: receiver → v (bounded short of the interface vertex).
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'if (causticTransmissiveLegBlocked(hitPos + normal * 1e-3, wi, distA - 2e-3)) { continue; }',
    );
    // leg B: v → light.
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'if (causticTransmissiveLegBlocked(v + dirB * 1e-3, dirB, distB - 2e-3)) { continue; }',
    );
    // A transmissive interface facet (or a mirror facet) is stepped through; only an
    // opaque diffuse occluder blocks the connection.
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'let passThrough = segMat.transmission >= REFRACT_TRANSMIT_MIN ||',
    );
  });

  it('manifoldNeeContribution sums the refraction caustic for ANY receiver', () => {
    // It runs after the reflection caustic, independent of the legacy transmissive
    // cone-search gate — so a diffuse floor catches a water-surface caustic.
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('total = total + pointLightRefractionCaustic(');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,',
    );
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('anisotropy, anisotropyRotation,');
  });

  it('legacy transmissive cone-search uses packed RGB directional records', () => {
    const code = PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL
      .split('\n')
      .filter((line) => !line.trim().startsWith('//'))
      .join('\n');

    expect(code).toContain(
      'for (var dirIdx = 0u; dirIdx < params.directionalLightCount; dirIdx = dirIdx + 1u)',
    );
    expect(code).toContain('let dDirAD = directionalLights[dBase];');
    expect(code).toContain('let dIrrMean = directionalLights[dBase + 1u];');
    expect(code).toContain('let dirShadowDisabled = dDirAD.w < 0.0;');
    expect(code).toContain('let lightRadiance = dIrrMean.rgb * align;');
    expect(code).not.toContain('let lightRadiance = vec3f(params.lightDir.w) * align;');
    expect(code).not.toContain('params.lightDir.w <= 1e-6');
  });
});

describe('MNEE refraction caustic — trace-kernel composition', () => {
  it('composes the refraction caustic + its MNEE Newton dependency in scope', () => {
    // pointLightRefractionCaustic calls mneeNewtonSolve + mnee_safe_normalize
    // (MNEE_NEWTON_WGSL) + frDielectric (material.wgsl.ts) — all concatenated ahead.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn pointLightRefractionCaustic(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn mneeNewtonSolve(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn frDielectric(');
    // Ordering: the Newton solve + frDielectric must precede the caustic fn.
    const idxNewton = PT_WEBGPU_TRACE_WGSL.indexOf('fn mneeNewtonSolve(');
    const idxFr = PT_WEBGPU_TRACE_WGSL.indexOf('fn frDielectric(');
    const idxRefr = PT_WEBGPU_TRACE_WGSL.indexOf('fn pointLightRefractionCaustic(');
    expect(idxNewton).toBeGreaterThan(0);
    expect(idxFr).toBeGreaterThan(0);
    expect(idxRefr).toBeGreaterThan(idxNewton);
    expect(idxRefr).toBeGreaterThan(idxFr);
  });

  it('the BDPT-on variant ALSO carries the refraction caustic (caustic runs under BDPT)', () => {
    const bdptOn = composePtWebgpuTraceWgsl(true);
    expect(bdptOn).toContain('fn pointLightRefractionCaustic(');
    expect(bdptOn).toContain('fn refractionFocusingDet(');
  });
});
