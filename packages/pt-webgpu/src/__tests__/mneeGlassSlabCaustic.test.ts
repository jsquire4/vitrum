// mneeGlassSlabCaustic.test.ts — Phase I.1 sibling: the REAL MNEE point-light
// 2-VERTEX GLASS-SLAB caustic (the canonical glass caustic beyond a single water
// surface) wired into the beauty pass alongside the reflection + refraction caustics
// (caustic strategy mode 1).
//
// The EXECUTED radiometric correctness check runs on GPU: a {point light + glass
// slab + diffuse floor below} scene rendered through pt-webgpu with
// causticStrategy:'manifold-nee', A/B'd against a DETERMINISTIC FORWARD-TRACED floor-
// flux reference (light → iface1 → glass → iface2 → floor splat) — there is no
// analytic image for a 2-interface refraction caustic. That A/B lives in wsl-gpu
// scripts/mnee-glass-slab-caustic-ab.ts; the focusing+Fresnel-product formula was
// proven OFFLINE FIRST in mnee-glass-slab-focusing-derivation.ts (integral ratio +
// LS-slope 1.000 vs a pure-JS forward-traced slab grid). Here we pin the HOST-side
// WGSL so a refactor can't silently unwire it, and we pin the physics that makes the
// slab chain DIFFERENT from the single interface (the per-interface Fresnel
// transmittance PRODUCT + the CHAIN focusing Jacobian — copying the single-interface
// form would be wrong).
import { describe, it, expect } from 'vitest';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL } from '../wgsl/pathTrace/caustic.wgsl.js';
import {
  PT_WEBGPU_TRACE_WGSL,
  composePtWebgpuTraceWgsl,
} from '../wgsl/pathTraceBruteforce.wgsl.js';

describe('MNEE glass-slab caustic — kernel wiring (Phase I.1 sibling)', () => {
  it('caustic module defines the glass-slab-caustic fn + the chain-focusing helper', () => {
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('fn pointLightGlassSlabCaustic(');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('fn slabChainFocusingDet(');
    // The DELTA connection: throughput · f_r · E (E = I·T1·T2·|dω_L/dA_recv|) — no MIS,
    // no pdf division (a point-light 2-vertex specular caustic is unreachable by any
    // other technique, exactly like the single-interface cases).
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('contribution = contribution + throughput * fr * e;');
  });

  it('accumulates E = I · T1 · T2 · |dω_L/dA_recv| — per-interface Fresnel PRODUCT × chain focusing', () => {
    // (1) Fresnel TRANSMITTANCE PRODUCT 1−Fr over BOTH interfaces (NOT a single T).
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let T1 = 1.0 - frDielectric(cosI1, eta1i / eta1t);');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let T2 = 1.0 - frDielectric(cosI2, eta2i / eta2t);');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let T = T1 * T2;');
    // (2) the CHAIN focusing Jacobian (re-solves the WHOLE chain — NOT the single-
    // interface refractionFocusingDet).
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'let focDet = slabChainFocusingDet(v1, v2, ifaceN1, tu1, tv1, ifaceN2, tu2, tv2, hitPos, recvTu, recvTv, lightPos, eta1i, eta1t, eta2i, eta2t);',
    );
    // E = I · (T1·T2) · focDet — no separate cosθ_recv (it is inside the Jacobian).
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let e = lightI * T * focDet;');
  });

  it('the chain focusing Jacobian is |∂ω_L/∂u × ∂ω_L/∂v| by FD through the CHAIN Newton solve', () => {
    // Perturb the receiver, re-solve BOTH chain vertices warm-started at (v1,v2),
    // measure how ω_L = normalize(v1 − light) moves; the cross-product magnitude is
    // the focusing factor. Crucially it re-solves the CHAIN (mneeNewtonSolveChain2),
    // and tracks ω_L off v1 (the light-side vertex).
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let w0 = mnee_safe_normalize(v1Solved - light);');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let ru = mneeNewtonSolveChain2(v1Solved, n1, tu1, tv1, v2Solved, n2, tu2, tv2, light, recv + recvTu * eps');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let dwu = (mnee_safe_normalize(ru.v1 - light) - w0) / eps;');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('return length(cross(dwu, dwv));');
  });

  it('seeds the slab lower interface + probes UP for the upper interface + the air→glass→air η chain', () => {
    // SEED the LOWER interface (the plane the floor sees → v2): a transmissive,
    // non-metallic surface.
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('const SLAB_TRANSMIT_MIN');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'if (mat2lower.transmission < SLAB_TRANSMIT_MIN || mat2lower.metallic > 0.5) {',
    );
    // PROBE up through the glass (−ifaceN2) for the UPPER interface (v1's plane).
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let upInGlass = -ifaceN2;');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let probeRay = Ray(ifaceP2 - ifaceN2 * 1e-3, upInGlass);');
    // η chain: air→glass at iface 1, glass→air at iface 2.
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let eta1i = 1.0; let eta1t = iorGlass;');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let eta2i = iorGlass; let eta2t = 1.0;');
  });

  it('block-tridiagonal 4-DOF chain Newton + forward-consistency through BOTH refractions', () => {
    // The coupled 2-vertex chain solve (mneeNewtonSolveChain2), reused from the
    // GPU-validated chain core (mnee-chain-validate.ts).
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'let chain = mneeNewtonSolveChain2(ifaceP1, ifaceN1, tu1, tv1, ifaceP2, ifaceN2, tu2, tv2, lightPos, hitPos, eta1i, eta1t, eta2i, eta2t,',
    );
    // Forward-consistency: light→v1 refracts air→glass and must aim at v2, then v1→v2
    // refracts glass→air and must aim at recv — confirming the real branch.
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let refr1 = refract(wiTravel1, n1Refr, eta1i / eta1t);');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('let refr2 = refract(wiTravel2, n2Refr, eta2i / eta2t);');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('if (dot(safe_normalize(refr2), safe_normalize(toRecv)) < 0.99) { continue; }');
  });

  it('visibility-tests the TWO external connection legs (the v1→v2 leg is interior glass)', () => {
    // leg A: receiver → v2 (bounded short of interface 2), through the shared helper.
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'if (causticTransmissiveLegBlocked(hitPos + normal * 1e-3, wi, distA - 2e-3)) { continue; }',
    );
    // leg B: v1 → light. (No separate v1→v2 test — that segment is the glass interior,
    // i.e. the connection itself.)
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'if (causticTransmissiveLegBlocked(v1 + dirB * 1e-3, dirB, distB - 2e-3)) { continue; }',
    );
  });

  it('manifoldNeeContribution sums the glass-slab caustic for ANY receiver', () => {
    // It runs after the reflection + refraction caustics, independent of the legacy
    // transmissive cone-search gate — so a diffuse floor catches a glass-slab caustic.
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'total = total + pointLightGlassSlabCaustic(rng, hitPos, normal, wo, baseColor, roughness, metallic, throughput);',
    );
  });

  it('the single-interface refraction caustic GUARDS against the chain double-count', () => {
    // The slab render A/B caught a double-count: the floor's seed found the slab
    // BOTTOM and pointLightRefractionCaustic solved it as a LONE interface, adding a
    // spurious second path (~the real slab caustic — ratio 2.17 instead of ~1). The
    // guard skips the single-interface solve when the light→v leg CROSSES a
    // transmissive facet (⇒ it is a multi-interface chain the slab kernel owns). The
    // water-surface scene is unaffected (its light→v leg is direct air). This pin
    // guards the discriminator so a refactor can't silently re-introduce the 2.17×.
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain('fn causticSegmentCrossesTransmissive(');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'if (causticSegmentCrossesTransmissive(lightPos + lvDir * 1e-3, lvDir, lvDist - 2e-3)) { continue; }',
    );
  });
});

describe('MNEE glass-slab caustic — trace-kernel composition', () => {
  it('composes the slab caustic + the MNEE chain Newton dependency in scope', () => {
    // pointLightGlassSlabCaustic calls mneeNewtonSolveChain2 + mnee_safe_normalize
    // (MNEE_CHAIN_WGSL / MNEE_NEWTON_WGSL) + frDielectric (material.wgsl.ts) — all
    // concatenated ahead.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn pointLightGlassSlabCaustic(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn mneeNewtonSolveChain2(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('fn mneeChainResidual4d(');
    // Ordering: the chain solve must precede the caustic fn that calls it, and the
    // single-vertex Newton (which the chain reuses via mnee_safe_normalize) precedes
    // the chain.
    const idxNewton = PT_WEBGPU_TRACE_WGSL.indexOf('fn mnee_safe_normalize(');
    const idxChain = PT_WEBGPU_TRACE_WGSL.indexOf('fn mneeNewtonSolveChain2(');
    const idxSlab = PT_WEBGPU_TRACE_WGSL.indexOf('fn pointLightGlassSlabCaustic(');
    expect(idxNewton).toBeGreaterThan(0);
    expect(idxChain).toBeGreaterThan(idxNewton);
    expect(idxSlab).toBeGreaterThan(idxChain);
  });

  it('the BDPT-on variant ALSO carries the glass-slab caustic (caustic runs under BDPT)', () => {
    const bdptOn = composePtWebgpuTraceWgsl(true);
    expect(bdptOn).toContain('fn pointLightGlassSlabCaustic(');
    expect(bdptOn).toContain('fn slabChainFocusingDet(');
    expect(bdptOn).toContain('fn mneeNewtonSolveChain2(');
  });
});
