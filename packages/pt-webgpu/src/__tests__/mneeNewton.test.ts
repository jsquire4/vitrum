// mneeNewton.test.ts — the real-MNEE core: the half-vector Newton solve. The
// EXECUTED correctness check runs on GPU (wsl-gpu scripts/mnee-newton-validate.ts,
// lavapipe: residual→0 + the converged vertex == the analytic mirror reflection
// point to ~1e-5). Here we pin the host-side packing + the kernel composition.
import { describe, it, expect } from 'vitest';
import {
  MNEE_NEWTON_WGSL,
  MNEE_NEWTON_HARNESS_WGSL,
  packMneeHarnessInput,
  MNEE_HARNESS_INPUT_FLOATS,
  MNEE_NEWTON_MAX_ITERS,
} from '../wgsl/pathTrace/mneeNewton.wgsl.js';

describe('MNEE half-vector Newton solve (real-MNEE core)', () => {
  it('packs a config into the 12-float vec4-aligned record', () => {
    const r = packMneeHarnessInput([0, 0, 1], [1, 0, 1], [0, 0, 0]);
    expect(r).toHaveLength(MNEE_HARNESS_INPUT_FLOATS);
    expect(r.slice(0, 4)).toEqual([0, 0, 1, 0]);   // receiver.xyz, pad
    expect(r.slice(4, 8)).toEqual([1, 0, 1, 0]);   // light.xyz, pad
    expect(r.slice(8, 12)).toEqual([0, 0, 0, 0]);  // planePoint.xyz, pad
  });

  it('the solve is a half-vector Newton iteration (constraint + FD Jacobian)', () => {
    // The tangential half-vector residual (zero ⇔ h ∥ nm ⇔ reflection law).
    expect(MNEE_NEWTON_WGSL).toContain('fn mneeHalfVectorResidual2d(');
    expect(MNEE_NEWTON_WGSL).toContain('let h = mnee_safe_normalize(wi + wo)');
    expect(MNEE_NEWTON_WGSL).toContain('let hTan = h - dot(h, nm) * nm');
    // The Newton step: FD Jacobian + J·δ = −r.
    expect(MNEE_NEWTON_WGSL).toContain('fn mneeNewtonReflect(');
    expect(MNEE_NEWTON_WGSL).toContain('let det = j00 * j11 - j01 * j10');
    expect(MNEE_NEWTON_WGSL).toContain('if (rmag < 1e-5) { return out; }'); // convergence exit
  });

  it('harness kernel runs the solve over a flat mirror (nm=+z)', () => {
    expect(MNEE_NEWTON_HARNESS_WGSL).toContain(MNEE_NEWTON_WGSL); // byte-identical core
    expect(MNEE_NEWTON_HARNESS_WGSL).toContain('let nm = vec3f(0.0, 0.0, 1.0)');
    expect(MNEE_NEWTON_HARNESS_WGSL).toContain(`mneeNewtonReflect(c.planePoint, nm, tu, tv, c.recv, c.light, ${MNEE_NEWTON_MAX_ITERS}u)`);
    expect(MNEE_NEWTON_MAX_ITERS).toBe(16);
  });
});
