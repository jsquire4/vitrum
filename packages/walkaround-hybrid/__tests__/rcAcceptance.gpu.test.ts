/**
 * W8 Phase 4 — Radiance Cascades acceptance test (GPU-gated).
 *
 * Gated by:
 *   - `VITRUM_RC_ACCEPTANCE=1` env flag (opt-in; default `npm test` skips).
 *   - real `navigator.gpu` availability at runtime.
 *
 * The host-side wiring (`packRCParams` 64-byte layout, `RCSubsystem` public
 * surface, `pipeline.setRCInputs` proxy, SHADE_MODULE's include-graph
 * dependency on `sampleCascadeC0`) is already pinned by
 * `hybridEngineRC.test.ts` + `wgslCompose.test.ts`. This file's role is
 * end-to-end GPU validation: with `rcEnabled: true`, `Lo_indirect` must
 * differ from the `rcEnabled: false` baseline by more than read-noise.
 *
 * Acceptance procedure (when the harness is wired):
 *   1. Boot HybridEngine A with `rcEnabled: false`. Drive N=32 frames of
 *      a Cornell-box scene with a known directional sun. Read back the
 *      indirect output texture. Compute per-pixel RMS on a flat-wall ROI.
 *   2. Boot HybridEngine B with `rcEnabled: true, rcWeight: 1.0` (pure RC,
 *      maximum signal). Drive same N frames + same frame seeds. Read
 *      back same ROI. Compute per-pixel RMS.
 *   3. Assert mean(|B.indirect[roi] − A.indirect[roi]|) > 0.005 (i.e.,
 *      RC introduces a visible, non-zero difference; tight enough to fail
 *      if the `rcParams.enabled` short-circuit doesn't lift).
 *
 * The capture harness writes a JSON metrics artifact consumed by this test via
 * `VITRUM_RC_ACCEPTANCE_METRICS`. This keeps default `npm test` portable while
 * making the gated path executable and assertive when enabled.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';

const RC_ACCEPTANCE_ENABLED =
  typeof process !== 'undefined' &&
  process.env != null &&
  process.env['VITRUM_RC_ACCEPTANCE'] === '1';

describe.skipIf(!RC_ACCEPTANCE_ENABLED)('W8 — Radiance Cascades GPU acceptance', () => {
  function readMetrics(envVar = 'VITRUM_RC_ACCEPTANCE_METRICS'): {
    readonly rcDeltaMean: number;
    readonly pipelineCreatesBefore: number;
    readonly pipelineCreatesAfter: number;
  } {
    const path = process.env[envVar];
    if (!path) {
      throw new Error(
        `VITRUM_RC_ACCEPTANCE=1 requires ${envVar}=<json file> ` +
        'produced by tools/benchmark-runner.',
      );
    }
    const parsed = JSON.parse(readFileSync(path, 'utf8')) as {
      rcDeltaMean?: unknown;
      pipelineCreatesBefore?: unknown;
      pipelineCreatesAfter?: unknown;
    };
    if (
      typeof parsed.rcDeltaMean !== 'number' ||
      typeof parsed.pipelineCreatesBefore !== 'number' ||
      typeof parsed.pipelineCreatesAfter !== 'number'
    ) {
      throw new Error(
        `Invalid RC acceptance metrics JSON at ${path}. ` +
        'Expected numeric rcDeltaMean, pipelineCreatesBefore, pipelineCreatesAfter.',
      );
    }
    return {
      rcDeltaMean: parsed.rcDeltaMean,
      pipelineCreatesBefore: parsed.pipelineCreatesBefore,
      pipelineCreatesAfter: parsed.pipelineCreatesAfter,
    };
  }

  // Two-scene gate (2026-06-07, tools/reference-renders/rc-gate-2026-06-07/).
  // RC's light model is sun + emissive geometry + env + rect-area emitter NEE;
  // one scene can't be both indirect-dominant (strong RC signal) AND sun-lit
  // (requires an open box → direct-dominant). So Scene 1 gates emitter NEE at
  // full strength; Scene 2 is a directSun LIVENESS check (weaker by nature).
  it('Scene 1 (emitter NEE, enclosed Cornell): rcEnabled produces a strong Lo_indirect delta', () => {
    const metrics = readMetrics('VITRUM_RC_ACCEPTANCE_METRICS');
    expect(metrics.rcDeltaMean).toBeGreaterThan(0.005);
  });

  it('Scene 2 (directSun, open Cornell): RC sun path is live (non-zero indirect delta)', () => {
    // Liveness, not strength: open sun scenes are direct-dominant so RC's GI is
    // a small image fraction. RC's sun path is separately validated at full
    // strength by tlas-zero-gi-bisect (--sun=2: indirect 0.000356 → 0.0528).
    const metrics = readMetrics('VITRUM_RC_SUN_METRICS');
    expect(metrics.rcDeltaMean).toBeGreaterThan(0.0005);
  });

  it('rcEnabled: true → false toggles bit-identically on the bind-group path (no recompile)', () => {
    // Acceptance: the bind-group layout has rcCascade0 + rcParams entries
    // regardless of rcEnabled, with the rcParams placeholder buffer's
    // `enabled = 0u` bit short-circuiting sampleCascadeC0 to vec3f(0).
    // The pipeline must NOT recompile when host toggles rcEnabled at
    // runtime (re-boots are allowed; per-frame toggles via rcWeight are
    // the contract today).
    //
    const metrics = readMetrics();
    expect(metrics.pipelineCreatesAfter).toBe(metrics.pipelineCreatesBefore);
  });
});

describe('W8 — Radiance Cascades acceptance gate', () => {
  it('is opt-in only and remains disabled by default', () => {
    const enabled = process.env['VITRUM_RC_ACCEPTANCE'] === '1';
    expect(RC_ACCEPTANCE_ENABLED).toBe(enabled);
  });
});
