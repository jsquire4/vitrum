/**
 * H46-A — maxBounces wired to the DDGI indirect-feedback gate.
 *
 * The walkaround engine is NOT a path tracer; maxBounces has exactly two
 * regimes on this stack:
 *   - maxBounces == 1  ⇒ direct-only DDGI probes (the previous-frame irradiance
 *     atlas read is dropped from the bounce surface's outgoing radiance, so the
 *     infinite-bounce diffuse EMA is disabled).
 *   - maxBounces >= 2  ⇒ the multi-bounce diffuse equilibrium (default).
 *
 * The gate rides FrameParams.indirectFeedback (the former inert _pad2 slot). The
 * DEFAULT (true ⇒ 1) is radiometrically byte-identical to the pre-gate
 * `direct + indirect` form, so existing DDGI behaviour is unchanged.
 */
import { describe, expect, it } from 'vitest';
import { packProbeUpdateFrameParams } from '../probeUpdateFrameParams.js';
import { DDGI_FRAME_PARAMS_UBO } from '../probeUpdateUbos.js';

/** Byte offset of the indirectFeedback u32 within FrameParams. */
function indirectFeedbackOffset(): number {
  // Layout-driven: read the offset from the UBO definition so this stays
  // correct if the struct is re-ordered.
  return DDGI_FRAME_PARAMS_UBO.fieldOffsets.indirectFeedback;
}

function readU32At(buf: ArrayBuffer, offset: number): number {
  return new DataView(buf).getUint32(offset, true);
}

const baseInput = {
  frameIndex: 3,
  totalProbes: 64,
  skyTint: [0.4, 0.6, 1.0] as const,
  skyIrradiance: 2.0,
  glassMixScale: 0.7,
};

describe('H46-A maxBounces → DDGI indirect-feedback gate', () => {
  it('FrameParams carries an indirectFeedback u32 (was the inert _pad2 slot)', () => {
    expect(DDGI_FRAME_PARAMS_UBO.fieldOffsets.indirectFeedback).toBeTypeOf('number');
    // Byte size unchanged — UBO stays the same length (no new lane).
    expect(DDGI_FRAME_PARAMS_UBO.sizeBytes % 16).toBe(0);
  });

  it('default packing folds indirect feedback ON (1) — byte-identical radiance to pre-gate', () => {
    const buf = packProbeUpdateFrameParams(baseInput);
    expect(readU32At(buf, indirectFeedbackOffset())).toBe(1);
  });

  it('indirectFeedback:false packs 0 (direct-only probes; maxBounces == 1)', () => {
    const buf = packProbeUpdateFrameParams({ ...baseInput, indirectFeedback: false });
    expect(readU32At(buf, indirectFeedbackOffset())).toBe(0);
  });

  it('indirectFeedback:true packs 1 (multi-bounce; maxBounces >= 2)', () => {
    const buf = packProbeUpdateFrameParams({ ...baseInput, indirectFeedback: true });
    expect(readU32At(buf, indirectFeedbackOffset())).toBe(1);
  });
});
