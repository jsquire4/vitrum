/**
 * frameParamsPackerSizeGuard.test.ts — D8.11
 *
 * Pins the module-load-time allocation guard added to frameParamsPacker.ts:
 * Pins that the generated WGSL struct, CPU ArrayBuffer, and GPU allocation use
 * one exact size, with no stale capacity standing in for removed semantics.
 */
import { describe, expect, it } from 'vitest';
import {
  FRAME_PARAMS_BUFFER_ALLOC_BYTES,
  FRAME_PARAMS_MAX_SLOT,
} from '../frameParamsPacker.js';
import { FRAME_PARAMS_BYTE_SIZE } from '../scene/frameParamsLayout.js';
import { FRAME_PARAMS_F32_SLOTS } from '../scene/frameParamsLayout.generated.js';

describe('FrameParamsPacker allocation guard (D8.11)', () => {
  it('allocates exactly the generated FrameParams size', () => {
    expect(FRAME_PARAMS_BUFFER_ALLOC_BYTES).toBe(FRAME_PARAMS_BYTE_SIZE);
    expect(FRAME_PARAMS_BUFFER_ALLOC_BYTES).toBe(384);
  });

  it('derives the highest slot from the exact allocation', () => {
    expect(FRAME_PARAMS_MAX_SLOT).toBe(95);
  });

  it('does not retain trailing allocation beyond the WGSL struct', () => {
    expect(FRAME_PARAMS_BYTE_SIZE).toBe(FRAME_PARAMS_BUFFER_ALLOC_BYTES);
  });

  it('FRAME_PARAMS_BYTE_SIZE derives slot count within FRAME_PARAMS_MAX_SLOT + 1', () => {
    const effectiveSlots = FRAME_PARAMS_BYTE_SIZE / 4;
    expect(effectiveSlots).toBe(FRAME_PARAMS_MAX_SLOT + 1);
  });

  it('FRAME_PARAMS_F32_SLOTS matches FRAME_PARAMS_BYTE_SIZE / 4', () => {
    // The generated file exports both; verify they are consistent.
    expect(FRAME_PARAMS_F32_SLOTS).toBe(FRAME_PARAMS_BYTE_SIZE / 4);
  });

  it('importing frameParamsPacker does not throw (module-load guard passes)', () => {
    // If the module-level throw fired, this import would have failed and the
    // test suite itself would not have loaded.  This test is a signal that
    // the guard is currently GREEN (no throw).
    expect(FRAME_PARAMS_BUFFER_ALLOC_BYTES).toBeGreaterThan(0);
  });
});
