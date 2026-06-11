/**
 * frameParamsPackerSizeGuard.test.ts — D8.11
 *
 * Pins the module-load-time allocation guard added to frameParamsPacker.ts:
 *   - FRAME_PARAMS_BYTE_SIZE (auto-generated struct size) must not exceed the
 *     512-byte ArrayBuffer that `packFrameParams` allocates.
 *   - The highest slot index derived from FRAME_PARAMS_BYTE_SIZE must fit
 *     within slots 0..127 (the 512-byte allocation).
 *
 * If a generator update pushes the struct past 512 bytes, the module-level
 * `throw` in frameParamsPacker.ts will fire at import time; the guard test
 * below verifies the constants stay in range so a stale-generated file
 * is caught before reaching production.
 */
import { describe, expect, it } from 'vitest';
import {
  FRAME_PARAMS_BUFFER_ALLOC_BYTES,
  FRAME_PARAMS_MAX_SLOT,
} from '../frameParamsPacker.js';
import { FRAME_PARAMS_BYTE_SIZE } from '../scene/frameParamsLayout.js';
import { FRAME_PARAMS_F32_SLOTS } from '../scene/frameParamsLayout.generated.js';

describe('FrameParamsPacker allocation guard (D8.11)', () => {
  it('FRAME_PARAMS_BUFFER_ALLOC_BYTES is 512 (the ArrayBuffer size in packFrameParams)', () => {
    expect(FRAME_PARAMS_BUFFER_ALLOC_BYTES).toBe(512);
  });

  it('FRAME_PARAMS_MAX_SLOT is 127 (highest slot index in the 512-byte buffer)', () => {
    // 512 bytes / 4 bytes per slot = 128 slots (indices 0..127).
    expect(FRAME_PARAMS_MAX_SLOT).toBe(127);
  });

  it('FRAME_PARAMS_BYTE_SIZE does not exceed the 512-byte buffer allocation', () => {
    expect(FRAME_PARAMS_BYTE_SIZE).toBeLessThanOrEqual(FRAME_PARAMS_BUFFER_ALLOC_BYTES);
  });

  it('FRAME_PARAMS_BYTE_SIZE derives slot count within FRAME_PARAMS_MAX_SLOT + 1', () => {
    // effectiveSlots = FRAME_PARAMS_BYTE_SIZE / 4 must be ≤ 128 (slots 0..127).
    const effectiveSlots = FRAME_PARAMS_BYTE_SIZE / 4;
    expect(effectiveSlots).toBeLessThanOrEqual(FRAME_PARAMS_MAX_SLOT + 1);
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
