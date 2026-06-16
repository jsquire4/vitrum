// recordUnpack.test.ts — pins the NRC record gap-detection + dense-repack loop
// (D7.6, extracted from NrcSubsystem.trainFromRecords into recordUnpack.ts).
//
// Record layout (recordStride f32s): [ inW input | 3 target | 3 world pos ].
// A6 empty-slot semantics: a slot is EMPTY iff every encoded-input float is zero
// (the GPU zero-initializes the record buffer and nrcWriteRecord only runs when
// the spread heuristic fired); zero-TARGET records are VALID samples.

import { describe, it, expect } from 'vitest';
import { unpackRecords } from '../recordUnpack.ts';

const OUT_W = 3;

/** Build a raw record buffer [cap × stride] with the given records placed at
 *  their slot indices (gaps stay zero). */
function makeRaw(
  cap: number, inW: number,
  records: { slot: number; input: number[]; target: number[]; pos: number[] }[],
): { raw: Float32Array; stride: number } {
  const stride = inW + OUT_W + 3;
  const raw = new Float32Array(cap * stride);
  for (const r of records) {
    const base = r.slot * stride;
    raw.set(r.input, base);
    raw.set(r.target, base + inW);
    raw.set(r.pos, base + inW + OUT_W);
  }
  return { raw, stride };
}

describe('unpackRecords — NRC record gap detection + dense repack', () => {
  const inW = 5;
  const cap = 4;

  const recA = { slot: 0, input: [0.5, 1, 2, 3, 4], target: [9, 8, 7], pos: [-1, 2, 3] };
  // slot 1 left empty (gap)
  const recB = { slot: 2, input: [0.25, 0, 0, 0, 0], target: [0, 0, 0], pos: [4, 5, 6] };
  // slot 3: input[0] === 0, later feature non-zero → VALID by the A6 criterion.
  const recC = { slot: 3, input: [0, 9, 9, 9, 9], target: [1, 1, 1], pos: [7, 7, 7] };

  it('repacks non-empty records densely, skipping gaps', () => {
    const { raw, stride } = makeRaw(cap, inW, [recA, recB, recC]);
    const { x, y, pos, filled } = unpackRecords(raw, cap, stride, inW);

    expect(filled).toBe(3);
    // dense sample 0 = slot 0
    expect(Array.from(x.subarray(0, inW))).toEqual(recA.input);
    expect(Array.from(y.subarray(0, OUT_W))).toEqual(recA.target);
    expect(Array.from(pos.subarray(0, 3))).toEqual(recA.pos);
    // dense sample 1 = slot 2 (slot 1 gap skipped); zero target is KEPT (A6)
    expect(Array.from(x.subarray(inW, 2 * inW))).toEqual(recB.input);
    expect(Array.from(y.subarray(OUT_W, 2 * OUT_W))).toEqual(recB.target);
    expect(Array.from(pos.subarray(3, 6))).toEqual(recB.pos);
    // dense sample 2 = slot 3 (input[0] is zero, later features prove it is filled)
    expect(Array.from(x.subarray(2 * inW, 3 * inW))).toEqual(recC.input);
    expect(Array.from(y.subarray(2 * OUT_W, 3 * OUT_W))).toEqual(recC.target);
    expect(Array.from(pos.subarray(6, 9))).toEqual(recC.pos);
  });

  it('keeps records whose first encoded feature is zero but later features are nonzero', () => {
    const { raw, stride } = makeRaw(cap, inW, [recA, recB, recC]);
    const { x, y, pos, filled } = unpackRecords(raw, cap, stride, inW);

    expect(filled).toBe(3);
    expect(Array.from(x.subarray(2 * inW, 3 * inW))).toEqual(recC.input);
    expect(Array.from(y.subarray(2 * OUT_W, 3 * OUT_W))).toEqual(recC.target);
    expect(Array.from(pos.subarray(6, 9))).toEqual(recC.pos);
    for (let s = filled; s < cap; s++) {
      expect(Array.from(x.subarray(s * inW, (s + 1) * inW))).toEqual([0, 0, 0, 0, 0]);
      expect(Array.from(y.subarray(s * OUT_W, (s + 1) * OUT_W))).toEqual([0, 0, 0]);
      expect(Array.from(pos.subarray(s * 3, (s + 1) * 3))).toEqual([0, 0, 0]);
    }
  });

  it('treats only an all-zero encoded input as an empty slot', () => {
    const emptyInputNonzeroTail = {
      slot: 3,
      input: [0, 0, 0, 0, 0],
      target: [1, 1, 1],
      pos: [7, 7, 7],
    };
    const { raw, stride } = makeRaw(cap, inW, [recA, recB, emptyInputNonzeroTail]);
    const { x, y, pos, filled } = unpackRecords(raw, cap, stride, inW);

    expect(filled).toBe(2);
    for (let s = filled; s < cap; s++) {
      expect(Array.from(x.subarray(s * inW, (s + 1) * inW))).toEqual([0, 0, 0, 0, 0]);
      expect(Array.from(y.subarray(s * OUT_W, (s + 1) * OUT_W))).toEqual([0, 0, 0]);
      expect(Array.from(pos.subarray(s * 3, (s + 1) * 3))).toEqual([0, 0, 0]);
    }
  });

  it('returns filled=0 for an all-zero buffer', () => {
    const stride = inW + OUT_W + 3;
    const raw = new Float32Array(cap * stride);
    const { filled, x } = unpackRecords(raw, cap, stride, inW);
    expect(filled).toBe(0);
    expect(x.every((v) => v === 0)).toBe(true);
  });

  it('zeroes and re-uses caller-provided staging arrays (the subsystem path)', () => {
    const { raw, stride } = makeRaw(cap, inW, [recA]);
    const out = {
      x: new Float32Array(cap * inW).fill(123),
      y: new Float32Array(cap * OUT_W).fill(123),
      pos: new Float32Array(cap * 3).fill(123),
    };
    const res = unpackRecords(raw, cap, stride, inW, out);
    expect(res.x).toBe(out.x);
    expect(res.y).toBe(out.y);
    expect(res.pos).toBe(out.pos);
    expect(res.filled).toBe(1);
    expect(Array.from(out.x.subarray(0, inW))).toEqual(recA.input);
    // stale 123s from the previous frame must be gone everywhere
    expect(out.x.subarray(inW).every((v) => v === 0)).toBe(true);
    expect(out.y.subarray(OUT_W).every((v) => v === 0)).toBe(true);
    expect(out.pos.subarray(3).every((v) => v === 0)).toBe(true);
  });
});
