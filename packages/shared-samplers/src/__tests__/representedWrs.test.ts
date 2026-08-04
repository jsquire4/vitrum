import { describe, expect, it } from 'vitest';
import {
  REPRESENTED_WRS_BUCKET_COUNT,
  createRepresentedWrsStateF32,
  representedWrsLogSelectionProbability,
  representedWrsLogWeightSumF32,
  representedWrsSelectedLogCorrection,
  updateRepresentedWrsF32,
} from '../representedWrs.js';
import { REPRESENTED_WRS_WGSL } from '../wgsl/representedWrs.wgsl.js';

const ticketRandom = (ticket: number): (() => number) =>
  () => ticket / REPRESENTED_WRS_BUCKET_COUNT;

const logBucketProbability = (count: number): number =>
  Math.fround(Math.log2(Math.fround(count / REPRESENTED_WRS_BUCKET_COUNT)));

describe('represented finite-RNG weighted reservoir sampling', () => {
  it('selects the first finite candidate without consuming WRS RNG', () => {
    const state = createRepresentedWrsStateF32();
    let rngCalls = 0;
    const selected = updateRepresentedWrsF32(state, -7.25, () => {
      rngCalls += 1;
      return 0;
    });

    expect(selected).toBe(true);
    expect(rngCalls).toBe(0);
    expect(state).toEqual({
      maxLogWeight: -7.25,
      scaledWeightSum: 1,
      selectedLogWeight: -7.25,
      logSelectionProbability: 0,
      logSelectionProbabilityLow: 0,
      hasSelection: true,
    });
  });

  it('gives a roughly 2^-30 newcomer one bucket and records both exact branches', () => {
    const replace = createRepresentedWrsStateF32();
    expect(updateRepresentedWrsF32(replace, 0, ticketRandom(0))).toBe(true);
    expect(updateRepresentedWrsF32(replace, -30, ticketRandom(0))).toBe(true);
    expect(replace.selectedLogWeight).toBe(-30);
    expect(replace.logSelectionProbability).toBe(-24);

    const keep = createRepresentedWrsStateF32();
    expect(updateRepresentedWrsF32(keep, 0, ticketRandom(0))).toBe(true);
    expect(updateRepresentedWrsF32(keep, -30, ticketRandom(1))).toBe(false);
    expect(keep.selectedLogWeight).toBe(0);
    expect(keep.logSelectionProbability).toBe(
      logBucketProbability(REPRESENTED_WRS_BUCKET_COUNT - 1),
    );
    expect(keep.logSelectionProbability).toBeLessThan(0);
  });

  it('caps a huge newcomer at B - 1 so the final ticket still keeps', () => {
    const state = createRepresentedWrsStateF32();
    expect(updateRepresentedWrsF32(state, 0, ticketRandom(0))).toBe(true);
    expect(
      updateRepresentedWrsF32(
        state,
        200,
        ticketRandom(REPRESENTED_WRS_BUCKET_COUNT - 1),
      ),
    ).toBe(false);
    expect(state.selectedLogWeight).toBe(0);
    expect(state.logSelectionProbability).toBe(-24);
  });

  it('retains repeated one-bucket keep factors in the double-single residual', () => {
    const state = createRepresentedWrsStateF32();
    expect(updateRepresentedWrsF32(state, 0, ticketRandom(0))).toBe(true);
    expect(updateRepresentedWrsF32(state, -30, ticketRandom(0))).toBe(true);

    const keepCount = 4096;
    for (let i = 0; i < keepCount; i += 1) {
      expect(updateRepresentedWrsF32(state, -30, ticketRandom(1))).toBe(false);
    }

    const exactBinary64 =
      -24 + keepCount * Math.log2((REPRESENTED_WRS_BUCKET_COUNT - 1) / REPRESENTED_WRS_BUCKET_COUNT);
    const represented = representedWrsLogSelectionProbability(state);
    expect(state.logSelectionProbabilityLow).not.toBe(0);
    expect(Math.abs(represented - exactBinary64)).toBeLessThan(5e-10);
    expect(
      Math.abs(
        representedWrsSelectedLogCorrection(state) -
          (state.selectedLogWeight - exactBinary64),
      ),
    ).toBeLessThan(5e-10);

    let ordinaryF32 = Math.fround(-24);
    const keepLogF32 = logBucketProbability(REPRESENTED_WRS_BUCKET_COUNT - 1);
    for (let i = 0; i < keepCount; i += 1) {
      ordinaryF32 = Math.fround(ordinaryF32 + keepLogF32);
    }
    expect(Math.abs(represented - exactBinary64)).toBeLessThan(
      Math.abs(ordinaryF32 - exactBinary64),
    );
  });

  it('resets occurrence probability on a late replacement, then accumulates keeps', () => {
    const state = createRepresentedWrsStateF32();
    expect(updateRepresentedWrsF32(state, 0, ticketRandom(0))).toBe(true);
    expect(updateRepresentedWrsF32(state, -1, ticketRandom(0))).toBe(true);
    const replacementLogProbability = state.logSelectionProbability;
    expect(state.selectedLogWeight).toBe(-1);
    expect(replacementLogProbability).toBeLessThan(0);

    expect(
      updateRepresentedWrsF32(
        state,
        -2,
        ticketRandom(REPRESENTED_WRS_BUCKET_COUNT - 1),
      ),
    ).toBe(false);
    expect(state.selectedLogWeight).toBe(-1);
    expect(state.logSelectionProbability).toBeLessThan(replacementLogProbability);
    expect(representedWrsLogWeightSumF32(state)).toBe(
      Math.fround(Math.log2(Math.fround(1.75))),
    );
    expect(representedWrsSelectedLogCorrection(state)).toBe(
      state.selectedLogWeight -
        state.logSelectionProbability -
        state.logSelectionProbabilityLow,
    );
  });

  it('ignores non-finite candidate logs without state changes or RNG draws', () => {
    const empty = createRepresentedWrsStateF32();
    let rngCalls = 0;
    const rng = (): number => {
      rngCalls += 1;
      return 0;
    };
    for (const value of [
      Number.NaN,
      Number.POSITIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
      -Math.fround(3.4028234663852886e38),
      1e100,
    ]) {
      expect(updateRepresentedWrsF32(empty, value, rng)).toBe(false);
    }
    expect(empty).toEqual(createRepresentedWrsStateF32());
    expect(rngCalls).toBe(0);
    expect(representedWrsLogWeightSumF32(empty)).toBe(Number.NEGATIVE_INFINITY);
    expect(representedWrsLogSelectionProbability(empty)).toBe(Number.NEGATIVE_INFINITY);
    expect(representedWrsSelectedLogCorrection(empty)).toBe(Number.NEGATIVE_INFINITY);

    expect(updateRepresentedWrsF32(empty, 0, rng)).toBe(true);
    const beforeInvalid = { ...empty };
    expect(updateRepresentedWrsF32(empty, Number.NaN, rng)).toBe(false);
    expect(empty).toEqual(beforeInvalid);
    expect(rngCalls).toBe(0);
  });

  it('rejects invalid later RNG values before mutating live state', () => {
    const state = createRepresentedWrsStateF32();
    expect(updateRepresentedWrsF32(state, 0, ticketRandom(0))).toBe(true);
    for (const value of [-1, 1, Number.NaN, Number.POSITIVE_INFINITY]) {
      const before = { ...state };
      expect(() => updateRepresentedWrsF32(state, -1, () => value)).toThrow(/finite value in/);
      expect(state).toEqual(before);
    }
  });

  it('rejects externally corrupted compensated live state', () => {
    const state = createRepresentedWrsStateF32();
    expect(updateRepresentedWrsF32(state, 0, ticketRandom(0))).toBe(true);
    state.logSelectionProbabilityLow = 1;
    expect(() => updateRepresentedWrsF32(state, -1, ticketRandom(0))).toThrow(/live Float32/);
  });

  it('publishes the same 24-bit state machine and integer-ticket branch in WGSL', () => {
    expect(REPRESENTED_WRS_WGSL).toContain('const REPRESENTED_WRS_BUCKET_BITS: u32 = 24u;');
    expect(REPRESENTED_WRS_WGSL).toContain(
      'const REPRESENTED_WRS_BUCKET_COUNT: u32 = 16777216u;',
    );
    for (const field of [
      'maxLogWeight: f32',
      'scaledWeightSum: f32',
      'selectedLogWeight: f32',
      'logSelectionProbability: f32',
      'logSelectionProbabilityLow: f32',
      'hasSelection: bool',
    ]) {
      expect(REPRESENTED_WRS_WGSL).toContain(field);
    }
    expect(REPRESENTED_WRS_WGSL).toContain('u32(ceil(f32(REPRESENTED_WRS_BUCKET_COUNT)');
    expect(REPRESENTED_WRS_WGSL).toContain('REPRESENTED_WRS_BUCKET_COUNT - 1u');
    expect(REPRESENTED_WRS_WGSL).toContain('let ticket = pcgNext(rng) >> 8u;');
    expect(REPRESENTED_WRS_WGSL).toContain(
      'f32(bucketCount) / f32(REPRESENTED_WRS_BUCKET_COUNT)',
    );
    expect(REPRESENTED_WRS_WGSL).toContain(
      'fn representedWrsAddLogTerm(hi: f32, lo: f32, term: f32) -> vec2f',
    );
    expect(REPRESENTED_WRS_WGSL).toContain(
      '(*wrs).logSelectionProbabilityLow = nextLogProbability.y;',
    );
    expect(REPRESENTED_WRS_WGSL).toContain('candidateLogWeight > -3.402823466e38');
    expect(REPRESENTED_WRS_WGSL).not.toContain('rand_f32(rng)');
  });
});
