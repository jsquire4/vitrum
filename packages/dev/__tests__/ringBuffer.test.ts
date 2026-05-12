/**
 * ringBuffer.test.ts — Unit tests for RingBuffer (the moving-average engine
 * in FrameTimeHUD).
 *
 * RingBuffer is the only non-trivial algorithmic piece in @vitrum/dev that can
 * be tested purely in Node (no DOM, no GPU, no React). Test it exhaustively.
 */

import { describe, it, expect } from 'vitest';
import { RingBuffer } from '../src/react/FrameTimeHUD.js';

describe('RingBuffer', () => {
  it('mean() on empty buffer returns 0', () => {
    const rb = new RingBuffer(10);
    expect(rb.mean()).toBe(0);
    expect(rb.filled).toBe(0);
  });

  it('mean() of a single entry equals that entry', () => {
    const rb = new RingBuffer(10);
    rb.push(42);
    expect(rb.mean()).toBeCloseTo(42, 10);
    expect(rb.filled).toBe(1);
  });

  it('mean() of capacity-many entries is the arithmetic mean', () => {
    const rb = new RingBuffer(4);
    rb.push(10);
    rb.push(20);
    rb.push(30);
    rb.push(40);
    expect(rb.mean()).toBeCloseTo(25, 10);
    expect(rb.filled).toBe(4);
  });

  it('evicts the oldest entry once capacity is exceeded', () => {
    const rb = new RingBuffer(3);
    rb.push(100); // will be evicted
    rb.push(10);
    rb.push(20);
    rb.push(30);
    // mean should be (10 + 20 + 30) / 3 = 20, not including 100
    expect(rb.mean()).toBeCloseTo(20, 10);
    expect(rb.filled).toBe(3);
  });

  it('running mean updates correctly across many pushes', () => {
    const cap = 60;
    const rb = new RingBuffer(cap);
    // Push 0..119: ring wraps twice
    for (let i = 0; i < 120; i++) rb.push(i);
    // Window now contains [60..119]; mean = (60 + 119) / 2 = 89.5
    expect(rb.mean()).toBeCloseTo(89.5, 8);
    expect(rb.filled).toBe(cap);
  });

  it('mean() remains stable when all entries are the same value', () => {
    const rb = new RingBuffer(5);
    for (let i = 0; i < 200; i++) rb.push(16.67);
    expect(rb.mean()).toBeCloseTo(16.67, 5);
  });

  it('capacity of 1 always returns the last pushed value', () => {
    const rb = new RingBuffer(1);
    rb.push(5);
    rb.push(99);
    expect(rb.mean()).toBeCloseTo(99, 10);
  });

  it('filled never exceeds capacity', () => {
    const rb = new RingBuffer(3);
    for (let i = 0; i < 100; i++) {
      rb.push(i);
      expect(rb.filled).toBeLessThanOrEqual(3);
    }
  });

  it('handles fractional values without precision loss beyond Float64', () => {
    const rb = new RingBuffer(3);
    rb.push(0.1);
    rb.push(0.2);
    rb.push(0.3);
    // (0.1 + 0.2 + 0.3) / 3 = 0.2; Float64 is precise enough
    expect(rb.mean()).toBeCloseTo(0.2, 12);
  });
});
