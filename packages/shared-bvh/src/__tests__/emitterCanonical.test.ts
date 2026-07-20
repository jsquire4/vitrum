// T1-2 — unit tests for the backend-neutral emitter canonicalizer.

import { describe, expect, it } from 'vitest';
import type { Scene, SceneEmitter } from '@vitrum/core';
import {
  emitterToCanonical,
  canonicalizeEmitter,
  canonicalMeshAreaIncluded,
} from '../emitterCanonical.js';

function scene(emitters: SceneEmitter[]): Scene {
  return { primitives: [], emitters, environment: { kind: 'none' } } as unknown as Scene;
}

const REC709 = (r: number, g: number, b: number): number =>
  0.2126 * r + 0.7152 * g + 0.0722 * b;

describe('emitterToCanonical', () => {
  it('normalizes the directional toward-light vector as -direction/|direction|', () => {
    const c = canonicalizeEmitter(
      { kind: 'directional', id: 'd', color: [1, 1, 1], intensity: 2, direction: [0, -2, 0] },
      false,
    )!;
    expect(c.kind).toBe('directional');
    expect(c.towardLight![0]).toBeCloseTo(0, 12);
    expect(c.towardLight![1]).toBeCloseTo(1, 12);
    expect(c.towardLight![2]).toBeCloseTo(0, 12);
    expect(c.area).toBe(0);
    expect(c.power).toBeCloseTo(REC709(2, 2, 2), 12);
  });

  it('derives the spot inner/outer cone from angle + penumbra', () => {
    const angle = 0.7;
    const penumbra = 0.25;
    const c = canonicalizeEmitter(
      {
        kind: 'spot',
        id: 's',
        color: [1, 1, 1],
        intensity: 1,
        position: [0, 0, 0],
        direction: [0, -1, 0],
        angle,
        penumbra,
      },
      false,
    )!;
    expect(c.cone!.outerAngle).toBeCloseTo(angle, 12);
    expect(c.cone!.innerAngle).toBeCloseTo(angle * (1 - penumbra), 12);
    expect(c.cone!.cosOuter).toBeCloseTo(Math.cos(angle), 12);
    expect(c.cone!.cosInner).toBeCloseTo(Math.cos(angle * (1 - penumbra)), 12);
  });

  it('computes rect-area power = luminance(radiance)·|u×v|', () => {
    const c = canonicalizeEmitter(
      {
        kind: 'rect-area',
        id: 'r',
        color: [1, 0.5, 0.25],
        intensity: 3,
        position: [0, 0, 0],
        uAxis: [2, 0, 0],
        vAxis: [0, 0, 3],
      },
      false,
    )!;
    const area = 6; // |[2,0,0] x [0,0,3]| = 6
    expect(c.area).toBeCloseTo(area, 12);
    expect(c.power).toBeCloseTo(REC709(3, 1.5, 0.75) * area, 12);
  });

  it('computes disc-area power = luminance(radiance)·π·r²', () => {
    const c = canonicalizeEmitter(
      {
        kind: 'disc-area',
        id: 'disc',
        color: [1, 1, 1],
        intensity: 2,
        position: [0, 0, 0],
        normal: [0, 1, 0],
        radius: 0.5,
      },
      false,
    )!;
    const area = Math.PI * 0.25;
    expect(c.area).toBeCloseTo(area, 12);
    expect(c.power).toBeCloseTo(REC709(2, 2, 2) * area, 12);
  });

  it('resolves SHADOW-01 castShadow:false', () => {
    const c = canonicalizeEmitter(
      { kind: 'point', id: 'p', color: [1, 1, 1], intensity: 1, position: [0, 0, 0], castShadow: false },
      false,
    )!;
    expect(c.shadowDisabled).toBe(true);
  });

  it('excludes mesh-area by default and opts in when requested', () => {
    const emitters: SceneEmitter[] = [
      { kind: 'point', id: 'p', color: [1, 1, 1], intensity: 1, position: [0, 0, 0] },
      { kind: 'mesh-area', id: 'm', color: [1, 1, 1], intensity: 1, meshId: 'x' },
    ];
    const excluded = emitterToCanonical(scene(emitters), false);
    const included = emitterToCanonical(scene(emitters), true);
    expect(excluded).toHaveLength(1);
    expect(included).toHaveLength(2);
    expect(canonicalMeshAreaIncluded(excluded)).toBe(false);
    expect(canonicalMeshAreaIncluded(included)).toBe(true);
  });
});
