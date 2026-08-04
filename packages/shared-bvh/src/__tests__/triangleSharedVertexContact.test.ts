import { describe, expect, it } from 'vitest';
import {
  coplanarTrianglesContactBeyondSharedVertex,
  nonCoplanarTrianglesContactBeyondSharedVertex,
} from '../triangleSharedVertexContact.js';

const ORIGIN = [0, 0, 0] as const;

describe('shared-vertex triangle self-contact', () => {
  it('detects a coplanar incident-edge crossing away from the shared vertex', () => {
    const a = [ORIGIN, [2, 0, 0], [0, 2, 0]] as const;
    // Both nonshared vertices lie outside A. Their connecting edge crosses A,
    // while each triangle also has edges incident to the shared origin.
    const b = [ORIGIN, [1, -1, 0], [1, 3, 0]] as const;
    expect(coplanarTrianglesContactBeyondSharedVertex(a, b, ORIGIN)).toBe(true);
  });

  it('does not turn the one permitted coplanar shared point into self-contact', () => {
    const a = [ORIGIN, [2, 0, 0], [0, 2, 0]] as const;
    const b = [ORIGIN, [-2, 0, 0], [0, -2, 0]] as const;
    expect(coplanarTrianglesContactBeyondSharedVertex(a, b, ORIGIN)).toBe(false);
  });

  it('distinguishes a noncoplanar second contact from the shared point alone', () => {
    const a = [ORIGIN, [2, 0, 0], [0, 2, 0]] as const;
    const crossing = [ORIGIN, [1, 1, 1], [1, 1, -1]] as const;
    const separated = [ORIGIN, [-1, -1, 1], [-1, -1, -1]] as const;
    expect(nonCoplanarTrianglesContactBeyondSharedVertex(
      a, crossing, ORIGIN, Number.EPSILON * 64,
    )).toBe(true);
    expect(nonCoplanarTrianglesContactBeyondSharedVertex(
      a, separated, ORIGIN, Number.EPSILON * 64,
    )).toBe(false);
  });
});
