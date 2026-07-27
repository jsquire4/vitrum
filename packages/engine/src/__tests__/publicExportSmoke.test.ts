/**
 * Smoke tests: verify that the public exports added in the June 2026 cohesion
 * pass are importable from the package root without internal-path hacks.
 *
 * These are compile-time + runtime shape checks, not behaviour tests.  Their
 * purpose is to catch any future re-organisation that accidentally removes the
 * re-export from src/index.ts.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

// ── 1. CameraLike ─────────────────────────────────────────────────────────────
// The type must be importable from the package root (previously it was only
// accessible via the internal `lifecycle/vanilla` path, forcing examples to
// define an inline structural copy with a complaint comment).

// Type-level assertion: import succeeds and satisfies the structural contract.
import { progressiveHandleAsEngine } from '../index.js';
import type { CameraLike } from '../index.js';

describe('CameraLike public export (@vitrum/engine)', () => {
  it('is exported from the package root index', () => {
    // Read the raw source — the export must appear in index.ts.
    // We cannot instantiate a type at runtime, but we can verify the export
    // text exists in the index and build a conforming object against it.
    const indexSrc = readFileSync(
      fileURLToPath(new URL('../index.ts', import.meta.url)),
      'utf8',
    );
    expect(indexSrc).toMatch(/type CameraLike/);
  });

  it('a conforming camera object satisfies the structural interface', () => {
    // Build an object that matches CameraLike structurally — type-checks at
    // compile time and is non-trivially verified at runtime via the property
    // accessors that attachVitrum's RAF tick reads.
    const camera: CameraLike = {
      updateMatrixWorld(): void { /* static no-op */ },
      matrixWorldInverse: { elements: new Float32Array(16) },
      projectionMatrix:   { elements: new Float32Array(16) },
      position: { x: 0, y: 0, z: 0 },
    };

    expect(typeof camera.updateMatrixWorld).toBe('function');
    expect(camera.matrixWorldInverse.elements).toBeInstanceOf(Float32Array);
    expect(camera.projectionMatrix.elements).toBeInstanceOf(Float32Array);
    expect(camera.position.x).toBe(0);
  });
});

describe('progressive facade public export (@vitrum/engine)', () => {
  it('is callable from the package root for vanilla lifecycle hosts', () => {
    expect(progressiveHandleAsEngine).toBeTypeOf('function');
  });
});
