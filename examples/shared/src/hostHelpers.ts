/**
 * Shared host glue for vitrum example apps.
 *
 * These helpers were duplicated across cornell-box, two-engines-one-scene,
 * neural-denoiser, hero-product-viz, and hero-viewer. Hoisted here so a
 * helper bug fix only needs to happen once.
 */

import type { Mat4 } from '@vitrum/core';
import type * as THREE from 'three';

/**
 * Convert a THREE.Matrix4 to vitrum's Mat4 (a typed Float32Array of length 16,
 * column-major to match THREE.js's `.elements`). The two row/col conventions
 * are identical, so this is a `new Float32Array(m.elements)` wrap.
 */
export function mat4FromThree(m: THREE.Matrix4): Mat4 {
  return new Float32Array(m.elements);
}

/**
 * Resize a canvas's backing-store dimensions to match its CSS display size at
 * the current device pixel ratio, capped at 2× (rendering at 3× on a retina
 * MBP is mostly wasted samples). Returns true if the canvas was resized,
 * false if already at the target size — useful for triggering
 * `engine.setSize(...)` only on actual changes.
 */
export function resizeCanvasToDisplaySize(canvas: HTMLCanvasElement): boolean {
  const dpr = Math.min(window.devicePixelRatio, 2);
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width === w && canvas.height === h) return false;
  canvas.width = w;
  canvas.height = h;
  return true;
}

/**
 * Parse a query-string integer with a default. Returns `defaultValue` if the
 * value is missing, not a finite integer, or ≤ 0.
 *
 * Each example had its own slight variant of this (cornell-box uses a
 * factored `parseNumber`; hero-* uses fresh `Number(raw)`); the contract is
 * the same and the drift is harmless but argued for consolidation.
 */
export function parsePositiveInt(raw: string | null | undefined, defaultValue: number): number {
  if (raw == null) return defaultValue;
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0 || !Number.isInteger(n)) return defaultValue;
  return n;
}
