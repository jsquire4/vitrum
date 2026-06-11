/**
 * Legacy THREE-shaped adapter helpers, moved out of `HybridEngineLifecycle.ts`
 * (2026-06-10 dead-code sweep G0.4). Zero production callers — kept only for
 * hosts/tests that still hold a THREE-style scene graph and want to derive
 * `DDGILight`s from it. Structural and Three-free (duck-typed, no `three` import).
 */

import type { DDGILight } from '../ddgi/types.js';

interface ThreeObjectLike {
  readonly type?: string;
  readonly isPointLight?: boolean;
  readonly isRectAreaLight?: boolean;
  readonly position?: { readonly x: number; readonly y: number; readonly z: number };
  readonly color?: { readonly r: number; readonly g: number; readonly b: number };
  readonly intensity?: number;
  readonly width?: number;
  readonly height?: number;
  readonly matrixWorld?: { readonly elements: ArrayLike<number> };
  updateMatrixWorld?: (force?: boolean) => void;
  traverseVisible: (cb: (obj: ThreeObjectLike) => void) => void;
}

function matrixPosition(m: ArrayLike<number> | undefined): { x: number; y: number; z: number } {
  return { x: m?.[12] ?? 0, y: m?.[13] ?? 0, z: m?.[14] ?? 0 };
}

function isPointLightLike(obj: ThreeObjectLike): boolean {
  return obj.isPointLight === true || obj.type === 'PointLight';
}

function isRectAreaLightLike(obj: ThreeObjectLike): boolean {
  return obj.isRectAreaLight === true || obj.type === 'RectAreaLight';
}

/** Compatibility helper for tests/legacy adapters; structural and Three-free. */
export function collectDDGILightsFromThreeRoot(root: ThreeObjectLike): DDGILight[] {
  const out: DDGILight[] = [];
  root.updateMatrixWorld?.(true);
  root.traverseVisible((obj) => {
    if (isPointLightLike(obj) && obj.position != null && obj.color != null) {
      out.push({
        kind: 'fixture',
        intensity: obj.intensity ?? 1,
        on: true,
        position: { x: obj.position.x, y: obj.position.y, z: obj.position.z },
        color: { r: obj.color.r, g: obj.color.g, b: obj.color.b },
      });
      return;
    }
    if (isRectAreaLightLike(obj) && obj.color != null) {
      const width = obj.width ?? 0;
      const height = obj.height ?? 0;
      const p = matrixPosition(obj.matrixWorld?.elements);
      out.push({
        kind: 'fixture',
        intensity: (obj.intensity ?? 1) * width * height,
        on: true,
        position: p,
      });
    }
  });
  return out;
}
