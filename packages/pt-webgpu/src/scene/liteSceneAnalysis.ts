// Lite-tier scene-analysis free functions.
//
// Extracted verbatim from `index.ts` (T3-B god-file split, 2026-07-20). These are
// pure CPU predicates over the core `Scene`/`ScenePrimitive` contract with no GPU
// or engine-instance dependency — they decide whether the lite kernel can bake a
// primitive's per-vertex colors into a single flat color (the lite packer has no
// per-vertex-color path). Behaviour is byte-identical to the former inline fns;
// only the home changed.

import type { Scene, ScenePrimitive } from '@vitrum/core';

/**
 * True when `primitive`'s vertex colors are uniform enough for the lite tier to
 * bake into a single constant color (or it has no colors). The lite kernel has no
 * per-vertex-color attribute, so a primitive with varying vertex colors cannot be
 * represented faithfully on lite.
 */
export function liteCanBakeVertexColors(primitive: ScenePrimitive): boolean {
  const colors = (primitive as { readonly colors?: Float32Array }).colors;
  const positions = (primitive as { readonly positions?: Float32Array }).positions;
  if (colors == null || colors.length === 0) return true;
  if (positions == null || positions.length === 0) return false;
  const vertexCount = Math.floor(positions.length / 3);
  const stride = colors.length >= vertexCount * 4
    ? 4
    : colors.length >= vertexCount * 3
      ? 3
      : 0;
  if (vertexCount === 0 || stride === 0) return false;
  const r = colors[0] ?? 1;
  const g = colors[1] ?? 1;
  const b = colors[2] ?? 1;
  const eps = 1e-6;
  for (let i = 0; i < vertexCount; i += 1) {
    const o = i * stride;
    if (
      Math.abs((colors[o] ?? 1) - r) > eps ||
      Math.abs((colors[o + 1] ?? 1) - g) > eps ||
      Math.abs((colors[o + 2] ?? 1) - b) > eps
    ) {
      return false;
    }
    if (stride === 4 && Math.abs((colors[o + 3] ?? 1) - 1) > eps) {
      return false;
    }
  }
  return true;
}

/**
 * Sorted list of primitive ids whose vertex colors the lite tier cannot bake
 * (varying colors with no lite per-vertex path). Drives a lite-tier warning.
 */
export function collectLiteUnsupportedVertexColorPrimitiveIds(scene: Scene): string[] {
  const ids: string[] = [];
  for (const primitive of scene.primitives) {
    const colors = (primitive as { readonly colors?: Float32Array }).colors;
    if (colors != null && colors.length > 0 && !liteCanBakeVertexColors(primitive)) ids.push(primitive.id);
  }
  return ids.sort();
}
