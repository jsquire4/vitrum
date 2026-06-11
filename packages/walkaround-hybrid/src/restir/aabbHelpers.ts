/**
 * Mutable AABB helpers for the RestirBvhSnapshot build path.
 *
 * These operate on `{min: {x,y,z}, max: {x,y,z}}` semantics (not the
 * `PlainAabb` array-tuple form from @vitrum/shared-bvh, which has different
 * semantics). Moved from restirBvhSnapshot.ts (D6.11) to be reusable within
 * the restir/ directory without pulling them into the public package API.
 */

export interface MutableRestirAabb {
  min: { x: number; y: number; z: number };
  max: { x: number; y: number; z: number };
}

export function makeEmptyAabb(): MutableRestirAabb {
  return {
    min: { x: Infinity, y: Infinity, z: Infinity },
    max: { x: -Infinity, y: -Infinity, z: -Infinity },
  };
}

export function setAabb(
  out: MutableRestirAabb,
  minX: number,
  minY: number,
  minZ: number,
  maxX: number,
  maxY: number,
  maxZ: number,
): void {
  out.min.x = minX;
  out.min.y = minY;
  out.min.z = minZ;
  out.max.x = maxX;
  out.max.y = maxY;
  out.max.z = maxZ;
}

export function copyBoxLike(
  out: MutableRestirAabb,
  box: { readonly min: { x: number; y: number; z: number }; readonly max: { x: number; y: number; z: number } },
): void {
  setAabb(out, box.min.x, box.min.y, box.min.z, box.max.x, box.max.y, box.max.z);
}

export function isAabbEmpty(
  box: { readonly min: { x: number; y: number; z: number }; readonly max: { x: number; y: number; z: number } },
): boolean {
  return box.max.x < box.min.x || box.max.y < box.min.y || box.max.z < box.min.z;
}
