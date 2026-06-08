/** Plain world-space axis-aligned bounding box. */
export interface PlainAabb {
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
}

export function clonePlainAabb(aabb: PlainAabb): PlainAabb {
  return {
    min: [aabb.min[0], aabb.min[1], aabb.min[2]],
    max: [aabb.max[0], aabb.max[1], aabb.max[2]],
  };
}
