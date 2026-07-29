/** CPU oracle for the DI temporal surface-correspondence gate. */

export type DiCorrespondenceVec3 = readonly [number, number, number];

export interface DiCorrespondenceSurface {
  readonly hit: boolean;
  readonly position: DiCorrespondenceVec3;
  readonly normal: DiCorrespondenceVec3;
  readonly depth: number;
  readonly triangleId: number;
  readonly instanceId: number;
  readonly materialKey: number;
}

function distance(a: DiCorrespondenceVec3, b: DiCorrespondenceVec3): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function dot(a: DiCorrespondenceVec3, b: DiCorrespondenceVec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function diTemporalSurfaceCorresponds(
  current: DiCorrespondenceSurface,
  previousPixelRecastNow: DiCorrespondenceSurface,
  previousRayOrigin: DiCorrespondenceVec3,
  sceneDepthToleranceFloor: number,
): boolean {
  if (!current.hit || !previousPixelRecastNow.hit) return false;
  if (
    current.instanceId !== previousPixelRecastNow.instanceId ||
    current.triangleId !== previousPixelRecastNow.triangleId ||
    current.materialKey !== previousPixelRecastNow.materialKey
  ) {
    return false;
  }
  const expectedPreviousDepth = distance(current.position, previousRayOrigin);
  const depthDifference = Math.abs(
    previousPixelRecastNow.depth - expectedPreviousDepth,
  );
  const worldDifference = distance(
    previousPixelRecastNow.position,
    current.position,
  );
  const depthTolerance = Math.max(
    sceneDepthToleranceFloor * 4,
    expectedPreviousDepth * 0.02,
  );
  const worldTolerance = Math.max(
    sceneDepthToleranceFloor * 8,
    expectedPreviousDepth * 0.02,
  );
  return (
    Number.isFinite(expectedPreviousDepth) &&
    Number.isFinite(depthDifference) &&
    Number.isFinite(worldDifference) &&
    depthDifference <= depthTolerance &&
    worldDifference <= worldTolerance &&
    dot(previousPixelRecastNow.normal, current.normal) >= 0.9
  );
}
