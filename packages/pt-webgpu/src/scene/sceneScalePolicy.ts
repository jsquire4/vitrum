/** Smallest positive normal IEEE-754 binary32 value. */
export const F32_MIN_NORMAL = 1.1754943508222875e-38;

/** Canonical ±1 Cornell-box bounding-sphere radius. */
export const PT_WEBGPU_CORNELL_SCENE_RADIUS = Math.sqrt(3);

/**
 * The historical 1 mm secondary-ray offset expressed as a fraction of the
 * canonical Cornell radius. This preserves the established Cornell result while
 * making the policy proportional to the actual packed scene.
 */
export const PT_WEBGPU_RAY_BIAS_PER_SCENE_RADIUS =
  1e-3 / PT_WEBGPU_CORNELL_SCENE_RADIUS;

function assertPositiveFiniteF32(value: number, label: string): void {
  const packed = Math.fround(value);
  if (!(packed > 0) || !Number.isFinite(packed)) {
    throw new RangeError(
      `pt-webgpu ${label} must remain positive and finite after f32 packing.`,
    );
  }
}

/**
 * Preserve every finite positive packed-scene radius. A truly degenerate root
 * receives only a coordinate-relative f32-safe radius; no world-unit floor is
 * imposed on valid tiny scenes.
 */
export function resolvePtWebgpuSceneRadius(
  center: readonly [number, number, number],
  rawRadius: number,
): number {
  if (
    center.length !== 3 ||
    !center.every(Number.isFinite) ||
    !Number.isFinite(rawRadius) ||
    rawRadius < 0
  ) {
    throw new RangeError(
      'pt-webgpu scene scale requires a finite center and non-negative radius.',
    );
  }
  for (const coordinate of center) {
    if (!Number.isFinite(Math.fround(coordinate))) {
      throw new RangeError(
        'pt-webgpu scene center must remain finite after f32 packing.',
      );
    }
  }
  if (rawRadius > 0) {
    assertPositiveFiniteF32(rawRadius, 'scene radius');
    // Both values are uploaded as f32. Reject a nominally finite JavaScript
    // scale whose derived traversal threshold would silently round to zero.
    assertPositiveFiniteF32(
      rawRadius * PT_WEBGPU_RAY_BIAS_PER_SCENE_RADIUS * 0.1,
      'scene-relative traversal threshold',
    );
    return rawRadius;
  }
  const coordinateScale = Math.max(
    Math.abs(center[0]),
    Math.abs(center[1]),
    Math.abs(center[2]),
  );
  const fallback = Math.max(coordinateScale * 2 ** -20, F32_MIN_NORMAL);
  assertPositiveFiniteF32(fallback, 'degenerate-scene radius');
  return fallback;
}

function f32UlpAtMagnitude(value: number): number {
  const magnitude = Math.abs(Math.fround(value));
  if (magnitude === 0 || magnitude < 2 ** -126) return 2 ** -149;
  return 2 ** (Math.floor(Math.log2(magnitude)) - 23);
}

/**
 * Resolve the secondary-ray origin bias uploaded to FrameParams. The ordinary
 * term follows scene extent; the coordinate term guarantees that applying the
 * scalar along a unit normal changes at least one component of a translated
 * f32 hit point instead of rounding straight back to the surface.
 */
export function ptWebgpuRayOriginBias(
  sceneRadius: number,
  sceneCenter: readonly [number, number, number] = [0, 0, 0],
): number {
  if (!Number.isFinite(sceneRadius) || !(sceneRadius > 0)) {
    throw new RangeError(
      'pt-webgpu ray bias requires a finite positive scene radius.',
    );
  }
  if (
    sceneCenter.length !== 3 ||
    !sceneCenter.every((coordinate) =>
      Number.isFinite(coordinate) &&
      Number.isFinite(Math.fround(coordinate)))
  ) {
    throw new RangeError(
      'pt-webgpu ray bias requires a finite f32-representable scene center.',
    );
  }
  const coordinateMagnitude = Math.max(
    Math.abs(sceneCenter[0]),
    Math.abs(sceneCenter[1]),
    Math.abs(sceneCenter[2]),
  );
  // Every unit vector has a component >= 1/sqrt(3). Two ULPs at the largest
  // coordinate therefore move at least its dominant normal component by more
  // than one ULP, including round-to-nearest ties.
  const coordinateBias = 2 * f32UlpAtMagnitude(coordinateMagnitude);
  const result = Math.max(
    sceneRadius * PT_WEBGPU_RAY_BIAS_PER_SCENE_RADIUS,
    coordinateBias,
  );
  assertPositiveFiniteF32(result, 'ray origin bias');
  return result;
}

/**
 * Closest-hit traversal t-min derived from the same scene-relative policy as
 * the secondary-ray origin offset. Keeping the two related prevents a fixed
 * metre-scale triangle threshold from rejecting tiny scenes or leaking through
 * very large ones.
 */
export function ptWebgpuRayTMin(sceneRadius: number): number {
  const result = ptWebgpuRayOriginBias(sceneRadius) * 0.1;
  assertPositiveFiniteF32(result, 'ray traversal threshold');
  return result;
}

/**
 * Infinite directional/environment roots are sampled on a world-area disk.
 * Their area density and reciprocal must both be representable in the current
 * f32 BDPT/SPPM payload. Reject the estimator/scene combination explicitly
 * instead of silently publishing an invalid light root.
 */
export function assertPtWebgpuDistantLaunchDiskRepresentable(
  sceneRadius: number,
): void {
  const radius = Math.fround(sceneRadius);
  assertPositiveFiniteF32(radius, 'distant-emitter launch radius');
  const area = Math.fround(
    Math.fround(Math.fround(Math.PI) * radius) * radius,
  );
  const inverseArea = Math.fround(1 / area);
  if (
    !(area > 0) ||
    !Number.isFinite(area) ||
    !(inverseArea > 0) ||
    !Number.isFinite(inverseArea)
  ) {
    throw new RangeError(
      'pt-webgpu distant directional/environment launch-disk area and its ' +
        'reciprocal must both be representable as positive finite f32 values.',
    );
  }
}
