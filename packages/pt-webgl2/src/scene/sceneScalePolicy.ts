import type { Scene } from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import { classifyAreaVectorF32 } from './areaEmitterGeometry.js';
import {
  WEBGL2_F32_MAX,
  WEBGL2_F32_MIN_NORMAL,
  requireFiniteFloat32,
  requireNonNegativeFloat32,
} from './float32Policy.js';

export { WEBGL2_F32_MIN_NORMAL } from './float32Policy.js';

const CORNELL_ROOT_RADIUS_WITH_MARGIN = Math.sqrt(3) * 1.001;
const TRANSPORT_BOUNDS_MARGIN = 1.001;
const STEP_RAY_ORIGIN_RELATIVE_FACTOR = Math.fround(
  4 * 1.192092896e-7,
);

/**
 * Preserve the historical 1e-4 Cornell-box ray offset as a ratio of the
 * renderer's margin-expanded scene radius.
 */
export const WEBGL2_RAY_BIAS_PER_SCENE_RADIUS =
  1e-4 / CORNELL_ROOT_RADIUS_WITH_MARGIN;

export interface Webgl2TransportBounds {
  readonly center: readonly [number, number, number];
  readonly radius: number;
  /** Exact finite endpoint bounds retained for per-frame camera preflight. */
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
}

export interface Webgl2CoordinateBounds {
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
}

interface BoundsAccumulator {
  minX: number;
  minY: number;
  minZ: number;
  maxX: number;
  maxY: number;
  maxZ: number;
  count: number;
}

function includeTransportPoint(
  bounds: BoundsAccumulator,
  x: number,
  y: number,
  z: number,
  context: string,
): void {
  const qx = requireFiniteFloat32(x, `${context}[0]`);
  const qy = requireFiniteFloat32(y, `${context}[1]`);
  const qz = requireFiniteFloat32(z, `${context}[2]`);
  bounds.minX = Math.min(bounds.minX, qx);
  bounds.minY = Math.min(bounds.minY, qy);
  bounds.minZ = Math.min(bounds.minZ, qz);
  bounds.maxX = Math.max(bounds.maxX, qx);
  bounds.maxY = Math.max(bounds.maxY, qy);
  bounds.maxZ = Math.max(bounds.maxZ, qz);
  bounds.count += 1;
}

function includeTransportRange(
  bounds: BoundsAccumulator,
  center: readonly [number, number, number],
  extent: readonly [number, number, number],
  context: string,
): void {
  for (let axis = 0; axis < 3; axis += 1) {
    const absoluteEndpoint = Math.abs(center[axis]!) + extent[axis]!;
    requireFiniteFloat32(
      absoluteEndpoint,
      `${context} sampled endpoint axis ${axis}`,
    );
  }
  includeTransportPoint(
    bounds,
    center[0] - extent[0],
    center[1] - extent[1],
    center[2] - extent[2],
    `${context} minimum endpoint`,
  );
  includeTransportPoint(
    bounds,
    center[0] + extent[0],
    center[1] + extent[1],
    center[2] + extent[2],
    `${context} maximum endpoint`,
  );
}

function includeAnalyticEmitterBounds(
  bounds: BoundsAccumulator,
  scene: Scene,
): void {
  for (const emitter of scene.emitters) {
    const context = `pt-webgl2 ${emitter.kind} emitter "${String(emitter.id)}"`;
    switch (emitter.kind) {
      case 'rect-area': {
        const center = emitter.position.map((value, axis) =>
          requireFiniteFloat32(value, `${context} position[${axis}]`),
        ) as [number, number, number];
        const fullU = emitter.uAxis.map((value) => Math.fround(2 * value));
        const fullV = emitter.vAxis.map((value) => Math.fround(2 * value));
        const extent = [0, 1, 2].map((axis) =>
          Math.fround(
            Math.fround(Math.abs(fullU[axis]!) * 0.5) +
            Math.fround(Math.abs(fullV[axis]!) * 0.5),
          ),
        ) as [number, number, number];
        includeTransportRange(bounds, center, extent, context);
        break;
      }
      case 'disc-area': {
        const center = emitter.position.map((value, axis) =>
          requireFiniteFloat32(value, `${context} position[${axis}]`),
        ) as [number, number, number];
        const radius = requireNonNegativeFloat32(
          emitter.radius,
          `${context} radius`,
        );
        includeTransportRange(
          bounds,
          center,
          [radius, radius, radius],
          context,
        );
        break;
      }
      case 'point':
      case 'spot':
        includeTransportPoint(
          bounds,
          emitter.position[0],
          emitter.position[1],
          emitter.position[2],
          `${context} position`,
        );
        break;
      case 'directional':
      case 'mesh-area':
        break;
      default: {
        const exhaustive: never = emitter;
        void exhaustive;
      }
    }
  }
}

function hasActiveDistantSource(scene: Scene): boolean {
  if (
    (scene.environment.kind === 'hdri' ||
      scene.environment.kind === 'procedural-sky') &&
    (scene.environment.intensity ?? 1) > 0
  ) {
    return true;
  }
  return scene.emitters.some(
    (emitter) =>
      emitter.kind === 'directional' &&
      emitter.intensity > 0 &&
      emitter.color.some((channel) => channel > 0),
  );
}

function validateBdptDistantLaunch(
  bounds: Webgl2TransportBounds,
  scene: Scene,
): void {
  if (!hasActiveDistantSource(scene)) return;
  const radius = Math.fround(bounds.radius);
  const launchArea = classifyAreaVectorF32(
    [radius, 0, 0],
    [0, radius, 0],
    Math.PI,
  );
  if (!launchArea.valid) {
    throw new RangeError(
      '@vitrum/pt-webgl2: the scene radius cannot represent the BDPT ' +
      `distant-emitter launch disk and reciprocal PDF in float32 (${launchArea.reason}).`,
    );
  }
  const maximumAxisExcursion = Math.SQRT2 * radius;
  requireFiniteFloat32(
    maximumAxisExcursion,
    'pt-webgl2 BDPT distant-emitter launch excursion',
  );
  for (let axis = 0; axis < 3; axis += 1) {
    requireFiniteFloat32(
      Math.abs(bounds.center[axis]!) + maximumAxisExcursion,
      `pt-webgl2 BDPT distant-emitter launch coordinate[${axis}]`,
    );
  }
}

function validateCoordinateBounds(
  bounds: Webgl2CoordinateBounds,
  context: string,
): Webgl2CoordinateBounds {
  const min = [0, 1, 2].map((axis) =>
    requireFiniteFloat32(
      bounds.min[axis]!,
      `${context} minimum[${axis}]`,
    ),
  ) as [number, number, number];
  const max = [0, 1, 2].map((axis) =>
    requireFiniteFloat32(
      bounds.max[axis]!,
      `${context} maximum[${axis}]`,
    ),
  ) as [number, number, number];
  for (let axis = 0; axis < 3; axis += 1) {
    if (min[axis]! > max[axis]!) {
      throw new RangeError(
        `${context} minimum[${axis}] exceeds maximum[${axis}].`,
      );
    }
  }
  return { min, max };
}

/**
 * Prove that every point in an axis-aligned domain has room for the exact
 * coordinate-relative `stepRayOrigin` offset used by the GLSL common helpers:
 *
 *   coordinateStep = max(abs(point)) * (4 * FLT_EPSILON)
 *   step = max(RAY_OFFSET, coordinateStep, FLT_MIN_NORMAL)
 *   result = point + unitOffset * step
 *
 * Surface normals and launch directions are unit vectors, so checking both
 * outward signs with a component magnitude of one is the worst case.
 */
function validateStepRayOriginHeadroom(
  domain: Webgl2CoordinateBounds,
  rayOriginBias: number,
  context: string,
): Webgl2CoordinateBounds {
  const checked = validateCoordinateBounds(domain, context);
  const maximumAbsoluteCoordinate = Math.max(
    ...checked.min.map(Math.abs),
    ...checked.max.map(Math.abs),
  );
  const coordinateStep = requireNonNegativeFloat32(
    Math.fround(
      Math.fround(maximumAbsoluteCoordinate) *
        STEP_RAY_ORIGIN_RELATIVE_FACTOR,
    ),
    `${context} coordinate-relative ray-origin step`,
  );
  const packedBias = requireNonNegativeFloat32(
    rayOriginBias,
    `${context} base ray-origin bias`,
  );
  const step = Math.max(
    packedBias,
    coordinateStep,
    WEBGL2_F32_MIN_NORMAL,
  );

  const expandedMin = [0, 0, 0] as [number, number, number];
  const expandedMax = [0, 0, 0] as [number, number, number];
  for (let axis = 0; axis < 3; axis += 1) {
    const positiveOutward = Math.fround(checked.max[axis]! + step);
    const negativeOutward = Math.fround(checked.min[axis]! - step);
    if (!Number.isFinite(positiveOutward) || !Number.isFinite(negativeOutward)) {
      throw new RangeError(
        `@vitrum/pt-webgl2: ${context} has insufficient absolute-coordinate ` +
        `headroom for stepRayOrigin on axis ${axis}.`,
      );
    }
    expandedMin[axis] = negativeOutward;
    expandedMax[axis] = positiveOutward;
  }
  return { min: expandedMin, max: expandedMax };
}

function validateTraversableSeparation(
  first: Webgl2CoordinateBounds,
  second: Webgl2CoordinateBounds,
  context: string,
): void {
  const maximumSeparation = [0, 1, 2].map((axis) =>
    Math.max(
      Math.abs(first.min[axis]! - second.max[axis]!),
      Math.abs(first.max[axis]! - second.min[axis]!),
    ),
  ) as [number, number, number];
  for (let axis = 0; axis < 3; axis += 1) {
    requireFiniteFloat32(
      maximumSeparation[axis]!,
      `${context} separation[${axis}]`,
    );
  }
  const maximumDistance = requireFiniteFloat32(
    Math.hypot(...maximumSeparation),
    `${context} distance`,
  );
  if (maximumDistance >= WEBGL2_F32_MAX) {
    throw new RangeError(
      `@vitrum/pt-webgl2: the ${context} distance collides with the ` +
      'max-float distance sentinel.',
    );
  }
}

/**
 * Validate the complete finite domain joining all primary-ray origins to every
 * retained geometry/analytic-light endpoint. This mirrors the binary32
 * subtraction boundary used by BVH slabs, reserves the shader's max-float
 * distance sentinel, and proves secondary-ray origin offsets cannot overflow.
 */
export function validateWebgl2CameraTransportDomain(
  cameraOrigins: Webgl2CoordinateBounds,
  transport: Webgl2TransportBounds,
): void {
  const origins = validateCoordinateBounds(
    cameraOrigins,
    'pt-webgl2 primary-ray origin domain',
  );
  const finiteTransport = validateCoordinateBounds(
    transport,
    'pt-webgl2 retained transport domain',
  );
  const packedBias = Math.fround(webgl2RayOriginBias(transport.radius));
  const expandedOrigins = validateStepRayOriginHeadroom(
    origins,
    packedBias,
    'primary-ray origin domain',
  );
  const expandedTransport = validateStepRayOriginHeadroom(
    finiteTransport,
    packedBias,
    'retained transport domain',
  );
  validateTraversableSeparation(
    expandedOrigins,
    expandedTransport,
    'camera-to-transport',
  );
}

/**
 * Compute the common finite transport domain consumed by ray-distance,
 * finite-light, and BDPT launch arithmetic. Geometry and analytic emitters are
 * included together so individually representable positions cannot overflow
 * their shader subtraction or sampled endpoint construction.
 *
 * Call this during CPU preflight, before allocating or mutating GPU resources.
 */
export function computeWebgl2TransportBounds(
  pack: WorldSpaceMergeResult,
  scene: Scene,
  options: { readonly bdpt?: boolean } = {},
): Webgl2TransportBounds {
  const bounds: BoundsAccumulator = {
    minX: Number.POSITIVE_INFINITY,
    minY: Number.POSITIVE_INFINITY,
    minZ: Number.POSITIVE_INFINITY,
    maxX: Number.NEGATIVE_INFINITY,
    maxY: Number.NEGATIVE_INFINITY,
    maxZ: Number.NEGATIVE_INFINITY,
    count: 0,
  };
  const positions = pack.positions;
  const stride = pack.positionStrideFloats;
  for (let i = 0; i + 2 < positions.length; i += stride) {
    includeTransportPoint(
      bounds,
      positions[i]!,
      positions[i + 1]!,
      positions[i + 2]!,
      `pt-webgl2 merged position ${Math.floor(i / stride)}`,
    );
  }
  includeAnalyticEmitterBounds(bounds, scene);

  if (bounds.count === 0) {
    const empty = {
      center: [0, 0, 0],
      radius: 1,
      min: [0, 0, 0],
      max: [0, 0, 0],
    } as const;
    const expandedEmpty = validateStepRayOriginHeadroom(
      empty,
      Math.fround(webgl2RayOriginBias(empty.radius)),
      'finite transport domain',
    );
    validateTraversableSeparation(
      expandedEmpty,
      expandedEmpty,
      'finite transport-domain',
    );
    if (options.bdpt === true) validateBdptDistantLaunch(empty, scene);
    return empty;
  }

  const spanX = bounds.maxX - bounds.minX;
  const spanY = bounds.maxY - bounds.minY;
  const spanZ = bounds.maxZ - bounds.minZ;
  requireFiniteFloat32(spanX, 'pt-webgl2 transport span[0]');
  requireFiniteFloat32(spanY, 'pt-webgl2 transport span[1]');
  requireFiniteFloat32(spanZ, 'pt-webgl2 transport span[2]');
  const diagonal = Math.hypot(spanX, spanY, spanZ);
  requireFiniteFloat32(diagonal, 'pt-webgl2 transport bounding-box diagonal');

  const center: readonly [number, number, number] = [
    requireFiniteFloat32(
      bounds.minX * 0.5 + bounds.maxX * 0.5,
      'pt-webgl2 transport center[0]',
    ),
    requireFiniteFloat32(
      bounds.minY * 0.5 + bounds.maxY * 0.5,
      'pt-webgl2 transport center[1]',
    ),
    requireFiniteFloat32(
      bounds.minZ * 0.5 + bounds.maxZ * 0.5,
      'pt-webgl2 transport center[2]',
    ),
  ];
  const rawRadius = diagonal * 0.5 * TRANSPORT_BOUNDS_MARGIN;
  const radius = resolveWebgl2SceneRadius(center, rawRadius);
  const guardedTransportDiameter = requireFiniteFloat32(
    radius * 2,
    'pt-webgl2 guarded transport diameter',
  );
  if (guardedTransportDiameter >= WEBGL2_F32_MAX) {
    throw new RangeError(
      '@vitrum/pt-webgl2: the finite transport domain collides with the ' +
      'max-float distance sentinel.',
    );
  }
  if (rawRadius > 0 && !(webgl2RayOriginBias(radius) < radius)) {
    throw new RangeError(
      '@vitrum/pt-webgl2: the scene extent is smaller than its minimum ' +
      'representable ray-origin offset.',
    );
  }
  const result = {
    center,
    radius,
    min: [bounds.minX, bounds.minY, bounds.minZ],
    max: [bounds.maxX, bounds.maxY, bounds.maxZ],
  } as const;
  const expandedTransport = validateStepRayOriginHeadroom(
    result,
    Math.fround(webgl2RayOriginBias(radius)),
    'finite transport domain',
  );
  validateTraversableSeparation(
    expandedTransport,
    expandedTransport,
    'finite transport-domain',
  );
  if (options.bdpt === true) validateBdptDistantLaunch(result, scene);
  return result;
}

/** Preserve finite positive radii; only a degenerate root receives a fallback. */
export function resolveWebgl2SceneRadius(
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
      'pt-webgl2 scene scale requires a finite center and non-negative radius.',
    );
  }
  for (let axis = 0; axis < 3; axis += 1) {
    requireFiniteFloat32(
      center[axis]!,
      `pt-webgl2 scene center[${axis}]`,
    );
  }
  if (rawRadius > 0) {
    requireNonNegativeFloat32(rawRadius, 'pt-webgl2 scene radius');
    return rawRadius;
  }
  const coordinateScale = Math.max(
    Math.abs(center[0]),
    Math.abs(center[1]),
    Math.abs(center[2]),
  );
  const fallback = Math.max(
    coordinateScale * 2 ** -20,
    WEBGL2_F32_MIN_NORMAL,
  );
  requireNonNegativeFloat32(fallback, 'pt-webgl2 fallback scene radius');
  return fallback;
}

/** Resolve the scene-relative base offset uploaded to every trace program. */
export function webgl2RayOriginBias(sceneRadius: number): number {
  if (!Number.isFinite(sceneRadius) || !(sceneRadius > 0)) {
    throw new RangeError(
      'pt-webgl2 ray bias requires a finite positive scene radius.',
    );
  }
  requireNonNegativeFloat32(sceneRadius, 'pt-webgl2 ray-bias scene radius');
  const bias = Math.max(
    sceneRadius * WEBGL2_RAY_BIAS_PER_SCENE_RADIUS,
    WEBGL2_F32_MIN_NORMAL,
  );
  requireNonNegativeFloat32(bias, 'pt-webgl2 ray origin bias');
  return bias;
}
