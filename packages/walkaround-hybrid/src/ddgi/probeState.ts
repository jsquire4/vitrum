/**
 * DDGI probe relocation/classification state and the CPU reference algorithm.
 *
 * One otherwise-unused rgba16float texel in each probe's irradiance-atlas
 * ring is reserved for relocation/classification state:
 *   xyz = bounded relocation offset normalized by grid spacing
 *   w   = 1 for active, 0 for inactive
 *
 * The algorithm is an original, WebGPU-bounded adaptation of the probe
 * relocation and classification passes described by Majercik et al.,
 * "Scaling Probe-Based Real-Time Dynamic Global Illumination for Production"
 * (JCGT 2021), and NVIDIA RTXGI's reference
 * `ProbeRelocationCS.hlsl` / `ProbeClassificationCS.hlsl`:
 * https://github.com/NVIDIAGameWorks/RTXGI-DDGI/tree/main/rtxgi-sdk/shaders/ddgi
 *
 * RTXGI source is referenced for provenance only; this implementation uses
 * Vitrum's own ray record, round-robin schedule, scalar grid spacing, and
 * conservative inactive-probe policy.
 */

import {
  float16BitsToFloat32,
  float32ToFloat16Bits,
} from '@vitrum/shared-denoisers';
import {
  IRR_PROBE_STATE_LOCAL_X,
  IRR_PROBE_STATE_LOCAL_Y,
  IRR_STRIDE,
} from './ddgiAtlasLayout.js';

/** Explicit snapshot representation: four Float32 lanes per probe. */
export const DDGI_EXPLICIT_PROBE_STATE_BYTES = 16;
/** In-atlas representation: one rgba16float texel per probe. */
export const DDGI_PACKED_PROBE_STATE_BYTES = 8;
export const DDGI_PROBE_STATE_ACTIVE = 1;
export const DDGI_PROBE_STATE_INACTIVE = 0;
export const DDGI_PROBE_BACKFACE_THRESHOLD = 0.25;
export const DDGI_PROBE_MAX_OFFSET_NORMALIZED = 0.45;
export const DDGI_PROBE_MAX_RELOCATION_STEP_NORMALIZED = 0.20;
/**
 * Probe-clearance and blend-validity distance as a fraction of grid spacing.
 * Keeping this dimensionless is required for equivalent scenes expressed in
 * different world-unit scales.
 */
export const DDGI_PROBE_MIN_HIT_DISTANCE_NORMALIZED = 0.05;
export const DDGI_PROBE_MISS_DISTANCE = 1.0e19;
/** Maximum half-float rounding error for normalized values in [-0.5, 0.5]. */
export const DDGI_PROBE_STATE_MAX_QUANTIZATION_ERROR_NORMALIZED = 1 / 8192;
/**
 * A conservative eight binary32 ulps near normalized magnitude 0.45. Persisted explicit state is
 * Float32, so a mathematically boundary-clamped vector can land this far
 * outside under independent component rounding. Larger overshoot is invalid.
 */
export const DDGI_PROBE_STATE_F32_RADIUS_TOLERANCE_NORMALIZED = 2 ** -22;

export type ProbeStateVec3 = readonly [number, number, number];

export interface ProbeClassificationRay {
  readonly direction: ProbeStateVec3;
  /** Negative values are back-face hits; >= DDGI_PROBE_MISS_DISTANCE is a miss. */
  readonly hitDistance: number;
}

export interface ProbeClassificationResult {
  readonly offset: ProbeStateVec3;
  readonly active: boolean;
  readonly validRayCount: number;
  readonly backfaceCount: number;
}

function length3(v: ProbeStateVec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function normalize3(v: ProbeStateVec3): ProbeStateVec3 {
  if (!finiteVec3(v)) return [0, 0, 0];
  const maxComponent = Math.max(Math.abs(v[0]), Math.abs(v[1]), Math.abs(v[2]));
  if (!(maxComponent > 0) || !Number.isFinite(maxComponent)) return [0, 0, 0];
  const scaled: ProbeStateVec3 = [
    v[0] / maxComponent,
    v[1] / maxComponent,
    v[2] / maxComponent,
  ];
  const scaledLength = length3(scaled);
  if (!(scaledLength > 0) || !Number.isFinite(scaledLength)) return [0, 0, 0];
  return [
    scaled[0] / scaledLength,
    scaled[1] / scaledLength,
    scaled[2] / scaledLength,
  ];
}

function addScaled(a: ProbeStateVec3, b: ProbeStateVec3, scale: number): ProbeStateVec3 {
  return [a[0] + b[0] * scale, a[1] + b[1] * scale, a[2] + b[2] * scale];
}

function dot3(a: ProbeStateVec3, b: ProbeStateVec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function finiteVec3(v: ProbeStateVec3): boolean {
  return Number.isFinite(v[0]) && Number.isFinite(v[1]) && Number.isFinite(v[2]);
}

/** Clamp a relocation offset to the closed 0.45-cell sphere. */
export function clampProbeRelocationOffset(
  offset: ProbeStateVec3,
  spacing: number,
): ProbeStateVec3 {
  if (!Number.isFinite(spacing) || spacing <= 0 || !finiteVec3(offset)) return [0, 0, 0];
  const maxLength = DDGI_PROBE_MAX_OFFSET_NORMALIZED * spacing;
  const len = length3(offset);
  if (!(len > maxLength)) return offset;
  const scale = maxLength / len;
  return [offset[0] * scale, offset[1] * scale, offset[2] * scale];
}

/**
 * Deterministic CPU oracle for the GPU classify/relocate pass.
 *
 * Inactive probes are still ray traced by the producer on their next stratum:
 * relocation can therefore move them out of geometry and classification can
 * reactivate them. This function deliberately does not carry temporal hidden
 * state beyond the supplied offset.
 */
export function classifyAndRelocateProbe(
  currentOffset: ProbeStateVec3,
  rays: readonly ProbeClassificationRay[],
  spacing: number,
): ProbeClassificationResult {
  if (!Number.isFinite(spacing) || spacing <= 0) {
    throw new RangeError('DDGI probe spacing must be finite and > 0.');
  }

  const offset = clampProbeRelocationOffset(currentOffset, spacing);
  const minFrontDistance =
    spacing * DDGI_PROBE_MIN_HIT_DISTANCE_NORMALIZED;
  const maxStep = spacing * DDGI_PROBE_MAX_RELOCATION_STEP_NORMALIZED;

  let closestBackfaceDistance = Number.POSITIVE_INFINITY;
  let closestBackfaceDirection: ProbeStateVec3 = [0, 0, 0];
  let closestFrontfaceDistance = Number.POSITIVE_INFINITY;
  let closestFrontfaceDirection: ProbeStateVec3 = [0, 0, 0];
  let farthestFrontfaceDistance = 0;
  let farthestFrontfaceDirection: ProbeStateVec3 = [0, 0, 0];
  let backfaceCount = 0;
  let validRayCount = 0;
  let inspectedRayCount = 0;
  let nearbyFrontface = false;

  for (const ray of rays) {
    if (!Number.isFinite(ray.hitDistance) || !finiteVec3(ray.direction)) continue;
    const direction = normalize3(ray.direction);
    if (length3(direction) === 0) continue;
    inspectedRayCount += 1;
    const distance = ray.hitDistance;
    if (distance < 0) {
      backfaceCount += 1;
      const absoluteDistance = Math.abs(distance);
      if (absoluteDistance < closestBackfaceDistance) {
        closestBackfaceDistance = absoluteDistance;
        closestBackfaceDirection = direction;
      }
      continue;
    }
    // Relocation considers every finite nonnegative ray. In particular, a
    // sub-clearance front hit identifies the wall to escape while a miss is
    // the open direction to move toward. MIN_HIT_DISTANCE is a relocation
    // clearance threshold, not a validity cutoff.
    if (distance < closestFrontfaceDistance) {
      closestFrontfaceDistance = distance;
      closestFrontfaceDirection = direction;
    }
    if (distance > farthestFrontfaceDistance) {
      farthestFrontfaceDistance = distance;
      farthestFrontfaceDirection = direction;
    }
    if (
      distance >= minFrontDistance &&
      distance < DDGI_PROBE_MISS_DISTANCE
    ) {
      validRayCount += 1;
    }

    // Distance from the probe to the first axis-aligned voxel plane crossed by
    // this ray. A hit before that plane means geometry occupies this probe cell.
    const maxAxis = Math.max(
      Math.abs(direction[0]),
      Math.abs(direction[1]),
      Math.abs(direction[2]),
    );
    const voxelPlaneDistance = spacing / maxAxis;
    if (
      distance < DDGI_PROBE_MISS_DISTANCE &&
      distance <= voxelPlaneDistance
    ) {
      nearbyFrontface = true;
    }
  }

  const backfaceFraction =
    inspectedRayCount > 0 ? backfaceCount / inspectedRayCount : 1;
  let candidate = offset;

  if (
    closestBackfaceDistance < Number.POSITIVE_INFINITY &&
    backfaceFraction > DDGI_PROBE_BACKFACE_THRESHOLD
  ) {
    const step = Math.min(
      closestBackfaceDistance + minFrontDistance * 0.5,
      maxStep,
    );
    candidate = addScaled(offset, closestBackfaceDirection, step);
  } else if (
    closestFrontfaceDistance < minFrontDistance &&
    farthestFrontfaceDistance > 0 &&
    dot3(closestFrontfaceDirection, farthestFrontfaceDirection) <= 0
  ) {
    candidate = addScaled(
      offset,
      farthestFrontfaceDirection,
      Math.min(farthestFrontfaceDistance, maxStep),
    );
  } else if (length3(offset) > 0) {
    const clearance = Number.isFinite(closestFrontfaceDistance)
      ? Math.max(0, closestFrontfaceDistance - minFrontDistance)
      : maxStep;
    const moveBack = Math.min(clearance, length3(offset), maxStep);
    candidate = addScaled(offset, normalize3([-offset[0], -offset[1], -offset[2]]), moveBack);
  }

  return {
    offset: clampProbeRelocationOffset(candidate, spacing),
    active:
      inspectedRayCount > 0 &&
      backfaceFraction <= DDGI_PROBE_BACKFACE_THRESHOLD &&
      nearbyFrontface,
    validRayCount,
    backfaceCount,
  };
}

/** Build a tightly packed explicit Float32 snapshot block with zero offsets. */
export function buildInitialProbeStateData(
  probeCount: number,
  active = false,
): Float32Array<ArrayBuffer> {
  const elementCount = probeCount * 4;
  const byteLength = elementCount * Float32Array.BYTES_PER_ELEMENT;
  if (
    !Number.isSafeInteger(probeCount) ||
    probeCount < 0 ||
    !Number.isSafeInteger(elementCount) ||
    !Number.isSafeInteger(byteLength)
  ) {
    throw new RangeError('DDGI probe count must be a non-negative safe integer.');
  }
  const data = new Float32Array(elementCount);
  if (active) {
    for (let probe = 0; probe < probeCount; probe += 1) {
      data[probe * 4 + 3] = DDGI_PROBE_STATE_ACTIVE;
    }
  }
  return data;
}

/**
 * Validate persisted/read-back state before GPU upload. Offsets must be finite
 * and remain inside the same 0.45-cell bound enforced by the compute shader;
 * classification is an exact 0/1 lane, not an arbitrary blend weight.
 */
export function isValidProbeStateData(
  data: Float32Array,
  spacing: number,
): boolean {
  if (!Number.isFinite(spacing) || spacing <= 0 || data.length % 4 !== 0) {
    return false;
  }
  const maxOffset =
    spacing *
    (
      DDGI_PROBE_MAX_OFFSET_NORMALIZED +
      DDGI_PROBE_STATE_F32_RADIUS_TOLERANCE_NORMALIZED
    );
  for (let index = 0; index < data.length; index += 4) {
    const x = data[index]!;
    const y = data[index + 1]!;
    const z = data[index + 2]!;
    const active = data[index + 3]!;
    if (
      !Number.isFinite(x) ||
      !Number.isFinite(y) ||
      !Number.isFinite(z) ||
      Math.hypot(x, y, z) > maxOffset ||
      (active !== DDGI_PROBE_STATE_INACTIVE &&
        active !== DDGI_PROBE_STATE_ACTIVE)
    ) {
      return false;
    }
  }
  return true;
}

export interface PackedProbeStateAtlasLayout {
  readonly dimsX: number;
  readonly dimsY: number;
  readonly dimsZ: number;
  readonly irradianceWidth: number;
  readonly irradianceHeight: number;
  readonly spacing: number;
}

function validatePackedProbeStateAtlasLayout(
  layout: PackedProbeStateAtlasLayout,
): number {
  const { dimsX, dimsY, dimsZ, irradianceWidth, irradianceHeight, spacing } =
    layout;
  const probePlane = dimsX * dimsY;
  const probeCount = probePlane * dimsZ;
  const stackedRows = dimsY * dimsZ;
  const expectedWidth = dimsX * IRR_STRIDE;
  const expectedHeight = stackedRows * IRR_STRIDE;
  const atlasTexelCount = irradianceWidth * irradianceHeight;
  const atlasElementCount = atlasTexelCount * 4;
  const atlasByteLength = atlasElementCount * Uint16Array.BYTES_PER_ELEMENT;
  const explicitElementCount = probeCount * 4;
  const explicitByteLength =
    explicitElementCount * Float32Array.BYTES_PER_ELEMENT;
  if (
    !Number.isSafeInteger(dimsX) || dimsX <= 0 ||
    !Number.isSafeInteger(dimsY) || dimsY <= 0 ||
    !Number.isSafeInteger(dimsZ) || dimsZ <= 0 ||
    !Number.isSafeInteger(probePlane) ||
    !Number.isSafeInteger(probeCount) ||
    !Number.isSafeInteger(stackedRows) ||
    !Number.isSafeInteger(expectedWidth) ||
    !Number.isSafeInteger(expectedHeight) ||
    !Number.isSafeInteger(irradianceWidth) ||
    irradianceWidth !== expectedWidth ||
    !Number.isSafeInteger(irradianceHeight) ||
    irradianceHeight !== expectedHeight ||
    !Number.isSafeInteger(atlasTexelCount) ||
    !Number.isSafeInteger(atlasElementCount) ||
    !Number.isSafeInteger(atlasByteLength) ||
    !Number.isSafeInteger(explicitElementCount) ||
    !Number.isSafeInteger(explicitByteLength) ||
    !Number.isFinite(spacing) ||
    spacing <= 0 ||
    !Number.isFinite(Math.fround(spacing)) ||
    !(Math.fround(spacing) > 0)
  ) {
    throw new RangeError('DDGI packed probe-state atlas layout is invalid.');
  }
  return probeCount;
}

/** RGBA element offset of a probe's reserved (4,4) irradiance-ring texel. */
export function packedProbeStateElementOffset(
  probeIndex: number,
  layout: PackedProbeStateAtlasLayout,
): number {
  const probeCount = validatePackedProbeStateAtlasLayout(layout);
  if (
    !Number.isSafeInteger(probeIndex) ||
    probeIndex < 0 ||
    probeIndex >= probeCount
  ) {
    throw new RangeError('DDGI packed probe-state index is out of range.');
  }
  const px = probeIndex % layout.dimsX;
  const yz = Math.floor(probeIndex / layout.dimsX);
  const atlasX = px * IRR_STRIDE + IRR_PROBE_STATE_LOCAL_X;
  const atlasY = yz * IRR_STRIDE + IRR_PROBE_STATE_LOCAL_Y;
  return (atlasY * layout.irradianceWidth + atlasX) * 4;
}

/**
 * Encode explicit world-space Float32 probe state into the irradiance atlas.
 * Offsets are normalized by spacing before f16 storage, keeping every lane
 * inside [-0.45, 0.45] regardless of scene scale; active remains exact 0/1.
 */
export function writePackedProbeStateToIrradianceAtlas(
  irradianceData: Uint16Array,
  probeStateData: Float32Array,
  layout: PackedProbeStateAtlasLayout,
): void {
  const probeCount = validatePackedProbeStateAtlasLayout(layout);
  if (
    irradianceData.length !==
      layout.irradianceWidth * layout.irradianceHeight * 4 ||
    probeStateData.length !== probeCount * 4 ||
    !isValidProbeStateData(probeStateData, layout.spacing)
  ) {
    throw new RangeError('DDGI probe state cannot be packed into this irradiance atlas.');
  }
  for (let probe = 0; probe < probeCount; probe += 1) {
    const source = probe * 4;
    const target = packedProbeStateElementOffset(probe, layout);
    irradianceData[target] = float32ToFloat16Bits(
      probeStateData[source]! / layout.spacing,
    );
    irradianceData[target + 1] = float32ToFloat16Bits(
      probeStateData[source + 1]! / layout.spacing,
    );
    irradianceData[target + 2] = float32ToFloat16Bits(
      probeStateData[source + 2]! / layout.spacing,
    );
    irradianceData[target + 3] = float32ToFloat16Bits(
      probeStateData[source + 3]!,
    );
  }
}

/** Decode the packed normalized f16 state back to explicit world-space f32. */
export function readPackedProbeStateFromIrradianceAtlas(
  irradianceData: Uint16Array,
  layout: PackedProbeStateAtlasLayout,
): Float32Array<ArrayBuffer> {
  const probeCount = validatePackedProbeStateAtlasLayout(layout);
  if (
    irradianceData.length !==
    layout.irradianceWidth * layout.irradianceHeight * 4
  ) {
    throw new RangeError('DDGI irradiance atlas data length is invalid.');
  }
  const out = new Float32Array(probeCount * 4);
  for (let probe = 0; probe < probeCount; probe += 1) {
    const source = packedProbeStateElementOffset(probe, layout);
    const target = probe * 4;
    const decoded = clampProbeRelocationOffset([
      float16BitsToFloat32(irradianceData[source]!) * layout.spacing,
      float16BitsToFloat32(irradianceData[source + 1]!) * layout.spacing,
      float16BitsToFloat32(irradianceData[source + 2]!) * layout.spacing,
    ], layout.spacing);
    let x = Math.fround(decoded[0]);
    let y = Math.fround(decoded[1]);
    let z = Math.fround(decoded[2]);
    const maxRadius =
      layout.spacing * DDGI_PROBE_MAX_OFFSET_NORMALIZED;
    const roundedRadius = Math.hypot(x, y, z);
    if (roundedRadius > maxRadius) {
      // Assigning the clamped double components into a Float32Array can round
      // their vector a few ulps back outside the closed physical bound. Leave
      // a tiny inward f32 guard (far below the f16 quantization budget).
      const guardedRadius = Math.max(0, maxRadius - layout.spacing / 1_048_576);
      const scale = guardedRadius / roundedRadius;
      x = Math.fround(x * scale);
      y = Math.fround(y * scale);
      z = Math.fround(z * scale);
    }
    if (Math.hypot(x, y, z) > maxRadius) {
      throw new RangeError(
        'DDGI packed probe offset cannot be represented inside its physical bound.',
      );
    }
    out[target] = x;
    out[target + 1] = y;
    out[target + 2] = z;
    out[target + 3] = float16BitsToFloat32(irradianceData[source + 3]!);
  }
  return out;
}
