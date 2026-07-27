/**
 * Typed analytic primitive parameter layouts — single source for encode/decode.
 */

import type { AnalyticShape } from './primitives.js';

export const ANALYTIC_PARAM_LENGTH: Readonly<Record<AnalyticShape, number>> = {
  sphere: 4,
  box: 6,
  capsule: 7,
  cylinder: 5,
  'h-channel-came': 4,
};

export type SphereAnalyticParams = readonly [cx: number, cy: number, cz: number, radius: number];
export type BoxAnalyticParams = readonly [cx: number, cy: number, cz: number, hx: number, hy: number, hz: number];
export type CapsuleAnalyticParams = readonly [ax: number, ay: number, az: number, bx: number, by: number, bz: number, radius: number];
export type CylinderAnalyticParams = readonly [cx: number, cy: number, cz: number, radius: number, halfHeight: number];
export type HChannelCameParams = readonly [length: number, railWidth: number, blockHeight: number, webThickness: number];

export type AnalyticParamsByShape = {
  sphere: SphereAnalyticParams;
  box: BoxAnalyticParams;
  capsule: CapsuleAnalyticParams;
  cylinder: CylinderAnalyticParams;
  'h-channel-came': HChannelCameParams;
};

/** Encode typed params to the packed Float32Array consumed by backends. */
export function encodeAnalyticParams<S extends AnalyticShape>(
  shape: S,
  params: AnalyticParamsByShape[S],
): Float32Array {
  const len = ANALYTIC_PARAM_LENGTH[shape];
  if (params.length !== len) {
    throw new Error(
      `encodeAnalyticParams: shape "${shape}" expects ${len} values, got ${params.length}`,
    );
  }
  const packed = Float32Array.from(params);
  validateAnalyticParams(shape, packed);
  return packed;
}

/** Decode packed params after validating length for the shape. */
export function decodeAnalyticParams<S extends AnalyticShape>(
  shape: S,
  packed: Float32Array,
): AnalyticParamsByShape[S] {
  validateAnalyticParams(shape, packed);
  return Array.from(packed) as unknown as AnalyticParamsByShape[S];
}

export function validateAnalyticParams(shape: AnalyticShape, packed: Float32Array): void {
  if (!Object.prototype.hasOwnProperty.call(ANALYTIC_PARAM_LENGTH, shape)) {
    throw new TypeError(`validateAnalyticParams: unsupported shape "${String(shape)}"`);
  }
  if (!(packed instanceof Float32Array)) {
    throw new TypeError(
      `validateAnalyticParams: shape "${shape}" params must be a Float32Array`,
    );
  }
  const len = ANALYTIC_PARAM_LENGTH[shape];
  if (packed.length !== len) {
    throw new Error(
      `validateAnalyticParams: shape "${shape}" expects ${len} values, got ${packed.length}`,
    );
  }
  for (let index = 0; index < packed.length; index += 1) {
    const value = packed[index];
    if (!Number.isFinite(value)) {
      throw new RangeError(
        `validateAnalyticParams: shape "${shape}" params[${index}] must be finite (got ${String(value)})`,
      );
    }
  }

  const positive = (index: number, label: string): void => {
    const value = packed[index]!;
    if (!(value > 0)) {
      throw new RangeError(
        `validateAnalyticParams: shape "${shape}" ${label} must be > 0 (got ${value})`,
      );
    }
  };
  switch (shape) {
    case 'sphere':
      positive(3, 'radius');
      break;
    case 'box':
      positive(3, 'half-width');
      positive(4, 'half-height');
      positive(5, 'half-depth');
      break;
    case 'capsule':
      positive(6, 'radius');
      break;
    case 'cylinder':
      positive(3, 'radius');
      positive(4, 'half-height');
      break;
    case 'h-channel-came': {
      positive(0, 'length');
      positive(1, 'railWidth');
      positive(2, 'blockHeight');
      positive(3, 'webThickness');
      const webThickness = packed[3]!;
      if (webThickness >= Math.min(packed[1]!, packed[2]!)) {
        throw new RangeError(
          `validateAnalyticParams: shape "h-channel-came" webThickness must be smaller than ` +
            `railWidth and blockHeight (got ${webThickness})`,
        );
      }
      break;
    }
  }
}
