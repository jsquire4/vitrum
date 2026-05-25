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
  return Float32Array.from(params);
}

/** Decode packed params after validating length for the shape. */
export function decodeAnalyticParams<S extends AnalyticShape>(
  shape: S,
  packed: Float32Array,
): AnalyticParamsByShape[S] {
  const len = ANALYTIC_PARAM_LENGTH[shape];
  if (packed.length !== len) {
    throw new Error(
      `decodeAnalyticParams: shape "${shape}" expects ${len} values, got ${packed.length}`,
    );
  }
  return Array.from(packed) as unknown as AnalyticParamsByShape[S];
}

export function validateAnalyticParams(shape: AnalyticShape, packed: Float32Array): void {
  const len = ANALYTIC_PARAM_LENGTH[shape];
  if (packed.length !== len) {
    throw new Error(
      `validateAnalyticParams: shape "${shape}" expects ${len} values, got ${packed.length}`,
    );
  }
}
