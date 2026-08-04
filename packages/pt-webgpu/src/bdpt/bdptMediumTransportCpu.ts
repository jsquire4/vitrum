/** CPU oracle for the directional medium endpoint state used by WebGPU BDPT. */

export type BdptMediumVec3 = readonly [number, number, number];

export interface BdptEndpointMediumCpu {
  readonly matId: number;
  readonly initialDistance: number;
  readonly remainingDistance: number;
}

export interface BdptMediumEndpointCpu {
  readonly isMedium: boolean;
  readonly normal: BdptMediumVec3;
  readonly active: BdptEndpointMediumCpu;
  readonly incident: BdptEndpointMediumCpu;
  readonly transmitted: BdptEndpointMediumCpu;
}

export const BDPT_NO_MEDIUM_CPU = 0xffffffff;

export const BDPT_MEDIUM_BOUNDARY_KIND_TLAS_CPU = 0;
export const BDPT_MEDIUM_BOUNDARY_KIND_ANALYTIC_CPU = 1;
export const BDPT_MEDIUM_BOUNDARY_KIND_INVALID_CPU = 0xffffffff;

export interface BdptMediumBoundaryLayerCpu {
  readonly matId: number;
  readonly boundaryKind: number;
  readonly boundaryIndex: number;
}

export interface BdptMediumBoundaryCrossingCpu extends BdptMediumBoundaryLayerCpu {
  readonly entering: boolean;
}

export type BdptMediumStackTransitionCpu =
  | {
      readonly ok: true;
      readonly stack: readonly BdptMediumBoundaryLayerCpu[];
    }
  | {
      readonly ok: false;
      readonly reason: 'invalid-boundary' | 'stack-overflow' | 'boundary-mismatch';
      /** The exact input stack; rejected transitions never mutate or reorder it. */
      readonly stack: readonly BdptMediumBoundaryLayerCpu[];
    };

/**
 * CPU oracle for the production WGSL medium crossing rule.
 *
 * A boundary is `(kind,index)`, not a material id: two nested instances may
 * intentionally share one material. Enter pushes one exact boundary; exit may
 * pop only the current top. Invalid identities, overflow, underflow, and an
 * out-of-order exit fail without scanning, removing, or reordering the stack.
 */
export function transitionBdptMediumStackCpu(
  stack: readonly BdptMediumBoundaryLayerCpu[],
  crossing: BdptMediumBoundaryCrossingCpu,
  stackLimit = 8,
): BdptMediumStackTransitionCpu {
  if (crossing.boundaryKind === BDPT_MEDIUM_BOUNDARY_KIND_INVALID_CPU) {
    return { ok: false, reason: 'invalid-boundary', stack };
  }
  if (crossing.entering) {
    if (stack.length >= stackLimit) {
      return { ok: false, reason: 'stack-overflow', stack };
    }
    return {
      ok: true,
      stack: [
        ...stack,
        {
          matId: crossing.matId,
          boundaryKind: crossing.boundaryKind,
          boundaryIndex: crossing.boundaryIndex,
        },
      ],
    };
  }

  const top = stack.at(-1);
  if (
    top == null ||
    top.matId !== crossing.matId ||
    top.boundaryKind !== crossing.boundaryKind ||
    top.boundaryIndex !== crossing.boundaryIndex
  ) {
    return { ok: false, reason: 'boundary-mismatch', stack };
  }
  return { ok: true, stack: stack.slice(0, -1) };
}

export const BDPT_NO_ENDPOINT_MEDIUM_CPU: BdptEndpointMediumCpu = {
  matId: BDPT_NO_MEDIUM_CPU,
  initialDistance: Number.MAX_VALUE,
  remainingDistance: Number.MAX_VALUE,
};

function dot(a: BdptMediumVec3, b: BdptMediumVec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function selectBdptEndpointMediumCpu(
  endpoint: BdptMediumEndpointCpu,
  directionToOther: BdptMediumVec3,
): BdptEndpointMediumCpu {
  if (endpoint.isMedium) return endpoint.active;
  return dot(endpoint.normal, directionToOther) >= 0 ? endpoint.incident : endpoint.transmitted;
}

export function sharedBdptEdgeMediumCpu(
  a: BdptMediumEndpointCpu,
  directionAToB: BdptMediumVec3,
  b: BdptMediumEndpointCpu,
  directionBToA: BdptMediumVec3,
): BdptEndpointMediumCpu | null {
  const sideA = selectBdptEndpointMediumCpu(a, directionAToB);
  const sideB = selectBdptEndpointMediumCpu(b, directionBToA);
  if (sideA.matId !== sideB.matId) return null;
  if (sideA.matId === BDPT_NO_MEDIUM_CPU) return BDPT_NO_ENDPOINT_MEDIUM_CPU;
  if (
    !Number.isFinite(sideA.initialDistance) ||
    !Number.isFinite(sideB.initialDistance) ||
    !Number.isFinite(sideA.remainingDistance) ||
    !Number.isFinite(sideB.remainingDistance) ||
    sideA.initialDistance < 0 || sideB.initialDistance < 0 ||
    sideA.remainingDistance < 0 || sideB.remainingDistance < 0
  ) return null;
  const aUnbounded = sideA.initialDistance === Number.MAX_VALUE;
  const bUnbounded = sideB.initialDistance === Number.MAX_VALUE;
  if (aUnbounded || bUnbounded) {
    return aUnbounded && bUnbounded
      ? { ...BDPT_NO_ENDPOINT_MEDIUM_CPU, matId: sideA.matId }
      : null;
  }
  if (
    sideA.initialDistance !== sideB.initialDistance ||
    sideA.remainingDistance > sideA.initialDistance ||
    sideB.remainingDistance > sideB.initialDistance
  ) return null;
  return {
    matId: sideA.matId,
    initialDistance: sideA.initialDistance,
    remainingDistance: Math.max(
      sideA.remainingDistance + sideB.remainingDistance - sideA.initialDistance,
      0,
    ),
  };
}

export function bdptEffectiveMediumDistanceCpu(
  distance: number,
  medium: BdptEndpointMediumCpu,
): number {
  return Math.min(Math.max(distance, 0), Math.max(medium.remainingDistance, 0));
}

export function bdptSegmentDistanceDensityCpu(
  sigmaT: number,
  distance: number,
  remainingDistance: number,
  destinationIsMedium: boolean,
): number {
  const effectiveDistance = Math.min(Math.max(distance, 0), Math.max(remainingDistance, 0));
  if (sigmaT <= 0) return destinationIsMedium ? 0 : 1;
  const survival = Math.exp(-sigmaT * effectiveDistance);
  return destinationIsMedium ? sigmaT * survival : survival;
}

export function bdptConnectionTransmittanceCpu(
  sigmaT: readonly [number, number, number],
  distance: number,
  medium: BdptEndpointMediumCpu,
): readonly [number, number, number] {
  const effectiveDistance = bdptEffectiveMediumDistanceCpu(distance, medium);
  return [
    Math.exp(-sigmaT[0] * effectiveDistance),
    Math.exp(-sigmaT[1] * effectiveDistance),
    Math.exp(-sigmaT[2] * effectiveDistance),
  ];
}

export function bdptHgPhaseCpu(cosThetaRaw: number, gRaw: number): number {
  const g = Math.max(-0.999999, Math.min(0.999999, gRaw));
  const a = Math.abs(g);
  const cosTheta = Math.max(-1, Math.min(1, cosThetaRaw));
  const alignedCos = g >= 0 ? cosTheta : -cosTheta;
  const oneMinusA = 1 - a;
  const denom = oneMinusA * oneMinusA + 2 * a * (1 - alignedCos);
  return (oneMinusA * (1 + a)) / (4 * Math.PI * denom * Math.sqrt(denom));
}
