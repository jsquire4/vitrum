import type { Scene } from '@vitrum/core';

export interface OneEdgeReconnectionUnsupportedMaterial {
  readonly primitiveId: string;
  readonly transmission: number;
  readonly hasTransmissionMap: boolean;
}

/**
 * Structured construction/mutation error for the exact production domain of
 * pt-webgpu's one-edge GRIS reconnection estimator.
 */
export class OneEdgeReconnectionDomainError extends TypeError {
  readonly code = 'pt-webgpu.one-edge-reconnection.transmission-unsupported';
  readonly details: {
    readonly strategy: 'opaque-one-edge-gris-reconnection';
    readonly unsupportedMaterials: readonly OneEdgeReconnectionUnsupportedMaterial[];
  };

  constructor(unsupportedMaterials: readonly OneEdgeReconnectionUnsupportedMaterial[]) {
    const ids = unsupportedMaterials.map((entry) => entry.primitiveId).join(', ');
    super(
      '[vitrum/pt-webgpu] oneEdgeReconnectionReuse supports finite opaque ' +
      'one-edge reconnection only; transmissive materials require a multi-event ' +
      `shift/replay strategy and are rejected before GPU publication (primitive(s): ${ids}).`,
    );
    this.name = 'OneEdgeReconnectionDomainError';
    this.details = Object.freeze({
      strategy: 'opaque-one-edge-gris-reconnection',
      unsupportedMaterials: Object.freeze(
        unsupportedMaterials.map((entry) => Object.freeze({ ...entry })),
      ),
    });
  }
}

/**
 * Reject every material whose effective transmission factor can be non-zero.
 * A transmission map is multiplicative, so it is relevant only when the
 * authored scalar factor is positive.
 */
export function assertOneEdgeReconnectionSceneSupported(scene: Scene): void {
  const unsupported: OneEdgeReconnectionUnsupportedMaterial[] = [];
  for (const primitive of scene.primitives) {
    const transmission = primitive.material.transmission ?? 0;
    if (transmission <= 0) continue;
    unsupported.push({
      primitiveId: primitive.id,
      transmission,
      hasTransmissionMap: primitive.material.transmissionMap != null,
    });
  }
  if (unsupported.length > 0) {
    throw new OneEdgeReconnectionDomainError(unsupported);
  }
}
