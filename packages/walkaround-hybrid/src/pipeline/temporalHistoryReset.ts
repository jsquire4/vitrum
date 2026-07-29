import type { CollectedBvhMutation } from './CollectingBvhUpdateSink.js';
import type { FrameResources } from './resourceManager.js';

/**
 * Camera-only motion vectors cannot represent object, skinning, or transform
 * edits. Until previous object-space positions exist, every reachable scene
 * mutation is therefore a fail-safe temporal-history boundary.
 */
export function sceneMutationRequiresTemporalReset(
  mutation: CollectedBvhMutation,
): boolean {
  return (
    mutation.resetAccumulator ||
    mutation.nodes != null ||
    mutation.positions != null ||
    mutation.learningPositions != null ||
    mutation.normals != null ||
    mutation.tlas != null ||
    mutation.replacement != null ||
    mutation.material != null ||
    mutation.atlas != null ||
    mutation.emitters != null
  );
}

/** Clear every persistent DI/GI reservoir before the reset frame dispatches. */
export function clearTemporalReservoirHistory(
  encoder: GPUCommandEncoder,
  resources: Pick<FrameResources, 'restirDI' | 'restirGI'>,
): void {
  encoder.clearBuffer(resources.restirDI.reservoirCurrentBuffer);
  encoder.clearBuffer(resources.restirDI.reservoirPreviousBuffer);
  encoder.clearBuffer(resources.restirDI.reservoirSpatialBuffer);
  encoder.clearBuffer(resources.restirGI.reservoirGiCurrentBuffer);
  encoder.clearBuffer(resources.restirGI.reservoirGiPreviousBuffer);
  encoder.clearBuffer(resources.restirGI.reservoirGiSpatialBuffer);
}
