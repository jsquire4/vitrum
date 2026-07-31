import type { Scene } from '@vitrum/core';

/**
 * Camera-side dielectric throughput in the bounded GI reservoir cannot be
 * reconnection-shifted without storing the full refractive prefix. Such scenes
 * therefore require reservoir scale 1; auto mode may still reduce the whole
 * internal shading resolution.
 */
export function sceneRequiresExactGiCameraPrefixes(scene: Scene): boolean {
  return scene.primitives.some((primitive) => {
    if (!('material' in primitive)) return false;
    const material = primitive.material;
    return (material.transmission ?? 0) > 0
      || material.transmissionMap !== undefined;
  });
}

/**
 * Decide whether an incremental scene transition must join a transactional
 * frame-resource replan so the planner can select a new reservoir scale and,
 * when necessary, a new whole-graph internal resolution.
 */
export function sceneTransitionRequiresGiScaleReplan(
  previousScene: Scene,
  nextScene: Scene,
  currentReservoirScale: number,
  configuredReservoirScale: number | undefined,
): boolean {
  const previousRequiresExact =
    sceneRequiresExactGiCameraPrefixes(previousScene);
  const nextRequiresExact = sceneRequiresExactGiCameraPrefixes(nextScene);

  if (nextRequiresExact && currentReservoirScale !== 1) {
    return true;
  }
  return previousRequiresExact
    && !nextRequiresExact
    && configuredReservoirScale === undefined;
}
