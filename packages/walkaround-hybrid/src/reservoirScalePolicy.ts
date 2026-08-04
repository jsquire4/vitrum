import type { Scene } from '@vitrum/core';

/**
 * Camera-side dielectric throughput is evaluated natively in shadeMain. A
 * checkerboard gap pixel has no exact primary refractive prefix to reproject,
 * so material transmission forces full-rate internal shading while remaining
 * independent of the DI/GI reservoir-grid scale.
 */
export function sceneRequiresFullRateGlassShading(scene: Scene): boolean {
  return scene.primitives.some((primitive) => {
    if (!('material' in primitive)) return false;
    const material = primitive.material;
    // The map modulates the scalar factor; it cannot create transmission from
    // the contract's opaque default. Match sampleTransmissionMapForHit rather
    // than disabling checkerboard for an inert map attached to factor zero.
    return (material.transmission ?? 0) > 0;
  });
}
