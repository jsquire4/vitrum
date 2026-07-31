import type { MaterialSpec } from '@vitrum/core';
import { materialSpecSkipEmitter } from '@vitrum/shared-bvh';

/**
 * True when this material's camera/BSDF-hit emission must not be assigned a
 * mesh-light NEE proposal.
 *
 * `skipEmitter` explicitly removes an implicit emitter. Filtered or mipmapped
 * emissive maps remain valid surface textures, but their continuously varying
 * radiance cannot use the constant-radiance texel-cell distribution built by
 * `meshAreaLights.ts`. Those maps therefore use the unbiased forward estimator
 * only.
 */
export function materialEmissionExcludedFromMeshNee(material: MaterialSpec): boolean {
  if (materialSpecSkipEmitter(material)) return true;
  const ref = material.emissiveMap;
  if (ref == null) return false;
  return (
    (ref.magFilter ?? 'nearest') !== 'nearest' ||
    (ref.minFilter ?? 'nearest') !== 'nearest' ||
    (ref.mipFilter ?? 'none') !== 'none'
  );
}

/**
 * An explicit `mesh-area` emitter is authoritative over the material's
 * implicit-only `extensions.skipEmitter` hint. Return a backend-local clone
 * without that hint while preserving every other material field/extension.
 */
export function materialWithExplicitMeshEmitterAuthority(
  material: MaterialSpec,
): MaterialSpec {
  if (!materialSpecSkipEmitter(material)) return material;
  const { extensions: _extensions, ...withoutExtensions } = material;
  const extensions = { ...material.extensions };
  delete extensions['skipEmitter'];
  return Object.keys(extensions).length > 0
    ? { ...withoutExtensions, extensions }
    : withoutExtensions;
}
