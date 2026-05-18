/**
 * Dichroic LUT extension converter — stained-glass-studio host extension.
 *
 * The stained-glass dichroic body baker emits two 256×1 RGBA-float
 * DataTextures pre-convolving the TMM × CIE 1931 standard observer. Raster
 * backends bind the LUTs directly; PT backends evaluate TMM in-shader from
 * the (already-stamped) `thinFilmStack` and ignore the LUT.
 *
 * The converter survives the THREE → vitrum → THREE round trip via
 * `Material.extensions.dichroicLUTs`:
 *
 *   THREE.MeshPhysicalMaterial.userData['vitrumDichroicReflectanceLUT']
 *   THREE.MeshPhysicalMaterial.userData['vitrumDichroicTransmittanceLUT']
 *     ↕
 *   vitrum.Material.extensions['dichroicLUTs']
 *     = { reflectance: <texture handle?>, transmittance: <texture handle?> }
 *
 * Texture handles are opaque (the contract is "vitrum.Material.TextureRef =
 * unknown"). The converter copies them through verbatim — no interpretation,
 * no validation — so consumers downstream of the round trip can re-bind them
 * verbatim against whichever GPU pipeline they own.
 *
 * Provenance: stained-glass dichroic addendum (2026-05-12); TMM × CIE 1931
 * standard observer integration.
 */

import type * as THREE from 'three';
import type { Material as VitrumMaterial } from '@vitrum/core';
import type { MaterialExtensionConverter } from '@vitrum/three-bindings';

/**
 * Stable id for the dichroic-LUT extension. Used as the key inside
 * `Material.extensions`.
 */
export const DICHROIC_LUTS_EXTENSION_ID = 'dichroicLUTs';

/**
 * Canonical userData keys used by the stained-glass baker on the THREE side.
 */
export const STAINED_GLASS_USER_DATA_KEYS = {
  DICHROIC_REFLECTANCE_LUT:   'vitrumDichroicReflectanceLUT',
  DICHROIC_TRANSMITTANCE_LUT: 'vitrumDichroicTransmittanceLUT',
} as const;

/**
 * Shape of `Material.extensions['dichroicLUTs']`. Both LUT slots are optional
 * — the baker may emit only the reflectance LUT for one-sided coatings.
 */
export interface DichroicLUTsExtension {
  /** Pre-convolved angle-indexed reflectance LUT (e.g. THREE.DataTexture). */
  readonly reflectance?: unknown;
  /** Pre-convolved angle-indexed transmittance LUT (e.g. THREE.DataTexture). */
  readonly transmittance?: unknown;
}

/**
 * Opt-in extension converter that wires the stained-glass dichroic LUTs
 * through `@vitrum/three-bindings`'s pluggable converter registry. Hosts that
 * want dichroic LUTs to survive the round trip pass this into the
 * `extensionConverters` option of `sceneFromThreeJS` / `convertMaterial` /
 * `vitrumSceneToThree` / `vitrumMaterialToThree`.
 */
export const dichroicLUTsExtensionConverter: MaterialExtensionConverter = {
  id: 'stained-glass.dichroic-luts',

  forward(threeMat: THREE.Material): Readonly<Record<string, unknown>> | undefined {
    const ud = (threeMat.userData ?? {}) as Record<string, unknown>;
    const reflectance = ud[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_REFLECTANCE_LUT];
    const transmittance = ud[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_TRANSMITTANCE_LUT];
    if (reflectance == null && transmittance == null) return undefined;
    const ext: { reflectance?: unknown; transmittance?: unknown } = {};
    if (reflectance != null) ext.reflectance = reflectance;
    if (transmittance != null) ext.transmittance = transmittance;
    return { [DICHROIC_LUTS_EXTENSION_ID]: ext };
  },

  reverse(vitrumMat: VitrumMaterial, threeMat: THREE.Material): void {
    const dichroic = vitrumMat.extensions?.[DICHROIC_LUTS_EXTENSION_ID] as
      | DichroicLUTsExtension
      | undefined;
    if (dichroic == null) return;
    const ud: Record<string, unknown> = threeMat.userData ?? {};
    if (dichroic.reflectance != null) {
      ud[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_REFLECTANCE_LUT] = dichroic.reflectance;
    }
    if (dichroic.transmittance != null) {
      ud[STAINED_GLASS_USER_DATA_KEYS.DICHROIC_TRANSMITTANCE_LUT] = dichroic.transmittance;
    }
    threeMat.userData = ud;
  },
};
