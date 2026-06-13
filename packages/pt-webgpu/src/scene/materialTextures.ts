// materialTextures.ts — P2 host-side texture collection for pt-webgpu.
//
// Given the scene's MaterialSpec[], dedup the texture-map source handles into an
// upload-ordered list and pack a per-material descriptor buffer (texture indices
// + alpha-mode + the KHR_texture_transform UV transform / texCoord). The GPU
// upload step (follow-on) turns `sources` into a texture_2d_array; the WGSL
// sampler reads the descriptor buffer (this layout) to sample with the right
// index + UVs. Materials with no maps get index -1 → the sampler skips them, so
// a textureless scene stays byte-identical to the pre-P2 parametric path.

import type { MaterialSpec, TextureRef } from '@vitrum/core';
import type { MaterialTextureLayerUvScale } from './materialTextureArray.js';

/**
 * vec4s per material in the descriptor buffer (MUST match the WGSL
 * `MATERIAL_TEX_VEC4_STRIDE`):
 *   0: {baseColorIdx, normalIdx, ormIdx, emissiveIdx}   (-1 = no map)
 *   1: {alphaMode (0 opaque/1 mask/2 blend), alphaCutoff, opacity, texCoord}
 *   2: {offsetX, offsetY, scaleX, scaleY}                (baseColor UV transform)
 *   3: {rotation, aoMapIdx, lightMapIdx, bumpMapIdx}     ← D3 (-1 = no map)
 *   4: {aoMapIntensity, lightMapIntensity, bumpScale, envMapIntensity}  ← D3
 *   5: {anisotropy, anisotropyRotation, anisotropyMapIdx, normalScale}  ← D3/PTWG-MAT
 *   6: {alphaMapIdx, transmissionMapIdx, _, _}            (-1 = no map)
 *   7: {baseColorUvScale.xy, emissiveUvScale.xy}           (per-layer UV-fit)
 *   8: {normalUvScale.xy, ormUvScale.xy}
 *   9: {aoUvScale.xy, lightMapUvScale.xy}
 *  10: {bumpUvScale.xy, anisotropyUvScale.xy}
 *  11: {alphaUvScale.xy, transmissionUvScale.xy}
 *
 * D3 (reserved-field consumption) bumped the stride 4 → 6:
 *   - vec4 #3.yzw + vec4 #4.xyz: aoMap / lightMap / bumpMap layer indices and
 *     their intensity / scale scalars. All three maps are LINEAR-space (occlusion,
 *     baked-radiance-as-data, height field) so they share the LINEAR texture array
 *     index space (materialTexturesLinear). A material lacking a given map carries
 *     index -1 (the WGSL sampler returns a no-op), so absent-field scenes stay
 *     byte-identical to the pre-D3 path.
 *   - vec4 #4.w: per-material envMapIntensity (default 1).
 *   - vec4 #5: anisotropy / anisotropyRotation scalars + the optional
 *     anisotropyMap layer index (KHR_materials_anisotropy: RG = tangent rotation
 *     direction, B = strength), also in the LINEAR array. anisotropy == 0 (default)
 *     means the anisotropic GGX path is never taken → byte-identical.
 *   - vec4 #5.w: glTF normalTexture.scale / MaterialSpec.normalScale. Default 1,
 *     so legacy normal-mapped scenes remain byte-identical unless authored scale
 *     asks to dampen or amplify the tangent-space xy perturbation.
 *   - vec4 #6.x: standalone alphaMap layer in the LINEAR array (coverage data,
 *     not color). It multiplies baseColor alpha and opacity in alphaMode mask/blend.
 *   - vec4 #6.y: transmissionMap layer in the LINEAR array. It multiplies the
 *     scalar `MaterialSpec.transmission` (glTF KHR_materials_transmission R channel).
 *   - vec4 #7–#11: per-map UV-fit scales. Heterogeneous texture arrays copy each
 *     source into a max-sized layer; these scales remap repeat-wrapped UVs into
 *     the copied source rectangle instead of sampling padded black texels.
 */
export const MATERIAL_TEX_VEC4_STRIDE = 12;
export const MATERIAL_TEX_FLOAT_STRIDE = MATERIAL_TEX_VEC4_STRIDE * 4;

const ALPHA_MODE_INDEX: Readonly<Record<'opaque' | 'mask' | 'blend', number>> = {
  opaque: 0,
  mask: 1,
  blend: 2,
};

export interface CollectedTextures {
  /** Unique sRGB-decoded texture sources (baseColor + emissive), upload order. */
  readonly sources: unknown[];
  /** Unique LINEAR texture sources (normal + ORM — must NOT be sRGB-decoded),
   *  a separate index space → its own texture_2d_array. */
  readonly linearSources: unknown[];
  /** Per-material descriptor floats (MATERIAL_TEX_FLOAT_STRIDE per material). */
  readonly descriptors: Float32Array;
}

function uvFitScaleFor(
  scales: readonly MaterialTextureLayerUvScale[],
  layerIdx: number,
): MaterialTextureLayerUvScale {
  if (layerIdx < 0 || layerIdx >= scales.length) return [1, 1];
  return scales[layerIdx] ?? [1, 1];
}

function writeUvFitPair(
  descriptors: Float32Array,
  offset: number,
  scale: MaterialTextureLayerUvScale,
): void {
  descriptors[offset] = scale[0];
  descriptors[offset + 1] = scale[1];
}

function writeDefaultUvFitPairs(descriptors: Float32Array, b: number): void {
  for (let offset = b + 28; offset < b + 48; offset += 2) {
    descriptors[offset] = 1;
    descriptors[offset + 1] = 1;
  }
}

/** Fill per-map UV-fit descriptor lanes after the texture arrays reveal their
 *  actual per-layer source rects. Same-size layers remain [1,1]. */
export function applyMaterialTextureUvFitScales(
  descriptors: Float32Array,
  sRgbLayerScales: readonly MaterialTextureLayerUvScale[],
  linearLayerScales: readonly MaterialTextureLayerUvScale[],
): void {
  const materialCount = Math.floor(descriptors.length / MATERIAL_TEX_FLOAT_STRIDE);
  for (let mi = 0; mi < materialCount; mi += 1) {
    const b = mi * MATERIAL_TEX_FLOAT_STRIDE;
    // sRGB array maps: baseColor and emissive.
    writeUvFitPair(descriptors, b + 28, uvFitScaleFor(sRgbLayerScales, descriptors[b + 0] ?? -1));
    writeUvFitPair(descriptors, b + 30, uvFitScaleFor(sRgbLayerScales, descriptors[b + 3] ?? -1));
    // Linear array maps: normal, ORM, AO, light, bump, anisotropy, alpha, transmission.
    writeUvFitPair(descriptors, b + 32, uvFitScaleFor(linearLayerScales, descriptors[b + 1] ?? -1));
    writeUvFitPair(descriptors, b + 34, uvFitScaleFor(linearLayerScales, descriptors[b + 2] ?? -1));
    writeUvFitPair(descriptors, b + 36, uvFitScaleFor(linearLayerScales, descriptors[b + 13] ?? -1));
    writeUvFitPair(descriptors, b + 38, uvFitScaleFor(linearLayerScales, descriptors[b + 14] ?? -1));
    writeUvFitPair(descriptors, b + 40, uvFitScaleFor(linearLayerScales, descriptors[b + 15] ?? -1));
    writeUvFitPair(descriptors, b + 42, uvFitScaleFor(linearLayerScales, descriptors[b + 22] ?? -1));
    writeUvFitPair(descriptors, b + 44, uvFitScaleFor(linearLayerScales, descriptors[b + 24] ?? -1));
    writeUvFitPair(descriptors, b + 46, uvFitScaleFor(linearLayerScales, descriptors[b + 25] ?? -1));
  }
}

/** Collect + dedup material texture sources and pack the per-material descriptors.
 *  Two index spaces: sRGB (baseColor/emissive) and linear (normal/ORM) — they
 *  upload to separate arrays so each is sampled in the correct colour space. */
export function collectMaterialTextures(materials: ReadonlyArray<MaterialSpec>): CollectedTextures {
  const sources: unknown[] = [];
  const linearSources: unknown[] = [];
  const makeIndexer = (list: unknown[]) => {
    const handleToIdx = new Map<unknown, number>();
    return (ref: TextureRef | undefined): number => {
      const handle = ref?.handle;
      if (handle == null) return -1;
      let i = handleToIdx.get(handle);
      if (i === undefined) {
        i = list.length;
        list.push(handle);
        handleToIdx.set(handle, i);
      }
      return i;
    };
  };
  const indexOf = makeIndexer(sources);        // sRGB array
  const indexOfLinear = makeIndexer(linearSources); // linear array

  // H51-B — once-warn when a material has both roughnessMap AND metallicMap set
  // to DISTINCT handles. pt-webgpu uses a single ORM texture slot (the glTF
  // combined roughness-metallic, G=roughness/B=metallic); when both maps are
  // provided and they differ, metallicMap is silently dropped (roughnessMap wins
  // via the ?? fallback). Warn once per scene-pack call so the host is aware.
  let warnedOrmSplit = false;

  const descriptors = new Float32Array(materials.length * MATERIAL_TEX_FLOAT_STRIDE);
  materials.forEach((m, mi) => {
    // H51-B: warn once on the first material with distinct roughnessMap + metallicMap.
    if (
      !warnedOrmSplit &&
      m.roughnessMap != null &&
      m.metallicMap != null &&
      m.roughnessMap.handle !== m.metallicMap.handle
    ) {
      warnedOrmSplit = true;
      console.warn(
        '[vitrum/pt-webgpu] A material provides separate roughnessMap and metallicMap ' +
          'pointing to different texture handles. pt-webgpu uses a single ORM slot ' +
          '(glTF combined roughness-metallic texture: G=roughness, B=metallic). ' +
          'roughnessMap is used; metallicMap is ignored. Supply a pre-combined ORM ' +
          'texture as roughnessMap (or as metallicMap when roughnessMap is absent) ' +
          'to include both channels.',
      );
    }
    const b = mi * MATERIAL_TEX_FLOAT_STRIDE;
    const bc = m.baseColorMap;
    descriptors[b + 0] = indexOf(bc);            // baseColorIdx (sRGB array)
    descriptors[b + 1] = indexOfLinear(m.normalMap);                 // normalIdx (linear array)
    descriptors[b + 2] = indexOfLinear(m.roughnessMap ?? m.metallicMap); // ormIdx (linear; glTF MR texture: G=rough, B=metal)
    descriptors[b + 3] = indexOf(m.emissiveMap); // emissiveIdx (sRGB array — same layers as baseColor)
    descriptors[b + 4] = ALPHA_MODE_INDEX[m.alphaMode ?? 'opaque'];
    descriptors[b + 5] = m.alphaCutoff ?? 0.5;
    descriptors[b + 6] = m.opacity ?? 1;
    descriptors[b + 7] = bc?.texCoord ?? 0;
    const t = bc?.transform;
    descriptors[b + 8] = t?.offset?.[0] ?? 0;
    descriptors[b + 9] = t?.offset?.[1] ?? 0;
    descriptors[b + 10] = t?.scale?.[0] ?? 1;
    descriptors[b + 11] = t?.scale?.[1] ?? 1;
    descriptors[b + 12] = t?.rotation ?? 0;
    // D3 — vec4 #3.yzw + #4.xyz: aoMap / lightMap / bumpMap (all LINEAR-space data:
    // occlusion factor, baked outgoing radiance, height field) routed through the
    // linear texture array. Index -1 when absent → the WGSL sampler is a no-op.
    descriptors[b + 13] = indexOfLinear(m.aoMap);
    descriptors[b + 14] = indexOfLinear(m.lightMap);
    descriptors[b + 15] = indexOfLinear(m.bumpMap);
    descriptors[b + 16] = m.aoMapIntensity ?? 1;
    descriptors[b + 17] = m.lightMapIntensity ?? 1;
    descriptors[b + 18] = m.bumpScale ?? 1;
    descriptors[b + 19] = m.envMapIntensity ?? 1;
    // D3 — vec4 #5: anisotropy scalars + optional KHR_materials_anisotropy map.
    descriptors[b + 20] = m.anisotropy ?? 0;
    descriptors[b + 21] = m.anisotropyRotation ?? 0;
    descriptors[b + 22] = indexOfLinear(m.anisotropyMap);
    descriptors[b + 23] = m.normalScale ?? 1;
    // Standalone alphaMap is coverage data (linear), sampled with the shared v1
    // material texture UV transform. BaseColor alpha still participates too.
    descriptors[b + 24] = indexOfLinear(m.alphaMap);
    descriptors[b + 25] = indexOfLinear(m.transmissionMap);
    descriptors[b + 26] = 0;
    descriptors[b + 27] = 0;
    writeDefaultUvFitPairs(descriptors, b);
  });

  return { sources, linearSources, descriptors };
}
