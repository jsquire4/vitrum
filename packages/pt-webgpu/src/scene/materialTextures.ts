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

/**
 * vec4s per material in the descriptor buffer (MUST match the future WGSL
 * `MATERIAL_TEX_VEC4_STRIDE`):
 *   0: {baseColorIdx, normalIdx, ormIdx, emissiveIdx}   (-1 = no map)
 *   1: {alphaMode (0 opaque/1 mask/2 blend), alphaCutoff, opacity, texCoord}
 *   2: {offsetX, offsetY, scaleX, scaleY}                (baseColor UV transform)
 *   3: {rotation, _pad, _pad, _pad}
 */
export const MATERIAL_TEX_VEC4_STRIDE = 4;
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
  });

  return { sources, linearSources, descriptors };
}
