/**
 * Per-engine vertex / index / color packing helpers for ReSTIR.
 *
 * Sits on top of `@vitrum/shared-bvh`'s `buildSceneBVH` shared-core output
 * and re-packs the layouts ReSTIR's WGSL shaders consume:
 *   - position.w slot ← packed UV pair
 *   - bvhIndex.w slot ← RGBA8 baseColor | (trans4 | texType4)
 *   - bvh_beer    u32 ← Beer-Lambert visible color per triangle
 *
 * Extracted from `bvhCompute.ts` (was ~270 lines of inline packing).
 */

import * as THREE from 'three';

/** Default warm-gray fallback color (sRGB byte values) when a triangle has
 *  no material or unrecognised material type. Matches the old in-file
 *  `WARM_GRAY_DEFAULT_*` constants. File-local — no external consumers
 *  (2026-05-18 dead-code sweep). */
const WARM_GRAY_DEFAULT_R = 153;
const WARM_GRAY_DEFAULT_G = 148;
const WARM_GRAY_DEFAULT_B = 140;

/**
 * Apply Beer-Lambert absorption to an attenuation color given a sample
 * thickness / attenuation-distance pair. Returns the input color unchanged
 * if any required parameter is missing or non-finite.
 *
 * File-local — no external consumers (2026-05-18 dead-code sweep).
 */
function applyBeerLambert(
  attCol: THREE.Color,
  thickness: number | undefined,
  attDist: number | undefined,
): THREE.Color {
  if (thickness === undefined || attDist === undefined) return attCol;
  if (!Number.isFinite(thickness) || !Number.isFinite(attDist)) return attCol;
  if (thickness <= 0 || attDist <= 0) return attCol;
  const k = thickness / attDist;
  return new THREE.Color(
    Math.pow(Math.max(1e-6, attCol.r), k),
    Math.pow(Math.max(1e-6, attCol.g), k),
    Math.pow(Math.max(1e-6, attCol.b), k),
  );
}

/**
 * Pack UV (16-bit unorm pair) into the .w slot of every vec4f position.
 * See bvhCompute.ts for the rationale (single storage buffer per stage).
 */
export function packUVIntoPositionW(
  positions: Float32Array,
  uvAttr: THREE.BufferAttribute | undefined,
  vertCount: number,
): Float32Array<ArrayBuffer> {
  const out = new Float32Array(positions.length);
  out.set(positions);
  const u32View = new Uint32Array(out.buffer);

  const sourceUvs = uvAttr
    ? new Float32Array(uvAttr.array)
    : new Float32Array(vertCount * 2);

  for (let i = 0; i < vertCount; i++) {
    let u = sourceUvs[i * 2 + 0]!;
    let v = sourceUvs[i * 2 + 1]!;
    u = u - Math.floor(u);
    v = v - Math.floor(v);
    const u16 = Math.min(0xFFFF, Math.max(0, Math.round(u * 0xFFFF))) & 0xFFFF;
    const v16 = Math.min(0xFFFF, Math.max(0, Math.round(v * 0xFFFF))) & 0xFFFF;
    u32View[i * 4 + 3] = (v16 << 16) | u16;
  }
  return out;
}

/**
 * Resolve a triangle's RGB color for packing. Shared between packBVHIndexW
 * (raw attenuation color) and packBVHBeerColors (Beer-Lambert tinted).
 */
function resolveTriColor(mat: THREE.Material, applyBeer: boolean): THREE.Color {
  const physMat = mat as THREE.MeshPhysicalMaterial;
  const stdMat  = mat as THREE.MeshStandardMaterial;
  const transmission = (physMat.transmission ?? 0);
  const isTransmissive = transmission > 0.01;
  const attenColor = (physMat as { attenuationColor?: THREE.Color }).attenuationColor;
  if (isTransmissive && attenColor) {
    if (applyBeer) {
      return applyBeerLambert(
        attenColor,
        (physMat as { thickness?: number }).thickness,
        (physMat as { attenuationDistance?: number }).attenuationDistance,
      );
    }
    return attenColor;
  }
  return physMat.color ?? stdMat?.color ?? new THREE.Color(0.6, 0.58, 0.55);
}

/** Pack one triangle's index lanes + material byte into an existing vec4u buffer. */
export function packBVHIndexWTri(
  indexBuf: Uint32Array,
  indices: Uint32Array,
  triMaterialId: Uint32Array,
  materials: readonly THREE.Material[],
  tri: number,
): void {
  const base4 = tri * 4;
  indexBuf[base4 + 0] = indices[tri * 3 + 0]!;
  indexBuf[base4 + 1] = indices[tri * 3 + 1]!;
  indexBuf[base4 + 2] = indices[tri * 3 + 2]!;

  const matId = triMaterialId[tri]!;
  const mat = materials[matId];
  let r = WARM_GRAY_DEFAULT_R, g = WARM_GRAY_DEFAULT_G, b = WARM_GRAY_DEFAULT_B;
  let transmission = 0;
  let texTypeId = 0;
  let isMetal = 0;
  if (mat) {
    const physMat = mat as THREE.MeshPhysicalMaterial;
    const stdMat  = mat as THREE.MeshStandardMaterial;
    transmission = (physMat.transmission ?? 0);
    const color = resolveTriColor(mat, /* applyBeer */ false);
    r = Math.round(color.r * 255) & 0xFF;
    g = Math.round(color.g * 255) & 0xFF;
    b = Math.round(color.b * 255) & 0xFF;
    const surfTex = (mat.userData as { surfaceTextureId?: number } | undefined)?.surfaceTextureId;
    texTypeId = (typeof surfTex === 'number' ? surfTex : 0) & 0x7;
    const metalness = (stdMat?.metalness ?? 0);
    isMetal = metalness > 1e-4 ? 1 : 0;
  }
  const trans4 = Math.min(15, Math.round(transmission * 15)) & 0xF;
  const lowByte = ((trans4 << 4) | (isMetal << 3) | (texTypeId & 0x7)) & 0xFF;
  indexBuf[base4 + 3] = (r << 24) | (g << 16) | (b << 8) | lowByte;
}

/** Pack one triangle's Beer-Lambert visible color into a parallel u32 buffer. */
export function packBVHBeerColorTri(
  beerBuf: Uint32Array,
  triMaterialId: Uint32Array,
  materials: readonly THREE.Material[],
  tri: number,
): void {
  const matId = triMaterialId[tri]!;
  const mat = materials[matId];
  let r = WARM_GRAY_DEFAULT_R, g = WARM_GRAY_DEFAULT_G, b = WARM_GRAY_DEFAULT_B;
  if (mat) {
    const color = resolveTriColor(mat, /* applyBeer */ true);
    r = Math.round(Math.min(1, color.r) * 255) & 0xFF;
    g = Math.round(Math.min(1, color.g) * 255) & 0xFF;
    b = Math.round(Math.min(1, color.b) * 255) & 0xFF;
  }
  beerBuf[tri] = (r << 24) | (g << 16) | (b << 8);
}

/**
 * Re-pack `bvhIndex.w` and `bvh_beer` for a contiguous triangle subrange.
 * Used by the material-only `updatePrimitive` fast path.
 */
export function repackBVHMaterialRange(
  indexBuf: Uint32Array,
  beerBuf: Uint32Array,
  indices: Uint32Array,
  triMaterialId: Uint32Array,
  materials: readonly THREE.Material[],
  triStart: number,
  triCount: number,
): void {
  const triEnd = triStart + triCount;
  for (let t = triStart; t < triEnd; t++) {
    packBVHIndexWTri(indexBuf, indices, triMaterialId, materials, t);
    packBVHBeerColorTri(beerBuf, triMaterialId, materials, t);
  }
}

/**
 * Pack vertex indices + RGBA8 baseColor + (trans4|texType4) into vec4u
 * per-triangle (4 u32 = 16 bytes per triangle).
 */
export function packBVHIndexW(
  indices: Uint32Array,
  triMaterialId: Uint32Array,
  materials: THREE.Material[],
  triCount: number,
): Uint32Array<ArrayBuffer> {
  const indexBuf = new Uint32Array(triCount * 4);

  for (let t = 0; t < triCount; t++) {
    packBVHIndexWTri(indexBuf, indices, triMaterialId, materials, t);
  }
  return indexBuf;
}

/**
 * Pack the Beer-Lambert visible color per triangle into a parallel u32 buffer.
 * Read by shade.wgsl Lo_emit on a primary glass hit.
 */
export function packBVHBeerColors(
  triMaterialId: Uint32Array,
  materials: THREE.Material[],
  triCount: number,
): Uint32Array<ArrayBuffer> {
  const beerBuf = new Uint32Array(triCount);
  for (let t = 0; t < triCount; t++) {
    packBVHBeerColorTri(beerBuf, triMaterialId, materials, t);
  }
  return beerBuf;
}

/**
 * Emissive radiance Le (HDR, `emissive.rgb · emissiveIntensity`) of a THREE
 * material, or `null` when the material is not a self-emissive surface. This
 * mirrors the EMISSIVE branch of `classifyTriangleEmitter` (emitterList.ts) EXACTLY
 * — so the camera-visible glow Le equals the radiance ReSTIR-DI samples for that
 * emitter — but deliberately EXCLUDES the transmissive "sun-attenuated secondary
 * emitter" branch: glass self-emission to the camera is already handled by
 * shade.wgsl `lo_emit` (Beer-Lambert), so packing it here would double-count.
 */
export function materialEmissiveLe(mat: THREE.Material): [number, number, number] | null {
  const meshMat = mat as THREE.MeshStandardMaterial;
  const em = meshMat.emissive;
  if (!em) return null;
  const ei = meshMat.emissiveIntensity;
  if (!(ei && ei > 0)) return null;
  if (em.r <= 0 && em.g <= 0 && em.b <= 0) return null;
  return [em.r * ei, em.g * ei, em.b * ei];
}

/**
 * Pack per-triangle emissive radiance Le into a parallel rgba32float buffer
 * (stride 4: rgb + 0 pad). Read by shade.wgsl `lo_emitterGlow` on a primary hit
 * so emissive-mesh surfaces are CAMERA-VISIBLE (the real-time analogue of the
 * pt-webgpu camera-visible-emitters fix). Non-emissive triangles are zero. HDR
 * (emissiveIntensity may exceed 1), hence float — not the LDR `bvh_beer` u32.
 */
export function packBVHEmissiveLe(
  triMaterialId: Uint32Array,
  materials: readonly THREE.Material[],
  triCount: number,
): Float32Array<ArrayBuffer> {
  const out = new Float32Array(triCount * 4);
  for (let t = 0; t < triCount; t++) {
    const mat = materials[triMaterialId[t]!];
    if (!mat) continue;
    const le = materialEmissiveLe(mat);
    if (le == null) continue;
    out[t * 4 + 0] = le[0];
    out[t * 4 + 1] = le[1];
    out[t * 4 + 2] = le[2];
  }
  return out;
}
