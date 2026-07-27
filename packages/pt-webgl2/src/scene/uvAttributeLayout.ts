import {
  getPrimitiveUvSet,
  sparseArrayOwnIndices,
  type MaterialSpec,
  type Scene,
  type ScenePrimitive,
  type TextureRef,
} from '@vitrum/core';
import type { WorldSpaceMergeResult } from '@vitrum/shared-bvh';
import { MATERIAL_MAP_FIELD_ORDER } from '../glsl/shader/structs/materialStride.js';

/** Fixed compatibility lanes retained from the original five-layer payload. */
export const ATTR_LAYER_NORMAL = 0;
export const ATTR_LAYER_TANGENT = 1;
export const ATTR_LAYER_UV = 2;
export const ATTR_LAYER_COLOR = 3;
export const ATTR_LAYER_UV1 = 4;
export const ATTR_FIXED_LAYER_COUNT = 5;

/**
 * Dense scene-local addressing for arbitrary authored TextureRef.texCoord ids.
 *
 * UV0 and UV1 retain their historical layers (2 and 4). Every other distinct
 * non-negative texCoord used or supplied by the scene receives one dense layer
 * starting at 5. Consequently a sparse authored id such as TEXCOORD_10000 costs
 * one array layer rather than 9,999 empty layers.
 */
export interface UvAttributeLayout {
  readonly layerByTexCoord: ReadonlyMap<number, number>;
  /** Merged stride-2 streams keyed by authored texCoord. Includes 0 and 1. */
  readonly mergedByTexCoord: ReadonlyMap<number, Float32Array>;
  /** Streams for dense attribute layers 5..N, in layer order. */
  readonly extraUvLayers: readonly Float32Array[];
  readonly layerCount: number;
}

type MeshLikePrimitive = Extract<
  ScenePrimitive,
  { positions: Float32Array; material: MaterialSpec }
>;

function isMeshLikePrimitive(primitive: ScenePrimitive): primitive is MeshLikePrimitive {
  return primitive.kind === 'mesh' ||
    primitive.kind === 'instanced-mesh' ||
    primitive.kind === 'skinned-mesh';
}

function textureRefForField(material: MaterialSpec, field: string): TextureRef | undefined {
  return material[field as keyof MaterialSpec] as TextureRef | undefined;
}

function addTextureRefTexCoord(ids: Set<number>, ref: TextureRef | undefined): void {
  if (ref == null) return;
  const texCoord = ref.texCoord ?? 0;
  if (!Number.isSafeInteger(texCoord) || texCoord < 0) {
    throw new RangeError(
      `pt-webgl2: TextureRef.texCoord must be a non-negative safe integer (got ${String(texCoord)})`,
    );
  }
  ids.add(texCoord);
}

/** All texCoord ids referenced by the 21 mapped-rich slots plus layer normals. */
export function collectMaterialTexCoords(materials: readonly MaterialSpec[]): ReadonlySet<number> {
  const ids = new Set<number>([0, 1]);
  for (const material of materials) {
    for (const field of MATERIAL_MAP_FIELD_ORDER) {
      addTextureRefTexCoord(ids, textureRefForField(material, field));
    }
    addTextureRefTexCoord(ids, material.frontLayer?.normalMap);
    addTextureRefTexCoord(ids, material.backLayer?.normalMap);
  }
  return ids;
}

function collectPrimitiveTexCoords(scene: Scene, ids: Set<number>): void {
  for (const primitive of scene.primitives) {
    if (!isMeshLikePrimitive(primitive)) continue;
    if (primitive.uvs != null) ids.add(0);
    if (primitive.uv1 != null) ids.add(1);
    const uvSets = primitive.uvSets;
    if (uvSets == null) continue;
    // Work scales with supplied streams rather than the highest authored
    // texCoord array index, including ordinary-property indices >= 2^32-1.
    for (const texCoord of sparseArrayOwnIndices(uvSets)) {
      if (uvSets[texCoord] != null) {
        ids.add(texCoord);
      }
    }
  }
}

function copyPrimitiveUvRange(
  destination: Float32Array,
  source: Float32Array | undefined,
  vertexStart: number,
  vertexCount: number,
  localVertexCount: number,
): void {
  if (source == null) return;
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const local = Math.min(vertex, Math.max(localVertexCount - 1, 0));
    const src = local * 2;
    const dst = (vertexStart + vertex) * 2;
    destination[dst] = source[src] ?? 0;
    destination[dst + 1] = source[src + 1] ?? 0;
  }
}

/** Build dense UV layers aligned exactly to mergeWorldSpaceFromCore's vertices. */
export function buildUvAttributeLayout(
  scene: Scene,
  merged: Pick<
    WorldSpaceMergeResult,
    'uvs' | 'meshVertexRanges' | 'vertexCount'
  >,
  materials: readonly MaterialSpec[],
): UvAttributeLayout {
  const texCoords = new Set<number>(collectMaterialTexCoords(materials));
  collectPrimitiveTexCoords(scene, texCoords);

  const extras = [...texCoords]
    .filter((texCoord) => texCoord >= 2)
    .sort((a, b) => a - b);
  const layerByTexCoord = new Map<number, number>([
    [0, ATTR_LAYER_UV],
    [1, ATTR_LAYER_UV1],
  ]);
  for (let i = 0; i < extras.length; i += 1) {
    layerByTexCoord.set(extras[i]!, ATTR_FIXED_LAYER_COUNT + i);
  }

  const mergedByTexCoord = new Map<number, Float32Array>();
  const uv0 = merged.uvs ?? new Float32Array(merged.vertexCount * 2);
  mergedByTexCoord.set(0, uv0);
  // UV1 retains its compatibility fallback to UV0. Higher sets deliberately
  // default to zero where a primitive does not supply that authored stream.
  const uv1 = new Float32Array(merged.vertexCount * 2);
  const extraStreams = new Map<number, Float32Array>();
  for (const texCoord of extras) {
    extraStreams.set(texCoord, new Float32Array(merged.vertexCount * 2));
  }

  const primitiveById = new Map(
    scene.primitives
      .filter(isMeshLikePrimitive)
      .map((primitive) => [String(primitive.id), primitive] as const),
  );
  // The merge can omit a primitive/instance whose triangles are all invalid.
  // Join through the producer's explicit provenance rather than assuming range
  // ordinal == source ordinal; that assumption silently shifted every later UV.
  for (const range of merged.meshVertexRanges) {
    const primitive = primitiveById.get(String(range.sourcePrimitiveId ?? range.name));
    if (primitive == null) continue;
    const localVertexCount = Math.floor(primitive.positions.length / 3);
    copyPrimitiveUvRange(
      uv1,
      getPrimitiveUvSet(primitive, 1) ?? getPrimitiveUvSet(primitive, 0),
      range.vertexStart,
      range.vertexCount,
      localVertexCount,
    );
    for (const texCoord of extras) {
      copyPrimitiveUvRange(
        extraStreams.get(texCoord)!,
        getPrimitiveUvSet(primitive, texCoord),
        range.vertexStart,
        range.vertexCount,
        localVertexCount,
      );
    }
  }

  mergedByTexCoord.set(1, uv1);
  for (const texCoord of extras) {
    mergedByTexCoord.set(texCoord, extraStreams.get(texCoord)!);
  }
  return {
    layerByTexCoord,
    mergedByTexCoord,
    extraUvLayers: extras.map((texCoord) => extraStreams.get(texCoord)!),
    layerCount: ATTR_FIXED_LAYER_COUNT + extras.length,
  };
}

export function sameUvAttributeLayout(
  a: ReadonlyMap<number, number> | null | undefined,
  b: ReadonlyMap<number, number>,
): boolean {
  if (a == null || a.size !== b.size) return false;
  for (const [texCoord, layer] of b) {
    if (a.get(texCoord) !== layer) return false;
  }
  return true;
}
