import {
  solveSkin,
  type PrimitiveUvSets,
  type SkinnedMeshPrimitive,
} from '@vitrum/core';

export interface SolvedSkinnedPrimitiveAttributes {
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly tangents?: Float32Array;
  readonly uvs?: Float32Array;
  readonly uv1?: Float32Array;
  readonly uvSets?: PrimitiveUvSets;
}

const IDENTITY_MATRIX = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

export function hasActiveMorphTargets(
  primitive: SkinnedMeshPrimitive,
): boolean {
  if (
    primitive.morphTargets == null ||
    primitive.morphTargets.length === 0 ||
    primitive.morphWeights == null
  ) {
    return false;
  }
  return primitive.morphWeights.some((weight) => weight !== 0);
}

/**
 * Resolve the render attributes of a skinned primitive.
 *
 * A zero-bone primitive is a valid rest-pose/morph-only carrier in the existing
 * ingestion contract. When its morph weights are active, synthesize one identity
 * influence per vertex and reuse core's canonical morph implementation instead
 * of silently dropping the morph streams.
 */
export function solveSkinnedPrimitive(
  primitive: SkinnedMeshPrimitive,
): SolvedSkinnedPrimitiveAttributes {
  if (primitive.bones.length > 0) return solveSkin(primitive);
  if (!hasActiveMorphTargets(primitive)) {
    return {
      positions: primitive.positions,
      normals: primitive.normals,
      ...(primitive.tangents != null ? { tangents: primitive.tangents } : {}),
      ...(primitive.uvs != null ? { uvs: primitive.uvs } : {}),
      ...(primitive.uv1 != null ? { uv1: primitive.uv1 } : {}),
      ...(primitive.uvSets != null ? { uvSets: primitive.uvSets } : {}),
    };
  }
  if (primitive.boneInverses.length !== 0) {
    throw new Error(
      'solveSkinnedPrimitive: zero-bone primitive must have zero boneInverses.',
    );
  }

  const vertexCount = primitive.positions.length / 3;
  const influencesPerVertex = primitive.skinInfluencesPerVertex ?? 4;
  if (!Number.isSafeInteger(influencesPerVertex) || influencesPerVertex <= 0) {
    throw new Error(
      `solveSkinnedPrimitive: skinInfluencesPerVertex must be a positive safe integer ` +
      `(got ${String(influencesPerVertex)}).`,
    );
  }
  const skinIndices = new Uint32Array(vertexCount * influencesPerVertex);
  const skinWeights = new Float32Array(vertexCount * influencesPerVertex);
  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    skinWeights[vertex * influencesPerVertex] = 1;
  }
  return solveSkin({
    ...primitive,
    skinIndices,
    skinWeights,
    skinInfluencesPerVertex: influencesPerVertex,
    bones: IDENTITY_MATRIX,
    boneInverses: IDENTITY_MATRIX,
  });
}
