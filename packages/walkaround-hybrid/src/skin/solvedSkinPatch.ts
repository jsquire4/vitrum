import {
  sparseArrayOwnIndices,
  type SkinnedMeshPrimitive,
} from '@vitrum/core';

export interface SolvedSkinStreams {
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly tangents?: Float32Array;
  readonly uvs?: Float32Array;
  readonly uv1?: Float32Array;
  readonly uvSets?: SkinnedMeshPrimitive['uvSets'];
  readonly colors?: Float32Array;
  readonly colorSets?: SkinnedMeshPrimitive['colorSets'];
}

function hasDefinedMorphLane(
  lanes:
    | SkinnedMeshPrimitive['morphTargetUvSets']
    | SkinnedMeshPrimitive['morphTargetColorSets'],
): boolean {
  if (lanes == null) return false;
  for (const index of sparseArrayOwnIndices(lanes)) {
    const targets = lanes[index];
    if (targets != null && targets.length > 0) return true;
  }
  return false;
}

/** True when a non-zero morph can replace a renderer-owned UV/color stream. */
export function hasMorphControlledRenderStreams(
  primitive: SkinnedMeshPrimitive,
): boolean {
  return (
    (primitive.morphTargetUvs?.length ?? 0) > 0 ||
    (primitive.morphTargetUv1s?.length ?? 0) > 0 ||
    hasDefinedMorphLane(primitive.morphTargetUvSets) ||
    (primitive.morphTargetColors?.length ?? 0) > 0 ||
    hasDefinedMorphLane(primitive.morphTargetColorSets)
  );
}

/** True when at least one authored morph influence is currently non-zero. */
export function hasActiveSkinMorph(
  primitive: SkinnedMeshPrimitive,
): boolean {
  return (
    (primitive.morphTargets?.length ?? 0) > 0 &&
    primitive.morphWeights != null &&
    primitive.morphWeights.some((weight) => weight !== 0)
  );
}

function legacyAliasNeedsRestore(
  rendered: Float32Array | undefined,
  authoredAlias: Float32Array | undefined,
  authoredSetLane: Float32Array | undefined,
): boolean {
  if (authoredAlias != null) return rendered !== authoredAlias;
  // A lane-only authored primitive begins without the legacy alias, while an
  // accepted restore may retain the base lane as a coherent alias. Both are
  // clean steady states; a morphed allocation is neither reference.
  return rendered != null && rendered !== authoredSetLane;
}

/**
 * True when the persistent renderer scene still owns UV/color allocations
 * produced by a prior non-zero morph solve.
 *
 * Identity is intentional: an accepted restore republishes the authored set
 * containers and aliases. This makes the first active→inactive transition
 * (zero weights or cleared definitions) rebuild the attribute payload once,
 * while subsequent inactive skin ticks remain on the position/normal refit
 * path.
 */
export function needsAuthoredMorphStreamRestore(
  authored: SkinnedMeshPrimitive,
  rendered: SkinnedMeshPrimitive | null,
): boolean {
  if (rendered == null) return false;

  // Do not gate these comparisons on the CURRENT morph definitions. A valid
  // animation edit can clear those definitions in the same frame that must
  // retire arrays derived from the preceding active morph.
  if (rendered.uvSets !== authored.uvSets) return true;
  if (
    legacyAliasNeedsRestore(
      rendered.uvs,
      authored.uvs,
      authored.uvSets?.[0],
    )
  ) {
    return true;
  }
  if (
    legacyAliasNeedsRestore(
      rendered.uv1,
      authored.uv1,
      authored.uvSets?.[1],
    )
  ) {
    return true;
  }

  if (rendered.colorSets !== authored.colorSets) return true;
  return legacyAliasNeedsRestore(
    rendered.colors,
    authored.colors,
    authored.colorSets?.[0],
  );
}

/**
 * Build the renderer-facing streams for one solved pose.
 *
 * `solveSkin()` deliberately omits UV/color outputs when every morph weight is
 * zero. That is allocation-efficient for a fresh scene, but a persistent
 * render scene may still own arrays from the preceding non-zero morph frame,
 * including when the next patch removes the morph definitions themselves.
 * `restoreAuthoredMorphStreams` republishes the complete authored base stream
 * surface in that transition so no prior morphed stream can remain live.
 */
export function solvedSkinRenderPatch(
  primitive: SkinnedMeshPrimitive,
  solved: SolvedSkinStreams,
  restoreAuthoredMorphStreams = false,
): Partial<SkinnedMeshPrimitive> {
  const uvSets = solved.uvSets ??
    (restoreAuthoredMorphStreams
      ? primitive.uvSets
      : undefined);
  const uvs = solved.uvs ??
    (restoreAuthoredMorphStreams
      ? (primitive.uvs ?? primitive.uvSets?.[0])
      : undefined);
  const uv1 = solved.uv1 ??
    (restoreAuthoredMorphStreams
      ? (primitive.uv1 ?? primitive.uvSets?.[1])
      : undefined);
  const colorSets = solved.colorSets ??
    (restoreAuthoredMorphStreams
      ? primitive.colorSets
      : undefined);
  const colors = solved.colors ??
    (restoreAuthoredMorphStreams
      ? (primitive.colors ?? primitive.colorSets?.[0])
      : undefined);
  return {
    positions: solved.positions,
    normals: solved.normals,
    ...(solved.tangents != null ? { tangents: solved.tangents } : {}),
    ...(uvs != null ? { uvs } : {}),
    ...(uv1 != null ? { uv1 } : {}),
    ...(uvSets != null ? { uvSets } : {}),
    ...(colors != null ? { colors } : {}),
    ...(colorSets != null ? { colorSets } : {}),
  };
}
