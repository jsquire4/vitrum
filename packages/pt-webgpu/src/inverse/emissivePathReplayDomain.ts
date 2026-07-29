import type {
  InverseSessionDiagnostic,
  MaterialSpec,
  Scene,
  ScenePrimitive,
} from '@vitrum/core';
import {
  invertMat4,
  mergeWorldSpaceFromCore,
} from '@vitrum/shared-bvh';
import { applySolveSkinToScene } from '../scene/uploadSceneBuffers.js';

export interface EmissivePathReplayDomainIssue {
  readonly code: InverseSessionDiagnostic['code'];
  readonly message: string;
  readonly details: Record<
    string,
    string | number | boolean | readonly string[]
  >;
}

export function isTriangleBackedForEmissiveReplay(
  primitive: ScenePrimitive,
): boolean {
  return (
    primitive.kind === 'mesh' ||
    primitive.kind === 'skinned-mesh' ||
    primitive.kind === 'instanced-mesh'
  );
}

export function emissiveReplayPrimitiveIssue(
  primitive: ScenePrimitive,
): EmissivePathReplayDomainIssue | null {
  if (primitive.kind === 'analytic') {
    return {
      code: 'path-replay-unsupported-primitive',
      message: `analytic primitive "${primitive.id}" has no exact triangle replay`,
      details: {
        primitiveId: primitive.id,
        primitiveKind: primitive.kind,
        analyticShape: primitive.shape,
      },
    };
  }
  if (!isTriangleBackedForEmissiveReplay(primitive)) {
    return {
      code: 'path-replay-unsupported-primitive',
      message: `primitive "${primitive.id}" is not triangle-backed`,
      details: {
        primitiveId: primitive.id,
        primitiveKind: (primitive as { readonly kind: string }).kind,
      },
    };
  }
  if (
    primitive.kind === 'instanced-mesh' &&
    primitive.instances.length === 0
  ) {
    return {
      code: 'path-replay-unsupported-scene-geometry',
      message: `instanced primitive "${primitive.id}" has no instances`,
      details: {
        primitiveId: primitive.id,
        primitiveKind: primitive.kind,
        instanceCount: 0,
      },
    };
  }
  const transforms =
    primitive.kind === 'instanced-mesh'
      ? primitive.instances
      : primitive.transform == null
        ? []
        : [primitive.transform];
  for (let index = 0; index < transforms.length; index += 1) {
    const transform = transforms[index]!;
    if (
      !Array.from(transform).every(Number.isFinite) ||
      invertMat4(transform) == null
    ) {
      return {
        code: 'path-replay-unsupported-scene-geometry',
        message:
          `primitive "${primitive.id}" has a non-invertible replay transform`,
        details: {
          primitiveId: primitive.id,
          primitiveKind: primitive.kind,
          transformIndex: index,
          feature: 'singular-transform',
        },
      };
    }
  }
  return null;
}

function unsupportedMaterialTransport(
  material: MaterialSpec,
): { readonly reason: string; readonly feature: string } | null {
  if (material.shadingModel !== 'unlit') {
    return {
      reason:
        'lit shading can receive implicit emissive-mesh NEE that the primary-hit identity does not differentiate',
      feature: 'lit-receiver',
    };
  }
  if ((material.alphaMode ?? 'opaque') !== 'opaque') {
    return {
      reason: 'non-opaque alpha changes primary-hit visibility',
      feature: 'alpha-visibility',
    };
  }
  if ((material.transmission ?? 0) > 0 || material.transmissionMap != null) {
    return {
      reason: 'transmission changes camera transport',
      feature: 'transmission',
    };
  }
  if (
    material.thicknessMap != null ||
    (material.thickness ?? 0) > 0 ||
    material.spectralAttenuation != null ||
    material.dispersionAbbeNumber != null
  ) {
    return {
      reason: 'volume, spectral attenuation, or dispersion changes transport',
      feature: 'volume-or-spectral-transport',
    };
  }
  if (
    (material.scatteringCoefficient ?? 0) > 0 ||
    material.scatteringCoefficientRGB != null
  ) {
    return {
      reason: 'participating-medium scattering changes transport',
      feature: 'scattering',
    };
  }
  if (
    material.frontLayer != null ||
    material.backLayer != null ||
    (material.thinFilmStack?.layers.length ?? 0) > 0
  ) {
    return {
      reason: 'layered material transport is outside the primary-hit identity',
      feature: 'material-layers',
    };
  }
  if (material.displacementMap != null) {
    return {
      reason: 'displacement can change replayed geometry',
      feature: 'displacement',
    };
  }
  if (
    material.extensions != null &&
    Object.keys(material.extensions).length > 0
  ) {
    return {
      reason: 'opaque material extensions have no certified replay semantics',
      feature: 'material-extensions',
    };
  }
  return null;
}

interface ReplayRangeBounds {
  readonly primitiveId: string;
  readonly instanceIndex: number;
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
}

/**
 * The forward BVH is free to visit equal-distance leaves in a different order
 * from the adjoint's brute-force triangle loop. Within one primitive that tie
 * is harmless because every triangle has the same optimized material slot.
 * Across primitives it can change which material receives the derivative.
 *
 * Reject any cross-primitive world-space AABB overlap. This is intentionally
 * conservative: disjoint AABBs prove that two primitives cannot share a hit
 * point, while overlapping bounds merely mean the dangerous tie cannot be
 * excluded without replaying every possible camera ray.
 */
function ambiguousCrossPrimitiveHitIssue(
  scene: Scene,
): EmissivePathReplayDomainIssue | null {
  if (scene.primitives.length < 2) return null;

  let merged: ReturnType<typeof mergeWorldSpaceFromCore>;
  try {
    merged = mergeWorldSpaceFromCore(applySolveSkinToScene(scene), {
      positionStride: 4,
      filter: isTriangleBackedForEmissiveReplay,
    });
  } catch (error) {
    return {
      code: 'path-replay-unsupported-scene-geometry',
      message:
        'world-space replay geometry could not be proven before equal-distance tie validation',
      details: {
        feature: 'replay-geometry-validation',
        cause: error instanceof Error ? error.message : String(error),
      },
    };
  }

  const bounds: ReplayRangeBounds[] = [];
  for (const range of merged.meshVertexRanges) {
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    const end = range.vertexStart + range.vertexCount;
    for (let vertex = range.vertexStart; vertex < end; vertex += 1) {
      const offset = vertex * merged.positionStrideFloats;
      const x = merged.positions[offset]!;
      const y = merged.positions[offset + 1]!;
      const z = merged.positions[offset + 2]!;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      minZ = Math.min(minZ, z);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      maxZ = Math.max(maxZ, z);
    }
    if (range.vertexCount > 0) {
      bounds.push({
        primitiveId: range.sourcePrimitiveId ?? range.name,
        instanceIndex: range.sourceInstanceIndex ?? 0,
        min: [minX, minY, minZ],
        max: [maxX, maxY, maxZ],
      });
    }
  }

  for (let left = 0; left < bounds.length; left += 1) {
    const a = bounds[left]!;
    for (let right = left + 1; right < bounds.length; right += 1) {
      const b = bounds[right]!;
      if (a.primitiveId === b.primitiveId) continue;
      const overlap =
        a.min[0] <= b.max[0] && a.max[0] >= b.min[0] &&
        a.min[1] <= b.max[1] && a.max[1] >= b.min[1] &&
        a.min[2] <= b.max[2] && a.max[2] >= b.min[2];
      if (!overlap) continue;
      return {
        code: 'path-replay-unsupported-scene-geometry',
        message:
          `world-space bounds for primitives "${a.primitiveId}" and ` +
          `"${b.primitiveId}" overlap, so an equal-distance material tie ` +
          'between forward BVH order and adjoint triangle order cannot be excluded',
        details: {
          feature: 'ambiguous-equal-distance-hit',
          primitiveIds: [a.primitiveId, b.primitiveId],
          leftInstanceIndex: a.instanceIndex,
          rightInstanceIndex: b.instanceIndex,
        },
      };
    }
  }
  return null;
}

export function emissiveReplaySceneIssue(
  scene: Scene,
): EmissivePathReplayDomainIssue | null {
  for (const primitive of scene.primitives) {
    const primitiveIssue = emissiveReplayPrimitiveIssue(primitive);
    if (primitiveIssue != null) return primitiveIssue;

    const materialIssue = unsupportedMaterialTransport(primitive.material);
    if (materialIssue != null) {
      return {
        code:
          materialIssue.feature === 'alpha-visibility'
            ? 'path-replay-unsupported-visibility'
            : materialIssue.feature === 'displacement'
              ? 'path-replay-unsupported-geometry'
              : materialIssue.feature === 'lit-receiver'
                ? 'path-replay-unsupported-receiver'
                : 'path-replay-unsupported-transport',
        message:
          `primitive "${primitive.id}" uses ${materialIssue.reason}`,
        details: {
          primitiveId: primitive.id,
          feature: materialIssue.feature,
        },
      };
    }
  }
  return ambiguousCrossPrimitiveHitIssue(scene);
}

export function emissiveReplayTargetIssue(
  scene: Scene,
  primitive: ScenePrimitive,
): EmissivePathReplayDomainIssue | null {
  if (primitive.material.emissiveMap != null) {
    return {
      code: 'path-replay-unsupported-material',
      message:
        `primitive "${primitive.id}" uses an emissive map outside the certified spatially constant emission domain`,
      details: {
        primitiveId: primitive.id,
        feature: 'emissive-map',
      },
    };
  }

  const material = primitive.material;
  if (
    (material.clearcoat ?? 0) > 0 ||
    material.clearcoatMap != null ||
    material.clearcoatRoughnessMap != null ||
    material.clearcoatNormalMap != null
  ) {
    return {
      code: 'path-replay-unsupported-material',
      message:
        `primitive "${primitive.id}" uses clearcoat emission attenuation`,
      details: {
        primitiveId: primitive.id,
        feature: 'clearcoat',
      },
    };
  }

  const foldedEmitter = scene.emitters.find(
    (emitter) =>
      emitter.kind === 'mesh-area' &&
      emitter.meshId === primitive.id,
  );
  if (foldedEmitter != null) {
    return {
      code: 'path-replay-unsupported-emitter',
      message:
        `mesh-area emitter "${foldedEmitter.id}" folds primitive "${primitive.id}" emission outside the material derivative`,
      details: {
        primitiveId: primitive.id,
        emitterId: foldedEmitter.id,
        emitterKind: foldedEmitter.kind,
      },
    };
  }

  const emissiveIntensity = material.emissiveIntensity ?? 1;
  if (!Number.isFinite(emissiveIntensity) || emissiveIntensity < 0) {
    return {
      code: 'path-replay-unsupported-material',
      message:
        `primitive "${primitive.id}" has a non-finite or negative emissive intensity`,
      details: {
        primitiveId: primitive.id,
        feature: 'emissive-intensity',
      },
    };
  }
  return null;
}
