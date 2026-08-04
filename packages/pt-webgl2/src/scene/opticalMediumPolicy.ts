import type { Scene } from '@vitrum/core';
import {
  assertOpticalMediumTopology,
  lowerTransmissiveAnalyticPrimitives,
  materialDefinesBulkOpticalMedium,
  type OpticalMediumTopologyAnalysis,
  type WorldSpaceMergeResult,
} from '@vitrum/shared-bvh';
import { PT_WEBGL2_MAX_NESTED_MEDIA } from '../supportManifest.js';

/**
 * Enforce the exact closed/laminar topology promised by the live WebGL2
 * capability record. Callers must invoke this on a complete candidate scene
 * before any texture, BVH, program, or retained-scene state is changed.
 */
export function assertWebGl2OpticalMediumTopology(
  scene: Scene,
  method: string,
): OpticalMediumTopologyAnalysis {
  const representedScene = lowerTransmissiveAnalyticPrimitives(scene);
  return assertOpticalMediumTopology(representedScene, {
    maxNestedMedia: PT_WEBGL2_MAX_NESTED_MEDIA,
    backend: 'pt-webgl2',
    method,
    // Every transmissive analytic is lowered to core's canonical generated
    // triangles above, even when an unrelated authored fallback exists.
    analyticGeometry: 'generated-triangle',
    // mergeWorldSpaceFromCore evaluates transforms in JS f64 and stores one
    // final f32 result. TLAS shader-f32 emulation is a different geometry.
    transformArithmetic: 'merged-world-f64-to-f32',
  });
}

/**
 * Project the shared validator's exact component membership into the merged,
 * BVH-reordered triangle stream. Zero means non-bulk; component ids are stable,
 * 1-based analysis ordinals. This intentionally consumes the validator result
 * instead of recomputing connectivity in the backend.
 */
export function buildWebGl2OpticalComponentIds(
  scene: Scene,
  merged: WorldSpaceMergeResult,
  method: string,
): Uint32Array {
  const analysis = assertWebGl2OpticalMediumTopology(scene, method);
  const mergedComponentIds = new Uint32Array(merged.triangleCount);

  for (let componentOrdinal = 0; componentOrdinal < analysis.components.length; componentOrdinal += 1) {
    const component = analysis.components[componentOrdinal]!;
    const matchingRanges = merged.meshVertexRanges.filter((range) =>
      (range.sourcePrimitiveId ?? range.name) === component.primitiveId &&
      (range.sourceInstanceIndex ?? 0) === component.instanceIndex);
    if (matchingRanges.length !== 1) {
      throw new Error(
        `[pt-webgl2 ${method}] optical component mapping for primitive ` +
          `"${component.primitiveId}" instance ${component.instanceIndex} ` +
          `resolved ${matchingRanges.length} merged ranges; expected exactly one.`,
      );
    }
    const range = matchingRanges[0]!;
    // Both representation variants expose source-triangle ordinals. A
    // generated analytic is valid here only because the caller's merged range
    // was produced by the exact same canonical analyticPrimitiveToMesh result.
    // Shared boundary ids are dense and zero-based. The texture ABI reserves
    // zero for non-bulk triangles, so store the lossless 1-based encoding.
    const componentId = component.boundaryId + 1;
    for (const sourceTriangle of component.representation.sourceTriangles) {
      if (sourceTriangle < 0 || sourceTriangle >= range.triCount) {
        throw new Error(
          `[pt-webgl2 ${method}] optical component source triangle ` +
            `${sourceTriangle} is outside merged range [0, ${range.triCount}).`,
        );
      }
      const mergedTriangle = range.triStart + sourceTriangle;
      if (mergedComponentIds[mergedTriangle] !== 0) {
        throw new Error(
          `[pt-webgl2 ${method}] merged triangle ${mergedTriangle} belongs to ` +
            'more than one validated optical component.',
        );
      }
      mergedComponentIds[mergedTriangle] = componentId;
    }
  }

  for (let triangle = 0; triangle < merged.triangleCount; triangle += 1) {
    const materialId = merged.mergedTriMaterialId[triangle];
    const material = materialId === undefined ? undefined : merged.materials[materialId];
    const shouldBeBulk = material != null && materialDefinesBulkOpticalMedium(material);
    const hasComponent = mergedComponentIds[triangle] !== 0;
    if (shouldBeBulk !== hasComponent) {
      throw new Error(
        `[pt-webgl2 ${method}] merged triangle ${triangle} bulk/component ` +
          `membership mismatch (${String(shouldBeBulk)} vs ${String(hasComponent)}).`,
      );
    }
  }

  const reordered = new Uint32Array(merged.triangleCount);
  for (let triangle = 0; triangle < merged.triangleCount; triangle += 1) {
    const mergedTriangle = merged.bvhTriToMergedTri[triangle];
    if (mergedTriangle === undefined || mergedTriangle >= mergedComponentIds.length) {
      throw new Error(
        `[pt-webgl2 ${method}] BVH triangle ${triangle} has invalid merged ` +
          `source ${String(mergedTriangle)}.`,
      );
    }
    reordered[triangle] = mergedComponentIds[mergedTriangle]!;
  }
  return reordered;
}

/**
 * Dense identity of the represented primitive-instance range owning each BVH
 * triangle. Unlike bulk component identity this is defined for thin sheets and
 * opaque surfaces too. Exact edge/vertex continuation tokens use it to avoid
 * conflating two distinct sheet instances that happen to share coordinates.
 */
export function buildWebGl2RepresentedPrimitiveInstanceIds(
  merged: WorldSpaceMergeResult,
  method: string,
): Uint32Array {
  const mergedRangeIds = new Uint32Array(merged.triangleCount);
  for (let rangeIndex = 0; rangeIndex < merged.meshVertexRanges.length; rangeIndex += 1) {
    const range = merged.meshVertexRanges[rangeIndex]!;
    const encodedRangeId = rangeIndex + 1;
    for (let localTriangle = 0; localTriangle < range.triCount; localTriangle += 1) {
      const mergedTriangle = range.triStart + localTriangle;
      if (mergedTriangle < 0 || mergedTriangle >= merged.triangleCount) {
        throw new Error(
          `[pt-webgl2 ${method}] represented range ${rangeIndex} triangle ` +
            `${mergedTriangle} is outside [0, ${merged.triangleCount}).`,
        );
      }
      if (mergedRangeIds[mergedTriangle] !== 0) {
        throw new Error(
          `[pt-webgl2 ${method}] merged triangle ${mergedTriangle} belongs to ` +
            'more than one represented primitive-instance range.',
        );
      }
      mergedRangeIds[mergedTriangle] = encodedRangeId;
    }
  }
  for (let triangle = 0; triangle < mergedRangeIds.length; triangle += 1) {
    if (mergedRangeIds[triangle] === 0) {
      throw new Error(
        `[pt-webgl2 ${method}] merged triangle ${triangle} has no represented ` +
          'primitive-instance range identity.',
      );
    }
  }
  const reordered = new Uint32Array(merged.triangleCount);
  for (let triangle = 0; triangle < merged.triangleCount; triangle += 1) {
    const mergedTriangle = merged.bvhTriToMergedTri[triangle];
    if (mergedTriangle === undefined || mergedTriangle >= mergedRangeIds.length) {
      throw new Error(
        `[pt-webgl2 ${method}] BVH triangle ${triangle} has invalid merged ` +
          `source ${String(mergedTriangle)}.`,
      );
    }
    reordered[triangle] = mergedRangeIds[mergedTriangle]!;
  }
  return reordered;
}
