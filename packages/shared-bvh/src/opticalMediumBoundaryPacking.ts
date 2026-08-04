import type { Scene } from '@vitrum/core';
import type { ScenePackResult } from './scenePack.js';
import type { WorldSpaceMergeResult } from './worldSpaceMerge.js';
import type {
  OpticalMediumComponentAnalysis,
  OpticalMediumTopologyAnalysis,
} from './opticalMediumTopology.js';

/** GPU sentinel. Valid boundary identities are encoded as `boundaryId + 1`. */
export const OPTICAL_MEDIUM_INVALID_ENCODED_BOUNDARY_ID = 0;
export const OPTICAL_MEDIUM_MAX_BOUNDARY_ID = 0xffff_fffe;
export const OPTICAL_INVALID_REPRESENTED_PRIMITIVE_INSTANCE_ID = 0;
export const OPTICAL_MAX_REPRESENTED_PRIMITIVE_INSTANCE_ORDINAL = 0xffff_fffe;

export function encodeOpticalRepresentedPrimitiveInstanceId(ordinal: number): number {
  if (
    !Number.isSafeInteger(ordinal) || ordinal < 0 ||
    ordinal > OPTICAL_MAX_REPRESENTED_PRIMITIVE_INSTANCE_ORDINAL
  ) {
    throw new RangeError(
      'encodeOpticalRepresentedPrimitiveInstanceId: ordinal must be a u32-encodable ' +
        `non-negative integer (got ${String(ordinal)}).`,
    );
  }
  return ordinal + 1;
}

/**
 * Scene-pack addressing required to turn a triangle/TLAS hit into the exact
 * component identity validated by {@link analyzeOpticalMediumTopology}.
 *
 * For triangle hits:
 * `instanceBoundaryIdBasePlusOne[instance] +
 * triangleComponentIndexPlusOne[triangle] - 1` is the encoded analyzed
 * boundary ID (`boundaryId + 1`). Direct-BLAS traversal uses implicit instance
 * slot zero. Zero in either lane means the hit is not a bulk boundary.
 */
export interface PackedOpticalMediumBoundaryIds {
  readonly triangleComponentIndexPlusOne: Uint32Array;
  /**
   * Stable represented primitive-instance/range scope for direct traversal.
   * TLAS traversal uses its packed instance slot instead.
   */
  readonly triangleRepresentedPrimitiveInstanceIds: Uint32Array;
  readonly instanceBoundaryIdBasePlusOne: Uint32Array;
}

export function encodeOpticalMediumBoundaryId(boundaryId: number): number {
  if (
    !Number.isSafeInteger(boundaryId) || boundaryId < 0 ||
    boundaryId > OPTICAL_MEDIUM_MAX_BOUNDARY_ID
  ) {
    throw new RangeError(
      `encodeOpticalMediumBoundaryId: boundaryId must be an integer in ` +
        `[0, ${OPTICAL_MEDIUM_MAX_BOUNDARY_ID}] (got ${String(boundaryId)}).`,
    );
  }
  return boundaryId + 1;
}

export function decodeOpticalMediumBoundaryId(encodedBoundaryId: number): number | null {
  if (
    !Number.isSafeInteger(encodedBoundaryId) || encodedBoundaryId < 0 ||
    encodedBoundaryId > 0xffff_ffff
  ) {
    throw new RangeError(
      `decodeOpticalMediumBoundaryId: value must be a u32 (got ${String(encodedBoundaryId)}).`,
    );
  }
  return encodedBoundaryId === OPTICAL_MEDIUM_INVALID_ENCODED_BOUNDARY_ID
    ? null
    : encodedBoundaryId - 1;
}

function componentsByInstance(
  components: readonly OpticalMediumComponentAnalysis[],
): Map<number, readonly OpticalMediumComponentAnalysis[]> {
  const mutable = new Map<number, OpticalMediumComponentAnalysis[]>();
  for (const component of components) {
    const entries = mutable.get(component.instanceIndex) ?? [];
    entries.push(component);
    mutable.set(component.instanceIndex, entries);
  }
  const result = new Map<number, readonly OpticalMediumComponentAnalysis[]>();
  for (const [instanceIndex, entries] of mutable) {
    result.set(instanceIndex, [...entries].sort(
      (a, b) => a.componentIndex - b.componentIndex,
    ));
  }
  return result;
}

function assertContiguousComponentIds(
  primitiveId: string,
  instanceIndex: number,
  components: readonly OpticalMediumComponentAnalysis[],
): number {
  if (components.length === 0) return OPTICAL_MEDIUM_INVALID_ENCODED_BOUNDARY_ID;
  const base = components[0]!.boundaryId;
  encodeOpticalMediumBoundaryId(base);
  for (let index = 0; index < components.length; index += 1) {
    const component = components[index]!;
    if (component.componentIndex !== index || component.boundaryId !== base + index) {
      throw new RangeError(
        `packOpticalMediumBoundaryIds: primitive "${primitiveId}" instance ` +
          `${instanceIndex} does not have contiguous component/boundary identities.`,
      );
    }
  }
  return base;
}

function sourceTriangleComponentMap(
  primitiveId: string,
  instanceIndex: number,
  components: readonly OpticalMediumComponentAnalysis[],
): Map<number, number> {
  const result = new Map<number, number>();
  for (const component of components) {
    for (const sourceTriangle of component.representation.sourceTriangles) {
      if (result.has(sourceTriangle)) {
        throw new RangeError(
          `packOpticalMediumBoundaryIds: primitive "${primitiveId}" source triangle ` +
            `${sourceTriangle} belongs to more than one component.`,
        );
      }
      result.set(sourceTriangle, component.componentIndex);
    }
  }
  return result;
}

function assertSameTrianglePartition(
  primitiveId: string,
  reference: ReadonlyMap<number, number>,
  candidate: ReadonlyMap<number, number>,
  instanceIndex: number,
): void {
  if (reference.size !== candidate.size) {
    throw new RangeError(
      `packOpticalMediumBoundaryIds: primitive "${primitiveId}" instance ` +
        `${instanceIndex} has a different represented triangle partition.`,
    );
  }
  for (const [triangle, component] of reference) {
    if (candidate.get(triangle) !== component) {
      throw new RangeError(
        `packOpticalMediumBoundaryIds: primitive "${primitiveId}" instance ` +
          `${instanceIndex} maps source triangle ${triangle} to a different component.`,
      );
    }
  }
}

/**
 * Build immutable GPU-ready identity lanes for one exact `ScenePackResult` and
 * topology analysis. The pack's reorder/source and instance/source arrays are
 * validated before any output is returned, so callers can publish the three
 * arrays transactionally with the geometry they describe.
 */
export function packOpticalMediumBoundaryIds(
  scene: Scene,
  pack: Pick<
    ScenePackResult,
    | 'triangleCount'
    | 'triangleSourceIndices'
    | 'trianglePrimitiveIndices'
    | 'instancePrimitiveIndices'
    | 'instanceSourceIndices'
  >,
  analysis: OpticalMediumTopologyAnalysis,
): PackedOpticalMediumBoundaryIds {
  if (
    pack.triangleSourceIndices.length !== pack.triangleCount ||
    pack.trianglePrimitiveIndices.length !== pack.triangleCount
  ) {
    throw new RangeError(
      'packOpticalMediumBoundaryIds: triangle source-address arrays must match triangleCount.',
    );
  }
  if (pack.instancePrimitiveIndices.length !== pack.instanceSourceIndices.length) {
    throw new RangeError(
      'packOpticalMediumBoundaryIds: instance primitive/source arrays must have equal length.',
    );
  }
  if (analysis.componentCount !== analysis.components.length) {
    throw new RangeError(
      'packOpticalMediumBoundaryIds: analysis componentCount does not match components.length.',
    );
  }

  const primitiveComponents = new Map<string, OpticalMediumComponentAnalysis[]>();
  for (const component of analysis.components) {
    const entries = primitiveComponents.get(component.primitiveId) ?? [];
    entries.push(component);
    primitiveComponents.set(component.primitiveId, entries);
  }

  const trianglePartitionByPrimitive = new Map<string, ReadonlyMap<number, number>>();
  const componentInstancesByPrimitive = new Map<
    string,
    ReadonlyMap<number, readonly OpticalMediumComponentAnalysis[]>
  >();
  for (const [primitiveId, components] of primitiveComponents) {
    const instances = componentsByInstance(components);
    componentInstancesByPrimitive.set(primitiveId, instances);
    let reference: ReadonlyMap<number, number> | null = null;
    for (const [instanceIndex, instanceComponents] of instances) {
      assertContiguousComponentIds(primitiveId, instanceIndex, instanceComponents);
      const partition = sourceTriangleComponentMap(
        primitiveId,
        instanceIndex,
        instanceComponents,
      );
      if (reference == null) reference = partition;
      else assertSameTrianglePartition(primitiveId, reference, partition, instanceIndex);
    }
    if (reference != null) trianglePartitionByPrimitive.set(primitiveId, reference);
  }

  const triangleComponentIndexPlusOne = new Uint32Array(pack.triangleCount);
  for (let triangle = 0; triangle < pack.triangleCount; triangle += 1) {
    const primitiveIndex = pack.trianglePrimitiveIndices[triangle]!;
    const primitive = scene.primitives[primitiveIndex];
    if (primitive == null) {
      throw new RangeError(
        `packOpticalMediumBoundaryIds: triangle ${triangle} references missing scene primitive ${primitiveIndex}.`,
      );
    }
    const partition = trianglePartitionByPrimitive.get(primitive.id);
    if (partition == null) continue;
    const sourceTriangle = pack.triangleSourceIndices[triangle]!;
    const componentIndex = partition.get(sourceTriangle);
    if (componentIndex === undefined) {
      throw new RangeError(
        `packOpticalMediumBoundaryIds: bulk primitive "${primitive.id}" packed source ` +
          `triangle ${sourceTriangle} is absent from its analyzed closed components.`,
      );
    }
    triangleComponentIndexPlusOne[triangle] = componentIndex + 1;
  }

  const instanceBoundaryIdBasePlusOne = new Uint32Array(pack.instancePrimitiveIndices.length);
  for (let instance = 0; instance < instanceBoundaryIdBasePlusOne.length; instance += 1) {
    const primitiveIndex = pack.instancePrimitiveIndices[instance]!;
    const primitive = scene.primitives[primitiveIndex];
    if (primitive == null) {
      throw new RangeError(
        `packOpticalMediumBoundaryIds: instance ${instance} references missing scene primitive ${primitiveIndex}.`,
      );
    }
    const sourceInstance = pack.instanceSourceIndices[instance]!;
    const components = componentInstancesByPrimitive.get(primitive.id)?.get(sourceInstance);
    if (components == null || components.length === 0) continue;
    const boundaryBase = assertContiguousComponentIds(
      primitive.id,
      sourceInstance,
      components,
    );
    instanceBoundaryIdBasePlusOne[instance] = encodeOpticalMediumBoundaryId(boundaryBase);
  }

  const triangleRepresentedPrimitiveInstanceIds = new Uint32Array(pack.triangleCount);
  for (let triangle = 0; triangle < pack.triangleCount; triangle += 1) {
    triangleRepresentedPrimitiveInstanceIds[triangle] =
      encodeOpticalRepresentedPrimitiveInstanceId(
        pack.trianglePrimitiveIndices[triangle]!,
      );
  }
  return {
    triangleComponentIndexPlusOne,
    triangleRepresentedPrimitiveInstanceIds,
    instanceBoundaryIdBasePlusOne,
  };
}

/**
 * Pack identity for a world-space merged BVH. Each merged range is already one
 * concrete primitive instance, so its analyzed final encoded boundary ID is
 * stored directly in the triangle lane and the implicit instance base is one.
 */
export function packMergedOpticalMediumBoundaryIds(
  scene: Scene,
  merged: Pick<
    WorldSpaceMergeResult,
    'triangleCount' | 'bvhTriToMergedTri' | 'meshVertexRanges'
  >,
  analysis: OpticalMediumTopologyAnalysis,
): PackedOpticalMediumBoundaryIds {
  if (merged.bvhTriToMergedTri.length !== merged.triangleCount) {
    throw new RangeError(
      'packMergedOpticalMediumBoundaryIds: BVH/source map must match triangleCount.',
    );
  }
  const primitiveById = new Map(scene.primitives.map((primitive) => [primitive.id, primitive]));
  const componentInstances = new Map<string, Map<number, readonly OpticalMediumComponentAnalysis[]>>();
  for (const component of analysis.components) {
    const instances = componentInstances.get(component.primitiveId) ??
      new Map<number, readonly OpticalMediumComponentAnalysis[]>();
    const entries = instances.get(component.instanceIndex) ?? [];
    instances.set(component.instanceIndex, [...entries, component]);
    componentInstances.set(component.primitiveId, instances);
  }

  const mergedTriangleBoundary = new Uint32Array(merged.triangleCount);
  const mergedTriangleRange = new Uint32Array(merged.triangleCount);
  const assigned = new Uint8Array(merged.triangleCount);
  for (let rangeIndex = 0; rangeIndex < merged.meshVertexRanges.length; rangeIndex += 1) {
    const range = merged.meshVertexRanges[rangeIndex]!;
    const primitiveId = range.sourcePrimitiveId ?? range.name;
    if (!primitiveById.has(primitiveId)) {
      throw new RangeError(
        `packMergedOpticalMediumBoundaryIds: range ${rangeIndex} references missing primitive "${primitiveId}".`,
      );
    }
    const instanceIndex = range.sourceInstanceIndex ?? 0;
    const components = componentInstances.get(primitiveId)?.get(instanceIndex) ?? [];
    const sourceToBoundary = new Map<number, number>();
    for (const component of components) {
      const encoded = encodeOpticalMediumBoundaryId(component.boundaryId);
      for (const sourceTriangle of component.representation.sourceTriangles) {
        sourceToBoundary.set(sourceTriangle, encoded);
      }
    }
    for (let localTriangle = 0; localTriangle < range.triCount; localTriangle += 1) {
      const mergedTriangle = range.triStart + localTriangle;
      if (mergedTriangle >= merged.triangleCount || assigned[mergedTriangle] !== 0) {
        throw new RangeError(
          `packMergedOpticalMediumBoundaryIds: range ${rangeIndex} has overlapping/out-of-range triangle ${mergedTriangle}.`,
        );
      }
      assigned[mergedTriangle] = 1;
      mergedTriangleRange[mergedTriangle] =
        encodeOpticalRepresentedPrimitiveInstanceId(rangeIndex);
      const encoded = sourceToBoundary.get(localTriangle) ?? 0;
      if (components.length > 0 && encoded === 0) {
        throw new RangeError(
          `packMergedOpticalMediumBoundaryIds: bulk range "${primitiveId}" instance ` +
            `${instanceIndex} source triangle ${localTriangle} is absent from its analyzed components.`,
        );
      }
      mergedTriangleBoundary[mergedTriangle] = encoded;
    }
  }
  if (assigned.some((value) => value === 0)) {
    throw new RangeError(
      'packMergedOpticalMediumBoundaryIds: merged ranges do not tile the triangle stream.',
    );
  }

  const triangleComponentIndexPlusOne = new Uint32Array(merged.triangleCount);
  const triangleRepresentedPrimitiveInstanceIds = new Uint32Array(merged.triangleCount);
  for (let triangle = 0; triangle < merged.triangleCount; triangle += 1) {
    const source = merged.bvhTriToMergedTri[triangle]!;
    if (source >= merged.triangleCount) {
      throw new RangeError(
        `packMergedOpticalMediumBoundaryIds: triangle ${triangle} maps out of range to ${source}.`,
      );
    }
    triangleComponentIndexPlusOne[triangle] = mergedTriangleBoundary[source]!;
    triangleRepresentedPrimitiveInstanceIds[triangle] = mergedTriangleRange[source]!;
  }
  return {
    triangleComponentIndexPlusOne,
    triangleRepresentedPrimitiveInstanceIds,
    // Direct traversal resolves encoded = 1 + storedEncoded - 1.
    instanceBoundaryIdBasePlusOne: new Uint32Array([1]),
  };
}

/** CPU oracle for the exact shader-side encoded base-plus-component lookup. */
export function resolvePackedOpticalMediumEncodedBoundaryId(
  packed: Pick<
    PackedOpticalMediumBoundaryIds,
    'triangleComponentIndexPlusOne' | 'instanceBoundaryIdBasePlusOne'
  >,
  triangleIndex: number,
  instanceIndex = 0,
): number {
  const componentPlusOne = packed.triangleComponentIndexPlusOne[triangleIndex] ?? 0;
  const basePlusOne = packed.instanceBoundaryIdBasePlusOne[instanceIndex] ?? 0;
  if (
    componentPlusOne === 0 ||
    basePlusOne === 0 ||
    basePlusOne + componentPlusOne - 1 > 0xffff_ffff
  ) {
    return OPTICAL_MEDIUM_INVALID_ENCODED_BOUNDARY_ID;
  }
  return basePlusOne + componentPlusOne - 1;
}
