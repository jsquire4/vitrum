/**
 * Core-scene ReSTIR BVH build path.
 *
 * This file is deliberately free of runtime `three` imports. The historical
 * raw-THREE scene graph path lives under `legacy/three`; the concrete
 * HybridEngine imports from here so core-scene rendering cannot pull a Three
 * dependency through a mixed module.
 */

import {
  sparseArrayOwnIndices,
  type EngineWarning,
  type MaterialSpec,
  type PrimitiveUvSets,
  type Scene,
} from '@vitrum/core';
import {
  collapseIndicesToStride3,
  isLeafSplit,
  materialSig,
  mergeUv1FromCore,
  mergeWorldSpaceFromCore,
  packSceneFromCore,
  rebuildPrimitiveBlas,
  resolveDisplacedGeometry,
  toProductionEmissiveRadiance,
  type PrimitiveTlasBinding,
  type ScenePackResult,
  type WorldSpaceMergeResult,
} from '@vitrum/shared-bvh';
import { buildAliasTable } from '@vitrum/shared-samplers';
import {
  packUVIntoPositionW,
  packUVIntoVec4W,
} from '../bvh/bvhPacking.js';
import {
  packBVHIndexWFromCore,
  packBVHBeerColorsFromCore,
  packBVHEmissiveLeFromCore,
  packBVHRoughMetalFromCore,
} from './packingHelpers.js';
import { packMaterialTextureAtlas } from '../bvh/materialTextureAtlasPack.js';
import { buildEmitterListFromCore, buildLightTreeBuffer } from './emitterList.js';
import {
  collectRectAreaEmitterTrisFromCore,
  enrichMeshVertexRangesWithCoreMatrix,
} from './bvhSceneHelpers.js';
import type {
  RebuiltEmitterBuffers,
  ReSTIRBvhMode,
  SceneBVHBuffers,
} from './bvhTypes.js';

export type { RebuiltEmitterBuffers, ReSTIRBvhMode, SceneBVHBuffers };

interface Vector3Like {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

export interface CoreBvhBuildOptions {
  bvhMode?: ReSTIRBvhMode;
  primaryLightDir?: Vector3Like;
  primaryLightIntensity?: number;
  onWarning?: (warning: EngineWarning) => void;
  warningPhase?: EngineWarning['phase'];
  warningMethod?: string;
  suppressMeshAreaMissingReferenceWarnings?: boolean;
}

type MaterialWithCastShadow = MaterialSpec & { readonly castShadow?: boolean };
interface MeshAreaLeOverride {
  readonly Le: [number, number, number];
  readonly castShadowDisabled: boolean;
}

function sceneHasCoreMeshes(scene: Scene): boolean {
  return scene.primitives.some(
    (p) => p.kind === 'mesh' || p.kind === 'skinned-mesh' || p.kind === 'instanced-mesh',
  );
}

export function resolveReSTIRBvhMode(scene: Scene, override?: ReSTIRBvhMode): ReSTIRBvhMode {
  if (override != null) return override;
  const meshLike = scene.primitives.filter(
    (p) => p.kind === 'mesh' || p.kind === 'skinned-mesh' || p.kind === 'instanced-mesh',
  );
  if (meshLike.some((p) => p.kind === 'instanced-mesh')) return 'tlas';
  if (meshLike.length > 1) return 'tlas';
  return 'merged';
}

function buildMergedPrimitiveRefitNodeIndices(
  merged: WorldSpaceMergeResult,
): ReadonlyMap<string, Uint32Array> {
  const totalNodes = merged.bvhNodes.length / 8;
  const words = new Uint32Array(
    merged.bvhNodes.buffer,
    merged.bvhNodes.byteOffset,
    merged.bvhNodes.length,
  );
  const ownerByMergedTriangle = new Int32Array(merged.triangleCount);
  ownerByMergedTriangle.fill(-1);
  merged.meshVertexRanges.forEach((range, owner) => {
    for (let tri = range.triStart; tri < range.triStart + range.triCount; tri += 1) {
      ownerByMergedTriangle[tri] = owner;
    }
  });
  const parents = new Int32Array(totalNodes);
  parents.fill(-1);
  const leavesByOwner = merged.meshVertexRanges.map(() => new Set<number>());
  const pending = totalNodes > 0 ? [0] : [];
  while (pending.length > 0) {
    const node = pending.pop()!;
    const base = node * 8;
    const splitOrCount = words[base + 7]!;
    if (!isLeafSplit(splitOrCount)) {
      const left = node + 1;
      const right = node + words[base + 6]!;
      parents[left] = node;
      parents[right] = node;
      pending.push(right, left);
      continue;
    }
    const triangleOffset = words[base + 6]!;
    const triangleCount = splitOrCount & 0xffff;
    for (
      let tri = triangleOffset;
      tri < triangleOffset + triangleCount;
      tri += 1
    ) {
      const mergedTri = merged.bvhTriToMergedTri[tri];
      const owner = mergedTri == null ? -1 : ownerByMergedTriangle[mergedTri]!;
      if (owner >= 0) leavesByOwner[owner]!.add(node);
    }
  }

  return new Map(merged.meshVertexRanges.map((range, owner) => {
    const affected = new Set<number>();
    for (const leaf of leavesByOwner[owner]!) {
      for (let node = leaf; node >= 0; node = parents[node]!) {
        affected.add(node);
      }
    }
    // Preorder indices guarantee children > parents, so descending order is
    // the required leaf-to-root refit order.
    return [
      range.name,
      Uint32Array.from([...affected].sort((a, b) => b - a)),
    ] as const;
  }));
}

function makeStorageHandle(
  data: ArrayBufferView,
  elementBytes: number,
): { cpuData: ArrayBuffer; byteLength: number; count: number } {
  return {
    cpuData: data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer,
    byteLength: data.byteLength,
    count: Math.floor(data.byteLength / elementBytes),
  };
}

function applyMeshAreaLeOverride(material: MaterialSpec, override: MeshAreaLeOverride): MaterialSpec {
  const castShadowDisabled =
    override.castShadowDisabled ||
    (material as MaterialWithCastShadow).castShadow === false;
  return {
    ...material,
    emissive: [override.Le[0], override.Le[1], override.Le[2]] as const,
    emissiveIntensity: 1,
    ...(castShadowDisabled ? { castShadow: false } : {}),
  };
}

function warnCoreBvh(options: Pick<CoreBvhBuildOptions, 'onWarning'>, warning: EngineWarning): void {
  if (options.onWarning) {
    try {
      options.onWarning(warning);
    } catch {
      // Host warning callbacks must not break BVH construction.
    }
    return;
  }
  console.warn(warning.message);
}

function vertexDisplacementWarningDetails(warning: string): Readonly<Record<string, unknown>> | null {
  if (!warning.includes('displacementMap')) return null;
  const details: Record<string, unknown> = {
    warning,
    source: 'shared-bvh',
    fallback: 'displacement warning retained',
  };
  const match = / displacementMap at (.+?)(?: handle | requests | has | displacementSubdivisions| triangle )/.exec(warning);
  if (match?.[1] !== undefined) details.sourcePath = match[1];
  if (/displacement skipped/i.test(warning) || warning.includes('Skipped.')) {
    details.fallback = 'displacement skipped';
  } else if (warning.includes('microdisplacement disabled')) {
    details.fallback = 'microdisplacement disabled';
  } else if (warning.includes('Falling back to vertex displacement')) {
    details.fallback = 'microdisplacement fallback to vertex displacement';
  }
  return details;
}

function warnScenePackWarnings(
  options: CoreBvhBuildOptions,
  warnings: readonly string[],
): void {
  for (const warning of warnings) {
    const displacementDetails = vertexDisplacementWarningDetails(warning);
    warnCoreBvh(options, {
      code: displacementDetails != null
        ? 'walkaround-hybrid.vertex-displacement-skipped'
        : 'walkaround-hybrid.scene-pack-warning',
      backend: 'walkaround-hybrid',
      phase: options.warningPhase ?? 'setScene',
      method: options.warningMethod ?? 'buildReSTIRSceneBVHForCoreScene',
      message: `[vitrum/walkaround-hybrid] ${warning}`,
      details: displacementDetails ?? {
        warning,
        source: 'shared-bvh',
        fallback: 'scene pack warning retained',
      },
    });
  }
}

function warnMissingMeshAreaEmitterReference(
  options: CoreBvhBuildOptions,
  emitterId: unknown,
  meshId: string,
  source: 'bvh-emissive-override' | 'emitter-list',
): void {
  if (options.suppressMeshAreaMissingReferenceWarnings === true) return;
  warnCoreBvh(options, {
    code: 'walkaround-hybrid.mesh-area-emitter-missing-mesh',
    backend: 'walkaround-hybrid',
    phase: options.warningPhase ?? 'setScene',
    method: options.warningMethod ?? 'buildReSTIRSceneBVHForCoreScene',
    message:
      `[vitrum/walkaround-hybrid] mesh-area emitter "${String(emitterId)}" ` +
      `references meshId="${meshId}" which matches no scene primitive; ` +
      `the emitter color/intensity is ignored for walkaround lighting.`,
    details: {
      emitterId: String(emitterId),
      meshId,
      source,
      fallback: 'emitter skipped',
    },
  });
}

function materialResolver(scene: Scene): {
  coreMaterials: MaterialSpec[];
  resolveMaterialId: (primitiveId: string) => number;
} {
  const coreMaterials: MaterialSpec[] = [];
  const byKey = new Map<string, number>();
  const duplicateIds = new Set<string>();
  for (const p of scene.primitives) {
    if (p.kind === 'mesh' || p.kind === 'instanced-mesh' || p.kind === 'skinned-mesh') {
      const id = String(p.id);
      if (byKey.has(id)) {
        duplicateIds.add(id);
      } else {
        byKey.set(id, coreMaterials.length);
      }
      // SHADOW-01 — slots are per-primitive here, so the primitive's castShadow
      // flag rides the material entry (consumed by packBVHRoughMetalFromCore's
      // bit-0 lane). Default (true/undefined) keeps the original object —
      // byte-identical pack for flag-less scenes.
      coreMaterials.push(
        p.castShadow === false
          ? ({ ...p.material, castShadow: false } as MaterialSpec)
          : p.material,
      );
    }
  }
  if (duplicateIds.size > 0) {
    throw new Error(
      `[ReSTIR bvhCore] duplicate mesh-like primitive id(s): ${[...duplicateIds].join(', ')}. ` +
      `Primitive ids must be unique so per-triangle material slots, TLAS bindings, ` +
      `mesh-area emitters, and incremental updates resolve the same primitive.`,
    );
  }
  return {
    coreMaterials,
    resolveMaterialId: (id) => {
      const idx = byKey.get(id);
      if (idx === undefined) {
        throw new Error(
          `[ReSTIR bvhCore] could not resolve material slot for primitive id "${id}". ` +
          `The scene packer passed an id that was not present in the validated mesh-like primitive set.`,
        );
      }
      return idx;
    },
  };
}

function makeMergedGeometry(
  min: readonly [number, number, number],
  max: readonly [number, number, number],
): SceneBVHBuffers['mergedGeometry'] {
  const boundingBox = {
    min: { x: min[0], y: min[1], z: min[2] },
    max: { x: max[0], y: max[1], z: max[2] },
  };
  return {
    boundingBox,
    computeBoundingBox() {},
  };
}

/**
 * H23 — Build a material-slot → Le override map from scene `mesh-area` emitters.
 *
 * For each `mesh-area` emitter in the scene, find the referenced primitive by id,
 * look up its material slot in the merged material array (by structural signature),
 * and record `emitter.color * emitter.intensity` as the Le override for that slot.
 *
 * Override rule: the emitter's Le (color*intensity) REPLACES the material's emissive
 * Le when the emitter specifies it. This is the override-vs-sum rule: apply either
 * emitter Le OR material emissive, not both summed. The host explicitly wired a
 * mesh-area emitter to make the mesh a light with a specific colour/intensity; the
 * material emissive is a style hint, not the physical source.
 *
 * Caveats documented:
 *  - Dedup collision: if two primitives share the same renderer-visible material
 *    signature, the override applies to ALL triangles with that material slot. This
 *    is an accepted edge case — mesh-area emitters are typically unique materials.
 *  - Missing reference: a meshId that matches no primitive is warned and skipped.
 *
 * @returns Map from material-slot-index to Le [r,g,b] override; empty when no
 *          mesh-area emitters are present.
 */
function buildMeshAreaLeOverrides(
  scene: Scene,
  mergedMaterials: readonly MaterialSpec[],
  options: CoreBvhBuildOptions = {},
): Map<number, MeshAreaLeOverride> {
  const meshAreaEmitters = scene.emitters.filter((e) => e.kind === 'mesh-area');
  if (meshAreaEmitters.length === 0) return new Map();

  const materialSlotSig = (m: MaterialSpec, castShadow?: boolean): string =>
    `${materialSig(m)}|castShadow=${(castShadow ?? (m as MaterialWithCastShadow).castShadow ?? true) ? 1 : 0}`;

  // Build material-signature → slot index in O(M) using the same materialSig
  // family as mergeWorldSpaceFromCore. The bare key covers default merged
  // material dedup; the castShadow-suffixed key covers callers that split
  // otherwise-identical material slots by shadow participation.
  const sigToSlot = new Map<string, number>();
  for (let s = 0; s < mergedMaterials.length; s++) {
    const material = mergedMaterials[s]!;
    const bare = materialSig(material);
    const shadowAware = materialSlotSig(material);
    if (!sigToSlot.has(bare)) sigToSlot.set(bare, s);
    if (!sigToSlot.has(shadowAware)) sigToSlot.set(shadowAware, s);
  }

  // Build primitive-id → material slot via one O(P) pass.
  const primitiveIdToMaterialSlot = new Map<string, number>();
  for (const p of scene.primitives) {
    if (p.kind === 'mesh' || p.kind === 'skinned-mesh' || p.kind === 'instanced-mesh') {
      const shadowAware = materialSlotSig(p.material, p.castShadow ?? true);
      const s = sigToSlot.get(shadowAware) ?? sigToSlot.get(materialSig(p.material));
      if (s !== undefined) {
        primitiveIdToMaterialSlot.set(String(p.id), s);
      }
    }
  }

  const overrides = new Map<number, MeshAreaLeOverride>();
  for (const e of meshAreaEmitters) {
    if (e.kind !== 'mesh-area') continue;
    const meshId = String(e.meshId);
    const slot = primitiveIdToMaterialSlot.get(meshId);
    if (slot === undefined) {
      warnMissingMeshAreaEmitterReference(options, e.id, meshId, 'bvh-emissive-override');
      continue;
    }
    const Le: [number, number, number] = [
      e.color[0] * e.intensity,
      e.color[1] * e.intensity,
      e.color[2] * e.intensity,
    ];
    overrides.set(slot, {
      Le,
      castShadowDisabled: e.castShadow === false,
    });
  }
  return overrides;
}

/**
 * H23 — Apply mesh-area emitter Le overrides to a primitive-ordered `coreMaterials`
 * array (as produced by `materialResolver`). Returns a patched copy where each
 * primitive id referenced by a `mesh-area` emitter has its emissive replaced by
 * `emitter.color * emitter.intensity`. Used by `packBVHEmissiveLeFromCore` so the
 * camera-visible emissive glow also reflects the emitter Le, not just the ReSTIR
 * emitter stream. Emitter ids that match no primitive are warned and skipped.
 */
function applyMeshAreaLeOverridesToCoreMaterials(
  scene: Scene,
  coreMaterials: readonly MaterialSpec[],
): readonly MaterialSpec[] {
  const meshAreaEmitters = scene.emitters.filter((e) => e.kind === 'mesh-area');
  if (meshAreaEmitters.length === 0) return coreMaterials;

  // Build primitive-id → coreMaterials slot index.
  const idToSlot = new Map<string, number>();
  let meshIdx = 0;
  for (const p of scene.primitives) {
    if (p.kind === 'mesh' || p.kind === 'skinned-mesh' || p.kind === 'instanced-mesh') {
      if (!idToSlot.has(String(p.id))) idToSlot.set(String(p.id), meshIdx);
      meshIdx++;
    }
  }

  const patched = [...coreMaterials] as MaterialSpec[];
  for (const e of meshAreaEmitters) {
    if (e.kind !== 'mesh-area') continue;
    const slot = idToSlot.get(String(e.meshId));
    if (slot === undefined) {
      // Warn already issued by buildMeshAreaLeOverrides for the emitter-list path.
      continue;
    }
    const Le: [number, number, number] = [e.color[0] * e.intensity, e.color[1] * e.intensity, e.color[2] * e.intensity];
    patched[slot] = applyMeshAreaLeOverride(patched[slot]!, {
      Le,
      castShadowDisabled: e.castShadow === false,
    });
  }
  return patched;
}

export function rebuildBvhEmissiveLeFromCoreScene(
  scene: Scene,
  bvh: Pick<SceneBVHBuffers, 'bvhMode' | 'coreMaterials' | 'triangleMaterialIds' | 'bvhBeerColors'>,
  options: CoreBvhBuildOptions = {},
): SceneBVHBuffers['bvhEmissiveLe'] {
  const triMaterialIds = new Uint32Array(bvh.triangleMaterialIds.cpuData);
  const triCount = bvh.bvhBeerColors.count;
  if (bvh.bvhMode === 'tlas') {
    const emissiveCoreMats = applyMeshAreaLeOverridesToCoreMaterials(scene, bvh.coreMaterials);
    return makeStorageHandle(
      packBVHEmissiveLeFromCore(triMaterialIds, emissiveCoreMats, triCount),
      16,
    );
  }

  const overrides = buildMeshAreaLeOverrides(scene, bvh.coreMaterials, {
    ...options,
    suppressMeshAreaMissingReferenceWarnings: true,
  });
  const emissiveMaterials = bvh.coreMaterials.map((material, slot) => {
    const override = overrides.get(slot);
    if (override == null) return toProductionEmissiveRadiance(material);
    return applyMeshAreaLeOverride(material, override);
  });
  return makeStorageHandle(
    packBVHEmissiveLeFromCore(triMaterialIds, emissiveMaterials, triCount),
    16,
  );
}

function buildTlasEmitterSourceTriMapper(
  merged: WorldSpaceMergeResult,
  primitiveTlasBindings: readonly PrimitiveTlasBinding[],
): (triIdx: number) => number {
  const bindingByPrimitiveId = new Map<string, PrimitiveTlasBinding>();
  for (const binding of primitiveTlasBindings) {
    if (!bindingByPrimitiveId.has(binding.primitiveId)) {
      bindingByPrimitiveId.set(binding.primitiveId, binding);
    }
  }

  const mergedTriToTlasTri = new Int32Array(merged.triangleCount);
  mergedTriToTlasTri.fill(-1);
  for (const range of merged.meshVertexRanges) {
    const binding = bindingByPrimitiveId.get(range.name);
    if (binding == null) continue;
    const triCount = Math.min(range.triCount, binding.triCount);
    for (let localTri = 0; localTri < triCount; localTri += 1) {
      const sourceTri = binding.triStart + localTri;
      mergedTriToTlasTri[range.triStart + localTri] =
        range.windingFlipped === true ? -(sourceTri + 2) : sourceTri;
    }
  }

  return (triIdx: number): number => {
    const mergedTri = merged.bvhTriToMergedTri[triIdx];
    if (mergedTri === undefined) return -1;
    return mergedTriToTlasTri[mergedTri] ?? -1;
  };
}

function coreEmitterBuffers(
  scene: Scene,
  options: {
    primaryLightDir?: Vector3Like;
    primaryLightIntensity?: number;
    packSourceTriIndex?: boolean;
    tlasPrimitiveBindings?: readonly PrimitiveTlasBinding[];
    onWarning?: (warning: EngineWarning) => void;
    warningPhase?: EngineWarning['phase'];
    warningMethod?: string;
    suppressMeshAreaMissingReferenceWarnings?: boolean;
  } = {},
): RebuiltEmitterBuffers {
  // This stream is only for ReSTIR light selection. Expanding instanced meshes
  // here keeps emissive instances visible to direct lighting while the render
  // BVH can still use the TLAS/BLAS path for traversal.
  const merged = mergeWorldSpaceFromCore(scene, {
    positionStride: 4,
    splitMaterialsByCastShadow: true,
  });
  const extraEmitters = collectRectAreaEmitterTrisFromCore(scene);
  // H23 — derive mesh-area emitter Le overrides. When a mesh-area emitter
  // references a primitive, override that material slot's emissive Le with
  // emitter.color * emitter.intensity (overrides material emissive; does NOT
  // double-apply — the merged material Le is replaced, not summed).
  const meshAreaOverrides = buildMeshAreaLeOverrides(scene, merged.materials, {
    ...options,
    warningMethod: options.warningMethod ?? 'rebuildEmitterBuffersFromCoreScene',
  });
  const productionMaterials = merged.materials.map((m, slot) => {
    const override = meshAreaOverrides.get(slot);
    if (override == null) return toProductionEmissiveRadiance(m);
    // Override: set emissive to the emitter Le and emissiveIntensity to 1
    // (toProductionEmissiveRadiance would keep ei=1 which is correct).
    return applyMeshAreaLeOverride(m, override);
  });
  const sourceTriIndexForTriangle =
    options.packSourceTriIndex === true && options.tlasPrimitiveBindings != null
      ? buildTlasEmitterSourceTriMapper(merged, options.tlasPrimitiveBindings)
      : undefined;
  const { emitterFloats, cdfArray, totalEmissivePower, treeInput } = buildEmitterListFromCore(
    merged.indices,
    merged.positions,
    merged.normals,
    merged.triMaterialId,
    productionMaterials,
    {
      ...options,
      extraEmitters,
      ...(sourceTriIndexForTriangle != null ? { sourceTriIndexForTriangle } : {}),
    },
  );
  const emitterCount = cdfArray.length;
  const emitterAlias = buildAliasTable(treeInput.powers);
  const lightTreeBuf = buildLightTreeBuffer(treeInput);
  return {
    emitters: {
      cpuData: emitterFloats.buffer as ArrayBuffer,
      byteLength: emitterFloats.byteLength,
      count: emitterCount,
    },
    emitterCdf: {
      cpuData: cdfArray.buffer as ArrayBuffer,
      byteLength: cdfArray.byteLength,
      count: emitterCount,
    },
    emitterAlias: {
      cpuData: emitterAlias.data,
      byteLength: emitterAlias.data.byteLength,
      count: emitterCount,
    },
    emitterCount,
    totalEmissivePower,
    lightTree: {
      cpuData: lightTreeBuf.nodes.buffer as ArrayBuffer,
      byteLength: lightTreeBuf.nodes.byteLength,
      count: Math.max(1, lightTreeBuf.nodeCount),
    },
    lightTreeNodeCount: lightTreeBuf.nodeCount,
    lightTreeEnabled: lightTreeBuf.enabled,
  };
}

/**
 * H15 — extract the uv0 layer from a stride-4 `ScenePackResult.uvs` array
 * (layout: [u0, v0, u1, v1] per vertex) into a stride-2 Float32Array that
 * `packUVIntoPositionW` can consume via its `{ array }` BufferAttributeLike.
 * The TLAS vertex ordering is inherited from `packSceneFromCore` → same
 * primitive-concat order as `geo.positions`, so vertex indices align 1:1.
 */
function stride4UvsToStride2Uv0(uvs4: Float32Array, vertCount: number): Float32Array {
  const out = new Float32Array(vertCount * 2);
  for (let i = 0; i < vertCount; i++) {
    out[i * 2] = uvs4[i * 4] ?? 0;
    out[i * 2 + 1] = uvs4[i * 4 + 1] ?? 0;
  }
  return out;
}

function stride4UvsToStride2Uv1(uvs4: Float32Array, vertCount: number): Float32Array {
  const out = new Float32Array(vertCount * 2);
  for (let i = 0; i < vertCount; i++) {
    out[i * 2] = uvs4[i * 4 + 2] ?? 0;
    out[i * 2 + 1] = uvs4[i * 4 + 3] ?? 0;
  }
  return out;
}

interface MneeFacetDomainInput {
  readonly triStart: number;
  readonly triCount: number;
  readonly instanceStart: number;
  readonly instanceCount: number;
}

/**
 * Pack disjoint facet domains plus a Walker/Vose entry per domain. The alias
 * weight is exactly the represented pair cardinality; the shared alias builder
 * stores the quantized distribution's represented PMF, so shader weighting
 * divides by the distribution that is actually on the wire.
 */
export function packMneeFacetDomains(
  inputs: readonly MneeFacetDomainInput[],
): Uint32Array {
  for (const domain of inputs) {
    for (const [name, value] of Object.entries(domain)) {
      if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
        throw new RangeError(
          `[HybridEngine] manifold facet-domain ${name} must be a nonnegative u32.`,
        );
      }
    }
    if (domain.triStart + domain.triCount > 0x1_0000_0000 ||
        domain.instanceStart + domain.instanceCount > 0x1_0000_0000) {
      throw new RangeError(
        '[HybridEngine] manifold facet-domain half-open range exceeds the u32 address space.',
      );
    }
  }
  const domains = inputs.filter(
    (domain) => domain.triCount > 0 && domain.instanceCount > 0,
  );
  const weights = domains.map((domain) => {
    const pairs = domain.triCount * domain.instanceCount;
    if (!Number.isSafeInteger(pairs) || pairs <= 0) {
      throw new RangeError(
        '[HybridEngine] manifold facet-domain cardinality must be a positive safe integer.',
      );
    }
    return pairs;
  });
  const alias = buildAliasTable(weights);
  const aliasWords = new Uint32Array(alias.data);
  const records = new Uint32Array(domains.length * 8);
  for (let index = 0; index < domains.length; index += 1) {
    const domain = domains[index]!;
    const base = index * 8;
    records[base] = domain.triStart >>> 0;
    records[base + 1] = domain.triCount >>> 0;
    records[base + 2] = domain.instanceStart >>> 0;
    records[base + 3] = domain.instanceCount >>> 0;
    records[base + 4] = aliasWords[index * 4]!;
    records[base + 5] = aliasWords[index * 4 + 1]!;
    records[base + 6] = aliasWords[index * 4 + 2]!;
    records[base + 7] = 0;
  }
  return records;
}

function packMergedMneeFacetDomains(triangleCount: number): Uint32Array {
  return packMneeFacetDomains([{
    triStart: 0,
    triCount: triangleCount,
    instanceStart: 0,
    instanceCount: 1,
  }]);
}

/** Compact exact representation of every `(triangle, instance)` TLAS pair. */
function packTlasMneeFacetDomains(
  bindings: readonly PrimitiveTlasBinding[],
): Uint32Array {
  let instanceStart = 0;
  const domains: MneeFacetDomainInput[] = [];
  for (const binding of bindings) {
    domains.push({
      triStart: binding.triStart,
      triCount: binding.triCount,
      instanceStart,
      instanceCount: binding.instanceCount,
    });
    instanceStart += binding.instanceCount;
    if (!Number.isSafeInteger(instanceStart) || instanceStart > 0xffff_ffff) {
      throw new RangeError(
        '[HybridEngine] manifold facet-domain instance range exceeds u32.',
      );
    }
  }
  return packMneeFacetDomains(domains);
}

interface CoreUvMergeRange {
  readonly name: string;
  readonly sourcePrimitiveId?: string;
  readonly vertexStart: number;
  readonly vertexCount: number;
}

/**
 * Rebuild vertex-aligned TEXCOORD_2+ streams in the exact merged/TLAS vertex
 * order. Missing per-primitive streams remain NaN so material-atlas packing
 * fails explicitly instead of silently sampling (0,0).
 */
function mergeHighUvSetsFromCore(
  scene: Scene,
  ranges: readonly CoreUvMergeRange[],
  totalVertexCount: number,
): ReadonlyMap<number, Float32Array> {
  const sourceByPrimitiveId = new Map<string, PrimitiveUvSets>();
  const highSetIndices = new Set<number>();
  for (const primitive of scene.primitives) {
    if (primitive.kind !== 'mesh' && primitive.kind !== 'skinned-mesh' && primitive.kind !== 'instanced-mesh') {
      continue;
    }
    const resolved = resolveDisplacedGeometry(primitive, () => undefined);
    const sets = resolved.baseUvSets;
    if (sets == null) continue;
    sourceByPrimitiveId.set(String(primitive.id), sets);
    for (const texCoord of sparseArrayOwnIndices(sets)) {
      if (texCoord < 2) continue;
      if (sets[texCoord] != null) highSetIndices.add(texCoord);
    }
  }

  const result = new Map<number, Float32Array>();
  for (const texCoord of [...highSetIndices].sort((a, b) => a - b)) {
    const out = new Float32Array(totalVertexCount * 2);
    out.fill(Number.NaN);
    for (const range of ranges) {
      const source = sourceByPrimitiveId.get(range.sourcePrimitiveId ?? range.name)?.[texCoord];
      if (source == null) continue;
      for (let vertex = 0; vertex < range.vertexCount; vertex += 1) {
        out[(range.vertexStart + vertex) * 2] = source[vertex * 2] ?? Number.NaN;
        out[(range.vertexStart + vertex) * 2 + 1] = source[vertex * 2 + 1] ?? Number.NaN;
      }
    }
    result.set(texCoord, out);
  }
  return result;
}

function buffersFromCoreScenePack(
  scene: Scene,
  geo: ScenePackResult,
  coreMaterials: readonly MaterialSpec[],
  options: CoreBvhBuildOptions,
): SceneBVHBuffers {
  const triCount = geo.triangleCount;
  const vertCount = geo.positions.length / 4;

  // H15 — pass the real UVs from ScenePackResult.uvs (stride-4 vec4f → extract
  // uv0 as stride-2) so every vertex's .w lane carries the packed UV pair instead
  // of (0,0). Vertex ordering: packSceneFromCore emits positions and uvs in the
  // same primitive-concat order, so indices align 1:1.
  const uv0Stride2 = stride4UvsToStride2Uv0(geo.uvs, vertCount);
  const positionsWithUV = packUVIntoPositionW(geo.positions, { array: uv0Stride2 }, vertCount);
  const uv1Stride2 = stride4UvsToStride2Uv1(geo.uvs, vertCount);
  const normalsWithUV1 = packUVIntoVec4W(geo.normals, { array: uv1Stride2 }, vertCount);
  const triIndices3 = collapseIndicesToStride3(geo.indices);
  const mneeFacetDomains = packTlasMneeFacetDomains(geo.primitiveTlasBindings);
  const rawMeshVertexRanges = geo.primitiveTlasBindings.map((b) => ({
    name: b.primitiveId,
    sourcePrimitiveId: b.primitiveId,
    vertexStart: b.vertexStart,
    vertexCount: b.vertexCount,
    triStart: b.triStart,
    triCount: b.triCount,
  }));
  const highUvSets = mergeHighUvSetsFromCore(scene, rawMeshVertexRanges, vertCount);

  const indexBuf = packBVHIndexWFromCore(triIndices3, geo.triMaterialIds, coreMaterials, triCount);
  const beerBuf = packBVHBeerColorsFromCore(geo.triMaterialIds, coreMaterials, triCount);
  const materialTextureAtlas = packMaterialTextureAtlas(coreMaterials, geo.triMaterialIds, triCount, {
    indices: triIndices3,
    uv0: uv0Stride2,
    uv1: uv1Stride2,
    uvSets: highUvSets,
  });
  // B1 — per-triangle roughness+metalness lane (diffuse-default invariant inside).
  const roughMetalBuf = packBVHRoughMetalFromCore(geo.triMaterialIds, coreMaterials, triCount);
  // H23 — apply mesh-area emitter Le overrides to the emissive-Le glow buffer so
  // the camera-visible glow on an emitter-referenced mesh reflects the emitter Le.
  const emissiveCoreMats = applyMeshAreaLeOverridesToCoreMaterials(scene, coreMaterials);
  const emissiveLeBuf = packBVHEmissiveLeFromCore(geo.triMaterialIds, emissiveCoreMats, triCount);

  const emitterSlice = coreEmitterBuffers(scene, {
    ...options,
    packSourceTriIndex: true,
    tlasPrimitiveBindings: geo.primitiveTlasBindings,
  });
  // `mergedGeometry` is the CPU fallback AABB consumed by cascade/probe
  // placement. It must enclose TLAS instances too; excluding instanced meshes
  // silently clipped scenes whose only/farthest geometry lived in the TLAS.
  const merged = mergeWorldSpaceFromCore(scene, { positionStride: 4 });
  warnScenePackWarnings(options, geo.warnings);

  return {
    bvhMode: 'tlas',
    bvhNodes: makeStorageHandle(geo.bvhNodes, 32),
    bvhIndex: makeStorageHandle(indexBuf, 16),
    bvhPositions: makeStorageHandle(positionsWithUV, 16),
    triangleMaterialIds: makeStorageHandle(geo.triMaterialIds, 4),
    bvhBeerColors: makeStorageHandle(beerBuf, 4),
    bvhEmissiveLe: makeStorageHandle(emissiveLeBuf, 16),
    materialTextureAtlas,
    bvhRoughMetal: makeStorageHandle(roughMetalBuf, 4),
    bvhNormals: makeStorageHandle(normalsWithUV1, 16),
    bvhTangents: makeStorageHandle(geo.tangents, 16),
    bvhColors: makeStorageHandle(geo.colors, 16),
    mneeFacetDomains: makeStorageHandle(mneeFacetDomains, 32),
    emitters: emitterSlice.emitters,
    emitterCdf: emitterSlice.emitterCdf,
    emitterAlias: emitterSlice.emitterAlias,
    emitterCount: emitterSlice.emitterCount,
    totalEmissivePower: emitterSlice.totalEmissivePower,
    lightTree: emitterSlice.lightTree,
    lightTreeNodeCount: emitterSlice.lightTreeNodeCount,
    lightTreeEnabled: emitterSlice.lightTreeEnabled,
    mergedGeometry: makeMergedGeometry(merged.boundingBox.min, merged.boundingBox.max),
    meshVertexRanges: enrichMeshVertexRangesWithCoreMatrix(scene, rawMeshVertexRanges),
    bvhIndicesStride3: triIndices3,
    buildMaterials: [],
    coreMaterials,
    emitterNormals: geo.normals,
    tlas: {
      nodes: makeStorageHandle(geo.tlasNodes, 32),
      instanceIndices: makeStorageHandle(geo.tlasInstanceIndices, 4),
      blasRoots: makeStorageHandle(geo.tlasBlasRoots, 4),
      worldToLocal: makeStorageHandle(geo.tlasInstanceWorldToLocal, 64),
      localToWorld: makeStorageHandle(geo.tlasInstanceLocalToWorld, 64),
      nodeCount: geo.tlasNodeCount,
    },
    primitiveTlasBindings: geo.primitiveTlasBindings,
    scenePack: geo,
    warnings: geo.warnings,
  };
}

function buildReSTIRSceneBVHFromCoreTlas(
  scene: Scene,
  options: CoreBvhBuildOptions = {},
): SceneBVHBuffers {
  const { coreMaterials, resolveMaterialId } = materialResolver(scene);
  const geo = packSceneFromCore(scene, { tlas: true, resolveMaterialId });
  return buffersFromCoreScenePack(scene, geo, coreMaterials, options);
}

function buildReSTIRSceneBVHFromCoreMerged(
  scene: Scene,
  options: CoreBvhBuildOptions = {},
): SceneBVHBuffers {
  // SHADOW-01 — split material slots by the primitive castShadow flag so
  // packBVHRoughMetalFromCore can pack the bit-0 castShadowDisabled lane even
  // when two primitives share a structurally-equal material. All-default scenes
  // produce the same slot grouping + identical packed bytes.
  const merged = mergeWorldSpaceFromCore(scene, {
    positionStride: 4,
    splitMaterialsByCastShadow: true,
  });
  const triCount = merged.indices.length / 3;
  const mneeFacetDomains = packMergedMneeFacetDomains(triCount);
  const vertCount = merged.positions.length / 4;
  // H15 — pass merged.uvs (stride-2, same vertex order as merged.positions) so
  // every vertex's .w lane carries the packed UV pair.  WorldSpaceMergeResult.uvs
  // is stride-2 (u0, v0 per vertex), which is exactly what packUVIntoPositionW's
  // { array } path expects (reads array[i*2] / array[i*2+1]).
  const positionsWithUV = packUVIntoPositionW(merged.positions, { array: merged.uvs }, vertCount);
  const mergedUv1 = mergeUv1FromCore(scene, merged.meshVertexRanges, merged.vertexCount);
  const highUvSets = mergeHighUvSetsFromCore(scene, merged.meshVertexRanges, merged.vertexCount);
  const normalsWithUV1 = packUVIntoVec4W(merged.normals, mergedUv1 == null ? undefined : { array: mergedUv1 }, vertCount);
  const indexBuf = packBVHIndexWFromCore(
    merged.indices,
    merged.triMaterialId,
    merged.materials,
    triCount,
  );
  const beerBuf = packBVHBeerColorsFromCore(merged.triMaterialId, merged.materials, triCount);
  const materialTextureAtlas = packMaterialTextureAtlas(merged.materials, merged.triMaterialId, triCount, {
    indices: merged.indices,
    uv0: merged.uvs,
    ...(mergedUv1 != null ? { uv1: mergedUv1 } : {}),
    uvSets: highUvSets,
  });
  // B1 — per-triangle roughness+metalness lane (diffuse-default invariant inside).
  const roughMetalBuf = packBVHRoughMetalFromCore(merged.triMaterialId, merged.materials, triCount);
  // H23 — apply mesh-area emitter Le overrides (same as TLAS path) so the emissive
  // glow buffer reflects the emitter Le for mesh-area-referenced primitives.
  const emissiveMergedMats = buildMeshAreaLeOverrides(scene, merged.materials, options);
  const mergedMatsForEmissive = merged.materials.map((m, slot) => {
    const override = emissiveMergedMats.get(slot);
    if (override == null) return toProductionEmissiveRadiance(m);
    return applyMeshAreaLeOverride(m, override);
  });
  const emissiveLeBuf = packBVHEmissiveLeFromCore(merged.triMaterialId, mergedMatsForEmissive, triCount);
  const emitterSlice = coreEmitterBuffers(scene, {
    ...options,
    packSourceTriIndex: true,
    suppressMeshAreaMissingReferenceWarnings: true,
  });
  warnScenePackWarnings(options, merged.warnings);

  return {
    bvhMode: 'merged',
    bvhNodes: makeStorageHandle(merged.bvhNodes, 32),
    bvhIndex: makeStorageHandle(indexBuf, 16),
    bvhPositions: makeStorageHandle(positionsWithUV, 16),
    triangleMaterialIds: makeStorageHandle(merged.triMaterialId, 4),
    bvhBeerColors: makeStorageHandle(beerBuf, 4),
    bvhEmissiveLe: makeStorageHandle(emissiveLeBuf, 16),
    materialTextureAtlas,
    bvhRoughMetal: makeStorageHandle(roughMetalBuf, 4),
    bvhNormals: makeStorageHandle(normalsWithUV1, 16),
    bvhTangents: makeStorageHandle(merged.tangents, 16),
    bvhColors: makeStorageHandle(merged.colors, 16),
    mneeFacetDomains: makeStorageHandle(mneeFacetDomains, 32),
    emitters: emitterSlice.emitters,
    emitterCdf: emitterSlice.emitterCdf,
    emitterAlias: emitterSlice.emitterAlias,
    emitterCount: emitterSlice.emitterCount,
    totalEmissivePower: emitterSlice.totalEmissivePower,
    lightTree: emitterSlice.lightTree,
    lightTreeNodeCount: emitterSlice.lightTreeNodeCount,
    lightTreeEnabled: emitterSlice.lightTreeEnabled,
    mergedGeometry: makeMergedGeometry(merged.boundingBox.min, merged.boundingBox.max),
    meshVertexRanges: enrichMeshVertexRangesWithCoreMatrix(scene, merged.meshVertexRanges),
    bvhIndicesStride3: merged.indices,
    primitiveRefitNodeIndices: buildMergedPrimitiveRefitNodeIndices(merged),
    buildMaterials: [],
    coreMaterials: merged.materials,
    emitterNormals: merged.normals,
    primitiveTlasBindings: [],
    warnings: merged.warnings,
  };
}

export function buildReSTIRSceneBVHForCoreScene(
  scene: Scene,
  options: CoreBvhBuildOptions = {},
): SceneBVHBuffers {
  if (!sceneHasCoreMeshes(scene)) {
    throw new Error(
      '[HybridEngine] BVH source unavailable: concrete walkaround-hybrid requires a core Scene with mesh primitives.',
    );
  }
  const mode = resolveReSTIRBvhMode(scene, options.bvhMode);
  return mode === 'tlas'
    ? buildReSTIRSceneBVHFromCoreTlas(scene, options)
    : buildReSTIRSceneBVHFromCoreMerged(scene, options);
}

export function rebuildReSTIRSceneBVHPrimitiveCore(
  scene: Scene,
  primitiveId: string,
  prev: SceneBVHBuffers,
  options: CoreBvhBuildOptions = {},
): SceneBVHBuffers | { ok: false; reason: string } {
  if (prev.scenePack == null) {
    return { ok: false, reason: 'previous buffers have no scenePack snapshot' };
  }
  const { coreMaterials, resolveMaterialId } = materialResolver(scene);
  const rebuilt = rebuildPrimitiveBlas(scene, primitiveId, prev.scenePack, {
    tlas: true,
    resolveMaterialId,
  });
  if (!rebuilt.ok) {
    return { ok: false, reason: rebuilt.reason };
  }
  return buffersFromCoreScenePack(scene, rebuilt.pack, coreMaterials, options);
}

export function rebuildEmitterBuffersFromCoreScene(
  scene: Scene,
  options: {
    primaryLightDir?: Vector3Like;
    primaryLightIntensity?: number;
    packSourceTriIndex?: boolean;
    tlasPrimitiveBindings?: readonly PrimitiveTlasBinding[];
    onWarning?: (warning: EngineWarning) => void;
    warningPhase?: EngineWarning['phase'];
    warningMethod?: string;
  } = {},
): RebuiltEmitterBuffers {
  return coreEmitterBuffers(scene, options);
}
