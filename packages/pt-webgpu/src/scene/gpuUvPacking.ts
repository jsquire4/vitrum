import type { Scene, ScenePrimitive } from '@vitrum/core';
import { resolveDisplacedGeometry } from '@vitrum/shared-bvh';

export interface GpuUvRange {
  readonly vertexStart: number;
  readonly vertexCount: number;
  readonly primitiveId?: string;
  readonly sourcePrimitiveId?: string;
  readonly name?: string;
}

function rangePrimitiveId(range: GpuUvRange): string | undefined {
  return range.primitiveId ?? range.sourcePrimitiveId ?? range.name;
}

function isMeshLikePrimitive(
  primitive: ScenePrimitive,
): primitive is Extract<
  ScenePrimitive,
  { kind: 'mesh' | 'skinned-mesh' | 'instanced-mesh' }
> {
  return primitive.kind === 'mesh' ||
    primitive.kind === 'skinned-mesh' ||
    primitive.kind === 'instanced-mesh';
}

function validateUvLayout(uvSetTexCoords: readonly number[]): void {
  if (uvSetTexCoords.length < 2 || uvSetTexCoords[0] !== 0 || uvSetTexCoords[1] !== 1) {
    throw new Error('pt-webgpu GPU UV layout must begin with authored TEXCOORD_0 and TEXCOORD_1.');
  }
  const seen = new Set<number>();
  for (const texCoord of uvSetTexCoords) {
    if (!Number.isSafeInteger(texCoord) || texCoord < 0 || seen.has(texCoord)) {
      throw new Error(
        `pt-webgpu GPU UV layout contains an invalid or duplicate texCoord (${String(texCoord)}).`,
      );
    }
    seen.add(texCoord);
  }
}

/**
 * Expand the historical one-vec4-per-vertex UV buffer into a compact scalable
 * layout without adding a bind-group slot:
 *
 *   record [0, vertexCount)                         = uv0.xy / uv1.zw
 *   record vertexCount + (gpuSlot - 2)*vertexCount = gpuSlot.xy
 *
 * `uvSetTexCoords[gpuSlot]` maps the compact slot back to the authored sparse
 * `TextureRef.texCoord`. Thus TEXCOORD_8192 costs one tail plane rather than
 * 8191 empty planes. Missing primitive streams remain zero-filled.
 */
export function packGpuUvSets(
  scene: Scene,
  primaryUvs: Float32Array,
  ranges: readonly GpuUvRange[],
  uvSetTexCoords: readonly number[],
): Float32Array {
  validateUvLayout(uvSetTexCoords);
  if (primaryUvs.length % 4 !== 0) {
    throw new Error(
      `pt-webgpu primary UV buffer length ${primaryUvs.length} is not vec4-aligned.`,
    );
  }
  if (uvSetTexCoords.length === 2) return primaryUvs;

  const vertexCount = primaryUvs.length / 4;
  const out = new Float32Array(primaryUvs.length * (uvSetTexCoords.length - 1));
  out.set(primaryUvs);

  const resolvedById = new Map<
    string,
    ReturnType<typeof resolveDisplacedGeometry>
  >();
  for (const primitive of scene.primitives) {
    if (!isMeshLikePrimitive(primitive)) continue;
    resolvedById.set(primitive.id, resolveDisplacedGeometry(primitive, () => {}));
  }

  for (const range of ranges) {
    const primitiveId = rangePrimitiveId(range);
    if (primitiveId == null) continue;
    const resolved = resolvedById.get(primitiveId);
    if (resolved == null) continue;
    const copyCount = Math.min(
      Math.max(0, range.vertexCount),
      Math.max(0, vertexCount - range.vertexStart),
    );
    for (let gpuSlot = 2; gpuSlot < uvSetTexCoords.length; gpuSlot += 1) {
      const authoredTexCoord = uvSetTexCoords[gpuSlot]!;
      const source = resolved.baseUvSets?.[authoredTexCoord];
      if (source == null) continue;
      const sourceVertexCount = Math.floor(source.length / 2);
      const vertices = Math.min(copyCount, sourceVertexCount);
      const planeStart = vertexCount + (gpuSlot - 2) * vertexCount;
      for (let localVertex = 0; localVertex < vertices; localVertex += 1) {
        const record = planeStart + range.vertexStart + localVertex;
        out[record * 4] = source[localVertex * 2] ?? 0;
        out[record * 4 + 1] = source[localVertex * 2 + 1] ?? 0;
      }
    }
  }
  return out;
}
