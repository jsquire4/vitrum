import type { MaterialSpec, Scene } from '@vitrum/core';
import type { PrimitiveTlasBinding } from '@vitrum/shared-bvh';
import { ROUGH_DIELECTRIC_SMOOTH_THRESHOLD } from '../math/roughDielectric.js';

/** Bit marker stored in the first vec4 after the real analytic parameters. */
export const MNEE_FACET_TABLE_MAGIC = 0x4d4e4545;
export const MNEE_GUIDED_FACET_PROBABILITY = 0.5;

/**
 * Conservative host mirror of the production Dirac-interface predicate.
 * Texture-backed scalar channels are treated as potentially active because the
 * opaque texture handle cannot be exhaustively classified during scene packing.
 */
export function materialMayProduceMneeDelta(material: MaterialSpec): boolean {
  if (material.thinFilmStack != null) return true;

  const mayTransmit =
    (material.transmission ?? 0) > 0 ||
    material.transmissionMap != null;
  const mayBeNonMetal = material.metallic < 1 || material.metallicMap != null;
  const faceMayBeSmooth =
    (material.frontLayer?.roughness != null &&
      material.frontLayer.roughness <= ROUGH_DIELECTRIC_SMOOTH_THRESHOLD) ||
    (material.backLayer?.roughness != null &&
      material.backLayer.roughness <= ROUGH_DIELECTRIC_SMOOTH_THRESHOLD);
  const mayBeSmooth =
    material.roughness <= ROUGH_DIELECTRIC_SMOOTH_THRESHOLD ||
    material.roughnessMap != null ||
    faceMayBeSmooth;

  return mayTransmit && mayBeNonMetal && mayBeSmooth;
}

/**
 * CPU oracle for the shader's guided/uniform mixture.
 *
 * A missing guide renormalizes the surviving uniform branch to one. When the
 * guide exists, the selected identity receives its uniform mass plus the guide
 * mass iff it is the guided identity.
 */
export function mneeConditionalFacetPmf(
  candidateCount: number,
  guidedCandidateIndex: number | null,
  selectedCandidateIndex: number,
): number {
  if (!Number.isInteger(candidateCount) || candidateCount <= 0) return 0;
  if (
    !Number.isInteger(selectedCandidateIndex) ||
    selectedCandidateIndex < 0 ||
    selectedCandidateIndex >= candidateCount
  ) return 0;
  if (guidedCandidateIndex == null) return 1 / candidateCount;
  if (
    !Number.isInteger(guidedCandidateIndex) ||
    guidedCandidateIndex < 0 ||
    guidedCandidateIndex >= candidateCount
  ) return 0;
  const uniformMass = (1 - MNEE_GUIDED_FACET_PROBABILITY) / candidateCount;
  return uniformMass +
    (selectedCandidateIndex === guidedCandidateIndex
      ? MNEE_GUIDED_FACET_PROBABILITY
      : 0);
}

/** Rejection threshold for an unbiased u32 draw in [0, bound). */
export function mneeBoundedU32Threshold(bound: number): number {
  if (!Number.isInteger(bound) || bound <= 0 || bound > 0xffff_ffff) {
    throw new RangeError('MNEE u32 bound must be an integer in [1, 2^32 - 1].');
  }
  return ((0xffff_ffff % bound) + 1) % bound;
}

/**
 * CPU oracle for the shader's modulo-rejection map. Null means draw again.
 * Inputs stay as JS-safe integers; the result covers the complete u32 domain.
 */
export function mneeBoundedU32Index(word: number, bound: number): number | null {
  if (!Number.isInteger(word) || word < 0 || word > 0xffff_ffff) {
    throw new RangeError('MNEE random word must be a u32.');
  }
  const threshold = mneeBoundedU32Threshold(bound);
  return word < threshold ? null : word % bound;
}

/**
 * CPU oracle for the scale-aware ray offset/tolerance used by manifold NEE.
 */
export function mneeScaleAwareEpsilon(
  triIntersectEpsilon: number,
  point: readonly [number, number, number],
  distanceHint: number,
): number {
  const coordinateScale = Math.max(Math.abs(point[0]), Math.abs(point[1]), Math.abs(point[2]));
  const distanceTolerance = Math.abs(distanceHint) * 4 * 2 ** -23;
  const ulpTolerance = coordinateScale * 4 * 2 ** -23;
  return Math.max(
    Math.max(triIntersectEpsilon, 0),
    distanceTolerance,
    ulpTolerance,
    2 ** -126,
  );
}

export interface MneeSolverToleranceOracle {
  readonly coordinateScale: number;
  readonly localSpan: number;
  readonly lengthFloor: number;
  readonly fdStep: number;
  readonly residualTolerance: number;
  readonly representable: boolean;
}

/** CPU oracle for the production Newton finite-difference/convergence scales. */
export function mneeSolverToleranceOracle(
  triIntersectEpsilon: number,
  points: readonly (readonly [number, number, number])[],
): MneeSolverToleranceOracle {
  let coordinateScale = 0;
  let localSpan = 0;
  for (let i = 0; i < points.length; i += 1) {
    const point = points[i]!;
    coordinateScale = Math.max(
      coordinateScale,
      Math.abs(point[0]),
      Math.abs(point[1]),
      Math.abs(point[2]),
    );
    for (let j = 0; j < i; j += 1) {
      const other = points[j]!;
      localSpan = Math.max(
        localSpan,
        Math.hypot(
          point[0] - other[0],
          point[1] - other[1],
          point[2] - other[2],
        ),
      );
    }
  }
  const f32Epsilon = 2 ** -23;
  const lengthFloor = Math.max(
    Math.max(triIntersectEpsilon, 0),
    coordinateScale * 4 * f32Epsilon,
    localSpan * 4 * f32Epsilon,
    2 ** -126,
  );
  return {
    coordinateScale,
    localSpan,
    lengthFloor,
    fdStep: Math.max(lengthFloor, localSpan * Math.sqrt(f32Epsilon)),
    residualTolerance: Math.max(
      16 * f32Epsilon,
      Math.min(lengthFloor / Math.max(localSpan, lengthFloor), 0.01),
    ),
    representable:
      Number.isFinite(coordinateScale) &&
      Number.isFinite(localSpan) &&
      localSpan > 2 ** -126 &&
      lengthFloor < localSpan * 0.01,
  };
}

type MneeMeshPrimitive = Extract<
  Scene['primitives'][number],
  { readonly kind: 'mesh' | 'instanced-mesh' | 'skinned-mesh' }
>;

function materialHasMneeNormalPerturbation(material: MaterialSpec): boolean {
  return (
    (material.normalMap != null && (material.normalScale ?? 1) !== 0) ||
    (material.bumpMap != null && (material.bumpScale ?? 1) !== 0) ||
    (material.frontLayer?.normalMap != null &&
      (material.frontLayer.normalScale ?? 1) !== 0) ||
    (material.backLayer?.normalMap != null &&
      (material.backLayer.normalScale ?? 1) !== 0)
  );
}

function firstNonGeometricNormalTriangle(primitive: MneeMeshPrimitive): number | null {
  const indices = primitive.indices;
  const triangleCount = Math.floor(
    (indices?.length ?? primitive.positions.length / 3) / 3,
  );
  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    const i0 = indices?.[triangle * 3] ?? triangle * 3;
    const i1 = indices?.[triangle * 3 + 1] ?? triangle * 3 + 1;
    const i2 = indices?.[triangle * 3 + 2] ?? triangle * 3 + 2;
    const ax = primitive.positions[i0 * 3];
    const ay = primitive.positions[i0 * 3 + 1];
    const az = primitive.positions[i0 * 3 + 2];
    const bx = primitive.positions[i1 * 3];
    const by = primitive.positions[i1 * 3 + 1];
    const bz = primitive.positions[i1 * 3 + 2];
    const cx = primitive.positions[i2 * 3];
    const cy = primitive.positions[i2 * 3 + 1];
    const cz = primitive.positions[i2 * 3 + 2];
    if (
      ax == null || ay == null || az == null ||
      bx == null || by == null || bz == null ||
      cx == null || cy == null || cz == null
    ) return triangle;
    const e0x = bx - ax;
    const e0y = by - ay;
    const e0z = bz - az;
    const e1x = cx - ax;
    const e1y = cy - ay;
    const e1z = cz - az;
    const fx = e0y * e1z - e0z * e1y;
    const fy = e0z * e1x - e0x * e1z;
    const fz = e0x * e1y - e0y * e1x;
    const faceLength = Math.hypot(fx, fy, fz);
    if (!(faceLength > 1e-20)) continue;
    const faceX = fx / faceLength;
    const faceY = fy / faceLength;
    const faceZ = fz / faceLength;
    for (const vertexIndex of [i0, i1, i2]) {
      const nx = primitive.normals[vertexIndex * 3];
      const ny = primitive.normals[vertexIndex * 3 + 1];
      const nz = primitive.normals[vertexIndex * 3 + 2];
      if (nx == null || ny == null || nz == null) return triangle;
      const normalLength = Math.hypot(nx, ny, nz);
      if (!(normalLength > 1e-20)) return triangle;
      const alignment =
        (nx * faceX + ny * faceY + nz * faceZ) / normalLength;
      if (alignment < 1 - 1e-4) return triangle;
    }
  }
  return null;
}

/**
 * Fail closed outside the implemented planar/geometric-normal manifold domain.
 * This is intentionally called on the CPU-solved scene before any GPU upload.
 */
export function assertMneeInterfaceDomainSupported(scene: Scene): void {
  const unsupported = scene.primitives.flatMap((primitive) =>
    primitive.kind === 'analytic' && materialMayProduceMneeDelta(primitive.material)
      ? [`${primitive.id} (${primitive.shape})`]
      : [],
  );
  if (unsupported.length > 0) {
    throw new Error(
      '@vitrum/pt-webgpu: causticStrategy="manifold-nee" supports planar mesh, ' +
        'instanced-mesh, and skinned-mesh delta interfaces. Analytic delta interfaces ' +
        'require a curved-surface manifold parameterization and are rejected before GPU mutation: ' +
        unsupported.join(', ') + '.',
    );
  }

  const shadingNormalViolations: string[] = [];
  for (const primitive of scene.primitives) {
    if (primitive.kind === 'analytic' || !materialMayProduceMneeDelta(primitive.material)) {
      continue;
    }
    if (materialHasMneeNormalPerturbation(primitive.material)) {
      shadingNormalViolations.push(
        `${primitive.id} (normal/bump/layer-normal map)`,
      );
      continue;
    }
    const triangle = firstNonGeometricNormalTriangle(primitive);
    if (triangle != null) {
      shadingNormalViolations.push(
        `${primitive.id} (triangle ${triangle} has a varying vertex normal)`,
      );
    }
  }
  if (shadingNormalViolations.length > 0) {
    throw new Error(
      '@vitrum/pt-webgpu: causticStrategy="manifold-nee" currently solves ' +
        'planar geometric-normal delta interfaces. Varying vertex normals and ' +
        'normal/bump/layer-normal maps require a varying-normal manifold Jacobian ' +
        'and are rejected before GPU mutation: ' +
        shadingNormalViolations.join(', ') + '.',
    );
  }
}

export interface MneeFacetCandidateTable {
  /** One vec4 metadata record followed by one vec4 per candidate. */
  readonly records: Float32Array;
  readonly candidateCount: number;
  readonly requiredBytes: number;
}

/**
 * Build the compact, exact TLAS membership table used by the shader proposal.
 * Each record bit-packs `(globalTriangleIndex, globalTlasInstanceIndex)` as u32
 * words in x/y. This removes the old triangle x instance Cartesian domain and
 * its invalid BLAS pairs without imposing a fixed content-count ceiling.
 */
export function buildMneeFacetCandidateTable(
  scene: Scene,
  bindings: readonly PrimitiveTlasBinding[],
  storageLimitBytes: number,
  prefixBytes = 0,
): MneeFacetCandidateTable {
  const primitiveById = new Map(scene.primitives.map((primitive) => [primitive.id, primitive]));
  let candidateCount = 0;
  for (const binding of bindings) {
    const primitive = primitiveById.get(binding.primitiveId);
    if (primitive == null || !materialMayProduceMneeDelta(primitive.material)) continue;
    candidateCount += binding.triCount * binding.instanceCount;
    if (!Number.isSafeInteger(candidateCount) || candidateCount > 0xffff_ffff) {
      throw new Error(
        '@vitrum/pt-webgpu: manifold-nee facet candidate count exceeds the u32 shader table domain.',
      );
    }
  }

  const tableBytes = (candidateCount + 1) * 16;
  const requiredBytes = prefixBytes + tableBytes;
  if (!Number.isSafeInteger(requiredBytes) || requiredBytes > storageLimitBytes) {
    throw new Error(
      '@vitrum/pt-webgpu: manifold-nee facet candidate table requires ' +
        `${requiredBytes} bytes, but this device permits ${storageLimitBytes} bytes ` +
        'for the shared storage-buffer allocation.',
    );
  }

  const records = new Float32Array((candidateCount + 1) * 4);
  const words = new Uint32Array(records.buffer);
  words[0] = MNEE_FACET_TABLE_MAGIC;
  words[1] = candidateCount;

  let writeRecord = 1;
  let globalInstance = 0;
  for (const binding of bindings) {
    const primitive = primitiveById.get(binding.primitiveId);
    const eligible = primitive != null && materialMayProduceMneeDelta(primitive.material);
    if (eligible) {
      for (let localInstance = 0; localInstance < binding.instanceCount; localInstance += 1) {
        const instanceIndex = globalInstance + localInstance;
        for (let localTri = 0; localTri < binding.triCount; localTri += 1) {
          const triIndex = binding.triStart + localTri;
          const wordBase = writeRecord * 4;
          words[wordBase] = triIndex >>> 0;
          words[wordBase + 1] = instanceIndex >>> 0;
          writeRecord += 1;
        }
      }
    }
    globalInstance += binding.instanceCount;
  }

  if (writeRecord !== candidateCount + 1) {
    throw new Error('@vitrum/pt-webgpu: internal manifold-nee facet table packing mismatch.');
  }
  return { records, candidateCount, requiredBytes };
}
