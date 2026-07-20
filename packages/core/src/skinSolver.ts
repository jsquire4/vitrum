/**
 * skinSolver.ts — CPU baseline per-frame linear-blend skinning solver.
 *
 * Reads a SkinnedMeshPrimitive (rest pose + current bones / boneInverses)
 * and produces deformed `positions` + `normals` (+ optional `tangents`)
 * Float32Arrays that hosts push through `engine.updatePrimitive(...)`.
 *
 * Algorithm: linear blend skinning (LBS) per glTF 2.0 / three.js convention,
 * preceded by morph-target blending when the primitive carries blend shapes,
 * and respecting `bindMatrix` / `bindMatrixInverse` when supplied:
 *
 *   morphedPos[v]    = restPos[v]    + Σ_t morphWeights[t] · morphTargets[t][v]
 *   morphedNormal[v] = restNormal[v] + Σ_t morphWeights[t] · morphTargetNormals[t][v]
 *   morphedTangent[v]= restTangent[v]+ Σ_t morphWeights[t] · morphTargetTangents[t][v]
 *   morphedUv[v]     = restUv[v]     + Σ_t morphWeights[t] · morphTargetUvs[t][v]
 *   skinVertex[v]   = bindMatrix       · morphedPos[v]            (skip if bindMatrix omitted)
 *   skinMatrix[v]   = Σ_k weights[v,k] · ( bones[idx[v,k]] · boneInverses[idx[v,k]] )
 *   skinnedWorld[v] = skinMatrix[v]    · skinVertex[v]            (w=1 transform)
 *   deformedPos[v]  = bindMatrixInverse · skinnedWorld[v]         (skip if omitted)
 *
 * Normals transform by the INVERSE-TRANSPOSE of the position transform's
 * linear (upper-3×3) part (PBR4e §3.10) — `n' = (L⁻¹)ᵀ · n` where
 * `L = bindMatrixInverse₃ · skinMatrix₃ · bindMatrix₃`. For rigid bones
 * (rotation-only, no scale) `(L⁻¹)ᵀ == L`, so the result matches a plain
 * upper-3×3 transform; for non-uniformly-scaled or sheared bones the
 * inverse-transpose is required to keep the normal perpendicular to the
 * deformed surface. For glTF-typical use bindMatrix is identity and the
 * pipeline collapses to `deformedPos = skinMatrix · morphedPos`,
 * `deformedNormal = (skinMatrix₃⁻¹)ᵀ · morphedNormal`.
 *
 * Per-bone `combined = bones · boneInverses` is precomputed once per call
 * (boneCount 4x4 matrix muls) instead of once per vertex×bone.
 *
 * Matrix layout: column-major Float32Array of length 16 (same as
 * THREE.Matrix4.elements). Entry at row r, column c is `m[r + c*4]`.
 *
 * Cost (rough): ~30 µs/1000 verts on modern hardware for the LBS inner
 * loop, plus boneCount · ~0.5 µs for the prep. Skinning a 30k-vert mesh
 * with 50 bones is ~1 ms per frame — squarely in the budget for a single
 * hero character. A GPU compute variant (`GpuSkinningSubsystem` in
 * `@vitrum/walkaround-hybrid`) is shipped and gives ~10× headroom.
 */

import type { SkinnedMeshPrimitive } from './scene/primitives.js';

/**
 * Multiply two column-major 4x4 matrices: `out[outOff..+16] = a[aOff..+16] · b[bOff..+16]`.
 * No scratch allocation; reads + writes via offsets so callers can point all
 * three slots at sub-ranges of larger packed buffers.
 */
function mat4Mul(
  out: Float32Array, outOff: number,
  a: Float32Array, aOff: number,
  b: Float32Array, bOff: number,
): void {
  const a00 = a[aOff +  0]!, a10 = a[aOff +  1]!, a20 = a[aOff +  2]!, a30 = a[aOff +  3]!;
  const a01 = a[aOff +  4]!, a11 = a[aOff +  5]!, a21 = a[aOff +  6]!, a31 = a[aOff +  7]!;
  const a02 = a[aOff +  8]!, a12 = a[aOff +  9]!, a22 = a[aOff + 10]!, a32 = a[aOff + 11]!;
  const a03 = a[aOff + 12]!, a13 = a[aOff + 13]!, a23 = a[aOff + 14]!, a33 = a[aOff + 15]!;

  for (let c = 0; c < 4; c++) {
    const b0 = b[bOff + 0 + c * 4]!;
    const b1 = b[bOff + 1 + c * 4]!;
    const b2 = b[bOff + 2 + c * 4]!;
    const b3 = b[bOff + 3 + c * 4]!;
    out[outOff + 0 + c * 4] = a00 * b0 + a01 * b1 + a02 * b2 + a03 * b3;
    out[outOff + 1 + c * 4] = a10 * b0 + a11 * b1 + a12 * b2 + a13 * b3;
    out[outOff + 2 + c * 4] = a20 * b0 + a21 * b1 + a22 * b2 + a23 * b3;
    out[outOff + 3 + c * 4] = a30 * b0 + a31 * b1 + a32 * b2 + a33 * b3;
  }
}

/** Pre-compute combined = bones · boneInverses, packed contiguously. */
export function combineSkinMatrices(
  bones: Float32Array,
  boneInverses: Float32Array,
  boneCount: number,
): Float32Array {
  const combined = new Float32Array(boneCount * 16);
  for (let i = 0; i < boneCount; i++) {
    const off = i * 16;
    mat4Mul(combined, off, bones, off, boneInverses, off);
  }
  return combined;
}

/**
 * Inverse-transpose of a 3×3 matrix — the correct normal transform under any
 * affine deformation whose linear part is `m` (Pharr et al., PBR4e §3.10:
 * normals transform by `(M⁻¹)ᵀ`, NOT by `M`). For a rigid (rotation-only)
 * `m` this equals `m` itself, so the result is identical to the old
 * upper-3×3 direct transform on un-scaled bones; for a scaled/sheared `m`
 * (non-uniform bone scale) it diverges, and only the inverse-transpose keeps
 * the normal perpendicular to the deformed surface.
 *
 * Inputs / output are row-major 3×3s laid out as
 * `[m00,m01,m02, m10,m11,m12, m20,m21,m22]`. On a singular `m` (|det| ≈ 0,
 * e.g. a bone scaled to zero on an axis) we fall back to the input matrix so
 * the caller still produces a finite (if approximate) normal rather than NaN.
 *
 * @returns the 9-element inverse-transpose, written into `out` (allocated if
 *   omitted).
 */
export function mat3InverseTranspose(m: ArrayLike<number>, out?: Float32Array): Float32Array {
  const r = out ?? new Float32Array(9);
  const m00 = m[0]!, m01 = m[1]!, m02 = m[2]!;
  const m10 = m[3]!, m11 = m[4]!, m12 = m[5]!;
  const m20 = m[6]!, m21 = m[7]!, m22 = m[8]!;

  // Cofactors (these directly give the adjugate; det·M⁻¹ = adjugateᵀ, and the
  // matrix of cofactors IS (det · M⁻¹)ᵀᵀ = det · (M⁻¹)ᵀ). Dividing the cofactor
  // matrix by det yields (M⁻¹)ᵀ exactly — which is what we want.
  const c00 = m11 * m22 - m12 * m21;
  const c01 = m12 * m20 - m10 * m22;
  const c02 = m10 * m21 - m11 * m20;
  const c10 = m02 * m21 - m01 * m22;
  const c11 = m00 * m22 - m02 * m20;
  const c12 = m01 * m20 - m00 * m21;
  const c20 = m01 * m12 - m02 * m11;
  const c21 = m02 * m10 - m00 * m12;
  const c22 = m00 * m11 - m01 * m10;

  const det = m00 * c00 + m01 * c01 + m02 * c02;
  if (Math.abs(det) < 1e-20) {
    // Singular linear part — fall back to the raw matrix (best-effort).
    r[0] = m00; r[1] = m01; r[2] = m02;
    r[3] = m10; r[4] = m11; r[5] = m12;
    r[6] = m20; r[7] = m21; r[8] = m22;
    return r;
  }
  const inv = 1 / det;
  // (M⁻¹)ᵀ row i = cofactor row i / det.
  r[0] = c00 * inv; r[1] = c01 * inv; r[2] = c02 * inv;
  r[3] = c10 * inv; r[4] = c11 * inv; r[5] = c12 * inv;
  r[6] = c20 * inv; r[7] = c21 * inv; r[8] = c22 * inv;
  return r;
}

/**
 * Accumulate a weighted morph-target stream into `base` in place:
 * `base[i] += Σ_t weights[t] · deltas[t][i]` (skipping zero-weight targets).
 *
 * Every one of solveSkin's five morph blends (positions, normals, tangents,
 * uvs, uv1) is this same loop; the only variation was the diagnostic strings
 * and the tangent xyzw→xyz pre-step (which the caller performs when building
 * `base`). `deltaName`/`baseName` reproduce the exact per-stream error text so
 * the pinned messages (`morphSolver.test.ts`) are byte-preserved. When
 * `checkCount` is true the count mismatch (`deltas.length !== tCount`) is
 * reported as `${deltaName} length N != morphTargets tCount` — the positions
 * stream passes `false` because its count is guarded upstream via morphWeights.
 */
function blendMorphStream(
  base: Float32Array,
  deltas: readonly Float32Array[],
  weights: ArrayLike<number>,
  tCount: number,
  deltaName: string,
  baseName: string,
  checkCount: boolean,
): void {
  if (checkCount && deltas.length !== tCount) {
    throw new Error(`solveSkin: ${deltaName} length ${deltas.length} != morphTargets ${tCount}.`);
  }
  for (let t = 0; t < tCount; t++) {
    const w = weights[t]!;
    if (w === 0) continue;
    const delta = deltas[t]!;
    if (delta.length !== base.length) {
      throw new Error(
        `solveSkin: ${deltaName}[${t}] length ${delta.length} != ${baseName} ${base.length}.`,
      );
    }
    for (let i = 0; i < base.length; i++) {
      base[i] = base[i]! + w * delta[i]!;
    }
  }
}

/**
 * Solve linear-blend skinning into `outPositions` + `outNormals`. Allocates
 * fresh buffers if either is omitted; otherwise writes in-place. Returns
 * the (possibly newly-allocated) output references so callers can chain.
 */
export function solveSkin(
  prim: SkinnedMeshPrimitive,
  outPositions?: Float32Array,
  outNormals?: Float32Array,
  outTangents?: Float32Array,
  outUvs?: Float32Array,
  outUv1?: Float32Array,
): {
  positions: Float32Array;
  normals: Float32Array;
  tangents?: Float32Array;
  uvs?: Float32Array;
  uv1?: Float32Array;
} {
  const vertCount = prim.positions.length / 3;
  if (prim.normals.length !== vertCount * 3) {
    throw new Error(
      `solveSkin: normals length ${prim.normals.length} does not match positions ${prim.positions.length}.`,
    );
  }
  if (prim.skinIndices.length !== vertCount * 4) {
    throw new Error(
      `solveSkin: skinIndices length ${prim.skinIndices.length} expected ${vertCount * 4}.`,
    );
  }
  if (prim.skinWeights.length !== vertCount * 4) {
    throw new Error(
      `solveSkin: skinWeights length ${prim.skinWeights.length} expected ${vertCount * 4}.`,
    );
  }
  const boneCount = prim.bones.length / 16;
  if (boneCount === 0 || boneCount * 16 !== prim.bones.length) {
    throw new Error(`solveSkin: bones length ${prim.bones.length} not a multiple of 16.`);
  }
  if (prim.boneInverses.length !== boneCount * 16) {
    throw new Error(
      `solveSkin: boneInverses length ${prim.boneInverses.length} != bones length ${prim.bones.length}.`,
    );
  }
  for (let i = 0; i < prim.skinIndices.length; i += 1) {
    const boneIndex = prim.skinIndices[i]!;
    if (boneIndex >= boneCount) {
      throw new Error(
        `solveSkin: skinIndices[${i}] references bone ${boneIndex}, but only ${boneCount} bones exist.`,
      );
    }
  }
  for (let i = 0; i < prim.skinWeights.length; i += 1) {
    const weight = prim.skinWeights[i]!;
    if (!Number.isFinite(weight) || weight < 0) {
      throw new Error(`solveSkin: skinWeights[${i}] is invalid (${weight}).`);
    }
  }
  for (let vertex = 0; vertex < vertCount; vertex += 1) {
    const offset = vertex * 4;
    const sum =
      prim.skinWeights[offset]! +
      prim.skinWeights[offset + 1]! +
      prim.skinWeights[offset + 2]! +
      prim.skinWeights[offset + 3]!;
    if (Math.abs(sum - 1) > 1e-4) {
      throw new Error(`solveSkin: skinWeights for vertex ${vertex} sum to ${sum}; expected 1.`);
    }
  }

  const positions = outPositions ?? new Float32Array(vertCount * 3);
  const normals = outNormals ?? new Float32Array(vertCount * 3);
  const hasTangents = prim.tangents != null;
  if (hasTangents && prim.tangents.length !== vertCount * 4) {
    throw new Error(
      `solveSkin: tangents length ${prim.tangents.length} expected ${vertCount * 4}.`,
    );
  }
  const tangents = hasTangents ? (outTangents ?? new Float32Array(vertCount * 4)) : undefined;
  if (prim.uvs != null && prim.uvs.length !== vertCount * 2) {
    throw new Error(
      `solveSkin: uvs length ${prim.uvs.length} expected ${vertCount * 2}.`,
    );
  }
  if (prim.uv1 != null && prim.uv1.length !== vertCount * 2) {
    throw new Error(
      `solveSkin: uv1 length ${prim.uv1.length} expected ${vertCount * 2}.`,
    );
  }
  if (positions.length !== vertCount * 3) {
    throw new Error(`solveSkin: outPositions length ${positions.length} expected ${vertCount * 3}.`);
  }
  if (normals.length !== vertCount * 3) {
    throw new Error(`solveSkin: outNormals length ${normals.length} expected ${vertCount * 3}.`);
  }
  if (tangents != null && tangents.length !== vertCount * 4) {
    throw new Error(`solveSkin: outTangents length ${tangents.length} expected ${vertCount * 4}.`);
  }
  if (outUvs != null && outUvs.length !== vertCount * 2) {
    throw new Error(`solveSkin: outUvs length ${outUvs.length} expected ${vertCount * 2}.`);
  }
  if (outUv1 != null && outUv1.length !== vertCount * 2) {
    throw new Error(`solveSkin: outUv1 length ${outUv1.length} expected ${vertCount * 2}.`);
  }

  const combined = combineSkinMatrices(prim.bones, prim.boneInverses, boneCount);

  // ── Morph-target pre-blend ─────────────────────────────────────────────
  // If the primitive carries morph targets and at least one non-zero weight,
  // we precompute morphedPos + morphedNormal arrays once; otherwise we read
  // directly from the rest pose to skip the allocation + indirection.
  let morphedPositions: Float32Array | null = null;
  let morphedNormals: Float32Array | null = null;
  let morphedTangents: Float32Array | null = null;
  let morphedUvs: Float32Array | null = null;
  let morphedUv1: Float32Array | null = null;
  if (prim.morphTargets != null && prim.morphWeights != null && prim.morphTargets.length > 0) {
    let anyActive = false;
    for (let t = 0; t < prim.morphWeights.length; t++) {
      if (prim.morphWeights[t]! !== 0) { anyActive = true; break; }
    }
    if (anyActive) {
      const tCount = prim.morphTargets.length;
      if (prim.morphWeights.length !== tCount) {
        throw new Error(
          `solveSkin: morphWeights length ${prim.morphWeights.length} != morphTargets ${tCount}.`,
        );
      }
      const mp: Float32Array = new Float32Array(prim.positions);
      morphedPositions = mp;
      blendMorphStream(mp, prim.morphTargets, prim.morphWeights, tCount, 'morphTargets', 'positions', false);
      if (prim.morphTargetNormals != null) {
        const mn: Float32Array = new Float32Array(prim.normals);
        morphedNormals = mn;
        blendMorphStream(mn, prim.morphTargetNormals, prim.morphWeights, tCount, 'morphTargetNormals', 'normals', true);
      }
      if (prim.morphTargetTangents != null && prim.tangents != null) {
        // Pre-step: tangents are stored xyzw (w = handedness); the morph blend
        // operates on the xyz direction only, so pack a vertCount*3 base first.
        const mt = new Float32Array(vertCount * 3);
        for (let v = 0; v < vertCount; v += 1) {
          mt[v * 3] = prim.tangents[v * 4]!;
          mt[v * 3 + 1] = prim.tangents[v * 4 + 1]!;
          mt[v * 3 + 2] = prim.tangents[v * 4 + 2]!;
        }
        morphedTangents = mt;
        blendMorphStream(mt, prim.morphTargetTangents, prim.morphWeights, tCount, 'morphTargetTangents', 'tangents', true);
      }
      if (prim.morphTargetUvs != null) {
        if (prim.uvs == null) {
          throw new Error('solveSkin: morphTargetUvs supplied but primitive has no uvs stream.');
        }
        const mu = new Float32Array(prim.uvs);
        morphedUvs = mu;
        blendMorphStream(mu, prim.morphTargetUvs, prim.morphWeights, tCount, 'morphTargetUvs', 'uvs', true);
      }
      if (prim.morphTargetUv1s != null) {
        if (prim.uv1 == null) {
          throw new Error('solveSkin: morphTargetUv1s supplied but primitive has no uv1 stream.');
        }
        const mu1 = new Float32Array(prim.uv1);
        morphedUv1 = mu1;
        blendMorphStream(mu1, prim.morphTargetUv1s, prim.morphWeights, tCount, 'morphTargetUv1s', 'uv1', true);
      }
    }
  }
  const restPositions = morphedPositions ?? prim.positions;
  const restNormals = morphedNormals ?? prim.normals;
  const restTangents = morphedTangents;

  // Bind-matrix support: when the SkinnedMesh was bound with a non-identity
  // bindMatrix, we must pre-transform rest positions to bind-pose-world
  // space, apply the LBS, then untransform back. glTF-typical hosts skip
  // these branches because `bindMatrix` is identity (omitted).
  const bm = prim.bindMatrix;
  const bmi = prim.bindMatrixInverse;
  const hasBind = bm != null && bmi != null;
  if (hasBind && (bm.length !== 16 || bmi.length !== 16)) {
    throw new Error(
      `solveSkin: bindMatrix / bindMatrixInverse must be 16-element column-major matrices.`,
    );
  }

  // Per-vertex accumulation. We do not allocate a per-vertex 4x4 skinMatrix;
  // instead we accumulate the 12 entries needed for point + direction
  // transforms directly (m00..m23, omitting the projective row).
  // Scratch for the per-vertex normal inverse-transpose (row-major 3×3 in,
  // 9-element out). Reused across vertices to avoid per-vertex allocation.
  const linRowMajor = new Float32Array(9);
  const normalMat = new Float32Array(9);

  for (let v = 0; v < vertCount; v++) {
    const px = restPositions[v * 3 + 0]!;
    const py = restPositions[v * 3 + 1]!;
    const pz = restPositions[v * 3 + 2]!;
    // Rest-pose normal (pre-bind, pre-skin). Normals are transformed by the
    // inverse-transpose of the FULL position-linear-part below, so we keep the
    // un-transformed rest normal here rather than incrementally pushing it
    // through bind / skin stages (which only happened to be correct for rigid
    // bones in the pre-inverse-transpose code).
    const nx = restNormals[v * 3 + 0]!;
    const ny = restNormals[v * 3 + 1]!;
    const nz = restNormals[v * 3 + 2]!;
    const hasVertexTangent = tangents != null && prim.tangents != null;
    const tx0 = restTangents?.[v * 3 + 0] ?? prim.tangents?.[v * 4 + 0] ?? 0;
    const ty0 = restTangents?.[v * 3 + 1] ?? prim.tangents?.[v * 4 + 1] ?? 0;
    const tz0 = restTangents?.[v * 3 + 2] ?? prim.tangents?.[v * 4 + 2] ?? 0;

    // Position in (possibly bind-pre-multiplied) space for the skin transform.
    let bpx = px, bpy = py, bpz = pz;
    if (hasBind) {
      bpx = bm[0]! * px + bm[4]! * py + bm[8]! * pz + bm[12]!;
      bpy = bm[1]! * px + bm[5]! * py + bm[9]! * pz + bm[13]!;
      bpz = bm[2]! * px + bm[6]! * py + bm[10]! * pz + bm[14]!;
    }

    // Accumulator entries: skin[r,c] for r in 0..3, c in 0..4 (we only need
    // the top 3 rows). Indexed below by `s<r><c>` where r ∈ {0,1,2}.
    let s00 = 0, s01 = 0, s02 = 0, s03 = 0;
    let s10 = 0, s11 = 0, s12 = 0, s13 = 0;
    let s20 = 0, s21 = 0, s22 = 0, s23 = 0;

    for (let k = 0; k < 4; k++) {
      const w = prim.skinWeights[v * 4 + k]!;
      if (w === 0) continue;
      const bIdx = prim.skinIndices[v * 4 + k]!;
      const off = bIdx * 16;
      // Column-major: m[r + c*4]
      s00 += w * combined[off + 0]!;
      s10 += w * combined[off + 1]!;
      s20 += w * combined[off + 2]!;
      s01 += w * combined[off + 4]!;
      s11 += w * combined[off + 5]!;
      s21 += w * combined[off + 6]!;
      s02 += w * combined[off + 8]!;
      s12 += w * combined[off + 9]!;
      s22 += w * combined[off + 10]!;
      s03 += w * combined[off + 12]!;
      s13 += w * combined[off + 13]!;
      s23 += w * combined[off + 14]!;
    }

    // Position transform (w = 1) → skinned-world space.
    let outX = s00 * bpx + s01 * bpy + s02 * bpz + s03;
    let outY = s10 * bpx + s11 * bpy + s12 * bpz + s13;
    let outZ = s20 * bpx + s21 * bpy + s22 * bpz + s23;

    // ── Normal transform via inverse-transpose of the combined linear part ──
    // Build L = the upper-3×3 of (bindMatrixInverse · skinMatrix · bindMatrix)
    // — the exact linear part the position transform applies to a direction.
    // For the glTF-typical identity-bind case this is just skinMatrix₃.
    // Row-major layout [m00,m01,m02, m10,m11,m12, m20,m21,m22].
    if (hasBind) {
      // bm3 (upper-3×3 of bindMatrix), column-major source.
      const b00 = bm[0]!, b01 = bm[4]!, b02 = bm[8]!;
      const b10 = bm[1]!, b11 = bm[5]!, b12 = bm[9]!;
      const b20 = bm[2]!, b21 = bm[6]!, b22 = bm[10]!;
      // sb = skinMatrix₃ · bm3.
      const sb00 = s00 * b00 + s01 * b10 + s02 * b20;
      const sb01 = s00 * b01 + s01 * b11 + s02 * b21;
      const sb02 = s00 * b02 + s01 * b12 + s02 * b22;
      const sb10 = s10 * b00 + s11 * b10 + s12 * b20;
      const sb11 = s10 * b01 + s11 * b11 + s12 * b21;
      const sb12 = s10 * b02 + s11 * b12 + s12 * b22;
      const sb20 = s20 * b00 + s21 * b10 + s22 * b20;
      const sb21 = s20 * b01 + s21 * b11 + s22 * b21;
      const sb22 = s20 * b02 + s21 * b12 + s22 * b22;
      // L = bmi3 · sb.
      const i00 = bmi[0]!, i01 = bmi[4]!, i02 = bmi[8]!;
      const i10 = bmi[1]!, i11 = bmi[5]!, i12 = bmi[9]!;
      const i20 = bmi[2]!, i21 = bmi[6]!, i22 = bmi[10]!;
      linRowMajor[0] = i00 * sb00 + i01 * sb10 + i02 * sb20;
      linRowMajor[1] = i00 * sb01 + i01 * sb11 + i02 * sb21;
      linRowMajor[2] = i00 * sb02 + i01 * sb12 + i02 * sb22;
      linRowMajor[3] = i10 * sb00 + i11 * sb10 + i12 * sb20;
      linRowMajor[4] = i10 * sb01 + i11 * sb11 + i12 * sb21;
      linRowMajor[5] = i10 * sb02 + i11 * sb12 + i12 * sb22;
      linRowMajor[6] = i20 * sb00 + i21 * sb10 + i22 * sb20;
      linRowMajor[7] = i20 * sb01 + i21 * sb11 + i22 * sb21;
      linRowMajor[8] = i20 * sb02 + i21 * sb12 + i22 * sb22;
    } else {
      linRowMajor[0] = s00; linRowMajor[1] = s01; linRowMajor[2] = s02;
      linRowMajor[3] = s10; linRowMajor[4] = s11; linRowMajor[5] = s12;
      linRowMajor[6] = s20; linRowMajor[7] = s21; linRowMajor[8] = s22;
    }
    mat3InverseTranspose(linRowMajor, normalMat);
    let dnx = normalMat[0]! * nx + normalMat[1]! * ny + normalMat[2]! * nz;
    let dny = normalMat[3]! * nx + normalMat[4]! * ny + normalMat[5]! * nz;
    let dnz = normalMat[6]! * nx + normalMat[7]! * ny + normalMat[8]! * nz;

    // Post-multiply position by bindMatrixInverse to return to mesh-local space.
    if (hasBind) {
      const tx = bmi[0]! * outX + bmi[4]! * outY + bmi[8]! * outZ + bmi[12]!;
      const ty = bmi[1]! * outX + bmi[5]! * outY + bmi[9]! * outZ + bmi[13]!;
      const tz = bmi[2]! * outX + bmi[6]! * outY + bmi[10]! * outZ + bmi[14]!;
      outX = tx; outY = ty; outZ = tz;
    }

    positions[v * 3 + 0] = outX;
    positions[v * 3 + 1] = outY;
    positions[v * 3 + 2] = outZ;
    const invLen = 1 / Math.sqrt(dnx * dnx + dny * dny + dnz * dnz + 1e-20);
    dnx *= invLen; dny *= invLen; dnz *= invLen;
    normals[v * 3 + 0] = dnx;
    normals[v * 3 + 1] = dny;
    normals[v * 3 + 2] = dnz;
    if (hasVertexTangent && tangents != null && prim.tangents != null) {
      let dtx = linRowMajor[0] * tx0 + linRowMajor[1] * ty0 + linRowMajor[2] * tz0;
      let dty = linRowMajor[3] * tx0 + linRowMajor[4] * ty0 + linRowMajor[5] * tz0;
      let dtz = linRowMajor[6] * tx0 + linRowMajor[7] * ty0 + linRowMajor[8] * tz0;
      const tInvLen = 1 / Math.sqrt(dtx * dtx + dty * dty + dtz * dtz + 1e-20);
      dtx *= tInvLen; dty *= tInvLen; dtz *= tInvLen;
      tangents[v * 4 + 0] = dtx;
      tangents[v * 4 + 1] = dty;
      tangents[v * 4 + 2] = dtz;
      tangents[v * 4 + 3] = prim.tangents[v * 4 + 3] ?? 1;
    }
  }

  const result: {
    positions: Float32Array;
    normals: Float32Array;
    tangents?: Float32Array;
    uvs?: Float32Array;
    uv1?: Float32Array;
  } = tangents != null ? { positions, normals, tangents } : { positions, normals };
  if (morphedUvs != null) {
    const uvs = outUvs ?? morphedUvs;
    if (uvs !== morphedUvs) uvs.set(morphedUvs);
    result.uvs = uvs;
  }
  if (morphedUv1 != null) {
    const uv1 = outUv1 ?? morphedUv1;
    if (uv1 !== morphedUv1) uv1.set(morphedUv1);
    result.uv1 = uv1;
  }
  return result;
}
