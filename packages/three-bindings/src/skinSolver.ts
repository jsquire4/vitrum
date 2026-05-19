/**
 * skinSolver.ts — CPU baseline per-frame linear-blend skinning solver.
 *
 * Reads a SkinnedMeshPrimitive (rest pose + current bones / boneInverses)
 * and produces deformed `positions` + `normals` Float32Arrays that hosts
 * push through `engine.updatePrimitive(id, { positions, normals })`.
 *
 * Algorithm: linear blend skinning (LBS) per glTF 2.0 / three.js convention,
 * preceded by morph-target blending when the primitive carries blend shapes:
 *
 *   morphedPos[v]    = restPos[v]    + Σ_t morphWeights[t] · morphTargets[t][v]
 *   morphedNormal[v] = restNormal[v] + Σ_t morphWeights[t] · morphTargetNormals[t][v]
 *   skinMatrix[v] = Σ_k weights[v,k] · ( bones[idx[v,k]] · boneInverses[idx[v,k]] )
 *   deformedPos[v]    = skinMatrix[v] · morphedPos[v]              (w=1 transform)
 *   deformedNormal[v] = mat3(skinMatrix[v]) · morphedNormal[v]     (w=0 transform)
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
 * hero character. The GPU compute variant (deferred) gives ~10× headroom.
 */

import type { SkinnedMeshPrimitive } from '@vitrum/core';

/** Multiply two column-major 4x4 matrices into `out` (`a · b`). */
function mat4Mul(out: Float32Array, a: Float32Array, b: Float32Array, bOff: number): void {
  // Loop unrolled for clarity; not the perf bottleneck (called once per bone).
  const a00 = a[0]!, a10 = a[1]!, a20 = a[2]!, a30 = a[3]!;
  const a01 = a[4]!, a11 = a[5]!, a21 = a[6]!, a31 = a[7]!;
  const a02 = a[8]!, a12 = a[9]!, a22 = a[10]!, a32 = a[11]!;
  const a03 = a[12]!, a13 = a[13]!, a23 = a[14]!, a33 = a[15]!;

  for (let c = 0; c < 4; c++) {
    const b0 = b[bOff + 0 + c * 4]!;
    const b1 = b[bOff + 1 + c * 4]!;
    const b2 = b[bOff + 2 + c * 4]!;
    const b3 = b[bOff + 3 + c * 4]!;
    out[0 + c * 4] = a00 * b0 + a01 * b1 + a02 * b2 + a03 * b3;
    out[1 + c * 4] = a10 * b0 + a11 * b1 + a12 * b2 + a13 * b3;
    out[2 + c * 4] = a20 * b0 + a21 * b1 + a22 * b2 + a23 * b3;
    out[3 + c * 4] = a30 * b0 + a31 * b1 + a32 * b2 + a33 * b3;
  }
}

/** Pre-compute combined = bones * boneInverses, packed contiguously. */
function combineSkinMatrices(
  bones: Float32Array,
  boneInverses: Float32Array,
  boneCount: number,
): Float32Array {
  const combined = new Float32Array(boneCount * 16);
  const bonesSlice = new Float32Array(16);
  for (let i = 0; i < boneCount; i++) {
    const off = i * 16;
    for (let k = 0; k < 16; k++) bonesSlice[k] = bones[off + k]!;
    const outView = combined.subarray(off, off + 16);
    // mat4Mul writes into `out` directly indexed [0..16); use a temporary
    // 16-length view to satisfy that contract.
    mat4Mul(outView, bonesSlice, boneInverses, off);
  }
  return combined;
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
): { positions: Float32Array; normals: Float32Array } {
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

  const positions = outPositions ?? new Float32Array(vertCount * 3);
  const normals = outNormals ?? new Float32Array(vertCount * 3);
  if (positions.length !== vertCount * 3) {
    throw new Error(`solveSkin: outPositions length ${positions.length} expected ${vertCount * 3}.`);
  }
  if (normals.length !== vertCount * 3) {
    throw new Error(`solveSkin: outNormals length ${normals.length} expected ${vertCount * 3}.`);
  }

  const combined = combineSkinMatrices(prim.bones, prim.boneInverses, boneCount);

  // ── Morph-target pre-blend ─────────────────────────────────────────────
  // If the primitive carries morph targets and at least one non-zero weight,
  // we precompute morphedPos + morphedNormal arrays once; otherwise we read
  // directly from the rest pose to skip the allocation + indirection.
  let morphedPositions: Float32Array | null = null;
  let morphedNormals: Float32Array | null = null;
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
      for (let t = 0; t < tCount; t++) {
        const w = prim.morphWeights[t]!;
        if (w === 0) continue;
        const delta = prim.morphTargets[t]!;
        if (delta.length !== mp.length) {
          throw new Error(
            `solveSkin: morphTargets[${t}] length ${delta.length} != positions ${mp.length}.`,
          );
        }
        for (let i = 0; i < mp.length; i++) {
          mp[i] = mp[i]! + w * delta[i]!;
        }
      }
      if (prim.morphTargetNormals != null && prim.morphTargetNormals.length === tCount) {
        const mn: Float32Array = new Float32Array(prim.normals);
        morphedNormals = mn;
        for (let t = 0; t < tCount; t++) {
          const w = prim.morphWeights[t]!;
          if (w === 0) continue;
          const delta = prim.morphTargetNormals[t]!;
          if (delta.length !== mn.length) {
            throw new Error(
              `solveSkin: morphTargetNormals[${t}] length ${delta.length} != normals ${mn.length}.`,
            );
          }
          for (let i = 0; i < mn.length; i++) {
            mn[i] = mn[i]! + w * delta[i]!;
          }
        }
      }
    }
  }
  const restPositions = morphedPositions ?? prim.positions;
  const restNormals = morphedNormals ?? prim.normals;

  // Per-vertex accumulation. We do not allocate a per-vertex 4x4 skinMatrix;
  // instead we accumulate the 12 entries needed for point + direction
  // transforms directly (m00..m23, omitting the projective row).
  for (let v = 0; v < vertCount; v++) {
    const px = restPositions[v * 3 + 0]!;
    const py = restPositions[v * 3 + 1]!;
    const pz = restPositions[v * 3 + 2]!;
    const nx = restNormals[v * 3 + 0]!;
    const ny = restNormals[v * 3 + 1]!;
    const nz = restNormals[v * 3 + 2]!;

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

    // Position transform (w = 1).
    positions[v * 3 + 0] = s00 * px + s01 * py + s02 * pz + s03;
    positions[v * 3 + 1] = s10 * px + s11 * py + s12 * pz + s13;
    positions[v * 3 + 2] = s20 * px + s21 * py + s22 * pz + s23;

    // Normal transform (w = 0). Uses the upper 3x3 of the skin matrix
    // directly; for rigid bones (rotations only) this is correct. For
    // scaled bones an inverse-transpose would be needed — a future
    // refinement when we ship scaled-bone test scenes.
    let dnx = s00 * nx + s01 * ny + s02 * nz;
    let dny = s10 * nx + s11 * ny + s12 * nz;
    let dnz = s20 * nx + s21 * ny + s22 * nz;
    const invLen = 1 / Math.sqrt(dnx * dnx + dny * dny + dnz * dnz + 1e-20);
    dnx *= invLen; dny *= invLen; dnz *= invLen;
    normals[v * 3 + 0] = dnx;
    normals[v * 3 + 1] = dny;
    normals[v * 3 + 2] = dnz;
  }

  return { positions, normals };
}
