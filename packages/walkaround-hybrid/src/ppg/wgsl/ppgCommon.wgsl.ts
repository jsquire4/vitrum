/**
 * Shared PPG WGSL helpers consumed by ppgSample and ppgUpdate.
 *
 * Each consumer module must declare the storage bindings for
 * `ppgCells` and `ppgKdNodes` before concatenating this string.
 * This module declares only functions that operate on those bindings.
 *
 * Decision: a single canonical kd-tree traversal eliminates the ~75-line
 * duplication that existed between ppgSample.wgsl.ts (ppgKdFindCell) and
 * ppgUpdate.wgsl.ts (ppgUpdateKdFindCell). Both now call
 * `ppgKdFindCellShared(worldPos, cellCount)`.
 */

export const PPG_COMMON_WGSL = /* wgsl */ `
fn ppgAxisComp(v: vec3f, axis: u32) -> f32 {
  if (axis == 0u) { return v.x; }
  if (axis == 1u) { return v.y; }
  return v.z;
}

fn ppgBruteFindCell(worldPos: vec3f, cellCount: u32) -> u32 {
  if (cellCount == 0u) { return 0u; }
  var bestIdx  = 0u;
  var bestDist2 = 1e38;
  for (var i = 0u; i < cellCount; i++) {
    let d = ppgCells[i].position - worldPos;
    let dist2 = dot(d, d);
    if (dist2 < bestDist2) {
      bestDist2 = dist2;
      bestIdx  = i;
    }
  }
  return bestIdx;
}

// Nearest-neighbour kd-tree traversal (iterative; stack prunes far subtree).
// Falls back to brute search when the tree is unbuilt (single leaf root) or
// the traversal stack would overflow.
fn ppgKdFindCellShared(worldPos: vec3f, cellCount: u32) -> u32 {
  let nk = arrayLength(&ppgKdNodes);
  if (nk == 0u || cellCount == 0u) { return 0u; }
  let root = ppgKdNodes[0];
  if (root.child0 == 0xFFFFFFFFu && root.child1 == 0xFFFFFFFFu) {
    return ppgBruteFindCell(worldPos, cellCount);
  }

  var bestIdx  = 0u;
  var bestDist2 = 1e38;

  var stN: array<u32, 48>;
  var stK: array<u32, 48>;
  var stFar: array<u32, 48>;
  var stD2: array<f32, 48>;
  var sp = 0u;

  stN[sp] = 0u;
  stK[sp] = 0u;
  stFar[sp] = 0u;
  stD2[sp] = 0.0;
  sp = sp + 1u;

  while (sp > 0u) {
    sp = sp - 1u;
    if (stK[sp] == 1u) {
      if (stD2[sp] < bestDist2 && sp < 48u) {
        stN[sp] = stFar[sp];
        stK[sp] = 0u;
        sp = sp + 1u;
      }
      continue;
    }

    let nid = stN[sp];
    if (nid >= nk) { continue; }
    let node = ppgKdNodes[nid];
    // WGSL reserves meta as a keyword; rename the local to nodeMeta.
    let nodeMeta = node.meta;
    if ((nodeMeta & 0x80000000u) != 0u) {
      let cellIdx = nodeMeta & 0x7FFFFFFFu;
      if (cellIdx < cellCount) {
        let d = ppgCells[cellIdx].position - worldPos;
        let dist2 = dot(d, d);
        if (dist2 < bestDist2) {
          bestDist2 = dist2;
          bestIdx = cellIdx;
        }
      }
      continue;
    }

    let axis = nodeMeta & 3u;
    let split = node.split;
    let c0 = node.child0;
    let c1 = node.child1;
    let d0 = ppgAxisComp(worldPos, axis) - split;
    let d2plane = d0 * d0;
    let nearI = select(c1, c0, d0 < 0.0);
    let farI = select(c0, c1, d0 < 0.0);

    if (sp + 2u > 48u) {
      return ppgBruteFindCell(worldPos, cellCount);
    }
    stFar[sp] = farI;
    stD2[sp] = d2plane;
    stK[sp] = 1u;
    sp = sp + 1u;
    stN[sp] = nearI;
    stK[sp] = 0u;
    stFar[sp] = 0u;
    stD2[sp] = 0.0;
    sp = sp + 1u;
  }
  return bestIdx;
}
`;
