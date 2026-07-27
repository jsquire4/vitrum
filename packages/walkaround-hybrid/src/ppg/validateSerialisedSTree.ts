import {
  DTREE_HEADER_F32,
  DTREE_NODE_F32,
  STREE_HEADER_F32,
  STREE_NODE_F32,
  type SerialisedSTree,
} from './serialise.js';

export interface SerialisedSTreeValidationOptions {
  maxSpatialCells: number;
  maxDTreeNodesPerCell: number;
  sceneBounds: {
    min: readonly [number, number, number];
    max: readonly [number, number, number];
  };
  epsilon?: number;
}

const FOUR_PI = 4 * Math.PI;

function reject(message: string): never {
  throw new RangeError(`Invalid PPG snapshot: ${message}`);
}

function requireFinite(value: number, label: string): number {
  if (!Number.isFinite(value)) reject(`${label} must be finite`);
  return value;
}

function requireInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value)) reject(`${label} must be a safe integer`);
  return value;
}

function near(a: number, b: number, epsilon: number): boolean {
  return Math.abs(a - b) <= epsilon * Math.max(1, Math.abs(a), Math.abs(b));
}

function requireNear(a: number, b: number, epsilon: number, label: string): void {
  if (!near(a, b, epsilon)) reject(`${label} mismatch (${a} != ${b})`);
}

function requireSameF32(a: number, b: number, label: string): void {
  if (
    !Number.isFinite(a) ||
    !Number.isFinite(b) ||
    Math.fround(a) !== Math.fround(b)
  ) {
    reject(`${label} mismatch (${a} != ${b})`);
  }
}

/**
 * Validate every structural, numeric, and capacity invariant consumed by the
 * PPG CPU and WGSL traversals. This function must run before deserialisation or
 * any GPU upload; malformed external buffers are never treated as trusted tree
 * objects merely because their TypeScript shape happens to match.
 */
export function validateSerialisedSTree(
  serialised: SerialisedSTree,
  options: SerialisedSTreeValidationOptions,
): void {
  const { sTreeBuf, dTreeBuf, dTreeOffsets } = serialised;
  const epsilon = options.epsilon ?? 1e-5;
  if (!(sTreeBuf instanceof Float32Array)) reject('sTreeBuf must be Float32Array');
  if (!(dTreeBuf instanceof Float32Array)) reject('dTreeBuf must be Float32Array');
  if (!(dTreeOffsets instanceof Uint32Array)) reject('dTreeOffsets must be Uint32Array');
  if (!Number.isSafeInteger(options.maxSpatialCells) || options.maxSpatialCells < 1) {
    reject('maxSpatialCells must be a positive safe integer');
  }
  if (!Number.isSafeInteger(options.maxDTreeNodesPerCell) || options.maxDTreeNodesPerCell < 1) {
    reject('maxDTreeNodesPerCell must be a positive safe integer');
  }
  if (!Number.isFinite(epsilon) || epsilon <= 0) reject('epsilon must be finite and positive');
  for (let axis = 0; axis < 3; axis++) {
    const lo = requireFinite(options.sceneBounds.min[axis]!, `sceneBounds.min[${axis}]`);
    const hi = requireFinite(options.sceneBounds.max[axis]!, `sceneBounds.max[${axis}]`);
    if (!(lo < hi)) reject(`sceneBounds axis ${axis} must have min < max`);
  }

  if (sTreeBuf.length < STREE_HEADER_F32) reject('sTreeBuf header is truncated');
  const spatialNodeCount = requireInteger(sTreeBuf[0]!, 'sTree node count');
  const dTreeCount = requireInteger(sTreeBuf[1]!, 'dTree count');
  if (spatialNodeCount < 1 || dTreeCount < 1) reject('tree counts must be positive');
  if (dTreeCount > options.maxSpatialCells) reject('dTree count exceeds maxSpatialCells');
  if (spatialNodeCount > options.maxSpatialCells * 2 - 1) reject('sTree node count exceeds binary-tree capacity');
  if (spatialNodeCount !== dTreeCount * 2 - 1) reject('sTree node/leaf count relation is invalid');
  if (sTreeBuf.length !== STREE_HEADER_F32 + spatialNodeCount * STREE_NODE_F32) {
    reject('sTreeBuf length does not match its header');
  }
  if (dTreeOffsets.length !== dTreeCount) reject('dTreeOffsets length does not match dTree count');
  if (sTreeBuf[2] !== 0 || sTreeBuf[3] !== 0) {
    reject('sTree reserved header words must be zero');
  }

  const sVisited = new Uint8Array(spatialNodeCount);
  const dTreeSeen = new Uint8Array(dTreeCount);
  let leafCount = 0;
  const sStack: Array<[number, number]> = [[0, 0]];
  while (sStack.length > 0) {
    const [index, depth] = sStack.pop()!;
    if (index < 0 || index >= spatialNodeCount) reject(`sTree child index ${index} is out of range`);
    if (depth > 31) reject('sTree depth exceeds the WGSL traversal limit');
    if (sVisited[index] !== 0) reject(`sTree node ${index} is cyclic or multiply referenced`);
    sVisited[index] = 1;
    const base = STREE_HEADER_F32 + index * STREE_NODE_F32;
    for (let field = 0; field < 11; field++) requireFinite(sTreeBuf[base + field]!, `sTree node ${index} field ${field}`);
    for (let field = 11; field < STREE_NODE_F32; field++) {
      if (sTreeBuf[base + field] !== 0) {
        reject(`sTree node ${index} reserved field ${field} must be zero`);
      }
    }
    const min = [sTreeBuf[base]!, sTreeBuf[base + 1]!, sTreeBuf[base + 2]!] as const;
    const max = [sTreeBuf[base + 4]!, sTreeBuf[base + 5]!, sTreeBuf[base + 6]!] as const;
    for (let axis = 0; axis < 3; axis++) if (!(min[axis]! < max[axis]!)) reject(`sTree node ${index} has an empty AABB`);
    const splitAxis = requireInteger(sTreeBuf[base + 7]!, `sTree node ${index} splitAxis`);
    const left = requireInteger(sTreeBuf[base + 8]!, `sTree node ${index} leftChild`);
    const right = requireInteger(sTreeBuf[base + 9]!, `sTree node ${index} rightChild`);
    const dTreeIndex = requireInteger(sTreeBuf[base + 10]!, `sTree node ${index} dTreeIndex`);
    if (splitAxis === -1) {
      if (left !== -1 || right !== -1) reject(`sTree leaf ${index} has child pointers`);
      if (dTreeIndex < 0 || dTreeIndex >= dTreeCount) reject(`sTree leaf ${index} has an invalid dTree index`);
      if (dTreeSeen[dTreeIndex] !== 0) reject(`dTree index ${dTreeIndex} is referenced by multiple leaves`);
      dTreeSeen[dTreeIndex] = 1;
      leafCount++;
      continue;
    }
    if (splitAxis !== 0 && splitAxis !== 1 && splitAxis !== 2) reject(`sTree node ${index} has an invalid split axis`);
    if (dTreeIndex !== -1) reject(`sTree interior ${index} has a dTree index`);
    if (left === right || left < 0 || right < 0 || left >= spatialNodeCount || right >= spatialNodeCount) {
      reject(`sTree interior ${index} has invalid children`);
    }
    const split = sTreeBuf[base + 3]!;
    if (!(split > min[splitAxis] && split < max[splitAxis])) reject(`sTree interior ${index} split is outside its AABB`);
    validateSpatialChild(sTreeBuf, index, left, splitAxis, split, true);
    validateSpatialChild(sTreeBuf, index, right, splitAxis, split, false);
    sStack.push([right, depth + 1], [left, depth + 1]);
  }
  if (leafCount !== dTreeCount) reject('reachable sTree leaf count does not match dTree count');
  if (sVisited.some((value) => value === 0)) reject('sTree contains unreachable nodes');
  if (dTreeSeen.some((value) => value === 0)) reject('sTree dTree-index mapping is not a permutation');
  const rootBase = STREE_HEADER_F32;
  for (let axis = 0; axis < 3; axis++) {
    requireSameF32(
      sTreeBuf[rootBase + axis]!,
      options.sceneBounds.min[axis]!,
      `root min axis ${axis}`,
    );
    requireSameF32(
      sTreeBuf[rootBase + 4 + axis]!,
      options.sceneBounds.max[axis]!,
      `root max axis ${axis}`,
    );
  }

  let expectedOffset = 0;
  for (let treeIndex = 0; treeIndex < dTreeCount; treeIndex++) {
    const offset = dTreeOffsets[treeIndex]!;
    if (offset !== expectedOffset) reject(`dTree offset ${treeIndex} is not contiguous`);
    if (offset + DTREE_HEADER_F32 > dTreeBuf.length) reject(`dTree ${treeIndex} header is truncated`);
    const nodeCount = requireInteger(dTreeBuf[offset]!, `dTree ${treeIndex} node count`);
    const headerLeafCount = requireInteger(dTreeBuf[offset + 1]!, `dTree ${treeIndex} leaf count`);
    const totalFlux = requireFinite(dTreeBuf[offset + 2]!, `dTree ${treeIndex} total flux`);
    if (dTreeBuf[offset + 3] !== 0) {
      reject(`dTree ${treeIndex} reserved header word must be zero`);
    }
    if (nodeCount < 1 || nodeCount > options.maxDTreeNodesPerCell) reject(`dTree ${treeIndex} node count exceeds its cap`);
    if (headerLeafCount < 1 || headerLeafCount > nodeCount) reject(`dTree ${treeIndex} leaf count is invalid`);
    if (totalFlux < 0) reject(`dTree ${treeIndex} total flux is negative`);
    const blockEnd = offset + DTREE_HEADER_F32 + nodeCount * DTREE_NODE_F32;
    if (blockEnd > dTreeBuf.length) reject(`dTree ${treeIndex} block is truncated`);

    const visited = new Uint8Array(nodeCount);
    const stack: Array<[number, number]> = [[0, 0]];
    let actualLeaves = 0;
    let leafFlux = 0;
    while (stack.length > 0) {
      const [nodeIndex, depth] = stack.pop()!;
      if (nodeIndex < 0 || nodeIndex >= nodeCount) reject(`dTree ${treeIndex} child index ${nodeIndex} is out of range`);
      if (depth > 31) reject(`dTree ${treeIndex} depth exceeds the WGSL traversal limit`);
      if (visited[nodeIndex] !== 0) reject(`dTree ${treeIndex} node ${nodeIndex} is cyclic or multiply referenced`);
      visited[nodeIndex] = 1;
      const base = offset + DTREE_HEADER_F32 + nodeIndex * DTREE_NODE_F32;
      for (let field = 0; field < DTREE_NODE_F32; field++) requireFinite(dTreeBuf[base + field]!, `dTree ${treeIndex} node ${nodeIndex} field ${field}`);
      const u0 = dTreeBuf[base]!, v0 = dTreeBuf[base + 1]!, u1 = dTreeBuf[base + 2]!, v1 = dTreeBuf[base + 3]!;
      const flux = dTreeBuf[base + 4]!, solidAngle = dTreeBuf[base + 5]!;
      const firstChild = requireInteger(dTreeBuf[base + 6]!, `dTree ${treeIndex} node ${nodeIndex} firstChild`);
      const leafFlag = dTreeBuf[base + 7]!;
      if (u0 < 0 || v0 < 0 || u1 > 1 || v1 > 1 || !(u0 < u1) || !(v0 < v1)) reject(`dTree ${treeIndex} node ${nodeIndex} has invalid UV bounds`);
      if (flux < 0) reject(`dTree ${treeIndex} node ${nodeIndex} has negative flux`);
      if (leafFlag === 1) {
        if (firstChild !== -1) reject(`dTree ${treeIndex} leaf ${nodeIndex} has children`);
        requireNear(solidAngle, FOUR_PI * (u1 - u0) * (v1 - v0), epsilon, `dTree ${treeIndex} leaf ${nodeIndex} solid angle`);
        actualLeaves++;
        leafFlux += flux;
      } else if (leafFlag === 0) {
        if (firstChild <= nodeIndex || firstChild + 3 >= nodeCount) reject(`dTree ${treeIndex} interior ${nodeIndex} has invalid children`);
        if (!(solidAngle <= 0)) reject(`dTree ${treeIndex} interior ${nodeIndex} has a positive solid angle`);
        validateDirectionalChildren(dTreeBuf, offset, treeIndex, nodeIndex, firstChild);
        let childFlux = 0;
        for (let child = 0; child < 4; child++) childFlux += dTreeBuf[offset + DTREE_HEADER_F32 + (firstChild + child) * DTREE_NODE_F32 + 4]!;
        requireNear(flux, childFlux, epsilon, `dTree ${treeIndex} interior ${nodeIndex} subtree flux`);
        stack.push([firstChild + 3, depth + 1], [firstChild + 2, depth + 1], [firstChild + 1, depth + 1], [firstChild, depth + 1]);
      } else {
        reject(`dTree ${treeIndex} node ${nodeIndex} has a non-boolean leaf flag`);
      }
    }
    if (visited.some((value) => value === 0)) reject(`dTree ${treeIndex} contains unreachable nodes`);
    if (actualLeaves !== headerLeafCount) reject(`dTree ${treeIndex} leaf count header is inconsistent`);
    requireNear(totalFlux, leafFlux, epsilon, `dTree ${treeIndex} total flux`);
    requireNear(totalFlux, dTreeBuf[offset + DTREE_HEADER_F32 + 4]!, epsilon, `dTree ${treeIndex} root flux`);
    expectedOffset = blockEnd;
  }
  if (expectedOffset !== dTreeBuf.length) reject('dTreeBuf has trailing or missing data');
}

function validateSpatialChild(
  buffer: Float32Array,
  parentIndex: number,
  childIndex: number,
  splitAxis: 0 | 1 | 2,
  split: number,
  left: boolean,
): void {
  const parent = STREE_HEADER_F32 + parentIndex * STREE_NODE_F32;
  const child = STREE_HEADER_F32 + childIndex * STREE_NODE_F32;
  for (let axis = 0; axis < 3; axis++) {
    const parentMin = buffer[parent + axis]!, parentMax = buffer[parent + 4 + axis]!;
    const expectedMin = axis === splitAxis && !left ? split : parentMin;
    const expectedMax = axis === splitAxis && left ? split : parentMax;
    requireSameF32(
      buffer[child + axis]!,
      expectedMin,
      `sTree child ${childIndex} min axis ${axis}`,
    );
    requireSameF32(
      buffer[child + 4 + axis]!,
      expectedMax,
      `sTree child ${childIndex} max axis ${axis}`,
    );
  }
}

function validateDirectionalChildren(
  buffer: Float32Array,
  offset: number,
  treeIndex: number,
  parentIndex: number,
  firstChild: number,
): void {
  const parent = offset + DTREE_HEADER_F32 + parentIndex * DTREE_NODE_F32;
  const u0 = buffer[parent]!, v0 = buffer[parent + 1]!, u1 = buffer[parent + 2]!, v1 = buffer[parent + 3]!;
  const uMid = (u0 + u1) * 0.5, vMid = (v0 + v1) * 0.5;
  const expected = [
    [u0, v0, uMid, vMid], [uMid, v0, u1, vMid],
    [u0, vMid, uMid, v1], [uMid, vMid, u1, v1],
  ] as const;
  for (let childOffset = 0; childOffset < 4; childOffset++) {
    const child = offset + DTREE_HEADER_F32 + (firstChild + childOffset) * DTREE_NODE_F32;
    for (let field = 0; field < 4; field++) {
      requireSameF32(
        buffer[child + field]!,
        expected[childOffset]![field]!,
        `dTree ${treeIndex} child bounds of node ${parentIndex}`,
      );
    }
  }
}
