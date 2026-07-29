import {
  BVH_NODE_FLOATS,
  BVH_TRAVERSAL_STACK_DEPTH,
} from './strides.js';

const LEAF_NODE_HIGH_WORD = 0xffff;

export interface BvhEncodingValidationOptions {
  /**
   * Roots of every concatenated BLAS tree. A standalone BVH defaults to root
   * zero. Scene packers must supply every primitive root.
   */
  readonly roots?: readonly number[];
  /**
   * Exact triangle-record capacity consumed by leaf offsets. When supplied,
   * leaf ranges must tile [0, triangleCount) exactly once.
   */
  readonly triangleCount?: number;
  /** Private-stack capacity of the consuming shader traversal. */
  readonly traversalStackCapacity?: number;
}

export interface BvhEncodingProof {
  readonly nodeCount: number;
  readonly rootCount: number;
  readonly interiorNodeCount: number;
  readonly leafNodeCount: number;
  readonly maxDepth: number;
  readonly maxTraversalStackEntries: number;
  readonly traversalStackCapacity: number;
  readonly triangleCount?: number;
}

function validationError(message: string): Error {
  return new Error(
    `[@vitrum/shared-bvh] validateBvhEncoding: ${message}`,
  );
}

function finiteSafeCount(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw validationError(`${label} must be a non-negative safe integer.`);
  }
}

/**
 * Prove that a packed binary BVH is structurally safe for vitrum's fixed-stack
 * WGSL traversals.
 *
 * In addition to the relative-right-child encoding, this validates buffer
 * extent, roots/reachability, acyclic exclusive ownership, split axes, finite
 * ordered/enclosing bounds, leaf triangle ranges, exact triangle coverage, and
 * the maximum live DFS stack. Publication paths call this unconditionally.
 */
export function validateBvhEncoding(
  nodeBytes: Float32Array | Uint32Array,
  totalNodes: number,
  options: BvhEncodingValidationOptions = {},
): BvhEncodingProof {
  finiteSafeCount(totalNodes, 'totalNodes');
  if (nodeBytes.length !== totalNodes * BVH_NODE_FLOATS) {
    throw validationError(
      `node buffer has ${nodeBytes.length} words; expected exactly ` +
        `${totalNodes * BVH_NODE_FLOATS} for ${totalNodes} node(s).`,
    );
  }
  const triangleCount = options.triangleCount;
  if (triangleCount != null) finiteSafeCount(triangleCount, 'triangleCount');
  const traversalStackCapacity =
    options.traversalStackCapacity ?? BVH_TRAVERSAL_STACK_DEPTH;
  if (
    !Number.isSafeInteger(traversalStackCapacity) ||
    traversalStackCapacity < 1
  ) {
    throw validationError(
      'traversalStackCapacity must be a positive safe integer.',
    );
  }

  const roots = options.roots == null
    ? (totalNodes === 0 ? [] : [0])
    : Array.from(options.roots);
  if (totalNodes === 0) {
    if (roots.length !== 0) {
      throw validationError('an empty node buffer cannot declare roots.');
    }
    if (triangleCount != null && triangleCount !== 0) {
      throw validationError(
        `empty node buffer cannot cover ${triangleCount} triangle(s).`,
      );
    }
    return {
      nodeCount: 0,
      rootCount: 0,
      interiorNodeCount: 0,
      leafNodeCount: 0,
      maxDepth: 0,
      maxTraversalStackEntries: 0,
      traversalStackCapacity,
      ...(triangleCount == null ? {} : { triangleCount }),
    };
  }
  if (roots.length === 0) {
    throw validationError('non-empty node buffer must declare at least one root.');
  }

  const u32 = nodeBytes instanceof Uint32Array
    ? nodeBytes
    : new Uint32Array(
        nodeBytes.buffer,
        nodeBytes.byteOffset,
        nodeBytes.length,
      );
  const f32 = nodeBytes instanceof Float32Array
    ? nodeBytes
    : new Float32Array(
        nodeBytes.buffer,
        nodeBytes.byteOffset,
        nodeBytes.length,
      );
  const visited = new Uint8Array(totalNodes);
  const leafRanges: Array<readonly [start: number, count: number]> = [];
  const rootSeen = new Set<number>();
  let interiorNodeCount = 0;
  let leafNodeCount = 0;
  let maxDepth = 0;
  let maxTraversalStackEntries = 0;

  for (const root of roots) {
    if (!Number.isSafeInteger(root) || root < 0 || root >= totalNodes) {
      throw validationError(
        `root ${String(root)} is outside [0, ${totalNodes - 1}].`,
      );
    }
    if (rootSeen.has(root)) {
      throw validationError(`root ${root} is declared more than once.`);
    }
    rootSeen.add(root);

    const stack: Array<{ readonly node: number; readonly depth: number }> = [
      { node: root, depth: 0 },
    ];
    while (stack.length > 0) {
      const current = stack.pop()!;
      const node = current.node;
      const depth = current.depth;
      if (visited[node] !== 0) {
        throw validationError(
          `node ${node} is reachable more than once (cycle, shared child, or ` +
            'overlapping concatenated roots).',
        );
      }
      visited[node] = 1;
      maxDepth = Math.max(maxDepth, depth);
      // Child visit order is ray-dependent. The arbitrary-order worst case is
      // the deepest root-to-leaf path plus the one pending sibling retained at
      // every ancestor. A single left-first CPU walk can materially
      // underestimate this for a deep-right/shallow-left tree.
      maxTraversalStackEntries = Math.max(
        maxTraversalStackEntries,
        depth + 1,
      );
      if (maxTraversalStackEntries > traversalStackCapacity) {
        throw validationError(
          `traversal requires ${maxTraversalStackEntries} live stack entries, ` +
            `exceeding shader capacity ${traversalStackCapacity}.`,
        );
      }
      const base = node * BVH_NODE_FLOATS;
      const minX = f32[base]!;
      const minY = f32[base + 1]!;
      const minZ = f32[base + 2]!;
      const maxX = f32[base + 3]!;
      const maxY = f32[base + 4]!;
      const maxZ = f32[base + 5]!;
      if (
        !Number.isFinite(minX) ||
        !Number.isFinite(minY) ||
        !Number.isFinite(minZ) ||
        !Number.isFinite(maxX) ||
        !Number.isFinite(maxY) ||
        !Number.isFinite(maxZ)
      ) {
        throw validationError(`node ${node} has non-finite bounds.`);
      }
      if (minX > maxX || minY > maxY || minZ > maxZ) {
        throw validationError(`node ${node} has inverted bounds.`);
      }

      const splitOrCount = u32[base + 7]!;
      const isLeaf = (splitOrCount >>> 16) === LEAF_NODE_HIGH_WORD;
      if (isLeaf) {
        leafNodeCount += 1;
        const count = splitOrCount & 0xffff;
        const offset = u32[base + 6]!;
        if (triangleCount != null) {
          if (offset > triangleCount || count > triangleCount - offset) {
            throw validationError(
              `leaf node ${node} triangle range [${offset}, ${offset + count}) ` +
                `exceeds triangle capacity ${triangleCount}.`,
            );
          }
          leafRanges.push([offset, count]);
        }
        continue;
      }

      interiorNodeCount += 1;
      if (splitOrCount > 2) {
        throw validationError(
          `interior node ${node} has invalid split axis ${splitOrCount}.`,
        );
      }
      const rightOffset = u32[base + 6]!;
      const leftChild = node + 1;
      const rightChild = node + rightOffset;
      if (
        rightOffset <= 1 ||
        leftChild >= totalNodes ||
        rightChild >= totalNodes
      ) {
        throw validationError(
          `interior node ${node} has invalid relative right-child offset ` +
            `${rightOffset}; child indices would be ${leftChild} and ` +
            `${rightChild} for ${totalNodes} node(s).`,
        );
      }
      for (const child of [leftChild, rightChild]) {
        const childBase = child * BVH_NODE_FLOATS;
        const childMinX = f32[childBase]!;
        const childMinY = f32[childBase + 1]!;
        const childMinZ = f32[childBase + 2]!;
        const childMaxX = f32[childBase + 3]!;
        const childMaxY = f32[childBase + 4]!;
        const childMaxZ = f32[childBase + 5]!;
        if (
          childMinX < minX ||
          childMinY < minY ||
          childMinZ < minZ ||
          childMaxX > maxX ||
          childMaxY > maxY ||
          childMaxZ > maxZ
        ) {
          throw validationError(
            `interior node ${node} bounds do not enclose child ${child}.`,
          );
        }
      }

      stack.push({ node: rightChild, depth: depth + 1 });
      stack.push({ node: leftChild, depth: depth + 1 });
    }
  }

  let visitedNodeCount = 0;
  for (const value of visited) visitedNodeCount += value;
  if (visitedNodeCount !== totalNodes) {
    throw validationError(
      `${totalNodes - visitedNodeCount} node(s) are unreachable from the ` +
        `${roots.length} declared root(s).`,
    );
  }
  if (triangleCount != null) {
    leafRanges.sort((a, b) => a[0] - b[0]);
    let triangleCursor = 0;
    for (const [start, count] of leafRanges) {
      if (start !== triangleCursor) {
        const problem = start < triangleCursor ? 'overlaps' : 'leaves a gap after';
        throw validationError(
          `leaf triangle range [${start}, ${start + count}) ${problem} ` +
            `triangle ${triangleCursor}.`,
        );
      }
      triangleCursor += count;
    }
    if (triangleCursor !== triangleCount) {
      throw validationError(
        `leaf ranges cover ${triangleCursor} triangle(s); expected ` +
          `${triangleCount}.`,
      );
    }
  }

  return {
    nodeCount: totalNodes,
    rootCount: roots.length,
    interiorNodeCount,
    leafNodeCount,
    maxDepth,
    maxTraversalStackEntries,
    traversalStackCapacity,
    ...(triangleCount == null ? {} : { triangleCount }),
  };
}
