/**
 * CPU-side 3D kd-tree over PPG spatial cell centroids for GPU nearest-cell lookup.
 *
 * Flat 16-byte nodes (WGSL `struct PPGKdNode`) written to `ppgKdBuffer`.
 * See `ppgSample.wgsl.ts` / `ppgUpdate.wgsl.ts` for traversal (O(log N) typical).
 */

import { PPG_KD_NODE_BYTE_STRIDE, PPG_KD_MAX_NODES, PPG_KD_LEAF_FLAG } from './types.js';

/**
 * Sentinel root node: both children UINT32_MAX ⇒ shaders fall back to brute-force scan.
 * Written by `createPPGBuffers` until the host uploads a real tree.
 */
export function encodePpgKdDisabledRoot(): Uint8Array {
  const u = new Uint32Array(4);
  u[0] = 0xff_ff_ff_ff;
  u[1] = 0xff_ff_ff_ff;
  u[2] = 0;
  u[3] = 0;
  return new Uint8Array(u.buffer);
}

interface KdGpuNode {
  readonly child0: number;
  readonly child1: number;
  readonly meta: number;
  readonly split: number;
}

function packNode(n: KdGpuNode): Uint8Array {
  const buf = new ArrayBuffer(PPG_KD_NODE_BYTE_STRIDE);
  const u = new DataView(buf);
  u.setUint32(0, n.child0 >>> 0, true);
  u.setUint32(4, n.child1 >>> 0, true);
  u.setUint32(8, n.meta >>> 0, true);
  u.setFloat32(12, n.split, true);
  return new Uint8Array(buf);
}

function getPositions(
  cells: ReadonlyArray<{ readonly position: readonly [number, number, number] }>,
  count: number,
): Float32Array {
  const p = new Float32Array(Math.max(count, 1) * 3);
  for (let i = 0; i < count; i++) {
    const pos = cells[i]!.position;
    p[i * 3] = pos[0];
    p[i * 3 + 1] = pos[1];
    p[i * 3 + 2] = pos[2];
  }
  return p;
}

/**
 * Build a kd-tree over cell indices [0 .. count-1] and return packed GPU bytes.
 * Root node index is always **0** (pre-order flat allocation).
 */
export function buildPpgKdTreeGpuBytes(
  cells: ReadonlyArray<{ readonly position: readonly [number, number, number] }>,
  count: number,
): Uint8Array {
  if (count <= 0) {
    return new Uint8Array(encodePpgKdDisabledRoot());
  }
  const positions = getPositions(cells, count);
  const indices: number[] = Array.from({ length: count }, (_, i) => i);
  const nodes: KdGpuNode[] = [];

  function build(sub: number[]): number {
    if (sub.length === 1) {
      const cellIdx = sub[0]!;
      const idx = nodes.length;
      nodes.push({
        child0: 0,
        child1: 0,
        meta: PPG_KD_LEAF_FLAG | (cellIdx >>> 0),
        split: 0,
      });
      return idx;
    }

    let axis = 0;
    let bestSpan = -1;
    for (let a = 0; a < 3; a++) {
      let minV = Infinity;
      let maxV = -Infinity;
      for (const i of sub) {
        const v = positions[i * 3 + a]!;
        minV = Math.min(minV, v);
        maxV = Math.max(maxV, v);
      }
      const span = maxV - minV;
      if (span > bestSpan) {
        bestSpan = span;
        axis = a;
      }
    }

    sub.sort((ia, ib) => positions[ia * 3 + axis]! - positions[ib * 3 + axis]!);
    const mid = Math.floor(sub.length / 2);
    const left = sub.slice(0, mid);
    const right = sub.slice(mid);
    const ia = left[left.length - 1]!;
    const ib = right[0]!;
    const split = 0.5 * (positions[ia * 3 + axis]! + positions[ib * 3 + axis]!);

    const myIdx = nodes.length;
    nodes.push({ child0: 0, child1: 0, meta: 0, split: 0 });
    const leftChild = build(left);
    const rightChild = build(right);
    nodes[myIdx] = { child0: leftChild, child1: rightChild, meta: axis & 3, split };
    return myIdx;
  }

  build(indices);

  const nodeBytes = nodes.length * PPG_KD_NODE_BYTE_STRIDE;
  const cap = PPG_KD_MAX_NODES * PPG_KD_NODE_BYTE_STRIDE;
  if (nodeBytes > cap) {
    throw new RangeError(
      `buildPpgKdTreeGpuBytes: node count ${nodes.length} exceeds cap ${PPG_KD_MAX_NODES}`,
    );
  }

  const out = new Uint8Array(nodeBytes);
  for (let i = 0; i < nodes.length; i++) {
    out.set(packNode(nodes[i]!), i * PPG_KD_NODE_BYTE_STRIDE);
  }
  return out;
}

/**
 * Brute-force nearest cell (CPU reference for tests). Matches legacy WGSL tie-breaking:
 * strict `<` on squared distance.
 */
export function ppgNearestCellIndexBrute(
  cells: ReadonlyArray<{ readonly position: readonly [number, number, number] }>,
  count: number,
  qx: number,
  qy: number,
  qz: number,
): number {
  if (count <= 0) return 0;
  let bestIdx = 0;
  let bestD2 = Infinity;
  for (let i = 0; i < count; i++) {
    const p = cells[i]!.position;
    const dx = p[0] - qx;
    const dy = p[1] - qy;
    const dz = p[2] - qz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < bestD2) {
      bestD2 = d2;
      bestIdx = i;
    }
  }
  return bestIdx;
}

type KdStackFrame = { readonly n: number; readonly k: number; readonly far: number; readonly d2: number };

function kdAxisComp(vx: number, vy: number, vz: number, axis: number): number {
  if (axis === 0) return vx;
  if (axis === 1) return vy;
  return vz;
}

/**
 * Nearest cell via kd-tree bytes built by {@link buildPpgKdTreeGpuBytes}.
 * Mirrors WGSL `ppgKdFindCell` / `ppgUpdateKdFindCell` (iterative NN + far prune).
 */
export function ppgNearestCellIndexKd(
  gpuBytes: Uint8Array,
  cells: ReadonlyArray<{ readonly position: readonly [number, number, number] }>,
  cellCount: number,
  qx: number,
  qy: number,
  qz: number,
): number {
  const stride = PPG_KD_NODE_BYTE_STRIDE;
  if (cellCount <= 0) return 0;
  if (gpuBytes.byteLength < stride) return 0;

  const nNodes = (gpuBytes.byteLength / stride) | 0;

  function readNode(i: number): KdGpuNode {
    const o = i * stride;
    const dv = new DataView(gpuBytes.buffer, gpuBytes.byteOffset + o, stride);
    return {
      child0: dv.getUint32(0, true),
      child1: dv.getUint32(4, true),
      meta: dv.getUint32(8, true),
      split: dv.getFloat32(12, true),
    };
  }

  const root = readNode(0);
  if (root.child0 === 0xffff_ffff && root.child1 === 0xffff_ffff) {
    return ppgNearestCellIndexBrute(cells, cellCount, qx, qy, qz);
  }

  let bestIdx = 0;
  let bestDist2 = Infinity;

  const stack: KdStackFrame[] = [];
  stack.push({ n: 0, k: 0, far: 0, d2: 0 });

  while (stack.length > 0) {
    const top = stack.pop()!;
    if (top.k === 1) {
      if (top.d2 < bestDist2 && stack.length < 48) {
        stack.push({ n: top.far, k: 0, far: 0, d2: 0 });
      }
      continue;
    }

    const nid = top.n;
    if (nid >= nNodes) continue;

    const node = readNode(nid);
    const meta = node.meta;
    if (meta & PPG_KD_LEAF_FLAG) {
      const cellIdx = meta & 0x7fff_ffff;
      if (cellIdx < cellCount) {
        const p = cells[cellIdx]!.position;
        const dx = p[0] - qx;
        const dy = p[1] - qy;
        const dz = p[2] - qz;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < bestDist2) {
          bestDist2 = d2;
          bestIdx = cellIdx;
        }
      }
      continue;
    }

    const axis = meta & 3;
    const split = node.split;
    const c0 = node.child0;
    const c1 = node.child1;
    const d0 = kdAxisComp(qx, qy, qz, axis) - split;
    const d2plane = d0 * d0;
    const nearI = d0 < 0 ? c0 : c1;
    const farI = d0 < 0 ? c1 : c0;

    if (stack.length + 2 > 48) {
      return ppgNearestCellIndexBrute(cells, cellCount, qx, qy, qz);
    }

    // Match WGSL: defer far subtree first, then push near (pop processes near first).
    stack.push({ n: 0, k: 1, far: farI, d2: d2plane });
    stack.push({ n: nearI, k: 0, far: 0, d2: 0 });
  }

  return bestIdx;
}
