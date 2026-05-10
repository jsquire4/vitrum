const LEAFNODE_FLAG = 0xffff0000;
const MAX_LEAF_TRIANGLES = 4;

interface TriangleRecord {
  readonly triIndex: number;
  readonly centroid: readonly [number, number, number];
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
}

interface NodeBuild {
  min: [number, number, number];
  max: [number, number, number];
  rightChildOrTriOffset: number;
  splitAxisOrTriCount: number;
}

function getPosition(positions: Float32Array, vertexIndex: number): readonly [number, number, number] {
  const base = vertexIndex * 4;
  return [positions[base] ?? 0, positions[base + 1] ?? 0, positions[base + 2] ?? 0];
}

function min3(a: readonly [number, number, number], b: readonly [number, number, number], c: readonly [number, number, number]): [number, number, number] {
  return [
    Math.min(a[0], b[0], c[0]),
    Math.min(a[1], b[1], c[1]),
    Math.min(a[2], b[2], c[2]),
  ];
}

function max3(a: readonly [number, number, number], b: readonly [number, number, number], c: readonly [number, number, number]): [number, number, number] {
  return [
    Math.max(a[0], b[0], c[0]),
    Math.max(a[1], b[1], c[1]),
    Math.max(a[2], b[2], c[2]),
  ];
}

function centroid(min: readonly [number, number, number], max: readonly [number, number, number]): [number, number, number] {
  return [
    0.5 * (min[0] + max[0]),
    0.5 * (min[1] + max[1]),
    0.5 * (min[2] + max[2]),
  ];
}

function dominantAxis(min: readonly [number, number, number], max: readonly [number, number, number]): 0 | 1 | 2 {
  const ex = max[0] - min[0];
  const ey = max[1] - min[1];
  const ez = max[2] - min[2];
  if (ex >= ey && ex >= ez) return 0;
  if (ey >= ez) return 1;
  return 2;
}

export interface CpuBvhBuildResult {
  readonly bvhNodes: Float32Array;
  readonly reorderedIndices: Uint32Array;
  readonly reorderedTriMaterialIds: Uint32Array;
}

export function buildCpuBvh(
  positions: Float32Array,
  indices: Uint32Array,
  triMaterialIds: Uint32Array,
): CpuBvhBuildResult {
  const triCount = Math.floor(indices.length / 4);
  if (triCount === 0) {
    const emptyNode = new Float32Array(8);
    return {
      bvhNodes: emptyNode,
      reorderedIndices: indices,
      reorderedTriMaterialIds: triMaterialIds,
    };
  }

  const records: TriangleRecord[] = [];
  for (let t = 0; t < triCount; t += 1) {
    const i0 = indices[t * 4] ?? 0;
    const i1 = indices[t * 4 + 1] ?? 0;
    const i2 = indices[t * 4 + 2] ?? 0;
    const a = getPosition(positions, i0);
    const b = getPosition(positions, i1);
    const c = getPosition(positions, i2);
    const triMin = min3(a, b, c);
    const triMax = max3(a, b, c);
    records.push({
      triIndex: t,
      min: triMin,
      max: triMax,
      centroid: centroid(triMin, triMax),
    });
  }

  const nodes: NodeBuild[] = [];
  const orderedTriangles: number[] = [];

  const build = (subset: TriangleRecord[]): number => {
    const nodeIndex = nodes.length;
    const node: NodeBuild = {
      min: [Infinity, Infinity, Infinity],
      max: [-Infinity, -Infinity, -Infinity],
      rightChildOrTriOffset: 0,
      splitAxisOrTriCount: 0,
    };
    for (const r of subset) {
      node.min[0] = Math.min(node.min[0], r.min[0]);
      node.min[1] = Math.min(node.min[1], r.min[1]);
      node.min[2] = Math.min(node.min[2], r.min[2]);
      node.max[0] = Math.max(node.max[0], r.max[0]);
      node.max[1] = Math.max(node.max[1], r.max[1]);
      node.max[2] = Math.max(node.max[2], r.max[2]);
    }
    nodes.push(node);

    if (subset.length <= MAX_LEAF_TRIANGLES) {
      const leafOffset = orderedTriangles.length;
      for (const r of subset) orderedTriangles.push(r.triIndex);
      node.rightChildOrTriOffset = leafOffset;
      node.splitAxisOrTriCount = LEAFNODE_FLAG | subset.length;
      return nodeIndex;
    }

    const cMin: [number, number, number] = [Infinity, Infinity, Infinity];
    const cMax: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (const r of subset) {
      cMin[0] = Math.min(cMin[0], r.centroid[0]);
      cMin[1] = Math.min(cMin[1], r.centroid[1]);
      cMin[2] = Math.min(cMin[2], r.centroid[2]);
      cMax[0] = Math.max(cMax[0], r.centroid[0]);
      cMax[1] = Math.max(cMax[1], r.centroid[1]);
      cMax[2] = Math.max(cMax[2], r.centroid[2]);
    }
    const axis = dominantAxis(cMin, cMax);
    const sorted = subset.slice().sort((a, b) => a.centroid[axis] - b.centroid[axis]);
    const mid = Math.floor(sorted.length / 2);
    const left = sorted.slice(0, mid);
    const right = sorted.slice(mid);

    build(left);
    const rightChild = build(right);
    node.rightChildOrTriOffset = rightChild;
    node.splitAxisOrTriCount = axis;
    return nodeIndex;
  };

  build(records);

  const reorderedIndices = new Uint32Array(indices.length);
  const reorderedTriMaterialIds = new Uint32Array(triMaterialIds.length);
  for (let newTri = 0; newTri < orderedTriangles.length; newTri += 1) {
    const oldTri = orderedTriangles[newTri] ?? 0;
    reorderedIndices[newTri * 4] = indices[oldTri * 4] ?? 0;
    reorderedIndices[newTri * 4 + 1] = indices[oldTri * 4 + 1] ?? 0;
    reorderedIndices[newTri * 4 + 2] = indices[oldTri * 4 + 2] ?? 0;
    reorderedIndices[newTri * 4 + 3] = 0;
    reorderedTriMaterialIds[newTri] = triMaterialIds[oldTri] ?? 0;
  }

  const nodeBuffer = new ArrayBuffer(nodes.length * 32);
  const dv = new DataView(nodeBuffer);
  for (let i = 0; i < nodes.length; i += 1) {
    const node = nodes[i];
    if (node == null) continue;
    const off = i * 32;
    dv.setFloat32(off + 0, node.min[0], true);
    dv.setFloat32(off + 4, node.min[1], true);
    dv.setFloat32(off + 8, node.min[2], true);
    dv.setFloat32(off + 12, node.max[0], true);
    dv.setFloat32(off + 16, node.max[1], true);
    dv.setFloat32(off + 20, node.max[2], true);
    dv.setUint32(off + 24, node.rightChildOrTriOffset >>> 0, true);
    dv.setUint32(off + 28, node.splitAxisOrTriCount >>> 0, true);
  }

  return {
    bvhNodes: new Float32Array(nodeBuffer),
    reorderedIndices,
    reorderedTriMaterialIds,
  };
}
