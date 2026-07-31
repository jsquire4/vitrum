#!/usr/bin/env -S deno run --unstable-webgpu --sloppy-imports --allow-read --allow-env --allow-write=tools/behavioral-gate
// @ts-nocheck
/**
 * CWBVH GPU parity oracle.
 *
 * Builds a mixed scene (opaque grid + one glass triangle), packs it through the
 * shared CPU CWBVH builder, dispatches `CWBVH_INTERSECT_WGSL` on a real WebGPU
 * adapter, and compares closest-hit + any-hit results against the CPU oracle.
 */

import {
  CWBVH_CHILD_EMPTY,
  CWBVH_CHILD_BOUNDS_U16,
  CWBVH_CHILDREN,
  CWBVH_CHILD_LEAF,
  CWBVH_CHILD_META_WORDS,
  CWBVH_CHILD_NODE,
  CWBVH_INTERSECT_WGSL,
  buildCompressedWideBvh,
  intersectCompressedWideBvhAnyHit,
  intersectCompressedWideBvhFirstHit,
  packCwbvhBuildBoundsForWgsl,
  reorderCwbvhTrianglePayloads,
} from "../../packages/shared-bvh/src/index.ts";
import { buildPackedScene } from "../../packages/pt-webgpu/src/scene/uploadSceneBuffers.ts";

const WRITE_STATUS = Deno.args.includes("--write-status");
const STATUS_PATH = new URL("./cwbvh-parity-status.json", import.meta.url);
const GLASS_PAYLOAD = 5 << 4;
const DIST_TOL = 1e-4;
const NORMAL_TOL = 2e-4;
const SEEDED_RAY_COUNT = 512;

function makeRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x1_0000_0000;
  };
}

function normalize3(x, y, z) {
  const invLength = 1 / Math.max(Math.hypot(x, y, z), 1e-30);
  return [x * invLength, y * invLength, z * invLength];
}

function makeScene() {
  const size = 5;
  const positions = [];
  for (let y = 0; y <= size; y += 1) {
    for (let x = 0; x <= size; x += 1) {
      positions.push(x, y, 0, 0);
    }
  }

  const rootZeroIndices = [];
  const rootZeroMaterialIds = [];
  const rootZeroPayloads = [];
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const v0 = y * (size + 1) + x;
      const v1 = v0 + 1;
      const v2 = v0 + size + 2;
      const v3 = v0 + size + 1;
      rootZeroIndices.push(v0, v1, v2, 0, v0, v2, v3, 0);
      rootZeroMaterialIds.push(0, 0);
      rootZeroPayloads.push(0, 0);
    }
  }

  const glassBase = positions.length / 4;
  positions.push(
    0, 0, 1, 0,
    0.8, 0, 1, 0,
    0, 0.8, 1, 0,
  );
  rootZeroIndices.push(glassBase, glassBase + 1, glassBase + 2, GLASS_PAYLOAD);
  rootZeroMaterialIds.push(1);
  rootZeroPayloads.push(GLASS_PAYLOAD);

  const secondRootBase = positions.length / 4;
  positions.push(
    10, 0, 0, 0,
    11, 0, 0, 0,
    10, 1, 0, 0,
  );
  const rootOneIndices = [secondRootBase, secondRootBase + 1, secondRootBase + 2, 0];
  const extremeBase = positions.length / 4;
  positions.push(
    100_000_000, 100_000_000, 100_000_000, 0,
    100_000_064, 100_000_000, 100_000_000, 0,
    100_000_000, 100_000_064, 100_000_000, 0,
  );
  rootOneIndices.push(extremeBase, extremeBase + 1, extremeBase + 2, 0);
  const rootOneMaterialIds = [0, 0];
  const rootOnePayloads = [0, 0];

  return {
    positions: new Float32Array(positions),
    rootZeroIndices: new Uint32Array(rootZeroIndices),
    rootZeroMaterialIds: new Uint32Array(rootZeroMaterialIds),
    rootZeroPayloads: new Uint32Array(rootZeroPayloads),
    rootOneIndices: new Uint32Array(rootOneIndices),
    rootOneMaterialIds: new Uint32Array(rootOneMaterialIds),
    rootOnePayloads: new Uint32Array(rootOnePayloads),
  };
}

function makeRays(nonzeroRoot, statusRoots) {
  const rays = [
    { label: "glass-short", origin: [0.2, 0.3, 2], direction: [0, 0, -1], tMax: 1.25 },
    { label: "glass-long", origin: [0.2, 0.3, 2], direction: [0, 0, -1], tMax: 3.0 },
    { label: "opaque-grid", origin: [2.2, 3.3, 2], direction: [0, 0, -1], tMax: 3.0 },
    { label: "miss-outside", origin: [6.0, 6.0, 2], direction: [0, 0, -1], tMax: 3.0 },
    { label: "short-before-surface", origin: [1.2, 1.2, 2], direction: [0, 0, -1], tMax: 0.5 },
    { label: "nonzero-root-triangle", origin: [10.25, 0.25, 2], direction: [0, 0, -1], tMax: 3.0, root: nonzeroRoot },
    { label: "zero-axis-boundary", origin: [0, 0, 2], direction: [0, 0, -1], tMax: 3.0 },
    { label: "zero-axis-parallel-outside", origin: [6, 2, 0], direction: [0, 1, 0], tMax: 20.0 },
    {
      label: "large-coordinate-root",
      origin: [100_000_016, 100_000_016, 100_001_024],
      direction: [0, 0, -1],
      tMax: 2_048,
      root: nonzeroRoot,
    },
    {
      label: "status-empty-live-child",
      origin: [0, 0, 2],
      direction: [0, 0, -1],
      tMax: 10,
      root: statusRoots.emptyLiveChild,
      expectedStatus: 2,
    },
    {
      label: "status-zero-count-leaf",
      origin: [0, 0, 2],
      direction: [0, 0, -1],
      tMax: 10,
      root: statusRoots.zeroCountLeaf,
      expectedStatus: 2,
    },
    {
      label: "status-invalid-parent-bounds",
      origin: [0, 0, 2],
      direction: [0, 0, -1],
      tMax: 10,
      root: statusRoots.invalidBounds,
      expectedStatus: 2,
    },
    {
      label: "status-stack-overflow",
      origin: [0, 0, 2],
      direction: [0, 0, -1],
      tMax: 10,
      root: statusRoots.stackOverflow,
      expectedStatus: 1,
    },
  ];

  const random = makeRandom(0xc0b7_5eed);
  for (let i = 0; i < SEEDED_RAY_COUNT; i += 1) {
    const useSecondRoot = i % 5 === 0;
    const targeted = (i & 1) === 0;
    if (useSecondRoot) {
      if (targeted) {
        const large = i % 20 === 0;
        const base = large ? 100_000_000 : 10;
        const span = large ? 48 : 0.75;
        const z = large ? 100_000_000 : 0;
        const height = large ? 256 + random() * 2048 : 1 + random() * 30;
        rays.push({
          label: `seeded-root1-hit-${i}`,
          origin: [base + random() * span * 0.45, (large ? base : 0) + random() * span * 0.45, z + height],
          direction: [0, 0, -1],
          tMax: height + 16,
          root: nonzeroRoot,
        });
      } else {
        rays.push({
          label: `seeded-root1-miss-${i}`,
          origin: [20 + random() * 40, -30 + random() * 60, 1 + random() * 40],
          direction: normalize3(random() * 2 - 1, random() * 2 - 1, -0.25 - random()),
          tMax: 80,
          root: nonzeroRoot,
        });
      }
      continue;
    }
    if (targeted) {
      const x = random() * 4.9;
      const y = random() * 4.9;
      const z = 1.5 + random() * 40;
      // Every fourth targeted ray has exact zero X/Y direction to exercise
      // slab-boundary behavior; the others use a shallow deterministic tilt.
      const direction = (i & 7) === 0
        ? [0, 0, -1]
        : normalize3((random() - 0.5) * 0.08, (random() - 0.5) * 0.08, -1);
      rays.push({
        label: `seeded-root0-hit-${i}`,
        origin: [x, y, z],
        direction,
        tMax: z + 4,
      });
    } else {
      rays.push({
        label: `seeded-root0-miss-${i}`,
        origin: [8 + random() * 80, 8 + random() * 80, 1 + random() * 80],
        direction: normalize3(random() * 2 - 1, random() * 2 - 1, -0.1 - random()),
        tMax: 160,
      });
    }
  }
  return rays;
}

function rayBuffer(rays) {
  const packed = new Float32Array(rays.length * 8);
  for (let i = 0; i < rays.length; i += 1) {
    const ray = rays[i];
    packed[i * 8 + 0] = ray.origin[0];
    packed[i * 8 + 1] = ray.origin[1];
    packed[i * 8 + 2] = ray.origin[2];
    packed[i * 8 + 3] = ray.tMax;
    packed[i * 8 + 4] = ray.direction[0];
    packed[i * 8 + 5] = ray.direction[1];
    packed[i * 8 + 6] = ray.direction[2];
    packed[i * 8 + 7] = ray.root ?? 0;
  }
  return packed;
}

function concatFloat32(a, b) {
  const out = new Float32Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function concatUint16(a, b) {
  const out = new Uint16Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function concatUint32(a, b) {
  const out = new Uint32Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function offsetCwbvhMeta(meta, nodeOffset, triangleOffset) {
  const out = new Uint32Array(meta);
  for (let i = 0; i < out.length; i += CWBVH_CHILD_META_WORDS) {
    const kind = out[i] ?? CWBVH_CHILD_EMPTY;
    if (kind === CWBVH_CHILD_NODE) {
      out[i + 1] = (out[i + 1] ?? 0) + nodeOffset;
    } else if (kind === CWBVH_CHILD_LEAF) {
      out[i + 1] = (out[i + 1] ?? 0) + triangleOffset;
    }
  }
  return out;
}

function offsetSourceTriangles(source, triangleOffset) {
  const out = new Uint32Array(source.length);
  for (let i = 0; i < source.length; i += 1) out[i] = (source[i] ?? 0) + triangleOffset;
  return out;
}

function concatCwbvhRoots(a, b) {
  const triangleOffset = a.reorderedToSourceTriangle.length;
  return {
    ...a,
    bvhNodes: concatFloat32(a.bvhNodes, b.bvhNodes),
    reorderedIndices: concatUint32(a.reorderedIndices, b.reorderedIndices),
    reorderedTriMaterialIds: concatUint32(a.reorderedTriMaterialIds, b.reorderedTriMaterialIds),
    reorderedToSourceTriangle: concatUint32(
      a.reorderedToSourceTriangle,
      offsetSourceTriangles(b.reorderedToSourceTriangle, triangleOffset),
    ),
    cwbvhNodeBounds: concatFloat32(a.cwbvhNodeBounds, b.cwbvhNodeBounds),
    cwbvhChildBounds: concatUint16(a.cwbvhChildBounds, b.cwbvhChildBounds),
    cwbvhChildMeta: concatUint32(
      a.cwbvhChildMeta,
      offsetCwbvhMeta(b.cwbvhChildMeta, a.cwbvhNodeCount, triangleOffset),
    ),
    cwbvhChildCount: concatUint32(a.cwbvhChildCount, b.cwbvhChildCount),
    cwbvhNodeCount: a.cwbvhNodeCount + b.cwbvhNodeCount,
  };
}

function syntheticStatusTree(kind) {
  // A valid linear comb forces the fixed-depth stack past capacity in O(depth
  // x fanout): seven unique sibling subtrees stay pending while slot 7
  // continues deeper. There are no cycles or repeated node references.
  const overflowInteriorCount = 10;
  const overflowSiblingCount = (overflowInteriorCount - 1) * 7 + 8;
  const nodeCount = kind === "stack-overflow"
    ? overflowInteriorCount + overflowSiblingCount
    : 1;
  const nodeBounds = new Float32Array(nodeCount * 6);
  const childBounds = new Uint16Array(nodeCount * 8 * 6);
  const childMeta = new Uint32Array(nodeCount * 8 * CWBVH_CHILD_META_WORDS);
  const childCount = new Uint32Array(nodeCount);
  for (let node = 0; node < nodeCount; node += 1) {
    nodeBounds.set([-1, -1, -1, 1, 1, 1], node * 6);
  }

  const setFullBounds = (node, slot) => {
    const base = (node * 8 + slot) * 6;
    childBounds[base + 3] = 0xffff;
    childBounds[base + 4] = 0xffff;
    childBounds[base + 5] = 0xffff;
  };

  if (kind === "empty-live-child") {
    childCount[0] = 1;
    setFullBounds(0, 0);
  } else if (kind === "zero-count-leaf") {
    childCount[0] = 1;
    setFullBounds(0, 0);
    childMeta[0] = CWBVH_CHILD_LEAF;
    childMeta[1] = 0;
    childMeta[2] = 0;
  } else if (kind === "invalid-bounds") {
    nodeBounds[0] = 2;
    nodeBounds[3] = 1;
  } else {
    let siblingNode = overflowInteriorCount;
    for (let level = 0; level < overflowInteriorCount; level += 1) {
      childCount[level] = 8;
      for (let slot = 0; slot < 8; slot += 1) {
        setFullBounds(level, slot);
        const meta = (level * 8 + slot) * CWBVH_CHILD_META_WORDS;
        childMeta[meta] = CWBVH_CHILD_NODE;
        if (slot === 7 && level + 1 < overflowInteriorCount) {
          childMeta[meta + 1] = level + 1;
        } else {
          childMeta[meta + 1] = siblingNode;
          childCount[siblingNode] = 1;
          setFullBounds(siblingNode, 0);
          const leafMeta = siblingNode * 8 * CWBVH_CHILD_META_WORDS;
          childMeta[leafMeta] = CWBVH_CHILD_LEAF;
          childMeta[leafMeta + 1] = 0;
          childMeta[leafMeta + 2] = 1;
          siblingNode += 1;
        }
      }
    }
  }

  return {
    bvhNodes: new Float32Array(0),
    reorderedIndices: kind === "stack-overflow" ? new Uint32Array([0, 1, 2, 0]) : new Uint32Array(0),
    reorderedTriMaterialIds: kind === "stack-overflow" ? new Uint32Array([0]) : new Uint32Array(0),
    reorderedToSourceTriangle: kind === "stack-overflow" ? new Uint32Array([0]) : new Uint32Array(0),
    cwbvhNodeBounds: nodeBounds,
    cwbvhChildBounds: childBounds,
    cwbvhChildMeta: childMeta,
    cwbvhChildCount: childCount,
    cwbvhNodeCount: nodeCount,
  };
}

function shaderCode(rayCount, nodeCount) {
  const raw = /* wgsl */ `
${CWBVH_INTERSECT_WGSL}

const CWBVH_GATE_RAY_COUNT: u32 = ${rayCount}u;
const CWBVH_GATE_NODE_COUNT: u32 = ${nodeCount}u;

struct CwbvhGateRay {
  originAndTMax: vec4f,
  direction: vec4f,
};

@group(0) @binding(0) var<storage, read> cwbvhNodeBounds: array<CwbvhNodeBounds>;
@group(0) @binding(1) var<storage, read> cwbvhChildBoundsPacked: array<u32>;
@group(0) @binding(2) var<storage, read> cwbvhChildMeta: array<CwbvhChildMeta>;
@group(0) @binding(3) var<storage, read> cwbvhChildCount: array<u32>;
@group(0) @binding(4) var<storage, read> bvh_index: array<vec4u>;
@group(0) @binding(5) var<storage, read> bvh_position: array<vec4f>;
@group(0) @binding(6) var<storage, read> cwbvhGateRays: array<CwbvhGateRay>;
@group(0) @binding(7) var<storage, read_write> cwbvhGateOut: array<vec4u>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let rayIndex = gid.x;
  if (rayIndex >= CWBVH_GATE_RAY_COUNT) {
    return;
  }

  let packedRay = cwbvhGateRays[rayIndex];
  var ray: CwbvhRay;
  ray.origin = packedRay.originAndTMax.xyz;
  ray.direction = packedRay.direction.xyz;
  let tMax = packedRay.originAndTMax.w;
  let root = u32(packedRay.direction.w);

  let closestNoSkip = cwbvhIntersectFirstHitFromRoot(
    ray,
    1.0e-5,
    CWBVH_GATE_NODE_COUNT,
    root,
    false,
  );
  // Visibility uses the same range-bounded closest-hit traversal as renderer
  // sidedness/alpha loops. Keeping one walker prevents layout guards and stack
  // behavior from drifting between closest-hit and any-hit implementations.
  let anyNoSkip = cwbvhIntersectFirstHitRangeFromRoot(
    ray,
    1.0e-5,
    1.0e-5,
    tMax,
    CWBVH_GATE_NODE_COUNT,
    root,
    false,
  );

  let closestSkip = cwbvhIntersectFirstHitFromRoot(
    ray,
    1.0e-5,
    CWBVH_GATE_NODE_COUNT,
    root,
    true,
  );
  let anySkip = cwbvhIntersectFirstHitRangeFromRoot(
    ray,
    1.0e-5,
    1.0e-5,
    tMax,
    CWBVH_GATE_NODE_COUNT,
    root,
    true,
  );

  cwbvhGateOut[rayIndex * 2u + 0u] = vec4u(
    closestNoSkip.status * 2u + select(0u, 1u, closestNoSkip.didHit),
    closestNoSkip.triIndex,
    bitcast<u32>(closestNoSkip.dist),
    anyNoSkip.status * 2u + select(0u, 1u, anyNoSkip.didHit),
  );
  cwbvhGateOut[rayIndex * 2u + 1u] = vec4u(
    closestSkip.status * 2u + select(0u, 1u, closestSkip.didHit),
    closestSkip.triIndex,
    bitcast<u32>(closestSkip.dist),
    anySkip.status * 2u + select(0u, 1u, anySkip.didHit),
  );
}
`;
  return raw;
}

function makeBuffer(device, label, data, usage) {
  const size = Math.max(4, (data.byteLength + 3) & ~3);
  const buffer = device.createBuffer({ label, size, usage });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function f32BitsToNumber(bits) {
  const buf = new ArrayBuffer(4);
  const view = new DataView(buf);
  view.setUint32(0, bits >>> 0, true);
  return view.getFloat32(0, true);
}

function gpuTriToSigned(tri) {
  return tri === 0xffffffff ? -1 : tri;
}

function compareHit(label, mode, gpu, cpu, mismatches) {
  if (gpu[0] > 1) {
    mismatches.push(`${label}/${mode}: GPU status=${gpu[0] >> 1} (non-COMPLETE)`);
    return;
  }
  const gpuDidHit = gpu[0] === 1;
  const gpuTri = gpuTriToSigned(gpu[1]);
  const gpuDist = f32BitsToNumber(gpu[2]);
  if (gpuDidHit !== cpu.didHit) {
    mismatches.push(`${label}/${mode}: didHit GPU=${gpuDidHit} CPU=${cpu.didHit}`);
    return;
  }
  if (!cpu.didHit) return;
  if (gpuTri !== cpu.triangleIndex) {
    mismatches.push(`${label}/${mode}: tri GPU=${gpuTri} CPU=${cpu.triangleIndex}`);
  }
  if (Math.abs(gpuDist - cpu.dist) > DIST_TOL) {
    mismatches.push(`${label}/${mode}: dist GPU=${gpuDist} CPU=${cpu.dist}`);
  }
}

function compareAny(label, mode, encoded, cpu, mismatches) {
  if (encoded > 1) {
    mismatches.push(`${label}/${mode}: GPU status=${encoded >> 1} (non-COMPLETE)`);
    return;
  }
  if ((encoded === 1) !== cpu) {
    mismatches.push(`${label}/${mode}: GPU=${encoded === 1} CPU=${cpu}`);
  }
}

function affine(c0, c1, c2, translation) {
  return new Float32Array([
    c0[0], c0[1], c0[2], 0,
    c1[0], c1[1], c1[2], 0,
    c2[0], c2[1], c2[2], 0,
    translation[0], translation[1], translation[2], 1,
  ]);
}

function transformPointCpu(matrix, point) {
  return [
    matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12],
    matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13],
    matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14],
  ];
}

function transformDirectionCpu(matrix, direction) {
  return normalize3(
    matrix[0] * direction[0] + matrix[4] * direction[1] + matrix[8] * direction[2],
    matrix[1] * direction[0] + matrix[5] * direction[1] + matrix[9] * direction[2],
    matrix[2] * direction[0] + matrix[6] * direction[1] + matrix[10] * direction[2],
  );
}

function transformNormalCpu(worldToLocal, normal) {
  // transpose(worldToLocal) * normal under the column-major contract used by
  // transformPointCpu. Normalisation intentionally happens only afterwards.
  return normalize3(
    worldToLocal[0] * normal[0] + worldToLocal[1] * normal[1] + worldToLocal[2] * normal[2],
    worldToLocal[4] * normal[0] + worldToLocal[5] * normal[1] + worldToLocal[6] * normal[2],
    worldToLocal[8] * normal[0] + worldToLocal[9] * normal[1] + worldToLocal[10] * normal[2],
  );
}

function subtract3(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function unpackPackedCwbvh(packed) {
  const childBounds = new Uint16Array(
    packed.cwbvhNodeCount * CWBVH_CHILDREN * CWBVH_CHILD_BOUNDS_U16,
  );
  for (let i = 0; i < packed.cwbvhChildBoundsPacked.length; i += 1) {
    const word = packed.cwbvhChildBoundsPacked[i];
    childBounds[i * 2] = word & 0xffff;
    childBounds[i * 2 + 1] = word >>> 16;
  }
  return {
    bvhNodes: packed.bvhNodes,
    reorderedIndices: packed.indices,
    reorderedTriMaterialIds: packed.triMaterialIds,
    reorderedToSourceTriangle: Uint32Array.from({ length: packed.triangleCount }, (_, i) => i),
    cwbvhNodeBounds: packed.cwbvhNodeBounds,
    cwbvhChildBounds: childBounds,
    cwbvhChildMeta: packed.cwbvhChildMeta,
    cwbvhChildCount: packed.cwbvhChildCount,
    cwbvhNodeCount: packed.cwbvhNodeCount,
  };
}

function tracePackedTlasCpu(packed, cwbvh, ray) {
  const tMin = 1e-5;
  let anyHit = false;
  let best = {
    didHit: false,
    dist: ray.tMax,
    triangleIndex: -1,
    instanceIndex: -1,
    frontFace: false,
    normal: [0, 1, 0],
  };
  for (let instanceIndex = 0; instanceIndex < packed.tlasBlasRoots.length; instanceIndex += 1) {
    const pairBase = instanceIndex * 4;
    const binaryRoot = packed.cwbvhTlasBlasRoots[pairBase + 1];
    const wideRoot = packed.cwbvhTlasBlasRoots[pairBase + 2];
    if (binaryRoot !== packed.tlasBlasRoots[instanceIndex]) {
      throw new Error(`TLAS CPU oracle root-pair mismatch for instance ${instanceIndex}`);
    }
    const w2l = packed.tlasInstanceWorldToLocal.subarray(instanceIndex * 16, instanceIndex * 16 + 16);
    const l2w = packed.tlasInstanceLocalToWorld.subarray(instanceIndex * 16, instanceIndex * 16 + 16);
    const localOrigin = transformPointCpu(w2l, ray.origin);
    const localDirection = transformDirectionCpu(w2l, ray.direction);
    const localStartPoint = transformPointCpu(w2l, [
      ray.origin[0] + ray.direction[0] * tMin,
      ray.origin[1] + ray.direction[1] * tMin,
      ray.origin[2] + ray.direction[2] * tMin,
    ]);
    const localEndPoint = transformPointCpu(w2l, [
      ray.origin[0] + ray.direction[0] * best.dist,
      ray.origin[1] + ray.direction[1] * best.dist,
      ray.origin[2] + ray.direction[2] * best.dist,
    ]);
    const localAnyEndPoint = transformPointCpu(w2l, [
      ray.origin[0] + ray.direction[0] * ray.tMax,
      ray.origin[1] + ray.direction[1] * ray.tMax,
      ray.origin[2] + ray.direction[2] * ray.tMax,
    ]);
    const localTMin = Math.max(dot3(subtract3(localStartPoint, localOrigin), localDirection), 0);
    const localTMax = Math.max(dot3(subtract3(localEndPoint, localOrigin), localDirection), localTMin);
    const localAnyTMax = Math.max(
      dot3(subtract3(localAnyEndPoint, localOrigin), localDirection),
      localTMin,
    );
    if (!anyHit) {
      const shiftedLocalOrigin = [
        localOrigin[0] + localDirection[0] * localTMin,
        localOrigin[1] + localDirection[1] * localTMin,
        localOrigin[2] + localDirection[2] * localTMin,
      ];
      anyHit = intersectCompressedWideBvhAnyHit(
        cwbvh,
        packed.positions,
        { origin: shiftedLocalOrigin, direction: localDirection },
        { root: wideRoot, tMax: Math.max(localAnyTMax - localTMin, 0) },
      );
    }
    const localHit = intersectCompressedWideBvhFirstHit(
      cwbvh,
      packed.positions,
      { origin: localOrigin, direction: localDirection },
      { root: wideRoot, tMin: localTMin, tMax: localTMax },
    );
    if (!localHit.didHit) continue;
    const localPosition = [
      localOrigin[0] + localDirection[0] * localHit.dist,
      localOrigin[1] + localDirection[1] * localHit.dist,
      localOrigin[2] + localDirection[2] * localHit.dist,
    ];
    const worldPosition = transformPointCpu(l2w, localPosition);
    const worldDistance = dot3(subtract3(worldPosition, ray.origin), ray.direction);
    if (worldDistance > tMin && worldDistance < best.dist) {
      const triBase = localHit.triangleIndex * 4;
      const ia = packed.indices[triBase] * 4;
      const ib = packed.indices[triBase + 1] * 4;
      const ic = packed.indices[triBase + 2] * 4;
      const pa = [packed.positions[ia], packed.positions[ia + 1], packed.positions[ia + 2]];
      const pb = [packed.positions[ib], packed.positions[ib + 1], packed.positions[ib + 2]];
      const pc = [packed.positions[ic], packed.positions[ic + 1], packed.positions[ic + 2]];
      const rawNormal = cross3(subtract3(pb, pa), subtract3(pc, pa));
      const localNormal = normalize3(rawNormal[0], rawNormal[1], rawNormal[2]);
      const worldNormal = transformNormalCpu(w2l, localNormal);
      best = {
        didHit: true,
        dist: worldDistance,
        triangleIndex: localHit.triangleIndex,
        instanceIndex,
        frontFace: dot3(ray.direction, worldNormal) < 0,
        normal: worldNormal,
      };
    }
  }
  return { ...best, anyHit };
}

function makeTlasSceneAndRays() {
  const material = { baseColor: [0.7, 0.7, 0.7], roughness: 0.5, metallic: 0 };
  const identity = affine(
    [1, 0, 0], [0, 1, 0], [0, 0, 1], [-8, -6, 2],
  );
  const nonuniformRotationShear = affine(
    [1.4, 0.35, 0.2], [-0.45, 0.9, 0.3], [0.25, -0.2, 1.1], [-2, 3, -3],
  );
  const mirroredRotationShear = affine(
    [-1.1, 0.25, 0.15], [0.3, 1.25, -0.2], [0.15, 0.4, 0.85], [6, -2, 1],
  );
  const singular = affine(
    [1, 0.25, 0], [2, 0.5, 0], [0, 0, 1], [30, 25, 8],
  );
  const secondBlasRotationShear = affine(
    [0.9, -0.6, 0.2], [0.5, 1.0, 0.1], [-0.15, 0.3, 1.4], [1, 10, -4],
  );
  const scene = {
    primitives: [
      {
        kind: "instanced-mesh",
        id: "tlas-instances",
        positions: new Float32Array([-1, -1, 0, 1, -1, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material,
        instances: [
          identity,
          nonuniformRotationShear,
          mirroredRotationShear,
          singular,
        ],
      },
      {
        kind: "mesh",
        id: "tlas-second-blas",
        positions: new Float32Array([-1, -1, 0, 1, -1, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material,
        transform: secondBlasRotationShear,
      },
    ],
    emitters: [],
    environment: { kind: "none" },
  };
  const packed = buildPackedScene(scene);
  const rawAuthoredInstanceCount = 5;
  const expectedPackedInstanceCount = 4;
  if (packed.tlasBlasRoots.length !== expectedPackedInstanceCount) {
    throw new Error(
      `TLAS gate expected ${expectedPackedInstanceCount} packed instances after singular skip, ` +
      `got ${packed.tlasBlasRoots.length}`,
    );
  }
  const singularSkipped = packed.warnings.some((warning) =>
    warning.includes("non-invertible instance transform") && warning.includes("skipping"));
  if (!singularSkipped) {
    throw new Error("TLAS gate did not observe the required singular-instance skip warning");
  }
  const random = makeRandom(0x71a5_5eed);
  const sourceRays = [];
  for (let i = 0; i < 192; i += 1) {
    const instance = i % packed.tlasBlasRoots.length;
    const l2w = packed.tlasInstanceLocalToWorld.subarray(instance * 16, instance * 16 + 16);
    const w2l = packed.tlasInstanceWorldToLocal.subarray(instance * 16, instance * 16 + 16);
    const wa = 0.15 + random() * 0.25;
    const wb = 0.15 + random() * 0.25;
    const wc = 1 - wa - wb;
    const localPoint = [-wa + wb, -wa - wb + wc, 0];
    const worldPoint = transformPointCpu(l2w, localPoint);
    const worldNormal = transformNormalCpu(w2l, [0, 0, 1]);
    const front = ((Math.floor(i / packed.tlasBlasRoots.length)) & 1) === 0;
    const rayDirection = front
      ? [-worldNormal[0], -worldNormal[1], -worldNormal[2]]
      : [worldNormal[0], worldNormal[1], worldNormal[2]];
    const height = 0.75 + random() * 12;
    sourceRays.push({
      label: `tlas-${front ? "front" : "back"}-${i}-instance-${instance}`,
      origin: [
        worldPoint[0] - rayDirection[0] * height,
        worldPoint[1] - rayDirection[1] * height,
        worldPoint[2] - rayDirection[2] * height,
      ],
      direction: rayDirection,
      tMax: height + 2,
    });
  }
  for (let i = 0; i < 32; i += 1) {
    const singularPoint = transformPointCpu(singular, [random() * 0.4 - 0.2, random() * 0.4 - 0.2, 0]);
    const height = 1 + random() * 8;
    sourceRays.push({
      label: `tlas-singular-skipped-${i}`,
      origin: [singularPoint[0], singularPoint[1], singularPoint[2] + height],
      direction: [0, 0, -1],
      tMax: height + 2,
    });
  }
  for (let i = 0; i < 32; i += 1) {
    sourceRays.push({
      label: `tlas-far-miss-${i}`,
      origin: [45 + random() * 30, -45 - random() * 30, 20 + random() * 30],
      direction: normalize3(random() * 0.2 - 0.1, random() * 0.2 - 0.1, -1),
      tMax: 100,
    });
  }
  if (sourceRays.length !== 256) {
    throw new Error(`TLAS gate requires exactly 256 rays, got ${sourceRays.length}`);
  }
  const data = rayBuffer(sourceRays);
  const rays = sourceRays.map((ray, i) => ({
    label: ray.label,
    origin: [data[i * 8], data[i * 8 + 1], data[i * 8 + 2]],
    tMax: data[i * 8 + 3],
    direction: [data[i * 8 + 4], data[i * 8 + 5], data[i * 8 + 6]],
  }));
  return {
    packed,
    rays,
    data,
    metadata: {
      rawAuthoredInstanceCount,
      packedInstanceCount: packed.tlasBlasRoots.length,
      hitRayCount: 192,
      singularMissRayCount: 32,
      farMissRayCount: 32,
      singularSkipped,
    },
  };
}

const TLAS_GATE_DECLARATIONS_WGSL = /* wgsl */ `
const GATE_TLAS_STACK_DEPTH: u32 = 64u;
const GATE_LEAF_FLAG: u32 = 0xffff0000u;
const GATE_ROOT_MAGIC: u32 = 0x43574256u;
const GATE_BINARY_FACTOR: u32 = 0x9e3779b1u;
const GATE_WIDE_FACTOR: u32 = 0x85ebca6bu;

struct GateTlasNode {
  boundsMin: array<f32, 3>,
  boundsMax: array<f32, 3>,
  rightChildOrOffset: u32,
  splitOrCount: u32,
};

struct GateRay {
  originAndTMax: vec4f,
  direction: vec4f,
};

struct GateTlasHit {
  status: u32,
  didHit: bool,
  dist: f32,
  triIndex: u32,
  instanceIndex: u32,
  normal: vec3f,
  frontFace: bool,
};

struct GateTlasAnyResult {
  status: u32,
  didHit: bool,
};

@group(0) @binding(0) var<storage, read> cwbvhNodeBounds: array<CwbvhNodeBounds>;
@group(0) @binding(1) var<storage, read> cwbvhChildBoundsPacked: array<u32>;
@group(0) @binding(2) var<storage, read> cwbvhChildMeta: array<CwbvhChildMeta>;
@group(0) @binding(3) var<storage, read> cwbvhChildCount: array<u32>;
@group(0) @binding(4) var<storage, read> bvh_index: array<vec4u>;
@group(0) @binding(5) var<storage, read> bvh_position: array<vec4f>;
@group(0) @binding(6) var<storage, read> gateTlasNodes: array<GateTlasNode>;
@group(0) @binding(7) var<storage, read> gateInstanceIndices: array<u32>;
@group(0) @binding(8) var<storage, read> gateBinaryRoots: array<u32>;
@group(0) @binding(9) var<storage, read> gateRootPairs: array<vec4u>;
@group(0) @binding(10) var<storage, read> gateWorldToLocal: array<vec4f>;
@group(0) @binding(11) var<storage, read> gateLocalToWorld: array<vec4f>;
@group(0) @binding(12) var<storage, read> gateRays: array<GateRay>;
@group(0) @binding(13) var<storage, read_write> gateOutput: array<vec4u>;

fn gateTransformPoint(c0: vec4f, c1: vec4f, c2: vec4f, c3: vec4f, p: vec3f) -> vec3f {
  let h = c0 * p.x + c1 * p.y + c2 * p.z + c3;
  return h.xyz / max(abs(h.w), 1e-8);
}

fn gateTransformDirection(c0: vec4f, c1: vec4f, c2: vec4f, d: vec3f) -> vec3f {
  return normalize((c0 * d.x + c1 * d.y + c2 * d.z).xyz);
}

fn gateTransformNormal(w0: vec4f, w1: vec4f, w2: vec4f, nLocal: vec3f) -> vec3f {
  // transpose(worldToLocal) under the same column-major contract as points.
  return normalize(vec3f(
    dot(w0.xyz, nLocal),
    dot(w1.xyz, nLocal),
    dot(w2.xyz, nLocal),
  ));
}

fn gateAabb(origin: vec3f, direction: vec3f, bmin: vec3f, bmax: vec3f, tMin: f32, tMax: f32) -> bool {
  let invDir = safeInvDir(direction);
  let t0 = (bmin - origin) * invDir;
  let t1 = (bmax - origin) * invDir;
  let near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
  let far = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  return !(near > far || far < tMin || near > tMax);
}

fn gateRootPairValid(instanceIndex: u32) -> bool {
  if (instanceIndex >= arrayLength(&gateRootPairs) || instanceIndex >= arrayLength(&gateBinaryRoots)) {
    return false;
  }
  let pair = gateRootPairs[instanceIndex];
  return pair.x == GATE_ROOT_MAGIC &&
    pair.y == gateBinaryRoots[instanceIndex] &&
    pair.z < GATE_CWBVH_NODE_COUNT &&
    pair.w == (pair.x ^ pair.y * GATE_BINARY_FACTOR ^ pair.z * GATE_WIDE_FACTOR);
}
`;

const TLAS_GATE_CLOSEST_WGSL = /* wgsl */ `
fn gateTlasClosest(origin: vec3f, direction: vec3f, tMin: f32, tMax: f32) -> GateTlasHit {
  var best: GateTlasHit;
  best.status = CWBVH_STATUS_COMPLETE;
  best.didHit = false;
  best.dist = tMax;
  best.triIndex = 0xffffffffu;
  best.instanceIndex = 0xffffffffu;
  best.normal = vec3f(0.0, 1.0, 0.0);
  best.frontFace = false;
  if (GATE_TLAS_NODE_COUNT == 0u || arrayLength(&gateTlasNodes) < GATE_TLAS_NODE_COUNT) {
    best.status = CWBVH_STATUS_INVALID_LAYOUT;
    return best;
  }
  var stack: array<u32, 64>;
  var stackPtr = 1u;
  stack[0] = 0u;
  while (stackPtr > 0u) {
    stackPtr = stackPtr - 1u;
    let nodeIndex = stack[stackPtr];
    if (nodeIndex >= GATE_TLAS_NODE_COUNT) {
      best.status = CWBVH_STATUS_INVALID_LAYOUT;
      return best;
    }
    let node = gateTlasNodes[nodeIndex];
    let bmin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let bmax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    if (!cwbvhBoundsAreValid(bmin, bmax)) {
      best.status = CWBVH_STATUS_INVALID_LAYOUT;
      return best;
    }
    if (!gateAabb(origin, direction, bmin, bmax, tMin, best.dist)) { continue; }
    if ((node.splitOrCount & GATE_LEAF_FLAG) == GATE_LEAF_FLAG) {
      let count = node.splitOrCount & 0xffffu;
      if (count == 0u || node.rightChildOrOffset > arrayLength(&gateInstanceIndices) || count > arrayLength(&gateInstanceIndices) - node.rightChildOrOffset) {
        best.status = CWBVH_STATUS_INVALID_LAYOUT;
        return best;
      }
      for (var i = 0u; i < count; i = i + 1u) {
        let instanceIndex = gateInstanceIndices[node.rightChildOrOffset + i];
        let matrixBase = instanceIndex * 4u;
        if (matrixBase + 3u >= arrayLength(&gateWorldToLocal) || matrixBase + 3u >= arrayLength(&gateLocalToWorld) || !gateRootPairValid(instanceIndex)) {
          best.status = CWBVH_STATUS_INVALID_LAYOUT;
          return best;
        }
        let w0 = gateWorldToLocal[matrixBase];
        let w1 = gateWorldToLocal[matrixBase + 1u];
        let w2 = gateWorldToLocal[matrixBase + 2u];
        let w3 = gateWorldToLocal[matrixBase + 3u];
        let l0 = gateLocalToWorld[matrixBase];
        let l1 = gateLocalToWorld[matrixBase + 1u];
        let l2 = gateLocalToWorld[matrixBase + 2u];
        let l3 = gateLocalToWorld[matrixBase + 3u];
        var localRay: CwbvhRay;
        localRay.origin = gateTransformPoint(w0, w1, w2, w3, origin);
        localRay.direction = gateTransformDirection(w0, w1, w2, direction);
        let localStart = gateTransformPoint(w0, w1, w2, w3, origin + direction * tMin);
        let localEnd = gateTransformPoint(w0, w1, w2, w3, origin + direction * best.dist);
        let localTMin = max(dot(localStart - localRay.origin, localRay.direction), 0.0);
        let localTMax = max(dot(localEnd - localRay.origin, localRay.direction), localTMin);
        let localHit = cwbvhIntersectFirstHitRangeFromRoot(
          localRay, 1e-5, localTMin, localTMax,
          GATE_CWBVH_NODE_COUNT, gateRootPairs[instanceIndex].z, false,
        );
        if (localHit.status != CWBVH_STATUS_COMPLETE) {
          best.status = localHit.status;
          return best;
        }
        if (!localHit.didHit) { continue; }
        let localPosition = localRay.origin + localRay.direction * localHit.dist;
        let worldPosition = gateTransformPoint(l0, l1, l2, l3, localPosition);
        let worldDistance = dot(worldPosition - origin, direction);
        if (worldDistance > tMin && worldDistance < best.dist) {
          let tri = bvh_index[localHit.triIndex].xyz;
          let pa = bvh_position[tri.x].xyz;
          let pb = bvh_position[tri.y].xyz;
          let pc = bvh_position[tri.z].xyz;
          let localAuthoredNormal = cross(pb - pa, pc - pa);
          let worldAuthoredNormal = gateTransformNormal(w0, w1, w2, localAuthoredNormal);
          best.didHit = true;
          best.dist = worldDistance;
          best.triIndex = localHit.triIndex;
          best.instanceIndex = instanceIndex;
          best.normal = worldAuthoredNormal;
          best.frontFace = dot(direction, worldAuthoredNormal) < 0.0;
        }
      }
    } else {
      let left = nodeIndex + 1u;
      let right = nodeIndex + node.rightChildOrOffset;
      if (node.rightChildOrOffset <= 1u || left >= GATE_TLAS_NODE_COUNT || right >= GATE_TLAS_NODE_COUNT) {
        best.status = CWBVH_STATUS_INVALID_LAYOUT;
        return best;
      }
      if (stackPtr + 2u > GATE_TLAS_STACK_DEPTH) {
        best.status = CWBVH_STATUS_STACK_OVERFLOW;
        return best;
      }
      stack[stackPtr] = right; stackPtr = stackPtr + 1u;
      stack[stackPtr] = left; stackPtr = stackPtr + 1u;
    }
  }
  return best;
}
`;

const TLAS_GATE_ANY_WGSL = /* wgsl */ `
fn gateTlasAny(origin: vec3f, direction: vec3f, tMin: f32, tMax: f32) -> GateTlasAnyResult {
  var result: GateTlasAnyResult;
  result.status = CWBVH_STATUS_COMPLETE;
  result.didHit = false;
  if (GATE_TLAS_NODE_COUNT == 0u || arrayLength(&gateTlasNodes) < GATE_TLAS_NODE_COUNT) {
    result.status = CWBVH_STATUS_INVALID_LAYOUT;
    return result;
  }

  var stack: array<u32, 64>;
  var stackPtr = 1u;
  stack[0] = 0u;
  while (stackPtr > 0u) {
    stackPtr = stackPtr - 1u;
    let nodeIndex = stack[stackPtr];
    if (nodeIndex >= GATE_TLAS_NODE_COUNT) {
      result.status = CWBVH_STATUS_INVALID_LAYOUT;
      return result;
    }
    let node = gateTlasNodes[nodeIndex];
    let bmin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let bmax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    if (!cwbvhBoundsAreValid(bmin, bmax)) {
      result.status = CWBVH_STATUS_INVALID_LAYOUT;
      return result;
    }
    if (!gateAabb(origin, direction, bmin, bmax, tMin, tMax)) { continue; }

    if ((node.splitOrCount & GATE_LEAF_FLAG) == GATE_LEAF_FLAG) {
      let count = node.splitOrCount & 0xffffu;
      if (count == 0u || node.rightChildOrOffset > arrayLength(&gateInstanceIndices) || count > arrayLength(&gateInstanceIndices) - node.rightChildOrOffset) {
        result.status = CWBVH_STATUS_INVALID_LAYOUT;
        return result;
      }
      for (var i = 0u; i < count; i = i + 1u) {
        let instanceIndex = gateInstanceIndices[node.rightChildOrOffset + i];
        let matrixBase = instanceIndex * 4u;
        if (matrixBase + 3u >= arrayLength(&gateWorldToLocal) || !gateRootPairValid(instanceIndex)) {
          result.status = CWBVH_STATUS_INVALID_LAYOUT;
          return result;
        }
        let w0 = gateWorldToLocal[matrixBase];
        let w1 = gateWorldToLocal[matrixBase + 1u];
        let w2 = gateWorldToLocal[matrixBase + 2u];
        let w3 = gateWorldToLocal[matrixBase + 3u];
        var localRay: CwbvhRay;
        localRay.origin = gateTransformPoint(w0, w1, w2, w3, origin);
        localRay.direction = gateTransformDirection(w0, w1, w2, direction);
        let localStart = gateTransformPoint(w0, w1, w2, w3, origin + direction * tMin);
        let localEnd = gateTransformPoint(w0, w1, w2, w3, origin + direction * tMax);
        let localTMin = max(dot(localStart - localRay.origin, localRay.direction), 0.0);
        let localTMax = max(dot(localEnd - localRay.origin, localRay.direction), localTMin);
        let localHit = cwbvhIntersectFirstHitRangeFromRoot(
          localRay, 1e-5, localTMin, localTMax,
          GATE_CWBVH_NODE_COUNT, gateRootPairs[instanceIndex].z, false,
        );
        if (localHit.status != CWBVH_STATUS_COMPLETE) {
          result.status = localHit.status;
          return result;
        }
        if (localHit.didHit) {
          result.didHit = true;
          return result;
        }
      }
    } else {
      let left = nodeIndex + 1u;
      let right = nodeIndex + node.rightChildOrOffset;
      if (node.rightChildOrOffset <= 1u || left >= GATE_TLAS_NODE_COUNT || right >= GATE_TLAS_NODE_COUNT) {
        result.status = CWBVH_STATUS_INVALID_LAYOUT;
        return result;
      }
      if (stackPtr + 2u > GATE_TLAS_STACK_DEPTH) {
        result.status = CWBVH_STATUS_STACK_OVERFLOW;
        return result;
      }
      stack[stackPtr] = right; stackPtr = stackPtr + 1u;
      stack[stackPtr] = left; stackPtr = stackPtr + 1u;
    }
  }
  return result;
}
`;

const TLAS_GATE_MAIN_WGSL = /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= GATE_RAY_COUNT) { return; }
  let packed = gateRays[gid.x];
  let closest = gateTlasClosest(
    packed.originAndTMax.xyz, packed.direction.xyz, 1e-5, packed.originAndTMax.w,
  );
  let anyHitResult = gateTlasAny(
    packed.originAndTMax.xyz, packed.direction.xyz, 1e-5, packed.originAndTMax.w,
  );
  gateOutput[gid.x * 3u] = vec4u(
    closest.status * 2u + select(0u, 1u, closest.didHit),
    closest.triIndex,
    bitcast<u32>(closest.dist),
    closest.instanceIndex,
  );
  gateOutput[gid.x * 3u + 1u] = vec4u(
    bitcast<u32>(closest.normal.x),
    bitcast<u32>(closest.normal.y),
    bitcast<u32>(closest.normal.z),
    select(0u, 1u, closest.frontFace),
  );
  gateOutput[gid.x * 3u + 2u] = vec4u(
    anyHitResult.status * 2u + select(0u, 1u, anyHitResult.didHit),
    closest.status,
    anyHitResult.status,
    0xc0b7a11eu,
  );
}
`;

function tlasShaderCode(rayCount, cwbvhNodeCount, tlasNodeCount) {
  return /* wgsl */ `
${CWBVH_INTERSECT_WGSL}
const GATE_RAY_COUNT: u32 = ${rayCount}u;
const GATE_CWBVH_NODE_COUNT: u32 = ${cwbvhNodeCount}u;
const GATE_TLAS_NODE_COUNT: u32 = ${tlasNodeCount}u;
${TLAS_GATE_DECLARATIONS_WGSL}
${TLAS_GATE_CLOSEST_WGSL}
${TLAS_GATE_ANY_WGSL}
${TLAS_GATE_MAIN_WGSL}
  `;
}

async function dispatchTlasDifferential(device, storageUsage, packed, rayData, rayCount) {
  const tlasNodeCount = Math.floor(packed.tlasNodes.length / 8);
  const module = device.createShaderModule({
    label: "cwbvh-tlas-parity-oracle",
    code: tlasShaderCode(rayCount, packed.cwbvhNodeCount, tlasNodeCount),
  });
  const info = await module.getCompilationInfo();
  const compileErrors = info.messages.filter((message) => message.type === "error");
  if (compileErrors.length > 0) {
    throw new Error(
      `CWBVH TLAS WGSL compile failure: ${compileErrors.map((error) => error.message).join(" | ")}`,
    );
  }
  device.pushErrorScope("validation");
  const pipeline = await device.createComputePipelineAsync({
    label: "cwbvh-tlas-parity-oracle",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const pipelineError = await device.popErrorScope();
  if (pipelineError) {
    throw new Error(`CWBVH TLAS pipeline failure: ${pipelineError.message ?? pipelineError}`);
  }
  const output = new Uint32Array(rayCount * 12);
  const result = device.createBuffer({
    label: "cwbvh-tlas-result",
    size: output.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: "cwbvh-tlas-readback",
    size: output.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const bindGroup = device.createBindGroup({
    label: "cwbvh-tlas-bindings",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: makeBuffer(device, "tlas-cwbvh-node-bounds", packed.cwbvhNodeBounds, storageUsage) } },
      { binding: 1, resource: { buffer: makeBuffer(device, "tlas-cwbvh-child-bounds", packed.cwbvhChildBoundsPacked, storageUsage) } },
      { binding: 2, resource: { buffer: makeBuffer(device, "tlas-cwbvh-child-meta", packed.cwbvhChildMeta, storageUsage) } },
      { binding: 3, resource: { buffer: makeBuffer(device, "tlas-cwbvh-child-count", packed.cwbvhChildCount, storageUsage) } },
      { binding: 4, resource: { buffer: makeBuffer(device, "tlas-indices", packed.indices, storageUsage) } },
      { binding: 5, resource: { buffer: makeBuffer(device, "tlas-positions", packed.positions, storageUsage) } },
      { binding: 6, resource: { buffer: makeBuffer(device, "tlas-nodes", packed.tlasNodes, storageUsage) } },
      { binding: 7, resource: { buffer: makeBuffer(device, "tlas-instance-indices", packed.tlasInstanceIndices, storageUsage) } },
      { binding: 8, resource: { buffer: makeBuffer(device, "tlas-binary-roots", packed.tlasBlasRoots, storageUsage) } },
      { binding: 9, resource: { buffer: makeBuffer(device, "tlas-root-pairs", packed.cwbvhTlasBlasRoots, storageUsage) } },
      { binding: 10, resource: { buffer: makeBuffer(device, "tlas-world-to-local", packed.tlasInstanceWorldToLocal, storageUsage) } },
      { binding: 11, resource: { buffer: makeBuffer(device, "tlas-local-to-world", packed.tlasInstanceLocalToWorld, storageUsage) } },
      { binding: 12, resource: { buffer: makeBuffer(device, "tlas-rays", rayData, storageUsage) } },
      { binding: 13, resource: { buffer: result } },
    ],
  });
  device.pushErrorScope("validation");
  device.pushErrorScope("internal");
  device.pushErrorScope("out-of-memory");
  const encoder = device.createCommandEncoder({ label: "cwbvh-tlas-parity-oracle" });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(rayCount / 64));
  pass.end();
  encoder.copyBufferToBuffer(result, 0, readback, 0, output.byteLength);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const errors = [
    await device.popErrorScope(),
    await device.popErrorScope(),
    await device.popErrorScope(),
  ].filter(Boolean);
  if (errors.length > 0) {
    throw new Error(`CWBVH TLAS GPU failure: ${errors.map((error) => error.message ?? error).join(" | ")}`);
  }
  await readback.mapAsync(GPUMapMode.READ);
  output.set(new Uint32Array(readback.getMappedRange()).slice(0, output.length));
  readback.unmap();
  return output;
}


const scene = makeScene();
const builtRootZeroBase = buildCompressedWideBvh(scene.positions, scene.rootZeroIndices, scene.rootZeroMaterialIds, {
  maxLeafTriangles: 1,
});
const builtRootZero = {
  ...builtRootZeroBase,
  reorderedIndices: reorderCwbvhTrianglePayloads(builtRootZeroBase, scene.rootZeroPayloads),
};
const builtRootOneBase = buildCompressedWideBvh(scene.positions, scene.rootOneIndices, scene.rootOneMaterialIds, {
  maxLeafTriangles: 1,
});
const builtRootOne = {
  ...builtRootOneBase,
  reorderedIndices: reorderCwbvhTrianglePayloads(builtRootOneBase, scene.rootOnePayloads),
};
const nonzeroRoot = builtRootZero.cwbvhNodeCount;
const validBuilt = concatCwbvhRoots(builtRootZero, builtRootOne);
const statusRoots = {};
let built = validBuilt;
for (const [name, kind] of [
  ["emptyLiveChild", "empty-live-child"],
  ["zeroCountLeaf", "zero-count-leaf"],
  ["invalidBounds", "invalid-bounds"],
  ["stackOverflow", "stack-overflow"],
]) {
  statusRoots[name] = built.cwbvhNodeCount;
  built = concatCwbvhRoots(built, syntheticStatusTree(kind));
}
const childBoundsPacked = packCwbvhBuildBoundsForWgsl(built);
const sourceRays = makeRays(nonzeroRoot, statusRoots);
const rayData = rayBuffer(sourceRays);
// CPU and GPU must consume byte-identical f32 rays. Comparing the original JS
// doubles against storage-buffer-rounded inputs produces false distance deltas
// at large coordinates even when the shader is bit-correct.
const rays = sourceRays.map((ray, i) => ({
  label: ray.label,
  origin: [rayData[i * 8 + 0], rayData[i * 8 + 1], rayData[i * 8 + 2]],
  tMax: rayData[i * 8 + 3],
  direction: [rayData[i * 8 + 4], rayData[i * 8 + 5], rayData[i * 8 + 6]],
  root: Math.trunc(rayData[i * 8 + 7]),
  expectedStatus: ray.expectedStatus,
}));
const outputWordsPerRay = 8;
const output = new Uint32Array(rays.length * outputWordsPerRay);

const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) {
  console.error("[cwbvh-parity] ERROR: no WebGPU adapter available");
  Deno.exit(1);
}
const maxStorageBuffersPerShaderStage = adapter.limits?.maxStorageBuffersPerShaderStage ?? 8;
const device = await adapter.requestDevice(
  maxStorageBuffersPerShaderStage > 8
    ? { requiredLimits: { maxStorageBuffersPerShaderStage } }
    : {},
);

const module = device.createShaderModule({
  label: "cwbvh-parity-oracle",
  code: shaderCode(rays.length, built.cwbvhNodeCount),
});
const info = await module.getCompilationInfo();
const errors = info.messages.filter((m) => m.type === "error");
if (errors.length > 0) {
  console.error("[cwbvh-parity] WGSL compile errors:");
  for (const e of errors) console.error(`  line ${e.lineNum}: ${e.message}`);
  Deno.exit(1);
}

device.pushErrorScope("validation");
const pipeline = await device.createComputePipelineAsync({
  label: "cwbvh-parity-oracle",
  layout: "auto",
  compute: { module, entryPoint: "main" },
});
const pipelineError = await device.popErrorScope();
if (pipelineError) {
  console.error(`[cwbvh-parity] pipeline error: ${pipelineError.message ?? pipelineError}`);
  Deno.exit(1);
}

const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const result = device.createBuffer({
  label: "cwbvh-parity-result",
  size: output.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});
const readback = device.createBuffer({
  label: "cwbvh-parity-readback",
  size: output.byteLength,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});

const bindGroup = device.createBindGroup({
  label: "cwbvh-parity-bindings",
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: makeBuffer(device, "cwbvhNodeBounds", built.cwbvhNodeBounds, storage) } },
    { binding: 1, resource: { buffer: makeBuffer(device, "cwbvhChildBoundsPacked", childBoundsPacked, storage) } },
    { binding: 2, resource: { buffer: makeBuffer(device, "cwbvhChildMeta", built.cwbvhChildMeta, storage) } },
    { binding: 3, resource: { buffer: makeBuffer(device, "cwbvhChildCount", built.cwbvhChildCount, storage) } },
    { binding: 4, resource: { buffer: makeBuffer(device, "bvh_index", built.reorderedIndices, storage) } },
    { binding: 5, resource: { buffer: makeBuffer(device, "bvh_position", scene.positions, storage) } },
    { binding: 6, resource: { buffer: makeBuffer(device, "cwbvhGateRays", rayData, storage) } },
    { binding: 7, resource: { buffer: result } },
  ],
});

device.pushErrorScope("validation");
device.pushErrorScope("internal");
device.pushErrorScope("out-of-memory");
const encoder = device.createCommandEncoder({ label: "cwbvh-parity-oracle" });
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.dispatchWorkgroups(Math.ceil(rays.length / 64), 1, 1);
pass.end();
encoder.copyBufferToBuffer(result, 0, readback, 0, output.byteLength);
device.queue.submit([encoder.finish()]);
await device.queue.onSubmittedWorkDone();
const oomError = await device.popErrorScope();
const internalError = await device.popErrorScope();
const validationError = await device.popErrorScope();
for (const err of [validationError, internalError, oomError]) {
  if (err) {
    console.error(`[cwbvh-parity] GPU error: ${err.message ?? err}`);
    Deno.exit(1);
  }
}

await readback.mapAsync(GPUMapMode.READ);
output.set(new Uint32Array(readback.getMappedRange()).slice(0, output.length));
readback.unmap();

const mismatches = [];
for (let i = 0; i < rays.length; i += 1) {
  const ray = rays[i];
  const gpuNoSkip = output.slice(i * outputWordsPerRay, i * outputWordsPerRay + 4);
  const gpuSkip = output.slice(i * outputWordsPerRay + 4, i * outputWordsPerRay + 8);
  if (ray.expectedStatus != null) {
    for (const [mode, gpu] of [["no-skip", gpuNoSkip], ["skip-glass", gpuSkip]]) {
      const closestStatus = gpu[0] >> 1;
      const anyStatus = gpu[3] >> 1;
      if (closestStatus !== ray.expectedStatus) {
        mismatches.push(`${ray.label}/${mode}/closest: status GPU=${closestStatus} expected=${ray.expectedStatus}`);
      }
      if (anyStatus !== ray.expectedStatus) {
        mismatches.push(`${ray.label}/${mode}/any: status GPU=${anyStatus} expected=${ray.expectedStatus}`);
      }
    }
    continue;
  }
  const root = ray.root ?? 0;
  const cpuNoSkip = intersectCompressedWideBvhFirstHit(built, scene.positions, ray, { root });
  const cpuSkip = intersectCompressedWideBvhFirstHit(built, scene.positions, ray, { root, skipGlass: true });
  const cpuAnyNoSkip = intersectCompressedWideBvhAnyHit(built, scene.positions, ray, { root, tMax: ray.tMax });
  const cpuAnySkip = intersectCompressedWideBvhAnyHit(built, scene.positions, ray, { root, tMax: ray.tMax, skipGlass: true });

  compareHit(ray.label, "closest", gpuNoSkip, cpuNoSkip, mismatches);
  compareHit(ray.label, "closest-skip-glass", gpuSkip, cpuSkip, mismatches);
  compareAny(ray.label, "any", gpuNoSkip[3], cpuAnyNoSkip, mismatches);
  compareAny(ray.label, "any-skip-glass", gpuSkip[3], cpuAnySkip, mismatches);
}

const tlasCase = makeTlasSceneAndRays();
const tlasCwbvh = unpackPackedCwbvh(tlasCase.packed);
const tlasGpu = await dispatchTlasDifferential(
  device,
  storage,
  tlasCase.packed,
  tlasCase.data,
  tlasCase.rays.length,
);
if (tlasGpu.length !== tlasCase.rays.length * 12) {
  throw new Error(
    `TLAS result buffer unreadable or truncated: got ${tlasGpu.length} words for ` +
    `${tlasCase.rays.length} rays`,
  );
}
for (let i = 0; i < tlasCase.rays.length; i += 1) {
  const ray = tlasCase.rays[i];
  const cpu = tracePackedTlasCpu(tlasCase.packed, tlasCwbvh, ray);
  const base = i * 12;
  const closestEncoded = tlasGpu[base];
  const gpuClosestStatus = closestEncoded >>> 1;
  const gpuDidHit = (closestEncoded & 1) === 1;
  const gpuTriIndex = gpuTriToSigned(tlasGpu[base + 1]);
  const gpuWorldT = f32BitsToNumber(tlasGpu[base + 2]);
  const gpuInstanceIndex = gpuTriToSigned(tlasGpu[base + 3]);
  const gpuNormal = [
    f32BitsToNumber(tlasGpu[base + 4]),
    f32BitsToNumber(tlasGpu[base + 5]),
    f32BitsToNumber(tlasGpu[base + 6]),
  ];
  const gpuFrontFace = tlasGpu[base + 7] === 1;
  const anyEncoded = tlasGpu[base + 8];
  const gpuAnyStatus = anyEncoded >>> 1;
  const gpuAnyHit = (anyEncoded & 1) === 1;
  const explicitClosestStatus = tlasGpu[base + 9];
  const explicitAnyStatus = tlasGpu[base + 10];
  const sentinel = tlasGpu[base + 11];

  if (gpuClosestStatus !== 0 || explicitClosestStatus !== 0) {
    mismatches.push(
      `${ray.label}/tlas-closest-status: encoded=${gpuClosestStatus} explicit=${explicitClosestStatus}`,
    );
  }
  if (gpuAnyStatus !== 0 || explicitAnyStatus !== 0) {
    mismatches.push(
      `${ray.label}/tlas-any-status: encoded=${gpuAnyStatus} explicit=${explicitAnyStatus}`,
    );
  }
  if (sentinel !== 0xc0b7a11e) {
    mismatches.push(`${ray.label}/tlas-readback-sentinel: GPU=0x${sentinel.toString(16)}`);
  }
  if (gpuDidHit !== cpu.didHit) {
    mismatches.push(`${ray.label}/tlas-closest: didHit GPU=${gpuDidHit} CPU=${cpu.didHit}`);
  }
  if (gpuAnyHit !== cpu.anyHit) {
    mismatches.push(`${ray.label}/tlas-any: didHit GPU=${gpuAnyHit} CPU=${cpu.anyHit}`);
  }
  const shouldHit = i < tlasCase.metadata.hitRayCount;
  if (cpu.didHit !== shouldHit || cpu.anyHit !== shouldHit) {
    mismatches.push(
      `${ray.label}/tlas-authored-expectation: closest=${cpu.didHit} any=${cpu.anyHit} expected=${shouldHit}`,
    );
  }
  if (!cpu.didHit || !gpuDidHit) continue;
  if (gpuTriIndex !== cpu.triangleIndex) {
    mismatches.push(`${ray.label}/tlas-triangle: GPU=${gpuTriIndex} CPU=${cpu.triangleIndex}`);
  }
  if (gpuInstanceIndex !== cpu.instanceIndex) {
    mismatches.push(`${ray.label}/tlas-instance: GPU=${gpuInstanceIndex} CPU=${cpu.instanceIndex}`);
  }
  if (!Number.isFinite(gpuWorldT) || Math.abs(gpuWorldT - cpu.dist) > DIST_TOL) {
    mismatches.push(`${ray.label}/tlas-world-t: GPU=${gpuWorldT} CPU=${cpu.dist}`);
  }
  if (gpuNormal.some((component) => !Number.isFinite(component))) {
    mismatches.push(`${ray.label}/tlas-normal: GPU contains non-finite component ${gpuNormal}`);
  } else {
    for (let component = 0; component < 3; component += 1) {
      if (Math.abs(gpuNormal[component] - cpu.normal[component]) > NORMAL_TOL) {
        mismatches.push(
          `${ray.label}/tlas-normal-${component}: GPU=${gpuNormal[component]} CPU=${cpu.normal[component]}`,
        );
      }
    }
  }
  if (gpuFrontFace !== cpu.frontFace) {
    mismatches.push(`${ray.label}/tlas-front-face: GPU=${gpuFrontFace} CPU=${cpu.frontFace}`);
  }
}

const status = {
  generatedAt: new Date().toISOString(),
  harness: "cwbvh-parity-oracle",
  verdict: mismatches.length === 0 ? "PASS" : "FAIL",
  command: "npm run behavioral-gate:cwbvh -- --write-status",
  rayCount: rays.length,
  rootCount: 2,
  statusRootCount: Object.keys(statusRoots).length,
  nonzeroRoot,
  cwbvhNodeCount: built.cwbvhNodeCount,
  triangleCount: Math.floor((scene.rootZeroIndices.length + scene.rootOneIndices.length) / 4),
  tlasRayCount: tlasCase.rays.length,
  tlasRawAuthoredInstanceCount: tlasCase.metadata.rawAuthoredInstanceCount,
  tlasPackedInstanceCount: tlasCase.metadata.packedInstanceCount,
  checks: {
    closestNoSkip: true,
    closestSkipGlass: true,
    anyNoSkip: true,
    anySkipGlass: true,
    nonzeroRootClosest: !mismatches.some((m) => m.startsWith("nonzero-root-triangle/closest")),
    nonzeroRootAny: !mismatches.some((m) => m.startsWith("nonzero-root-triangle/any")),
    emptyLiveChildInvalid: !mismatches.some((m) => m.startsWith("status-empty-live-child/")),
    zeroCountLeafInvalid: !mismatches.some((m) => m.startsWith("status-zero-count-leaf/")),
    invalidBoundsInvalid: !mismatches.some((m) => m.startsWith("status-invalid-parent-bounds/")),
    stackOverflowDistinct: !mismatches.some((m) => m.startsWith("status-stack-overflow/")),
    tlasIdentity: !mismatches.some((m) => m.includes("instance-0/tlas-")),
    tlasNonuniformRotationShear: !mismatches.some((m) => m.includes("instance-1/tlas-")),
    tlasMirroredAuthoredOrientation: !mismatches.some((m) => m.includes("instance-2/tlas-")),
    tlasSecondBlasRotationShear: !mismatches.some((m) => m.includes("instance-3/tlas-")),
    tlasSingularSkipped: tlasCase.metadata.singularSkipped &&
      !mismatches.some((m) => m.startsWith("tlas-singular-skipped-")),
    tlasWorldDistance: !mismatches.some((m) => m.includes("/tlas-world-t:")),
    tlasNormal: !mismatches.some((m) => m.includes("/tlas-normal")),
    tlasFrontFace: !mismatches.some((m) => m.includes("/tlas-front-face:")),
    tlasAnyAndStatus: !mismatches.some((m) =>
      m.includes("/tlas-any") || m.includes("/tlas-closest-status:")),
    tlasReadbackSentinel: !mismatches.some((m) => m.includes("/tlas-readback-sentinel:")),
  },
  mismatches,
};

if (WRITE_STATUS) {
  await Deno.writeTextFile(STATUS_PATH, `${JSON.stringify(status, null, 2)}\n`);
}

if (mismatches.length > 0) {
  console.error("[cwbvh-parity] FAIL");
  for (const mismatch of mismatches) console.error(`  ${mismatch}`);
  Deno.exit(1);
}

console.log("[cwbvh-parity] PASS");
console.log(`  rays          : ${rays.length}`);
console.log(`  roots         : 2 (nonzero root ${nonzeroRoot})`);
console.log(`  status roots  : ${Object.keys(statusRoots).length}`);
console.log(`  CWBVH nodes   : ${built.cwbvhNodeCount}`);
console.log(`  triangles     : ${Math.floor((scene.rootZeroIndices.length + scene.rootOneIndices.length) / 4)}`);
console.log(`  TLAS rays     : ${tlasCase.rays.length}`);
console.log(
  `  TLAS instances: ${tlasCase.metadata.packedInstanceCount} packed / ` +
  `${tlasCase.metadata.rawAuthoredInstanceCount} authored (singular skipped)`,
);
if (WRITE_STATUS) {
  console.log(`  status        : ${STATUS_PATH.pathname}`);
}
