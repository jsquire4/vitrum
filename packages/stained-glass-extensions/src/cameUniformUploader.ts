/**
 * Pack analytic H-channel came geometry into std140-aligned UBO arrays.
 */

export interface CameSegment {
  readonly startWorld: readonly [number, number, number];
  readonly endWorld: readonly [number, number, number];
  readonly railWidth: number;
  readonly blockHeight: number;
  readonly webThickness: number;
}

export interface CameNode {
  readonly position: readonly [number, number, number];
  readonly radius: number;
}

export interface CameUploadOptions {
  readonly maxSegments?: number;
  readonly maxNodes?: number;
}

export interface CamePackedUBO {
  readonly segments: Float32Array;
  readonly nodes: Float32Array;
  readonly segmentCount: number;
  readonly nodeCount: number;
}

const SEGMENT_FLOATS = 16;
const NODE_FLOATS = 4;
const DEFAULT_MAX_SEGMENTS = 500;
const DEFAULT_MAX_NODES = 200;

export function packCameUBO(
  segments: ReadonlyArray<CameSegment>,
  nodes: ReadonlyArray<CameNode>,
  opts?: CameUploadOptions,
): CamePackedUBO {
  const maxSeg = opts?.maxSegments ?? DEFAULT_MAX_SEGMENTS;
  const maxNode = opts?.maxNodes ?? DEFAULT_MAX_NODES;

  if (segments.length > maxSeg) {
    console.warn(
      `packCameUBO: received ${segments.length} segments but maxSegments=${maxSeg}; excess discarded.`,
    );
  }
  if (nodes.length > maxNode) {
    console.warn(
      `packCameUBO: received ${nodes.length} nodes but maxNodes=${maxNode}; excess discarded.`,
    );
  }

  const segCount = Math.min(segments.length, maxSeg);
  const nodeCount = Math.min(nodes.length, maxNode);

  const segBuf = new Float32Array(segCount * SEGMENT_FLOATS);
  for (let i = 0; i < segCount; i++) {
    const seg = segments[i]!;
    const base = i * SEGMENT_FLOATS;
    segBuf[base + 0] = seg.startWorld[0];
    segBuf[base + 1] = seg.startWorld[1];
    segBuf[base + 2] = seg.startWorld[2];
    segBuf[base + 3] = seg.railWidth;
    segBuf[base + 4] = seg.endWorld[0];
    segBuf[base + 5] = seg.endWorld[1];
    segBuf[base + 6] = seg.endWorld[2];
    segBuf[base + 7] = seg.blockHeight;
    segBuf[base + 8] = seg.webThickness;
  }

  const nodeBuf = new Float32Array(nodeCount * NODE_FLOATS);
  for (let i = 0; i < nodeCount; i++) {
    const node = nodes[i]!;
    const base = i * NODE_FLOATS;
    nodeBuf[base + 0] = node.position[0];
    nodeBuf[base + 1] = node.position[1];
    nodeBuf[base + 2] = node.position[2];
    nodeBuf[base + 3] = node.radius;
  }

  return {
    segments: segBuf,
    nodes: nodeBuf,
    segmentCount: segCount,
    nodeCount,
  };
}
