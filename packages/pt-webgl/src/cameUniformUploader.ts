/**
 * cameUniformUploader.ts — Pack analytic H-channel came geometry into a
 * std140-aligned UBO Float32Array for the fork's intersectCameSegment /
 * intersectCameNode shader routines.
 *
 * Background: "came" (from the French "came de plomb") is the H-profile lead
 * channel used to hold individual stained-glass cells. Each segment is a
 * straight run of H-channel between two nodes; each node is a solder joint
 * where multiple segments meet. The analytic intersection shader avoids
 * tessellating these into 1500+ tube/sphere meshes and tests them as
 * closed-form CSG primitives instead (~30%+ BVH traversal savings on a
 * typical window with 500 segments).
 *
 * UBO format (std140):
 *
 *   Segments — 16 floats each (std140 vec4 boundary alignment):
 *     [0..2]  startWorld.xyz
 *     [3]     railWidth
 *     [4..6]  endWorld.xyz
 *     [7]     blockHeight
 *     [8]     webThickness
 *     [9..15] padding to reach 16-float (64-byte) boundary
 *
 *   Nodes — 4 floats each (vec4 aligned):
 *     [0..2]  position.xyz
 *     [3]     radius
 *
 * This layout matches the GLSL struct declarations documented in
 * plan/sprint-5-pt-fork-patch.md:
 *
 *   struct CameSegment { vec3 startWorld; float railWidth;
 *                        vec3 endWorld; float blockHeight;
 *                        float webThickness; float _pad[7]; };
 *   struct CameNode    { vec3 position; float radius; };
 *
 * References:
 *   - Phase 6 Sprint 5 spec (plan/sprint-5-pt-fork-patch.md)
 *   - OpenGL 4.6 spec §7.6.2.2 "Standard Uniform Block Layout" (std140)
 */

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/** One H-channel came rail segment between two node positions. */
export interface CameSegment {
  /** World-space start point (one end of the rail run). */
  readonly startWorld: readonly [number, number, number];
  /** World-space end point (the other end of the rail run). */
  readonly endWorld: readonly [number, number, number];
  /** Full width of the H-channel rail face (mm converted to scene units
   *  by the host). Typical lead came: 4–10 mm. */
  readonly railWidth: number;
  /** Full height of the H-channel cross-section (web depth).
   *  Typical: 6–12 mm. */
  readonly blockHeight: number;
  /** Thickness of the H-profile web (centre rib). Typical: 1–2 mm. */
  readonly webThickness: number;
}

/** One solder node — a rounded joint where multiple segments meet. */
export interface CameNode {
  /** World-space centre of the solder bead. */
  readonly position: readonly [number, number, number];
  /** Sphere radius of the solder bead (typically max(railWidth, blockHeight) / 2). */
  readonly radius: number;
}

export interface CameUploadOptions {
  /** Hard cap on segments uploaded to the UBO. Default 500.
   *  Segments beyond the cap are silently discarded with a console.warn. */
  readonly maxSegments?: number;
  /** Hard cap on nodes uploaded to the UBO. Default 200.
   *  Nodes beyond the cap are silently discarded with a console.warn. */
  readonly maxNodes?: number;
}

/** Packed UBO buffers ready for gl.bufferData / gl.uniformBlockBinding. */
export interface CamePackedUBO {
  /** Segment data — Float32Array of `segmentCount * SEGMENT_FLOATS` floats. */
  readonly segments: Float32Array;
  /** Node data — Float32Array of `nodeCount * NODE_FLOATS` floats. */
  readonly nodes: Float32Array;
  /** Number of segments actually packed (≤ maxSegments, ≤ input length). */
  readonly segmentCount: number;
  /** Number of nodes actually packed (≤ maxNodes, ≤ input length). */
  readonly nodeCount: number;
}

// ────────────────────────────────────────────────────────────────────────────
// Layout constants
// ────────────────────────────────────────────────────────────────────────────

/**
 * std140 requires each struct member aligned to its "base alignment":
 *   - vec3  → 16-byte alignment (treated as vec4 in std140)
 *   - float → 4-byte alignment
 *
 * CameSegment:
 *   vec3  startWorld  → 4 floats (vec4 padded) = floats 0-3
 *   float railWidth   → 1 float              = float 3   (shares vec4 with startWorld)
 *   vec3  endWorld    → 4 floats (next vec4)  = floats 4-7 (endWorld.xyz + blockHeight)
 *   float blockHeight → float 7               (shares vec4 with endWorld)
 *   float webThick    → float 8
 *   padding           → floats 9-15 (pad to 64-byte / 16-float boundary)
 *
 * Total: 16 floats per segment = 64 bytes.  Clean std140.
 */
const SEGMENT_FLOATS = 16;

/**
 * CameNode:
 *   vec3 position + float radius = 4 floats = 16 bytes (one vec4).
 */
const NODE_FLOATS = 4;

const DEFAULT_MAX_SEGMENTS = 500;
const DEFAULT_MAX_NODES = 200;

// ────────────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────────────

/**
 * Pack came segments and nodes into std140-aligned Float32Arrays ready for
 * upload to the fork's CameUBO uniform block.
 *
 * Segments beyond `maxSegments` (default 500) and nodes beyond `maxNodes`
 * (default 200) are silently discarded. The counts are reported in the
 * returned `CamePackedUBO` so the caller can set the UBO count uniforms.
 *
 * @param segments - Came rail segments (order irrelevant; shader tests all).
 * @param nodes    - Solder nodes (order irrelevant; shader tests all).
 * @param opts     - Optional caps (maxSegments, maxNodes).
 */
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

    // floats 0-2: startWorld.xyz
    segBuf[base + 0] = seg.startWorld[0];
    segBuf[base + 1] = seg.startWorld[1];
    segBuf[base + 2] = seg.startWorld[2];
    // float 3: railWidth (fills the std140 vec4 padding slot after startWorld)
    segBuf[base + 3] = seg.railWidth;

    // floats 4-6: endWorld.xyz
    segBuf[base + 4] = seg.endWorld[0];
    segBuf[base + 5] = seg.endWorld[1];
    segBuf[base + 6] = seg.endWorld[2];
    // float 7: blockHeight (fills the std140 vec4 padding slot after endWorld)
    segBuf[base + 7] = seg.blockHeight;

    // float 8: webThickness
    segBuf[base + 8] = seg.webThickness;

    // floats 9-15: explicit padding (zero-init from new Float32Array)
  }

  const nodeBuf = new Float32Array(nodeCount * NODE_FLOATS);
  for (let i = 0; i < nodeCount; i++) {
    const node = nodes[i]!;
    const base = i * NODE_FLOATS;

    // floats 0-2: position.xyz
    nodeBuf[base + 0] = node.position[0];
    nodeBuf[base + 1] = node.position[1];
    nodeBuf[base + 2] = node.position[2];
    // float 3: radius
    nodeBuf[base + 3] = node.radius;
  }

  return {
    segments: segBuf,
    nodes: nodeBuf,
    segmentCount: segCount,
    nodeCount,
  };
}
