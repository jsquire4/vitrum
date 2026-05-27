/**
 * HybridEngineRC — Radiance Cascades subsystem for the HybridEngine path
 * (W8 sprint, started 2026-05-18).
 *
 * Owns:
 *   - One {@link RCDispatcher} per `GPUDevice`.
 *   - The 5 raw `GPUBuffer`s for the RC BVH (built from a `THREE.Scene` via
 *     `buildRCSceneBVH`, or shared from ReSTIR when `bvhMode === 'tlas'`).
 *   - The N raw cascade output `GPUBuffer`s (sized per `CASCADE_DIMS`).
 *
 * Lifecycle:
 *   1. `new RCSubsystem(device)` once per engine.
 *   2. `setScene(threeScene)` for merged single-root BVH, or
 *      `syncRestirBvhBuffers(sceneBVH)` when ReSTIR uses TLAS (C2).
 *   3. `dispatchFrame({ sunDirection, sunColor, frameSeed, triIntersectEpsilon })`
 *      per frame.
 *   4. `getCascadeC0Buffer()` returns the cascade-0 `GPUBuffer` for the
 *      shade pass to sample (W8 Phase 3 wiring).
 *   5. `dispose()` releases all GPU resources.
 *
 * Plan: `plan/w8-rc-mis-composition.md` (Phase 2 section).
 */

import type * as THREE from 'three';
import type { StorageBufferAttribute } from 'three/webgpu';
import { RCDispatcher, buildRCSceneBVH, packCascadeMaterials, CASCADE_DIMS, type SceneBVH, type CascadeDim } from '@vitrum/walkaround-rc';
import type { SceneBVHBuffers } from './restir/bvhCompute.js';
import { makeRestirBvhSnapshot, type RestirBvhSnapshot } from './restir/restirBvhSnapshot.js';

interface RCBVHBuffers {
  readonly bvhNodesBuf:      GPUBuffer;
  readonly bvhIndicesBuf:    GPUBuffer;
  readonly bvhPositionsBuf:  GPUBuffer;
  readonly materialsBuf:     GPUBuffer;
  readonly triMaterialIdBuf: GPUBuffer;
  readonly tlasNodesBuf?:     GPUBuffer;
  readonly tlasInstanceIndicesBuf?: GPUBuffer;
  readonly tlasBlasRootsBuf?: GPUBuffer;
  readonly tlasInstanceWorldToLocalBuf?: GPUBuffer;
  readonly tlasInstanceLocalToWorldBuf?: GPUBuffer;
}

interface RCFrameInputs {
  readonly sunDirection:        readonly [number, number, number];
  readonly sunColor:            readonly [number, number, number];
  readonly frameSeed:           number;
  readonly triIntersectEpsilon: number;
}

/**
 * Pack the WGSL {@link sampleCascadeC0.wgsl} `RCParams` struct (64 bytes).
 * Layout — must match the WGSL struct declaration exactly:
 *   probeOriginWorld: vec3f  (offset 0..11)
 *   rcWeight:         f32    (offset 12..15)
 *   roomSize:         vec3f  (offset 16..27)
 *   enabled:          u32    (offset 28..31)
 *   probeCount:       vec3u  (offset 32..43)
 *   raysPerProbe:     u32    (offset 44..47)
 *   rayGridSize:      u32    (offset 48..51)
 *   _pad0/1/2:        3 × u32 (offset 52..63)
 */
export function packRCParams(
  probeOriginWorld: readonly [number, number, number],
  roomSize:         readonly [number, number, number],
  probeCount:       readonly [number, number, number],
  raysPerProbe:     number,
  rcWeight:         number,
  enabled:          boolean,
): ArrayBuffer {
  const buf = new ArrayBuffer(64);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  f[0] = probeOriginWorld[0]; f[1] = probeOriginWorld[1]; f[2] = probeOriginWorld[2];
  f[3] = rcWeight;
  f[4] = roomSize[0]; f[5] = roomSize[1]; f[6] = roomSize[2];
  u[7] = enabled ? 1 : 0;
  u[8] = probeCount[0]; u[9] = probeCount[1]; u[10] = probeCount[2];
  u[11] = raysPerProbe;
  u[12] = Math.max(1, Math.round(Math.sqrt(raysPerProbe))); // rayGridSize
  // u[13..15] pad (already zero from ArrayBuffer init).
  return buf;
}

export class RCSubsystem {
  private readonly _device: GPUDevice;
  private readonly _cascadeDims: readonly CascadeDim[];
  private _dispatcher: RCDispatcher | null = null;
  private _bvhBuffers: RCBVHBuffers | null = null;
  private _cascadeBufs: GPUBuffer[] | null = null;
  private _probeOriginWorld: readonly [number, number, number] | null = null;
  private _roomSize:         readonly [number, number, number] | null = null;
  private _restirSnapshot: RestirBvhSnapshot | null = null;
  private _bvhMode: 'merged' | 'tlas' = 'merged';
  private _tlasNodeCount = 0;
  private _lastBvhVersion = 0;
  private _lastBlasVersion = -1;
  private _lastTlasVersion = -1;

  constructor(device: GPUDevice, cascadeDims: readonly CascadeDim[] = CASCADE_DIMS) {
    this._device = device;
    this._cascadeDims = cascadeDims;
  }

  buildRCInputs(rcWeight: number): { cascade0Buffer: GPUBuffer; paramsBytes: ArrayBuffer } | null {
    if (!this._cascadeBufs || !this._probeOriginWorld || !this._roomSize) return null;
    const c0 = this._cascadeDims[0]!;
    return {
      cascade0Buffer: this._cascadeBufs[0]!,
      paramsBytes: packRCParams(
        this._probeOriginWorld,
        this._roomSize,
        c0.probes,
        c0.rays,
        rcWeight,
        true,
      ),
    };
  }

  refitCascadeBounds(
    boundsMin: readonly [number, number, number],
    boundsMax: readonly [number, number, number],
  ): void {
    if (!this._cascadeBufs) return;
    this._probeOriginWorld = [boundsMin[0], boundsMin[1], boundsMin[2]];
    this._roomSize = [
      Math.max(boundsMax[0] - boundsMin[0], 1e-6),
      Math.max(boundsMax[1] - boundsMin[1], 1e-6),
      Math.max(boundsMax[2] - boundsMin[2], 1e-6),
    ];
  }

  /**
   * C2 — share ReSTIR BLAS/TLAS buffers for RC probe rays (multi-mesh / instanced).
   */
  syncRestirBvhBuffers(buffers: SceneBVHBuffers | null): void {
    if (buffers == null || buffers.bvhMode !== 'tlas') {
      this._restirSnapshot = null;
      this._bvhMode = 'merged';
      this._tlasNodeCount = 0;
      return;
    }
    const snap = makeRestirBvhSnapshot(buffers);
    this._restirSnapshot = snap;
    this._bvhMode = 'tlas';
    this._tlasNodeCount = snap.tlasNodeCount;

    const { min, max } = snap.boundingBox;
    this._probeOriginWorld = [min.x, min.y, min.z];
    this._roomSize = [
      Math.max(max.x - min.x, 1e-6),
      Math.max(max.y - min.y, 1e-6),
      Math.max(max.z - min.z, 1e-6),
    ];

    if (snap.contentVersion !== this._lastBvhVersion) {
      const tlasOnly =
        this._bvhBuffers != null &&
        snap.tlas != null &&
        snap.blasContentVersion === this._lastBlasVersion &&
        snap.tlasContentVersion !== this._lastTlasVersion;
      if (tlasOnly) {
        this._refitTlasGpuBuffers(snap.tlas);
      } else {
        this._disposeBvhBuffersOnly();
        this._bvhBuffers = this._uploadFromRestirSnapshot(snap);
        this._dispatcher?.dispose();
        this._dispatcher = null;
      }
      this._lastBvhVersion = snap.contentVersion;
      this._lastBlasVersion = snap.blasContentVersion;
      this._lastTlasVersion = snap.tlasContentVersion;
    }

    if (this._cascadeBufs == null) {
      this._cascadeBufs = this._allocateCascadeBuffers();
    }
    if (this._dispatcher == null) {
      this._dispatcher = new RCDispatcher(this._cascadeDims);
    }
  }

  setScene(threeScene: THREE.Scene): void {
    this._restirSnapshot = null;
    this._bvhMode = 'merged';
    this._tlasNodeCount = 0;
    this._lastBvhVersion = 0;
    this._lastBlasVersion = -1;
    this._lastTlasVersion = -1;
    this._disposeSceneBuffers();
    if (this._dispatcher) {
      this._dispatcher.dispose();
      this._dispatcher = null;
    }

    const bvh = buildRCSceneBVH(threeScene);
    this._bvhBuffers = this._uploadBVH(bvh);

    const { min, max } = bvh.bounds;
    this._probeOriginWorld = [min.x, min.y, min.z];
    this._roomSize = [
      Math.max(max.x - min.x, 1e-6),
      Math.max(max.y - min.y, 1e-6),
      Math.max(max.z - min.z, 1e-6),
    ];

    this._cascadeBufs = this._allocateCascadeBuffers();
    this._dispatcher = new RCDispatcher(this._cascadeDims);
  }

  dispatchFrame(inputs: RCFrameInputs): void {
    if (!this._dispatcher || !this._bvhBuffers || !this._cascadeBufs ||
        !this._probeOriginWorld || !this._roomSize) {
      return;
    }
    const bvh = this._bvhBuffers;
    this._dispatcher.dispatchFrameRaw({
      device: this._device,
      bvhNodesBuf:      bvh.bvhNodesBuf,
      bvhIndicesBuf:    bvh.bvhIndicesBuf,
      bvhPositionsBuf:  bvh.bvhPositionsBuf,
      materialsBuf:     bvh.materialsBuf,
      triMaterialIdBuf: bvh.triMaterialIdBuf,
      cascadeBufs:      this._cascadeBufs,
      probeOriginWorld: this._probeOriginWorld,
      roomSize:         this._roomSize,
      sunDirection:     inputs.sunDirection,
      sunColor:         inputs.sunColor,
      frameSeed:        inputs.frameSeed,
      triIntersectEpsilon: inputs.triIntersectEpsilon,
      bvhMode:          this._bvhMode,
      tlasNodeCount:    this._tlasNodeCount,
      ...(bvh.tlasNodesBuf != null
        ? {
            tlasNodesBuf: bvh.tlasNodesBuf,
            tlasInstanceIndicesBuf: bvh.tlasInstanceIndicesBuf!,
            tlasBlasRootsBuf: bvh.tlasBlasRootsBuf!,
            tlasInstanceWorldToLocalBuf: bvh.tlasInstanceWorldToLocalBuf!,
            tlasInstanceLocalToWorldBuf: bvh.tlasInstanceLocalToWorldBuf!,
          }
        : {}),
    });
  }

  getCascadeC0Buffer(): GPUBuffer | null {
    return this._cascadeBufs?.[0] ?? null;
  }

  getCascadeC0Dims(): { probes: readonly [number, number, number]; rays: number } | null {
    if (!this._cascadeBufs) return null;
    const c0 = this._cascadeDims[0]!;
    return { probes: c0.probes, rays: c0.rays };
  }

  getCascadeGeometry(): {
    probeOriginWorld: readonly [number, number, number];
    roomSize: readonly [number, number, number];
  } | null {
    if (!this._probeOriginWorld || !this._roomSize) return null;
    return { probeOriginWorld: this._probeOriginWorld, roomSize: this._roomSize };
  }

  dispose(): void {
    this._disposeSceneBuffers();
    if (this._dispatcher) {
      this._dispatcher.dispose();
      this._dispatcher = null;
    }
  }

  private _refitTlasGpuBuffers(
    tlas: NonNullable<RestirBvhSnapshot['tlas']>,
  ): void {
    const bvh = this._bvhBuffers;
    if (bvh?.tlasNodesBuf == null) return;
    const q = this._device.queue;
    q.writeBuffer(bvh.tlasNodesBuf, 0, tlas.nodes);
    q.writeBuffer(bvh.tlasInstanceWorldToLocalBuf!, 0, tlas.worldToLocal);
    q.writeBuffer(bvh.tlasInstanceLocalToWorldBuf!, 0, tlas.localToWorld);
  }

  private _uploadFromRestirSnapshot(snap: RestirBvhSnapshot): RCBVHBuffers {
    const device = this._device;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const upload = (label: string, data: ArrayBuffer): GPUBuffer => {
      const buf = device.createBuffer({
        label,
        size: Math.max(data.byteLength, 16),
        usage,
        mappedAtCreation: true,
      });
      new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data));
      buf.unmap();
      return buf;
    };
    const matFloats = packCascadeMaterials([...snap.materials]);
    const tlas = snap.tlas;
    return {
      bvhNodesBuf: upload('rc-restir-bvh-nodes', snap.bvhNodes),
      bvhIndicesBuf: upload('rc-restir-bvh-index', snap.bvhIndex),
      bvhPositionsBuf: upload('rc-restir-bvh-positions', snap.positions),
      materialsBuf: upload('rc-restir-bvh-materials', matFloats.buffer as ArrayBuffer),
      triMaterialIdBuf: upload('rc-restir-bvh-tri-mat', snap.triMaterialIds),
      ...(tlas != null
        ? {
            tlasNodesBuf: upload('rc-restir-tlas-nodes', tlas.nodes),
            tlasInstanceIndicesBuf: upload('rc-restir-tlas-inst', tlas.instanceIndices),
            tlasBlasRootsBuf: upload('rc-restir-tlas-blas', tlas.blasRoots),
            tlasInstanceWorldToLocalBuf: upload('rc-restir-tlas-w2l', tlas.worldToLocal),
            tlasInstanceLocalToWorldBuf: upload('rc-restir-tlas-l2w', tlas.localToWorld),
          }
        : {}),
    };
  }

  private _uploadBVH(bvh: SceneBVH): RCBVHBuffers {
    return {
      bvhNodesBuf:      this._uploadAttribute(bvh.bvhNodes,      'rc-bvh-nodes'),
      bvhIndicesBuf:    this._uploadAttribute(bvh.indices,        'rc-bvh-indices'),
      bvhPositionsBuf:  this._uploadAttribute(bvh.positions,      'rc-bvh-positions'),
      materialsBuf:     this._uploadAttribute(bvh.materials,      'rc-bvh-materials'),
      triMaterialIdBuf: this._uploadAttribute(bvh.triMaterialId,  'rc-bvh-tri-mat-id'),
    };
  }

  private _uploadAttribute(attr: StorageBufferAttribute, label: string): GPUBuffer {
    const arr = attr.array as Float32Array | Uint32Array;
    const buf = this._device.createBuffer({
      label,
      size:  arr.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength));
    buf.unmap();
    return buf;
  }

  private _allocateCascadeBuffers(): GPUBuffer[] {
    return this._cascadeDims.map((c, k) => {
      const totalRays = c.probes[0] * c.probes[1] * c.probes[2] * c.rays;
      return this._device.createBuffer({
        label: `rc-cascade-C${k}`,
        size:  totalRays * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    });
  }

  private _disposeBvhBuffersOnly(): void {
    if (!this._bvhBuffers) return;
    this._bvhBuffers.bvhNodesBuf.destroy();
    this._bvhBuffers.bvhIndicesBuf.destroy();
    this._bvhBuffers.bvhPositionsBuf.destroy();
    this._bvhBuffers.materialsBuf.destroy();
    this._bvhBuffers.triMaterialIdBuf.destroy();
    this._bvhBuffers.tlasNodesBuf?.destroy();
    this._bvhBuffers.tlasInstanceIndicesBuf?.destroy();
    this._bvhBuffers.tlasBlasRootsBuf?.destroy();
    this._bvhBuffers.tlasInstanceWorldToLocalBuf?.destroy();
    this._bvhBuffers.tlasInstanceLocalToWorldBuf?.destroy();
    this._bvhBuffers = null;
  }

  private _disposeSceneBuffers(): void {
    this._disposeBvhBuffersOnly();
    if (this._cascadeBufs) {
      for (const b of this._cascadeBufs) b.destroy();
      this._cascadeBufs = null;
    }
    this._probeOriginWorld = null;
    this._roomSize         = null;
    this._restirSnapshot = null;
    this._lastBvhVersion = 0;
    this._lastBlasVersion = -1;
    this._lastTlasVersion = -1;
  }
}
