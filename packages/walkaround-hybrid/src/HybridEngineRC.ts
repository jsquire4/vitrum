/**
 * HybridEngineRC — Radiance Cascades subsystem for the HybridEngine path
 * (W8 sprint, started 2026-05-18).
 *
 * Owns:
 *   - One {@link RCDispatcher} per `GPUDevice`.
 *   - The 5 raw `GPUBuffer`s for the RC BVH (built from a `THREE.Scene` via
 *     `buildRCSceneBVH`, then uploaded as new raw buffers since HybridEngine
 *     doesn't run a THREE WebGPU renderer that would lazily allocate them).
 *   - The N raw cascade output `GPUBuffer`s (sized per `CASCADE_DIMS`).
 *
 * Lifecycle:
 *   1. `new RCSubsystem(device)` once per engine.
 *   2. `setScene(threeScene)` whenever the BVH source changes — this
 *      rebuilds the BVH + cascade buffers (~50 ms for ~30K-tri scenes).
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
import { RCDispatcher } from './rc/cascadeDispatch.js';
import { buildRCSceneBVH, type SceneBVH } from './rc/bvhCompute.js';
import { CASCADE_DIMS } from './rc/cascadePyramid.js';

interface RCBVHBuffers {
  readonly bvhNodesBuf:      GPUBuffer;
  readonly bvhIndicesBuf:    GPUBuffer;
  readonly bvhPositionsBuf:  GPUBuffer;
  readonly materialsBuf:     GPUBuffer;
  readonly triMaterialIdBuf: GPUBuffer;
}

interface RCFrameInputs {
  readonly sunDirection:        readonly [number, number, number];
  readonly sunColor:            readonly [number, number, number];
  readonly frameSeed:           number;
  readonly triIntersectEpsilon: number;
}

export class RCSubsystem {
  private readonly _device: GPUDevice;
  private _dispatcher: RCDispatcher | null = null;
  private _bvhBuffers: RCBVHBuffers | null = null;
  private _cascadeBufs: GPUBuffer[] | null = null;
  private _probeOriginWorld: readonly [number, number, number] | null = null;
  private _roomSize:         readonly [number, number, number] | null = null;

  constructor(device: GPUDevice) {
    this._device = device;
  }

  /**
   * Build (or rebuild) the RC BVH + cascade buffers from a `THREE.Scene`.
   * Cost: ~50 ms for ~30K-tri scenes (matches `buildRCSceneBVH` cost).
   * Call when the scene source changes.
   */
  setScene(threeScene: THREE.Scene): void {
    this._disposeSceneBuffers();

    const bvh = buildRCSceneBVH(threeScene);
    this._bvhBuffers = this._uploadBVH(bvh);

    // Plain `CascadeAABB` extracted from the THREE.Box3 — matches the
    // Phase 1A `allocateCascades` contract.
    const { min, max } = bvh.bounds;
    this._probeOriginWorld = [min.x, min.y, min.z];
    // Floor each axis at 1µm against degenerate (flat-plane) scenes.
    this._roomSize = [
      Math.max(max.x - min.x, 1e-6),
      Math.max(max.y - min.y, 1e-6),
      Math.max(max.z - min.z, 1e-6),
    ];

    this._cascadeBufs = this._allocateCascadeBuffers();
    this._dispatcher  = new RCDispatcher();
  }

  /**
   * Dispatch the cascade compute pipeline for one frame.
   * No-op if `setScene` hasn't been called yet.
   *
   * The dispatcher submits its own command buffer (independent of the
   * walkaround pipeline's encoder); cascade writes are visible by the time
   * the next `getCascadeC0Buffer()` consumer reads them within the same
   * frame as long as the consumer's pass is encoded AFTER `dispatchFrame`
   * returns — WebGPU guarantees command-buffer ordering on the same queue.
   */
  dispatchFrame(inputs: RCFrameInputs): void {
    if (!this._dispatcher || !this._bvhBuffers || !this._cascadeBufs ||
        !this._probeOriginWorld || !this._roomSize) {
      return;
    }
    void this._dispatcher.dispatchFrameRaw({
      device: this._device,
      bvhNodesBuf:      this._bvhBuffers.bvhNodesBuf,
      bvhIndicesBuf:    this._bvhBuffers.bvhIndicesBuf,
      bvhPositionsBuf:  this._bvhBuffers.bvhPositionsBuf,
      materialsBuf:     this._bvhBuffers.materialsBuf,
      triMaterialIdBuf: this._bvhBuffers.triMaterialIdBuf,
      cascadeBufs:      this._cascadeBufs,
      probeOriginWorld: this._probeOriginWorld,
      roomSize:         this._roomSize,
      sunDirection:     inputs.sunDirection,
      sunColor:         inputs.sunColor,
      frameSeed:        inputs.frameSeed,
      triIntersectEpsilon: inputs.triIntersectEpsilon,
    });
  }

  /**
   * Cascade-0 `GPUBuffer` for the shade pass to bind (W8 Phase 3).
   * Returns null when `setScene` hasn't been called yet.
   */
  getCascadeC0Buffer(): GPUBuffer | null {
    return this._cascadeBufs?.[0] ?? null;
  }

  /** Cascade-0 dimensions — required by the shade pass to compute probe-index
   *  arithmetic for sampling. */
  getCascadeC0Dims(): { probes: readonly [number, number, number]; rays: number } | null {
    if (!this._cascadeBufs) return null;
    const c0 = CASCADE_DIMS[0]!;
    return { probes: c0.probes, rays: c0.rays };
  }

  /** Probe-grid origin + size in world units. Returns null when no scene set. */
  getCascadeGeometry(): {
    probeOriginWorld: readonly [number, number, number];
    roomSize: readonly [number, number, number];
  } | null {
    if (!this._probeOriginWorld || !this._roomSize) return null;
    return { probeOriginWorld: this._probeOriginWorld, roomSize: this._roomSize };
  }

  /** Release all GPU resources. */
  dispose(): void {
    this._disposeSceneBuffers();
    if (this._dispatcher) {
      this._dispatcher.dispose();
      this._dispatcher = null;
    }
  }

  // ── Private helpers ──────────────────────────────────────────────────────

  private _uploadBVH(bvh: SceneBVH): RCBVHBuffers {
    return {
      bvhNodesBuf:      this._uploadAttribute(bvh.bvhNodes,      'rc-bvh-nodes'),
      bvhIndicesBuf:    this._uploadAttribute(bvh.indices,        'rc-bvh-indices'),
      bvhPositionsBuf:  this._uploadAttribute(bvh.positions,      'rc-bvh-positions'),
      materialsBuf:     this._uploadAttribute(bvh.materials,      'rc-bvh-materials'),
      triMaterialIdBuf: this._uploadAttribute(bvh.triMaterialId,  'rc-bvh-tri-mat-id'),
    };
  }

  /**
   * Allocate a raw `GPUBuffer` and upload the StorageBufferAttribute's
   * typed-array contents. Avoids the THREE WebGPU renderer's lazy-allocation
   * path (which would never run in the HybridEngine context).
   */
  private _uploadAttribute(attr: StorageBufferAttribute, label: string): GPUBuffer {
    // StorageBufferAttribute.array can be Float32Array OR Uint32Array
    // (e.g. `triMaterialId` is u32). Use the underlying ArrayBuffer to
    // copy in one shot regardless of typed-array variant.
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
    return CASCADE_DIMS.map((c, k) => {
      const totalRays = c.probes[0] * c.probes[1] * c.probes[2] * c.rays;
      // vec4f per ray = 16 bytes. COPY_SRC so a future debug readback works.
      return this._device.createBuffer({
        label: `rc-cascade-C${k}`,
        size:  totalRays * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    });
  }

  private _disposeSceneBuffers(): void {
    if (this._bvhBuffers) {
      this._bvhBuffers.bvhNodesBuf.destroy();
      this._bvhBuffers.bvhIndicesBuf.destroy();
      this._bvhBuffers.bvhPositionsBuf.destroy();
      this._bvhBuffers.materialsBuf.destroy();
      this._bvhBuffers.triMaterialIdBuf.destroy();
      this._bvhBuffers = null;
    }
    if (this._cascadeBufs) {
      for (const b of this._cascadeBufs) b.destroy();
      this._cascadeBufs = null;
    }
    this._probeOriginWorld = null;
    this._roomSize         = null;
  }
}
