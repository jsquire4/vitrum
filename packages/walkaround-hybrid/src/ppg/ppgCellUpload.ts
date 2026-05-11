/**
 * CPU helpers for PPG spatial cell buffer uploads (Sprint 11).
 *
 * Encodes {@link PPGSpatialCell} records for WGSL and builds a uniform axis-aligned
 * grid over scene bounds for online path-guiding training.
 */

import { PPG_CELL_BYTE_STRIDE, PPG_MAX_SPATIAL_CELLS } from './types.js';

export interface PpgCellPosition {
  readonly position: readonly [number, number, number];
}

/**
 * World-space AABB from `SceneBVHBuffers.bvhPositions` (vec4 stride: xyz + packed w).
 */
export function aabbFromBvhPositions(
  positionsBytes: ArrayBuffer,
  vertexCount: number,
): { readonly min: readonly [number, number, number]; readonly max: readonly [number, number, number] } {
  const f = new Float32Array(positionsBytes, 0, Math.min(vertexCount * 4, (positionsBytes.byteLength / 4) | 0));
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < vertexCount; i++) {
    const x = f[i * 4]!;
    const y = f[i * 4 + 1]!;
    const z = f[i * 4 + 2]!;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  if (!Number.isFinite(minX) || !Number.isFinite(maxX)) {
    return {
      min: [0, 0, 0] as const,
      max: [1, 1, 1] as const,
    };
  }
  return { min: [minX, minY, minZ] as const, max: [maxX, maxY, maxZ] as const };
}

/**
 * Subdivide an AABB into at most `maxCells` cells with aspect-aware resolution
 * (longer edges get more divisions). Cell positions are voxel centers.
 */
export function buildPpgUniformGridCells(
  min: readonly [number, number, number],
  max: readonly [number, number, number],
  maxCells: number,
  paddingFrac = 0.02,
): PpgCellPosition[] {
  const cap = Math.max(1, Math.min(maxCells >>> 0, PPG_MAX_SPATIAL_CELLS));
  const px = Math.abs(max[0] - min[0]) * paddingFrac;
  const py = Math.abs(max[1] - min[1]) * paddingFrac;
  const pz = Math.abs(max[2] - min[2]) * paddingFrac;
  const lo: [number, number, number] = [min[0] - px, min[1] - py, min[2] - pz];
  const hi: [number, number, number] = [max[0] + px, max[1] + py, max[2] + pz];
  const lx = Math.max(1e-6, hi[0] - lo[0]);
  const ly = Math.max(1e-6, hi[1] - lo[1]);
  const lz = Math.max(1e-6, hi[2] - lo[2]);

  let nx = 1;
  let ny = 1;
  let nz = 1;
  for (;;) {
    const sx = lx / nx;
    const sy = ly / ny;
    const sz = lz / nz;
    let nx2 = nx;
    let ny2 = ny;
    let nz2 = nz;
    if (sx >= sy && sx >= sz) nx2++;
    else if (sy >= sz) ny2++;
    else nz2++;
    if (nx2 * ny2 * nz2 > cap) break;
    nx = nx2;
    ny = ny2;
    nz = nz2;
  }

  const out: PpgCellPosition[] = [];
  const fx = lx / nx;
  const fy = ly / ny;
  const fz = lz / nz;
  for (let iz = 0; iz < nz && out.length < cap; iz++) {
    for (let iy = 0; iy < ny && out.length < cap; iy++) {
      for (let ix = 0; ix < nx && out.length < cap; ix++) {
        out.push({
          position: [
            lo[0] + (ix + 0.5) * fx,
            lo[1] + (iy + 0.5) * fy,
            lo[2] + (iz + 0.5) * fz,
          ] as const,
        });
      }
    }
  }
  return out;
}

/**
 * Pack the first `activeCount` cells into a byte view sized to `cellBufferBytes`
 * (typically `PPGBuffers.cellBuffer.size`). Unused tail is zeroed.
 */
export function encodePpgCellGpuBytes(
  cells: ReadonlyArray<PpgCellPosition>,
  activeCount: number,
  cellBufferBytes: number,
): Uint8Array {
  const stride = PPG_CELL_BYTE_STRIDE;
  const maxCells = (cellBufferBytes / stride) | 0;
  if (activeCount < 0 || activeCount > maxCells) {
    throw new RangeError(
      `[PPG] encodePpgCellGpuBytes: activeCount ${activeCount} out of range [0, ${maxCells}]`,
    );
  }
  if (activeCount > cells.length) {
    throw new RangeError(
      `[PPG] encodePpgCellGpuBytes: activeCount ${activeCount} > cells.length ${cells.length}`,
    );
  }
  const out = new Uint8Array(cellBufferBytes);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < activeCount; i++) {
    const p = cells[i]!.position;
    const o = i * stride;
    dv.setFloat32(o, p[0], true);
    dv.setFloat32(o + 4, p[1], true);
    dv.setFloat32(o + 8, p[2], true);
    dv.setFloat32(o + 12, 0, true);
    dv.setUint32(o + 16, i >>> 0, true);
    dv.setUint32(o + 20, 0, true);
    dv.setUint32(o + 24, 0, true);
    dv.setUint32(o + 28, 0, true);
  }
  return out;
}
