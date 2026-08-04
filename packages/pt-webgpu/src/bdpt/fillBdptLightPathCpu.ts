/**
 * CPU reference packing of one invocation-private BDPT light path.
 * Production writes the same eight-row columns into bounded private WGSL memory;
 * this flat array exists only so tests can inspect the byte/row contract.
 * Flattened row-minor as `idx = col * 8 + row` (matches WGSL `bdptLightPathIndex`):
 * per light-vertex column, row 0 = pos (+ kind in .w), row 1 = normal + pdfFwd,
 * row 2 = throughput + pdfRev, row 3 = (A9) matId (.w) + wo-toward-prev (.xyz),
 * row 4 = hit-local material payload (triIndex, baryVW, instanceIndex),
 * row 5 = interface eta metadata, row 6 = both medium sides and remaining
 * budgets, row 7 = both original medium caps.
 * Bounce 0 is the emitter (matId < 0 ⇒ Lambertian/emission profile).
 */

import { sampleBdptBounce0Cpu } from './bdptEmitterPickCpu.js';

const KIND_INVALID = 3;
const KIND_LIGHT = 0;
const LIGHT_PATH_ROWS = 8;
/** A9 — row-3 .w sentinel marking the emitter vertex (Lambertian/emission). */
const LV_EMITTER_MATID = -1;

/** vec4f index into the flat light-path buffer for (col, row). */
export function bdptLightPathColumnIndex(col: number, row: number): number {
  return (col * LIGHT_PATH_ROWS + row) * 4;
}
export interface BdptMediumSideRow {
  readonly incidentMatId: number;
  readonly incidentRemainingDistance: number;
  readonly transmittedMatId: number;
  readonly transmittedRemainingDistance: number;
}

export interface BdptMediumCapRow {
  readonly incidentInitialDistance: number;
  readonly transmittedInitialDistance: number;
}

/** CPU oracle for WGSL row 6: two bitcast u32 IDs and two f32 budgets. */
export function writeBdptMediumSideRow(
  data: Float32Array,
  col: number,
  row: BdptMediumSideRow,
): void {
  const offset = bdptLightPathColumnIndex(col, 6);
  const words = new Uint32Array(data.buffer, data.byteOffset, data.length);
  words[offset + 0] = row.incidentMatId >>> 0;
  data[offset + 1] = row.incidentRemainingDistance;
  words[offset + 2] = row.transmittedMatId >>> 0;
  data[offset + 3] = row.transmittedRemainingDistance;
}

export function readBdptMediumSideRow(
  data: Float32Array,
  col: number,
): BdptMediumSideRow {
  const offset = bdptLightPathColumnIndex(col, 6);
  const words = new Uint32Array(data.buffer, data.byteOffset, data.length);
  return {
    incidentMatId: words[offset + 0]!,
    incidentRemainingDistance: data[offset + 1]!,
    transmittedMatId: words[offset + 2]!,
    transmittedRemainingDistance: data[offset + 3]!,
  };
}

/** CPU oracle for WGSL row 7: original incident/transmitted finite caps. */
export function writeBdptMediumCapRow(
  data: Float32Array,
  col: number,
  row: BdptMediumCapRow,
): void {
  const offset = bdptLightPathColumnIndex(col, 7);
  data[offset + 0] = row.incidentInitialDistance;
  data[offset + 1] = row.transmittedInitialDistance;
  data[offset + 2] = 0;
  data[offset + 3] = 0;
}

export function readBdptMediumCapRow(
  data: Float32Array,
  col: number,
): BdptMediumCapRow {
  const offset = bdptLightPathColumnIndex(col, 7);
  return {
    incidentInitialDistance: data[offset + 0]!,
    transmittedInitialDistance: data[offset + 1]!,
  };
}


/** Flat vec4f light-path buffer contents (`col * 8 + row` ordering). */
export function packBdptLightPathColumns(
  width: number,
  bounce0: ReturnType<typeof sampleBdptBounce0Cpu>,
): Float32Array {
  const data = new Float32Array(width * LIGHT_PATH_ROWS * 4);
  for (let col = 0; col < width; col += 1) {
    data[bdptLightPathColumnIndex(col, 0) + 3] = KIND_INVALID;
    // A9 — every column's row-3 .w defaults to the emitter sentinel (Lambertian).
    data[bdptLightPathColumnIndex(col, 3) + 3] = LV_EMITTER_MATID;
    // A neutral interface is required for invalid/emitter columns because the
    // connection shader reads eta_t/eta_i directly from row 5.
    const etaOffset = bdptLightPathColumnIndex(col, 5);
    data[etaOffset + 0] = 1;
    data[etaOffset + 1] = 1;
    data[etaOffset + 2] = 1;
  }
  if (bounce0 == null) return data;
  const col = 0;
    writeBdptMediumSideRow(data, col, {
      incidentMatId: 0xffffffff,
      incidentRemainingDistance: Math.fround(3.402823e38),
      transmittedMatId: 0xffffffff,
      transmittedRemainingDistance: Math.fround(3.402823e38),
    });
    writeBdptMediumCapRow(data, col, {
      incidentInitialDistance: Math.fround(3.402823e38),
      transmittedInitialDistance: Math.fround(3.402823e38),
    });
  const o0 = bdptLightPathColumnIndex(col, 0);
  const o1 = bdptLightPathColumnIndex(col, 1);
  const o2 = bdptLightPathColumnIndex(col, 2);
  const o3 = bdptLightPathColumnIndex(col, 3);
  data[o0 + 0] = bounce0.emitPos[0];
  data[o0 + 1] = bounce0.emitPos[1];
  data[o0 + 2] = bounce0.emitPos[2];
  data[o0 + 3] = KIND_LIGHT;
  data[o1 + 0] = bounce0.emitNormal[0];
  data[o1 + 1] = bounce0.emitNormal[1];
  data[o1 + 2] = bounce0.emitNormal[2];
  data[o1 + 3] = bounce0.pdfJoint;
  data[o2 + 0] = bounce0.emitRad[0];
  data[o2 + 1] = bounce0.emitRad[1];
  data[o2 + 2] = bounce0.emitRad[2];
  data[o2 + 3] = bounce0.pdfHemi;
  // A9/PTWG-BDPT-01 — bounce-0 is an emitter vertex: finite area emitters use
  // -2 so the connection treats row-2 throughput as Le/(pdfPick*pdfArea);
  // legacy pseudo emitters/point lights use -1.
  const endpointData = bounce0.endpointData ?? bounce0.emitNormal;
  data[o3 + 0] = endpointData[0];
  data[o3 + 1] = endpointData[1];
  data[o3 + 2] = endpointData[2];
  data[o3 + 3] = bounce0.lvMatId ?? LV_EMITTER_MATID;
  return data;
}
