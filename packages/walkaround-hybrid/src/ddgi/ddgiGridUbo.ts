/**
 * DDGI grid-params UBO layout (64 B) — shared by shade.wgsl and probeUpdateRays.
 * Lives under `ddgi/` so probeUpdatePass does not import `resourceManager`.
 */
import { defineUbo } from '@vitrum/shared-samplers';

const DDGI_GRID_UBO = defineUbo([
  { name: 'origin',   type: 'vec3f' },
  { name: 'spacing',  type: 'f32'   },
  { name: 'dimsX',    type: 'u32'   },
  { name: 'dimsY',    type: 'u32'   },
  { name: 'dimsZ',    type: 'u32'   },
  { name: '_pad0',    type: 'u32'   },
  { name: 'irrW',     type: 'f32'   },
  { name: 'irrH',     type: 'f32'   },
  { name: 'visW',     type: 'f32'   },
  { name: 'visH',     type: 'f32'   },
  { name: '_reserved0', type: 'f32' },
  { name: '_reserved1', type: 'f32' },
  { name: '_reserved2', type: 'f32' },
  { name: '_reserved3', type: 'f32' },
] as const);

export type DDGIGridParamsInput = {
  origin: { x: number; y: number; z: number };
  spacing: number;
  dims: { x: number; y: number; z: number };
  irradianceAtlasW: number;
  irradianceAtlasH: number;
  visibilityAtlasW: number;
  visibilityAtlasH: number;
};

function packGridParams(p: DDGIGridParamsInput): ArrayBuffer {
  const buf = new ArrayBuffer(DDGI_GRID_UBO.sizeBytes);
  DDGI_GRID_UBO.pack(new DataView(buf), 0, {
    origin: [p.origin.x, p.origin.y, p.origin.z] as const,
    spacing: p.spacing,
    dimsX: p.dims.x,
    dimsY: p.dims.y,
    dimsZ: p.dims.z,
    _pad0: 0,
    irrW: p.irradianceAtlasW,
    irrH: p.irradianceAtlasH,
    visW: p.visibilityAtlasW,
    visH: p.visibilityAtlasH,
    _reserved0: 0, _reserved1: 0, _reserved2: 0, _reserved3: 0,
  });
  return buf;
}

/** Zero-grid UBO — dimsX=1 gates `isDDGIWired()` false in shade.wgsl. */
export function buildDDGIPlaceholderUBO(): Float32Array<ArrayBuffer> {
  return new Float32Array(
    packGridParams({
      origin: { x: 0, y: 0, z: 0 },
      spacing: 24,
      dims: { x: 1, y: 1, z: 1 },
      irradianceAtlasW: 1,
      irradianceAtlasH: 1,
      visibilityAtlasW: 1,
      visibilityAtlasH: 1,
    }),
  );
}

/** Pack live probe-grid params for shade + probe update passes. */
export function packDDGIGridParams(p: DDGIGridParamsInput): ArrayBuffer {
  return packGridParams(p);
}
