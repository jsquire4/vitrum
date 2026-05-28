/**
 * Probe-update pass UBO layouts (W4c extract from probeUpdatePass.ts).
 */
import { defineUbo } from '@vitrum/shared-samplers';

export const PROBE_RAY_STRIDE_BYTES = 64;
export const DDGI_BORDER_UBO_BYTES = 32;

export const DDGI_BORDER_UBO = defineUbo([
  { name: 'numProbes',   type: 'u32' },
  { name: 'atlasWidth',  type: 'u32' },
  { name: 'atlasHeight', type: 'u32' },
  { name: '_pad0',       type: 'u32' },
  { name: 'gridDimX',    type: 'u32' },
  { name: 'gridDimY',    type: 'u32' },
  { name: 'gridDimZ',    type: 'u32' },
  { name: '_pad1',       type: 'u32' },
] as const);

export const DDGI_FRAME_PARAMS_UBO = defineUbo([
  { name: 'randomRotation', type: 'vec3f' },
  { name: 'frameIndex',     type: 'u32'   },
  { name: 'totalProbes',    type: 'u32'   },
  { name: 'probesPerFrame', type: 'u32'   },
  { name: '_pad0',          type: 'u32'   },
  { name: '_pad1',          type: 'u32'   },
  { name: 'skyTint',        type: 'vec3f' },
  { name: 'skyIrradiance',  type: 'f32'   },
  { name: 'glassMixScale',  type: 'f32'   },
  { name: '_pad2',          type: 'u32'   },
  { name: '_pad3',          type: 'u32'   },
  { name: '_pad4',          type: 'u32'   },
] as const);

export const DDGI_BLEND_PARAMS_UBO = defineUbo([
  { name: 'probesPerFrame', type: 'u32' },
  { name: 'hysteresis',     type: 'f32' },
] as const);
