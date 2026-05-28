/**
 * DDGI probe-update light UBO packing (up to 16 lights × 64 B).
 */
import type { DDGILight } from './types.js';

const MAX_DDGI_PROBE_LIGHTS = 16;
const LIGHT_STRIDE_FLOATS = 16;

export function packDDGIProbeLights(
  lights: readonly DDGILight[],
  sunIntensityMul: number,
): ArrayBuffer {
  const headerSize = 4;
  const data = new Float32Array(headerSize + MAX_DDGI_PROBE_LIGHTS * LIGHT_STRIDE_FLOATS);
  const udata = new Uint32Array(data.buffer);
  const active = lights.filter((l) => l.on);
  udata[0] = Math.min(active.length, MAX_DDGI_PROBE_LIGHTS);

  active.slice(0, MAX_DDGI_PROBE_LIGHTS).forEach((l, i) => {
    const base = headerSize + i * LIGHT_STRIDE_FLOATS;
    const ubase = base;
    if (l.kind === 'sun') {
      // Sun travel direction. When a `@vitrum/core` `directional` emitter
      // drives the sun, `coreEmitterToDDGILight` carries its real direction
      // here (already negated to a travel direction). The WGSL shader negates
      // again (`normalize(-light.direction)`) to recover the toward-light dir.
      // Absent direction → legacy hardcoded straight-down sun (0,-1,0) so a
      // host-supplied sun light with no direction is unchanged.
      const dir = l.direction;
      const col = l.color;
      udata[ubase] = 0;
      data[base + 4] = 0;
      data[base + 5] = 0;
      data[base + 6] = 0;
      data[base + 7] = l.intensity * sunIntensityMul;
      data[base + 8] = dir?.x ?? 0;
      data[base + 9] = dir?.y ?? -1;
      data[base + 10] = dir?.z ?? 0;
      data[base + 11] = 0;
      // Sun chroma: from the emitter when present; else the legacy warm-white
      // (1,0.95,0.85) the packer hardcoded before scene-directional wiring.
      data[base + 12] = col?.r ?? 1;
      data[base + 13] = col?.g ?? 0.95;
      data[base + 14] = col?.b ?? 0.85;
      data[base + 15] = 0;
    } else if (l.kind === 'fixture' || l.kind === 'teaLight') {
      udata[ubase] = 1;
      const pos = l.position;
      const col = l.color;
      data[base + 4] = pos?.x ?? 0;
      data[base + 5] = pos?.y ?? 0;
      data[base + 6] = pos?.z ?? 0;
      data[base + 7] = l.intensity;
      data[base + 8] = 0;
      data[base + 9] = 0;
      data[base + 10] = 0;
      data[base + 11] = 0;
      data[base + 12] = col?.r ?? 1;
      data[base + 13] = col?.g ?? 1;
      data[base + 14] = col?.b ?? 1;
      data[base + 15] = 0;
    }
  });
  return data.buffer;
}
