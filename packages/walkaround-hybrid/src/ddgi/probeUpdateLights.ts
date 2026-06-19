/**
 * DDGI probe-update light UBO packing (up to 16 lights × 64 B).
 */
import type { EngineWarning } from '@vitrum/core';
import type { DDGILight } from './types.js';

const MAX_DDGI_PROBE_LIGHTS = 16;
const LIGHT_STRIDE_FLOATS = 16;
export const DDGI_LIGHT_KIND_SUN = 0;
export const DDGI_LIGHT_KIND_POINT = 1;
export const DDGI_LIGHT_KIND_MASK = 0x7fffffff;
export const DDGI_LIGHT_CAST_SHADOW_DISABLED = 0x80000000;
export const DDGI_PROBE_LIGHTS_BUFFER_BYTES =
  (4 + MAX_DDGI_PROBE_LIGHTS * LIGHT_STRIDE_FLOATS) * Float32Array.BYTES_PER_ELEMENT;

export type DDGIProbeLightWarningSink = (warning: EngineWarning) => void;

function isPackableDDGILight(l: DDGILight): boolean {
  return l.kind === 'sun' || l.kind === 'fixture' || l.kind === 'teaLight';
}

export function packDDGIProbeLights(
  lights: readonly DDGILight[],
  sunIntensityMul: number,
  onWarning?: DDGIProbeLightWarningSink,
): ArrayBuffer {
  const headerSize = 4;
  const data = new Float32Array(headerSize + MAX_DDGI_PROBE_LIGHTS * LIGHT_STRIDE_FLOATS);
  const udata = new Uint32Array(data.buffer);
  const active = lights.filter((l) => l.on);
  const unsupportedKinds = [...new Set(active
    .filter((l) => !isPackableDDGILight(l))
    .map((l) => l.kind))];
  if (unsupportedKinds.length > 0) {
    emitProbeLightWarning(onWarning, {
      code: 'walkaround-hybrid.ddgi-unsupported-probe-light-kind',
      backend: 'walkaround-hybrid',
      phase: 'renderFrame',
      method: 'ProbeUpdatePass._uploadLights',
      message:
        `[DDGI] packDDGIProbeLights: unsupported DDGI light kind(s) ${unsupportedKinds.join(', ')} ` +
        'were ignored for probe-update GI.',
      details: { unsupportedKinds, activeLightCount: active.length },
    });
  }
  const packable = active.filter(isPackableDDGILight);
  // H18 Stage 1 — warn on truncation so hosts know lights beyond the cap are dropped.
  if (packable.length > MAX_DDGI_PROBE_LIGHTS) {
    emitProbeLightWarning(onWarning, {
      code: 'walkaround-hybrid.ddgi-probe-light-cap-exceeded',
      backend: 'walkaround-hybrid',
      phase: 'renderFrame',
      method: 'ProbeUpdatePass._uploadLights',
      message:
        `[DDGI] packDDGIProbeLights: scene has ${packable.length} active packable lights but the DDGI probe ` +
        `shader supports at most ${MAX_DDGI_PROBE_LIGHTS}. Lights beyond this cap are ignored ` +
        `for probe-update GI. Reduce your light count or raise MAX_DDGI_PROBE_LIGHTS.`,
      details: {
        activePackableLightCount: packable.length,
        maxProbeLights: MAX_DDGI_PROBE_LIGHTS,
        ignoredLightCount: packable.length - MAX_DDGI_PROBE_LIGHTS,
      },
    });
  }
  udata[0] = Math.min(packable.length, MAX_DDGI_PROBE_LIGHTS);

  packable.slice(0, MAX_DDGI_PROBE_LIGHTS).forEach((l, i) => {
    const base = headerSize + i * LIGHT_STRIDE_FLOATS;
    const ubase = base;
    const shadowFlag = l.castShadow === false ? DDGI_LIGHT_CAST_SHADOW_DISABLED : 0;
    if (l.kind === 'sun') {
      // Sun travel direction. When a `@vitrum/core` `directional` emitter
      // drives the sun, `coreEmitterToDDGILight` carries its real direction
      // here (already negated to a travel direction). The WGSL shader negates
      // again (`normalize(-light.direction)`) to recover the toward-light dir.
      // Absent direction → legacy hardcoded straight-down sun (0,-1,0) so a
      // host-supplied sun light with no direction is unchanged.
      const dir = l.direction;
      const col = l.color;
      udata[ubase] = (DDGI_LIGHT_KIND_SUN | shadowFlag) >>> 0;
      data[base + 4] = 0;
      data[base + 5] = 0;
      data[base + 6] = 0;
      data[base + 7] = l.intensity * sunIntensityMul;
      data[base + 8] = dir?.x ?? 0;
      data[base + 9] = dir?.y ?? -1;
      data[base + 10] = dir?.z ?? 0;
      data[base + 11] = typeof l.angularRadius === 'number' && Number.isFinite(l.angularRadius)
        ? Math.max(0, l.angularRadius)
        : 0;
      // Sun chroma: from the emitter when present; else the legacy warm-white
      // (1,0.95,0.85) the packer hardcoded before scene-directional wiring.
      data[base + 12] = col?.r ?? 1;
      data[base + 13] = col?.g ?? 0.95;
      data[base + 14] = col?.b ?? 0.85;
      data[base + 15] = 0;
    } else if (l.kind === 'fixture' || l.kind === 'teaLight') {
      udata[ubase] = (DDGI_LIGHT_KIND_POINT | shadowFlag) >>> 0;
      const pos = l.position;
      const col = l.color;
      // [1,2] = distance/decay, [8,9,10] = spot cone axis
      // (forward beam/travel; 0 for a point → no cone in the shader),
      // [11] = innerCone (cosInner), [15] = outerCone (cosOuter). These map
      // to the DDGILight WGSL struct's distance / decay / direction /
      // innerCone / outerCone.
      const spot = l.spotAxis;
      data[base + 1] = typeof l.distance === 'number' && l.distance > 0 ? l.distance : 0;
      data[base + 2] = typeof l.decay === 'number' ? l.decay : 2;
      data[base + 4] = pos?.x ?? 0;
      data[base + 5] = pos?.y ?? 0;
      data[base + 6] = pos?.z ?? 0;
      data[base + 7] = l.intensity;
      data[base + 8] = spot?.x ?? 0;
      data[base + 9] = spot?.y ?? 0;
      data[base + 10] = spot?.z ?? 0;
      data[base + 11] = l.spotCosInner ?? 0;
      data[base + 12] = col?.r ?? 1;
      data[base + 13] = col?.g ?? 1;
      data[base + 14] = col?.b ?? 1;
      data[base + 15] = l.spotCosOuter ?? 0;
    }
  });
  return data.buffer;
}

function emitProbeLightWarning(
  onWarning: DDGIProbeLightWarningSink | undefined,
  warning: EngineWarning,
): void {
  if (onWarning) {
    onWarning(warning);
    return;
  }
  console.warn(warning.message);
}
