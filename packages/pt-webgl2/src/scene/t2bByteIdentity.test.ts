// T2-B byte-identity golden — pins the FULL packed Float32Array output of the
// materials packer and the lights packer for representative fixtures, so the
// data-driven refactors of packLayerIds / packTextureTransforms (materialsTexture)
// and the writePositionType / writeColorIntensity helpers (lightsTexture) stay
// byte-for-byte identical (the packed texels are the wire format the GLSL decoder
// reads). The golden arrays are captured from pre-refactor code; any drift fails.
//
// To (re)capture: run with T2B_CAPTURE=1, inspect /tmp/t2bGoldens.json, then
// replace the pinned JSON intentionally. Under behavior-preserving refactors
// the goldens are frozen.

import { writeFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { packMaterialsTexture } from './materialsTexture.js';
import { packLightsTexture } from './lightsTexture.js';
import { emitters, layerOf, plainMaterial, richMaterial } from './t2bFixtures.js';
import goldens from './t2bGoldens.json' assert { type: 'json' };

// Serialize to a JSON-safe form: signed zero → +0, and non-finite floats →
// sentinel strings (JSON cannot represent -0 / Infinity / NaN). The GLSL decoder
// treats these texels numerically; the sentinels keep the golden lossless.
function serialize(a: Float32Array | Uint32Array): (number | string)[] {
  return Array.from(a as Float32Array, (v) => {
    if (v === 0) return 0;
    if (v === Infinity) return 'Inf';
    if (v === -Infinity) return '-Inf';
    if (Number.isNaN(v)) return 'NaN';
    return v;
  });
}

const materials = packMaterialsTexture([richMaterial, plainMaterial], layerOf).data;
const lights = packLightsTexture(emitters).data;

if (process.env['T2B_CAPTURE'] === '1') {
  writeFileSync(
    '/tmp/t2bGoldens.json',
    JSON.stringify({ materials: serialize(materials), lights: serialize(lights) }),
  );
  console.log('T2B_GOLDENS_JSON=/tmp/t2bGoldens.json');
}

describe('T2-B byte-identity goldens', () => {
  it('materials packer output is byte-identical to the pinned golden', () => {
    expect(serialize(materials)).toEqual(goldens.materials);
  });

  it('lights packer output is byte-identical to the pinned golden', () => {
    expect(serialize(lights)).toEqual(goldens.lights);
  });
});
