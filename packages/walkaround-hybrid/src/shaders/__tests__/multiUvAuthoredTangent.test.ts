import { describe, expect, it } from 'vitest';

import { makeProbeUpdateRaysWGSL } from '../../ddgi/wgsl/probeUpdateRays.wgsl.js';
import { MATERIAL_ATLAS_WGSL } from '../materialAtlas.wgsl.js';

function wgslFunctionBody(source: string, name: string): string {
  const signature = source.indexOf(`fn ${name}(`);
  expect(signature, `${name} signature`).toBeGreaterThanOrEqual(0);
  const open = source.indexOf('{', signature);
  expect(open, `${name} opening brace`).toBeGreaterThan(signature);
  let depth = 0;
  for (let i = open; i < source.length; i += 1) {
    if (source[i] === '{') depth += 1;
    if (source[i] === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(open + 1, i);
    }
  }
  throw new Error(`unterminated WGSL function ${name}`);
}

describe('authored tangent UV-lane contract', () => {
  it.each([
    [
      'main material path',
      MATERIAL_ATLAS_WGSL,
      'materialTangentFrameForHit',
      'preferAuthoredTangentFrameForHit',
      'MaterialTangentFrame',
    ],
    [
      'DDGI probe path',
      makeProbeUpdateRaysWGSL(256),
      'ddgiMaterialTangentFrameForHit',
      'ddgiPreferAuthoredTangentFrameForHit',
      'DdgiMaterialTangentFrame',
    ],
  ] as const)(
    '%s keeps selected nonzero UV lanes derivative-derived and uses authored tangents only for UV0',
    (_label, source, frameFunction, authoredFunction, frameType) => {
      const body = wgslFunctionBody(source, frameFunction);

      // The map metadata chooses a real lane and every triangle vertex resolves
      // that same lane before the derivative frame is constructed.
      expect(body).toContain('let texCoord = (flags >> 4u) & 0xFu;');
      expect(body.match(/materialResolveUv\(triIndex, texCoord,/g)).toHaveLength(3);

      // glTF TANGENT describes TEXCOORD_0. UV1+ must retain the frame above
      // instead of being overwritten by the authored UV0 frame.
      expect(body).toMatch(
        new RegExp(
          `if \\(texCoord == 0u\\) \\{\\s*return ${authoredFunction}` +
            `\\(hit, frameNormal, tangent, bitangent\\);\\s*\\}`,
        ),
      );
      expect(body).toContain(`return ${frameType}(tangent, bitangent);`);
    },
  );
});
