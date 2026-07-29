import { describe, expect, it } from 'vitest';
import * as ThinFilmModule from './shader/bsdf/thin_film_tmm.glsl.js';
import * as BsdfModule from './shader/bsdf/bsdf_functions.glsl.js';

const thin_film_tmm = (
  ThinFilmModule as unknown as Record<string, string>
)['thin_film_tmm']!;
const bsdf_functions = (
  BsdfModule as unknown as Record<string, string>
)['bsdf_functions']!;

describe('default-RGB thin-film transport', () => {
  it('evaluates channel-local visible wavelengths outside spectral mode', () => {
    const compact = thin_film_tmm.replace(/\s+/g, ' ');
    expect(compact).toContain('struct ThinFilmRgb');
    expect(compact).toContain('if ( uSpectralRendering != 0 )');
    expect(compact).toContain('thinFilmLayerCount, 650.0,');
    expect(compact).toContain('thinFilmLayerCount, 510.0,');
    expect(compact).toContain('thinFilmLayerCount, 475.0,');
    expect(compact).toContain(
      'result.reflectance = vec3( red.x, green.x, blue.x )',
    );
    expect(compact).toContain(
      'result.transmittance = vec3( red.y, green.y, blue.y )',
    );
  });

  it('routes every reflection and transmission call site through RGB-aware transport', () => {
    expect(bsdf_functions).not.toContain('vec2 thinFilmRt = thinFilmTMM(');
    expect(bsdf_functions.match(/thinFilmTMMRgb\(/g)).toHaveLength(4);
    expect(bsdf_functions).toContain('thinFilmRt.reflectance');
    expect(bsdf_functions).toContain('thinFilmRt.transmittance');
  });
});
