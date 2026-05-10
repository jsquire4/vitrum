import { describe, expect, it } from 'vitest';
import { OUTDOOR_HDRI_PRESETS, findHdriPresetById } from '../src/hdriPresets.js';

describe('hdriPresets', () => {
  it('exports four outdoor presets with dl.polyhaven.org HDR URLs', () => {
    expect(OUTDOOR_HDRI_PRESETS.length).toBe(4);
    for (const p of OUTDOOR_HDRI_PRESETS) {
      expect(p.id).toMatch(/^[a-z0-9_]+$/);
      expect(p.url).toMatch(/^https:\/\/dl\.polyhaven\.org\/file\/ph-assets\/HDRIs\/hdr\/1k\/.+\.hdr$/);
    }
  });

  it('findHdriPresetById resolves known ids', () => {
    expect(findHdriPresetById('venice_sunset')?.label).toBe('Venice Sunset');
    expect(findHdriPresetById('missing')).toBeUndefined();
  });
});
