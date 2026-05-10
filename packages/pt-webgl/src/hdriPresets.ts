import type { DataTexture, LoadingManager } from 'three';
import { EquirectangularReflectionMapping } from 'three';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';

export interface HdriPreset {
  readonly id: string;
  readonly label: string;
  /** HTTPS URL to a radiance `.hdr` (Poly Haven CDN, 1k). */
  readonly url: string;
}

/**
 * Outdoor / studio HDRIs for IBL — Sprint 1 (Phase 6).
 * URLs use Poly Haven `dl.polyhaven.org` (verified 200, `image/vnd.radiance`).
 */
export const OUTDOOR_HDRI_PRESETS: readonly HdriPreset[] = [
  {
    id: 'venice_sunset',
    label: 'Venice Sunset',
    url: 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/venice_sunset_1k.hdr',
  },
  {
    id: 'rogland_overcast',
    label: 'Rogland Overcast',
    url: 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/rogland_overcast_1k.hdr',
  },
  {
    id: 'kiara_1_dawn',
    label: 'Kiara 1 Dawn',
    url: 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/kiara_1_dawn_1k.hdr',
  },
  {
    id: 'music_hall_01',
    label: 'Music Hall 01',
    url: 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/music_hall_01_1k.hdr',
  },
] as const;

export function findHdriPresetById(id: string): HdriPreset | undefined {
  return OUTDOOR_HDRI_PRESETS.find((p) => p.id === id);
}

/** Loads an equirectangular HDR and sets reflection mapping for `scene.environment` / backgrounds. */
export async function loadHdriEquirect(
  url: string,
  opts?: { readonly manager?: LoadingManager },
): Promise<DataTexture> {
  const loader = new RGBELoader(opts?.manager);
  const tex = await loader.loadAsync(url);
  tex.mapping = EquirectangularReflectionMapping;
  return tex;
}
