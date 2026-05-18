/**
 * W7-H6 — BACK-COMPAT RE-EXPORT FAÇADE.
 *
 * The former monolithic `surfaceTextures.wgsl.ts` conflated two concerns:
 *   1. Library-general — `bvhTraceTintedVisibility` (per-channel BVH shadow
 *      walker, lives now in `./glassVisibility.wgsl.ts`).
 *   2. Host-app-specific — eight stained-glass procedural patterns +
 *      `surfaceTextureMod` switch (now in `./stained-glass/surfaceMods.wgsl.ts`).
 *
 * This file is kept solely to preserve the historical
 * `SURFACE_TEXTURES_WGSL` and `SURFACE_TEXTURES_MODULE` export names so
 * any external importer (and the bit-identical wgslCompose snapshot
 * tests) keeps working untouched. The string is recomposed from the two
 * split sources so the bytes are identical to the pre-split version.
 *
 * NEW CODE should import from the split files directly:
 *   - `./stained-glass/surfaceMods.wgsl.ts` for the host pattern catalogue
 *   - `./glassVisibility.wgsl.ts` for the library shadow walker
 *
 * @deprecated Use the split modules (`STAINED_GLASS_SURFACE_MODS_MODULE`,
 *             `GLASS_VISIBILITY_MODULE`) directly. The eventual home of the
 *             surfaceMods file is the `@vitrum/stained-glass-extensions`
 *             package (gated on W3-D2+D3 landing).
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import {
  STAINED_GLASS_SURFACE_MODS_WGSL,
  STAINED_GLASS_SURFACE_MODS_MODULE,
} from './stained-glass/surfaceMods.wgsl.js';
import {
  GLASS_VISIBILITY_WGSL,
  GLASS_VISIBILITY_MODULE,
} from './glassVisibility.wgsl.js';

// Re-export the split modules so consumers can opt into the new names
// without changing their import path.
export {
  STAINED_GLASS_SURFACE_MODS_WGSL,
  STAINED_GLASS_SURFACE_MODS_MODULE,
  GLASS_VISIBILITY_WGSL,
  GLASS_VISIBILITY_MODULE,
};

/**
 * @deprecated Use `STAINED_GLASS_SURFACE_MODS_WGSL + GLASS_VISIBILITY_WGSL`
 * (or, better, the include-graph modules) directly. Retained as the
 * exact concatenation of the two split sources so the pre-R6
 * bit-identical wgslCompose snapshot still matches.
 */
export const SURFACE_TEXTURES_WGSL: string =
  STAINED_GLASS_SURFACE_MODS_WGSL + GLASS_VISIBILITY_WGSL;

/**
 * @deprecated W7-H6 — kept only so existing consumers (notably the
 * `WGSL_MODULES` registry and the wgslCompose tests) continue to resolve
 * a single `surfaceTextures` name. The composed source equals the
 * concatenation of the split modules, mirroring the pre-W7-H6 string.
 * Prefer `STAINED_GLASS_SURFACE_MODS_MODULE` + `GLASS_VISIBILITY_MODULE`
 * in any new graph wiring.
 */
export const SURFACE_TEXTURES_MODULE: WgslModule = {
  name: 'surfaceTextures',
  source: SURFACE_TEXTURES_WGSL,
  requires: ['common'],
};
