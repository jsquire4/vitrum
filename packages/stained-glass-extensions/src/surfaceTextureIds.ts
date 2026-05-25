/**
 * Canonical SURFACE_TEXTURE_ID enum for stained-glass host workflows.
 *
 * Integer values are wire-level IDs consumed by BVH packing and shader switches.
 * Add new entries at the next unused integer; never renumber existing entries.
 */
export const SURFACE_TEXTURE_ID = {
  smooth: 0,
  hammered: 1,
  ripple: 2,
  granite: 3,
  baroque: 4,
  waterglass: 5,
  catspaw: 6,
  flemish: 7,
} as const;

export type SurfaceTextureName = keyof typeof SURFACE_TEXTURE_ID;
export type SurfaceTextureId = (typeof SURFACE_TEXTURE_ID)[SurfaceTextureName];
