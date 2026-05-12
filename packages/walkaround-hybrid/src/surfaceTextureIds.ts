/**
 * Canonical SURFACE_TEXTURE_ID enum — the wire contract between scene
 * bindings (THREE.Material.userData.surfaceTextureId), BVH packing
 * (`packBVHIndexW` writes the low 3 bits of bvhIndex[*].w), and the
 * WGSL `surfaceTextureMod` switch in
 * `./shaders/surfaceTextures.wgsl.ts`.
 *
 * Single source of truth, owned by the consumer (this package): the
 * integer values are baked into shader switch statements and BVH
 * uploads, so renumbering existing entries silently breaks rendering
 * for every host that ever stamped a face with the old code.
 *
 * Hosts that author scenes (e.g. the stainedGlass app's physics
 * package) should re-export this table rather than maintaining their
 * own copy. Add new textures at the next unused integer; never
 * renumber existing entries.
 *
 * Wire contract:
 *   - smooth     = 0   (flat, no modulation)
 *   - hammered   = 1
 *   - ripple     = 2
 *   - granite    = 3
 *   - baroque    = 4
 *   - waterglass = 5
 *   - catspaw    = 6
 *   - flemish    = 7
 *
 * The BVH packing reserves 3 bits (values 0..7), so adding entry 8+
 * also requires widening `texTypeId` in packingHelpers.ts and the
 * WGSL `decodeSurfaceTextureId` mask.
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
