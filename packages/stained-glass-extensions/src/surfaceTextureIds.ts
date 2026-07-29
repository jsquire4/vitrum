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

const VALID_SURFACE_TEXTURE_IDS: ReadonlySet<number> =
  new Set(Object.values(SURFACE_TEXTURE_ID));

/** Runtime guard for the exact wire-level surface-texture id set. */
export function isSurfaceTextureId(value: unknown): value is SurfaceTextureId {
  return typeof value === 'number'
    && Number.isInteger(value)
    && VALID_SURFACE_TEXTURE_IDS.has(value);
}

/**
 * Validate an untyped value before it enters the three-bit renderer lane.
 *
 * This deliberately rejects out-of-range integers instead of masking them:
 * silently mapping `8 → smooth` or `15 → flemish` would render a different
 * authored texture while leaving no diagnostic at the ingestion boundary.
 */
export function validateSurfaceTextureId(
  value: unknown,
  label = 'surfaceTextureId',
): SurfaceTextureId {
  if (typeof value !== 'number' || !Number.isInteger(value)) {
    throw new TypeError(
      `${label} must be an integer surface-texture id; received ${String(value)}.`,
    );
  }
  if (!VALID_SURFACE_TEXTURE_IDS.has(value)) {
    throw new RangeError(
      `${label} must be one of ${[...VALID_SURFACE_TEXTURE_IDS].join(', ')}; received ${value}.`,
    );
  }
  return value as SurfaceTextureId;
}
