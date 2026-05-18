/**
 * H-channel came — the stained-glass-studio analytic-rail primitive.
 *
 * This is a host-app-specific extension of the @vitrum/core AnalyticShape
 * union. Core defines AnalyticShape as an open-ended string union so packages
 * like this one can register additional discriminator tags without modifying
 * the core contract.
 *
 * Parameter layout (Float32Array, length 4):
 *   [length, railWidth, blockHeight, webThickness]
 *
 * Backends that opt in to supporting H-channel came do so via
 * `EngineCapabilities.supportedAnalyticShapes`; consumers detect the
 * primitive by calling `isHChannelCame(shape)`.
 */

/**
 * The canonical discriminator string for the H-channel came analytic shape.
 * Use this constant instead of the bare string literal in producers so the
 * spelling stays consistent across the codebase.
 */
export const STAINED_GLASS_H_CHANNEL_CAME = 'h-channel-came' as const;

/**
 * String literal type for the H-channel came analytic shape discriminator.
 */
export type StainedGlassHChannelCameTag = typeof STAINED_GLASS_H_CHANNEL_CAME;

/**
 * Returns true when the AnalyticPrimitive.shape discriminator names the
 * stained-glass H-channel came shape. Backends use this to gate
 * came-specific intersection / packing code.
 */
export function isHChannelCame(shape: string): shape is StainedGlassHChannelCameTag {
  return shape === STAINED_GLASS_H_CHANNEL_CAME;
}

/**
 * Field offsets into the AnalyticPrimitive.params Float32Array for the
 * H-channel came primitive. Exposed so backends pack / unpack against a
 * single canonical layout.
 */
export const STAINED_GLASS_H_CHANNEL_CAME_PARAMS = {
  LENGTH: 0,
  RAIL_WIDTH: 1,
  BLOCK_HEIGHT: 2,
  WEB_THICKNESS: 3,
  PARAM_COUNT: 4,
} as const;
