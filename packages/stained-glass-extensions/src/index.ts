/**
 * @vitrum/stained-glass-extensions — opt-in host-app extensions for the
 * stained-glass-studio renderer. Not part of the generic vitrum library;
 * provided as a reference implementation of the EngineOptions.extensions /
 * Material.extensions / AnalyticShape extension points.
 */

// D2 — H-channel came analytic shape (stained-glass rail primitive).
export {
  STAINED_GLASS_H_CHANNEL_CAME,
  STAINED_GLASS_H_CHANNEL_CAME_PARAMS,
  isHChannelCame,
  type StainedGlassHChannelCameTag,
} from './h-channel-came.js';

// D3 — Dichroic LUT extension converter (THREE ↔ vitrum round trip).
export {
  DICHROIC_LUTS_EXTENSION_ID,
  STAINED_GLASS_USER_DATA_KEYS,
  dichroicLUTsExtensionConverter,
  type DichroicLUTsExtension,
} from './dichroic-luts.js';
