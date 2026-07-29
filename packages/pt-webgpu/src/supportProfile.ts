/**
 * Side-effect-free support-profile surface for planners and scene adapters.
 *
 * Keep consumers on these backend-owned executable manifests instead of
 * copying profile deltas into a separate compatibility table.
 */
export {
  PT_WEBGPU_FULL_SUPPORT_MANIFEST,
  PT_WEBGPU_LITE_SUPPORT_MANIFEST,
} from './supportManifest.js';
export {
  PT_WEBGPU_LITE_EXTRA_UNSUPPORTED_MATERIAL_FIELDS,
} from './supportDetails.js';
