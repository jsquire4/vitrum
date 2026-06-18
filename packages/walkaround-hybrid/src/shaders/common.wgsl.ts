/**
 * Common WGSL aggregate shared across all ReSTIR compute passes.
 *
 * T9-stepA (split the `common.wgsl.ts` dumping-ground): the ~695-line single
 * template string that previously lived here was split into eleven focused
 * sibling modules under `shaders/`. This file is now a THIN AGGREGATE:
 *
 *   - `COMMON_WGSL` is the in-order concatenation of the eleven module
 *     strings. At the W1 split it stayed byte-identical to the pre-split
 *     string; later canonical-helper landings may intentionally change the
 *     bytes while preserving the same dependency order.
 *   - `COMMON_MODULE` no longer carries its own source; instead it `requires`
 *     the eleven focused modules in the original source order. `composeWgsl`
 *     emits each dependency exactly once, in declared order, then appends
 *     COMMON_MODULE's (empty) source — so `composeWgsl(COMMON_MODULE)` is also
 *     byte-identical to the old `COMMON_WGSL`.
 *
 * Why the focused modules declare `requires: []` rather than honest inter-
 * sibling deps: several modules forward-reference symbols that the original
 * source declared LATER (e.g. the reservoir update helpers call `rand_f32`,
 * which the shared-primitives module defines after them). WGSL resolves
 * module-scope functions regardless of declaration order, so the original
 * order compiles. Declaring honest `requires` would make the topo-sort hoist
 * those later modules earlier and CHANGE the byte order — breaking the ordering
 * guarantee. `common` therefore owns the canonical ordering by
 * listing the modules explicitly, and the focused modules are leaf entries.
 *
 * The focused modules (in aggregate order):
 *   walkaroundUbo · sceneTraversal · reservoirDi · reservoirGi ·
 *   sharedPrimitives · ggxBrdf · materialDecode · emitterSampling ·
 *   jacobianShift · cameraRays · welfordTail
 *
 * References:
 *   - three-mesh-bvh/src/webgpu/common_functions.wgsl.js — BVHNode struct
 *   - C-none/Web-RTRT reservoir.wgsl — encode/decode helpers
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { WALKAROUND_UBO_WGSL, WALKAROUND_UBO_MODULE } from './walkaroundUbo.wgsl.js';
import { SCENE_TRAVERSAL_WGSL, SCENE_TRAVERSAL_MODULE } from './sceneTraversal.wgsl.js';
import { RESERVOIR_DI_WGSL, RESERVOIR_DI_MODULE } from './reservoirDi.wgsl.js';
import { RESERVOIR_GI_WGSL, RESERVOIR_GI_MODULE } from './reservoirGi.wgsl.js';
import { SHARED_PRIMITIVES_WGSL, SHARED_PRIMITIVES_MODULE } from './sharedPrimitives.wgsl.js';
import { GGX_BRDF_WGSL, GGX_BRDF_MODULE } from './ggxBrdf.wgsl.js';
import { MATERIAL_DECODE_WGSL, MATERIAL_DECODE_MODULE } from './materialDecode.wgsl.js';
import { EMITTER_SAMPLING_WGSL, EMITTER_SAMPLING_MODULE } from './emitterSampling.wgsl.js';
import { JACOBIAN_SHIFT_WGSL, JACOBIAN_SHIFT_MODULE } from './jacobianShift.wgsl.js';
import { CAMERA_RAYS_WGSL, CAMERA_RAYS_MODULE } from './cameraRays.wgsl.js';
import { WELFORD_TAIL_WGSL, WELFORD_TAIL_MODULE } from './welfordTail.wgsl.js';

/**
 * The canonical ordering of the focused modules. `COMMON_WGSL` and
 * `COMMON_MODULE.requires` both derive from this single list so they can
 * never drift.
 */
const COMMON_MODULE_ORDER: readonly WgslModule[] = [
  WALKAROUND_UBO_MODULE,
  SCENE_TRAVERSAL_MODULE,
  RESERVOIR_DI_MODULE,
  RESERVOIR_GI_MODULE,
  SHARED_PRIMITIVES_MODULE,
  GGX_BRDF_MODULE,
  MATERIAL_DECODE_MODULE,
  EMITTER_SAMPLING_MODULE,
  JACOBIAN_SHIFT_MODULE,
  CAMERA_RAYS_MODULE,
  WELFORD_TAIL_MODULE,
];

/**
 * Built by concatenating the focused module sources in `COMMON_MODULE_ORDER`.
 * Kept as an explicit `+` chain (rather than `.map().join('')`) so the order
 * is auditable inline and matches the original single-string layout.
 */
export const COMMON_WGSL = /* wgsl */
  WALKAROUND_UBO_WGSL +
  SCENE_TRAVERSAL_WGSL +
  RESERVOIR_DI_WGSL +
  RESERVOIR_GI_WGSL +
  SHARED_PRIMITIVES_WGSL +
  GGX_BRDF_WGSL +
  MATERIAL_DECODE_WGSL +
  EMITTER_SAMPLING_WGSL +
  JACOBIAN_SHIFT_WGSL +
  CAMERA_RAYS_WGSL +
  WELFORD_TAIL_WGSL;

/** W1-R6 — declarative include-graph entry. Common is the root of the
 *  dependency tree; everything else opts in via `requires: ['common']`.
 *
 *  T9-stepA: `common` is now a thin aggregate. Its source is empty and it
 *  `requires` the eleven focused modules in `COMMON_MODULE_ORDER`. The
 *  composer emits the deps (each once, in this order) then appends the empty
 *  root source, preserving the aggregate dependency order. */
export const COMMON_MODULE: WgslModule = {
  name: 'common',
  source: '',
  requires: COMMON_MODULE_ORDER.map((m) => m.name),
};
