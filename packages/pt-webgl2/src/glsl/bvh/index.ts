// BVH traversal GLSL — the three-mesh-bvh `BVHShaderGLSL` port (plan/three-removal/
// 04-glsl-kernels.md §2). The four `.glsl.js` chunks are copied verbatim from
// three-mesh-bvh (MIT, (c) Garrett Johnson) so @vitrum/pt-webgl2 carries no
// three-mesh-bvh runtime dependency. They read ONLY their own `BVH` struct samplers
// (no THREE). Re-exported here under the UPPERCASE names the composer wires.
//
// The `.glsl.js` modules ship no type declarations (typed by glsl-modules.d.ts as a
// default-only wildcard). TS wildcard modules cannot declare named exports, so we pull
// the named string member via a namespace import cast; `pick()` resolves the
// `string | undefined` that `noUncheckedIndexedAccess` produces and fails loudly if a
// chunk's export name ever drifts.

import * as CommonNS from './common_functions.glsl.js';
import * as StructNS from './bvh_struct_definitions.glsl.js';
import * as RayNS from './bvh_ray_functions.glsl.js';
function pick(ns: unknown, name: string): string {
  const value = (ns as Record<string, unknown>)[name];
  if (typeof value !== 'string') {
    throw new Error(`pt-webgl2 bvh glsl: missing string export "${name}"`);
  }
  return value;
}

export const BVH_COMMON_FUNCTIONS: string = pick(CommonNS, 'common_functions');
export const BVH_STRUCT: string = pick(StructNS, 'bvh_struct_definitions');
export const BVH_RAY_FUNCTIONS: string = pick(RayNS, 'bvh_ray_functions');
