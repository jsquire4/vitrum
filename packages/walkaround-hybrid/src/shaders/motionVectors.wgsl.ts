import type { WgslModule } from '../pipeline/wgslComposer.js';

export const MOTION_VECTORS_WGSL = /* wgsl */ `
@group(0) @binding(0) var gNormalDepthIn: texture_2d<f32>;
@group(0) @binding(1) var motionVectorsOut: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> ubo: WalkaroundUBO;

@compute @workgroup_size(8, 8, 1)
fn motionVectorsMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = ubo.screenSize;
  if (any(gid.xy >= dims)) { return; }

  let nd = textureLoad(gNormalDepthIn, vec2i(gid.xy), 0);
  let depthSigned = nd.w;
  let depth = abs(depthSigned);

  // Sky sentinel from shade pass.
  if (depth <= 1e-6) {
    textureStore(motionVectorsOut, gid.xy, vec4f(0.0, 0.0, 0.0, 0.0));
    return;
  }

  let vp = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);
  let ray = generatePrimaryRay_common(
    gid.x, gid.y, dims.x, dims.y,
    ubo.cameraPos,
    invVP,
  );
  let worldPos = ray.origin + ray.direction * depth;

  let currClip = ubo.projMatrix * ubo.viewMatrix * vec4f(worldPos, 1.0);
  let prevClip = ubo.prevViewProjMatrix * vec4f(worldPos, 1.0);
  if (abs(currClip.w) <= 1e-6 || abs(prevClip.w) <= 1e-6) {
    textureStore(motionVectorsOut, gid.xy, vec4f(0.0, 0.0, 0.0, 0.0));
    return;
  }

  let currNdc = currClip.xy / currClip.w;
  let prevNdc = prevClip.xy / prevClip.w;
  // Reprojection delta expected by SVGF: previous - current.
  let motion = prevNdc - currNdc;
  textureStore(motionVectorsOut, gid.xy, vec4f(motion, 0.0, 0.0));
}
`;

/**
 * T9-stepC — `requires` narrowed from the full `common` aggregate to the
 * minimal subset this pass actually references (verified by the static
 * cross-module ident-resolution gate in __tests__/wgslCompose.test.ts):
 *   - `WalkaroundUBO`               → walkaroundUbo
 *   - `invertMat4_common` / `generatePrimaryRay_common`
 *                                   → cameraRays (which uses `Ray` +
 *                                     `safe_normalize`, hence sceneTraversal +
 *                                     sharedPrimitives transitively; and
 *                                     sharedPrimitives uses PI/INV_PI from
 *                                     walkaroundUbo).
 * The eight BVH/reservoir/BRDF/emitter/welford modules common otherwise
 * pulls are unreferenced by motion-vector reprojection, so they are dropped.
 */
export const MOTION_VECTORS_MODULE: WgslModule = {
  name: 'motionVectors',
  source: MOTION_VECTORS_WGSL,
  requires: ['walkaroundUbo', 'sceneTraversal', 'sharedPrimitives', 'cameraRays'],
};
