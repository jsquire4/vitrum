/**
 * FrameParams UBO struct + the 24 storage/uniform/texture binding declarations
 * for the brute-force path tracer's @group(0).
 *
 * This is the load-bearing host-GPU contract: any change here must be
 * mirrored in {@link buildFrameParamsBuffer} on the TypeScript side. Tests in
 * `frameParamsLayout.test.ts` verify the UBO layout, scalar offsets, and the
 * binding-index assignments.
 *
 * Also exports the material/spectral constants that every WGSL module touches
 * (LEAFNODE_FLAG, MATERIAL_VEC4_STRIDE, etc.) — they live with the bindings
 * because every consumer that indexes into `materials` or `bvhNodes` needs them.
 */
export const PT_WEBGPU_FRAME_PARAMS_WGSL = /* wgsl */ `
struct FrameParams {
  width: u32,
  height: u32,
  frameIndex: u32,
  frameSeed: u32,
  triangleCount: u32,
  maxBounces: u32,
  bvhNodeCount: u32,
  analyticCount: u32,
  pointLightCount: u32,
  spotLightCount: u32,
  rectAreaLightCount: u32,
  meshAreaLightCount: u32,
  mneeMaxIterations: u32,
  mneeMaxChainLength: u32,
  hasEnvironmentMap: u32,
  causticStrategy: u32,
  environmentMapWidth: u32,
  environmentMapHeight: u32,
  triIntersectEpsilon: f32, // UBO-plumbed (D12); default 1e-5 (metre-scale)
  _pad1: u32,
  cameraPos: vec4f,
  lightDir: vec4f,
  environmentTint: vec4f,
  environmentSun: vec4f,
  invViewProj: mat4x4f,
  viewProj: mat4x4f,
  prevViewProj: mat4x4f,
};

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform> params: FrameParams;
@group(0) @binding(2) var<storage, read_write> accumBuffer: array<vec4f>;
@group(0) @binding(3) var<storage, read> positions: array<vec4f>;
@group(0) @binding(4) var<storage, read> indices: array<vec4u>;
@group(0) @binding(5) var<storage, read> triMaterialIds: array<u32>;
@group(0) @binding(6) var<storage, read> materials: array<vec4f>;
@group(0) @binding(7) var<storage, read> bvhNodes: array<BVHNode>;
@group(0) @binding(8) var<storage, read> normals: array<vec4f>;
@group(0) @binding(9) var normalDepthTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var albedoTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var varianceTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(12) var motionVectorsTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(13) var<storage, read_write> varianceMomentsBuffer: array<vec4f>;
@group(0) @binding(14) var<storage, read> analyticHeaders: array<vec4f>;
@group(0) @binding(15) var<storage, read> analyticParams: array<vec4f>;
@group(0) @binding(16) var<storage, read> analyticLocalToWorld: array<vec4f>;
@group(0) @binding(17) var<storage, read> analyticWorldToLocal: array<vec4f>;
@group(0) @binding(18) var<storage, read> environmentMapTexels: array<vec4f>;
@group(0) @binding(19) var<storage, read> environmentMapCdf: array<f32>;
@group(0) @binding(20) var<storage, read> pointLights: array<vec4f>;
@group(0) @binding(21) var<storage, read> spotLights: array<vec4f>;
@group(0) @binding(22) var<storage, read> rectAreaLights: array<vec4f>;
@group(0) @binding(23) var<storage, read> meshAreaLights: array<vec4f>;

const LEAFNODE_FLAG = 0xffff0000u;
const MATERIAL_VEC4_STRIDE = 22u;
const MATERIAL_SCALAR_STRIDE = MATERIAL_VEC4_STRIDE * 4u;
const THIN_FILM_LAYER_LIMIT = 8u;
const THIN_FILM_SCALAR_BASE = 28u;
const SPECTRAL_SCALAR_BASE = 52u;
const SPECTRAL_SAMPLE_COUNT = 32u;
`;
