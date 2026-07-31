import { describe, expect, it } from 'vitest';
import { FRAME_PARAMS_BYTE_SIZE } from '../scene/frameParamsLayout.js';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { SPPM_PHOTON_PASS_WGSL } from '../wgsl/pathTrace/sppmBindings.wgsl.js';

/**
 * FrameParams layout contract test.
 *
 * FrameParams is a generated 368-byte semantic payload and exact-size engine
 * allocation. `cameraPos: vec3f` is immediately followed by
 * `environmentHdriIntensity: f32`, so the aligned fourth lane is live without
 * pretending the camera has a fourth component. Directional data lives only in
 * the dedicated storage/light textures.
 *
 * Per-light data lives in dedicated storage buffers at bind slots 20..23.
 * The .w-stuffing tricks (rectAreaPos.w hasFlag, meshAreaTriB.w envWidth, etc.)
 * are gone.
 *
 * We can't trivially call the host packer from a unit test (it's a private
 * instance method that depends on a real GPUDevice via `#sceneBuffers`).
 * Instead we (a) verify the WGSL struct field order matches the layout above
 * by parsing the struct definition, and (b) assert the struct does NOT still
 * reference any of the dropped per-light vec4 fields.
 */

function extractFrameParamsFields(wgsl: string): readonly string[] {
  const match = wgsl.match(/struct FrameParams\s*\{([\s\S]*?)\};/);
  if (match == null) {
    throw new Error('Could not find struct FrameParams in PT_WEBGPU_TRACE_WGSL');
  }
  const body = match[1] ?? '';
  return body
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith('//'))
    .map((line) => {
      // Field declarations are `name: type,`. Strip trailing comma and
      // any trailing comment.
      const noComment = line.split('//')[0] ?? line;
      return noComment.replace(/,\s*$/, '').trim();
    })
    .filter((line) => line.length > 0);
}

describe('FrameParams UBO layout (pt-webgpu)', () => {
  const fields = extractFrameParamsFields(PT_WEBGPU_TRACE_WGSL);

  it('declares the expected generated field order', () => {
    expect(fields).toEqual([
      'width: u32',
      'height: u32',
      'frameIndex: u32',
      'frameSeed: u32',
      'triangleCount: u32',
      'maxBounces: u32',
      'bvhNodeCount: u32',
      'analyticCount: u32',
      'pointLightCount: u32',
      'spotLightCount: u32',
      'rectAreaLightCount: u32',
      'meshAreaLightCount: u32',
      'mneeMaxIterations: u32',
      'mneeMaxChainLength: u32',
      'hasEnvironmentMap: u32',
      'causticStrategy: u32',
      'environmentMapWidth: u32',
      'environmentMapHeight: u32',
      'triIntersectEpsilon: f32',
      'tlasNodeCount: u32',
      'spectralEnabled: u32',
      'heroLambdaNm: f32',
      'heroPdf: f32',
      'bdptEnabled: u32',
      'bdptMaxLightBounces: u32',
      'bdptMaxEyeDepth: u32',
      'lightTreeEnabled: u32',
      'lightTreeNodeCount: u32',
      'cameraPos: vec3f',
      'environmentHdriIntensity: f32',
      'environmentTint: vec4f',
      'invViewProj: mat4x4f',
      'viewProj: mat4x4f',
      'prevViewProj: mat4x4f',
      // N-directional: packed directional count field added after prevViewProj.
      // Slot 96, the first scalar after the three mat4x4f blocks.
      'directionalLightCount: u32',
      // BDPT pseudo-distant emitter placement uses scene bounds, packed into
      // the existing scalar tail after directionalLightCount.
      'sceneCenterX: f32',
      'sceneCenterY: f32',
      'sceneCenterZ: f32',
      'sceneRadius: f32',
      'directLightingMode: u32',
      'rayOriginBias: f32',
      'environmentDistantPower: f32',
    ]);
  });

  it('drops every legacy single-light field and dead spectral constants', () => {
    // The Pass-1 cleanup removes these UBO fields and moves their data to
    // dedicated storage buffers (bindings 20..23). If any reappear in the
    // struct, the host packer offsets will silently mis-align.
    const droppedFields = [
      'pointLightPos',
      'pointLightRadiance',
      'spotLightPos',
      'spotLightDirection',
      'spotLightRadiance',
      'rectAreaPos',
      'rectAreaU',
      'rectAreaV',
      'rectAreaRadiance',
      'meshAreaTriA',
      'meshAreaTriB',
      'meshAreaTriC',
      'meshAreaRadiance',
      'lightDir',
      'cmfIntegralX',
      'cmfIntegralY',
      'cmfIntegralZ',
      'environmentSun',
    ];
    for (const dropped of droppedFields) {
      // The struct-field test above is positive; this one is defensive:
      // a future patch that re-adds, e.g., `meshAreaTriA: vec4f` would
      // silently break the host packer offset map.
      const re = new RegExp(`\\b${dropped}\\s*:`);
      expect(re.test(PT_WEBGPU_TRACE_WGSL)).toBe(false);
    }
  });

  it('exposes dedicated u32 fields for HDRI dims, caustic strategy, and MNEE knobs', () => {
    // These previously lived in `.w` lanes of unrelated vec4f fields.
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(/mneeMaxIterations\s*:\s*u32/);
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(/mneeMaxChainLength\s*:\s*u32/);
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(/hasEnvironmentMap\s*:\s*u32/);
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(/causticStrategy\s*:\s*u32/);
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(/environmentMapWidth\s*:\s*u32/);
    expect(PT_WEBGPU_TRACE_WGSL).toMatch(/environmentMapHeight\s*:\s*u32/);
  });

  it('reads per-light data from the storage-buffer arrays, not from params.<light>', () => {
    // After the Item-15 multi-light loop rewrite, rectAreaLights and
    // meshAreaLights are accessed via a loop variable index (rb = li * 4u),
    // not hardcoded [0]. The main direct-light loop still uses rb = ri * 4u.
    // We verify a representative sample is present and the .w-stuffing reads
    // are gone.
    // A4: point/spot light per-array reads (pointBase, spotBase) have MOVED to the
    // SPPM photon-emission compute pass (SPPM_PHOTON_PASS_WGSL).  The megakernel
    // caustic path now calls sppmGather() — no per-pixel point/spot loops.  Check
    // the photon-emission pass for the storage-array access patterns.
    expect(SPPM_PHOTON_PASS_WGSL).toContain('pointLights[pointBase].xyz');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('pointLights[pointBase + 1u].rgb');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('spotLights[spotBase].xyz');
    // spotBase + 1u row: saxisVec = spotLights[spotBase + 1u] holds direction + cosOuter.
    expect(SPPM_PHOTON_PASS_WGSL).toContain('spotLights[spotBase + 1u]');
    expect(SPPM_PHOTON_PASS_WGSL).toContain('spotLights[spotBase + 2u]');
    // Item 15: rectAreaLights and meshAreaLights now use loop-indexed rb/mb (megakernel).
    // 2026-06-10: disc-area native — rb+3u row is read as a full vec4 (rshape) so the
    // radiance (.rgb) + shape tag (.w) are extracted in one load; the substring
    // 'rectAreaLights[rb + 3u].rgb' no longer appears (it's 'rshape.rgb' instead).
    expect(PT_WEBGPU_TRACE_WGSL).toContain('rectAreaLights[rb].xyz');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('rectAreaLights[rb + 3u]');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('meshAreaLights[mb].xyz');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('sampleMeshAreaLightRadiance(');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('meshAreaLights[base + 6u].rgb');
    // No more reads of the dropped vec4f fields.
    expect(PT_WEBGPU_TRACE_WGSL).not.toMatch(/params\.pointLightPos\b/);
    expect(PT_WEBGPU_TRACE_WGSL).not.toMatch(/params\.spotLightPos\b/);
    expect(PT_WEBGPU_TRACE_WGSL).not.toMatch(/params\.rectAreaPos\b/);
    expect(PT_WEBGPU_TRACE_WGSL).not.toMatch(/params\.meshAreaTri[ABC]\b/);
    expect(PT_WEBGPU_TRACE_WGSL).not.toMatch(/params\.meshAreaRadiance\b/);
    expect(PT_WEBGPU_TRACE_WGSL).not.toMatch(/params\.rectAreaRadiance\b/);
    expect(PT_WEBGPU_TRACE_WGSL).not.toMatch(/params\.spotLightRadiance\b/);
    expect(PT_WEBGPU_TRACE_WGSL).not.toMatch(/params\.pointLightRadiance\b/);
    expect(PT_WEBGPU_TRACE_WGSL).not.toMatch(/params\.spotLightDirection\b/);
    expect(PT_WEBGPU_TRACE_WGSL).not.toMatch(/params\.rectAreaU\b/);
    expect(PT_WEBGPU_TRACE_WGSL).not.toMatch(/params\.rectAreaV\b/);
  });

  it('matches the exact WGSL-aligned byte size from the layout generator', () => {
    const u32Count = fields.filter((f) => /:\s*u32\b/.test(f)).length;
    const f32Count = fields.filter((f) => /:\s*f32\b/.test(f)).length;
    const vec3Count = fields.filter((f) => /:\s*vec3f\b/.test(f)).length;
    const vec4Count = fields.filter((f) => /:\s*vec4f\b/.test(f)).length;
    const mat4Count = fields.filter((f) => /:\s*mat4x4f\b/.test(f)).length;
    // Raw field sum (no inter-field alignment or struct-end padding). WGSL structs are
    // additionally padded to a multiple of their largest member's alignment (16 bytes
    // for mat4x4f). The cross-check in frameParamsSlotCrossCheck.test.ts applies the
    // full WGSL alignment rules; this test verifies the raw sum is close to and within
    // FRAME_PARAMS_BYTE_SIZE so stale fields are caught without false-failing on padding.
    const rawBytes =
      u32Count * 4 + f32Count * 4 + vec3Count * 12 + vec4Count * 16 + mat4Count * 64;
    // FRAME_PARAMS_BYTE_SIZE is the properly-padded size; raw < padded is expected
    // when the struct ends with a scalar (trailing u32 adds 4B raw but 16B padded).
    expect(rawBytes).toBeLessThanOrEqual(FRAME_PARAMS_BYTE_SIZE);
    expect(rawBytes).toBeGreaterThan(FRAME_PARAMS_BYTE_SIZE - 16); // at most 16B of end-padding
    expect(FRAME_PARAMS_BYTE_SIZE).toBe(368);
  });
});
