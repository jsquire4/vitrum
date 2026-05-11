import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_TRACE_WGSL } from '../wgsl/pathTraceBruteforce.wgsl.js';

/**
 * FrameParams layout contract test.
 *
 * After the Pass-1 4.71 deferral cleanup, the FrameParams UBO is a fixed
 * 336-byte payload in a 512-byte buffer with this layout:
 *
 *   u32 slot 0..19   (offsets 0..79):
 *     0  width
 *     1  height
 *     2  frameIndex
 *     3  frameSeed
 *     4  triangleCount
 *     5  maxBounces
 *     6  bvhNodeCount
 *     7  analyticCount
 *     8  pointLightCount
 *     9  spotLightCount
 *     10 rectAreaLightCount
 *     11 meshAreaLightCount
 *     12 mneeMaxIterations
 *     13 mneeMaxChainLength
 *     14 hasEnvironmentMap   (0/1)
 *     15 causticStrategy     (0=none, 1=manifold-nee, 2=photon-map)
 *     16 environmentMapWidth
 *     17 environmentMapHeight
 *     18 _pad0
 *     19 _pad1
 *
 *   f32 slot 20..23 cameraPos    (xyz, .w = 1)
 *   f32 slot 24..27 lightDir     (xyz, .w = averageDirectionalIrradiance)
 *   f32 slot 28..31 environmentTint (xyz, .w = 0 — unused)
 *   f32 slot 32..35 environmentSun  (xyz sunDir, .w sunStrength)
 *
 *   f32 slot 36..51 invViewProj   (mat4x4f, 16 floats)
 *   f32 slot 52..67 viewProj      (mat4x4f, 16 floats)
 *   f32 slot 68..83 prevViewProj  (mat4x4f, 16 floats)
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

  it('declares the expected field order — u32 counts then vec4f camera/light/env, then 3 matrices', () => {
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
      '_pad0: u32',
      '_pad1: u32',
      'cameraPos: vec4f',
      'lightDir: vec4f',
      'environmentTint: vec4f',
      'environmentSun: vec4f',
      'invViewProj: mat4x4f',
      'viewProj: mat4x4f',
      'prevViewProj: mat4x4f',
    ]);
  });

  it('drops every legacy single-light vec4f field', () => {
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
    ];
    for (const dropped of droppedFields) {
      // The struct-field test above is positive; this one is defensive:
      // a future patch that re-adds, e.g., `meshAreaTriA: vec4f` would
      // silently break the host packer offset map.
      const re = new RegExp(`\\b${dropped}\\s*:\\s*vec4f\\b`);
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
    // After the rewrite, the WGSL code reads:
    //   pointLights[0].xyz / pointLights[1].rgb        (point light 0)
    //   spotLights[0].xyz / spotLights[1].{xyz,w} / spotLights[2].rgb
    //   rectAreaLights[0..3]  / meshAreaLights[0..3]
    // We verify a representative sample is present and the .w-stuffing reads
    // are gone.
    expect(PT_WEBGPU_TRACE_WGSL).toContain('pointLights[0].xyz');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('pointLights[1].rgb');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('spotLights[0].xyz');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('spotLights[1].w');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('spotLights[2].rgb');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('rectAreaLights[0].xyz');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('rectAreaLights[3].rgb');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('meshAreaLights[0].xyz');
    expect(PT_WEBGPU_TRACE_WGSL).toContain('meshAreaLights[3].rgb');
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

  it('fits inside the 512-byte UBO with the new layout (336 bytes used)', () => {
    // Field sizes (WGSL std140-ish for uniform buffers, vec4-aligned):
    //   20 u32 = 80 bytes
    //   4  vec4f = 64 bytes
    //   3  mat4x4f = 192 bytes
    //   total = 336 bytes, fits in 512.
    const u32Count = fields.filter((f) => /:\s*u32\b/.test(f)).length;
    const vec4Count = fields.filter((f) => /:\s*vec4f\b/.test(f)).length;
    const mat4Count = fields.filter((f) => /:\s*mat4x4f\b/.test(f)).length;
    const bytes = u32Count * 4 + vec4Count * 16 + mat4Count * 64;
    expect(bytes).toBe(336);
    expect(bytes).toBeLessThanOrEqual(512);
  });
});
