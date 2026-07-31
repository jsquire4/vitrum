/**
 * uboSentinelRoundTrip.test.ts — I3.3/D3.17 offset-drift guard.
 *
 * Pins the WalkaroundUBO layout in three ways:
 *
 *   1. WGSL struct offset consistency — every field's `// offset NNN` comment
 *      in walkaroundUbo.wgsl.ts matches the running offset derived from the
 *      WGSL types' byte sizes (mat4x4f=64, vec3f=12, vec2u=8, u32/f32=4).
 *      If a field's comment drifts from its real offset the struct and host
 *      packer silently disagree — this catches that before a GPU run.
 *
 *   2. Packer sentinel round-trip — `packWalkaroundUBO` is called with a
 *      synthetic PipelineFrameInputs where every numeric field carries a
 *      distinct sentinel value (1000+k for ascending k). The produced
 *      ArrayBuffer is inspected at each field's documented byte offset; the
 *      value there must equal the sentinel that was written to the matching
 *      input field. This pins packer-index ↔ WGSL-offset agreement field by
 *      field. Any swap, off-by-one in index, or wrong view type (f32 vs u32)
 *      is caught here without a GPU.
 *
 *   3. Total size — the buffer is exactly WALKAROUND_UBO_SIZE_BYTES, and
 *      the last WGSL field ends exactly at the byte length declared by the
 *      shared constant.
 *
 * IMPORTANT: if this test fails due to a real offset mismatch (a field's
 * comment offset does not match its layout offset, or the sentinel at the
 * documented offset is wrong) that is a GENUINE BUG in the packer — do NOT
 * adjust offsets to make the test pass without understanding the root cause.
 */

import { describe, expect, it } from 'vitest';
import { packWalkaroundUBO, updateUBO } from '../src/pipeline/uboUpdater.js';
import { WALKAROUND_UBO_SIZE_BYTES } from '../src/pipeline/constants.js';
import { WALKAROUND_UBO_WGSL } from '../src/shaders/walkaroundUbo.wgsl.js';
import type { PipelineFrameInputs } from '../src/pipeline/WalkaroundGPUPipeline.js';

// ────────────────────────────────────────────────────────────────────────────
// Part 1: WGSL struct offset consistency
// ────────────────────────────────────────────────────────────────────────────

/**
 * Parse the WalkaroundUBO struct from the WGSL source text.
 * Returns an array of { name, wgslType, commentedOffset } for every field that
 * has an `// offset NNN` comment.
 */
function parseWgslStructFields(wgsl: string): Array<{
  name: string;
  wgslType: string;
  commentedOffset: number;
}> {
  // Match lines like: "  fieldName:   wgslType,   //  offset NNN"
  // The offset comment may be followed by a dash and description.
  const lineRe = /^\s+(\w+)\s*:\s*(\S+),.*\/\/\s*offset\s+(\d+)/;
  const fields: Array<{ name: string; wgslType: string; commentedOffset: number }> = [];

  // Extract only the struct body (between "struct WalkaroundUBO {" and closing "};")
  const structMatch = wgsl.match(/struct WalkaroundUBO \{([\s\S]*?)\};/);
  if (!structMatch) throw new Error('WalkaroundUBO struct not found in WGSL');
  const body = structMatch[1]!;

  for (const line of body.split('\n')) {
    const m = line.match(lineRe);
    if (m) {
      fields.push({
        name: m[1]!,
        wgslType: m[2]!,
        commentedOffset: parseInt(m[3]!, 10),
      });
    }
  }
  return fields;
}

/**
 * Compute the byte size of a WGSL type (std140-like sequential packing as used
 * in the WalkaroundUBO, which has no inter-field padding beyond what vec3f
 * implies via its 16-byte alignment in WGSL uniform structs).
 *
 * The WalkaroundUBO layout note says vec3f fields are followed by a scalar
 * that occupies the 4-byte pad slot. We derive sizes purely from the types
 * declared in the struct: the struct is laid out sequentially with each field
 * at the running offset indicated by its comment. We validate the comment is
 * consistent with sequential accumulation.
 *
 * Sizes:
 *   mat4x4f  = 64 bytes (4×4 f32)
 *   vec3f    = 12 bytes (3 f32, 4-byte aligned — the pad is the NEXT scalar)
 *   vec4f    = 16 bytes
 *   vec2u    = 8 bytes
 *   vec3u    = 12 bytes
 *   f32/u32  = 4 bytes
 */
function wgslTypeByteSize(type: string): number {
  switch (type) {
    case 'mat4x4f': return 64;
    case 'vec3f':   return 12;
    case 'vec4f':   return 16;
    case 'vec2u':   return 8;
    case 'vec3u':   return 12;
    case 'f32':     return 4;
    case 'u32':     return 4;
    default: throw new Error(`Unknown WGSL type for size computation: ${type}`);
  }
}

describe('WalkaroundUBO — WGSL struct offset consistency', () => {
  it('every field comment offset matches sequential layout (vec3f/scalar pair rule)', () => {
    const fields = parseWgslStructFields(WALKAROUND_UBO_WGSL);
    expect(fields.length).toBeGreaterThan(0);

    let runningOffset = 0;
    for (const field of fields) {
      // The commented offset must equal the running offset.
      expect(
        field.commentedOffset,
        `Field "${field.name}" (${field.wgslType}) comment says offset ${field.commentedOffset} ` +
        `but sequential layout gives ${runningOffset}`,
      ).toBe(runningOffset);

      runningOffset += wgslTypeByteSize(field.wgslType);
    }

    // After all fields the running offset is the total struct size.
    expect(runningOffset).toBe(WALKAROUND_UBO_SIZE_BYTES);
  });

  it('struct size constant matches WALKAROUND_UBO_SIZE_BYTES (432)', () => {
    expect(WALKAROUND_UBO_SIZE_BYTES).toBe(432);
    // The WGSL source also documents the size inline.
    expect(WALKAROUND_UBO_WGSL).toContain('struct size 432 bytes');
  });

  it('last field (sunAngular) at offset 416 + 16 bytes ends exactly at 432', () => {
    const fields = parseWgslStructFields(WALKAROUND_UBO_WGSL);
    const last = fields[fields.length - 1]!;
    expect(last.name).toBe('sunAngular');
    expect(last.commentedOffset).toBe(416);
    expect(last.commentedOffset + wgslTypeByteSize(last.wgslType)).toBe(432);
  });
});

// ────────────────────────────────────────────────────────────────────────────
// Part 2: Packer sentinel round-trip
// ────────────────────────────────────────────────────────────────────────────

/**
 * Build a PipelineFrameInputs where every scalar field carries a distinct
 * sentinel value (1000+k, incrementing k). The sentinels are chosen so that
 * their IEEE-754 f32 representation is exact (integers in [0..2^24)), which
 * lets us compare them as f32 OR u32 bit-pattern without rounding issues.
 *
 * Return value: the inputs AND the expected sentinel map from "field name as
 * documented in uboUpdater.ts comment" → value, so we can assert each byte
 * offset.
 */
function buildSentinelInputs(): {
  inputs: PipelineFrameInputs;
  expected: Record<string, number>;
} {
  let k = 0;
  const s = (): number => 1000 + k++;

  // mat4x4f sentinel matrices — we'll put a known float32 in every slot.
  // Use 1100..1115 for viewMatrix, 1116..1131 for projMatrix, 1132..1147 for prevViewProjMatrix.
  const viewMatrix = Float32Array.from({ length: 16 }, () => s());
  // Reset k so projMatrix starts immediately after viewMatrix sentinels.
  const projMatrix = Float32Array.from({ length: 16 }, () => s());
  const prevViewProjMatrix = Float32Array.from({ length: 16 }, () => s());

  // Camera pos: vec3f (offsets 192, 196, 200 → f32[48], f32[49], f32[50])
  const cpx = s(), cpy = s(), cpz = s();
  // frameSeed: u32 → u32[51]
  const frameSeed = s();
  // screenWidth: u32 → u32[52], screenHeight: u32 → u32[53]
  const screenWidth = s(), screenHeight = s();
  // emitterCount: u32 → u32[54]
  const emitterCount = s();
  // offset 220 is a retired zero ABI pad.
  // primaryLightDir: vec3f → f32[56..58]
  const pldx = s(), pldy = s(), pldz = s();
  // primaryLightIntensity: f32 → f32[59]
  const primaryLightIntensity = s();
  // skyTint: vec3f → f32[60..62]
  const stR = s(), stG = s(), stB = s();
  // skyIrradiance: f32 → f32[63]
  const skyIrradiance = s();
  // emitterDist2Floor: f32 → f32[64]
  const emitterDist2Floor = s();
  // directFireflyClamp: f32 → f32[65]
  const directFireflyClamp = s();
  // causticBoost: f32 → f32[66]
  const causticBoost = s();
  // causticVisClamp: f32 → f32[67]
  const causticVisClamp = s();
  // temporalMClampDI: u32 → u32[68]
  const temporalMClampDI = s();
  // spatialReuseRadiusPx: f32 → f32[69]
  const spatialReuseRadiusPx = s();
  // spatialDepthTolFloor: f32 → f32[70]
  const spatialDepthTolFloor = s();
  // triIntersectEpsilon: f32 → f32[71]
  const triIntersectEpsilon = s();
  // glassMixScale: f32 → f32[72]
  const glassMixScale = s();
  // restirGiWCap: f32 → f32[73]
  const restirGiWCap = s();
  // restirGiIrrClamp: f32 → f32[74]
  const restirGiIrrClamp = s();
  // restirGiMClamp: u32 → u32[75]
  const restirGiMClamp = s();
  // restirGiSpatialRadiusPx: f32 → f32[76]
  const restirGiSpatialRadiusPx = s();
  // restirGiSpatialNormalDotMin: f32 → f32[77]
  const restirGiSpatialNormalDotMin = s();
  // restirGiSpatialCoplanarTol: f32 → f32[78]
  const restirGiSpatialCoplanarTol = s();
  // frameParity (checkerboard): u32 → u32[79] — tested separately (gate-driven); use 0 in sentinels
  // indirectFireflyClamp: vec3f → f32[80..82]
  const ifcR = s(), ifcG = s(), ifcB = s();
  // checkerboardOn: u32 → u32[83] — gate-driven; 0 in sentinels
  // bvhMode: u32 → u32[84]
  const bvhMode = s();
  // tlasNodeCount: u32 → u32[85]
  const tlasNodeCount = s();
  // stainedGlassFlags: u32 → u32[86]
  const stainedGlassFlags = s();
  // ppgEnabled: u32 → u32[87] — gate-driven; 0 in sentinels
  // ppgMixAlpha: f32 → f32[88] — gate-driven; 0 in sentinels
  // lightTreeEnabled: u32 → u32[89]
  const lightTreeEnabled = s();
  // lightTreeNodeCount: u32 → u32[90]
  const lightTreeNodeCount = s();
  // restirReservoirScale: u32 â†’ u32[91], default 1.
  // regirOrigin: vec3f → f32[92..94]
  const rorX = s(), rorY = s(), rorZ = s();
  // regirInvCellSize: f32 → f32[95]
  const regirInvCellSize = s();
  // regirDims: vec3u → u32[96..98]
  const rdX = s(), rdY = s(), rdZ = s();
  // regirEnabled: u32 → u32[99] — driven by regir.enabled; tested separately
  // regirCandidatesPerCell: u32 → u32[100]
  const candidatesPerCell = s();
  // regirSurvivorsPerCell: u32 → u32[101]
  const survivorsPerCell = s();
  // regirGridFloatOffset: u32 → u32[102]
  const gridFloatOffset = s();
  // rayOriginBias: f32 → f32[103]
  const rayOriginBias = s();
  // sunAngular.x: f32 → f32[104]
  const sunAngularRadius = s();

  const _m4 = new Float32Array(16).fill(0); // zero mat4 available for matrix-field assertions

  const inputs: PipelineFrameInputs = {
    camera: {
      viewMatrix,
      projMatrix,
      prevViewProjMatrix,
      cameraPos: [cpx, cpy, cpz],
    },
    screen: {
      screenWidth,
      screenHeight,
      frameSeed,
      swapChainView: {} as GPUTextureView,
      swapChainFormat: 'bgra8unorm',
    },
    lighting: {
      emitterCount,
      primaryLightDir: [pldx, pldy, pldz],
      primaryLightIntensity,
      sunAngularRadius,
      skyTint: [stR, stG, stB],
      skyIrradiance,
      emitterDist2Floor,
      directFireflyClamp,
      causticBoost,
      causticVisClamp,
      lightTreeEnabled,
      lightTreeNodeCount,
    },
    restirDI: {
      temporalMClampDI,
      spatialReuseRadiusPx,
      spatialDepthTolFloor,
    },
    restirGI: {
      restirGiWCap,
      restirGiIrrClamp,
      restirGiMClamp,
      restirGiSpatialRadiusPx,
      restirGiSpatialNormalDotMin,
      restirGiSpatialCoplanarTol,
    },
    gtao: {
      gtaoRadiusPx: 32, gtaoIntensity: 2, gtaoDepthThreshold: 2,
      gtaoBilateralDepthSigma: 0.25,
      adaptiveSamplingThresholdLow: 0.01, adaptiveSamplingThresholdHigh: 0.1,
    },
    filter: {
      triIntersectEpsilon,
      rayOriginBias,
      glassMixScale,
      indirectFireflyClamp: [ifcR, ifcG, ifcB],
      atrousDirectSigmas: [128, 5, 0.05],
      atrousIndirectSigmas: [32, 20, 0.5],
      stainedGlassFlags,
    },
    bvh: { bvhMode, tlasNodeCount },
    composite: { tonemapMode: 0, exposure: 1.0, outputColorSpace: 0 },
  } as unknown as PipelineFrameInputs;

  // regir: pass enabled=true with the sentinel values so all grid fields are written.
  const regir = {
    enabled: true,
    origin: [rorX, rorY, rorZ] as readonly [number, number, number],
    invCellSize: regirInvCellSize,
    dims: [rdX, rdY, rdZ] as readonly [number, number, number],
    candidatesPerCell,
    survivorsPerCell,
    gridFloatOffset,
  };

  // Map from descriptive field name → expected u32 word index × 4 (= byte offset)
  // and expected value. We split into f32-viewed and u32-viewed fields.
  // Convention: offsetBytes is the WGSL struct offset from the comment.
  const expected: Record<string, number> = {
    // mat4x4f fields — sentinels are the array values themselves.
    // We check one representative element per matrix (the first).
    'viewMatrix[0]@0':          viewMatrix[0]!,
    'viewMatrix[15]@60':        viewMatrix[15]!,
    'projMatrix[0]@64':         projMatrix[0]!,
    'projMatrix[15]@124':       projMatrix[15]!,
    'prevViewProjMatrix[0]@128': prevViewProjMatrix[0]!,
    'prevViewProjMatrix[15]@188': prevViewProjMatrix[15]!,
    // Scalar fields by byte offset (= u32 index × 4)
    'cameraPos.x@192':          cpx,
    'cameraPos.y@196':          cpy,
    'cameraPos.z@200':          cpz,
    'frameSeed@204':            frameSeed,
    'screenWidth@208':          screenWidth,
    'screenHeight@212':         screenHeight,
    'emitterCount@216':         emitterCount,
    '_abiPadEmitterPower@220':  0,
    'primaryLightDir.x@224':    pldx,
    'primaryLightDir.y@228':    pldy,
    'primaryLightDir.z@232':    pldz,
    'primaryLightIntensity@236': primaryLightIntensity,
    'skyTint.x@240':            stR,
    'skyTint.y@244':            stG,
    'skyTint.z@248':            stB,
    'skyIrradiance@252':        skyIrradiance,
    'emitterDist2Floor@256':    emitterDist2Floor,
    'directFireflyClamp@260':   directFireflyClamp,
    'causticBoost@264':         causticBoost,
    'causticVisClamp@268':      causticVisClamp,
    'temporalMClampDI@272':     temporalMClampDI,
    'spatialReuseRadiusPx@276': spatialReuseRadiusPx,
    'spatialDepthTolFloor@280': spatialDepthTolFloor,
    'triIntersectEpsilon@284':  triIntersectEpsilon,
    'glassMixScale@288':        glassMixScale,
    'restirGiWCap@292':         restirGiWCap,
    'restirGiIrrClamp@296':     restirGiIrrClamp,
    'restirGiMClamp@300':       restirGiMClamp,
    'restirGiSpatialRadiusPx@304': restirGiSpatialRadiusPx,
    'restirGiSpatialNormalDotMin@308': restirGiSpatialNormalDotMin,
    'restirGiSpatialCoplanarTol@312': restirGiSpatialCoplanarTol,
    // offset 316 = frameParity — 0 because checkerboard is passed as OFF default
    'indirectFireflyClamp.x@320': ifcR,
    'indirectFireflyClamp.y@324': ifcG,
    'indirectFireflyClamp.z@328': ifcB,
    // offset 332 = checkerboardOn — 0 because checkerboard is passed as OFF default
    'bvhMode@336':              bvhMode,
    'tlasNodeCount@340':        tlasNodeCount,
    'stainedGlassFlags@344':    stainedGlassFlags,
    // offset 348 = ppgEnabled — 0 because ppg is passed as OFF default
    // offset 352 = ppgMixAlpha — 0 because ppg is passed as OFF default
    'lightTreeEnabled@356':     lightTreeEnabled,
    'lightTreeNodeCount@360':   lightTreeNodeCount,
    'restirReservoirScale@364': 1,
    'regirOrigin.x@368':        rorX,
    'regirOrigin.y@372':        rorY,
    'regirOrigin.z@376':        rorZ,
    'regirInvCellSize@380':     regirInvCellSize,
    'regirDims.x@384':          rdX,
    'regirDims.y@388':          rdY,
    'regirDims.z@392':          rdZ,
    'regirEnabled@396':         1, // regir.enabled=true ⇒ 1
    'regirCandidatesPerCell@400': candidatesPerCell,
    'regirSurvivorsPerCell@404': survivorsPerCell,
    'regirGridFloatOffset@408': gridFloatOffset,
    'rayOriginBias@412':         rayOriginBias,
    'sunAngular.x@416':         sunAngularRadius,
  };

  return { inputs, expected, regir } as unknown as {
    inputs: PipelineFrameInputs;
    expected: Record<string, number>;
  };
}

describe('packWalkaroundUBO — sentinel round-trip (packer index ↔ WGSL offset)', () => {
  it('total buffer size is exactly WALKAROUND_UBO_SIZE_BYTES (432)', () => {
    const { inputs } = buildSentinelInputs() as unknown as {
      inputs: PipelineFrameInputs;
      expected: Record<string, number>;
    };
    const buf = packWalkaroundUBO(inputs);
    expect(buf.byteLength).toBe(WALKAROUND_UBO_SIZE_BYTES);
    expect(WALKAROUND_UBO_SIZE_BYTES).toBe(432);
  });

  it('every documented field has its sentinel at the correct byte offset', () => {
    // Build sentinel inputs and the regir override.
    let k = 0;
    const s = (): number => 1000 + k++;
    const viewMatrix = Float32Array.from({ length: 16 }, () => s());
    const projMatrix = Float32Array.from({ length: 16 }, () => s());
    const prevViewProjMatrix = Float32Array.from({ length: 16 }, () => s());
    const cpx = s(), cpy = s(), cpz = s();
    const frameSeed = s();
    const screenWidth = s(), screenHeight = s();
    const emitterCount = s();
    // offset 220 is a retired zero ABI pad.
    const pldx = s(), pldy = s(), pldz = s();
    const primaryLightIntensity = s();
    const stR = s(), stG = s(), stB = s();
    const skyIrradiance = s();
    const emitterDist2Floor = s();
    const directFireflyClamp = s();
    const causticBoost = s();
    const causticVisClamp = s();
    const temporalMClampDI = s();
    const spatialReuseRadiusPx = s();
    const spatialDepthTolFloor = s();
    const triIntersectEpsilon = s();
    const glassMixScale = s();
    const restirGiWCap = s();
    const restirGiIrrClamp = s();
    const restirGiMClamp = s();
    const restirGiSpatialRadiusPx = s();
    const restirGiSpatialNormalDotMin = s();
    const restirGiSpatialCoplanarTol = s();
    const ifcR = s(), ifcG = s(), ifcB = s();
    const bvhMode = s();
    const tlasNodeCount = s();
    const stainedGlassFlags = s();
    const lightTreeEnabled = s();
    const lightTreeNodeCount = s();
    // offset 364 is the live reservoir-scale control (default 1).
    const rorX = s(), rorY = s(), rorZ = s();
    const regirInvCellSize = s();
    const rdX = s(), rdY = s(), rdZ = s();
    const candidatesPerCell = s();
    const survivorsPerCell = s();
    const gridFloatOffset = s();
    const rayOriginBias = s();

    const inputs: PipelineFrameInputs = {
      camera: { viewMatrix, projMatrix, prevViewProjMatrix, cameraPos: [cpx, cpy, cpz] },
      screen: {
        screenWidth, screenHeight, frameSeed,
        swapChainView: {} as GPUTextureView, swapChainFormat: 'bgra8unorm',
      },
      lighting: {
        emitterCount,
        primaryLightDir: [pldx, pldy, pldz], primaryLightIntensity,
        skyTint: [stR, stG, stB], skyIrradiance,
        emitterDist2Floor, directFireflyClamp, causticBoost, causticVisClamp,
        lightTreeEnabled, lightTreeNodeCount,
      },
      restirDI: { temporalMClampDI, spatialReuseRadiusPx, spatialDepthTolFloor },
      restirGI: {
        restirGiWCap, restirGiIrrClamp, restirGiMClamp,
        restirGiSpatialRadiusPx, restirGiSpatialNormalDotMin, restirGiSpatialCoplanarTol,
      },
      gtao: {
        gtaoRadiusPx: 32, gtaoIntensity: 2, gtaoDepthThreshold: 2,
        gtaoBilateralDepthSigma: 0.25,
        adaptiveSamplingThresholdLow: 0.01, adaptiveSamplingThresholdHigh: 0.1,
      },
      filter: {
        triIntersectEpsilon, rayOriginBias, glassMixScale,
        indirectFireflyClamp: [ifcR, ifcG, ifcB],
        atrousDirectSigmas: [128, 5, 0.05], atrousIndirectSigmas: [32, 20, 0.5],
        stainedGlassFlags,
      },
      bvh: { bvhMode, tlasNodeCount },
      composite: { tonemapMode: 0, exposure: 1.0, outputColorSpace: 0 },
    } as unknown as PipelineFrameInputs;

    const regirState = {
      enabled: true,
      origin: [rorX, rorY, rorZ] as readonly [number, number, number],
      invCellSize: regirInvCellSize,
      dims: [rdX, rdY, rdZ] as readonly [number, number, number],
      candidatesPerCell, survivorsPerCell, gridFloatOffset,
    };

    const buf = packWalkaroundUBO(
      inputs,
      { enabled: false, mixAlpha: 0 }, // ppg OFF — slots 348/352 stay 0
      regirState,
      { enabled: false, frameParity: 0 }, // checkerboard OFF — slots 316/332 stay 0
    );

    const f32v = new Float32Array(buf);
    const u32v = new Uint32Array(buf);

    // Helper: read a f32 at byte offset.
    const rf = (byteOffset: number): number => f32v[byteOffset / 4]!;
    // Helper: read a u32 at byte offset.
    const ru = (byteOffset: number): number => u32v[byteOffset / 4]!;

    // ── mat4x4f fields ──────────────────────────────────────────────────────
    // offset 0: viewMatrix (64 bytes = 16×f32)
    for (let i = 0; i < 16; i++) {
      expect(rf(0 + i * 4)).toBe(viewMatrix[i]);
    }
    // offset 64: projMatrix (64 bytes)
    for (let i = 0; i < 16; i++) {
      expect(rf(64 + i * 4)).toBe(projMatrix[i]);
    }
    // offset 128: prevViewProjMatrix (64 bytes)
    for (let i = 0; i < 16; i++) {
      expect(rf(128 + i * 4)).toBe(prevViewProjMatrix[i]);
    }

    // ── Scalar fields ────────────────────────────────────────────────────────
    // cameraPos (vec3f): offset 192, 196, 200
    expect(rf(192)).toBe(cpx);
    expect(rf(196)).toBe(cpy);
    expect(rf(200)).toBe(cpz);
    // frameSeed (u32): offset 204
    expect(ru(204)).toBe(frameSeed);
    // screenSize (vec2u): offset 208, 212
    expect(ru(208)).toBe(screenWidth);
    expect(ru(212)).toBe(screenHeight);
    // emitterCount (u32): offset 216
    expect(ru(216)).toBe(emitterCount);
    // retired emitter-power mirror: explicit zero ABI pad at offset 220
    expect(rf(220)).toBe(0);
    // sunDirection/primaryLightDir (vec3f): offset 224, 228, 232
    expect(rf(224)).toBe(pldx);
    expect(rf(228)).toBe(pldy);
    expect(rf(232)).toBe(pldz);
    // sunIntensity/primaryLightIntensity (f32): offset 236
    expect(rf(236)).toBe(primaryLightIntensity);
    // skyTint (vec3f): offset 240, 244, 248
    expect(rf(240)).toBe(stR);
    expect(rf(244)).toBe(stG);
    expect(rf(248)).toBe(stB);
    // skyIrradiance (f32): offset 252
    expect(rf(252)).toBe(skyIrradiance);
    // emitterDist2Floor (f32): offset 256
    expect(rf(256)).toBe(emitterDist2Floor);
    // directFireflyClamp (f32): offset 260
    expect(rf(260)).toBe(directFireflyClamp);
    // causticBoost (f32): offset 264
    expect(rf(264)).toBe(causticBoost);
    // causticVisClamp (f32): offset 268
    expect(rf(268)).toBe(causticVisClamp);
    // temporalMClampDI (u32): offset 272
    expect(ru(272)).toBe(temporalMClampDI);
    // spatialReuseRadiusPx (f32): offset 276
    expect(rf(276)).toBe(spatialReuseRadiusPx);
    // spatialDepthTolFloor (f32): offset 280
    expect(rf(280)).toBe(spatialDepthTolFloor);
    // triIntersectEpsilon (f32): offset 284
    expect(rf(284)).toBe(triIntersectEpsilon);
    // glassMixScale (f32): offset 288
    expect(rf(288)).toBe(glassMixScale);
    // restirGiWCap (f32): offset 292
    expect(rf(292)).toBe(restirGiWCap);
    // restirGiIrrClamp (f32): offset 296
    expect(rf(296)).toBe(restirGiIrrClamp);
    // restirGiMClamp (u32): offset 300
    expect(ru(300)).toBe(restirGiMClamp);
    // restirGiSpatialRadiusPx (f32): offset 304
    expect(rf(304)).toBe(restirGiSpatialRadiusPx);
    // restirGiSpatialNormalDotMin (f32): offset 308
    expect(rf(308)).toBe(restirGiSpatialNormalDotMin);
    // restirGiSpatialCoplanarTol (f32): offset 312
    expect(rf(312)).toBe(restirGiSpatialCoplanarTol);
    // frameParity (u32): offset 316 — 0 because checkerboard OFF
    expect(ru(316)).toBe(0);
    // indirectFireflyClamp (vec3f): offset 320, 324, 328
    expect(rf(320)).toBe(ifcR);
    expect(rf(324)).toBe(ifcG);
    expect(rf(328)).toBe(ifcB);
    // checkerboardOn (u32): offset 332 — 0 because checkerboard OFF
    expect(ru(332)).toBe(0);
    // bvhMode (u32): offset 336
    expect(ru(336)).toBe(bvhMode);
    // tlasNodeCount (u32): offset 340
    expect(ru(340)).toBe(tlasNodeCount);
    // stainedGlassFlags (u32): offset 344
    expect(ru(344)).toBe(stainedGlassFlags);
    // ppgEnabled (u32): offset 348 — 0 because ppg OFF
    expect(ru(348)).toBe(0);
    // ppgMixAlpha (f32): offset 352 — 0 because ppg OFF
    expect(rf(352)).toBe(0);
    // lightTreeEnabled (u32): offset 356
    expect(ru(356)).toBe(lightTreeEnabled);
    // lightTreeNodeCount (u32): offset 360
    expect(ru(360)).toBe(lightTreeNodeCount);
    // independent ReSTIR reservoir scale (default 1)
    expect(ru(364)).toBe(1);
    // regirOrigin (vec3f): offset 368, 372, 376
    expect(rf(368)).toBe(rorX);
    expect(rf(372)).toBe(rorY);
    expect(rf(376)).toBe(rorZ);
    // regirInvCellSize (f32): offset 380
    expect(rf(380)).toBe(regirInvCellSize);
    // regirDims (vec3u): offset 384, 388, 392
    expect(ru(384)).toBe(rdX);
    expect(ru(388)).toBe(rdY);
    expect(ru(392)).toBe(rdZ);
    // regirEnabled (u32): offset 396 — 1 because regir.enabled=true
    expect(ru(396)).toBe(1);
    // regirCandidatesPerCell (u32): offset 400
    expect(ru(400)).toBe(candidatesPerCell);
    // regirSurvivorsPerCell (u32): offset 404
    expect(ru(404)).toBe(survivorsPerCell);
    // regirGridFloatOffset (u32): offset 408
    expect(ru(408)).toBe(gridFloatOffset);
    // scene-relative secondary-ray origin offset at 412.
    expect(rf(412)).toBe(rayOriginBias);
  });

  it('ppg ON populates ppgEnabled=1 and ppgMixAlpha at offsets 348/352', () => {
    const m = new Float32Array(16).fill(0);
    const base = {
      camera: { viewMatrix: m, projMatrix: m, prevViewProjMatrix: m, cameraPos: [0, 0, 0] },
      screen: { screenWidth: 64, screenHeight: 64, frameSeed: 1, swapChainView: {} as GPUTextureView, swapChainFormat: 'bgra8unorm' },
      lighting: { emitterCount: 0, primaryLightDir: [0, 1, 0], primaryLightIntensity: 1, skyTint: [0, 0, 0], skyIrradiance: 0, emitterDist2Floor: 0.01, directFireflyClamp: 4, causticBoost: 1, causticVisClamp: 1 },
      restirDI: { temporalMClampDI: 20, spatialReuseRadiusPx: 30, spatialDepthTolFloor: 0.05 },
      restirGI: { restirGiWCap: 16, restirGiIrrClamp: 5, restirGiMClamp: 50, restirGiSpatialRadiusPx: 12, restirGiSpatialNormalDotMin: 0.9, restirGiSpatialCoplanarTol: 0.05 },
      gtao: { gtaoRadiusPx: 32, gtaoIntensity: 2, gtaoDepthThreshold: 2, gtaoBilateralDepthSigma: 0.25, adaptiveSamplingThresholdLow: 0.01, adaptiveSamplingThresholdHigh: 0.1 },
      filter: { triIntersectEpsilon: 1e-5, rayOriginBias: 1e-3, glassMixScale: 0.7, indirectFireflyClamp: [1, 1, 1], atrousDirectSigmas: [128, 5, 0.05], atrousIndirectSigmas: [32, 20, 0.5], stainedGlassFlags: 0 },
      bvh: { bvhMode: 0, tlasNodeCount: 0 },
      composite: { tonemapMode: 0, exposure: 1.0, outputColorSpace: 0 },
    } as unknown as PipelineFrameInputs;

    const buf = packWalkaroundUBO(base, { enabled: true, mixAlpha: 0.75 });
    const f32v = new Float32Array(buf);
    const u32v = new Uint32Array(buf);
    expect(u32v[348 / 4]).toBe(1);        // ppgEnabled at offset 348
    expect(f32v[352 / 4]).toBeCloseTo(0.75); // ppgMixAlpha at offset 352
  });

  it('checkerboard ON populates frameParity/checkerboardOn at offsets 316/332', () => {
    const m = new Float32Array(16).fill(0);
    const base = {
      camera: { viewMatrix: m, projMatrix: m, prevViewProjMatrix: m, cameraPos: [0, 0, 0] },
      screen: { screenWidth: 64, screenHeight: 64, frameSeed: 1, swapChainView: {} as GPUTextureView, swapChainFormat: 'bgra8unorm' },
      lighting: { emitterCount: 0, primaryLightDir: [0, 1, 0], primaryLightIntensity: 1, skyTint: [0, 0, 0], skyIrradiance: 0, emitterDist2Floor: 0.01, directFireflyClamp: 4, causticBoost: 1, causticVisClamp: 1 },
      restirDI: { temporalMClampDI: 20, spatialReuseRadiusPx: 30, spatialDepthTolFloor: 0.05 },
      restirGI: { restirGiWCap: 16, restirGiIrrClamp: 5, restirGiMClamp: 50, restirGiSpatialRadiusPx: 12, restirGiSpatialNormalDotMin: 0.9, restirGiSpatialCoplanarTol: 0.05 },
      gtao: { gtaoRadiusPx: 32, gtaoIntensity: 2, gtaoDepthThreshold: 2, gtaoBilateralDepthSigma: 0.25, adaptiveSamplingThresholdLow: 0.01, adaptiveSamplingThresholdHigh: 0.1 },
      filter: { triIntersectEpsilon: 1e-5, rayOriginBias: 1e-3, glassMixScale: 0.7, indirectFireflyClamp: [1, 1, 1], atrousDirectSigmas: [128, 5, 0.05], atrousIndirectSigmas: [32, 20, 0.5], stainedGlassFlags: 0 },
      bvh: { bvhMode: 0, tlasNodeCount: 0 },
      composite: { tonemapMode: 0, exposure: 1.0, outputColorSpace: 0 },
    } as unknown as PipelineFrameInputs;

    const buf = packWalkaroundUBO(base, undefined, undefined, { enabled: true, frameParity: 1 });
    const u32v = new Uint32Array(buf);
    expect(u32v[316 / 4]).toBe(1);  // frameParity at offset 316
    expect(u32v[332 / 4]).toBe(1);  // checkerboardOn at offset 332
  });

  it('updateUBO produces identical bytes to packWalkaroundUBO (delegate is byte-exact)', () => {
    // Verify updateUBO still works after the refactor by checking it calls
    // writeBuffer with the same bytes packWalkaroundUBO returns.
    const m = new Float32Array(16).fill(0);
    const base = {
      camera: { viewMatrix: m, projMatrix: m, prevViewProjMatrix: m, cameraPos: [1, 2, 3] },
      screen: { screenWidth: 100, screenHeight: 200, frameSeed: 42, swapChainView: {} as GPUTextureView, swapChainFormat: 'bgra8unorm' },
      lighting: { emitterCount: 3, primaryLightDir: [0, 1, 0], primaryLightIntensity: 2, skyTint: [0.1, 0.2, 0.3], skyIrradiance: 1, emitterDist2Floor: 0.01, directFireflyClamp: 4, causticBoost: 1, causticVisClamp: 1 },
      restirDI: { temporalMClampDI: 20, spatialReuseRadiusPx: 30, spatialDepthTolFloor: 0.05 },
      restirGI: { restirGiWCap: 16, restirGiIrrClamp: 5, restirGiMClamp: 50, restirGiSpatialRadiusPx: 12, restirGiSpatialNormalDotMin: 0.9, restirGiSpatialCoplanarTol: 0.05 },
      gtao: { gtaoRadiusPx: 32, gtaoIntensity: 2, gtaoDepthThreshold: 2, gtaoBilateralDepthSigma: 0.25, adaptiveSamplingThresholdLow: 0.01, adaptiveSamplingThresholdHigh: 0.1 },
      filter: { triIntersectEpsilon: 1e-5, rayOriginBias: 1e-3, glassMixScale: 0.7, indirectFireflyClamp: [1, 1, 1], atrousDirectSigmas: [128, 5, 0.05], atrousIndirectSigmas: [32, 20, 0.5], stainedGlassFlags: 0 },
      bvh: { bvhMode: 0, tlasNodeCount: 0 },
      composite: { tonemapMode: 0, exposure: 1.0, outputColorSpace: 0 },
    } as unknown as PipelineFrameInputs;

    const expected = new Uint8Array(packWalkaroundUBO(base));

    const captured = new Uint8Array(WALKAROUND_UBO_SIZE_BYTES);
    const fakeDevice = {
      queue: { writeBuffer: (_b: unknown, _o: number, data: ArrayBuffer) => captured.set(new Uint8Array(data)) },
    } as unknown as GPUDevice;
    updateUBO(fakeDevice, {} as GPUBuffer, base);

    expect(captured).toEqual(expected);
  });
});
