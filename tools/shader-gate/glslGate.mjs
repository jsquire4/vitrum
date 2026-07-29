#!/usr/bin/env -S deno run --sloppy-imports --allow-read --allow-env --allow-run --allow-write=/tmp
// @ts-nocheck — this is a .mjs file; deno runs it as JS with --sloppy-imports.
/**
 * tools/shader-gate/glslGate.mjs
 *
 * GLSL ES 3.00 compile gate for @vitrum/pt-webgl2.
 *
 * Composes the full production GLSL fragment program (preamble + compat defines + body)
 * for every production-reachable feature-flag combination and validates each one with
 * `glslangValidator` (from the `glslang-tools` system package on Ubuntu, or equivalent).
 *
 * Validator choice: glslangValidator (KhronosGroup/glslang)
 *   • Level: full GLSL ES 3.00 parse + type-check (not just syntax).
 *   • No SPIR-V output — we only validate; the flag combo below targets OpenGL GLSL ES.
 *   • Stronger than any pure-JS parser (glslx/glsl-parser are parse-only, no type checking;
 *     @webgpu/glslang compiles to SPIR-V and requires #version 310+, not 300).
 *   • Freely installable on all CI and dev Linux environments via `apt-get install -y glslang-tools`.
 *   • Limits: validates GLSL semantics but not WebGL2-specific extensions or uniform block rules
 *     that the real ANGLE/driver would enforce. Full driver validation still requires a real
 *     WebGL2 context (the wsl-gpu T1 smoke provides that gate).
 *
 * Prerequisite (install once):
 *   sudo apt-get install -y glslang-tools
 *
 * Usage (from repo root):
 *   npm run shader-gate:glsl
 *
 * Self-test mode (proves detection works):
 *   npm run shader-gate:glsl -- --self-test
 *
 * The deno.json in this directory provides the @vitrum/* import-map so transitive
 * imports from the production TS files resolve correctly.
 */

// ── Parse flags ──────────────────────────────────────────────────────────────
const selfTest = Deno.args.includes("--self-test");

// ── Locate glslangValidator ───────────────────────────────────────────────────
async function findGlslangValidator() {
  // Try PATH first (covers system install + any user-local bin)
  for (const candidate of ["glslangValidator", "glslangvalidator"]) {
    try {
      const cmd = new Deno.Command(candidate, { args: ["--version"] });
      const { success } = await cmd.output();
      if (success) return candidate;
    } catch {
      // not found
    }
  }
  return null;
}

const glslangBin = await findGlslangValidator();
if (!glslangBin) {
  console.error("[glsl-gate] ERROR: glslangValidator not found in PATH.");
  console.error(
    "  Install it with:  sudo apt-get install -y glslang-tools",
  );
  console.error(
    "  (On macOS:        brew install glslang)",
  );
  Deno.exit(1);
}

// ── Import the production pt-webgl2 GLSL machinery ───────────────────────────
// These imports use the @vitrum/* mappings from deno.json in this directory.
// --sloppy-imports lets Deno load the .ts files without a build step.
import {
  composeNeeCandidateGlsl,
  composeNeeResolveGlsl,
  composeTraceGlsl,
} from "../../packages/pt-webgl2/src/glsl/composeTraceGlsl.ts";
import {
  buildVertexSource,
  buildFragmentSource,
} from "../../packages/pt-webgl2/src/gl/glProgram.ts";
import {
  DEFAULT_TRACE_FEATURES,
  featureDefines,
} from "../../packages/pt-webgl2/src/featureTypes.ts";
import { FULLSCREEN_VERT } from "../../packages/pt-webgl2/src/gl/fullscreenQuad.ts";

// ── Production-reachable feature-combination matrix ──────────────────────────
// From featureTypes.ts and index.ts:
//   - bdpt: driven by opts.bdpt (true/false). At compose time, bdpt=true adds the
//     bdpt_light_subpath + bdpt_connection chunks; bdpt=false omits them entirely.
//     Both are production-reachable. MIS and Russian roulette are fixed production
//     invariants rather than cache-key dimensions.
//   - dof: driven by opts.dof (truthy/null). FEATURE_DOF resolves in #if blocks.
//   - cameraType: driven by opts.cameraType (0=perspective, 1=ortho, 2=equirect).
//     CAMERA_TYPE resolves in #if blocks inside getCameraRay().
//   - randomType: driven by opts.sampling for production PCG(0) and Sobol(1).
//   - material tier + fog: derived from the installed Scene. Every one of the four
//     tiers is production-reachable, and texture-free participating media select
//     scalarRichMaterials + fog.
//
// Composition branches on BDPT, RNG source, and material tier. The generated
// preamble also gates DOF, camera, and fog paths. The matrix below exercises every
// production-reachable dimension plus the heaviest BDPT+Sobol interaction:
//
//   Combo              bdpt  dof   cameraType  RNG    notes
//   ─────────────────────────────────────────────────────────────────
//   baseline            F    F       0         PCG    default production path
//   bdpt-on             T    F       0         PCG    BDPT chunks + fog path
//   dof-on              F    T       0         PCG    FEATURE_DOF=1 paths
//   cameraType-ortho    F    F       1         PCG    CAMERA_TYPE=1 paths
//   cameraType-equirect F    F       2         PCG    CAMERA_TYPE=2 paths
//   bdpt-sobol          T    F       0         Sobol  heaviest live interaction
//   sobol-on            F    F       0         Sobol  Sobol RNG composition
//   material-basic      F    F       0         PCG    basic material tier
//   scalar-rich-fog     F    F       0         PCG    scalar-rich + medium tier
//   material-mapped-pbr F    F       0         PCG    mapped base-PBR tier

const COMBOS = [
  {
    name: "baseline",
    features: { ...DEFAULT_TRACE_FEATURES },
  },
  {
    name: "bdpt-on",
    features: { ...DEFAULT_TRACE_FEATURES, bdpt: true, fog: true },
  },
  {
    name: "dof-on",
    features: { ...DEFAULT_TRACE_FEATURES, dof: true },
  },
  {
    name: "cameraType-ortho",
    features: { ...DEFAULT_TRACE_FEATURES, cameraType: 1 },
  },
  {
    name: "cameraType-equirect",
    features: { ...DEFAULT_TRACE_FEATURES, cameraType: 2 },
  },
  {
    name: "bdpt-sobol",
    features: { ...DEFAULT_TRACE_FEATURES, bdpt: true, randomType: 1 },
  },
  {
    name: "sobol-on",
    features: { ...DEFAULT_TRACE_FEATURES, randomType: 1 },
  },
  {
    name: "material-basic",
    features: {
      ...DEFAULT_TRACE_FEATURES,
      basicMaterials: true,
      mappedRichMaterials: false,
      fog: false,
    },
  },
  {
    name: "scalar-rich-fog",
    features: {
      ...DEFAULT_TRACE_FEATURES,
      scalarRichMaterials: true,
      mappedRichMaterials: false,
      fog: true,
    },
  },
  {
    name: "material-mapped-pbr",
    features: {
      ...DEFAULT_TRACE_FEATURES,
      mappedPbrMaterials: true,
      mappedRichMaterials: false,
      fog: false,
    },
  },
];

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Write src to a temp file and return the path. */
async function writeTmp(name, src) {
  const path = `/tmp/glsl-gate-${name}.glsl`;
  await Deno.writeTextFile(path, src);
  return path;
}

/**
 * Compile a GLSL source string with glslangValidator.
 * Returns { ok: boolean, output: string }.
 *
 * glslangValidator flags used:
 *   -S <stage>  — shader stage (vert / frag)
 *   <file>      — the temp file to validate
 *
 * No --target-env flag: without it, glslangValidator validates GLSL in-place
 * without targeting a SPIR-V environment, which correctly accepts #version 300 es
 * and enforces GLSL ES 3.00 type rules. (With --target-env opengl or vulkan it
 * refuses #version 300 es and requires 310+.)
 *
 * Exit code 0 = success, non-zero = one or more errors.
 */
async function validateGlsl(stageName, src, tmpName) {
  const stage = stageName === "vert" ? "vert" : "frag";
  const tmpPath = await writeTmp(tmpName, src);
  const cmd = new Deno.Command(glslangBin, {
    args: ["-S", stage, tmpPath],
    stdout: "piped",
    stderr: "piped",
  });
  const { success, stdout, stderr } = await cmd.output();
  const output = (new TextDecoder().decode(stdout) + new TextDecoder().decode(stderr)).trim();
  // Clean up temp file
  try { await Deno.remove(tmpPath); } catch { /* ignore */ }
  return { ok: success, output };
}

// ── Shader inventory ──────────────────────────────────────────────────────────
// Each entry: { name, vertSrc, fragSrc }
const shaders = [];

for (const combo of COMBOS) {
  const { name, features } = combo;
  const defines = featureDefines(features);
  const variants = [
    {
      suffix: "main",
      defines: { ...defines, NEE_CANDIDATE_PASS: 0 },
      body: composeTraceGlsl(features),
    },
    {
      suffix: "nee-candidate",
      defines: { ...defines, NEE_CANDIDATE_PASS: 1 },
      body: composeNeeCandidateGlsl(features),
    },
    {
      suffix: "nee-resolve",
      defines: { ...defines, NEE_CANDIDATE_PASS: 0 },
      body: composeNeeResolveGlsl(features),
    },
  ];
  for (const variant of variants) {
    const defineMap = new Map(Object.entries(variant.defines));
    const vertSrc = buildVertexSource(defineMap, FULLSCREEN_VERT);
    const fragSrc = buildFragmentSource(defineMap, variant.body);
    shaders.push({ name: `${name}/${variant.suffix}`, vertSrc, fragSrc });
  }
}

// ── Self-test: inject a broken fragment program ───────────────────────────────
if (selfTest) {
  // Use the baseline defines with an intentional type error injected into the body.
  const brokenDefines = featureDefines(COMBOS[0].features);
  const brokenDefineMap = new Map(Object.entries(brokenDefines));
  // A minimal broken fragment shader body: references an undefined variable so
  // glslangValidator must report an error. The error is injected AFTER the normal
  // composeTraceGlsl body so the preamble structure is unchanged — we're testing
  // that the gate detects errors anywhere in the composed output.
  const brokenFragBody = `
// SELF-TEST: intentional type error injected by --self-test.
// 'absolutely_undefined_glsl_identifier' is not declared — glslangValidator must reject.
void __selfTestBrokenChunk() {
  vec4 x = absolutely_undefined_glsl_identifier;
}
` + composeTraceGlsl(COMBOS[0].features);

  const brokenFragSrc = buildFragmentSource(brokenDefineMap, brokenFragBody);
  const brokenVertSrc = buildVertexSource(brokenDefineMap, FULLSCREEN_VERT);
  shaders.push({
    name: "__self-test/intentionally-broken",
    vertSrc: brokenVertSrc,
    fragSrc: brokenFragSrc,
    expectError: true,
  });
  console.log("[glsl-gate] --self-test: added 1 intentionally-broken shader");
}

// ── Compile loop ──────────────────────────────────────────────────────────────
let passed = 0;
let failed = 0;
const errors = [];

for (const entry of shaders) {
  const { name, vertSrc, fragSrc, expectError } = entry;
  const safeName = name.replace(/\//g, "-").replace(/[^a-zA-Z0-9_-]/g, "_");

  // Validate vertex + fragment. In production both must compile together; we
  // validate them independently because glslangValidator's -l (link) mode
  // requires matching in/out varyings which we do not need to enforce here —
  // we care about GLSL parse + type errors, not link-time interface matching.
  const vertResult = await validateGlsl("vert", vertSrc, `${safeName}-vert`);
  const fragResult = await validateGlsl("frag", fragSrc, `${safeName}-frag`);
  const ok = vertResult.ok && fragResult.ok;

  if (expectError) {
    // In self-test mode we expect this specific shader to fail.
    if (ok) {
      console.error(`[glsl-gate] --self-test FAIL: ${name} was expected to error but compiled OK.`);
    } else {
      console.log(`[glsl-gate] --self-test: ${name} correctly detected as broken.`);
    }
    errors.push({ name, ok, expectError });
    continue;
  }

  if (!ok) {
    failed++;
    const out = [vertResult.ok ? "" : `  [vert]\n${vertResult.output}`, fragResult.ok ? "" : `  [frag]\n${fragResult.output}`]
      .filter(Boolean)
      .join("\n");
    const errMsg = `[glsl-gate] FAIL  ${name}\n${out}`;
    console.error(errMsg);
    errors.push({ name, messages: out });
  } else {
    passed++;
    console.log(`[glsl-gate] OK    ${name}`);
  }
}

// ── Summary ───────────────────────────────────────────────────────────────────
const total = passed + failed;
console.log("");
console.log(`[glsl-gate] ${total} shader combination(s) compiled — ${passed} OK, ${failed} FAILED`);

if (selfTest) {
  const selfTestEntry = errors.find((e) => e.name === "__self-test/intentionally-broken");
  if (!selfTestEntry) {
    console.error("[glsl-gate] --self-test FAILED: injected broken shader not processed.");
    Deno.exit(1);
  }
  if (selfTestEntry.ok) {
    console.error("[glsl-gate] --self-test FAILED: injected broken shader was NOT detected!");
    Deno.exit(1);
  }
  const realFailures = errors.filter((e) => e.name !== "__self-test/intentionally-broken" && !e.expectError);
  if (realFailures.length > 0) {
    console.error("[glsl-gate] --self-test ABORTED: production shaders also failed:");
    for (const e of realFailures) console.error(`  ${e.name}`);
    Deno.exit(1);
  }
  console.log("[glsl-gate] --self-test PASSED: injected error was correctly detected.");
  Deno.exit(0);
}

if (failed > 0) {
  Deno.exit(1);
}
Deno.exit(0);
