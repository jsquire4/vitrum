/**
 * tools/shader-gate/nagaFix.mjs
 *
 * Minimal port of the naga/wgpu-native WGSL compatibility transforms from
 * ~/projects/wsl-gpu/capture-worker/three-headless-shim/wgsl-naga-gap-fix.ts
 *
 * WHY THIS EXISTS: naga (the Rust WGSL compiler used by wgpu/WebGPU on
 * Vulkan/lavapipe/dzn) rejects `ptr<storage, T, access>` as function
 * parameters unless the non-core WGSL feature `unrestricted_pointer_parameters`
 * is enabled. WebGPU does not expose that feature flag. vitrum's walkaround-
 * hybrid shaders use shared-bvh's traversal helpers which take the BVH storage
 * buffers as ptr<storage> params — legal in Dawn/Tint (Chrome) but rejected by
 * naga. The wsl-gpu T1 smoke applies this fix at runtime before compilation;
 * the shader gate applies it here for the same reason.
 *
 * The fix is mechanical (zero semantic change):
 *   1. Alpha-rename GRIS-prefixed scene globals (sgi_/tgi_ → canonical names)
 *      so the post-strip traversal bodies resolve.
 *   2. Remove BVH traversal function definitions from shaders that lack BVH
 *      storage bindings (dead code in atrous/motionVectors/etc.).
 *   3. Remove dead V3-variant BVH functions (different storage layout, never
 *      called by current walkaround-hybrid shaders).
 *   4. Strip ptr<storage> parameters from affected function signatures.
 *   5. Replace (*name)[] pointer dereferences with name[] direct access.
 *   6. Remove &name / forwarded-name arguments at call sites.
 *   7. Fix WGSL reserved keyword 'target' → 'tgt' (Naga treats it as reserved).
 *   8. Fix inter-stage struct references (comp-frag references CompositeVaryings
 *      defined in comp-vert; inject the struct into the fragment shader).
 *
 * The source of truth for this logic is the wsl-gpu shim at:
 *   ~/projects/wsl-gpu/capture-worker/three-headless-shim/wgsl-naga-gap-fix.ts
 * Keep the two in sync when new BVH binding names or storage patterns appear.
 */

/** Storage binding names that vitrum passes as ptr<storage> function args. */
const BVH_BINDING_NAMES = [
  "bvh_index",
  "bvh_position",
  "bvh",
  "tlasNodes",
  "tlasInstanceIndices",
  "tlasBlasRoots",
  "tlasInstanceWorldToLocal",
  "tlasInstanceLocalToWorld",
];

/** GRIS per-pass prefixes used in spatialGiGris / temporalGiGris. */
const GRIS_SCENE_PREFIXES = ["sgi_", "tgi_"];

/** RC probe-ray per-pass renames (rc_bvh → bvh etc.). */
const RC_SCENE_GLOBAL_RENAMES = [
  ["rc_geom_position", "bvh_position"],
  ["rc_geom_index", "bvh_index"],
  ["rc_bvh", "bvh"],
];

// ── helpers ──────────────────────────────────────────────────────────────────

function findMatchingBrace(src, openIdx) {
  let depth = 1;
  let i = openIdx + 1;
  while (i < src.length && depth > 0) {
    if (src[i] === "{") depth++;
    else if (src[i] === "}") depth--;
    i++;
  }
  return depth === 0 ? i : -1;
}

function locateFunctionSpan(src, funcName) {
  const fnIdx = src.indexOf(`fn ${funcName}(`);
  if (fnIdx === -1) return null;
  const openBrace = src.indexOf("{", fnIdx);
  if (openBrace === -1) return null;
  const end = findMatchingBrace(src, openBrace);
  if (end === -1) return null;
  // Walk back over comment/blank lines before the function
  const before = src.slice(0, fnIdx);
  const lines = before.split("\n");
  let lineIdx = lines.length - 1;
  while (lineIdx >= 0) {
    const line = lines[lineIdx].trimStart();
    if (line.startsWith("//") || line === "") lineIdx--;
    else break;
  }
  const commentStart = lines.slice(0, lineIdx + 1).join("\n").length + 1;
  return { commentStart, openBrace, end };
}

function removeFunctionBody(wgsl, funcName) {
  const span = locateFunctionSpan(wgsl, funcName);
  if (span === null) return wgsl;
  return wgsl.slice(0, span.commentStart) + wgsl.slice(span.end);
}

// ── transform steps ───────────────────────────────────────────────────────────

function renameGrisPrefixedSceneGlobals(wgsl) {
  let result = wgsl;
  let count = 0;
  for (const prefix of GRIS_SCENE_PREFIXES) {
    const declaresPrefixed = BVH_BINDING_NAMES.some((n) =>
      new RegExp(`var<storage[^>]*>\\s+${prefix}${n}\\b`).test(result)
    );
    if (!declaresPrefixed) continue;
    // Sort longest-first so e.g. sgi_bvh_index is renamed before sgi_bvh
    const sorted = [...BVH_BINDING_NAMES].sort((a, b) => b.length - a.length);
    for (const name of sorted) {
      const pat = new RegExp(`\\b${prefix}${name}\\b`, "g");
      const before = result;
      result = result.replace(pat, name);
      if (result !== before) count++;
    }
  }
  return { result, count };
}

function renameRcPrefixedSceneGlobals(wgsl) {
  let result = wgsl;
  let count = 0;
  if (!/var<storage[^>]*>\s+rc_bvh\b/.test(result)) return { result, count };
  const ordered = [...RC_SCENE_GLOBAL_RENAMES].sort((a, b) => b[0].length - a[0].length);
  for (const [from, to] of ordered) {
    const pat = new RegExp(`\\b${from}\\b`, "g");
    const before = result;
    result = result.replace(pat, to);
    if (result !== before) count++;
  }
  return { result, count };
}

function removeBvhFunctionsIfNoBindings(wgsl) {
  const hasBvhBindings =
    (/var<storage[^>]*>\s+\w*bvh/.test(wgsl) ||
      /var<storage[^>]*>\s+\w*tlasNodes/.test(wgsl)) &&
    wgsl.includes("BVHNode");
  if (hasBvhBindings) return { result: wgsl, count: 0 };

  const hasBvhFunctions =
    wgsl.includes("fn bvhIntersectFirstHit(") ||
    wgsl.includes("fn bvhIntersectFirstHitAtRoot(") ||
    wgsl.includes("fn bvhIntersectAny(") ||
    wgsl.includes("fn traceTlasFirstHit(") ||
    wgsl.includes("fn traceTlasAny(") ||
    wgsl.includes("fn traceSceneFirstHit(") ||
    wgsl.includes("fn traceSceneAny(");
  if (!hasBvhFunctions) return { result: wgsl, count: 0 };

  let result = wgsl;
  let count = 0;
  const toRemove = [
    "bvhIntersectFirstHit",
    "bvhIntersectFirstHitAtRoot",
    "bvhIntersectAny",
    "bvhIntersectAnyAtRoot",
    "bvhIntersectFirstHitV3",
    "traceTlasFirstHit",
    "traceTlasAny",
    "traceSceneFirstHit",
    "traceSceneAny",
    "bvhTraceTintedVisibility",
  ];
  for (const funcName of toRemove) {
    const before = result;
    result = removeFunctionBody(result, funcName);
    if (result !== before) count++;
  }
  return { result, count };
}

function removeV3Functions(wgsl) {
  let result = wgsl;
  let count = 0;
  for (const funcName of ["bvhIntersectFirstHitV3", "bvhIntersectAnyV3"]) {
    const span = locateFunctionSpan(result, funcName);
    if (span === null) continue;
    result = result.slice(0, span.commentStart) + result.slice(span.end);
    count++;
  }
  return { result, count };
}

function removePtrStorageParams(wgsl) {
  let result = wgsl;
  let count = 0;
  for (const name of BVH_BINDING_NAMES) {
    // Pattern A: parameter is NOT last (has trailing comma)
    const patA = new RegExp(
      `\\b${name}\\s*:\\s*ptr<storage,\\s*(?:[^<>]*(?:<[^<>]*>)*[^<>]*),\\s*(?:read|read_write)>\\s*,\\s*`,
      "g",
    );
    const before = result;
    result = result.replace(patA, "");
    if (result !== before) { count++; continue; }
    // Pattern B: parameter IS last (preceded by comma)
    const patB = new RegExp(
      `,\\s*\\b${name}\\s*:\\s*ptr<storage,\\s*(?:[^<>]*(?:<[^<>]*>)*[^<>]*),\\s*(?:read|read_write)>`,
      "g",
    );
    const beforeB = result;
    result = result.replace(patB, () => { count++; return ""; });
    if (result !== beforeB) count++;
  }
  return { result, count };
}

function replacePtrDerefs(wgsl) {
  let result = wgsl;
  let count = 0;
  for (const name of BVH_BINDING_NAMES) {
    const patBracket = new RegExp(`\\(\\*${name}\\)\\[`, "g");
    const before = result;
    result = result.replace(patBracket, `${name}[`);
    if (result !== before) count++;
    const patDot = new RegExp(`\\(\\*${name}\\)\\.`, "g");
    const before2 = result;
    result = result.replace(patDot, `${name}.`);
    if (result !== before2) count++;
  }
  return { result, count };
}

function removeStorageArgs(wgsl) {
  let result = wgsl;
  let count = 0;
  for (const name of BVH_BINDING_NAMES) {
    // &name, (not last arg)
    const caseA = new RegExp(`&${name}\\s*,\\s*`, "g");
    const beforeA = result;
    result = result.replace(caseA, "");
    if (result !== beforeA) count++;
    // , &name (last arg)
    const caseB = new RegExp(`,\\s*&${name}\\b`, "g");
    const beforeB = result;
    result = result.replace(caseB, "");
    if (result !== beforeB) count++;
    // forwarded without &, not-last arg
    const caseC = new RegExp(`\\b${name}(?!\\s*[.\\[])\\s*,\\s*`, "g");
    const beforeC = result;
    result = result.replace(caseC, "");
    if (result !== beforeC) count++;
    // forwarded without &, last arg
    const caseD = new RegExp(`,\\s*\\b${name}(?!\\s*[.\\[])\\b`, "g");
    const beforeD = result;
    result = result.replace(caseD, "");
    if (result !== beforeD) count++;
  }
  return { result, count };
}

/**
 * Specialize generic `buf: ptr<storage, array<T>, ...>` reservoir/cdf functions
 * by creating per-binding specialized copies and replacing call sites.
 *
 * After the BVH ptr<storage> params are stripped in Steps 1-3, the remaining
 * ptr<storage, array<u32>> reservoir helpers (loadReservoirDI_rw,
 * storeReservoirDI_rw, etc.) still fail naga.  This function:
 *   1. Finds every function whose first param is ptr<storage, array<T>>.
 *   2. For each, finds all call-site binding names (from &bindingName args).
 *   3. Emits a specialized copy per binding with the ptr param removed and the
 *      param name → binding name in the body.
 *   4. Removes the original generic function.
 *   5. Rewrites call sites to use the specialized name.
 *
 * Port of specializeReservoirFunctions from wsl-gpu naga-gap-fix.ts.
 */
function specializeReservoirFunctions(wgsl) {
  let result = wgsl;
  let count = 0;

  // Find all functions whose first param is ptr<storage, array<T>, read|read_write>
  const funcPat =
    /fn\s+(\w+)\s*\(\s*(\w+)\s*:\s*ptr<storage,\s*array<[^>]+>,\s*(?:read_write|read)\s*>/g;

  const funcsToPatch = [];
  const wgslCopy = result;
  let match;
  while ((match = funcPat.exec(wgslCopy)) !== null) {
    const paramName = match[2];
    if (!BVH_BINDING_NAMES.includes(paramName)) {
      funcsToPatch.push({ funcName: match[1], paramName });
    }
  }

  for (const { funcName, paramName } of funcsToPatch) {
    // Find unique call-site binding names: funcName(&bindingName, ...)
    const callPat = new RegExp(`\\b${funcName}\\s*\\(&(\\w+)`, "g");
    const uniqueBindings = new Set();
    const resultCopy = result;
    let callMatch;
    while ((callMatch = callPat.exec(resultCopy)) !== null) {
      uniqueBindings.add(callMatch[1]);
    }

    if (uniqueBindings.size === 0) {
      // Dead code — no call sites.  Remove the function so naga doesn't see it.
      const fnIdxDead = result.indexOf(`fn ${funcName}(`);
      if (fnIdxDead === -1) continue;
      const openBraceDead = result.indexOf("{", fnIdxDead);
      if (openBraceDead === -1) continue;
      const endDead = findMatchingBrace(result, openBraceDead);
      if (endDead === -1) continue;
      result = result.slice(0, fnIdxDead) + result.slice(endDead);
      count++;
      continue;
    }

    // Extract the original function
    const fnIdx = result.indexOf(`fn ${funcName}(`);
    if (fnIdx === -1) continue;
    const openBrace = result.indexOf("{", fnIdx);
    if (openBrace === -1) continue;
    const fnSig = result.slice(fnIdx, openBrace);
    const fnEnd = findMatchingBrace(result, openBrace);
    if (fnEnd === -1) continue;
    const fnBody = result.slice(openBrace, fnEnd);

    // Build specialized copies
    const paramEscaped = paramName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    for (const bindingName of uniqueBindings) {
      const specialName = `${funcName}_${bindingName}`;
      let newSig = fnSig
        .replace(
          new RegExp(
            `${paramEscaped}\\s*:\\s*ptr<storage,\\s*array<[^>]+>,\\s*(?:read_write|read)\\s*>\\s*,?\\s*`,
          ),
          "",
        )
        .replace(/fn\s+\w+\s*\(/, `fn ${specialName}(`);
      const newBody = fnBody
        .replace(new RegExp(`\\(\\*${paramEscaped}\\)\\[`, "g"), `${bindingName}[`)
        .replace(new RegExp(`\\b${paramEscaped}\\[`, "g"), `${bindingName}[`);
      result = result.slice(0, fnIdx) + `${newSig}${newBody}\n\n` + result.slice(fnIdx);
      count++;
    }

    // Remove original generic function (now after the inserted specialized copies)
    const origFnIdx = result.lastIndexOf(`fn ${funcName}(`);
    if (origFnIdx === -1) continue;
    const origOpenBrace = result.indexOf("{", origFnIdx);
    if (origOpenBrace === -1) continue;
    const origEnd = findMatchingBrace(result, origOpenBrace);
    if (origEnd === -1) continue;
    result = result.slice(0, origFnIdx) + result.slice(origEnd);

    // Rewrite call sites
    for (const bindingName of uniqueBindings) {
      const specialName = `${funcName}_${bindingName}`;
      result = result.replace(
        new RegExp(`\\b${funcName}\\s*\\(\\s*&${bindingName}\\s*,\\s*`, "g"),
        `${specialName}(`,
      );
      result = result.replace(
        new RegExp(`\\b${funcName}\\s*\\(\\s*&${bindingName}\\s*\\)`, "g"),
        `${specialName}()`,
      );
    }
  }

  return { result, count };
}

/**
 * Fix arrayLength() calls: `arrayLength(name)` → `arrayLength(&name)` for
 * global var<storage> variables. Naga requires the & for global vars in arrayLength.
 */
function fixArrayLength(wgsl) {
  let result = wgsl;
  let count = 0;
  for (const name of BVH_BINDING_NAMES) {
    // Match arrayLength(name) where name is NOT already preceded by &
    const pat = new RegExp(`arrayLength\\((?!&)${name}\\)`, "g");
    const before = result;
    result = result.replace(pat, `arrayLength(&${name})`);
    if (result !== before) count++;
  }
  return { result, count };
}

function fixReservedKeywords(wgsl) {
  let result = wgsl;
  let count = 0;
  const letPat = /\blet\s+target\s*=/g;
  if (letPat.test(result)) {
    result = result.replace(/\blet\s+target\s*=/g, "let tgt =");
    result = result.replace(/\btarget\b/g, "tgt");
    count++;
  }
  return { result, count };
}

function fixInterStageStructs(wgsl) {
  let result = wgsl;
  let count = 0;
  if (result.includes("CompositeVaryings") && !result.includes("struct CompositeVaryings")) {
    const injection = [
      "// [naga-fix] Injected struct for inter-stage compatibility",
      "struct CompositeVaryings {",
      "  @builtin(position) clip: vec4f,",
      "  @location(0) uv: vec2f,",
      "}",
      "",
    ].join("\n");
    result = injection + result;
    count++;
  }
  return { result, count };
}

// ── main export ──────────────────────────────────────────────────────────────

/**
 * Apply all naga/wgpu-native WGSL compatibility transforms to a composed WGSL
 * string.  Returns the patched string and a brief log of what was changed.
 *
 * This is a port of `applyNagaStoragePtrFix` from:
 *   ~/projects/wsl-gpu/capture-worker/three-headless-shim/wgsl-naga-gap-fix.ts
 *
 * The transforms are purely mechanical (no semantic change): they strip
 * ptr<storage> function parameters and rewrite the bodies + call sites to use
 * the module-scope globals directly — which naga allows, unlike ptr<storage>
 * params (the WGSL `unrestricted_pointer_parameters` extension not exposed by
 * WebGPU).
 */
export function applyNagaFix(wgsl) {
  let fixed = wgsl;

  // Step -3: RC probe-ray scene-global rename
  fixed = renameRcPrefixedSceneGlobals(fixed).result;
  // Step -2: GRIS prefix-rename (sgi_/tgi_ → canonical)
  fixed = renameGrisPrefixedSceneGlobals(fixed).result;
  // Step -1: Remove BVH traversal functions from shaders without BVH bindings
  fixed = removeBvhFunctionsIfNoBindings(fixed).result;
  // Step 0: Remove dead V3 BVH functions
  fixed = removeV3Functions(fixed).result;
  // Step 1: Strip ptr<storage> parameters from function signatures
  fixed = removePtrStorageParams(fixed).result;
  // Step 2: Replace (*name)[] pointer dereferences with name[]
  fixed = replacePtrDerefs(fixed).result;
  // Step 3: Remove &name / forwarded-name arguments at call sites
  fixed = removeStorageArgs(fixed).result;
  // Step 3b: Specialize reservoir/cdf functions with ptr<storage, array<T>> params
  fixed = specializeReservoirFunctions(fixed).result;
  // Step 3a: Fix arrayLength(name) → arrayLength(&name)
  fixed = fixArrayLength(fixed).result;
  // Step 3c-a: Fix reserved keyword 'target' → 'tgt'
  fixed = fixReservedKeywords(fixed).result;
  // Step 3c: Fix inter-stage struct references (comp-frag)
  fixed = fixInterStageStructs(fixed).result;

  return fixed;
}
