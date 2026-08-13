import { describe, expect, it } from 'vitest';
import { composePtWebgpuTraceWgsl } from '../wgsl/pathTraceBruteforce.wgsl.js';
import { composePtWebgpuTraceLiteWgsl } from '../wgsl/pathTraceBruteforceLite.wgsl.js';

/**
 * INLINE BUDGET GATE
 *
 * Mesa's NIR (lavapipe AND dzn — both are Mesa, and both are what WebGPU on this
 * project's Linux/WSL runners actually uses) fully inlines every function call
 * before it optimises. The quantity a driver has to chew through is therefore NOT
 * the WGSL source size but the size of `main` after every call site is expanded:
 *
 *     inlined(f) = own(f) + SUM over each call site to g of inlined(g)
 *
 * That MULTIPLIES along the call graph. A helper called from 6 places, itself
 * calling something 4 times, contributes 24 copies — so adding one call edge in a
 * hot function can add megabytes of post-inline code while the source grows by a
 * few hundred bytes.
 *
 * This is not hypothetical. Between 2026-07-20 and 2026-07-28 the full-tier kernel
 * grew 408K -> 582K of source (+43%) while its inlined size went 15.2M -> well over
 * 100M. Pipeline creation stopped converging: it consumed the entire WSL VM (30GB
 * + 8GB swap), the kernel OOM-killer escalated to init.scope, and the distro died.
 * `traceTier: 'full'` could not build a pipeline at all for roughly six weeks, and
 * nothing in CI noticed, because every source-level metric still looked reasonable.
 *
 * This gate is cheap (pure string analysis, no GPU, milliseconds) and is the only
 * check that tracks the quantity that actually breaks.
 *
 * IF THIS FAILS: do not simply raise the ceiling. Find the new duplicate call
 * sites and collapse them — a repeated call with identical or near-identical
 * arguments becomes a short loop, which NIR inlines ONCE instead of N times. See
 * the INLINE-BUDGET comments in material.wgsl.ts / bsdf.wgsl.ts for worked
 * examples. Raising the ceiling without understanding why is how the renderer
 * became uncompilable last time.
 */

const WGSL_BUILTINS = new Set([
  'vec2', 'vec3', 'vec4', 'vec2f', 'vec3f', 'vec4f', 'vec2i', 'vec3i', 'vec4i',
  'vec2u', 'vec3u', 'vec4u', 'mat2x2', 'mat3x3', 'mat4x4', 'array', 'f32', 'i32',
  'u32', 'bool', 'abs', 'min', 'max', 'clamp', 'mix', 'step', 'smoothstep', 'sqrt',
  'inverseSqrt', 'pow', 'exp', 'exp2', 'log', 'log2', 'sin', 'cos', 'tan', 'asin',
  'acos', 'atan', 'atan2', 'floor', 'ceil', 'round', 'trunc', 'fract', 'modf',
  'sign', 'dot', 'cross', 'normalize', 'length', 'distance', 'reflect', 'refract',
  'faceForward', 'determinant', 'transpose', 'select', 'all', 'any', 'saturate',
  'textureSample', 'textureLoad', 'textureStore', 'textureDimensions',
  'textureSampleLevel', 'textureNumLayers', 'textureNumLevels', 'atomicAdd',
  'atomicLoad', 'atomicStore', 'atomicMax', 'atomicMin', 'atomicExchange',
  'workgroupBarrier', 'storageBarrier', 'bitcast', 'countOneBits', 'reverseBits',
  'fma', 'unpack4x8unorm', 'pack4x8unorm', 'unpack2x16float', 'pack2x16float',
  'firstLeadingBit', 'firstTrailingBit', 'extractBits', 'insertBits', 'ldexp',
  'frexp', 'arrayLength', 'degrees', 'radians', 'sinh', 'cosh', 'tanh', 'if',
  'for', 'while', 'switch', 'return', 'let', 'var', 'const', 'struct', 'fn',
  'loop', 'break', 'continue', 'discard',
]);

interface Fn {
  own: number;
  calls: Map<string, number>;
}

function parseFunctions(src: string): Map<string, Fn> {
  const bodies = new Map<string, { text: string; own: number }>();
  const re = /^fn\s+([A-Za-z0-9_]+)\s*\(/gm;
  let m: RegExpExecArray | null;
  while ((m = re.exec(src)) !== null) {
    const start = m.index;
    let index = src.indexOf('{', start);
    if (index < 0) continue;
    let depth = 0;
    for (; index < src.length; index += 1) {
      if (src[index] === '{') depth += 1;
      else if (src[index] === '}') {
        depth -= 1;
        if (depth === 0) {
          index += 1;
          break;
        }
      }
    }
    // A name may be declared more than once across mutually exclusive
    // compositions; the last definition in the composed string is the live one.
    bodies.set(m[1], { text: src.slice(start, index), own: index - start });
  }
  const names = new Set(bodies.keys());
  const fns = new Map<string, Fn>();
  for (const [name, body] of bodies) {
    const calls = new Map<string, number>();
    const inner = body.text.slice(body.text.indexOf('{'));
    const callRe = /\b([A-Za-z_][A-Za-z0-9_]*)\s*\(/g;
    let c: RegExpExecArray | null;
    while ((c = callRe.exec(inner)) !== null) {
      const callee = c[1];
      if (!names.has(callee) || WGSL_BUILTINS.has(callee)) continue;
      calls.set(callee, (calls.get(callee) ?? 0) + 1);
    }
    fns.set(name, { own: body.own, calls });
  }
  return fns;
}

export function inlinedEntryPointSize(src: string, entry = 'main'): number {
  const fns = parseFunctions(src);
  const memo = new Map<string, number>();
  const visit = (name: string): number => {
    const cached = memo.get(name);
    if (cached !== undefined) return cached;
    const fn = fns.get(name);
    if (!fn) return 0;
    // Seed with own size first: WGSL forbids recursion, but a defensive seed
    // keeps a malformed graph from hanging the suite.
    memo.set(name, fn.own);
    let total = fn.own;
    for (const [callee, multiplicity] of fn.calls) {
      total += multiplicity * visit(callee);
    }
    memo.set(name, total);
    return total;
  };
  return visit(entry);
}

// Ceilings sit above today's measured values with headroom for ordinary feature
// work, and far below the ~100M+ that stopped converging. Full tier measured
// 28.0M and builds a real pipeline in ~4 minutes at a 10GB cap.
const FULL_TIER_INLINE_CEILING = 36_000_000;
const LITE_TIER_INLINE_CEILING = 20_000_000;

describe('pt-webgpu inline budget (NIR full-inlining gate)', () => {
  it('keeps the full-tier kernel inside its post-inline budget', () => {
    const inlined = inlinedEntryPointSize(composePtWebgpuTraceWgsl(false, {}));
    expect(inlined).toBeGreaterThan(0);
    expect(inlined).toBeLessThan(FULL_TIER_INLINE_CEILING);
  });

  it('keeps the BDPT-on full-tier variant inside a bounded budget', () => {
    const inlined = inlinedEntryPointSize(composePtWebgpuTraceWgsl(true, {}));
    expect(inlined).toBeGreaterThan(0);
    // BDPT composes an additional estimator, so it carries its own larger ceiling.
    expect(inlined).toBeLessThan(60_000_000);
  });

  it('keeps the lite-tier kernel inside its post-inline budget', () => {
    const inlined = inlinedEntryPointSize(composePtWebgpuTraceLiteWgsl({}));
    expect(inlined).toBeGreaterThan(0);
    expect(inlined).toBeLessThan(LITE_TIER_INLINE_CEILING);
  });

  /**
   * KNOWN BLIND SPOT of the inline metric above: it sums FUNCTION bodies only, so
   * module-scope constant data is invisible to it. That is not academic — the
   * 'sobol' sampling variant embeds a 4096-entry direction-number table as a
   * module-scope `const array<u32, 4096>`. It measures ~28M inlined (about the
   * same as the default PCG variant, which builds fine) yet still exhausted a 14GB
   * budget during pipeline creation, because materialising a constant table of
   * that size is its own cost entirely separate from call-site inlining.
   *
   * This ratchet does not shrink the existing table; it stops a BIGGER one landing
   * unnoticed. A table beyond this size almost certainly belongs in a storage
   * buffer rather than in the shader source.
   */
  it('does not grow module-scope constant tables beyond the current worst case', () => {
    const sources = [
      composePtWebgpuTraceWgsl(false, {}),
      composePtWebgpuTraceWgsl(false, { sampling: 'sobol' }),
    ];
    for (const src of sources) {
      const sizes = [...src.matchAll(/array\s*<\s*[^,<>]+,\s*(\d+)\s*>/g)].map((m) =>
        Number(m[1]),
      );
      const largest = sizes.length > 0 ? Math.max(...sizes) : 0;
      expect(largest).toBeLessThanOrEqual(4096);
    }
  });

  it('reports the dominant duplicated call edges when a budget is exceeded', () => {
    // Self-check of the analyser: main must expand far beyond its own body, or the
    // parser has silently stopped finding call edges and the gate is vacuous.
    const src = composePtWebgpuTraceWgsl(false, {});
    const inlined = inlinedEntryPointSize(src);
    const ownMain = /^fn main\s*\(|@compute[\s\S]*?fn main\s*\(/m.test(src);
    expect(ownMain).toBe(true);
    expect(inlined).toBeGreaterThan(1_000_000);
  });
});
