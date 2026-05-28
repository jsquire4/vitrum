/**
 * uboCodegen.ts — Single-source-of-truth helper for WGSL UBO bindings.
 *
 * ─── Why this exists ─────────────────────────────────────────────────────────
 * Every WGSL uniform buffer in the codebase historically required FIVE hand-
 * mirrored declarations:
 *   1. A TypeScript docstring describing the std140/WGSL byte layout.
 *   2. A TypeScript `interface XxxUniforms` mirroring the field set.
 *   3. A `XXX_SIZE_BYTES` numeric constant.
 *   4. A `packXxxUniforms(view, offset, value)` function with hand-rolled
 *      `DataView.setUint32` / `DataView.setFloat32` calls at literal offsets.
 *   5. A `struct Xxx { ... }` declaration in the WGSL shader string.
 *
 * If any one of these five drifts (e.g., a developer adds a field to (2) but
 * forgets (5)), there is no compile-time error — only silent runtime GPU
 * corruption. The W2-C13 sweep finding identified ~20+ UBO instances in the
 * codebase, every one of them a tripwire.
 *
 * `defineUbo` generates all five from a single field-spec list and returns:
 *   - `sizeBytes`   — the computed total std140-aligned size
 *   - `pack(view, offset, value)`   — write fields → DataView
 *   - `unpack(view, offset)`        — read back fields ← DataView (round-trip)
 *   - `wgsl(structName, opts?)`     — emit the WGSL `struct` declaration
 *   - `fieldOffsets`                — map of name → byte offset (for tests
 *                                     and host-side scratch-buffer layout)
 *
 * The TypeScript value shape is inferred from the field-spec tuple via
 * conditional types, so consumers get full IntelliSense on `pack`/`unpack`
 * arguments with no manually-written interface.
 *
 * ─── WGSL uniform address-space layout rules (used here) ─────────────────────
 * WGSL `uniform` follows std140-style rules:
 *   - f32 / u32 / i32 : align  4, size  4
 *   - vec2<f32>       : align  8, size  8
 *   - vec3<f32>       : align 16, size 12  (pads to 16 in arrays/structs)
 *   - vec4<f32>       : align 16, size 16
 *   - mat4x4<f32>     : align 16, size 64
 * The struct's own size is rounded up to its largest member alignment, and
 * every uniform buffer binding must be ≥ 16 bytes — we enforce both.
 *
 * Reference: WGSL §14.4.4 Address-space layout constraints
 *   https://www.w3.org/TR/WGSL/#address-space-layout-constraints
 *
 * ─── Rollout status (W2-C13) ─────────────────────────────────────────────────
 * `defineUbo` is the canonical UBO codegen helper. Current adopters:
 *   - `packages/walkaround-hybrid/src/**` — ~8 UBOs migrated.
 *   - `packages/shared-denoisers/src/svgfRealBindings.ts`
 * NOT yet migrated (still hand-rolled DataView packers):
 *   - `packages/shared-denoisers/src/atrousVarianceBindings.ts`
 *   - `packages/pt-webgpu/src/index.ts` (FrameParams)
 *   - `packages/walkaround-rc/src/cascadeDispatch.ts` (Cascade/Merge UBOs;
 *     was at `walkaround-hybrid/src/rc/` pre-W8-followup, moved 2026-05-18)
 * Earlier revisions of this docstring claimed atrousVarianceBindings.ts was
 * the proof-of-concept adopter; that migration was planned but not landed.
 */

// ─── Field-type vocabulary ────────────────────────────────────────────────────

/**
 * Permitted UBO field types. Each maps to a fixed (align, size) pair under
 * the WGSL uniform layout rules.
 *
 * Reserved for follow-up: `vec2u`, `vec4u`, `mat3x3f`, `array<T, N>`. Add
 * them here when a real UBO needs them — don't speculate.
 */
export type UboFieldType =
  | 'f32'
  | 'u32'
  | 'i32'
  | 'vec2f'
  | 'vec3f'
  | 'vec4f'
  | 'mat4x4f';

interface LayoutInfo {
  readonly align: number;
  readonly size: number;
  readonly wgsl: string;
}

/**
 * Layout table for each field type.
 *
 * vec3f keeps `size: 12` but `align: 16` — the std140 "vec3 padded by a
 * trailing f32" rule. The next field starts at the next 16-byte boundary,
 * but the vec3 itself only consumes 12 written bytes; the 4 trailing bytes
 * are unwritten (driver leaves them at whatever the buffer was initialized
 * to, which for our use is zero-initialised scratch).
 */
const FIELD_LAYOUT: { readonly [K in UboFieldType]: LayoutInfo } = {
  f32:     { align:  4, size:  4, wgsl: 'f32'        },
  u32:     { align:  4, size:  4, wgsl: 'u32'        },
  i32:     { align:  4, size:  4, wgsl: 'i32'        },
  vec2f:   { align:  8, size:  8, wgsl: 'vec2<f32>'  },
  vec3f:   { align: 16, size: 12, wgsl: 'vec3<f32>'  },
  vec4f:   { align: 16, size: 16, wgsl: 'vec4<f32>'  },
  mat4x4f: { align: 16, size: 64, wgsl: 'mat4x4<f32>' },
} as const;

/** Minimum size of any WebGPU uniform-buffer binding (driver requirement). */
const MIN_UNIFORM_BUFFER_SIZE = 16;

// ─── Field-spec types ─────────────────────────────────────────────────────────

export interface UboFieldSpec<Name extends string = string, T extends UboFieldType = UboFieldType> {
  readonly name: Name;
  readonly type: T;
}

/**
 * Maps a single field spec to the runtime value type. Scalars are `number`;
 * vectors are fixed-length tuples; mat4x4f is a 16-element tuple in
 * column-major order (matching WGSL's storage convention).
 */
type FieldValue<T extends UboFieldType> =
  T extends 'f32' | 'u32' | 'i32' ? number :
  T extends 'vec2f' ? readonly [number, number] :
  T extends 'vec3f' ? readonly [number, number, number] :
  T extends 'vec4f' ? readonly [number, number, number, number] :
  T extends 'mat4x4f' ? readonly [
    number, number, number, number,
    number, number, number, number,
    number, number, number, number,
    number, number, number, number,
  ] :
  never;

/**
 * Object type derived from a tuple of field specs. Each spec contributes
 * a `[name]: FieldValue<type>` entry.
 *
 * Implementation note: we use `F[number]` and discriminate by `name` so the
 * value type narrows per-key. The simpler `keyof F as ...` form unions all
 * value types together when TypeScript can't statically prove each name is
 * unique, which produces a `number | tuple` for every field — breaking
 * IntelliSense on the consumer side.
 *
 * Example:
 *   const fields = [
 *     { name: 'iteration',  type: 'u32' },
 *     { name: 'sigmaColor', type: 'f32' },
 *   ] as const;
 *   type V = UboValue<typeof fields>;
 *   // = { iteration: number; sigmaColor: number }
 */
export type UboValue<F extends readonly UboFieldSpec[]> = {
  readonly [N in F[number]['name']]:
    Extract<F[number], { name: N }> extends UboFieldSpec<string, infer T>
      ? FieldValue<T>
      : never;
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function alignUp(offset: number, alignment: number): number {
  return Math.ceil(offset / alignment) * alignment;
}

interface ResolvedField {
  readonly name: string;
  readonly type: UboFieldType;
  readonly offset: number;
  readonly align: number;
  readonly size: number;
}

/**
 * Walk the field list, computing per-field offsets and the total struct size.
 *
 * The total size is rounded up to the largest member alignment, and then
 * bumped to at least MIN_UNIFORM_BUFFER_SIZE so the buffer is a valid
 * WebGPU uniform binding.
 */
function computeLayout(fields: readonly UboFieldSpec[]): {
  readonly resolved: readonly ResolvedField[];
  readonly sizeBytes: number;
} {
  if (fields.length === 0) {
    throw new Error('defineUbo: field list must not be empty');
  }
  const resolved: ResolvedField[] = [];
  let cursor = 0;
  let maxAlign = 0;
  const seen = new Set<string>();
  for (const spec of fields) {
    if (seen.has(spec.name)) {
      throw new Error(`defineUbo: duplicate field name "${spec.name}"`);
    }
    seen.add(spec.name);
    const layout = FIELD_LAYOUT[spec.type];
    if (!layout) {
      throw new Error(`defineUbo: unknown field type "${spec.type}" for field "${spec.name}"`);
    }
    const offset = alignUp(cursor, layout.align);
    resolved.push({
      name: spec.name,
      type: spec.type,
      offset,
      align: layout.align,
      size: layout.size,
    });
    cursor = offset + layout.size;
    if (layout.align > maxAlign) maxAlign = layout.align;
  }
  // Round struct size up to the max member alignment (std140 / WGSL §14.4.4).
  const structSize = alignUp(cursor, Math.max(maxAlign, 4));
  // WebGPU minimum uniform-buffer-binding size.
  const sizeBytes = Math.max(structSize, MIN_UNIFORM_BUFFER_SIZE);
  return { resolved, sizeBytes };
}

// ─── Public API ───────────────────────────────────────────────────────────────

/** Options accepted by the generated `wgsl()` method. */
export interface UboWgslOptions {
  /** Indentation string per line (default: two spaces). */
  readonly indent?: string;
  /**
   * If true, emit `_padN: u32` filler fields for std140 gaps that arise
   * when, e.g., a vec3 is followed by another scalar. Most callers don't
   * need these (WGSL fills them implicitly) but explicit padding helps
   * cross-language ports. Default: false.
   */
  readonly explicitPadding?: boolean;
}

export interface UboDefinition<F extends readonly UboFieldSpec[]> {
  /** Total byte size of the UBO (≥ 16, rounded up to max member alignment). */
  readonly sizeBytes: number;
  /** Map of field name → byte offset, for callers that need raw offsets. */
  readonly fieldOffsets: { readonly [K in keyof UboValue<F>]: number };
  /**
   * Pack a value object into a DataView at the given byte offset.
   *
   * The view's buffer must contain at least `offset + sizeBytes` bytes.
   * Bytes that fall in std140 padding regions are explicitly zero-filled
   * for deterministic output.
   */
  readonly pack: (view: DataView, offset: number, value: UboValue<F>) => void;
  /**
   * Read a value object back from a DataView at the given byte offset.
   *
   * `pack` followed by `unpack` is a round-trip identity (modulo
   * float-precision saturation for non-finite inputs, which callers
   * shouldn't be writing to a uniform anyway).
   */
  readonly unpack: (view: DataView, offset: number) => UboValue<F>;
  /**
   * Emit the WGSL `struct` declaration for this UBO.
   *
   * @param structName - the WGSL struct name (e.g., `'AtrousVarianceAtrousUBO'`).
   * @param opts       - optional indent / explicit-padding overrides.
   */
  readonly wgsl: (structName: string, opts?: UboWgslOptions) => string;
}

/**
 * Construct a UBO definition from a field-spec list.
 *
 * The returned object is the single source of truth for both the host-side
 * packer and the shader-side struct. There is no longer a way to have one
 * drift from the other without producing a TypeScript compile error.
 *
 * @example
 *   const VarianceUBO = defineUbo([
 *     { name: 'frameCount', type: 'u32' },
 *   ] as const);
 *   // → sizeBytes 16 (padded up from 4 to the min uniform binding size)
 *   // → pack({ frameCount: 3 })   writes u32 LE at offset 0
 *   // → wgsl('VarianceUBO')       returns 'struct VarianceUBO { frameCount: u32, };'
 */
export function defineUbo<const F extends readonly UboFieldSpec[]>(fields: F): UboDefinition<F> {
  const { resolved, sizeBytes } = computeLayout(fields);

  const fieldOffsets = Object.fromEntries(resolved.map((f) => [f.name, f.offset])) as {
    readonly [K in keyof UboValue<F>]: number;
  };

  function pack(view: DataView, offset: number, value: UboValue<F>): void {
    if (offset < 0 || offset + sizeBytes > view.byteLength) {
      throw new RangeError(
        `defineUbo.pack: offset ${offset} + sizeBytes ${sizeBytes} exceeds view.byteLength ${view.byteLength}`,
      );
    }
    // Zero-fill the destination region first so std140 padding bytes are
    // deterministic regardless of buffer history.
    new Uint8Array(view.buffer, view.byteOffset + offset, sizeBytes).fill(0);
    for (const f of resolved) {
      const v = (value as unknown as Record<string, unknown>)[f.name];
      const base = offset + f.offset;
      switch (f.type) {
        case 'f32':
          view.setFloat32(base, v as number, true);
          break;
        case 'u32':
          view.setUint32(base, (v as number) >>> 0, true);
          break;
        case 'i32':
          view.setInt32(base, (v as number) | 0, true);
          break;
        case 'vec2f': {
          const tup = v as readonly number[];
          view.setFloat32(base + 0, tup[0]!, true);
          view.setFloat32(base + 4, tup[1]!, true);
          break;
        }
        case 'vec3f': {
          const tup = v as readonly number[];
          view.setFloat32(base + 0, tup[0]!, true);
          view.setFloat32(base + 4, tup[1]!, true);
          view.setFloat32(base + 8, tup[2]!, true);
          // bytes [base+12, base+16) are std140 pad — left at zero by the fill above.
          break;
        }
        case 'vec4f': {
          const tup = v as readonly number[];
          view.setFloat32(base +  0, tup[0]!, true);
          view.setFloat32(base +  4, tup[1]!, true);
          view.setFloat32(base +  8, tup[2]!, true);
          view.setFloat32(base + 12, tup[3]!, true);
          break;
        }
        case 'mat4x4f': {
          const tup = v as readonly number[];
          for (let i = 0; i < 16; i++) view.setFloat32(base + i * 4, tup[i]!, true);
          break;
        }
        default: {
          // Exhaustiveness guard — TS will narrow to `never` here.
          const _exhaustive: never = f.type;
          throw new Error(`defineUbo.pack: unhandled field type ${_exhaustive as string}`);
        }
      }
    }
  }

  function unpack(view: DataView, offset: number): UboValue<F> {
    if (offset < 0 || offset + sizeBytes > view.byteLength) {
      throw new RangeError(
        `defineUbo.unpack: offset ${offset} + sizeBytes ${sizeBytes} exceeds view.byteLength ${view.byteLength}`,
      );
    }
    const out: Record<string, unknown> = {};
    for (const f of resolved) {
      const base = offset + f.offset;
      switch (f.type) {
        case 'f32': out[f.name] = view.getFloat32(base, true); break;
        case 'u32': out[f.name] = view.getUint32(base, true); break;
        case 'i32': out[f.name] = view.getInt32(base, true); break;
        case 'vec2f': out[f.name] = [
          view.getFloat32(base + 0, true), view.getFloat32(base + 4, true),
        ] as const; break;
        case 'vec3f': out[f.name] = [
          view.getFloat32(base + 0, true), view.getFloat32(base + 4, true),
          view.getFloat32(base + 8, true),
        ] as const; break;
        case 'vec4f': out[f.name] = [
          view.getFloat32(base +  0, true), view.getFloat32(base +  4, true),
          view.getFloat32(base +  8, true), view.getFloat32(base + 12, true),
        ] as const; break;
        case 'mat4x4f': {
          const m: number[] = new Array(16);
          for (let i = 0; i < 16; i++) m[i] = view.getFloat32(base + i * 4, true);
          out[f.name] = m;
          break;
        }
        default: {
          const _exhaustive: never = f.type;
          throw new Error(`defineUbo.unpack: unhandled field type ${_exhaustive as string}`);
        }
      }
    }
    return out as UboValue<F>;
  }

  function wgsl(structName: string, opts: UboWgslOptions = {}): string {
    const indent = opts.indent ?? '  ';
    const explicitPadding = opts.explicitPadding ?? false;
    const lines: string[] = [`struct ${structName} {`];
    let padId = 0;
    let writtenCursor = 0;
    for (const f of resolved) {
      if (explicitPadding && f.offset > writtenCursor) {
        // Emit a u32 pad per 4 bytes of gap.
        const gapBytes = f.offset - writtenCursor;
        const pads = Math.floor(gapBytes / 4);
        for (let i = 0; i < pads; i++) {
          lines.push(`${indent}_pad${padId++}: u32,`);
        }
      }
      lines.push(`${indent}${f.name}: ${FIELD_LAYOUT[f.type].wgsl},`);
      writtenCursor = f.offset + f.size;
    }
    if (explicitPadding && writtenCursor < sizeBytes) {
      const gapBytes = sizeBytes - writtenCursor;
      const pads = Math.floor(gapBytes / 4);
      for (let i = 0; i < pads; i++) {
        lines.push(`${indent}_pad${padId++}: u32,`);
      }
    }
    lines.push('};');
    return lines.join('\n');
  }

  return { sizeBytes, fieldOffsets, pack, unpack, wgsl };
}
