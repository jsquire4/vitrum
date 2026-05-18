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
 *   - vec2<f32|u32>   : align  8, size  8
 *   - vec3<f32|u32>   : align 16, size 12  (pads to 16 in arrays/structs)
 *   - vec4<f32|u32>   : align 16, size 16
 *   - mat4x4<f32>     : align 16, size 64
 *   - array<T, N>     : align 16, stride = roundUp(sizeof(T), 16)
 * The struct's own size is rounded up to its largest member alignment, and
 * every uniform buffer binding must be ≥ 16 bytes — we enforce both.
 *
 * Reference: WGSL §14.4.4 Address-space layout constraints
 *   https://www.w3.org/TR/WGSL/#address-space-layout-constraints
 *
 * ─── W2-C13 vocabulary extension ─────────────────────────────────────────────
 * The vocabulary now includes `vec2u`, `vec3u`, `vec4u`, and
 * `array<struct, N>`. These unblock the remaining hand-rolled packers
 * identified by the W2-C13 follow-up sweep:
 *   - WalkaroundUBO (304 B) — uses `vec2u screenSize`.
 *   - DDGILightUniforms (1028 B padded) — array<DDGILight, 16>.
 *   - walkaround-rc MergeUniforms / CascadeUniforms — use `vec3u` (these
 *     are read-only-storage bindings, but the layout helper still applies
 *     for the host packers when those packers migrate).
 */

// ─── Field-type vocabulary ────────────────────────────────────────────────────

/**
 * Permitted scalar / vector / matrix UBO field types. Each maps to a fixed
 * (align, size) pair under the WGSL uniform layout rules. Struct-of-array
 * support is layered on top via a discriminated union (see `UboArrayFieldSpec`).
 *
 * Reserved for follow-up: `mat3x3f` (no concrete UBO needs it yet).
 */
export type UboFieldType =
  | 'f32'
  | 'u32'
  | 'i32'
  | 'vec2f'
  | 'vec3f'
  | 'vec4f'
  | 'vec2u'
  | 'vec3u'
  | 'vec4u'
  | 'mat4x4f';

interface LayoutInfo {
  readonly align: number;
  readonly size: number;
  readonly wgsl: string;
}

/**
 * Layout table for each field type.
 *
 * vec3f / vec3u keep `size: 12` but `align: 16` — the std140 "vec3 padded by
 * a trailing scalar" rule. The next field starts at the next 16-byte
 * boundary, but the vec3 itself only consumes 12 written bytes; the 4
 * trailing bytes are unwritten (driver leaves them at whatever the buffer
 * was initialized to, which for our use is zero-initialised scratch).
 */
const FIELD_LAYOUT: { readonly [K in UboFieldType]: LayoutInfo } = {
  f32:     { align:  4, size:  4, wgsl: 'f32'        },
  u32:     { align:  4, size:  4, wgsl: 'u32'        },
  i32:     { align:  4, size:  4, wgsl: 'i32'        },
  vec2f:   { align:  8, size:  8, wgsl: 'vec2<f32>'  },
  vec3f:   { align: 16, size: 12, wgsl: 'vec3<f32>'  },
  vec4f:   { align: 16, size: 16, wgsl: 'vec4<f32>'  },
  vec2u:   { align:  8, size:  8, wgsl: 'vec2<u32>'  },
  vec3u:   { align: 16, size: 12, wgsl: 'vec3<u32>'  },
  vec4u:   { align: 16, size: 16, wgsl: 'vec4<u32>'  },
  mat4x4f: { align: 16, size: 64, wgsl: 'mat4x4<f32>' },
} as const;

/** Minimum size of any WebGPU uniform-buffer binding (driver requirement). */
const MIN_UNIFORM_BUFFER_SIZE = 16;

/**
 * std140 / WGSL alignment of an array element when the element is an
 * aggregate (struct). Per WGSL §14.4.4, the array alignment is the
 * alignment of the element type, and for struct elements the alignment
 * is the max of the struct's member alignments — but the array stride
 * is rounded up to a multiple of 16 (RoundUp(AlignOf(element), 16)).
 */
const ARRAY_OF_STRUCT_ALIGN = 16;

// ─── Field-spec types ─────────────────────────────────────────────────────────

/**
 * Plain scalar / vector / matrix field. The legacy spec shape — what every
 * pre-W2-C13 caller writes.
 */
export interface UboFieldSpec<Name extends string = string, T extends UboFieldType = UboFieldType> {
  readonly name: Name;
  readonly type: T;
}

/**
 * Array-of-struct field. `element` is the inner struct's `UboDefinition`
 * (typically itself produced by a nested `defineUbo` call); `count` is the
 * compile-time fixed array length.
 *
 * The std140 array stride is `max(16, roundUp(element.sizeBytes, 16))`.
 * For an element whose own size is already a multiple of 16 (the common
 * case for nested `defineUbo` outputs), stride == element.sizeBytes.
 */
export interface UboArrayFieldSpec<
  Name extends string = string,
  F extends readonly AnyUboFieldSpec[] = readonly AnyUboFieldSpec[],
  Count extends number = number,
> {
  readonly name: Name;
  readonly type: 'array';
  readonly element: UboDefinition<F>;
  readonly count: Count;
}

/** Discriminated union of every supported field-spec shape. */
export type AnyUboFieldSpec =
  | UboFieldSpec<string, UboFieldType>
  | UboArrayFieldSpec<string, readonly AnyUboFieldSpec[], number>;

/**
 * Maps a single scalar/vector/matrix field spec to the runtime value type.
 * Scalars are `number`; vectors are fixed-length tuples; mat4x4f is a
 * 16-element tuple in column-major order (matching WGSL's storage convention).
 *
 * Integer vector tuples (`vec2u`/`vec3u`/`vec4u`) use the same numeric
 * tuple shape; callers pass JS numbers which are coerced via `>>> 0`
 * inside `pack`.
 */
type ScalarFieldValue<T extends UboFieldType> =
  T extends 'f32' | 'u32' | 'i32' ? number :
  T extends 'vec2f' | 'vec2u' ? readonly [number, number] :
  T extends 'vec3f' | 'vec3u' ? readonly [number, number, number] :
  T extends 'vec4f' | 'vec4u' ? readonly [number, number, number, number] :
  T extends 'mat4x4f' ? readonly [
    number, number, number, number,
    number, number, number, number,
    number, number, number, number,
    number, number, number, number,
  ] :
  never;

/**
 * Maps any field spec (scalar/vector/matrix or array-of-struct) to its
 * runtime value type. Array-of-struct fields become a readonly array of
 * the inner struct's value type, inferred from the element definition.
 */
type AnyFieldValue<S extends AnyUboFieldSpec> =
  S extends UboArrayFieldSpec<string, infer EF, number>
    ? readonly UboValue<EF>[]
    : S extends UboFieldSpec<string, infer T>
      ? ScalarFieldValue<T>
      : never;

/**
 * Object type derived from a tuple of field specs. Each spec contributes
 * a `[name]: AnyFieldValue<spec>` entry.
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
export type UboValue<F extends readonly AnyUboFieldSpec[]> = {
  readonly [N in F[number]['name']]:
    Extract<F[number], { name: N }> extends infer S
      ? S extends AnyUboFieldSpec
        ? AnyFieldValue<S>
        : never
      : never;
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function alignUp(offset: number, alignment: number): number {
  return Math.ceil(offset / alignment) * alignment;
}

/**
 * std140 array stride for an element struct: the element size rounded up
 * to a multiple of 16 bytes (the array element alignment for struct
 * elements). Always ≥ 16.
 */
function arrayStride(elementSize: number): number {
  return Math.max(ARRAY_OF_STRUCT_ALIGN, alignUp(elementSize, ARRAY_OF_STRUCT_ALIGN));
}

/**
 * Resolved field after layout: includes the byte offset, alignment, and
 * size. For array fields, also carries the element definition + count +
 * stride so pack/unpack can iterate.
 */
type ResolvedField =
  | {
      readonly kind: 'scalar';
      readonly name: string;
      readonly type: UboFieldType;
      readonly offset: number;
      readonly align: number;
      readonly size: number;
    }
  | {
      readonly kind: 'array';
      readonly name: string;
      readonly offset: number;
      readonly align: number;
      readonly size: number;
      readonly element: UboDefinition<readonly AnyUboFieldSpec[]>;
      readonly count: number;
      readonly stride: number;
    };

/**
 * Walk the field list, computing per-field offsets and the total struct size.
 *
 * The total size is rounded up to the largest member alignment, and then
 * bumped to at least MIN_UNIFORM_BUFFER_SIZE so the buffer is a valid
 * WebGPU uniform binding.
 */
function computeLayout(fields: readonly AnyUboFieldSpec[]): {
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
    if (spec.type === 'array') {
      const arr = spec;
      if (!arr.element || typeof arr.element.sizeBytes !== 'number') {
        throw new Error(`defineUbo: array field "${arr.name}" missing valid element UboDefinition`);
      }
      if (!Number.isInteger(arr.count) || arr.count <= 0) {
        throw new Error(`defineUbo: array field "${arr.name}" must have integer count > 0 (got ${arr.count})`);
      }
      const stride = arrayStride(arr.element.sizeBytes);
      const align = ARRAY_OF_STRUCT_ALIGN;
      const size = stride * arr.count;
      const offset = alignUp(cursor, align);
      resolved.push({
        kind: 'array',
        name: arr.name,
        offset,
        align,
        size,
        element: arr.element as UboDefinition<readonly AnyUboFieldSpec[]>,
        count: arr.count,
        stride,
      });
      cursor = offset + size;
      if (align > maxAlign) maxAlign = align;
      continue;
    }
    const layout = FIELD_LAYOUT[spec.type];
    if (!layout) {
      throw new Error(`defineUbo: unknown field type "${spec.type}" for field "${spec.name}"`);
    }
    const offset = alignUp(cursor, layout.align);
    resolved.push({
      kind: 'scalar',
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

export interface UboDefinition<F extends readonly AnyUboFieldSpec[]> {
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
 *
 * @example
 *   const Light = defineUbo([
 *     { name: 'kind',     type: 'u32'   },
 *     { name: 'position', type: 'vec3f' },
 *   ] as const);
 *   const Lights = defineUbo([
 *     { name: 'count', type: 'u32' },
 *     { name: 'items', type: 'array', element: Light, count: 16 },
 *   ] as const);
 */
export function defineUbo<const F extends readonly AnyUboFieldSpec[]>(fields: F): UboDefinition<F> {
  const { resolved, sizeBytes } = computeLayout(fields);

  const fieldOffsets = Object.fromEntries(resolved.map((f) => [f.name, f.offset])) as {
    readonly [K in keyof UboValue<F>]: number;
  };

  function packScalar(
    view: DataView,
    base: number,
    type: UboFieldType,
    v: unknown,
  ): void {
    switch (type) {
      case 'f32':
        view.setFloat32(base, v as number, true);
        return;
      case 'u32':
        view.setUint32(base, (v as number) >>> 0, true);
        return;
      case 'i32':
        view.setInt32(base, (v as number) | 0, true);
        return;
      case 'vec2f': {
        const tup = v as readonly number[];
        view.setFloat32(base + 0, tup[0]!, true);
        view.setFloat32(base + 4, tup[1]!, true);
        return;
      }
      case 'vec3f': {
        const tup = v as readonly number[];
        view.setFloat32(base + 0, tup[0]!, true);
        view.setFloat32(base + 4, tup[1]!, true);
        view.setFloat32(base + 8, tup[2]!, true);
        // bytes [base+12, base+16) are std140 pad — left at zero by zero-fill.
        return;
      }
      case 'vec4f': {
        const tup = v as readonly number[];
        view.setFloat32(base +  0, tup[0]!, true);
        view.setFloat32(base +  4, tup[1]!, true);
        view.setFloat32(base +  8, tup[2]!, true);
        view.setFloat32(base + 12, tup[3]!, true);
        return;
      }
      case 'vec2u': {
        const tup = v as readonly number[];
        view.setUint32(base + 0, tup[0]! >>> 0, true);
        view.setUint32(base + 4, tup[1]! >>> 0, true);
        return;
      }
      case 'vec3u': {
        const tup = v as readonly number[];
        view.setUint32(base + 0, tup[0]! >>> 0, true);
        view.setUint32(base + 4, tup[1]! >>> 0, true);
        view.setUint32(base + 8, tup[2]! >>> 0, true);
        // bytes [base+12, base+16) std140 pad — left at zero by zero-fill.
        return;
      }
      case 'vec4u': {
        const tup = v as readonly number[];
        view.setUint32(base +  0, tup[0]! >>> 0, true);
        view.setUint32(base +  4, tup[1]! >>> 0, true);
        view.setUint32(base +  8, tup[2]! >>> 0, true);
        view.setUint32(base + 12, tup[3]! >>> 0, true);
        return;
      }
      case 'mat4x4f': {
        const tup = v as readonly number[];
        for (let i = 0; i < 16; i++) view.setFloat32(base + i * 4, tup[i]!, true);
        return;
      }
      default: {
        // Exhaustiveness guard — TS will narrow to `never` here.
        const _exhaustive: never = type;
        throw new Error(`defineUbo.pack: unhandled field type ${_exhaustive as string}`);
      }
    }
  }

  function unpackScalar(view: DataView, base: number, type: UboFieldType): unknown {
    switch (type) {
      case 'f32': return view.getFloat32(base, true);
      case 'u32': return view.getUint32(base, true);
      case 'i32': return view.getInt32(base, true);
      case 'vec2f': return [
        view.getFloat32(base + 0, true), view.getFloat32(base + 4, true),
      ] as const;
      case 'vec3f': return [
        view.getFloat32(base + 0, true), view.getFloat32(base + 4, true),
        view.getFloat32(base + 8, true),
      ] as const;
      case 'vec4f': return [
        view.getFloat32(base +  0, true), view.getFloat32(base +  4, true),
        view.getFloat32(base +  8, true), view.getFloat32(base + 12, true),
      ] as const;
      case 'vec2u': return [
        view.getUint32(base + 0, true), view.getUint32(base + 4, true),
      ] as const;
      case 'vec3u': return [
        view.getUint32(base + 0, true), view.getUint32(base + 4, true),
        view.getUint32(base + 8, true),
      ] as const;
      case 'vec4u': return [
        view.getUint32(base +  0, true), view.getUint32(base +  4, true),
        view.getUint32(base +  8, true), view.getUint32(base + 12, true),
      ] as const;
      case 'mat4x4f': {
        const m: number[] = new Array(16);
        for (let i = 0; i < 16; i++) m[i] = view.getFloat32(base + i * 4, true);
        return m as readonly number[];
      }
      default: {
        const _exhaustive: never = type;
        throw new Error(`defineUbo.unpack: unhandled field type ${_exhaustive as string}`);
      }
    }
  }

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
      if (f.kind === 'scalar') {
        packScalar(view, base, f.type, v);
      } else {
        // array-of-struct field
        const arr = v as readonly unknown[];
        if (!Array.isArray(arr)) {
          throw new TypeError(
            `defineUbo.pack: field "${f.name}" expected an array; got ${typeof arr}`,
          );
        }
        const writeCount = Math.min(arr.length, f.count);
        for (let i = 0; i < writeCount; i++) {
          // Element pack writes into the slot starting at base + i*stride.
          // Each slot is exactly element.sizeBytes wide; any padding to
          // reach stride was already zero-filled by the outer Uint8Array.fill(0)
          // above.
          f.element.pack(view, base + i * f.stride, arr[i] as never);
        }
        // Unused trailing slots (if arr.length < f.count) stay zero-filled.
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
      if (f.kind === 'scalar') {
        out[f.name] = unpackScalar(view, base, f.type);
      } else {
        const items: unknown[] = new Array(f.count);
        for (let i = 0; i < f.count; i++) {
          items[i] = f.element.unpack(view, base + i * f.stride);
        }
        out[f.name] = items;
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
      if (f.kind === 'scalar') {
        lines.push(`${indent}${f.name}: ${FIELD_LAYOUT[f.type].wgsl},`);
        writtenCursor = f.offset + f.size;
      } else {
        // array<Element, N>. The element WGSL type name must be supplied
        // externally — we reference it by a conventional name derived from
        // the field. Emitting the inner struct decl is out of scope (callers
        // emit nested structs explicitly to keep ownership simple).
        lines.push(
          `${indent}${f.name}: array<${arrayElementTypeName(f.name)}, ${f.count}>,`,
        );
        writtenCursor = f.offset + f.size;
      }
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

/**
 * Convention for naming the element-struct WGSL type referenced by an
 * array<…> field. Capitalises the field name's singular form ("items" →
 * "Item", "lights" → "Light"). Callers can override by inspecting the
 * emitted WGSL string and substituting — for the existing migrations the
 * inner struct WGSL is hand-written in the shader file, so the conventional
 * name only needs to match what the shader expects.
 */
function arrayElementTypeName(fieldName: string): string {
  // Trim a trailing 's' if present, then PascalCase the first letter.
  const trimmed = fieldName.endsWith('s') ? fieldName.slice(0, -1) : fieldName;
  return trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
}
