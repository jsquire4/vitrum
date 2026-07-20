/**
 * D6-5 — parity guard between the consumed-material-field allowlist and its
 * keyed doc index. `CONSUMED_MATERIAL_FIELDS` is DERIVED from
 * `CONSUMED_MATERIAL_FIELD_DOCS` keys, so this test pins that:
 *   - every allowlist member has a non-empty doc entry, and
 *   - the doc record has no orphan keys (docs ⊇ Set and Set ⊇ docs).
 * This makes the Set and the docs structurally impossible to drift.
 */
import { describe, expect, it } from 'vitest';
import {
  CONSUMED_MATERIAL_FIELDS,
  CONSUMED_MATERIAL_FIELD_DOCS,
} from '../restir/consumedMaterialFields.js';

describe('consumedMaterialField docs ↔ allowlist parity (D6-5)', () => {
  it('every allowlist field has a non-empty doc entry', () => {
    for (const field of CONSUMED_MATERIAL_FIELDS) {
      const doc = CONSUMED_MATERIAL_FIELD_DOCS[field];
      expect(doc, `missing doc for consumed field "${field}"`).toBeDefined();
      expect(doc!.length, `empty doc for consumed field "${field}"`).toBeGreaterThan(0);
    }
  });

  it('the doc record has no keys outside the allowlist', () => {
    for (const field of Object.keys(CONSUMED_MATERIAL_FIELD_DOCS)) {
      expect(CONSUMED_MATERIAL_FIELDS.has(field), `doc key "${field}" is not in the allowlist`).toBe(true);
    }
  });

  it('the allowlist and doc keys are the same set', () => {
    expect(CONSUMED_MATERIAL_FIELDS.size).toBe(Object.keys(CONSUMED_MATERIAL_FIELD_DOCS).length);
  });
});
