/**
 * D6.9 — PPG descent drift gate.
 *
 * The sTree and dTree descent functions in ppgUpdate.wgsl.ts and ppgPdf.wgsl.ts
 * are semantically identical — only the buffer binding names differ
 * (ppgSTreeBuf / ppgDTreeBuf in the update kernel vs ppgSTreeBuf_gi /
 * ppgDTreeBuf_gi in the pdf kernel). This test extracts those function bodies,
 * strips the binding-name tokens and any comment-only differences, and asserts
 * the normalised bodies are identical.
 *
 * A failure here means one descent was edited without mirroring the change in
 * the other — see the MUST-MATCH comments in both files.
 */

import { describe, expect, it } from 'vitest';
import { buildPpgUpdateWgsl } from '../ppgUpdate.wgsl.js';
import { PPG_PDF_WGSL } from '../ppgPdf.wgsl.js';

/** Extract the body of a named WGSL `fn` (from `fn name(` to its matching `}`). */
function extractFnBody(wgsl: string, fnName: string): string {
  const start = wgsl.indexOf(`fn ${fnName}(`);
  if (start === -1) throw new Error(`Function '${fnName}' not found in WGSL`);
  // Find the opening brace of the function body.
  const braceStart = wgsl.indexOf('{', start);
  if (braceStart === -1) throw new Error(`No opening brace for '${fnName}'`);
  // Walk forward matching braces to find the closing brace.
  let depth = 0;
  let i = braceStart;
  for (; i < wgsl.length; i++) {
    if (wgsl[i] === '{') depth++;
    else if (wgsl[i] === '}') {
      depth--;
      if (depth === 0) break;
    }
  }
  return wgsl.slice(braceStart, i + 1);
}

/**
 * Normalise a WGSL function body for comparison:
 * 1. Strip single-line (//) comments so comment-only differences are ignored.
 * 2. Collapse all whitespace runs to a single space.
 * 3. Replace binding-name tokens so ppgSTreeBuf and ppgSTreeBuf_gi are treated
 *    as equivalent (same for ppgDTreeBuf / ppgDTreeBuf_gi, and ppgDTreeOffsets /
 *    ppgDTreeOffsets_gi).
 */
function normalise(body: string): string {
  return body
    .replace(/\/\/[^\n]*/g, '')          // strip line comments
    .replace(/\s+/g, ' ')                 // collapse whitespace
    .trim()
    // Canonicalize binding-name tokens so _gi variants compare equal.
    .replace(/ppgSTreeBuf_gi/g, 'ppgSTreeBuf')
    .replace(/ppgDTreeBuf_gi/g, 'ppgDTreeBuf')
    .replace(/ppgDTreeOffsets_gi/g, 'ppgDTreeOffsets')
    // Canonicalize the parameter name used for the octahedral UV input.
    // The update kernel uses `uv`; the pdf kernel uses `octUV`; both
    // are the same semantic variable — normalise to `uv` for comparison.
    .replace(/\boctUV\b/g, 'uv');
}

describe('PPG descent drift gate (D6.9)', () => {
  const updateWgsl = buildPpgUpdateWgsl(341);
  const pdfWgsl = PPG_PDF_WGSL;

  it('sTree descent bodies are identical (modulo buffer-name tokens)', () => {
    const updateBody = extractFnBody(updateWgsl, 'sTreeFindLeafBase');
    const pdfBody    = extractFnBody(pdfWgsl,    'ppgSTreeFindLeafBase');
    expect(normalise(updateBody)).toBe(normalise(pdfBody));
  });

  it('dTree descent bodies are identical (modulo buffer-name tokens)', () => {
    const updateBody = extractFnBody(updateWgsl, 'dTreeFindLeafBase');
    const pdfBody    = extractFnBody(pdfWgsl,    'ppgDTreeFindLeafBase');
    expect(normalise(updateBody)).toBe(normalise(pdfBody));
  });
});
