// @ts-check

export class CaptureSurfaceResolutionError extends Error {
  /** @param {string} message */
  constructor(message) {
    super(message);
    this.name = 'CaptureSurfaceResolutionError';
  }
}

/**
 * Resolve exactly one visible Playwright surface. Hidden matches are ignored,
 * but zero or multiple visible matches are proof-invalid.
 *
 * @param {{ locator(selector: string): any }} page
 * @param {string} selector
 */
export async function resolveUniqueVisibleCaptureSurface(page, selector) {
  if (typeof selector !== 'string' || selector.trim().length === 0) {
    throw new CaptureSurfaceResolutionError('capture selector must be a non-empty string');
  }
  const matches = page.locator(selector);
  const count = await matches.count();
  const visible = [];
  for (let index = 0; index < count; index += 1) {
    const candidate = matches.nth(index);
    if (await candidate.isVisible()) visible.push(candidate);
  }
  if (visible.length !== 1) {
    throw new CaptureSurfaceResolutionError(
      `capture selector ${JSON.stringify(selector)} resolved ${count} total and ` +
        `${visible.length} visible surfaces; exactly one visible surface is required`,
    );
  }
  return visible[0];
}
