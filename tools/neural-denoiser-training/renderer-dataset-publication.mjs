import { lstat, mkdir, mkdtemp, rename, rm } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';

async function pathExists(path) {
  try {
    await lstat(path);
    return true;
  } catch (error) {
    if (error?.code === 'ENOENT') return false;
    throw error;
  }
}

/**
 * Build one complete renderer dataset in a same-filesystem sibling directory,
 * then publish it with one directory rename.
 *
 * The final output path must not already exist. Refusing replacement is
 * deliberate: replacing a populated directory is not one atomic,
 * cross-platform operation, and mutating it in place can leave an old manifest
 * describing a mixture of old and new artifacts after a failed rerun.
 *
 * @template T
 * @param {string} outputPath
 * @param {(stagingPath: string) => Promise<T>} build
 * @returns {Promise<{ readonly outputPath: string, readonly result: T }>}
 */
export async function publishRendererDatasetGeneration(outputPath, build) {
  if (typeof outputPath !== 'string' || outputPath.trim().length === 0) {
    throw new TypeError('renderer dataset output path must be non-empty');
  }
  if (typeof build !== 'function') {
    throw new TypeError('renderer dataset generation builder must be a function');
  }

  const finalPath = resolve(outputPath);
  const parentPath = dirname(finalPath);
  const outputName = basename(finalPath);
  if (outputName.length === 0) {
    throw new RangeError('renderer dataset output must name a directory');
  }

  await mkdir(parentPath, { recursive: true });
  if (await pathExists(finalPath)) {
    throw new Error(
      `renderer dataset output already exists: ${finalPath}. ` +
        'Choose a new --out path or explicitly remove the prior generation.',
    );
  }

  // mkdtemp appends a unique suffix. Keeping the staging directory beside the
  // final path guarantees the publish rename stays on one filesystem/volume on
  // both POSIX and Windows.
  const stagingPath = await mkdtemp(join(parentPath, `.${outputName}.vitrum-generation-`));
  let published = false;
  let failed = false;
  let result;
  let primaryError;
  try {
    result = await build(stagingPath);
    // The destination was required to be absent above. A concurrent publisher
    // can still win the race; in that case rename fails without altering its
    // populated generation and this staging tree is removed below.
    await rename(stagingPath, finalPath);
    published = true;
  } catch (error) {
    failed = true;
    primaryError = error;
  }

  if (!published) {
    try {
      await rm(stagingPath, { recursive: true, force: true });
    } catch (cleanupError) {
      if (failed) {
        throw new AggregateError(
          [primaryError, cleanupError],
          'renderer dataset generation failed and staging cleanup also failed',
        );
      }
      throw cleanupError;
    }
  }
  if (failed) throw primaryError;

  return { outputPath: finalPath, result };
}
