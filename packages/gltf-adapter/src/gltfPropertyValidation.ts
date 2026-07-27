/** Validate the structural part of glTFProperty.extensions used by extensions. */
export function validateGltfPropertyExtensions(
  value: Readonly<Record<string, unknown>>,
  path: string,
): void {
  if (!Object.prototype.hasOwnProperty.call(value, 'extensions')) return;
  const extensions = value.extensions;
  if (extensions === undefined) return;
  if (extensions == null || typeof extensions !== 'object' || Array.isArray(extensions)) {
    throw new TypeError(`[vitrum/gltf-adapter] ${path}.extensions must be an object.`);
  }
  for (const key of Reflect.ownKeys(extensions)) {
    if (!Object.prototype.propertyIsEnumerable.call(extensions, key)) continue;
    if (typeof key !== 'string' || key.length === 0) {
      throw new TypeError(
        `[vitrum/gltf-adapter] ${path}.extensions contains an invalid extension name.`,
      );
    }
    const extension = (extensions as Record<string, unknown>)[key];
    if (extension == null || typeof extension !== 'object' || Array.isArray(extension)) {
      throw new TypeError(`[vitrum/gltf-adapter] ${path}.extensions.${key} must be an object.`);
    }
  }
}
