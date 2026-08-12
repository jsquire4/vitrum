import { asMat4, type CameraDescriptor, type Mat4 } from '@vitrum/core';
import type { GltfCamera, GltfJson } from './gltfTypes.js';
import type { ImportResourceLedger } from './importResourceBudget.js';

export interface GltfPerspectiveCameraProjection {
  readonly yfov?: number;
  readonly znear?: number;
  readonly zfar?: number;
  readonly aspectRatio?: number;
}

export interface GltfOrthographicCameraProjection {
  readonly xmag?: number;
  readonly ymag?: number;
  readonly znear?: number;
  readonly zfar?: number;
}

export interface GltfSceneCamera {
  readonly cameraIndex: number;
  readonly nodeIndex: number;
  readonly path: string;
  readonly nodePath: string;
  readonly type: 'perspective' | 'orthographic';
  readonly worldMatrix: Mat4;
  readonly name?: string;
  readonly nodeName?: string;
  readonly perspective?: GltfPerspectiveCameraProjection;
  readonly orthographic?: GltfOrthographicCameraProjection;
}

export interface GltfCameraMetadataIssue {
  readonly path: string;
  readonly message: string;
}

/**
 * Validate the camera objects referenced by the selected scene against the
 * glTF 2.0 camera schema and its cross-field projection constraints.
 *
 * Import validation is deliberately scene-scoped: malformed cameras that are
 * not reachable from the selected scene do not prevent that scene from being
 * imported.
 */
export function validateGltfCameraMetadata(
  gltf: GltfJson,
  cameraIndices: ReadonlySet<number>,
): readonly GltfCameraMetadataIssue[] {
  const issues: GltfCameraMetadataIssue[] = [];
  const cameras: readonly unknown[] = Array.isArray(gltf.cameras) ? gltf.cameras : [];

  for (const cameraIndex of [...cameraIndices].sort((a, b) => a - b)) {
    const path = `cameras[${cameraIndex}]`;
    const camera = cameras[cameraIndex];
    if (!isRecord(camera)) {
      issues.push(cameraIssue(path, 'must be an object.'));
      continue;
    }

    if (camera.type !== 'perspective' && camera.type !== 'orthographic') {
      issues.push(cameraIssue(
        `${path}.type`,
        'is required and must be "perspective" or "orthographic".',
      ));
      continue;
    }

    if (camera.type === 'perspective') {
      if (camera.orthographic !== undefined) {
        issues.push(cameraIssue(
          `${path}.orthographic`,
          'must not be defined when type is "perspective".',
        ));
      }
      validatePerspectiveProjection(camera.perspective, `${path}.perspective`, issues);
    } else {
      if (camera.perspective !== undefined) {
        issues.push(cameraIssue(
          `${path}.perspective`,
          'must not be defined when type is "orthographic".',
        ));
      }
      validateOrthographicProjection(camera.orthographic, `${path}.orthographic`, issues);
    }
  }

  return issues;
}

export function collectSceneCameras(
  gltf: GltfJson,
  worldTransforms: ReadonlyMap<number, Mat4>,
  resourceLedger?: ImportResourceLedger,
  allocationPath = 'scene cameras',
): GltfSceneCamera[] {
  const cameras = gltf.cameras ?? [];
  const nodes = gltf.nodes ?? [];
  const result: GltfSceneCamera[] = [];
  for (const [nodeIndex, worldMatrix] of worldTransforms) {
    const node = nodes[nodeIndex];
    if (!node) continue;
    const candidateCameraIndex = node.camera;
    if (
      candidateCameraIndex === undefined ||
      !Number.isInteger(candidateCameraIndex) ||
      candidateCameraIndex < 0
    ) {
      continue;
    }
    const cameraIndex = candidateCameraIndex;
    const camera = cameras[cameraIndex];
    if (!camera) continue;
    if (camera.type !== 'perspective' && camera.type !== 'orthographic') continue;
    const type = camera.type;
    resourceLedger?.chargeDecodedGeometryBytes(
      worldMatrix.byteLength,
      `${allocationPath}.nodes[${nodeIndex}].worldMatrix`,
    );
    result.push({
      cameraIndex,
      nodeIndex,
      path: `cameras[${cameraIndex}]`,
      nodePath: `nodes[${nodeIndex}]`,
      type,
      worldMatrix: asMat4(new Float32Array(worldMatrix)),
      ...(typeof camera.name === 'string' ? { name: camera.name } : {}),
      ...(typeof node.name === 'string' ? { nodeName: node.name } : {}),
      ...(type === 'perspective'
        ? { perspective: extractPerspectiveCameraProjection(camera) }
        : {}),
      ...(type === 'orthographic'
        ? { orthographic: extractOrthographicCameraProjection(camera) }
        : {}),
    });
  }
  return result;
}

function extractPerspectiveCameraProjection(
  camera: GltfCamera,
): GltfPerspectiveCameraProjection {
  const source = camera.perspective as Record<string, unknown>;
  return {
    yfov: source.yfov as number,
    znear: source.znear as number,
    ...finiteField(source, 'zfar'),
    ...finiteField(source, 'aspectRatio'),
  };
}

function extractOrthographicCameraProjection(
  camera: GltfCamera,
): GltfOrthographicCameraProjection {
  const source = camera.orthographic as Record<string, unknown>;
  return {
    xmag: source.xmag as number,
    ymag: source.ymag as number,
    znear: source.znear as number,
    zfar: source.zfar as number,
  };
}

function validatePerspectiveProjection(
  value: unknown,
  path: string,
  issues: GltfCameraMetadataIssue[],
): void {
  if (!isRecord(value)) {
    issues.push(cameraIssue(path, 'is required and must be an object for a perspective camera.'));
    return;
  }

  const yfov = requireFiniteNumber(value, 'yfov', path, issues);
  const znear = requireFiniteNumber(value, 'znear', path, issues);
  const zfar = optionalFiniteNumber(value, 'zfar', path, issues);
  const aspectRatio = optionalFiniteNumber(value, 'aspectRatio', path, issues);

  if (yfov !== undefined && !(yfov > 0)) {
    issues.push(cameraIssue(`${path}.yfov`, 'must be greater than 0 radians.'));
  }
  if (znear !== undefined && !(znear > 0)) {
    issues.push(cameraIssue(`${path}.znear`, 'must be greater than 0.'));
  }
  if (zfar !== undefined && !(zfar > 0)) {
    issues.push(cameraIssue(`${path}.zfar`, 'must be greater than 0 when defined.'));
  }
  if (zfar !== undefined && znear !== undefined && !(zfar > znear)) {
    issues.push(cameraIssue(`${path}.zfar`, 'must be greater than znear.'));
  }
  if (aspectRatio !== undefined && !(aspectRatio > 0)) {
    issues.push(cameraIssue(`${path}.aspectRatio`, 'must be greater than 0 when defined.'));
  }
}

function validateOrthographicProjection(
  value: unknown,
  path: string,
  issues: GltfCameraMetadataIssue[],
): void {
  if (!isRecord(value)) {
    issues.push(cameraIssue(path, 'is required and must be an object for an orthographic camera.'));
    return;
  }

  const xmag = requireFiniteNumber(value, 'xmag', path, issues);
  const ymag = requireFiniteNumber(value, 'ymag', path, issues);
  const znear = requireFiniteNumber(value, 'znear', path, issues);
  const zfar = requireFiniteNumber(value, 'zfar', path, issues);

  if (xmag !== undefined && xmag === 0) {
    issues.push(cameraIssue(`${path}.xmag`, 'must not be 0.'));
  }
  if (ymag !== undefined && ymag === 0) {
    issues.push(cameraIssue(`${path}.ymag`, 'must not be 0.'));
  }
  if (znear !== undefined && znear < 0) {
    issues.push(cameraIssue(`${path}.znear`, 'must be greater than or equal to 0.'));
  }
  if (zfar !== undefined && !(zfar > 0)) {
    issues.push(cameraIssue(`${path}.zfar`, 'must be greater than 0.'));
  }
  if (zfar !== undefined && znear !== undefined && !(zfar > znear)) {
    issues.push(cameraIssue(`${path}.zfar`, 'must be greater than znear.'));
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function requireFiniteNumber(
  source: Record<string, unknown>,
  name: string,
  path: string,
  issues: GltfCameraMetadataIssue[],
): number | undefined {
  if (!(name in source)) {
    issues.push(cameraIssue(`${path}.${name}`, 'is required.'));
    return undefined;
  }
  return finiteNumber(source[name], `${path}.${name}`, issues);
}

function optionalFiniteNumber(
  source: Record<string, unknown>,
  name: string,
  path: string,
  issues: GltfCameraMetadataIssue[],
): number | undefined {
  if (!(name in source)) return undefined;
  return finiteNumber(source[name], `${path}.${name}`, issues);
}

function finiteNumber(
  value: unknown,
  path: string,
  issues: GltfCameraMetadataIssue[],
): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    issues.push(cameraIssue(path, 'must be a finite number.'));
    return undefined;
  }
  return value;
}

function cameraIssue(path: string, detail: string): GltfCameraMetadataIssue {
  return {
    path,
    message: `[vitrum/gltf-adapter] ${path} ${detail}`,
  };
}

/**
 * Flatten a glTF scene camera (nested projection bag + extra import metadata)
 * into the core host {@link CameraDescriptor} that `cameraToFrameMatrices` consumes.
 */
export function gltfSceneCameraToDescriptor(camera: GltfSceneCamera): CameraDescriptor {
  if (camera.type === 'perspective') {
    const yfov = camera.perspective?.yfov;
    const znear = camera.perspective?.znear;
    if (!(typeof yfov === 'number' && yfov > 0) || !(typeof znear === 'number' && znear > 0)) {
      throw new RangeError(
        `${camera.path} perspective camera requires yfov > 0 and znear > 0.`,
      );
    }
    return {
      type: 'perspective',
      worldMatrix: camera.worldMatrix,
      yfov,
      znear,
      ...(typeof camera.perspective?.zfar === 'number' ? { zfar: camera.perspective.zfar } : {}),
      ...(typeof camera.perspective?.aspectRatio === 'number'
        ? { aspectRatio: camera.perspective.aspectRatio }
        : {}),
    };
  }
  const xmag = camera.orthographic?.xmag;
  const ymag = camera.orthographic?.ymag;
  const znear = camera.orthographic?.znear;
  const zfar = camera.orthographic?.zfar;
  if (
    !(typeof xmag === 'number' && xmag > 0) ||
    !(typeof ymag === 'number' && ymag > 0) ||
    typeof znear !== 'number' ||
    !(typeof zfar === 'number' && zfar > znear)
  ) {
    throw new RangeError(
      `${camera.path} orthographic camera requires xmag > 0, ymag > 0, and zfar > znear.`,
    );
  }
  return {
    type: 'orthographic',
    worldMatrix: camera.worldMatrix,
    xmag,
    ymag,
    znear,
    zfar,
  };
}

function finiteField<TName extends string>(
  source: Record<string, unknown>,
  name: TName,
): Partial<Record<TName, number>> {
  const value = source[name];
  return typeof value === 'number' && Number.isFinite(value)
    ? { [name]: value } as Partial<Record<TName, number>>
    : {};
}
