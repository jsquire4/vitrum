// overlayMaterialInspector.ts — material inspector panel overlay.

import type { Scene } from '@vitrum/core';
import type { DebuggableEngine } from '../types.js';
import { makePanel, makeTitle, makeRow } from './domUtils.js';
import { materialPatchStatus, formatMaterial, formatPrimitiveDetails, findCanvas } from './debugUtils.js';

export function addMaterialInspectorFallback(
  engine: DebuggableEngine,
  container: HTMLElement,
  explicitCanvas: HTMLCanvasElement | null,
  scene: Scene | undefined,
  add: (el: HTMLElement) => void,
  cleanupFns: Array<() => void>,
): void {
  const panel = makePanel({
    top: '96px',
    right: '8px',
    minWidth: '250px',
    maxWidth: '300px',
  });
  panel.append(makeTitle('Material inspector'));

  const pickRow = makeRow('pick api', '-');
  const selectedRow = makeRow('selected', '-');
  const kindRow = makeRow('kind', '-');
  const materialRow = makeRow('material', '-');
  const detailsRow = makeRow('details', '-');
  const editRow = makeRow('mat patch', '-');
  panel.append(pickRow.el, selectedRow.el, kindRow.el, materialRow.el, detailsRow.el, editRow.el);
  add(panel);

  const canvas = explicitCanvas ?? findCanvas(container);
  const hasPickApi = typeof engine.debug?.pickPrimitive === 'function';

  const renderSelection = (primitiveId: string | null): void => {
    pickRow.setValue(hasPickApi ? 'ready' : 'unavailable');
    editRow.setValue(materialPatchStatus(engine));

    if (!hasPickApi) {
      selectedRow.setValue('disabled');
      kindRow.setValue('debug.pickPrimitive missing');
      materialRow.setValue(scene == null ? 'pass scene for lookup' : `${scene.primitives.length} scene prims`);
      detailsRow.setValue('-');
      return;
    }
    if (canvas == null) {
      selectedRow.setValue('disabled');
      kindRow.setValue('pass canvas');
      materialRow.setValue(scene == null ? 'pass scene for lookup' : `${scene.primitives.length} scene prims`);
      detailsRow.setValue('-');
      return;
    }
    if (scene == null) {
      selectedRow.setValue(primitiveId ?? 'none');
      kindRow.setValue('no scene snapshot');
      materialRow.setValue('-');
      detailsRow.setValue('-');
      return;
    }
    if (primitiveId == null) {
      selectedRow.setValue('none');
      kindRow.setValue('click canvas');
      materialRow.setValue('-');
      detailsRow.setValue('-');
      return;
    }

    const primitive = scene.primitives.find((p) => p.id === primitiveId);
    selectedRow.setValue(primitiveId);
    if (primitive == null) {
      kindRow.setValue('not in scene');
      materialRow.setValue('-');
      detailsRow.setValue('-');
      return;
    }

    kindRow.setValue(primitive.kind);
    materialRow.setValue(formatMaterial(primitive.material));
    detailsRow.setValue(formatPrimitiveDetails(primitive));
  };

  renderSelection(null);

  if (!hasPickApi || canvas == null) return;

  const onClick = (e: MouseEvent): void => {
    const rect = canvas.getBoundingClientRect();
    const width = rect.width || canvas.width || 1;
    const height = rect.height || canvas.height || 1;
    const x = (e.clientX - rect.left) * ((canvas.width || width) / width);
    const y = (e.clientY - rect.top) * ((canvas.height || height) / height);
    const primitiveId = engine.debug?.pickPrimitive?.(x, y) ?? null;
    renderSelection(primitiveId);
  };
  canvas.addEventListener('click', onClick);
  cleanupFns.push(() => canvas.removeEventListener('click', onClick));
}
