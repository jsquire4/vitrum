// domUtils.ts — low-level DOM element factories shared by all vanilla overlays.

export function makePanel(style: Record<string, string>): HTMLDivElement {
  return makeDiv({
    position: 'absolute',
    background: 'rgba(0,0,0,0.76)',
    color: '#e0e0e0',
    fontFamily: 'monospace',
    fontSize: '11px',
    padding: '8px 10px',
    borderRadius: '4px',
    userSelect: 'none',
    zIndex: '9997',
    lineHeight: '1.55',
    boxSizing: 'border-box',
    ...style,
  });
}

export function makeDiv(style: Record<string, string>): HTMLDivElement {
  const el = document.createElement('div');
  Object.assign(el.style, style);
  return el;
}

export function makeTitle(text: string): HTMLDivElement {
  const el = document.createElement('div');
  el.style.cssText = 'font-weight:bold;margin-bottom:5px;color:#fff';
  el.textContent = text;
  return el;
}

export function makeDivider(): HTMLDivElement {
  return makeDiv({
    borderTop: '1px solid rgba(255,255,255,0.16)',
    margin: '5px 0',
    height: '0',
  });
}

export function makeMuted(text: string): HTMLSpanElement {
  const el = document.createElement('span');
  el.style.cssText = 'color:#888;font-style:italic';
  el.textContent = text;
  return el;
}

export function makeRow(
  label: string,
  initialValue: string,
): { el: HTMLDivElement; setValue(v: string): void } {
  const el = document.createElement('div');
  el.style.cssText = 'display:flex;justify-content:space-between;gap:10px';
  const labelSpan = document.createElement('span');
  labelSpan.style.cssText = 'color:#888;white-space:nowrap';
  labelSpan.textContent = label;
  const valueSpan = document.createElement('span');
  valueSpan.style.cssText = 'color:#e0e0e0;text-align:right;overflow-wrap:anywhere';
  valueSpan.textContent = initialValue;
  el.append(labelSpan, valueSpan);
  return {
    el,
    setValue(v: string): void {
      valueSpan.textContent = v;
    },
  };
}
