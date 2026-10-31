/** Downloads: CSV and JSON of any payload, PNG of the scene. */

export function download(filename: string, content: Blob | string, type = "text/plain"): void {
  const blob = content instanceof Blob ? content : new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadJson(filename: string, payload: unknown): void {
  download(filename, JSON.stringify(payload, null, 2), "application/json");
}

function cell(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** A CSV from columns and rows, with provenance as leading comment lines. */
export function toCsv(
  columns: string[],
  rows: unknown[][],
  meta: Record<string, unknown> = {},
): string {
  const head = Object.entries(meta)
    .filter(([, v]) => typeof v !== "object")
    .map(([k, v]) => `# ${k}: ${String(v)}`);
  return (
    [...head, columns.join(","), ...rows.map((row) => row.map(cell).join(","))].join("\n") + "\n"
  );
}

export function downloadCsv(
  filename: string,
  columns: string[],
  rows: unknown[][],
  meta?: Record<string, unknown>,
): void {
  download(filename, toCsv(columns, rows, meta), "text/csv");
}

/** Compose the WebGL canvas and the SVG overlay into one PNG. */
export async function sceneToPng(
  canvas: HTMLCanvasElement,
  overlay: SVGSVGElement,
  background: string,
): Promise<Blob> {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const out = document.createElement("canvas");
  out.width = width * dpr;
  out.height = height * dpr;
  const ctx = out.getContext("2d");
  if (!ctx) throw new Error("no 2d context for export");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.drawImage(canvas, 0, 0, out.width, out.height);
  const clone = overlay.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));
  inlineComputedColours(overlay, clone);
  const svg = new XMLSerializer().serializeToString(clone);
  const image = new Image();
  const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml;charset=utf-8" }));
  await new Promise<void>((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = () => reject(new Error("overlay could not be rasterised"));
    image.src = url;
  });
  ctx.drawImage(image, 0, 0, out.width, out.height);
  URL.revokeObjectURL(url);
  return new Promise((resolve, reject) =>
    out.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("png failed"))), "image/png"),
  );
}

/** CSS variables do not survive serialisation; bake the computed values in. */
function inlineComputedColours(source: Element, target: Element): void {
  const sourceNodes = [source, ...source.querySelectorAll("*")];
  const targetNodes = [target, ...target.querySelectorAll("*")];
  for (let i = 0; i < sourceNodes.length && i < targetNodes.length; i++) {
    const computed = getComputedStyle(sourceNodes[i]);
    const el = targetNodes[i] as HTMLElement;
    for (const prop of [
      "stroke",
      "fill",
      "font-family",
      "font-size",
      "opacity",
      "stroke-width",
      "stroke-dasharray",
    ]) {
      const value = computed.getPropertyValue(prop);
      if (value) el.style.setProperty(prop, value);
    }
  }
}
