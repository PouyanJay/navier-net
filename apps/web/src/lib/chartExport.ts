/** Client-side chart exports: standalone SVG, high-resolution PNG, and the
 * charted data as CSV/JSON. Charts style themselves through CSS custom
 * properties, so a downloaded SVG inlines the computed styles — the file must
 * look right outside the app's stylesheet. */

/** Presentation properties worth carrying into a standalone SVG file. */
const STYLE_PROPS = [
  "fill",
  "stroke",
  "stroke-width",
  "stroke-dasharray",
  "stroke-linejoin",
  "stroke-linecap",
  "font-family",
  "font-size",
  "font-weight",
  "letter-spacing",
  "text-anchor",
  "opacity",
] as const;

/** How much larger than CSS pixels a downloaded PNG renders (print-ready). */
export const PNG_SCALE = 3;

export function serializeSvg(svg: SVGSVGElement, background: string): string {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  const sources = svg.querySelectorAll<SVGElement>("*");
  const targets = clone.querySelectorAll<SVGElement>("*");
  sources.forEach((source, index) => {
    const computed = getComputedStyle(source);
    for (const property of STYLE_PROPS) {
      const value = computed.getPropertyValue(property);
      if (value && value !== "none" && value !== "normal")
        targets[index].setAttribute(property, value);
    }
  });
  const viewBox = svg.viewBox.baseVal;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", String(viewBox.width));
  clone.setAttribute("height", String(viewBox.height));
  // The app draws charts on the dark canvas; a standalone file needs that
  // ground or light-on-transparent text disappears in most viewers.
  const backdrop = clone.ownerDocument.createElementNS(
    "http://www.w3.org/2000/svg",
    "rect",
  );
  backdrop.setAttribute("width", "100%");
  backdrop.setAttribute("height", "100%");
  backdrop.setAttribute("fill", background);
  clone.insertBefore(backdrop, clone.firstChild);
  return new XMLSerializer().serializeToString(clone);
}

export function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function downloadText(
  filename: string,
  text: string,
  mime: string,
): void {
  downloadBlob(filename, new Blob([text], { type: mime }));
}

/** Rasterize a serialized SVG at PNG_SCALE× its viewBox size. */
export function svgTextToPng(
  svgText: string,
  width: number,
  height: number,
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = width * PNG_SCALE;
      canvas.height = height * PNG_SCALE;
      const context = canvas.getContext("2d");
      if (!context) return reject(new Error("canvas unavailable"));
      context.scale(PNG_SCALE, PNG_SCALE);
      context.drawImage(image, 0, 0, width, height);
      canvas.toBlob(
        (blob) =>
          blob ? resolve(blob) : reject(new Error("PNG encoding failed")),
        "image/png",
      );
    };
    image.onerror = () => reject(new Error("could not rasterize the SVG"));
    image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgText)}`;
  });
}

/** Upscale a raster chart (field maps) into a smooth high-resolution PNG. */
export function canvasToPng(source: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const target = document.createElement("canvas");
    const width = source.clientWidth || source.width;
    const height = source.clientHeight || source.height;
    target.width = width * PNG_SCALE;
    target.height = height * PNG_SCALE;
    const context = target.getContext("2d");
    if (!context) return reject(new Error("canvas unavailable"));
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(source, 0, 0, target.width, target.height);
    target.toBlob(
      (blob) =>
        blob ? resolve(blob) : reject(new Error("PNG encoding failed")),
      "image/png",
    );
  });
}

/** Rows → RFC-ish CSV: header from the union of keys, values quoted as needed. */
export function rowsToCsv(rows: Record<string, unknown>[]): string {
  if (rows.length === 0) return "";
  const keys = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const cell = (value: unknown): string => {
    if (value == null) return "";
    const text = String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  return [
    keys.join(","),
    ...rows.map((row) => keys.map((key) => cell(row[key])).join(",")),
  ].join("\n");
}
