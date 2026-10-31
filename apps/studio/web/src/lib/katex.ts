import katex from "katex";

/** TeX to HTML, never throwing: a bad formula renders as its source. */
export function tex(source: string, display = false): string {
  if (!source) return "";
  try {
    return katex.renderToString(source, { throwOnError: false, displayMode: display });
  } catch {
    return `<code>${source}</code>`;
  }
}
