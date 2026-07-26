import katex from "katex";

interface EquationBlockProps {
  tex: string;
  display?: boolean;
}

/** Render a LaTeX expression with KaTeX (display math by default). */
export function EquationBlock({ tex, display = true }: EquationBlockProps) {
  const html = katex.renderToString(tex, {
    displayMode: display,
    throwOnError: false,
  });
  return <div className="eq" dangerouslySetInnerHTML={{ __html: html }} />;
}
