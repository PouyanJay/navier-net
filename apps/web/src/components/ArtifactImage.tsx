import { useState, type ImgHTMLAttributes } from "react";

interface ArtifactImageProps extends ImgHTMLAttributes<HTMLImageElement> {
  alt: string;
}

/**
 * An <img> for pipeline artifacts that fails loudly: if the artifact can't be
 * fetched (deleted run dir, API hiccup), a visible note replaces the broken
 * image instead of a silent broken-image glyph.
 */
export function ArtifactImage({ alt, ...img }: ArtifactImageProps) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <span className="img-fallback" role="img" aria-label={alt}>
        {alt} could not be loaded
      </span>
    );
  }
  return <img alt={alt} {...img} onError={() => setFailed(true)} />;
}
