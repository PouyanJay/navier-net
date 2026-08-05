import { useCallback, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

/** Gap between the anchor and the panel, in px. */
const OFFSET = 8;
/** Keep the panel this far from the viewport edge. */
const MARGIN = 10;
const WIDTH = 340;

interface Placement {
  left: number;
  top: number;
}

/**
 * An `i` glyph whose detail opens in a layer above the page.
 *
 * Portalled to the body rather than positioned inside its anchor: the stage
 * rail scrolls, so an absolutely-positioned panel is clipped by that scroll
 * container — near the foot of the rail it simply disappears below the edge.
 * Escaping the container is the only fix that holds wherever the row sits.
 *
 * Opens on hover and on keyboard focus, closes on Escape, and flips above its
 * anchor when there is no room below.
 */
export function InfoPopover({
  label,
  children,
}: {
  /** What the glyph announces, e.g. "Momentum detail". */
  label: string;
  children: ReactNode;
}) {
  const anchor = useRef<HTMLButtonElement>(null);
  const [at, setAt] = useState<Placement | null>(null);

  const open = useCallback(() => {
    const box = anchor.current?.getBoundingClientRect();
    if (!box) return;
    // Right-aligned to the glyph, so the panel opens back over the row it
    // belongs to rather than off the side of a narrow rail.
    const left = Math.min(
      Math.max(MARGIN, box.right - WIDTH),
      window.innerWidth - WIDTH - MARGIN,
    );
    const below = window.innerHeight - box.bottom;
    // A rough panel height is enough to choose a side; the real one is not
    // known until it is painted, and being a little conservative here only
    // means flipping up slightly early.
    const flip = below < 220 && box.top > below;
    setAt({
      left,
      top: flip ? box.top - OFFSET : box.bottom + OFFSET,
    });
  }, []);

  const close = useCallback(() => setAt(null), []);

  return (
    <>
      <button
        ref={anchor}
        type="button"
        className="infob"
        aria-label={label}
        aria-expanded={at !== null}
        onPointerEnter={open}
        onPointerLeave={close}
        onFocus={open}
        onBlur={close}
        onClick={() => (at ? close() : open())}
        onKeyDown={(event) => {
          if (event.key === "Escape") close();
        }}
      >
        <span aria-hidden="true">i</span>
      </button>
      {at &&
        createPortal(
          <div
            className="infopop"
            role="tooltip"
            style={{
              left: at.left,
              top: at.top,
              width: WIDTH,
              // Flipping up means the panel's BOTTOM meets the anchor.
              transform:
                at.top < (anchor.current?.getBoundingClientRect().top ?? 0)
                  ? "translateY(-100%)"
                  : undefined,
            }}
          >
            {children}
          </div>,
          document.body,
        )}
    </>
  );
}
