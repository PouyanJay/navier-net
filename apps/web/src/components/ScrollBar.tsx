import { useEffect, useRef, useState, type RefObject } from "react";

type Target = RefObject<HTMLElement | null>;

export interface ScrollExtent {
  /** Fraction of the content that fits on screen; 1 when nothing overflows. */
  visible: number;
  /** How far through the scrollable range we are, 0 to 1. */
  progress: number;
}

/** Measure a horizontal scroller, re-measuring on scroll and on resize. */
export function useScrollExtent(
  target: Target,
  itemCount: number,
): ScrollExtent {
  const [extent, setExtent] = useState<ScrollExtent>({
    visible: 1,
    progress: 0,
  });

  useEffect(() => {
    const element = target.current;
    if (!element) return;
    const measure = () => {
      const scrollable = element.scrollWidth - element.clientWidth;
      setExtent({
        visible:
          element.scrollWidth > 0
            ? element.clientWidth / element.scrollWidth
            : 1,
        progress: scrollable > 0 ? element.scrollLeft / scrollable : 0,
      });
    };
    measure();
    element.addEventListener("scroll", measure, { passive: true });
    const observer = new ResizeObserver(measure); // a panel resize changes the fit
    observer.observe(element);
    return () => {
      element.removeEventListener("scroll", measure);
      observer.disconnect();
    };
  }, [target, itemCount]);

  return extent;
}

/**
 * Let a plain mouse wheel drive a horizontal-only scroller.
 *
 * A wheel emits deltaY; a strip that scrolls only in x would ignore it and let
 * the page scroll instead, which reads as "the strip doesn't scroll". The event
 * is only swallowed while the strip can still move that way, so reaching either
 * end hands scrolling back to the page rather than trapping the pointer.
 */
export function useWheelToHorizontal(target: Target): void {
  useEffect(() => {
    const element = target.current;
    if (!element) return;
    const onWheel = (event: WheelEvent) => {
      if (event.deltaY === 0 || Math.abs(event.deltaX) > Math.abs(event.deltaY))
        return;
      const limit = element.scrollWidth - element.clientWidth;
      if (limit <= 0) return;
      const next = element.scrollLeft + event.deltaY;
      if (
        (next <= 0 && element.scrollLeft <= 0) ||
        (next >= limit && element.scrollLeft >= limit)
      )
        return;
      event.preventDefault();
      element.scrollLeft = Math.max(0, Math.min(limit, next));
    };
    // Not passive: the whole point is to take the event over from the page.
    element.addEventListener("wheel", onWheel, { passive: false });
    return () => element.removeEventListener("wheel", onWheel);
  }, [target]);
}

interface ScrollBarProps {
  target: Target;
  extent: ScrollExtent;
  /** Names the control for assistive tech, e.g. "Scroll the frame strip". */
  label: string;
  /** `id` of the scroller this drives; the caller sets it on that element. */
  controls: string;
  /** Fraction of a viewport to move per arrow key or track click page. */
  step?: number;
}

const DEFAULT_STEP = 0.25;

/**
 * A slim horizontal scrollbar for a scroller whose native bar is hidden.
 *
 * Hiding the native bar is deliberate (on macOS it is an overlay that vanishes
 * at rest, so overflow goes unnoticed) but it also removes the only pointer
 * affordance; hence a real control here: draggable thumb, click-to-page track,
 * and arrow/Home/End keys. Renders nothing when the content fits.
 */
export function ScrollBar({
  target,
  extent,
  label,
  controls,
  step = DEFAULT_STEP,
}: ScrollBarProps) {
  const track = useRef<HTMLDivElement>(null);
  const dragOffset = useRef<number | null>(null);

  if (extent.visible >= 1) return null;

  const thumbFraction = Math.max(extent.visible, 0.08); // stays grabbable when tiny
  const travel = 1 - thumbFraction;

  /** Move the scroller so the thumb's left edge sits at `fraction` of the track. */
  const scrollToThumb = (fraction: number) => {
    const element = target.current;
    if (!element || travel <= 0) return;
    const progress = Math.max(0, Math.min(1, fraction / travel));
    element.scrollLeft = progress * (element.scrollWidth - element.clientWidth);
  };

  /** Where the pointer sits along the track, as a 0..1 fraction. */
  const fractionAt = (clientX: number) => {
    const box = track.current?.getBoundingClientRect();
    return box && box.width > 0 ? (clientX - box.left) / box.width : 0;
  };

  const nudge = (direction: number) => {
    const element = target.current;
    if (!element) return;
    element.scrollBy({
      left: direction * element.clientWidth * step,
      behavior: "smooth",
    });
  };

  function onThumbPointerDown(event: React.PointerEvent<HTMLDivElement>) {
    event.preventDefault(); // no text selection while dragging
    event.currentTarget.setPointerCapture(event.pointerId);
    dragOffset.current = fractionAt(event.clientX) - extent.progress * travel;
  }

  function onThumbPointerMove(event: React.PointerEvent<HTMLDivElement>) {
    if (dragOffset.current === null) return;
    scrollToThumb(fractionAt(event.clientX) - dragOffset.current);
  }

  function endDrag(event: React.PointerEvent<HTMLDivElement>) {
    dragOffset.current = null;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
  }

  function onTrackPointerDown(event: React.PointerEvent<HTMLDivElement>) {
    if (event.target !== event.currentTarget) return; // the thumb handles its own
    scrollToThumb(fractionAt(event.clientX) - thumbFraction / 2); // centre on the click
  }

  function onKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    const element = target.current;
    if (!element) return;
    const keys: Record<string, () => void> = {
      ArrowLeft: () => nudge(-1),
      ArrowRight: () => nudge(1),
      Home: () => element.scrollTo({ left: 0, behavior: "smooth" }),
      End: () =>
        element.scrollTo({
          left: element.scrollWidth - element.clientWidth,
          behavior: "smooth",
        }),
    };
    const action = keys[event.key];
    if (!action) return;
    event.preventDefault();
    action();
  }

  return (
    <div
      ref={track}
      className="hbar"
      role="scrollbar"
      tabIndex={0}
      aria-label={label}
      aria-controls={controls}
      aria-orientation="horizontal"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(extent.progress * 100)}
      onPointerDown={onTrackPointerDown}
      onKeyDown={onKeyDown}
    >
      <div
        className="hbar-thumb"
        style={{
          width: `${thumbFraction * 100}%`,
          left: `${extent.progress * travel * 100}%`,
        }}
        onPointerDown={onThumbPointerDown}
        onPointerMove={onThumbPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      />
    </div>
  );
}
