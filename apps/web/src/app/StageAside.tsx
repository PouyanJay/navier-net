import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";

import { initialAsideCollapsed, storeAsideCollapsed } from "./asidePreference";

/** What a stage tells the shell about the aside it is filling. */
export interface AsideHeading {
  title: string;
  subtitle?: string;
  /** Expanded width in px, when this stage's content needs more room than the
   * default. A stage that omits it gets the shell's own width. */
  width?: number;
}

export interface AsideSlot {
  /** Where a stage's rail content is mounted; null until the shell renders it. */
  node: HTMLElement | null;
  /** Claim the rail with a heading, or release it with null. The shell owns the
   * header and the collapse control, so it needs the heading, not just the
   * body. */
  claim: (heading: AsideHeading | null) => void;
}

const NO_SLOT: AsideSlot = { node: null, claim: () => {} };

const AsideSlotContext = createContext<AsideSlot>(NO_SLOT);

/**
 * The shell's half of the aside: which stage has claimed it, whether it is
 * folded, and the slot its body mounts into.
 *
 * Kept here rather than in AppShell so the claim-and-fold logic lives beside
 * the `StageAside` it serves, and so the shell stays a composer of chrome.
 */
export function useAsideState() {
  const [node, setNode] = useState<HTMLElement | null>(null);
  const [heading, setHeading] = useState<AsideHeading | null>(null);
  const [collapsed, setCollapsed] = useState(initialAsideCollapsed);

  const toggle = useCallback(() => {
    setCollapsed((current) => {
      storeAsideCollapsed(!current);
      return !current;
    });
  }, []);

  // `claim` is identity-stable, so a stage's effect runs when its own heading
  // changes rather than on every shell render.
  const slot = useMemo(() => ({ node, claim: setHeading }), [node]);

  return { heading, collapsed, toggle, slot, setNode };
}

export function AsideSlotProvider({
  slot,
  children,
}: {
  slot: AsideSlot;
  children: ReactNode;
}) {
  return (
    <AsideSlotContext.Provider value={slot}>
      {children}
    </AsideSlotContext.Provider>
  );
}

/**
 * A stage's secondary rail: the fixed column between the pipeline rail and the
 * canvas.
 *
 * Rendered as a portal into the shell rather than in the page flow, so the rail
 * scrolls independently of the canvas and keeps its width while the canvas
 * scrolls — the thing that separates a sidebar from a narrow first column.
 *
 * A stage that renders no `StageAside` has no rail at all; the shell collapses
 * back to two columns, so this costs nothing on the stages that do not use it.
 */
export function StageAside({
  title,
  subtitle,
  children,
}: AsideHeading & { children: ReactNode }) {
  const slot = useContext(AsideSlotContext);
  const { claim } = slot;

  useEffect(() => {
    claim({ title, subtitle });
    return () => claim(null);
  }, [claim, title, subtitle]);

  return slot.node ? createPortal(children, slot.node) : null;
}
