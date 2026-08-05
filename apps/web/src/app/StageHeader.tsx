import {
  createContext,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";

/**
 * The slot a stage fills with the identity of the thing it is showing.
 *
 * The four pipeline stages used to open with an `<h1>` repeating the label the
 * rail had just used to get you there, plus a paragraph explaining the stage.
 * That is about 140px above the fold carrying nothing about the run. A stage
 * names its OBJECT here instead — the series, the run, the checkpoint — and the
 * navigation keeps the job of naming the page.
 *
 * Portalled rather than rendered in place so the identity sits in the shell's
 * header row, beside the stage's forward action, instead of pushing it down.
 */
const StageHeaderSlot = createContext<HTMLElement | null>(null);

export function useStageHeaderSlot() {
  const [node, setNode] = useState<HTMLElement | null>(null);
  return useMemo(() => ({ node, setNode }), [node]);
}

export function StageHeaderProvider({
  node,
  children,
}: {
  node: HTMLElement | null;
  children: ReactNode;
}) {
  return (
    <StageHeaderSlot.Provider value={node}>{children}</StageHeaderSlot.Provider>
  );
}

export function StageHeader({ children }: { children: ReactNode }) {
  const node = useContext(StageHeaderSlot);
  return node ? createPortal(children, node) : null;
}
