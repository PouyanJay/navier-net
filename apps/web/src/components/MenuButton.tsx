import { useEffect, useId, useRef, useState, type ReactNode } from "react";

import { HugeiconsIcon, MenuOpenIcon } from "./icons";

export interface MenuAction {
  /** Stable key, also used to build the item's element id. */
  id: string;
  label: string;
  /** A second line under the label, for what taking it produces. */
  hint?: string;
  onSelect: () => void;
}

interface MenuButtonProps {
  /** The trigger's accessible name. */
  label: string;
  /** What the trigger reads; omit for an icon-only button. */
  text?: string;
  icon?: ReactNode;
  actions: readonly MenuAction[];
  /** Which edge the menu hangs from. "end" for a trigger at the row's right. */
  align?: "start" | "end";
}

/**
 * A button that opens a menu of actions, to DESIGN_SYSTEM §8.5.
 *
 * The sibling of `Select`: same trigger, same popover, but the items DO
 * something rather than choose a value, so nothing is ever marked as current
 * and there is no reserved check column.
 */
export function MenuButton({
  label,
  text,
  icon,
  actions,
  align = "start",
}: MenuButtonProps) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const host = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const listId = useId();

  useEffect(() => {
    if (!open) return;
    const away = (event: PointerEvent) => {
      if (!host.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", away);
    return () => document.removeEventListener("pointerdown", away);
  }, [open]);

  const take = (index: number) => {
    actions[index]?.onSelect();
    setOpen(false);
    trigger.current?.focus();
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (!open) {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
        event.preventDefault();
        setActive(0);
        setOpen(true);
      }
      return;
    }
    const last = actions.length - 1;
    const moves: Record<string, number> = {
      ArrowDown: Math.min(last, active + 1),
      ArrowUp: Math.max(0, active - 1),
      Home: 0,
      End: last,
    };
    if (event.key in moves) {
      event.preventDefault();
      setActive(moves[event.key]);
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      take(active);
      return;
    }
    if (event.key === "Escape" || event.key === "Tab") {
      setOpen(false);
      if (event.key === "Escape") trigger.current?.focus();
    }
  };

  return (
    <div className="pick" ref={host}>
      <button
        ref={trigger}
        type="button"
        className="pick-btn"
        aria-label={label}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? listId : undefined}
        onClick={() => {
          setActive(0);
          setOpen(!open);
        }}
        onKeyDown={onKeyDown}
      >
        {icon}
        {text && <span className="pick-val">{text}</span>}
        <HugeiconsIcon icon={MenuOpenIcon} size={14} aria-hidden="true" />
      </button>

      {open && (
        <div
          className={
            align === "end" ? "menu pick-menu pick-menu-end" : "menu pick-menu"
          }
          id={listId}
          role="menu"
        >
          {actions.map((action, index) => (
            <button
              key={action.id}
              id={`${listId}-${action.id}`}
              type="button"
              role="menuitem"
              className={index === active ? "menu-item active" : "menu-item"}
              onPointerEnter={() => setActive(index)}
              onClick={() => take(index)}
            >
              <span className="pick-item-main">
                {action.label}
                {action.hint && <em>{action.hint}</em>}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
