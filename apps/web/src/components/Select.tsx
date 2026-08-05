import { useEffect, useId, useRef, useState } from "react";

import { ChosenIcon, HugeiconsIcon, MenuOpenIcon } from "./icons";

export interface SelectChoice<T extends string> {
  value: T;
  label: string;
  /** A second line under the label, for what the option actually shows. */
  hint?: string;
}

interface SelectProps<T extends string> {
  /** The control's accessible name. It is never rendered as a caption. */
  label: string;
  value: T;
  options: readonly SelectChoice<T>[];
  onChange: (value: T) => void;
  /** Extra class on the trigger, for callers that size it themselves. */
  className?: string;
}

/**
 * A listbox the app draws itself, to DESIGN_SYSTEM §8.5.
 *
 * A native `<select>` renders the operating system's menu: system font, system
 * metrics, system colours, in neither theme and on neither surface. Everywhere
 * a chooser sits in the product's own chrome, this is what it looks like.
 *
 * The chosen option keeps a reserved 16px check column with `visibility`
 * rather than `display`, so the labels do not shift as the choice moves.
 */
export function Select<T extends string>({
  label,
  value,
  options,
  onChange,
  className,
}: SelectProps<T>) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(() =>
    Math.max(
      0,
      options.findIndex((option) => option.value === value),
    ),
  );
  const host = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const listId = useId();
  const chosen = options.find((option) => option.value === value);

  // Any press outside closes it. Pointerdown, not click, so the menu is gone
  // before whatever was underneath it reacts.
  useEffect(() => {
    if (!open) return;
    const away = (event: PointerEvent) => {
      if (!host.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", away);
    return () => document.removeEventListener("pointerdown", away);
  }, [open]);

  const openAt = (index: number) => {
    setActive(index);
    setOpen(true);
  };
  const choose = (index: number) => {
    const option = options[index];
    if (option) onChange(option.value);
    setOpen(false);
    trigger.current?.focus();
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    const current = options.findIndex((option) => option.value === value);
    if (!open) {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
        event.preventDefault();
        openAt(Math.max(0, current));
      }
      return;
    }
    const last = options.length - 1;
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
      choose(active);
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
        className={className ? `pick-btn ${className}` : "pick-btn"}
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listId : undefined}
        // Focus stays on the trigger while the list is open, so the active
        // option is named here rather than by moving focus into the menu.
        aria-activedescendant={
          open ? `${listId}-${options[active]?.value}` : undefined
        }
        onClick={() =>
          open
            ? setOpen(false)
            : openAt(
                Math.max(
                  0,
                  options.findIndex((option) => option.value === value),
                ),
              )
        }
        onKeyDown={onKeyDown}
      >
        <span className="pick-val">{chosen?.label ?? ""}</span>
        <HugeiconsIcon icon={MenuOpenIcon} size={14} aria-hidden="true" />
      </button>

      {open && (
        <div className="menu pick-menu" id={listId} role="listbox">
          {options.map((option, index) => {
            const isChosen = option.value === value;
            return (
              <button
                key={option.value}
                id={`${listId}-${option.value}`}
                type="button"
                role="option"
                aria-selected={isChosen}
                className={index === active ? "menu-item active" : "menu-item"}
                // The pointer moves the active option so hover and keyboard
                // never disagree about which one is about to be taken.
                onPointerEnter={() => setActive(index)}
                onClick={() => choose(index)}
              >
                <HugeiconsIcon
                  icon={ChosenIcon}
                  size={13}
                  className="pick-tick"
                  style={{ visibility: isChosen ? "visible" : "hidden" }}
                  aria-hidden="true"
                />
                <span className="pick-item-main">
                  {option.label}
                  {option.hint && <em>{option.hint}</em>}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
