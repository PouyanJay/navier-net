import { useEffect, useRef, useState } from "react";

export interface PaletteAction {
  group: string;
  label: string;
  shortcut?: string;
  run: () => void;
}

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  actions: PaletteAction[];
}

/** ⌘K command palette: filter, arrow-select, run. */
export function CommandPalette({ open, onClose, actions }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [index, setIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const matches = actions.filter((action) =>
    action.label.toLowerCase().includes(query.toLowerCase()),
  );

  useEffect(() => {
    if (open) {
      setQuery("");
      setIndex(0);
      inputRef.current?.focus();
    }
  }, [open]);

  if (!open) return null;

  const run = (action: PaletteAction) => {
    onClose();
    action.run();
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Escape") onClose();
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setIndex((i) => Math.min(i + 1, matches.length - 1));
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      setIndex((i) => Math.max(i - 1, 0));
    }
    if (event.key === "Enter" && matches[index]) run(matches[index]);
  };

  let lastGroup = "";
  return (
    <div className="pal-ov" onClick={onClose}>
      <div
        className="pal"
        role="dialog"
        aria-label="Command palette"
        onClick={(event) => event.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        <input
          ref={inputRef}
          value={query}
          placeholder="Search or run a command…"
          aria-label="Search commands"
          onChange={(event) => {
            setQuery(event.target.value);
            setIndex(0);
          }}
        />
        <div className="pal-list" role="listbox" aria-label="Commands">
          {matches.length === 0 && <p className="pal-empty">No matching command.</p>}
          {matches.map((action, i) => {
            const header = action.group !== lastGroup ? action.group : null;
            lastGroup = action.group;
            return (
              <div key={`${action.group}:${action.label}`}>
                {header && <div className="pal-group">{header}</div>}
                <button
                  type="button"
                  role="option"
                  aria-selected={i === index}
                  className="pal-item"
                  data-selected={i === index || undefined}
                  onMouseEnter={() => setIndex(i)}
                  onClick={() => run(action)}
                >
                  <span>{action.label}</span>
                  {action.shortcut && <kbd>{action.shortcut}</kbd>}
                </button>
              </div>
            );
          })}
        </div>
        <div className="pal-foot">↑↓ navigate · ↵ select · esc close</div>
      </div>
    </div>
  );
}
