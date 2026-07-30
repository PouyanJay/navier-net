import { type ReactNode, useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { errorMessage } from "../lib/errors";
import { Button } from "./Button";
import { Callout } from "./Callout";

interface ConfirmDeleteDialogProps {
  /** Modal heading, e.g. "Delete run". */
  title: string;
  /** The destructive button's label, e.g. "Delete run". */
  confirmLabel: string;
  /** The specific consequence, e.g. "Delete <b>demo_run</b> and all its outputs?". */
  children: ReactNode;
  /** Runs the delete; rejects to surface the reason without closing. On success the
   * caller unmounts this dialog. */
  onConfirm: () => Promise<void>;
  onClose: () => void;
}

/**
 * A confirm-and-delete overlay for any destructive, irreversible action. Portalled to
 * <body>, focus-managed (Cancel autofocused, Escape closes), and it keeps itself open on
 * failure to show why. The one place the platform gates a delete behind explicit
 * approval, so every delete reads the same.
 */
export function ConfirmDeleteDialog({
  title,
  confirmLabel,
  children,
  onConfirm,
  onClose,
}: ConfirmDeleteDialogProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !busy) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy, onClose]);

  const runConfirm = () => {
    setBusy(true);
    setError(null);
    // On success the caller unmounts this dialog, so we leave `busy` set; only a
    // failure returns control here to surface the reason.
    onConfirm().catch((err) => {
      setError(errorMessage(err));
      setBusy(false);
    });
  };

  return createPortal(
    <div
      className="modal-ov"
      role="presentation"
      // Contain clicks so a portalled event does not bubble to whatever opened the
      // dialog (portals re-dispatch up the component tree).
      onClick={(e) => e.stopPropagation()}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && !busy) onClose();
      }}
    >
      <div
        className="modal"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-del-title"
        aria-describedby="confirm-del-desc"
      >
        <div className="hd">
          <h2 id="confirm-del-title">{title}</h2>
          <span className="sub">permanent</span>
        </div>
        <div className="body">
          <p id="confirm-del-desc" className="del-msg">
            {children}
          </p>
          <Callout tone="caution">This cannot be undone.</Callout>
          {error && <Callout tone="error">{error}</Callout>}
          <div className="pform-actions">
            <Button variant="danger" onClick={runConfirm} disabled={busy}>
              {busy ? "Deleting…" : confirmLabel}
            </Button>
            <Button onClick={onClose} disabled={busy} autoFocus>
              Cancel
            </Button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
