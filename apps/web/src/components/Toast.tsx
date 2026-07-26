import {
  createContext,
  useCallback,
  useContext,
  useRef,
  useState,
  type ReactNode,
} from "react";

const TOAST_MS = 3600;

export interface ToastMessage {
  id: number;
  title: string;
  sub?: string;
  tone: "ok" | "err" | "default";
}

type PushToast = (
  title: string,
  sub?: string,
  tone?: ToastMessage["tone"],
) => void;

const ToastContext = createContext<PushToast>(() => {});

/** Fire-and-forget notifications (bottom-right, auto-dismiss). */
export function useToast(): PushToast {
  return useContext(ToastContext);
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const nextId = useRef(0);

  const push = useCallback<PushToast>((title, sub, tone = "default") => {
    const id = nextId.current++;
    setToasts((prev) => [...prev, { id, title, sub, tone }]);
    window.setTimeout(
      () => setToasts((prev) => prev.filter((t) => t.id !== id)),
      TOAST_MS,
    );
  }, []);

  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((toast) => (
          <div key={toast.id} className="toast" data-tone={toast.tone}>
            <b>{toast.title}</b>
            {toast.sub && <small>{toast.sub}</small>}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
