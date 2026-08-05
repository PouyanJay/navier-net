import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";

// Self-hosted, no CDN, no FOUT (DESIGN_SYSTEM §4.1). The two variable faces
// ship one file across their whole weight range; the mono is subset to the
// three weights actually used.
// The weight axis only. Instrument Sans's full file carries a width axis this
// app never varies (19KB vs 11KB), and Newsreader's carries an optical-size
// axis (87KB vs 36KB) for type that is only ever set at one size.
import "@fontsource-variable/instrument-sans/wght.css";
import "@fontsource-variable/instrument-sans/wght-italic.css";
import "@fontsource-variable/newsreader/wght.css";
import "@fontsource-variable/newsreader/wght-italic.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import "@fontsource/jetbrains-mono/600.css";
import "katex/dist/katex.min.css";

import "./tokens.css";
import { App } from "./App";
import { ToastProvider } from "./components/Toast";
import { applyTheme, initialTheme } from "./theme";

// Set the theme before first paint to avoid a flash.
applyTheme(initialTheme());

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <ToastProvider>
        <App />
      </ToastProvider>
    </BrowserRouter>
  </StrictMode>,
);
