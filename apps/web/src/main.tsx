import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";

import "@fontsource-variable/geist";
import "@fontsource-variable/geist/wght-italic.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import "@fontsource/jetbrains-mono/600.css";
import "@fontsource/crimson-pro/400.css";
import "@fontsource/crimson-pro/400-italic.css";
import "@fontsource/crimson-pro/600.css";
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
