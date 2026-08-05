import { Component, type ReactNode } from "react";

import { Callout } from "./Callout";

interface ErrorBoundaryProps {
  /** Names the surface in the fallback ("the Training tab"). */
  label: string;
  /** Remounts children when it changes (e.g. run/tab id), clearing the error. */
  resetKey?: string;
  children: ReactNode;
}

interface ErrorBoundaryState {
  message: string | null;
}

/**
 * Contains a render crash to the panel that threw it: one broken tab must
 * never blank the whole workspace. The error is surfaced, not swallowed.
 */
export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { message: null };

  static getDerivedStateFromError(error: unknown): ErrorBoundaryState {
    return { message: error instanceof Error ? error.message : String(error) };
  }

  componentDidUpdate(previous: ErrorBoundaryProps) {
    if (previous.resetKey !== this.props.resetKey && this.state.message)
      this.setState({ message: null });
  }

  render() {
    if (this.state.message)
      return (
        <Callout tone="error" title={`${this.props.label} crashed`}>
          {this.state.message}. The rest of the page is unaffected; switch tabs
          or reselect the run to retry.
        </Callout>
      );
    return this.props.children;
  }
}
