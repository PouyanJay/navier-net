/** Normalize any thrown value to a display string. */
export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
