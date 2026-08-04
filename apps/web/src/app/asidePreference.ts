/** Whether the stage aside is folded, remembered across visits.
 *
 * A researcher who folds it away to give the canvas the full width should not
 * have to fold it again on the next visit. Unlike the theme's helper this one
 * guards the storage calls, because the aside is chrome a refused store must
 * not be able to take down.
 */

const STORAGE_KEY = "naviernet-aside-collapsed";

export function initialAsideCollapsed(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    // Private browsing and similar can refuse storage entirely; the rail's
    // default state is not worth failing the whole shell over.
    return false;
  }
}

export function storeAsideCollapsed(collapsed: boolean): void {
  try {
    localStorage.setItem(STORAGE_KEY, collapsed ? "1" : "0");
  } catch {
    /* see above: an unavailable store just means the choice is not remembered */
  }
}
