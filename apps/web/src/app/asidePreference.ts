/** Whether the secondary rail is collapsed, remembered across visits.
 *
 * Kept beside the theme's own storage helper and read the same way: a
 * researcher who folds the rail away to give the frame strip the full canvas
 * should not have to fold it again on the next visit.
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
