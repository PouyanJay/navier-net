/**
 * The app's icon vocabulary, mapped onto HugeIcons.
 *
 * Every glyph the interface uses is named here for its ROLE, not its picture,
 * which is what DESIGN_SYSTEM.md §7 asks for. Two things follow from that:
 * only the glyphs listed here reach the bundle, and swapping icon libraries is
 * an edit to this one file rather than a search across the app.
 *
 * HugeIcons draws on a 24 grid in `currentColor` at 1.5 stroke, so icons
 * inherit their parent's ink and need no per-state styling.
 */

export { HugeiconsIcon } from "@hugeicons/react";

export { GithubIcon as SourceIcon } from "@hugeicons/core-free-icons";
export { Moon02Icon as ThemeDarkIcon } from "@hugeicons/core-free-icons";
export { Sun03Icon as ThemeLightIcon } from "@hugeicons/core-free-icons";
export { Share08Icon as ShareIcon } from "@hugeicons/core-free-icons";
export { FileExportIcon as ExportReportIcon } from "@hugeicons/core-free-icons";
/** The workspace's own settings and account, at the foot of the pipeline rail. */
export { Settings02Icon as SystemIcon } from "@hugeicons/core-free-icons";
