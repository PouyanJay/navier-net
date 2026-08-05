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

/* The four pipeline stages. Each glyph names what the stage holds, not its
   position: a strip of camera frames, the function the network approximates,
   the compute that trains it, and the curves it is judged on. */
export { FilmRoll01Icon as StageDatasetsIcon } from "@hugeicons/core-free-icons";
export { FunctionIcon as StagePhysicsIcon } from "@hugeicons/core-free-icons";
export { CpuChargeIcon as StageSolverIcon } from "@hugeicons/core-free-icons";
export { ChartLineData01Icon as StageResultsIcon } from "@hugeicons/core-free-icons";
