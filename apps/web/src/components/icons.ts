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

/* Status glyphs for callouts and the destructive/rebuild actions. Each names
   what it means, not what it looks like, so a callout tone maps straight onto
   one and swapping libraries never touches a view. */
export { InformationCircleIcon as InfoIcon } from "@hugeicons/core-free-icons";
export { Alert02Icon as WarningIcon } from "@hugeicons/core-free-icons";
export { AlertCircleIcon as ErrorIcon } from "@hugeicons/core-free-icons";
export { Delete02Icon as DeleteIcon } from "@hugeicons/core-free-icons";
export { Refresh01Icon as RerunIcon } from "@hugeicons/core-free-icons";
export { FloppyDiskIcon as SaveIcon } from "@hugeicons/core-free-icons";
/** The recorded sheet of values behind a series, opened for editing. */
export { File01Icon as ConditionsIcon } from "@hugeicons/core-free-icons";

/* Menu furniture: the glyph that says a control opens a list, and the one that
   marks the option currently chosen in it. */
export { ArrowDown01Icon as MenuOpenIcon } from "@hugeicons/core-free-icons";
export { Tick02Icon as ChosenIcon } from "@hugeicons/core-free-icons";

/* Chart actions: take a copy of what is on screen, and open it larger. */
export { Download04Icon as DownloadIcon } from "@hugeicons/core-free-icons";
export { Maximize01Icon as ExpandIcon } from "@hugeicons/core-free-icons";
