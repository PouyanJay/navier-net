/** The field table: every user-facing input, declared once on the Python side
 * (`naviernet.config.fields`) and rendered here from a committed copy.
 *
 * The copy is `fieldTable.json`, written by `scripts/export_field_table.py`;
 * the Python test suite fails when it is stale, so what the forms offer is
 * what the pipeline's schema declares: the same keys, options, bounds and
 * defaults, with no second hand-typed list to drift. `GET /api/config/fields`
 * serves the same table for any other client.
 */

import table from "./fieldTable.json";

export type FieldKind = "float" | "int" | "bool" | "enum";
export type FieldSection = "conditions" | "physics" | "training" | "acceptance";
/** Who sets the value: the series' conditions file, the launch form, or the
 * advanced overrides only. */
export type FieldOwner = "series" | "form" | "override";

export interface FieldSpec {
  /** The Hydra path the value composes to, e.g. "experiment.T_inlet_C". */
  key: string;
  /** The short name the request models and the forms use. */
  name: string;
  label: string;
  kind: FieldKind;
  section: FieldSection;
  owner: FieldOwner;
  help: string;
  unit: string;
  /** The admitted values of an enum; empty otherwise. */
  options: string[];
  lo: number | null;
  hi: number | null;
  /** Null is a value: "not set, take the derived default". */
  optional: boolean;
  /** Baked into the tensors; an edit needs a re-preprocess. */
  baked: boolean;
  /** Reachable, but not a form control. */
  advanced: boolean;
  /** The schema's default; null when the series config must supply it. */
  default: number | string | boolean | null;
}

export const FIELD_TABLE: readonly FieldSpec[] = table as FieldSpec[];

const BY_NAME = new Map(FIELD_TABLE.map((spec) => [spec.name, spec]));

/** The entries matching every given filter, in table order. */
export function fieldSpecs(filter: {
  section?: FieldSection;
  owner?: FieldOwner;
  advanced?: boolean;
}): FieldSpec[] {
  return FIELD_TABLE.filter(
    (spec) =>
      (filter.section === undefined || spec.section === filter.section) &&
      (filter.owner === undefined || spec.owner === filter.owner) &&
      (filter.advanced === undefined || spec.advanced === filter.advanced),
  );
}

/** The entry for a short name; throws on a name the table lacks, which is a
 * bug in the caller, never a runtime condition. */
export function fieldByName(name: string): FieldSpec {
  const spec = BY_NAME.get(name);
  if (!spec) throw new Error(`no field table entry named ${name}`);
  return spec;
}

/** An option value as the interface reads it: "no_slip" → "no slip". */
export const optionLabel = (value: string): string => value.replace(/_/g, " ");

/** The option list a select renders for an enum entry. */
export const fieldOptions = (spec: FieldSpec) =>
  spec.options.map((value) => ({ value, label: optionLabel(value) }));
