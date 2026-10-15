/** The field table copy the forms render from, held against the forms. */

import { describe, expect, it } from "vitest";

import { FIELD_TABLE, fieldByName, fieldSpecs } from "../src/lib/fields";
import { ALL_CONDITIONS, BAKED_KEYS } from "../src/views/datasets/conditions";
import { FORM_DEFAULTS } from "../src/views/solver/form";

describe("the field table", () => {
  it("names every entry once", () => {
    const names = FIELD_TABLE.map((spec) => spec.name);
    expect(new Set(names).size).toBe(names.length);
  });

  it("offers each enum its default among its options", () => {
    for (const spec of FIELD_TABLE.filter((s) => s.kind === "enum")) {
      expect(spec.options.length).toBeGreaterThanOrEqual(2);
      expect(spec.options).toContain(spec.default);
    }
  });

  it("starts the solver form at the table's defaults for every form-owned entry", () => {
    // The form's literal defaults exist for typing; the table is the truth.
    for (const spec of fieldSpecs({ owner: "form" })) {
      expect(
        FORM_DEFAULTS[spec.name as keyof typeof FORM_DEFAULTS],
        `FORM_DEFAULTS.${spec.name}`,
      ).toBe(spec.default);
    }
  });

  it("renders every series condition in the conditions editor", () => {
    const editable = new Set(ALL_CONDITIONS.map((spec) => spec.key));
    for (const spec of fieldSpecs({ section: "conditions", advanced: false })) {
      expect(editable.has(spec.name), `${spec.name} in the editor`).toBe(true);
    }
  });

  it("reads the baked fields off the table rather than a second list", () => {
    expect([...BAKED_KEYS].sort()).toEqual(
      FIELD_TABLE.filter((s) => s.baked)
        .map((s) => s.name)
        .sort(),
    );
  });

  it("throws on a name the table lacks, as a caller bug", () => {
    expect(() => fieldByName("no_such_input")).toThrow(/no field table entry/);
  });
});
