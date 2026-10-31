import { describe, expect, it } from "vitest";

import { decodeGrid } from "../src/lib/api/client";

function pack(header: object, planes: Float32Array[]): ArrayBuffer {
  const head = new TextEncoder().encode(JSON.stringify(header));
  const body = planes.reduce((n, p) => n + p.byteLength, 0);
  const out = new Uint8Array(8 + head.length + body);
  out.set(new TextEncoder().encode("NVST"), 0);
  new DataView(out.buffer).setUint32(4, head.length, true);
  out.set(head, 8);
  let offset = 8 + head.length;
  for (const p of planes) {
    out.set(new Uint8Array(p.buffer), offset);
    offset += p.byteLength;
  }
  return out.buffer;
}

describe("decodeGrid", () => {
  it("reads the header and one plane per quantity, even with an odd header length", () => {
    const header = {
      shape: [2, 3],
      extent_um: [0, 3, 0, 2],
      quantities: [{ id: "u" }, { id: "p" }],
      run: "r",
      frame: 1,
      t_ms: 0,
      t_star: 0,
      boundary_um: [],
      nose_um: null,
    };
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([7, 8, 9, 10, 11, 12]);
    const grid = decodeGrid(pack({ ...header, pad: "x" }, [a, b]));
    expect(grid.header.shape).toEqual([2, 3]);
    expect(Array.from(grid.planes[0])).toEqual([1, 2, 3, 4, 5, 6]);
    expect(Array.from(grid.planes[1])).toEqual([7, 8, 9, 10, 11, 12]);
  });

  it("rejects a buffer without the studio magic", () => {
    const bad = new TextEncoder().encode("JUNKJUNKJUNK").buffer;
    expect(() => decodeGrid(bad)).toThrow(/magic/);
  });
});
