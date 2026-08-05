/**
 * Generates the brand asset set from the mark's geometry.
 *
 *   node scripts/build-brand.mjs
 *
 * The favicons are rasterised here rather than by a toolchain dependency: the
 * mark is eleven circles, so a supersampled scanline fill and a zlib deflate is
 * the whole rasteriser, and the repo stays free of an image library it would
 * otherwise use exactly once.
 *
 * tests/brand.test.ts asserts the committed SVG still matches the component's
 * geometry, so this script drifting from BrandMark.tsx is a test failure rather
 * than a silently stale asset.
 */

import { deflateSync } from "node:zlib";
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";

// Kept in step with src/components/BrandMark.tsx by tests/brand.test.ts.
const VIEWBOX = 24;
const CENTRE = VIEWBOX / 2;
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
const CORE_RADIUS = 1.5;
const ARM_GROWTH = 0.83;
const CORE_DOT = 2.15;
const DOT_TAPER = 0.1;
const FADE = 0.055;
const COUNT = 11;

/** The light and dark theme's --primary, so the mark matches the app it opens. */
const INK_LIGHT = "#2e6cca";
const INK_DARK = "#7fb2f0";
/** One value for the raster fallbacks, which cannot switch on the theme. */
const INK_FLAT = [0x3f, 0x83, 0xdb];

export function points(count = COUNT) {
  return Array.from({ length: count }, (_, i) => {
    const angle = i * GOLDEN_ANGLE;
    const radius = CORE_RADIUS + i * ARM_GROWTH;
    return {
      cx: CENTRE + radius * Math.cos(angle),
      cy: CENTRE + radius * Math.sin(angle),
      r: Math.max(0.4, CORE_DOT - i * DOT_TAPER),
      opacity: Math.max(0.15, 1 - i * FADE),
    };
  });
}

const round = (n) => Number(n.toFixed(3));

/**
 * The SVG favicon carries both inks and switches on the OS theme, which is the
 * one place a favicon can honour a theme at all.
 */
export function buildSvg() {
  const circles = points()
    .map(
      (p) =>
        `  <circle cx="${round(p.cx)}" cy="${round(p.cy)}" r="${round(p.r)}" opacity="${round(p.opacity)}"/>`,
    )
    .join("\n");

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${VIEWBOX} ${VIEWBOX}" role="img" aria-label="NavierNet">
  <style>
    circle { fill: ${INK_LIGHT}; }
    @media (prefers-color-scheme: dark) { circle { fill: ${INK_DARK}; } }
  </style>
${circles}
</svg>
`;
}

/** Renders the mark to an RGBA buffer, supersampled for antialiased edges. */
function raster(size, samples = 4) {
  const scale = size / VIEWBOX;
  const pixels = Buffer.alloc(size * size * 4);
  const dots = points().map((p) => ({
    cx: p.cx * scale,
    cy: p.cy * scale,
    r: p.r * scale,
    opacity: p.opacity,
  }));
  const step = 1 / samples;
  const offset = step / 2;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let coverage = 0;
      for (let sy = 0; sy < samples; sy++) {
        for (let sx = 0; sx < samples; sx++) {
          const px = x + offset + sx * step;
          const py = y + offset + sy * step;
          // A sample takes the strongest dot covering it, so overlapping dots
          // compose the way the SVG's painted circles do rather than summing.
          let best = 0;
          for (const dot of dots) {
            const dx = px - dot.cx;
            const dy = py - dot.cy;
            if (dx * dx + dy * dy <= dot.r * dot.r && dot.opacity > best) {
              best = dot.opacity;
            }
          }
          coverage += best;
        }
      }
      const alpha = coverage / (samples * samples);
      if (alpha <= 0) continue;
      const i = (y * size + x) * 4;
      pixels[i] = INK_FLAT[0];
      pixels[i + 1] = INK_FLAT[1];
      pixels[i + 2] = INK_FLAT[2];
      pixels[i + 3] = Math.round(alpha * 255);
    }
  }
  return pixels;
}

const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

function crc32(buf) {
  let c = 0xffffffff;
  for (const byte of buf) c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([length, body, crc]);
}

export function buildPng(size) {
  const pixels = raster(size);
  // Each scanline is prefixed with filter type 0 (None).
  const raw = Buffer.alloc(size * (size * 4 + 1));
  for (let y = 0; y < size; y++) {
    raw[y * (size * 4 + 1)] = 0;
    pixels.copy(raw, y * (size * 4 + 1) + 1, y * size * 4, (y + 1) * size * 4);
  }

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // colour type: RGBA
  ihdr[10] = 0; // deflate
  ihdr[11] = 0; // adaptive filtering
  ihdr[12] = 0; // no interlace

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

/** An .ico wrapping a PNG, which every browser in support has accepted for years. */
export function buildIco(size) {
  const png = buildPng(size);
  const header = Buffer.alloc(6);
  header.writeUInt16LE(0, 0); // reserved
  header.writeUInt16LE(1, 2); // type: icon
  header.writeUInt16LE(1, 4); // one image

  const entry = Buffer.alloc(16);
  entry[0] = size === 256 ? 0 : size;
  entry[1] = size === 256 ? 0 : size;
  entry[2] = 0; // palette
  entry[3] = 0; // reserved
  entry.writeUInt16LE(1, 4); // colour planes
  entry.writeUInt16LE(32, 6); // bits per pixel
  entry.writeUInt32BE(0, 8);
  entry.writeUInt32LE(png.length, 8);
  entry.writeUInt32LE(header.length + entry.length, 12);

  return Buffer.concat([header, entry, png]);
}

const ASSETS = [
  ["public/brand/navnet-mark.svg", () => Buffer.from(buildSvg(), "utf8")],
  ["public/favicon-32.png", () => buildPng(32)],
  ["public/apple-touch-icon.png", () => buildPng(180)],
  ["public/favicon.ico", () => buildIco(48)],
];

if (process.argv[1]?.endsWith("build-brand.mjs")) {
  for (const [path, build] of ASSETS) {
    const full = resolve(path);
    mkdirSync(dirname(full), { recursive: true });
    const data = build();
    writeFileSync(full, data);
    console.log(`${path.padEnd(34)} ${String(data.length).padStart(6)} bytes`);
  }
}
