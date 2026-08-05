/** Types for the brand asset generator, which is plain ESM so `node` can run it. */

export interface GeneratedPoint {
  cx: number;
  cy: number;
  r: number;
  opacity: number;
}

export function points(count?: number): GeneratedPoint[];
export function buildSvg(): string;
export function buildPng(size: number): Buffer;
export function buildIco(size: number): Buffer;
