import * as d3 from "d3";
import { useEffect, useRef } from "react";

import type { ModelArchitecture } from "../../lib/api";

const WIDTH = 680;
const HEIGHT = 300;
const MARGIN = { top: 16, right: 16, bottom: 40, left: 16 };
const INNER_W = WIDTH - MARGIN.left - MARGIN.right;
const INNER_H = HEIGHT - MARGIN.top - MARGIN.bottom;
const MAX_NODES = 6; // display cap per column; real counts go in the labels

interface Column {
  label: string;
  nodes: string[]; // node labels (empty string = anonymous)
  isField: boolean; // output column, drawn in the field color
  tooltip: string; // hover readout with the real dimensions
}

type G = d3.Selection<SVGGElement, unknown, null, undefined>;

function columns(model: ModelArchitecture): Column[] {
  const anon = Array.from({ length: MAX_NODES }, () => "");
  return [
    {
      label: "input · x, y, t",
      nodes: ["x", "y", "t"],
      isField: false,
      tooltip: "nondimensional space-time input (x*, y*, t*)",
    },
    {
      label: `Fourier · ${2 * model.fourier_feats}`,
      nodes: anon,
      isField: false,
      tooltip: `${model.fourier_feats} Fourier pairs · scale σ_B = ${model.fourier_scale}`,
    },
    {
      label: `hidden · ${model.hidden} × ${model.layers}`,
      nodes: anon,
      isField: false,
      tooltip: `${model.layers} hidden layers × ${model.hidden} units${
        model.nodewise_activation ? " · adaptive tanh" : ""
      }`,
    },
    {
      label: `fields · ${model.fields.length}`,
      nodes: model.fields.slice(0, MAX_NODES),
      isField: true,
      tooltip: `one MLP head per field: ${model.fields.join(", ")}`,
    },
  ];
}

function nodeY(index: number, total: number): number {
  if (total === 1) return INNER_H / 2;
  const span = Math.min(INNER_H, total * 34);
  const top = (INNER_H - span) / 2;
  return top + (index / (total - 1)) * span;
}

function drawEdges(g: G, cols: Column[], x: (i: number) => number): void {
  const edges = g.append("g").attr("class", "topo-edges");
  for (let c = 0; c < cols.length - 1; c++) {
    const a = cols[c].nodes;
    const b = cols[c + 1].nodes;
    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < b.length; j++) {
        edges
          .append("line")
          .attr("x1", x(c))
          .attr("y1", nodeY(i, a.length))
          .attr("x2", x(c + 1))
          .attr("y2", nodeY(j, b.length));
      }
    }
  }
}

function drawColumn(g: G, col: Column, cx: number): void {
  const group = g.append("g");
  col.nodes.forEach((label, i) => {
    const cy = nodeY(i, col.nodes.length);
    group
      .append("circle")
      .attr("class", col.isField ? "topo-node field" : "topo-node")
      .attr("cx", cx)
      .attr("cy", cy)
      .attr("r", 7)
      // Real dimensions on hover (and as the node's accessible name).
      .append("title")
      .text(col.tooltip);
    if (label) {
      group
        .append("text")
        .attr("class", "topo-node-label")
        .attr("x", cx)
        .attr("y", cy)
        .attr("dy", "0.32em")
        .attr("text-anchor", "middle")
        .text(label);
    }
  });
  group
    .append("text")
    .attr("class", "topo-axis")
    .attr("x", cx)
    .attr("y", INNER_H + 24)
    .attr("text-anchor", "middle")
    .text(col.label);
}

/**
 * Schematic of the field-ensemble: input (x,y,t) -> Fourier features -> hidden
 * layers (adaptive tanh) -> per-field outputs. D3 owns geometry; CSS tokens own
 * color. Display nodes are capped; real widths are in the column labels.
 */
export function TopologyChart({ model }: { model: ModelArchitecture }) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    const cols = columns(model);
    const g = svg
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);
    const x = d3
      .scalePoint<number>()
      .domain(cols.map((_, i) => i))
      .range([40, INNER_W - 40]);
    const cx = (i: number) => x(i) ?? 0;

    drawEdges(g, cols, cx);
    cols.forEach((col, i) => drawColumn(g, col, cx(i)));
  }, [model]);

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="Model topology: input to Fourier features to hidden layers to per-field outputs."
    />
  );
}
