import type { ReactNode } from "react";

export interface Column<T> {
  header: string;
  cell: (row: T) => ReactNode;
  /** Right-align + monospace, for numeric columns. */
  num?: boolean;
}

interface TableProps<T> {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  rowTone?: (row: T) => "amber" | undefined;
  caption?: string;
}

/** A real semantic <table> for tabular data (enterprise-ui §3). */
export function Table<T>({
  columns,
  rows,
  rowKey,
  rowTone,
  caption,
}: TableProps<T>) {
  return (
    <div className="tbl-wrap">
      <table className="tbl">
        {caption && <caption className="visually-hidden">{caption}</caption>}
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={col.header}
                className={col.num ? "num" : undefined}
                scope="col"
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={rowKey(row)} data-tone={rowTone?.(row)}>
              {columns.map((col) => (
                <td key={col.header} className={col.num ? "num" : undefined}>
                  {col.cell(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
