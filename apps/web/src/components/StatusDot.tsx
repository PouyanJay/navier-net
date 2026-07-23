type DotTone = "default" | "accent" | "green" | "amber" | "red";

interface StatusDotProps {
  tone?: DotTone;
  label: string;
}

/**
 * A colored dot with a text label. State is conveyed by the label as well as the
 * color, never by color alone (enterprise-ui §3).
 */
export function StatusDot({ tone = "default", label }: StatusDotProps) {
  return (
    <span className="dot" data-tone={tone}>
      {label}
    </span>
  );
}
