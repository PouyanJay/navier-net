import type { ButtonHTMLAttributes } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "primary";
}

/** A real <button>; `primary` uses the accent for the main action on a surface. */
export function Button({ variant = "default", className, ...rest }: ButtonProps) {
  const classes = ["btn", variant === "primary" ? "primary" : "", className]
    .filter(Boolean)
    .join(" ");
  return <button className={classes} {...rest} />;
}
