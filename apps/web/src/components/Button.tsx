import type { ButtonHTMLAttributes } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "primary";
  /** `sm` is the compact form for secondary actions sitting in a panel header. */
  size?: "default" | "sm";
}

/** A real <button>; `primary` uses the accent for the main action on a surface. */
export function Button({
  variant = "default",
  size = "default",
  className,
  ...rest
}: ButtonProps) {
  const classes = [
    "btn",
    variant === "primary" ? "primary" : "",
    size === "sm" ? "sm" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");
  return <button className={classes} {...rest} />;
}
