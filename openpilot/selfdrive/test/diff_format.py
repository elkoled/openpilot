# the layout of the diff report comments


def format_report(title: str, description: str, changes: list[list[str]], n_passed: int, errors: list[str], notes: list[str] | None = None,
                  n_changed: int | None = None) -> str:
  icon = "⚠️" if changes else "✅"
  lines = [
    f"## {title}",
    description,
    "Please review any changes carefully to ensure they are expected.\n",
    f"{icon}  {len(changes) if n_changed is None else n_changed} changed, {n_passed} passed, {len(errors)} errors",
  ]
  lines += [f"\nERROR {e}" for e in errors]
  lines += notes or []
  if changes:
    lines.append("<details><summary><b>Show changes</b></summary>\n\n```")
    for change in changes:
      lines += change
    lines.append("```\n</details>")
  return "\n".join(lines)
