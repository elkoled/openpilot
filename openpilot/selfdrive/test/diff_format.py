# the layout of the diff report comments


def format_report(title: str, description: str, changes: list[tuple[str, list[str]]], n_passed: int, errors: list[str],
                  max_len: int | None = None) -> str:
  # changes: each changed item's name and its lines, max_len: as many changes as fit
  icon = "⚠️" if changes else "✅"
  lines = [
    f"## {title}",
    description,
    "Please review any changes carefully to ensure they are expected.\n",
    f"{icon}  {len(changes)} changed, {n_passed} passed, {len(errors)} errors",
  ]
  if errors:
    lines.append("<details><summary><b>Show errors</b></summary>\n\n```")
    lines += [f"\n{e}" for e in dict.fromkeys(errors)]
    lines.append("```\n</details>")
  if changes:
    lines.append("<details><summary><b>Show changes</b></summary>\n\n```")
    size = sum(len(line) + 1 for line in lines)
    for i, (name, change) in enumerate(changes):
      size += len(name) + 2 + sum(len(line) + 1 for line in change)
      # leave room for the closing lines
      if max_len is not None and size > max_len - 100:
        lines.append(f"\n(... {len(changes) - i} more)")
        break
      lines += [f"\n{name}", *change]
    lines.append("```\n</details>")
  return "\n".join(lines)
