"""Generate API reference pages for the documentation.

This script is executed by mkdocs-gen-files plugin during the build process.
It automatically creates documentation pages for all Python modules in the
counterfactuals package.
"""

from pathlib import Path

import mkdocs_gen_files

# Navigation builder for literate-nav
nav = mkdocs_gen_files.Nav()

# Source directory
src = Path("counterfactuals")

# Modules to exclude from documentation
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
]

# Files to skip
SKIP_FILES = [
    "__init__.py",
    "__main__.py",
]


def should_skip(path: Path) -> bool:
    """Check if a file should be skipped."""
    if path.name in SKIP_FILES:
        return True
    if path.name.startswith("_") and path.name != "__init__.py":
        return True
    for pattern in EXCLUDE_PATTERNS:
        if path.match(pattern):
            return True
    return False


# Process all Python files
for path in sorted(src.rglob("*.py")):
    # Skip excluded files
    if should_skip(path):
        continue

    # Build module path (e.g., counterfactuals/datasets/base.py -> counterfactuals.datasets.base)
    module_path = path.relative_to(src.parent).with_suffix("")

    # Build documentation path
    doc_path = path.relative_to(src.parent).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    # Build navigation parts
    parts = tuple(module_path.parts)

    # Skip if it's just the package __init__
    if parts[-1] == "__init__":
        parts = parts[:-1]
        if not parts:
            continue
        doc_path = doc_path.with_name("index.md")
        full_doc_path = Path("reference", doc_path)

    # Add to navigation
    nav[parts] = doc_path.as_posix()

    # Create the documentation page
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        fd.write(f"# {parts[-1]}\n\n")
        fd.write(f"::: {identifier}\n")

    # Set edit path to the source file
    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Write navigation file
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
