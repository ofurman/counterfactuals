#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <input_dir> <output_file>" >&2
  echo "Concatenate .md files in <input_dir> into <output_file>." >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 1
fi

input_dir="$1"
output_file="$2"

if [[ ! -d "$input_dir" ]]; then
  echo "Error: input directory not found: $input_dir" >&2
  exit 1
fi

shopt -s nullglob
md_files=("$input_dir"/*.md)
shopt -u nullglob

if [[ ${#md_files[@]} -eq 0 ]]; then
  echo "Error: no .md files found in $input_dir" >&2
  exit 1
fi

{
  for file_path in "${md_files[@]}"; do
    file_name="$(basename "$file_path")"
    echo "----- $file_name -----"
    cat "$file_path"
    echo
    echo
  done
} >"$output_file"

echo "Wrote concatenated markdown to $output_file"
