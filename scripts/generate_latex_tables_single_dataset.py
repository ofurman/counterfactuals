"""Generate LaTeX tables from markdown metric tables in `results/`.

This script aggregates per-dataset/per-method markdown tables (typically produced by
`scripts/calculate_metrics.py`) and emits two LaTeX tables:

1) Proximity metrics (by default: metrics starting with `proximity_`)
2) Other metrics (everything else)

It supports include/exclude metric lists and can fill missing method/dataset combinations with `--`.
Empty markdown cells are rendered as `nan` (e.g. `||` in a markdown row).
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from omegaconf import OmegaConf

# Select metrics that will be plotted in the output tables.
# Keep this list empty to include all available metrics.
METRICS_TO_SHOW: list[str] = [
    "validity",
    "proximity_euclidean_jaccard",
    "log_density_cf",
]

MetricDirection = Literal["up", "down"]

# User-facing names (edit as needed).
METHOD_NAME_MAP: dict[str, str] = {
    "GLOBE_CE": "GLOBE-CE",
    "WACH_OURS": "WACH",
    "DiceExplainerWrapper": "DiCE",
    "CaseBasedSACE": "SACE",
    "CEM_CF": "CEM",
}

METRIC_TEX_MAP: dict[str, str] = {
    "coverage": "Cov.",
    "validity": "Valid.",
    "actionability": "Act.",
    "sparsity": "Sparse.",
    "proximity_euclidean_hamming": "Euc.-Ham.",
    "proximity_euclidean_jaccard": "Euc.-Jac.",
    "proximity_l1_jaccard": "L1-Jac.",
    "proximity_mad_jaccard": "MAD-Jac.",
    "proximity_l2_jaccard": "L2-Jac.",
    "proximity_mad_hamming": "MAD-Ham.",
    "prob_plausibility": "Prob. Plaus.",
    "log_density_cf": "Log Dens.",
    "lof_scores_cf": "LOF",
    "isolation_forest_scores_cf": "IsoForest",
    "search_time": "Time(s)",
    "cf_search_time": "Time(s)",
    "number_of_instances": "N",
}

DECIMAL_PLACES_MAX = 2
SCIENTIFIC_NOTATION_MIN = 1e-3
SCIENTIFIC_NOTATION_MAX = 1e6


@dataclass(frozen=True)
class MetricMeta:
    """Presentation metadata for a metric."""

    key: str
    tex: str
    direction: MetricDirection


@dataclass(frozen=True)
class ParsedStat:
    """A parsed mean ± std value."""

    mean: float
    std: float
    mean_str: str
    std_str: str


@dataclass(frozen=True)
class Cell:
    """A single metric cell."""

    raw: str
    kind: Literal["missing", " nan", "stat", "number", "text"]
    stat: ParsedStat | None = None
    number: float | None = None


@dataclass(frozen=True)
class RecordKey:
    """Identifies a (model, dataset, method) row."""

    model: str
    dataset: str
    method: str


def _escape_latex(text: str) -> str:
    """Escape common LaTeX special characters.

    Args:
        text: Input text.

    Returns:
        Escaped text.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _split_csvish(value: str) -> list[str]:
    """Split a comma-separated string, keeping empty fields.

    This is only used for interpreting user-provided lists where `a,,b` should
    be preserved as an empty entry.
    """
    return [part.strip() for part in value.split(",")]


def _split_md_row(line: str) -> list[str]:
    """Split a markdown table row into cells.

    Args:
        line: A markdown row, typically starting and ending with `|`.

    Returns:
        List of cell strings (trimmed).
    """
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_md_separator_row(line: str) -> bool:
    """Heuristically detect the header separator row of a markdown table."""
    cells = _split_md_row(line)
    if not cells:
        return False
    for cell in cells:
        normalized = cell.replace(":", "").replace("-", "").strip()
        if normalized != "":
            return False
    return True


def parse_markdown_tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Parse GitHub-style markdown tables.

    Args:
        text: Full markdown file contents.

    Returns:
        List of (headers, rows) tables.
    """
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    tables: list[tuple[list[str], list[list[str]]]] = []
    i = 0
    while i < len(lines) - 1:
        line = lines[i].strip()
        if not line.startswith("|") or "|" not in line:
            i += 1
            continue
        if i + 1 >= len(lines) or not _is_md_separator_row(lines[i + 1]):
            i += 1
            continue

        headers = _split_md_row(lines[i])
        rows: list[list[str]] = []
        i += 2
        while i < len(lines):
            row_line = lines[i].strip()
            if not row_line.startswith("|"):
                break
            row = _split_md_row(lines[i])
            if len(row) < len(headers):
                row = row + [""] * (len(headers) - len(row))
            elif len(row) > len(headers):
                row = row[: len(headers)]
            rows.append(row)
            i += 1
        tables.append((headers, rows))
        continue
    return tables


def _try_parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def parse_cell(raw: str) -> Cell:
    """Parse a markdown cell to a structured representation.

    Rules:
    - Empty string -> `nan`
    - '-' or '--' -> missing (`--`)
    - 'mean ± std' or 'mean +/- std' -> stat
    - numeric -> number
    - otherwise -> text
    """
    s = raw.strip()
    if s == "":
        return Cell(raw=raw, kind="nan")
    # Some upstream exporters represent missing values as empty comma-separated fields (",,")
    # that may end up embedded in markdown cells; treat these as NaN.
    if s.strip(",") == "" and "," in s:
        return Cell(raw=raw, kind="nan")
    if s in {"-", "--"}:
        return Cell(raw=raw, kind="missing")

    for sep in ("±", "+/-"):
        if sep in s:
            left, right = s.split(sep, 1)
            mean_str = left.strip()
            std_str = right.strip()
            mean = _try_parse_float(mean_str)
            std = _try_parse_float(std_str)
            if mean is None or std is None:
                return Cell(raw=raw, kind="text")
            return Cell(
                raw=raw,
                kind="stat",
                stat=ParsedStat(mean=mean, std=std, mean_str=mean_str, std_str=std_str),
            )

    number = _try_parse_float(s)
    if number is not None:
        return Cell(raw=raw, kind="number", number=number)

    if s.lower() in {"nan", "inf", "-inf"}:
        return Cell(raw=s.lower(), kind="text")

    return Cell(raw=raw, kind="text")


def _latex_math_number(value: str) -> str:
    s = value.strip().lower()
    if s == "inf":
        return r"\infty"
    if s == "-inf":
        return r"-\infty"
    return value.strip()


def _format_number(value: float) -> str:
    if not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if value == 0:
        return "0.00"
    abs_value = abs(value)
    if abs_value >= SCIENTIFIC_NOTATION_MAX or abs_value < SCIENTIFIC_NOTATION_MIN:
        exponent = int(math.floor(math.log10(abs_value)))
        mantissa = value / (10**exponent)
        mantissa_str = f"{mantissa:.{DECIMAL_PLACES_MAX}f}"
        return f"{mantissa_str}\\times 10^{{{exponent}}}"
    return f"{value:.{DECIMAL_PLACES_MAX}f}"


def cell_to_latex(cell: Cell, *, bold: bool) -> str:
    """Render a cell to LaTeX.

    Args:
        cell: Parsed cell.
        bold: Whether to boldface (best-in-column).

    Returns:
        LaTeX string for the cell (no trailing `\\`).
    """
    if cell.kind == "missing":
        return "--"
    if cell.kind == "nan":
        return "nan"
    if cell.kind == "stat" and cell.stat is not None:
        # Important: `\pmnan` would be parsed as a single (undefined) control sequence.
        # Always separate `\pm` from the following token by bracing the RHS.
        mean_str = _format_number(cell.stat.mean)
        std_str = _format_number(cell.stat.std)
        inner = f"{_latex_math_number(mean_str)}\\pm{{{_latex_math_number(std_str)}}}"
        if bold:
            inner = rf"\boldsymbol{{{inner}}}"
        return f"${inner}$"
    if cell.kind == "number" and cell.number is not None:
        inner = _latex_math_number(_format_number(cell.number))
        if bold:
            inner = rf"\boldsymbol{{{inner}}}"
        return f"${inner}$"

    # Text / special
    if cell.raw.strip().lower() in {"nan", "inf", "-inf"}:
        inner = _latex_math_number(cell.raw.strip().lower())
        if bold:
            inner = rf"\boldsymbol{{{inner}}}"
        return f"${inner}$"
    return _escape_latex(cell.raw.strip())


def default_direction(metric_key: str) -> MetricDirection:
    """Guess a metric direction (up/down) from its name."""
    key = metric_key.lower()
    if key.startswith("proximity_"):
        return "down"
    if "distance" in key:
        return "down"
    if "time" in key or "runtime" in key:
        return "down"
    if "cost" in key:
        return "down"
    if "lof" in key:
        return "down"
    return "up"


def default_metric_tex(metric_key: str) -> str:
    """Provide a compact default LaTeX header label for common metrics."""
    if metric_key in METRIC_TEX_MAP:
        return METRIC_TEX_MAP[metric_key]
    # Avoid LaTeX errors like "Missing $ inserted" from underscores in plain text headers.
    return metric_key.replace("_", r"\_")


def _arrow(direction: MetricDirection) -> str:
    return r"\uparrow" if direction == "up" else r"\downarrow"


def load_metrics_config(path: Path) -> list[str]:
    """Load metric keys (ordered) from a metrics config YAML."""
    conf = OmegaConf.load(path)
    metrics = list(conf.metrics_to_compute)
    metrics = [metric for metric in metrics if metric != "number_of_instances"]
    return [m for m in metrics if not str(m).endswith("_test")]


def _iter_metric_files(results_dir: Path, pattern: str) -> Iterable[Path]:
    yield from results_dir.rglob(pattern)


def parse_result_filename(path: Path) -> RecordKey | None:
    """Parse `dataset_method_model_metrics.md` into its components."""
    if not path.name.endswith("_metrics.md"):
        return None
    stem = path.stem
    if not stem.endswith("_metrics"):
        return None
    base = stem[: -len("_metrics")]
    parts = base.split("_")
    if len(parts) < 3:
        return None

    model = parts[-1]
    remaining = parts[:-1]

    def norm(value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum())

    folder_norm = norm(path.parent.name)

    # Prefer splitting based on the immediate parent folder name, which typically identifies the method.
    # This prevents incorrect splits when methods contain underscores (e.g. GLOBE_CE).
    best_split: tuple[bool, int, int] | None = None
    # tuple = (exact_match, norm_length, split_index)
    for i in range(1, len(remaining)):
        candidate_method = "_".join(remaining[i:])
        candidate_norm = norm(candidate_method)
        if candidate_norm == "":
            continue
        matches = (
            candidate_norm == folder_norm
            or (candidate_norm in folder_norm)
            or (folder_norm in candidate_norm)
        )
        if not matches:
            continue
        score = (candidate_norm == folder_norm, len(candidate_norm), i)
        if best_split is None or score > best_split:
            best_split = score

    if best_split is not None:
        split_index = best_split[2]
        dataset = "_".join(remaining[:split_index])
        method = "_".join(remaining[split_index:])
        return RecordKey(model=model, dataset=dataset, method=method)

    # Fallback heuristic: assume the last token before the model is the method.
    method = remaining[-1]
    dataset = "_".join(remaining[:-1])
    return RecordKey(model=model, dataset=dataset, method=method)


def load_records(
    results_dir: Path,
    *,
    file_glob: str,
) -> tuple[dict[RecordKey, dict[str, Cell]], set[str]]:
    """Load all records from results markdown files.

    Returns:
        (records, discovered_metrics)
    """
    records: dict[RecordKey, dict[str, Cell]] = {}
    discovered_metrics: set[str] = set()

    for path in sorted(_iter_metric_files(results_dir, file_glob)):
        key = parse_result_filename(path)
        if key is None:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logging.warning("Failed to read %s: %s", path, exc)
            continue

        tables = parse_markdown_tables(text)
        if not tables:
            logging.warning("No markdown tables found in %s", path)
            continue
        headers, rows = tables[0]
        if not rows:
            logging.warning("Empty markdown table in %s", path)
            continue

        row = rows[0]
        metrics = {headers[i]: parse_cell(row[i]) for i in range(len(headers))}
        records[key] = metrics
        for metric in headers:
            if not metric.endswith("_test"):
                discovered_metrics.add(metric)

    return records, discovered_metrics


def _unique_in_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _parse_list_args(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend([item for item in _split_csvish(value) if item != ""])
    return out


def build_metric_meta(
    metric_keys: Iterable[str],
    overrides: dict[str, dict[str, Any]],
) -> dict[str, MetricMeta]:
    """Build metadata for each metric."""
    meta: dict[str, MetricMeta] = {}
    for key in metric_keys:
        override = overrides.get(key, {})
        direction: MetricDirection = override.get("direction", default_direction(key))
        tex = override.get("tex", default_metric_tex(key))
        meta[key] = MetricMeta(key=key, tex=tex, direction=direction)
    return meta


def _cell_mean(cell: Cell) -> float | None:
    if cell.kind == "stat" and cell.stat is not None and math.isfinite(cell.stat.mean):
        return cell.stat.mean
    if cell.kind == "number" and cell.number is not None and math.isfinite(cell.number):
        return cell.number
    return None


def compute_bold_cells(
    row_keys: list[RecordKey],
    values: dict[RecordKey, dict[str, Cell]],
    metric_keys: list[str],
    metric_meta: dict[str, MetricMeta],
) -> dict[tuple[RecordKey, str], bool]:
    """Decide which cells are best-in-column within each (model, dataset)."""
    best: dict[tuple[str, str, str], float] = {}
    tol = 1e-12

    # First pass: compute best value per (model, dataset, metric)
    for rk in row_keys:
        row = values.get(rk, {})
        for metric in metric_keys:
            cell = row.get(metric, Cell(raw="--", kind="missing"))
            mean = _cell_mean(cell)
            if mean is None:
                continue
            k = (rk.model, rk.dataset, metric)
            direction = metric_meta[metric].direction
            if k not in best:
                best[k] = mean
                continue
            current = best[k]
            if direction == "up":
                if mean > current + tol:
                    best[k] = mean
            else:
                if mean < current - tol:
                    best[k] = mean

    # Second pass: mark cells equal-to-best (ties)
    bold: dict[tuple[RecordKey, str], bool] = {}
    for rk in row_keys:
        row = values.get(rk, {})
        for metric in metric_keys:
            cell = row.get(metric, Cell(raw="--", kind="missing"))
            mean = _cell_mean(cell)
            k = (rk.model, rk.dataset, metric)
            if mean is None or k not in best:
                bold[(rk, metric)] = False
                continue
            bold[(rk, metric)] = abs(mean - best[k]) <= tol
    return bold


def _format_model_header(model_tex: str, *, ncols: int) -> str:
    return rf"\multicolumn{{{ncols}}}{{c}}{{{model_tex}}} \\"


def build_latex_table(
    row_keys: list[RecordKey],
    values: dict[RecordKey, dict[str, Cell]],
    *,
    metric_keys: list[str],
    metric_meta: dict[str, MetricMeta],
    caption: str,
    label: str,
    model: str | None = None,
    model_aliases: dict[str, str],
    dataset_aliases: dict[str, str],
    method_aliases: dict[str, str],
    escape_names: bool,
    bold_best: bool,
) -> str:
    """Build a LaTeX `table*` as a string."""
    ncols = 1 + len(metric_keys)
    col_spec = "l|" + ("r" * len(metric_keys))

    header_cells = ["Method"]
    for metric in metric_keys:
        mm = metric_meta[metric]
        header_cells.append(rf"{mm.tex}${_arrow(mm.direction)}$")
    header_row = " & ".join(header_cells) + r" \\"

    # Determine ordering/grouping
    if model is not None:
        row_keys = [rk for rk in row_keys if rk.model == model]
        models = [model]
    else:
        models = _unique_in_order([rk.model for rk in row_keys])
    datasets_by_model: dict[str, list[str]] = {m: [] for m in models}
    methods_by_model_dataset: dict[tuple[str, str], list[str]] = {}
    for rk in row_keys:
        if rk.dataset not in datasets_by_model[rk.model]:
            datasets_by_model[rk.model].append(rk.dataset)
        md_key = (rk.model, rk.dataset)
        methods_by_model_dataset.setdefault(md_key, [])
        if rk.method not in methods_by_model_dataset[md_key]:
            methods_by_model_dataset[md_key].append(rk.method)

    if bold_best:
        bold = compute_bold_cells(row_keys, values, metric_keys, metric_meta)
    else:
        bold = {(rk, metric): False for rk in row_keys for metric in metric_keys}

    def fmt_name(name: str, aliases: dict[str, str]) -> str:
        rendered = aliases.get(name, name)
        return rendered if not escape_names else _escape_latex(rendered)

    lines: list[str] = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{center}")
    lines.append(r"\begin{sc}")
    lines.append(r"\begin{scriptsize}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(header_row)
    lines.append(r"\midrule")

    for model in models:
        if model is None:
            model_tex = fmt_name(model, model_aliases)
            lines.append(_format_model_header(model_tex, ncols=ncols))
            lines.append(r"\midrule")

        for dataset in datasets_by_model[model]:
            methods = methods_by_model_dataset[(model, dataset)]
            for method in methods:
                rk = RecordKey(model=model, dataset=dataset, method=method)
                row = values.get(rk, {})
                method_tex = fmt_name(method, method_aliases)

                metric_cells: list[str] = []
                for metric in metric_keys:
                    cell = row.get(metric, Cell(raw="--", kind="missing"))
                    metric_cells.append(cell_to_latex(cell, bold=bold[(rk, metric)]))

                line = " & ".join([method_tex, *metric_cells]) + r" \\"
                lines.append(line)
            lines.append(r"\midrule")

    # Replace last midrule with bottomrule
    if lines and lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{scriptsize}")
    lines.append(r"\end{sc}")
    lines.append(r"\end{center}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def _parse_overrides_yaml(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    conf = OmegaConf.load(path)
    data = OmegaConf.to_container(conf, resolve=True)
    if not isinstance(data, dict):
        raise ValueError("Overrides YAML must be a mapping at the root.")
    metric_meta = data.get("metric_meta", {})
    if metric_meta is None:
        return {}
    if not isinstance(metric_meta, dict):
        raise ValueError("`metric_meta` must be a mapping.")
    out: dict[str, dict[str, Any]] = {}
    for k, v in metric_meta.items():
        if isinstance(v, dict):
            out[str(k)] = dict(v)
    return out


def _parse_aliases_yaml(path: Path | None, key: str) -> dict[str, str]:
    if path is None:
        return {}
    conf = OmegaConf.load(path)
    data = OmegaConf.to_container(conf, resolve=True)
    if not isinstance(data, dict):
        return {}
    value = data.get(key, {})
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables by aggregating markdown metric tables in `results/`."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to scan for markdown metric tables (default: results).",
    )
    parser.add_argument(
        "--file-glob",
        default="*_metrics.md",
        help="Glob to match metric markdown files (default: *_metrics.md).",
    )
    parser.add_argument(
        "--metrics-conf-path",
        default="counterfactuals/pipelines/conf/metrics/default.yaml",
        help="Metrics config (default: counterfactuals/pipelines/conf/metrics/default.yaml).",
    )
    parser.add_argument(
        "--include-metrics",
        action="append",
        default=[],
        help="Comma-separated list of metrics to include (repeated allowed). If set, restricts output.",
    )
    parser.add_argument(
        "--exclude-metrics",
        action="append",
        default=[],
        help="Comma-separated list of metrics to exclude (repeated allowed).",
    )
    parser.add_argument(
        "--include-methods",
        action="append",
        default=[],
        help="Comma-separated list of methods to include (repeated allowed). If set, restricts output.",
    )
    parser.add_argument(
        "--exclude-methods",
        action="append",
        default=[],
        help="Comma-separated list of methods to exclude (repeated allowed).",
    )
    parser.add_argument(
        "--drop-empty-rows",
        action="store_true",
        help="Omit rows where all metric values are missing.",
    )
    parser.add_argument(
        "--include-dataset",
        default=None,
        help="Single dataset to include.",
    )
    parser.add_argument(
        "--discriminative-model",
        default="MLR",
        help=(
            "Discriminative model to include: MultinomialLogisticRegression or "
            "MultilayerPecreptron (aliases accepted). Default: MLR."
        ),
    )
    parser.add_argument(
        "--caption",
        default="Metrics for a single dataset.",
        help="Caption for the generated table.",
    )
    parser.add_argument(
        "--label",
        default="tab:single_dataset_metrics",
        help="LaTeX label for the generated table.",
    )
    parser.add_argument(
        "--config-yaml",
        help=(
            "Optional YAML to provide `metric_meta`, `model_aliases`, `dataset_aliases`, "
            "and `method_aliases` mappings."
        ),
    )
    parser.add_argument(
        "--no-escape-names",
        action="store_true",
        help="Do not LaTeX-escape model/dataset/method names from aliases.",
    )
    parser.add_argument(
        "--no-bold-best",
        action="store_true",
        help="Disable bolding best-in-column values.",
    )
    parser.add_argument(
        "--output",
        help="Optional output .tex file. If omitted, prints to stdout.",
    )
    return parser.parse_args()


def _resolve_discriminative_model(model_name: str) -> str:
    normalized = model_name.strip().lower()
    aliases = {
        "mlr": "MultinomialLogisticRegression",
        "multinomiallogisticregression": "MultinomialLogisticRegression",
        "mlp": "MultilayerPecreptron",
        "multilayerpecreptron": "MultilayerPecreptron",
    }
    if normalized in aliases:
        return aliases[normalized]
    raise SystemExit(
        "Unsupported --discriminative-model. Use one of: MLR, MultinomialLogisticRegression, "
        "MLP, MultilayerPecreptron."
    )


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = _parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Results dir not found: {results_dir}")

    config_yaml = Path(args.config_yaml) if args.config_yaml else None
    metric_overrides = _parse_overrides_yaml(config_yaml)
    model_aliases = _parse_aliases_yaml(config_yaml, "model_aliases")
    dataset_aliases = _parse_aliases_yaml(config_yaml, "dataset_aliases")
    method_aliases = _parse_aliases_yaml(config_yaml, "method_aliases")

    # Helpful defaults for common classifier names.
    model_aliases = {
        "MultilayerPecreptron": "MLP",
        "MultinomialLogisticRegression": "LR",
        **model_aliases,
    }
    method_aliases = {**METHOD_NAME_MAP, **method_aliases}

    base_metrics = load_metrics_config(Path(args.metrics_conf_path))
    records, discovered = load_records(results_dir, file_glob=args.file_glob)
    if not records:
        raise SystemExit(
            f"No records found under {results_dir} matching {args.file_glob}"
        )

    # Prefer config ordering, but include any discovered (non-test) metrics too.
    all_metrics = _unique_in_order([*base_metrics, *sorted(discovered)])
    all_metrics = [m for m in all_metrics if not m.endswith("_test")]

    include_metrics = (
        _unique_in_order(METRICS_TO_SHOW)
        if METRICS_TO_SHOW
        else _parse_list_args(args.include_metrics)
    )
    exclude_metrics = set(_parse_list_args(args.exclude_metrics))

    include_methods = _parse_list_args(args.include_methods)
    exclude_methods = set(_parse_list_args(args.exclude_methods))

    if include_metrics:
        metric_keys = _unique_in_order(include_metrics)
    else:
        metric_keys = list(all_metrics)
    metric_keys = [
        m
        for m in metric_keys
        if m not in exclude_metrics
        and not m.endswith("_test")
        and m != "number_of_instances"
    ]

    metric_meta = build_metric_meta(metric_keys, metric_overrides)
    selected_model = _resolve_discriminative_model(args.discriminative_model)

    row_keys = sorted(
        records.keys(),
        key=lambda rk: (rk.model, rk.dataset, rk.method),
    )
    if args.include_dataset:
        row_keys = [rk for rk in row_keys if rk.dataset == args.include_dataset]
        if not row_keys:
            available_datasets = _unique_in_order(rk.dataset for rk in records.keys())
            raise SystemExit(
                f"No records found for dataset '{args.include_dataset}'. "
                f"Available datasets: {available_datasets}"
            )

    row_keys = [rk for rk in row_keys if rk.model == selected_model]
    if not row_keys:
        available_models = _unique_in_order(rk.model for rk in records.keys())
        raise SystemExit(
            f"No records found for discriminative model '{selected_model}'. "
            f"Available models: {available_models}"
        )

    datasets = _unique_in_order([rk.dataset for rk in row_keys])
    if len(datasets) != 1:
        raise SystemExit(
            f"Expected exactly one dataset for '{selected_model}', found {len(datasets)}: {datasets}"
        )
    selected_dataset = datasets[0]
    models = [selected_model]
    datasets_by_model: dict[str, list[str]] = {selected_model: [selected_dataset]}
    methods_by_model: dict[str, list[str]] = {selected_model: []}
    for rk in row_keys:
        if rk.method not in methods_by_model[selected_model]:
            methods_by_model[selected_model].append(rk.method)

    # Apply method include/exclude filters (preserve order where possible).
    if include_methods:
        include_methods_ordered = _unique_in_order(include_methods)
        for model in models:
            methods_by_model[model] = [
                m for m in include_methods_ordered if m in methods_by_model[model]
            ]
    for model in models:
        methods_by_model[model] = [
            m for m in methods_by_model[model] if m not in exclude_methods
        ]

    # Expand to a full grid so missing method/dataset combos show up as `--` rows.
    expanded_row_keys: list[RecordKey] = []
    for model in models:
        for dataset in datasets_by_model[model]:
            for method in methods_by_model[model]:
                expanded_row_keys.append(
                    RecordKey(model=model, dataset=dataset, method=method)
                )
    row_keys = expanded_row_keys
    if args.drop_empty_rows:
        filtered: list[RecordKey] = []
        for rk in row_keys:
            row = records.get(rk, {})
            has_value = any(
                row.get(metric, Cell(raw="--", kind="missing")).kind != "missing"
                for metric in metric_keys
            )
            if has_value:
                filtered.append(rk)
        row_keys = filtered

    latex = build_latex_table(
        row_keys,
        records,
        metric_keys=metric_keys,
        metric_meta=metric_meta,
        caption=args.caption,
        label=args.label,
        model=selected_model,
        model_aliases=model_aliases,
        dataset_aliases=dataset_aliases,
        method_aliases=method_aliases,
        escape_names=not args.no_escape_names,
        bold_best=not args.no_bold_best,
    )
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(latex, encoding="utf-8")
        logging.info("Wrote LaTeX tables to %s", out_path)
    else:
        print(latex)  # noqa: T201 - CLI tool output


if __name__ == "__main__":
    main()
