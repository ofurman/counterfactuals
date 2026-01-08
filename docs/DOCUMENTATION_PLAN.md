# Documentation Plan for Counterfactuals Library

## Overview

This plan outlines the structure and content for comprehensive documentation using **MkDocs Material** with **mkdocstrings** for automatic API documentation.

---

## 1. Technical Setup

### Required Dependencies

```bash
pip install mkdocs-material mkdocstrings[python] mkdocs-gen-files mkdocs-literate-nav mkdocs-section-index
```

### mkdocs.yml Configuration

```yaml
site_name: Counterfactuals
site_description: A Python library for counterfactual explanations
repo_url: https://github.com/ofurman/counterfactuals
repo_name: ofurman/counterfactuals

theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - navigation.top
    - search.suggest
    - content.code.copy
    - content.tabs.link
  palette:
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [.]
          options:
            docstring_style: google
            show_source: true
            show_root_heading: true
            members_order: source
  - gen-files:
      scripts:
        - docs/gen_ref_pages.py
  - literate-nav:
      nav_file: SUMMARY.md
  - section-index

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.arithmatex:
      generic: true
  - toc:
      permalink: true
  - attr_list
  - md_in_html

extra_javascript:
  - javascripts/mathjax.js
  - https://polyfill.io/v3/polyfill.min.js?features=es6
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js

nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting-started/installation.md
    - Quick Start: getting-started/quickstart.md
    - Core Concepts: getting-started/concepts.md
  - User Guide:
    - Overview: user-guide/index.md
    - Working with Datasets: user-guide/datasets.md
    - Training Models: user-guide/models.md
    - Generating Counterfactuals: user-guide/generating-counterfactuals.md
    - Evaluating Results: user-guide/evaluation.md
    - Running Pipelines: user-guide/pipelines.md
  - Methods:
    - Overview: methods/index.md
    - Local Methods:
      - PPCEF: methods/local/ppcef.md
      - DiCoFlex: methods/local/dicoflex.md
      - DICE: methods/local/dice.md
      - WACH: methods/local/wach.md
      - SACE: methods/local/sace.md
      - CEM: methods/local/cem.md
      - CEGP: methods/local/cegp.md
      - CET: methods/local/cet.md
      - CCHVAE: methods/local/cchvae.md
      - Artelt: methods/local/artelt.md
    - Global Methods:
      - GLOBE-CE: methods/global/globe-ce.md
      - AReS: methods/global/ares.md
    - Group Methods:
      - ReViCE (Group PPCEF): methods/group/revice.md
      - GLANCE: methods/group/glance.md
      - Group GLOBE-CE: methods/group/group-globe-ce.md
  - Datasets:
    - Overview: datasets/index.md
    - Classification Datasets: datasets/classification.md
    - Regression Datasets: datasets/regression.md
    - Custom Datasets: datasets/custom.md
  - Benchmarks:
    - Evaluation Metrics: benchmarks/metrics.md
    - Benchmark Results: benchmarks/results.md
    - Running Benchmarks: benchmarks/running.md
  - API Reference: reference/
  - Contributing: contributing.md
```

---

## 2. Documentation Structure

### Directory Layout

```
docs/
├── index.md                          # Homepage
├── getting-started/
│   ├── installation.md               # Installation guide
│   ├── quickstart.md                 # Quick start tutorial
│   └── concepts.md                   # Core concepts
├── user-guide/
│   ├── index.md                      # User guide overview
│   ├── datasets.md                   # Working with datasets
│   ├── models.md                     # Training discriminative/generative models
│   ├── generating-counterfactuals.md # Main usage guide
│   ├── evaluation.md                 # Evaluating counterfactuals
│   └── pipelines.md                  # Using Hydra pipelines
├── methods/
│   ├── index.md                      # Methods overview & comparison
│   ├── local/                        # Local method docs (10 files)
│   ├── global/                       # Global method docs (2 files)
│   └── group/                        # Group method docs (3 files)
├── datasets/
│   ├── index.md                      # Datasets overview
│   ├── classification.md             # Classification datasets
│   ├── regression.md                 # Regression datasets
│   └── custom.md                     # Adding custom datasets
├── benchmarks/
│   ├── metrics.md                    # Metric definitions
│   ├── results.md                    # Benchmark comparison tables
│   └── running.md                    # How to run benchmarks
├── reference/                        # Auto-generated API docs
│   └── SUMMARY.md                    # Auto-generated nav
├── gen_ref_pages.py                  # Script for API doc generation
└── contributing.md                   # Contributing guidelines
```

---

## 3. Content Plan by Section

### 3.1 Homepage (`index.md`)

- Library overview and purpose
- Key features highlight (17+ methods, 22 datasets, 18+ metrics)
- Visual diagram of library capabilities
- Quick installation snippet
- Links to main sections

### 3.2 Getting Started

#### `installation.md`
- Prerequisites (Python 3.10+)
- Installation via `uv sync` or pip
- Optional dependencies (LiCE, GPU support)
- Verification steps

#### `quickstart.md`
- End-to-end example: load dataset → train model → generate counterfactuals → evaluate
- Code snippets with explanations
- Expected output

#### `concepts.md`
- What are counterfactual explanations?
- Local vs Global vs Group explanations
- Generative models (flows) for plausibility
- Actionability constraints
- Key terminology glossary

### 3.3 User Guide

#### `datasets.md`
- Loading pre-configured datasets via `FileDataset`
- Dataset configuration YAML structure
- Feature types (numerical, categorical)
- Actionability and constraints
- Train/test splitting
- Cross-validation usage

#### `models.md`
- Discriminative models: LogisticRegression, MLP, NODE
- Generative models: KDE, MAF, RealNVP, NICE, CNF
- Training workflow
- Saving/loading models
- Model selection guidance

#### `generating-counterfactuals.md`
- Using `BaseCounterfactualMethod`
- The `explain()` method
- `ExplanationResult` structure
- Common parameters (alpha, beta, epochs)
- Handling multiple counterfactuals (K parameter)

#### `evaluation.md`
- MetricsOrchestrator usage
- Available metrics by category
- Interpreting results
- Custom metrics

#### `pipelines.md`
- Hydra configuration system
- Running pre-built pipelines
- Customizing pipeline configs
- MLflow logging

### 3.4 Methods Documentation

Each method page should include:
- **Overview**: What the method does, key paper reference
- **Algorithm**: Brief explanation with math notation
- **Usage Example**: Complete code snippet
- **Parameters**: Table of all parameters with descriptions
- **Strengths/Limitations**: When to use this method
- **References**: Original paper, related work

#### Priority Order for Method Documentation:
1. **PPCEF** - Flagship method
2. **DICE** - Popular baseline
3. **GLOBE-CE** - Global method
4. **DiCoFlex** - Feature flexibility
5. **ReViCE** - Group method
6. Remaining methods...

### 3.5 Datasets Documentation

#### `classification.md`
Table for each dataset:
| Dataset | Features | Classes | Size | Use Case |
|---------|----------|---------|------|----------|
| adult   | 14       | 2       | 48K  | Income prediction |
| compas  | 12       | 2       | 7K   | Recidivism |
...

#### `regression.md`
Similar table for regression datasets

#### `custom.md`
- YAML configuration template
- Required vs optional fields
- Feature constraints specification
- Integration with pipelines

### 3.6 Benchmarks

#### `metrics.md`
For each metric:
- Definition and formula
- Interpretation (higher/lower is better)
- When to use

Categories:
- Validity metrics
- Proximity metrics
- Sparsity metrics
- Plausibility metrics
- Diversity metrics

#### `results.md`
- Comparison tables from `metrics.md` (existing)
- Visualizations (if applicable)
- Method recommendations by use case

#### `running.md`
- How to reproduce benchmarks
- Adding new methods to benchmarks
- Exporting results

### 3.7 API Reference (Auto-generated)

Script `gen_ref_pages.py`:
```python
"""Generate API reference pages."""
from pathlib import Path
import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
src = Path("counterfactuals")

for path in sorted(src.rglob("*.py")):
    if path.name.startswith("_"):
        continue
    module_path = path.relative_to(src.parent).with_suffix("")
    doc_path = path.relative_to(src.parent).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)
    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        fd.write(f"::: {identifier}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
```

### 3.8 Contributing (`contributing.md`)
- Based on existing `AGENTS.md`
- Code style (Ruff, PEP 8)
- Type hints requirements
- Docstring format (Google style)
- Testing requirements
- PR process

---

## 4. Implementation Phases

### Phase 1: Foundation
- [ ] Set up MkDocs configuration
- [ ] Create directory structure
- [ ] Write homepage
- [ ] Write installation guide
- [ ] Set up API reference auto-generation

### Phase 2: Core Documentation
- [ ] Quick start tutorial
- [ ] Core concepts page
- [ ] User guide sections (5 pages)
- [ ] Datasets overview and reference

### Phase 3: Methods Documentation
- [ ] Methods overview with comparison table
- [ ] PPCEF documentation (flagship)
- [ ] DICE documentation (baseline)
- [ ] GLOBE-CE documentation
- [ ] Remaining local methods (7 pages)
- [ ] Remaining global/group methods (4 pages)

### Phase 4: Benchmarks & Polish
- [ ] Metrics documentation
- [ ] Benchmark results page
- [ ] Running benchmarks guide
- [ ] Contributing guide
- [ ] Review and cross-link all pages

### Phase 5: Enhancements
- [ ] Add diagrams (Mermaid)
- [ ] Add interactive examples (if applicable)
- [ ] Search optimization
- [ ] Versioning setup

---

## 5. Writing Guidelines

### Code Examples
- All code should be runnable
- Include imports
- Use consistent variable names
- Add comments for complex steps

### Admonitions
Use MkDocs admonitions for:
- `!!! note` - Additional information
- `!!! tip` - Best practices
- `!!! warning` - Common pitfalls
- `!!! example` - Code examples

### Cross-References
Link between pages using:
- `[text](../path/to/page.md)` for relative links
- `[text][identifier]` for reference-style links
- Auto-generated API links via mkdocstrings

### Mathematical Notation
Use MathJax for formulas:
```markdown
The loss function is defined as:

$$
\mathcal{L} = \alpha \cdot \text{validity} + \beta \cdot \text{proximity}
$$
```

---

## 6. Estimated Content

| Section | Pages | Priority |
|---------|-------|----------|
| Getting Started | 3 | High |
| User Guide | 6 | High |
| Methods | 16 | High |
| Datasets | 4 | Medium |
| Benchmarks | 3 | Medium |
| API Reference | Auto | High |
| Contributing | 1 | Low |
| **Total** | **33+** | |

---

## 7. Next Steps

1. **Create `mkdocs.yml`** with the configuration above
2. **Set up `docs/` directory structure**
3. **Write `index.md` homepage**
4. **Implement API reference generation**
5. **Start with Getting Started section**
6. **Iterate on methods documentation**

---

## 8. Notes

- Leverage existing docstrings (Google style) for API reference
- Migrate content from existing `metrics.md` and method READMEs
- Use notebooks as source for examples
- Consider adding a "Gallery" section with visualizations from notebooks
