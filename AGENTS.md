# AGENTS.md — is-it-true

## Project overview

`is-it-true` is an AI-powered fact-checking Python package. Users call `is_it_true(claim)` and get back a `FactCheckReport` with a verdict, confidence score, chain of evidence, and references. The agent performs multi-round web searches, extracts evidence, resolves contradictions, and synthesizes a final judgment.

Package manager: `uv`. Ruff is available in PATH but not listed as a project dependency.

## Common commands

```bash
uv sync                 # install dependencies
uv run ruff check .     # lint
uv run ruff format .    # format
uv run python -c "from is_it_true import is_it_true; print(is_it_true.__doc__)"  # smoke test
uv run is-it-true "claim to check"        # CLI
uv run is-it-true --format json "claim"   # CLI JSON output
uv run is-it-true --format html "claim"   # CLI HTML output
uv run is-it-true -f pdf -o report.pdf "claim"  # PDF output
```

Tests: none exist yet.

## Architecture

```
src/is_it_true/
├── __init__.py           # is_it_true() — async public API
├── agent.py              # investigate() — async orchestrator loop + _InvestigationState
├── cli.py                # CLI entry point (uv run is-it-true / pip script)
├── config.py             # build_model_config(), resolve_model(), env loading
├── display.py            # console, spin(), color helpers, print_*() summaries
├── logging.py            # OutputMode (CONSOLE/NONE/JSON_LINES), log, event
├── utils.py               # check_finish_reason(), parse_json_response(), domain_credibility_score(), verdict_style()
├── models.py             # Pydantic data models (FactCheckReport, ModelConfigDict, etc.)
├── operations.py         # search_queries_parallel(), enrich_sources(), filter_results()
├── engines/              # Search engine abstraction
│   ├── base.py           # SearchEngine ABC + SearchResult
│   ├── __init__.py       # get_search_engine() auto-select
│   ├── fallback.py       # FallbackEngine (auto-discovers engines in constructor)
│   ├── tavily.py         # Primary (keyless mode, include_raw_content)
│   ├── exa.py            # Secondary (deep-reasoning, highlights)
│   └── duckduckgo.py     # Fallback (trafilatura extraction inline)
├── formatters/           # Report export formats
│   ├── html.py           # Self-contained HTML with inline CSS
│   └── pdf.py            # PDF via ReportLab
├── subagents/            # 7 isolated LLM roles
│   ├── language_detector.py
│   ├── query_planner.py
│   ├── evidence_extractor.py
│   ├── gap_detector.py
│   ├── contradiction_resolver.py
│   ├── source_evaluator.py
│   └── verdict_judge.py
└── multimedia/
    └── image_analyzer.py # Vision LLM image analysis (7th role)
```

**8 LLM roles total**: 7 subagents (6 investigation + `language_detector`) + `image_analyzer`. The image_analyzer lives in `multimedia/` not `subagents/`.

## Key design decisions

### The investigation loop is iterative, not single-pass

Each round: Plan queries → Search → Extract evidence → Evaluate sources → Detect gaps → Resolve contradictions. Termination: no gaps, consistent evidence across rounds (> round 1) with no contradictions, max rounds, or dead-end (no new sources).

### Search engine priority (auto mode)

In auto mode, all available engines are discovered at startup and arranged into a **fallback chain**. If the primary engine fails a search at runtime, the next engine in the chain is tried automatically.

Chain build order:
1. Tavily with `TAVILY_API_KEY` — richest results (advanced depth, raw content)
2. Tavily keyless — free tier (1K credits/month, basic depth)
3. Exa with `EXA_API_KEY` — deep-reasoning context, highlights
4. DuckDuckGo — always available, no credentials needed

The chain composition is logged at startup as `[engine] discovered: ...` and `[engine] fallback chain: ...`. At each call, the engine that succeeds is printed as `[engine] using ...`.

Tavily and Exa return raw content directly; DuckDuckGo needs a separate trafilatura extraction step.

### Model config precedence

`is_it_true(model_config={...})` → `IS_IT_TRUE_<ROLE>_MODEL` env → `IS_IT_TRUE_DEFAULT_MODEL` env.

The config class is `ModelConfigDict` (not `ModelConfig`). Use `build_model_config(model_config)` to resolve, then `resolve_model(role, config)` to get a specific role's model.

Minimal setup: `export IS_IT_TRUE_DEFAULT_MODEL="openai/gpt-4o-mini"`.

The `image_analyzer` role needs a vision-capable model (e.g., `gpt-4o`).

The `language_detector` role defaults to the same model as `IS_IT_TRUE_DEFAULT_MODEL`; a lightweight model like `gpt-4o-mini` is sufficient.

### JSON parsing: json-repair + Pydantic

Sub-agent LLM responses are parsed via `parse_json_response(text, model)` in `utils.py:57`. This:

1. Strips markdown fences.
2. Repairs malformed JSON with `json_repair.repair_json()` — fixes unescaped quotes, trailing commas, missing brackets, single-quoted strings, etc.
3. Validates and coerces with a Pydantic model or `TypeAdapter`.

Response models live in `models.py` (e.g. `VerdictJudgeResponse`, `GapDetectorResponse`, `QueryPlannerResponse`). Each model defines field types and defaults so partial or slightly malformed LLM output is gracefully coerced.

When adding a new sub-agent:
- Define its response model in `models.py`.
- Use `parse_json_response(content, YourResponse)` instead of `try_parse_json`.
- `BaseModel` → returns `dict` via `.model_dump()`. `TypeAdapter` (for bare lists) → returns the validated value directly.

### Reasoning effort

Per-role reasoning/thinking effort can be set via environment variables, resolved by `resolve_reasoning(role)` in `config.py:73`:

- `IS_IT_TRUE_<ROLE>_REASONING_EFFORT` — per-role (e.g. `IS_IT_TRUE_VERDICT_JUDGE_REASONING_EFFORT=high`)
- `IS_IT_TRUE_REASONING_EFFORT` — global fallback

Valid values: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `default`. If unset, the model's default behavior is used.

### Trafilatura limitation

trafilatura works on server-rendered HTML only. For CSR/SPA sites, extraction returns empty. The DuckDuckGo engine falls back to the search snippet when trafilatura produces <200 chars.

### Image analysis is opt-in

Only triggered when `multimedia=True` AND the claim contains visual keywords. 16 keywords at `src/is_it_true/multimedia/image_analyzer.py:29-46`: photo, image, picture, video, chart, graph, screenshot, footage, visual, diagram, infographic, map, satellite, photograph, clip, recording. Checked by `claim_warrants_visual_analysis()`.

### Sync/async split

The public API is async. The CLI uses ``asyncio.run()`` to bridge to the sync entrypoint. When calling internal functions directly (e.g., ``plan_queries``, ``extract_evidence``), you must ``await`` them.

### FactCheckReport fields

The actual fields (src/is_it_true/models.py:66-74): `claim`, `language`, `verdict`, `confidence`, `summary`, `investigation_rounds` (list of `InvestigationRound`), `references` (list[str]), `contradictions_resolved`, `model_config_used`.

There is no top-level `evidence_chain` or `search_engine_used` field — evidence lives inside `investigation_rounds[].evidence`, and the engine is recorded per-round inside each `InvestigationRound`.

### CLI

The package registers a `is-it-true` console script (`pyproject.toml` line 22-23 → `src/is_it_true/cli.py:main()`). Args include `--engine`, `--rounds` (1-5), `--depth`, `--multimedia`, `--format` (console|json|html|pdf), `--log` (console|json|none), `--output`, and `--version`. Accepts claim as positional arg or from stdin.

Output modes:
- Default: styled rich console output with spinners and colors
- `--format json`: final report as JSON
- `--format html`: self-contained HTML document (defaults to `report.html`)
- `--format pdf`: PDF via ReportLab (defaults to `report.pdf`)
- `--log json`: outputs progress events as JSON lines (`{"event": "search_result", ...}`) for RPC/external consumption
- `--log none`: suppresses all progress messages and spinners
- `--output / -o`: write report to file instead of stdout (defaults to `report.<fmt>` for HTML/PDF)

### Caching and deduplication

`_InvestigationState` carries several caches across rounds:

- **`search_queries_cache`** (`set[str]`) — query strings already searched. Checked AFTER the query-planner LLM call (post-hoc filter, saves search-engine calls not LLM calls).
- **`seen_urls`** (`set[str]`) — URLs returned by any search. Mutated in-place by `search_queries_parallel()`. Deduplication is atomic within the asyncio event loop.
- **`resolved_contradiction_pairs`** (`set[frozenset[int]]`) — evidence-index pairs already resolved, preventing duplicate LLM calls when the gap detector re-flags the same pair in later rounds.
- **`filter_results()`** — always deduplicates URLs before scoring/capping.

Not cached (by design): the query-planner LLM response (cannot predict output without calling), source evaluator results per URL (URLs are already deduplicated), language detection result (one-shot).

### Language detection

Source language is auto-detected at the start of each investigation by the `language_detector` subagent (`src/is_it_true/subagents/language_detector.py`). Configurable via `IS_IT_TRUE_LANGUAGE_DETECTOR_MODEL` env var or `model_config={"language_detector": "..."}`.
