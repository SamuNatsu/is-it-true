# is-it-true

> **Warning**: This project is primarily generated through vibe coding with human guidance. It has not been thoroughly tested in production environments. Use at your own risk, and expect edge cases, bugs, and incomplete error handling.

AI-powered fact-checking agent with multi-round web investigation and full chain of evidence.

Takes any claim and returns a detailed report — verdict, confidence score, evidence tracing every source, and a summary of the reasoning.

## Requirements

- Python 3.10+
- A [litellm](https://github.com/BerriAI/litellm)-compatible model provider (OpenAI, Anthropic, Azure, Ollama, etc.)

## Installation

```bash
pip install is-it-true
# or
uv add is-it-true
```

Set your model provider:

```bash
export IS_IT_TRUE_DEFAULT_MODEL="openai/gpt-4o-mini"
```

## Quick start

```python
from is_it_true import is_it_true

report = await is_it_true("The Eiffel Tower grows 15 cm in summer due to thermal expansion")

print(report.verdict)      # e.g. "true"
print(report.confidence)   # 0.92
print(report.summary)      # narrative explanation
```

```bash
is-it-true "The Eiffel Tower grows in summer"
is-it-true --format html "some claim"        # writes report.html
is-it-true --format pdf "some claim"         # writes report.pdf
```

## How it works

The agent runs an **iterative investigation loop** over multiple rounds, each with dedicated LLM roles:

1. **Plan queries** — generate targeted English search queries from the claim
2. **Search** — query web sources through a fallback chain of search engines
3. **Filter & enrich** — deduplicate results, fetch full content, evaluate source credibility
4. **Extract evidence** — identify supporting, contradicting, and neutral facts per source
5. **Detect gaps** — find unanswered questions and flag contradictory evidence
6. **Resolve contradictions** — reconcile conflicting sources with reasoning
7. **Deliver verdict** — synthesise all evidence into a final judgment with confidence score

The loop terminates when gaps are closed, evidence is consistent, max rounds (default 3) are exhausted, or no new sources are found.

## API

```python
from is_it_true import is_it_true

report = await is_it_true(
    claim: str,
    *,
    search_engine: str = "auto",           # "auto", "tavily", "exa", "duckduckgo"
    max_rounds: int = 3,                   # 1–5
    depth: str = "thorough",               # "fast" or "thorough"
    multimedia: bool = False,              # enable image analysis
    multimedia_types: list[str] | None = None,
    model_config: ModelConfigDict | dict | None = None,
    log_mode: str = "console",             # "console", "json", or "none"
) -> FactCheckReport
```

### Return fields

| Field | Type | Description |
|---|---|---|
| `claim` | `str` | The original claim |
| `language` | `str` | Detected ISO 639-1 code (e.g. `"en"`) |
| `verdict` | `str` | `"true"`, `"mostly_true"`, `"mostly_false"`, `"false"`, `"misleading"`, `"unverified"` |
| `confidence` | `float` | 0.0–1.0 |
| `summary` | `str` | Narrative explanation of the verdict |
| `investigation_rounds` | `list[InvestigationRound]` | Round-by-round record with queries, evidence, and gaps |
| `references` | `list[str]` | Deduplicated source URLs |
| `contradictions_resolved` | `list[ContradictionResolution]` | Resolved contradictions with reasoning |
| `model_config_used` | `ModelConfigDict` | The effective model configuration |
| `total_token_usage` | `TokenUsage` | Aggregated input/output/cache token counts |

### Examples

```python
# Basic
report = await is_it_true("The Great Barrier Reef is visible from space")
print(f"{report.verdict} ({report.confidence:.0%}): {report.summary}")

# With model overrides
report = await is_it_true(
    "Python was named after Monty Python, not the snake",
    model_config={
        "verdict_judge": "openai/gpt-4o",
        "default": "openai/gpt-4o-mini",
    },
)

# With image analysis
report = await is_it_true("This viral image shows a real event", multimedia=True)

# With JSON lines progress
report = await is_it_true("some claim", log_mode="json")
```

## Configuration

Model selection resolves per role in this order (highest wins):

1. `model_config={"query_planner": "openai/gpt-4o"}` argument
2. `IS_IT_TRUE_<ROLE>_MODEL` environment variable
3. `IS_IT_TRUE_DEFAULT_MODEL` environment variable

### Per-role model config

```python
from is_it_true.models import ModelConfigDict

report = await is_it_true(
    "Claim to check...",
    model_config=ModelConfigDict(
        default="openai/gpt-4o-mini",
        verdict_judge="openai/gpt-4o",
        image_analyzer="openai/gpt-4o",  # must support vision
    ),
)
```

### Environment variables

| Variable | Purpose |
|---|---|
| `IS_IT_TRUE_DEFAULT_MODEL` | Fallback model for all roles |
| `IS_IT_TRUE_<ROLE>_MODEL` | Per-role model override (e.g. `_VERDICT_JUDGE_MODEL`) |
| `IS_IT_TRUE_<ROLE>_REASONING_EFFORT` | Per-role thinking effort (`none` to `xhigh`) |
| `IS_IT_TRUE_REASONING_EFFORT` | Global reasoning effort fallback |
| `TAVILY_API_KEY` | Tavily search API key |
| `EXA_API_KEY` | Exa search API key |

Available roles: `QUERY_PLANNER`, `EVIDENCE_EXTRACTOR`, `GAP_DETECTOR`, `CONTRADICTION_RESOLVER`, `SOURCE_EVALUATOR`, `VERDICT_JUDGE`, `IMAGE_ANALYZER`, `LANGUAGE_DETECTOR`.

## CLI

```bash
is-it-true "claim"                                # rich console output
is-it-true -f json "claim"                        # JSON to stdout
is-it-true -f html -o report.html "claim"         # HTML to file
is-it-true -f pdf "claim"                         # PDF (writes report.pdf)
is-it-true --log json -f json "claim"             # JSON lines progress + JSON report
is-it-true --log none -f json "claim"             # silent progress + JSON report
is-it-true -e duckduckgo -r 2 -d fast "claim"     # fast investigation
```

| Flag | Default | Description |
|---|---|---|
| `--engine` / `-e` | `auto` | `auto`, `tavily`, `exa`, or `duckduckgo` |
| `--rounds` / `-r` | 3 | Max rounds 1–5 |
| `--depth` / `-d` | `thorough` | `fast` or `thorough` |
| `--multimedia` | off | Enable image analysis |
| `--format` / `-f` | `console` | `console`, `json`, `html`, or `pdf` |
| `--log` | `console` | `console`, `json` (JSON lines), or `none` |
| `--output` / `-o` | stdout* | Write report to file (*`report.html`/`report.pdf` by default for HTML/PDF) |

## Search engines

In `auto` mode (default), all available engines form a **fallback chain**. If the primary engine fails, the next is tried automatically.

| Priority | Engine | Requires | Notes |
|---|---|---|---|
| 1 | Tavily | `TAVILY_API_KEY` | Advanced depth, raw content, images |
| 2 | Tavily keyless | — | Free tier (1,000 credits/month), basic depth |
| 3 | Exa | `EXA_API_KEY` | Deep-reasoning context with highlights |
| 4 | DuckDuckGo | — | Always available; trafilatura content extraction |

DuckDuckGo uses [trafilatura](https://trafilatura.readthedocs.io/) for content extraction. SPA/client-rendered pages may return empty results (minimum 200 characters required to keep extracted content).

## License

MIT
