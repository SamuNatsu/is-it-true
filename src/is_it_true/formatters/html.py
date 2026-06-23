"""Self-contained HTML report formatter.

Produces a single HTML document with inline CSS — no external resources
required. Suitable for direct display in a browser.
"""

from __future__ import annotations

import html as _html
from datetime import datetime, timezone

from ..models import FactCheckReport

_CSS = """\
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.6;color:#1a1a2e;background:#f8f9fa;padding:40px 20px}
.container{max-width:900px;margin:0 auto;background:#fff;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,.08);overflow:hidden}
.header{background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff;padding:32px 40px}
.header h1{font-size:1.6rem;font-weight:600;margin-bottom:8px}
.header .claim{font-size:1.1rem;opacity:.85;font-style:italic}
.meta{padding:20px 40px;background:#f0f2f5;display:flex;gap:32px;flex-wrap:wrap;font-size:.9rem;border-bottom:1px solid #e0e0e0}
.meta-item{display:flex;align-items:center;gap:6px}
.meta-label{color:#666;font-weight:500}
.meta-value{font-weight:600}
.verdict{display:inline-block;padding:4px 14px;border-radius:20px;font-weight:700;font-size:.9rem;text-transform:uppercase;letter-spacing:.5px}
.verdict-true,.verdict-mostly_true{background:#d4edda;color:#155724}
.verdict-false,.verdict-mostly_false{background:#f8d7da;color:#721c24}
.verdict-misleading,.verdict-unverified{background:#fff3cd;color:#856404}
.confidence-bar{display:inline-flex;align-items:center;gap:8px}
.confidence-value{font-size:1.3rem;font-weight:700}
.bar-track{width:120px;height:8px;background:#e0e0e0;border-radius:4px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px;transition:width .3s}
.bar-high{background:#28a745}.bar-mid{background:#ffc107}.bar-low{background:#dc3545}
.body{padding:32px 40px}
.summary{font-size:1.05rem;line-height:1.8;margin-bottom:28px;padding:20px;background:#f0f7ff;border-left:4px solid #2196f3;border-radius:4px}
.section-title{font-size:1.1rem;font-weight:600;margin:28px 0 12px;padding-bottom:6px;border-bottom:2px solid #e0e0e0;color:#333}
.token-summary{font-size:.85rem;color:#666;padding:8px 14px;background:#f8f9fa;border-radius:6px;margin-bottom:20px}
.round{border:1px solid #e8e8e8;border-radius:8px;margin-bottom:16px;overflow:hidden}
.round-header{background:#f8f9fa;padding:10px 16px;font-weight:600;font-size:.95rem;border-bottom:1px solid #e8e8e8;display:flex;justify-content:space-between}
.round-queries{padding:8px 16px}
.query-tag{display:inline-block;background:#e3f2fd;color:#1565c0;font-size:.8rem;padding:3px 10px;margin:3px 6px 3px 0;border-radius:12px}
.evidence-item{padding:12px 16px;border-bottom:1px solid #f0f0f0}
.evidence-item:last-child{border-bottom:none}
.evidence-badge{display:inline-block;font-size:.75rem;font-weight:600;padding:2px 8px;border-radius:10px;margin-right:8px}
.evidence-badge.supports{background:#d4edda;color:#155724}
.evidence-badge.contradicts{background:#f8d7da;color:#721c24}
.evidence-badge.neutral{background:#fff3cd;color:#856404}
.evidence-source{font-weight:600;font-size:.9rem}
.evidence-url{font-size:.75rem;color:#999;word-break:break-all}
.evidence-passage{font-size:.85rem;color:#444;margin:6px 0 0 0;padding:6px 10px;background:#fafafa;border-left:3px solid #ddd;border-radius:3px}
.evidence-visual{font-size:.8rem;color:#666;margin-top:4px;font-style:italic}
.gap-item{font-size:.85rem;padding:6px 12px;margin:12px 16px;background:#fff8e1;border-left:3px solid #ffc107;border-radius:3px}
.gap-question{font-weight:500}
.gap-reason{color:#888;font-size:.8rem}
.contradiction-item{font-size:.85rem;padding:8px 12px;margin:4px 0;background:#fce4ec;border-left:3px solid #e91e63;border-radius:3px}
.references{padding:0 16px 16px}
.ref-link{display:block;font-size:.8rem;color:#1565c0;padding:3px 0;word-break:break-all;text-decoration:none}
.ref-link:hover{text-decoration:underline}
.footer{padding:16px 40px;font-size:.75rem;color:#999;text-align:center;border-top:1px solid #e0e0e0}
@media print{body{background:#fff;padding:0}.container{box-shadow:none;border-radius:0}}
"""


def render_html(report: FactCheckReport) -> str:
    """Generate a self-contained HTML report from a FactCheckReport."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    claim_esc = _html.escape(report.claim)
    summary_esc = _html.escape(report.summary)
    lang = report.language

    verdict = report.verdict
    verdict_text = verdict.replace("_", " ").title()
    conf = report.confidence
    conf_pct = f"{conf:.0%}"
    bar_class = "bar-high" if conf >= 0.7 else "bar-mid" if conf >= 0.4 else "bar-low"

    return f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fact Check: {_html.escape(report.claim[:80])}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>Fact-Check Report</h1>
<div class="claim">{claim_esc}</div>
</div>
<div class="meta">
<div class="meta-item"><span class="meta-label">Verdict:</span>
<span class="verdict verdict-{verdict}">{verdict_text}</span></div>
<div class="meta-item"><span class="meta-label">Confidence:</span>
<span class="confidence-bar">
<span class="confidence-value">{conf_pct}</span>
<div class="bar-track"><div class="bar-fill {bar_class}" style="width:{int(conf * 100)}%"></div></div>
</span></div>
<div class="meta-item"><span class="meta-label">Language:</span><span class="meta-value">{lang}</span></div>
<div class="meta-item"><span class="meta-label">Rounds:</span><span class="meta-value">{len(report.investigation_rounds)}</span></div>
<div class="meta-item"><span class="meta-label">Sources:</span><span class="meta-value">{sum(len(r.sources_found) for r in report.investigation_rounds)}</span></div>
<div class="meta-item"><span class="meta-label">Evidence:</span><span class="meta-value">{sum(len(r.evidence) for r in report.investigation_rounds)}</span></div>
</div>
<div class="body">
<div class="summary">{summary_esc}</div>
{_render_token_summary(report)}
{_render_rounds(report)}
{_render_references(report)}
</div>
<div class="footer">Generated by is-it-true &middot; {now}</div>
</div>
</body>
</html>"""


def _render_token_summary(report: FactCheckReport) -> str:
    if not report.total_token_usage:
        return ""
    t = report.total_token_usage
    parts = [f"{t.input_tokens:,} input", f"{t.output_tokens:,} output"]
    if t.cache_read_tokens:
        parts.append(f"{t.cache_read_tokens:,} cache read")
    if t.cache_creation_tokens:
        parts.append(f"{t.cache_creation_tokens:,} cache write")
    total = t.input_tokens + t.output_tokens
    if total:
        parts.append(f"{total:,} total")
    return f'<div class="token-summary">Tokens: {" &middot; ".join(parts)}</div>'


def _render_rounds(report: FactCheckReport) -> str:
    if not report.investigation_rounds:
        return ""
    parts = ['<div class="section-title">Investigation Rounds</div>']
    for r in report.investigation_rounds:
        parts.append('<div class="round">')
        parts.append(
            f'<div class="round-header"><span>Round {r.round_number}</span>'
            f'<span style="font-weight:400;color:#888">{r.search_engine_used} &middot; '
            f"{len(r.search_queries)} queries &middot; {len(r.sources_found)} sources &middot; "
            f"{len(r.evidence)} evidence</span></div>"
        )
        if r.search_queries:
            parts.append('<div class="round-queries">')
            for q in r.search_queries:
                parts.append(f'<span class="query-tag">{_html.escape(q)}</span>')
            parts.append("</div>")
        for ev in r.evidence:
            parts.append(_render_evidence(ev))
        if r.gaps_identified:
            for g in r.gaps_identified:
                parts.append(
                    '<div class="gap-item">'
                    f'<span class="gap-question">{_html.escape(g.question)}</span>'
                    f'<br><span class="gap-reason">{_html.escape(g.reason)}</span>'
                    "</div>"
                )
        if r.token_usage:
            t = r.token_usage
            parts.append(
                f'<div class="token-summary" style="margin:8px 16px">'
                f"Round tokens: {t.input_tokens + t.output_tokens:,} total"
                f"</div>"
            )
        parts.append("</div>")
    return "\n".join(parts)


def _render_evidence(ev) -> str:
    from ..utils import evidence_label

    label = evidence_label(ev.supports_claim).lower()
    badge_class = f"evidence-badge {label}"
    parts = [
        '<div class="evidence-item">',
        f'<span class="{badge_class}">{label.upper()}</span>',
        f'<span class="evidence-source">{_html.escape(ev.source.title or "Untitled")}</span>',
        f'<div class="evidence-url">{_html.escape(ev.source.url)}</div>',
    ]
    for p in ev.key_passages:
        parts.append(f'<div class="evidence-passage">{_html.escape(p)}</div>')
    if ev.visual_findings:
        parts.append(f'<div class="evidence-visual">{_html.escape(ev.visual_findings)}</div>')
    parts.append("</div>")
    return "\n".join(parts)


def _render_references(report: FactCheckReport) -> str:
    if not report.references:
        return ""
    parts = ['<div class="section-title">References</div>', '<div class="references">']
    for i, ref in enumerate(report.references, 1):
        parts.append(
            f'<a class="ref-link" href="{_html.escape(ref)}" target="_blank">'
            f"[{i}] {_html.escape(ref)}</a>"
        )
    parts.append("</div>")
    return "\n".join(parts)
