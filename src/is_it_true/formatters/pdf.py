"""PDF report formatter — native PDF generation via ReportLab.

Produces a professional multi-section report with proper pagination,
consistent typography, and controlled page breaks.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from io import BytesIO

# Register Unicode-capable fonts that ship with ReportLab
import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..models import FactCheckReport
from ..utils import evidence_label

# Primary font: Vera (bundled with ReportLab, covers Latin + Cyrillic).
# Extended: package-embedded NotoSansCJK for East-Asian text.

_assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
_fonts_dir = os.path.join(os.path.dirname(reportlab.__file__), "fonts")

pdfmetrics.registerFont(TTFont("Vera", os.path.join(_fonts_dir, "Vera.ttf")))
pdfmetrics.registerFont(TTFont("Vera-Bold", os.path.join(_fonts_dir, "VeraBd.ttf")))
pdfmetrics.registerFont(TTFont("Vera-Italic", os.path.join(_fonts_dir, "VeraIt.ttf")))
pdfmetrics.registerFont(TTFont("Vera-BoldItalic", os.path.join(_fonts_dir, "VeraBI.ttf")))
pdfmetrics.registerFontFamily(
    "Vera",
    normal="Vera",
    bold="Vera-Bold",
    italic="Vera-Italic",
    boldItalic="Vera-BoldItalic",
)

_cjk_path = os.path.join(_assets_dir, "NotoSansSC.ttf")
_cjk_font: str | None = None
if os.path.exists(_cjk_path):
    try:
        pdfmetrics.registerFont(TTFont("NotoSansCJK", _cjk_path))
        _cjk_font = "NotoSansCJK"
    except Exception:
        pass

_BODY_FONT = _cjk_font or "Vera"
_BOLD_FONT = _cjk_font or "Vera-Bold"
_MONO_FONT = _cjk_font or "Vera"

_PAGE_W, _PAGE_H = A4
_MARGIN = 25 * mm

_VERDICT_COLORS: dict[str, colors.Color] = {
    "true": colors.HexColor("#28a745"),
    "mostly_true": colors.HexColor("#28a745"),
    "false": colors.HexColor("#dc3545"),
    "mostly_false": colors.HexColor("#dc3545"),
    "misleading": colors.HexColor("#856404"),
    "unverified": colors.HexColor("#856404"),
}

_EV_BG: dict[str, colors.Color] = {
    "SUPPORTS": colors.HexColor("#d4edda"),
    "CONTRADICTS": colors.HexColor("#f8d7da"),
    "NEUTRAL": colors.HexColor("#fff3cd"),
}


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontSize=20, spaceAfter=8 * mm, fontName=_BODY_FONT
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontSize=11,
            textColor=colors.HexColor("#555555"),
            fontName=_BODY_FONT,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontSize=14,
            spaceBefore=10 * mm,
            spaceAfter=4 * mm,
            fontName=_BODY_FONT,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontSize=12,
            spaceBefore=6 * mm,
            spaceAfter=2 * mm,
            fontName=_BODY_FONT,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontSize=9.5,
            leading=14,
            spaceAfter=3 * mm,
            fontName=_BODY_FONT,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["Normal"],
            fontSize=7.5,
            textColor=colors.HexColor("#999999"),
            fontName=_BODY_FONT,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Normal"],
            fontSize=8,
            fontName=_MONO_FONT,
            leading=10,
            leftIndent=6 * mm,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=base["Normal"],
            fontSize=8,
            leading=10,
            wordWrap="CJK",
            fontName=_BODY_FONT,
        ),
    }


def render_pdf(report: FactCheckReport) -> bytes:
    """Generate PDF bytes from a FactCheckReport."""
    buf = BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN,
        title=f"Fact Check: {report.claim[:80]}",
    )

    s = _build_styles()
    story: list = []

    _title_page(story, report, s)
    story.append(PageBreak())
    _summary_section(story, report, s)
    _rounds_section(story, report, s)

    if report.contradictions_resolved:
        story.append(PageBreak())
        story.append(Paragraph("Resolved Contradictions", s["h1"]))
        for c in report.contradictions_resolved:
            story.append(KeepTogether(_contradiction_block(c, s)))

    if report.references:
        story.append(PageBreak())
        story.append(Paragraph("References", s["h1"]))
        for i, ref in enumerate(report.references, 1):
            story.append(
                Paragraph(
                    f'[{i}] <a href="{_escape_html(ref)}" color="blue">{_escape_html(ref)}</a>',
                    s["code"],
                )
            )

    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------


def _title_page(story: list, report: FactCheckReport, s: dict) -> None:
    story.append(Spacer(1, 20 * mm))
    story.append(Paragraph("Fact-Check Report", s["title"]))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(report.claim, s["subtitle"]))
    story.append(Spacer(1, 10 * mm))

    verdict_text = report.verdict.replace("_", " ").title()
    color = _VERDICT_COLORS.get(report.verdict, colors.black)
    story.append(Paragraph(f"<b>Verdict:</b> <font color='{color}'>{verdict_text}</font>", s["h2"]))
    story.append(Paragraph(f"<b>Confidence:</b> {report.confidence:.0%}", s["h2"]))

    total_rounds = len(report.investigation_rounds)
    total_sources = sum(len(r.sources_found) for r in report.investigation_rounds)
    total_evidence = sum(len(r.evidence) for r in report.investigation_rounds)
    story.append(
        Paragraph(
            f"{total_rounds} round(s) &middot; {total_sources} source(s) "
            f"&middot; {total_evidence} evidence item(s)",
            s["small"],
        )
    )

    if report.total_token_usage:
        t = report.total_token_usage
        total = t.input_tokens + t.output_tokens
        parts = [
            f"Input: {t.input_tokens:,}",
            f"Output: {t.output_tokens:,}",
            f"Total: {total:,}",
        ]
        if t.cache_read_tokens:
            parts.append(f"Cache read: {t.cache_read_tokens:,}")
        story.append(Paragraph(" &middot; ".join(parts), s["small"]))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(f"Generated: {now}", s["small"]))


def _summary_section(story: list, report: FactCheckReport, s: dict) -> None:
    story.append(Paragraph("Summary", s["h1"]))
    story.append(Paragraph(report.summary, s["body"]))


def _rounds_section(story: list, report: FactCheckReport, s: dict) -> None:
    for rnd in report.investigation_rounds:
        story.append(PageBreak())
        story.append(KeepTogether(_round_header(rnd, s)))
        story.append(Spacer(1, 3 * mm))

        if rnd.search_queries:
            story.append(Paragraph("<b>Queries</b>", s["body"]))
            for q in rnd.search_queries:
                story.append(
                    Paragraph(
                        f"&bull; <font size='8' color='#1565c0'>{_escape_html(q)}</font>",
                        s["small"],
                    )
                )
            story.append(Spacer(1, 2 * mm))

        if rnd.evidence:
            story.append(Paragraph("<b>Evidence</b>", s["body"]))
            story.append(KeepTogether(_evidence_table(rnd, s)))

        if rnd.gaps_identified:
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph("<b>Gaps identified</b>", s["body"]))
            for g in rnd.gaps_identified:
                story.append(
                    Paragraph(
                        f"&bull; {_escape_html(g.question)} — <i>{_escape_html(g.reason)}</i>",
                        s["small"],
                    )
                )


def _round_header(rnd, s: dict) -> list:
    total_tok = (
        rnd.token_usage.input_tokens + rnd.token_usage.output_tokens if rnd.token_usage else 0
    )
    tok_str = f" &middot; {total_tok:,} tokens" if total_tok else ""
    return [
        Paragraph(
            f"<b>Round {rnd.round_number}</b> — "
            f"{rnd.search_engine_used} &middot; "
            f"{len(rnd.sources_found)} sources &middot; "
            f"{len(rnd.evidence)} evidence{tok_str}",
            s["h2"],
        )
    ]


def _evidence_table(rnd, s: dict) -> Table:
    header = ["Dir", "Source / Title", "Key Passages"]
    data: list = [header]
    for ev in rnd.evidence:
        label = evidence_label(ev.supports_claim)
        icon = {"SUPPORTS": "+", "CONTRADICTS": "\u2212", "NEUTRAL": "~"}.get(label, "?")
        title = _escape_html(ev.source.title or ev.source.url)
        passages = "\n".join(ev.key_passages[:2]) if ev.key_passages else "\u2014"
        data.append(
            [
                Paragraph(f"<b>{icon}</b>", s["cell"]),
                Paragraph(f"<b>{title}</b>", s["cell"]),
                Paragraph(_escape_html(passages), s["cell"]),
            ]
        )

    col_w = [_PAGE_W - 2 * _MARGIN]
    tbl = Table(
        data,
        colWidths=[32, (col_w[0] - 32) * 0.35, (col_w[0] - 32) * 0.65],
    )
    tbl.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), _BODY_FONT),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("FONTSIZE", (0, 0), (-1, 0), 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f2f5")),
                ("FONTNAME", (0, 0), (-1, 0), _BOLD_FONT),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#fafafa")],
                ),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    for i, ev in enumerate(rnd.evidence):
        label = evidence_label(ev.supports_claim)
        bg = _EV_BG.get(label, colors.white)
        tbl.setStyle(TableStyle([("BACKGROUND", (0, i + 1), (0, i + 1), bg)]))
    return tbl


def _contradiction_block(c, s: dict) -> list:
    return [
        Paragraph(
            f"<b>Contradiction:</b> {_escape_html(c.evidence_a.source.title)} vs "
            f"{_escape_html(c.evidence_b.source.title)}",
            s["body"],
        ),
        Paragraph(
            f"<b>Resolution:</b> {_escape_html(c.resolution)}<br/><b>Trusted:</b> {_escape_html(c.trusted_source)}",
            s["small"],
        ),
        Spacer(1, 2 * mm),
    ]


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )
