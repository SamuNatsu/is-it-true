"""Vision LLM image analysis — opt-in multimedia sub-agent.

Triggered only when ``multimedia=True`` AND the claim contains visual
keywords (e.g. "photo", "chart", "screenshot"). Downloads image URLs,
base64-encodes them, and sends them to a vision-capable LLM for analysis.

Each analysed image produces a verdict of supports/contradicts/neutral
plus a literal description and assessment. Results are written into
``evidence[].visual_findings``.
"""

from __future__ import annotations

import base64

import litellm

from .. import logging as log
from ..config import resolve_model, resolve_reasoning
from ..models import ImageAnalyzerResponse, ModelConfigDict, record_token_usage
from ..utils import check_finish_reason, parse_json_response

IMAGE_ANALYZER_SYSTEM = """You are an image analyzer for fact-checking. You will be shown an image and a claim.
Determine whether the image supports, contradicts, or is unrelated to the claim.

Consider:
- What is actually visible in the image (describe literally, don't infer beyond what you see)
- Whether the image appears to be authentic or possibly manipulated/synthetic
- Whether the image could be from a different context than what the claim suggests
- Text overlays, watermarks, or logos that indicate provenance

Return a JSON object:
{
  "supports_claim": true/false/null,
  "description": "Literal description of what is visible in the image",
  "assessment": "Analysis of whether this image supports or contradicts the claim"
}

supports_claim: true = image supports the claim, false = image contradicts, null = image is unrelated

Return ONLY the JSON object — no markdown fences, no other text."""

# Keywords that trigger visual analysis when present in the claim
VISUAL_CLAIM_KEYWORDS = [
    "photo",
    "image",
    "picture",
    "video",
    "chart",
    "graph",
    "screenshot",
    "footage",
    "visual",
    "diagram",
    "infographic",
    "map",
    "satellite",
    "photograph",
    "clip",
    "recording",
]


def claim_warrants_visual_analysis(claim: str) -> bool:
    """Check whether the claim contains keywords suggesting visual content."""
    claim_lower = claim.lower()
    return any(kw in claim_lower for kw in VISUAL_CLAIM_KEYWORDS)


def _build_user_prompt(claim: str, image_description: str = "") -> str:
    """Build the user prompt for image analysis, with optional metadata."""
    desc = f"\nImage metadata/description: {image_description}" if image_description else ""
    return f"Claim: {claim}{desc}\n\nAnalyze the attached image against this claim."


async def analyze_image(
    image_url: str,
    claim: str,
    description: str = "",
    model_config: ModelConfigDict | None = None,
) -> dict[str, bool | str | None] | None:
    """Analyse a single image against the claim.

    Returns a dict with ``supports_claim``, ``description``, ``assessment``
    keys, or ``None`` if the image couldn't be fetched or analysed.

    Requires a vision-capable model (e.g. gpt-4o).
    """
    config = model_config or ModelConfigDict()
    model = resolve_model("image_analyzer", config)
    reasoning_effort = resolve_reasoning("image_analyzer")

    # Download and base64-encode the image
    try:
        image_data, mime_type = await _fetch_image_base64(image_url)
    except Exception:
        return None

    messages = [
        {"role": "system", "content": IMAGE_ANALYZER_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _build_user_prompt(claim, description)},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                },
            ],
        },
    ]

    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=2000,
            temperature=0.0,
            reasoning_effort=reasoning_effort,
        )
        record_token_usage(response.usage)
        content = response.choices[0].message.content
        check_finish_reason(response.choices[0].finish_reason, "image analyzer")
        if not content:
            return None
        return _parse_response(content)
    except Exception as e:
        log.print(f"  image analyzer failed: {e}")
        return None


async def _fetch_image_base64(url: str) -> tuple[str, str]:
    """Download an image and return its base64-encoded string and MIME type."""
    import httpx

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        mime = resp.headers.get("content-type", "image/jpeg")
        return base64.b64encode(resp.content).decode("utf-8"), mime


def _parse_response(content: str) -> dict[str, bool | str | None]:
    """Parse image analysis JSON (validated via json-repair + Pydantic).

    Falls back to raw text description on failure.
    """
    data = parse_json_response(content, ImageAnalyzerResponse)
    if isinstance(data, dict):
        return data
    return {
        "supports_claim": None,
        "description": content[:500],
        "assessment": content[:500],
    }
