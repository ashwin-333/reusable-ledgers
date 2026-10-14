"""
Anthropic prompt-caching helpers (iteration-03 version).

Background: iteration 02 proved the obvious multipart layout
`[static-prefix | trajectory | static-suffix]` with cache_control on both
outer blocks produces writes-only (cache_creation_input_tokens > 0,
cache_read_input_tokens == 0) because the second cache breakpoint's key =
prefix + trajectory + suffix, which varies every call.

The canonical pattern that actually caches on Haiku 4.5 and Sonnet 4.6:

    system = [{"type":"text",
               "text": <ENTIRE static template, placeholder removed>,
               "cache_control": {"type":"ephemeral"}}]
    messages = [{"role":"user", "content": <ONLY the dynamic text>}]

The entire concatenated static template is ~2.5k tokens for the Universal
Ledger prompt, which crosses the ~2048-token minimum Haiku needs before
`cache_control` is honored. Caller also primes by awaiting the first call
AND sleeping briefly before the parallel wave (see `prime_and_gather`).

We keep `split_cached_template` and the legacy `user_content_sandwich`
helper (only for backwards compatibility); new code should use
`build_cached_system_and_user` / `build_cached_system_only`.
"""


from __future__ import annotations

import asyncio
from typing import Awaitable, Iterable


def split_cached_template(template: str, placeholder: str) -> tuple[str, str]:
    """Return (prefix, suffix) such that prefix + dynamic + suffix == filled template."""
    if placeholder not in template:
        raise ValueError(f"Placeholder {placeholder!r} not found in template")
    parts = template.split(placeholder, 1)
    if len(parts) != 2:
        raise ValueError(f"Placeholder {placeholder!r} must appear exactly once")
    return parts[0], parts[1]


def build_cached_system_and_user(
    template: str,
    placeholder: str,
    dynamic: str,
    *,
    dynamic_header: str = "INPUT",
) -> tuple[list[dict], str]:
    """
    Build (system_blocks, user_text) for Anthropic messages.create().

    system_blocks has ONE text block with cache_control covering the entire
    static template (prefix + suffix concatenated, with the placeholder
    removed). user_text is ONLY the dynamic content, wrapped in a small
    header so the model knows what it is.
    """
    prefix, suffix = split_cached_template(template, placeholder)
    static = (prefix.rstrip() + "\n\n" + suffix.lstrip()).strip()
    system_blocks = [
        {"type": "text", "text": static, "cache_control": {"type": "ephemeral"}}
    ]
    sep = "-" * 50
    user_text = f"### {dynamic_header}\n{sep}\n{dynamic.strip()}\n{sep}"
    return system_blocks, user_text


def build_cached_system_only(
    static_text: str,
    dynamic: str,
    *,
    dynamic_header: str = "INPUT",
) -> tuple[list[dict], str]:
    """Variant when the static text has no placeholder (e.g. monolithic rubric)."""
    system_blocks = [
        {"type": "text", "text": static_text.strip(), "cache_control": {"type": "ephemeral"}}
    ]
    sep = "-" * 50
    user_text = f"### {dynamic_header}\n{sep}\n{dynamic.strip()}\n{sep}"
    return system_blocks, user_text


# Kept for backwards compatibility only; new code should not use this.
def user_content_sandwich(prefix: str, dynamic: str, suffix: str) -> list[dict]:
    blocks: list[dict] = [
        {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": dynamic},
    ]
    if suffix:
        blocks.append({"type": "text", "text": suffix, "cache_control": {"type": "ephemeral"}})
    return blocks


async def prime_and_gather(
    first_coro: Awaitable,
    rest_coros: Iterable[Awaitable],
    *,
    sleep_after_first: float = 2.0,
):
    """
    Run the first coroutine to completion, sleep briefly so the Anthropic
    prompt-cache write has time to propagate, then gather the remaining
    coroutines in parallel. Returns [first_result, *rest_results] in order.
    """
    first = await first_coro
    rest_list = list(rest_coros)
    if not rest_list:
        return [first]
    if sleep_after_first > 0:
        await asyncio.sleep(sleep_after_first)
    rest = await asyncio.gather(*rest_list)
    return [first] + list(rest)


def normalize_length_bucket(label: str) -> str:
    """
    Collapse synonymous directory bucket names so per-bucket tables are comparable.

    Synthetic dirs use `20k-30k`; smoothed dirs use `20-30k` for the same length band.
    Synthetic `50k+` aligns with smoothed `over_50k`.
    """
    if not label:
        return label
    aliases = {
        "20-30k": "20k-30k",
        "30-40k": "30k-40k",
        "40-50k": "40k-50k",
        "over_50k": "50k+",
    }
    return aliases.get(label, label)
