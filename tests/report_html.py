"""Helpers for reading the rendered report through its STABLE test hooks.

docs/interfaces.md §11.11.5 pins the hooks tests may rely on: `main[data-status]`,
`main[data-mode]`, the section ids of §11.4, `[data-check-id]` +
`[data-severity]`, `[data-kpi]`, the figure div ids and `#created-at`.
Everything here goes through those hooks only, never through headings or
copy (wording is not a contract, §11.5.2). Not a test module (no `test_`
prefix), so pytest does not collect it.
"""

from __future__ import annotations

import html as html_lib
import re
from html.parser import HTMLParser
from typing import Any

SECTION_IDS = ("at-a-glance", "results", "segments", "calibration", "gates", "audit")
KPI_NAMES = ("reviewed", "capture", "lift", "residual")

# plotly.js's own banner; `plotly.offline.get_plotlyjs()` contains it exactly
# once, so it identifies the embedded bundle (figure snippets don't have it).
PLOTLY_BUNDLE_MARKER = "plotly.js v"

_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</script>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def strip_scripts(html: str) -> str:
    """Drop every <script>…</script> block (the 4.6 MB bundle + figure
    bootstraps) so structural parsing stays fast and script text can't
    produce false matches."""
    return _SCRIPT_RE.sub("", html)


def text(fragment: str) -> str:
    """Visible text of an HTML fragment: tags removed, entities decoded,
    whitespace collapsed."""
    return " ".join(html_lib.unescape(_TAG_RE.sub(" ", fragment)).split())


def section(html: str, section_id: str) -> str:
    """Raw HTML inside `<section id="...">` (sections are never nested)."""
    opener = f'<section id="{section_id}">'
    assert html.count(opener) == 1, f"expected exactly one {opener!r}"
    return html.split(opener, 1)[1].split("</section>", 1)[0]


def section_order(html: str) -> list[str]:
    """Section ids in document order."""
    found = [
        (html.index(f'<section id="{sid}">'), sid)
        for sid in SECTION_IDS
        if f'<section id="{sid}">' in html
    ]
    return [sid for _, sid in sorted(found)]


class _StartTags(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, Any]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))


def start_tags(html: str) -> list[tuple[str, dict[str, Any]]]:
    parser = _StartTags()
    parser.feed(strip_scripts(html))
    return parser.tags


def main_attrs(html: str) -> dict[str, Any]:
    mains = [attrs for tag, attrs in start_tags(html) if tag == "main"]
    assert len(mains) == 1
    return mains[0]


def rendered_checks(html: str) -> list[dict[str, Any]]:
    """Every rendered check, in document order: id, severity, open flag and
    classes, read from the `<details data-check-id=…>` element itself."""
    out = []
    for tag, attrs in start_tags(html):
        if tag == "details" and "data-check-id" in attrs:
            out.append(
                {
                    "id": attrs["data-check-id"],
                    "severity": attrs["data-severity"],
                    "open": "open" in attrs,
                    "class": attrs.get("class", ""),
                }
            )
    return out


def check_block(html: str, check_id: str) -> str:
    """Raw HTML of one check, from its opening tag to its `</details>`."""
    marker = f'data-check-id="{check_id}"'
    assert html.count(marker) == 1, f"expected exactly one {marker!r}"
    return html.split(marker, 1)[1].split("</details>", 1)[0]


def kpi_order(html: str) -> list[str]:
    return [attrs["data-kpi"] for tag, attrs in start_tags(html) if "data-kpi" in attrs]


def kpi_block(html: str, name: str) -> str:
    """Raw HTML of one KPI tile: from its data-kpi attribute to the next
    tile (or the checks list after the last tile)."""
    marker = f'data-kpi="{name}"'
    assert html.count(marker) == 1, f"expected exactly one {marker!r}"
    rest = html.split(marker, 1)[1]
    ends = [i for i in (rest.find("data-kpi="), rest.find('<div class="checks">')) if i >= 0]
    return rest[: min(ends)] if ends else rest


def first_figure_index(html: str) -> int:
    return html.index('class="plotly-graph-div"')
