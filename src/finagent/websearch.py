from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

WEB_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[WebResult] = []
        self._href: str | None = None
        self._title: list[str] = []
        self._snippet: list[str] = []
        self._mode: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        css_class = values.get("class", "") or ""
        if tag == "a" and "result__a" in css_class:
            self._href = values.get("href")
            self._title = []
            self._snippet = []
            self._mode = "title"
        elif "result__snippet" in css_class and self._href:
            self._mode = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._mode == "title" and self._href:
            self._mode = None
        if tag == "div" and self._mode == "snippet" and self._href:
            title = " ".join(self._title).strip()
            snippet = " ".join(self._snippet).strip()
            target_url = _result_url(self._href)
            if title and target_url:
                self.results.append(WebResult(title, target_url, snippet))
            self._href = None
            self._mode = None

    def handle_data(self, data: str) -> None:
        if self._mode == "title":
            self._title.append(data)
        elif self._mode == "snippet":
            self._snippet.append(data)


def _result_url(href: str | None) -> str | None:
    if not href:
        return None
    absolute = f"https:{href}" if href.startswith("//") else href
    parsed = urlparse(absolute)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [None])[0]
        return unquote(target) if target else None
    return absolute if absolute.startswith(("https://", "http://")) else None


def search_public_web(query: str, *, limit: int = 3) -> list[WebResult]:
    """A no-key public web search capability; results remain explicitly external evidence."""
    if not query.strip():
        return []
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    request = Request(url, headers={"User-Agent": WEB_USER_AGENT})
    try:
        with urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    parser = _DuckDuckGoParser()
    parser.feed(body)
    return [item for item in parser.results if item.url.startswith("http")][:limit]
