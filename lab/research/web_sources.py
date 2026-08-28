from __future__ import annotations

from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


class PublicResearchSources:
    """Low-dependency public research lookup. Failures are non-fatal."""

    def arxiv(self, query: str, max_results: int = 5) -> list[dict]:
        params = urlencode({
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
        })
        url = f"https://export.arxiv.org/api/query?{params}"
        req = Request(url, headers={"User-Agent": "AutonomousCryptoTradingLab/0.1"})
        try:
            with urlopen(req, timeout=10) as response:
                root = ET.fromstring(response.read())
        except Exception:
            return []
        ns = {"a": "http://www.w3.org/2005/Atom"}
        results = []
        for entry in root.findall("a:entry", ns):
            results.append({
                "title": (entry.findtext("a:title", "", ns) or "").strip(),
                "summary": (entry.findtext("a:summary", "", ns) or "").strip(),
                "published": entry.findtext("a:published", "", ns) or "",
                "url": entry.findtext("a:id", "", ns) or "",
            })
        return results
