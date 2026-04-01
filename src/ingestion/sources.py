"""Document source fetchers — SEC EDGAR filings and generic URL sources."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import xxhash

from config.settings import settings


class SECEdgarSource:
    """Fetches SEC EDGAR filings (10-K, 10-Q, 8-K) via the EDGAR full-text search API."""

    BASE_URL = "https://efts.sec.gov/LATEST/search-index"
    FILING_URL = "https://www.sec.gov/Archives/edgar/data"
    SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
    FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": settings.sec_edgar_user_agent,
            "Accept-Encoding": "gzip, deflate",
        })

    def search_filings(self, query: str, form_types: list[str] | None = None,
                       date_start: str | None = None, date_end: str | None = None,
                       max_results: int = 20) -> list[dict]:
        """Search SEC EDGAR for filings matching criteria."""
        params = {"q": query, "dateRange": "custom", "startdt": date_start or "2023-01-01",
                  "enddt": date_end or datetime.now().strftime("%Y-%m-%d")}
        if form_types:
            params["forms"] = ",".join(form_types)

        resp = self.session.get(self.FULL_TEXT_SEARCH, params=params)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for hit in data.get("hits", {}).get("hits", [])[:max_results]:
            source = hit.get("_source", {})
            results.append({
                "accession_number": source.get("file_num", ""),
                "form_type": source.get("form_type", ""),
                "company_name": source.get("display_names", [""])[0] if source.get("display_names") else "",
                "filing_date": source.get("file_date", ""),
                "file_url": source.get("file_url", ""),
                "source_type": "sec_filing",
            })
        return results

    def download_filing(self, file_url: str, landing_dir: str | None = None) -> dict:
        """Download a single filing to the landing zone."""
        landing = Path(landing_dir or settings.landing_dir)
        landing.mkdir(parents=True, exist_ok=True)

        url = file_url if file_url.startswith("http") else f"https://www.sec.gov{file_url}"
        resp = self.session.get(url)
        resp.raise_for_status()
        content = resp.content

        file_hash = xxhash.xxh64(content).hexdigest()
        filename = f"{file_hash}_{Path(url).name}"
        file_path = landing / filename
        file_path.write_bytes(content)

        # Respect SEC rate limits
        time.sleep(0.15)

        return {
            "source_url": url,
            "file_path": str(file_path),
            "file_hash": file_hash,
            "file_size_bytes": len(content),
            "download_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_type": "sec_filing",
        }


class GenericURLSource:
    """Fetches documents from arbitrary URLs (PDFs, HTML pages)."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "RAGPipeline/1.0"})

    def download(self, url: str, source_type: str = "generic",
                 landing_dir: str | None = None) -> dict:
        landing = Path(landing_dir or settings.landing_dir)
        landing.mkdir(parents=True, exist_ok=True)

        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        content = resp.content

        file_hash = xxhash.xxh64(content).hexdigest()
        ext = Path(url).suffix or ".html"
        filename = f"{file_hash}_{source_type}{ext}"
        file_path = landing / filename
        file_path.write_bytes(content)

        return {
            "source_url": url,
            "file_path": str(file_path),
            "file_hash": file_hash,
            "file_size_bytes": len(content),
            "download_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_type": source_type,
        }


class LocalFileSource:
    """Ingests documents from a local directory (for testing/development)."""

    def ingest_directory(self, directory: str, source_type: str = "local",
                         landing_dir: str | None = None) -> list[dict]:
        landing = Path(landing_dir or settings.landing_dir)
        landing.mkdir(parents=True, exist_ok=True)
        results = []

        source_dir = Path(directory)
        for file_path in source_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in (".txt", ".pdf", ".html", ".htm", ".md"):
                content = file_path.read_bytes()
                file_hash = xxhash.xxh64(content).hexdigest()
                dest = landing / f"{file_hash}_{file_path.name}"
                dest.write_bytes(content)

                results.append({
                    "source_url": f"file://{file_path.resolve()}",
                    "file_path": str(dest),
                    "file_hash": file_hash,
                    "file_size_bytes": len(content),
                    "download_timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_type": source_type,
                })

        return results
