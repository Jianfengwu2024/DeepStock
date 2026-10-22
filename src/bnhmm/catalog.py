"""Publish dataset descriptions to Quant Platform; keep every artifact local."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


DEFAULT_SERVER = "https://quant.hapgetaquant.com"


class CatalogError(RuntimeError):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class CatalogClient:
    def __init__(self, server: str, access_token: str, timeout: float = 30.0):
        parsed = urlsplit(server)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}
            or (parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"})
        ):
            raise ValueError("catalog server must be an HTTPS origin or a loopback HTTP origin")
        if not access_token.strip():
            raise ValueError("a Quant Platform access token is required")
        self.server = server.rstrip("/")
        self._token = access_token.strip()
        self.timeout = timeout
        self._opener = build_opener(_NoRedirect())

    def _request(self, path: str, payload: dict[str, Any] | None = None):
        request = Request(
            self.server + "/api/v1/deepstock" + path,
            data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Authorization": "Bearer " + self._token, "Content-Type": "application/json"},
            method="GET" if payload is None else "PUT",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            messages = {401: "login expired or invalid", 403: "fast_screener permission is required to read; administrator permission is required to publish"}
            raise CatalogError(messages.get(exc.code, f"catalog request failed (HTTP {exc.code})")) from None
        except (URLError, TimeoutError) as exc:
            raise CatalogError("catalog server could not be reached") from exc

    def list(self, category: str | None = None) -> dict[str, Any]:
        return self._request("/catalog" + ("?" + urlencode({"category": category}) if category else ""))

    def publish(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return self._request("/admin/catalog/" + quote(metadata["dataset_id"], safe=""), metadata)


def metadata_from_visual_manifest(
    manifest: dict[str, Any], *, category: str, name: str, description: str,
) -> dict[str, Any]:
    """Select descriptive fields explicitly; never upload source paths or images."""
    if manifest.get("status") != "complete" or manifest.get("errors"):
        raise ValueError("only a successfully completed dataset can be published")
    count = manifest.get("image_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("dataset must contain generated samples")
    config = manifest["config"]
    return {
        "dataset_id": manifest["dataset_id"],
        "category": category,
        "name": name,
        "description": description,
        "data_kind": "visual",
        "input_intervals": ["1m"],
        "start_date": config["start"],
        "end_date": config["end"],
        "sample_count": count,
        "feature_schema": {
            "protocol": manifest["protocol"],
            "tensor": manifest["tensor"],
            "encoding": manifest["encoding"],
            "symbol_count": manifest["symbol_count"],
            "generated_at": manifest["generated_at"],
            "end_date_exclusive": True,
        },
        "metadata_version": "1",
        "enabled": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=os.getenv("BNHMM_QUANT_SERVER", DEFAULT_SERVER))
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--category")
    publish = commands.add_parser("publish-visual")
    publish.add_argument("manifest", type=Path)
    publish.add_argument("--category", required=True)
    publish.add_argument("--name", required=True)
    publish.add_argument("--description", required=True)
    args = parser.parse_args(argv)
    try:
        metadata = None
        if args.command == "publish-visual":
            metadata = metadata_from_visual_manifest(
                json.loads(args.manifest.read_text(encoding="utf-8")),
                category=args.category, name=args.name, description=args.description,
            )
        token = os.getenv("BNHMM_QUANT_ACCESS_TOKEN") or getpass.getpass("Quant Platform access token (hidden): ")
        client = CatalogClient(args.server, token)
        result = client.list(args.category) if metadata is None else client.publish(metadata)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (CatalogError, ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
