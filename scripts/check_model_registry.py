#!/usr/bin/env python3
"""Ground-truth check of the model registry against the provider's own Models API.

    ANTHROPIC_API_KEY=... python scripts/check_model_registry.py

Exit 0 if every Anthropic id in the registry is served by your account, 1 otherwise (and prints which).
Prices cannot be read from the API: compare them with https://platform.claude.com/docs/en/about-claude/pricing.
"""
import json
import os
import sys
import urllib.request
from typing import Callable, Iterable, List

API = "https://api.anthropic.com/v1/models"


def fetch_all_ids(api_key: str, opener: Callable = urllib.request.urlopen) -> List[str]:
    ids, after = [], None
    while True:
        url = API + "?limit=100" + (f"&after_id={after}" if after else "")
        req = urllib.request.Request(url, headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"})  # noqa: S310 - fixed https URL
        with opener(req, timeout=30) as resp:  # noqa: S310 - fixed https URL
            page = json.load(resp)
        ids += [m["id"] for m in page.get("data", [])]
        if not page.get("has_more"):
            return ids
        after = page.get("last_id")


def missing_ids(registry_ids: Iterable[str], served: Iterable[str]) -> List[str]:
    served = set(served)
    return sorted(i for i in set(registry_ids) if i not in served)


def main() -> int:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2
    from sintetico.model_registry import MODEL_REGISTRY
    mine = {c.id for c in MODEL_REGISTRY.values() if c.provider == "anthropic"}
    gone = missing_ids(mine, fetch_all_ids(key))
    print(json.dumps({"checked": sorted(mine), "not_served": gone}, indent=2))
    return 1 if gone else 0


if __name__ == "__main__":
    sys.exit(main())
