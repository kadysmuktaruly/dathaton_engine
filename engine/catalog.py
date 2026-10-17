"""Catalog access. Legal references and type names come ONLY from rules/rules_catalog.json."""
from __future__ import annotations

import json
import re
from functools import lru_cache

from engine import config

# Context flags for `only_if` refs. This scene is a bank cash desk, not an exchange office,
# and there is no vault room, so those refs are off unless a rule sets them.
DEFAULT_CONTEXT = {"exchange_cash_desk": False, "vault_entered": False}

# Which catalog refs fit this scene best (points must exist in the catalog; order = priority).
# B5: the safe stands at the cashier's workplace, so p. 80 (keys to devices at the workplace)
# and p. 76 fit better than p. 75 (entry to a vault room).
PREFERRED_POINTS = {"B5": ["п. 80", "п. 76"]}

BANNED = re.compile(r"хищени|краж|мошенни|нарушение установлено|УК РК", re.I)


@lru_cache(maxsize=1)
def load() -> dict:
    with open(config.CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def violation(code: str) -> dict:
    for v in load()["violations"]:
        if v["code"] == code:
            return v
    raise KeyError(code)


def refs(code: str, context: dict | None = None) -> list[dict]:
    ctx = {**DEFAULT_CONTEXT, **(context or {})}
    return [r for r in violation(code)["refs"] if not r.get("only_if") or ctx.get(r["only_if"])]


def safe_refs(code: str, context: dict | None = None) -> list[dict]:
    """Applicable refs whose quote is allowed in user-facing text (no banned words)."""
    rs = [r for r in refs(code, context) if not BANNED.search(r["quote"])]
    pref = PREFERRED_POINTS.get(code, [])
    rs.sort(key=lambda r: pref.index(r["point"]) if r["point"] in pref else len(pref))
    return rs or refs(code, context)[:1]


def norm_text(code: str, context: dict | None = None) -> str:
    """'Правила № 231, п. 20 ч. 1' style citation built from the catalog (up to two refs)."""
    docs = load()["documents"]
    rs = safe_refs(code, context)
    parts = []
    for r in rs[:2]:
        parts.append(f"{docs[r['doc']]['short']}, {r['point']}")
    return "; ".join(parts)


def requirement(code: str, context: dict | None = None) -> str:
    """Short paraphrase of the requirement: the first applicable ref's quote."""
    q = safe_refs(code, context)[0]["quote"].rstrip(".")
    return q[0].lower() + q[1:]


def description(code: str, seen: str, context: dict | None = None) -> str:
    """'<что видно> Это может указывать на <требование>. Норма: <документ>, <пункт>.'"""
    seen = seen.strip().rstrip(".")
    if not seen.lower().startswith("признаки возможного нарушения"):
        seen = "Признаки возможного нарушения: " + seen[0].lower() + seen[1:]
    text = (f"{seen}. Это может указывать на несоблюдение требования: «{requirement(code, context)}». "
            f"Норма: {norm_text(code, context)}.")
    if BANNED.search(text):
        raise ValueError(f"banned wording in description: {text}")
    return text
