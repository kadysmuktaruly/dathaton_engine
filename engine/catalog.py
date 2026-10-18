"""Catalog access. Legal references and type names come ONLY from rules/rules_catalog.json.

Only codes marked `"active": true` may be reported (docs/master-document.md); the other codes stay
in the catalog as reference. A ref may come from the bank's internal documents (doc INTERNAL) and
have no quote; an entry's optional `requirement` then states the rule in plain words.
"""
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


def active_codes() -> set[str]:
    return {v["code"] for v in load()["violations"] if v.get("active")}


def active_types() -> set[str]:
    return {v["type"] for v in load()["violations"] if v.get("active")}


def refs(code: str, context: dict | None = None) -> list[dict]:
    ctx = {**DEFAULT_CONTEXT, **(context or {})}
    return [r for r in violation(code)["refs"] if not r.get("only_if") or ctx.get(r["only_if"])]


def safe_refs(code: str, context: dict | None = None) -> list[dict]:
    """Applicable refs whose quote is allowed in user-facing text (no banned words)."""
    rs = [r for r in refs(code, context) if not BANNED.search(r.get("quote", ""))]
    pref = PREFERRED_POINTS.get(code, [])
    rs.sort(key=lambda r: pref.index(r["point"]) if r["point"] in pref else len(pref))
    return rs or refs(code, context)[:1]


def norm_text(code: str, context: dict | None = None) -> str:
    """'Правила № 231, п. 20 ч. 1' style citation built from the catalog (up to two refs)."""
    docs = load()["documents"]
    parts = []
    for r in safe_refs(code, context)[:2]:
        short = docs[r["doc"]]["short"]
        parts.append(f"{short}, {r['point']}" if r.get("point") else short)
    return "; ".join(parts)


def requirement(code: str, context: dict | None = None) -> tuple[str, bool]:
    """(text, is_quote): the entry's plain-words `requirement` if it has one, else the first
    applicable quote (lower-cased first letter)."""
    v = violation(code)
    if v.get("requirement"):
        return v["requirement"].strip().rstrip("."), False
    quoted = [r for r in safe_refs(code, context) if r.get("quote")]
    if not quoted:
        raise ValueError(f"{code}: no quote and no requirement in the catalog")
    q = quoted[0]["quote"].rstrip(".")
    return q[0].lower() + q[1:], True


def description(code: str, seen: str, context: dict | None = None) -> str:
    """'<что видно> Это может указывать на <требование>. Норма: <документ>, <пункт>.'"""
    seen = seen.strip().rstrip(".")
    if not seen.lower().startswith("признаки возможного нарушения"):
        seen = "Признаки возможного нарушения: " + seen[0].lower() + seen[1:]
    req, is_quote = requirement(code, context)
    req = f": «{req}»" if is_quote else f" {req}"
    text = f"{seen}. Это может указывать на несоблюдение требования{req}. Норма: {norm_text(code, context)}."
    if BANNED.search(text):
        raise ValueError(f"banned wording in description: {text}")
    return text
