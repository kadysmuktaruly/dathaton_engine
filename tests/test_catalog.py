"""Catalog consistency with docs/master-document.md."""
import json

from engine import catalog, config

# Codes the master document keeps (incl. optional A4); everything else is reference only.
MASTER_CODES = {"A1", "A2", "A3", "A4", "A6", "A7", "B1", "B3", "B5", "B8", "B9", "C2", "D6", "D7"}


def test_active_codes_match_master_document():
    assert catalog.active_codes() == MASTER_CODES


def test_catalog_copies_identical():
    root = config.ROOT / "rules_catalog.json"
    assert json.loads(root.read_text(encoding="utf-8")) == catalog.load()


def test_types_unique():
    types = [v["type"] for v in catalog.load()["violations"]]
    assert len(types) == len(set(types))


def test_every_active_code_has_requirement_and_clean_wording():
    for code in catalog.active_codes():
        v = catalog.violation(code)
        for k in ("visual_cues", "not_flag", "timecode_rule"):
            assert v.get(k), (code, k)
        d = catalog.description(code, "в кадре видно действие")
        assert d.startswith("Признаки возможного нарушения") and "Норма:" in d
        assert not catalog.BANNED.search(d)


def test_refs_point_to_known_documents():
    docs = catalog.load()["documents"]
    for v in catalog.load()["violations"]:
        for r in v["refs"]:
            assert r["doc"] in docs, (v["code"], r["doc"])
            assert r.get("quote") or r["doc"] == "INTERNAL", (v["code"], r)


def test_client_id_check_not_in_do_not_flag():
    """A6 (no client ID) is a reported type now."""
    assert all("документа клиента" not in d["what"] for d in catalog.load()["do_not_flag"])
