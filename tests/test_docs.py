from pathlib import Path
import sintetico.autonomy as autonomy_mod
from sintetico.docs import ADRArchive, ADRGenerator, DocumentGenerator


def test_document_module_uses_real_signatures_and_real_test_examples():
    gen = DocumentGenerator(test_dirs=[Path(__file__).parent])
    md = gen.document_module(autonomy_mod)
    assert "threshold_for_error_cost(error_cost: float) -> float" in md
    assert "threshold_for_error_cost(" in md.split("threshold_for_error_cost(error_cost")[1]   # example from tests included


def test_adr_only_for_commits_that_sound_architectural():
    g = ADRGenerator()
    assert g.from_commit("fix typo in readme") is None
    adr = g.from_commit("Migrate sessions from Redis to Postgres")
    assert adr and adr.status == "proposed" and "Redis" in adr.title


def test_adr_with_llm_splits_paragraphs():
    g = ADRGenerator(llm_call=lambda p: "ctx para\n\ndecision para\n\nconsequence para")
    adr = g.from_commit("Introduce event bus architecture", "diff")
    assert (adr.context, adr.decision, adr.consequences) == ("ctx para", "decision para", "consequence para")


def test_adr_archive_numbering_and_search(tmp_path):
    a = ADRArchive(tmp_path / "adr")
    g = ADRGenerator()
    p1 = a.save(g.from_decision("Use Kafka for events", "volume", "adopt kafka", "ops cost"))
    p2 = a.save(g.from_decision("Use Postgres for state", "consistency", "adopt postgres", "backups"))
    assert p1.name.startswith("0001-") and p2.name.startswith("0002-")
    assert a.search("kafka events") == [p1]
    assert a.search("zzzz") == []
