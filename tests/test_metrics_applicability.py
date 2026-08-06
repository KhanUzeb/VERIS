import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veris.cache import cache_key
from veris.config import metrics_fingerprint
from veris.schemas import GoldenCase


def make_case():
    return GoldenCase(id="c1", input="what is 2+2", expected_output="4", retrieval_context=["math facts"])


def test_metrics_fingerprint_is_stable():
    a = metrics_fingerprint()
    b = metrics_fingerprint()
    assert a == b
    assert len(a) == 16


def test_cache_key_includes_metrics_fingerprint(monkeypatch):
    case = make_case()
    key_a = cache_key(case, "4", "groq/test-model")

    import veris.cache as cache_mod

    monkeypatch.setattr(cache_mod, "metrics_fingerprint", lambda: "deadbeefdeadbeef")
    key_b = cache_key(case, "4", "groq/test-model")
    assert key_a != key_b
