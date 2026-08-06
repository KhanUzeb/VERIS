import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veris.cache import ScoreCache, cache_key
from veris.schemas import GoldenCase


def make_case():
    return GoldenCase(id="c1", input="what is 2+2", expected_output="4", retrieval_context=["math facts"])


def test_cache_miss_then_hit(tmp_path):
    cache = ScoreCache(db_path=tmp_path / "cache.db")
    case = make_case()
    key = cache_key(case, "4", "groq/test-model")

    assert cache.get(key) is None
    cache.set(key, {"faithfulness": 0.9})
    assert cache.get(key) == {"faithfulness": 0.9}
    cache.close()


def test_cache_key_changes_with_output(tmp_path):
    case = make_case()
    key_a = cache_key(case, "4", "groq/test-model")
    key_b = cache_key(case, "5", "groq/test-model")
    assert key_a != key_b


def test_cache_key_changes_with_judge_model(tmp_path):
    case = make_case()
    key_a = cache_key(case, "4", "groq/model-a")
    key_b = cache_key(case, "4", "groq/model-b")
    assert key_a != key_b


def test_cache_stats_tracks_hits(tmp_path):
    cache = ScoreCache(db_path=tmp_path / "cache.db")
    case = make_case()
    key = cache_key(case, "4", "groq/test-model")
    cache.set(key, {"faithfulness": 0.9})
    cache.get(key)
    cache.get(key)
    stats = cache.stats()
    assert stats["cached_entries"] == 1
    assert stats["total_hits"] == 2
    cache.close()
