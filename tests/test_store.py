"""Starter tests for VectorStore.

These describe the v1 contract. They will fail with NotImplementedError
until you fill in store.py.

    pytest tests/test_store.py -v
"""

import json
from pathlib import Path

import numpy as np
import pytest

from simple_vector_search.store import VectorStore

EXAMPLE_STORE = Path(__file__).resolve().parents[1] / "examples" / "store.json"


def fake_embed(texts):
    """Deterministic stand-in so tests never need a real model.

    Maps a few known strings to 3-D vectors. Accepts str or list[str].
    """
    table = {
        "cat": np.array([0.9, 0.1, 0.2]),
        "dog": np.array([0.8, 0.2, 0.1]),
        "car": np.array([0.1, 0.9, 0.3]),
        "bike": np.array([0.2, 0.8, 0.4]),
    }
    if isinstance(texts, str):
        return table[texts]
    return np.stack([table[t] for t in texts])


@pytest.fixture
def store(tmp_path):
    return VectorStore(filename=str(tmp_path / "store.json"), dim=3)


def test_add_and_get(store):
    store.add("cat", vector=[0.9, 0.1, 0.2], metadata={"type": "animal"})
    row = store.get("cat")
    assert row is not None
    assert row["id"] == "cat"
    assert row["metadata"]["type"] == "animal"
    assert len(store) == 1


def test_add_duplicate_id_raises(store):
    store.add("cat", vector=[0.9, 0.1, 0.2])
    with pytest.raises(ValueError):
        store.add("cat", vector=[0.1, 0.1, 0.1])


def test_get_missing_returns_none(store):
    assert store.get("nope") is None


def test_rejects_wrong_dimension(store):
    with pytest.raises(ValueError):
        store.add("bad", vector=[1.0, 2.0])  # dim was set to 3


def test_infers_dim_from_first_vector(tmp_path):
    s = VectorStore(filename=str(tmp_path / "s.json"))
    s.add("a", vector=[1.0, 2.0, 3.0, 4.0])
    assert s.dim == 4
    with pytest.raises(ValueError):
        s.add("b", vector=[1.0, 2.0])


def test_update_and_delete(store):
    store.add("cat", vector=[0.9, 0.1, 0.2], metadata={"type": "animal"})
    store.update("cat", metadata={"type": "pet"})
    assert store.get("cat")["metadata"]["type"] == "pet"
    assert store.delete("cat") is True
    assert store.get("cat") is None
    assert store.delete("cat") is False


def test_update_missing_id_raises(store):
    with pytest.raises(KeyError):
        store.update("missing", metadata={"x": 1})


def test_add_batch(store):
    store.add_batch(
        [
            {"id": "cat", "vector": [0.9, 0.1, 0.2]},
            {"id": "dog", "vector": [0.8, 0.2, 0.1]},
        ]
    )
    assert len(store) == 2
    assert store.get("dog") is not None


def test_query_cosine_nearest_neighbor(store):
    store.add("cat", vector=[0.9, 0.1, 0.2])
    store.add("dog", vector=[0.8, 0.2, 0.1])
    store.add("car", vector=[0.1, 0.9, 0.3])
    hits = store.query(vector=[0.85, 0.15, 0.15], metric="cosine", top_k=2)
    assert [h["id"] for h in hits] == ["cat", "dog"]
    assert hits[0]["score"] >= hits[1]["score"]


def test_query_euclidean_nearest_neighbor(store):
    store.add("cat", vector=[0.9, 0.1, 0.2])
    store.add("car", vector=[0.1, 0.9, 0.3])
    hits = store.query(vector=[0.85, 0.15, 0.15], metric="euclidean", top_k=1)
    assert hits[0]["id"] == "cat"
    # euclidean: smaller score is closer
    assert hits[0]["score"] >= 0.0


def test_query_metadata_filter(store):
    store.add("cat", vector=[0.9, 0.1, 0.2], metadata={"type": "animal"})
    store.add("dog", vector=[0.8, 0.2, 0.1], metadata={"type": "animal"})
    store.add("car", vector=[0.1, 0.9, 0.3], metadata={"type": "vehicle"})
    hits = store.query(
        vector=[0.85, 0.15, 0.15],
        metric="cosine",
        top_k=5,
        filter={"type": "vehicle"},
    )
    assert [h["id"] for h in hits] == ["car"]


def test_query_unknown_metric_raises(store):
    store.add("cat", vector=[0.9, 0.1, 0.2])
    with pytest.raises(ValueError):
        store.query(vector=[0.9, 0.1, 0.2], metric="manhattan")


def test_save_and_reload(tmp_path):
    path = tmp_path / "store.json"
    s = VectorStore(filename=str(path), dim=3)
    s.add("cat", vector=[0.9, 0.1, 0.2], document="a cat", metadata={"k": "v"})
    s.save()

    raw = json.loads(path.read_text())
    assert isinstance(raw, list)
    assert raw[0]["id"] == "cat"

    loaded = VectorStore(filename=str(path))
    assert len(loaded) == 1
    assert loaded.get("cat")["document"] == "a cat"


def test_add_via_embed_fn(tmp_path):
    s = VectorStore(filename=str(tmp_path / "s.json"), dim=3, embed_fn=fake_embed)
    s.add("c", document="cat", metadata={"type": "animal"})
    row = s.get("c")
    assert row["vector"] == pytest.approx([0.9, 0.1, 0.2])


def test_query_via_text(tmp_path):
    s = VectorStore(filename=str(tmp_path / "s.json"), dim=3, embed_fn=fake_embed)
    s.add("c", document="cat")
    s.add("d", document="dog")
    s.add("v", document="car")
    hits = s.query(text="cat", metric="cosine", top_k=1)
    assert hits[0]["id"] == "c"


def _write(tmp_path, payload, name="store.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload)
    return str(path)


def test_load_missing_file_starts_empty(tmp_path):
    s = VectorStore(filename=str(tmp_path / "nope.json"))
    assert s._records == {}
    assert s.dim is None


def test_load_reads_records(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "id": "cat",
                "vector": [0.9, 0.1, 0.2],
                "document": "a cat",
                "metadata": {"type": "animal"},
            }
        ],
    )
    s = VectorStore(filename=path)
    row = s._records["cat"]
    assert row["vector"] == pytest.approx([0.9, 0.1, 0.2])
    assert row["document"] == "a cat"
    assert row["metadata"] == {"type": "animal"}


def test_load_defaults_document_and_metadata(tmp_path):
    path = _write(tmp_path, [{"id": "cat", "vector": [0.9, 0.1, 0.2]}])
    row = VectorStore(filename=path)._records["cat"]
    assert row["document"] is None
    assert row["metadata"] == {}


def test_load_infers_dim_when_none(tmp_path):
    path = _write(tmp_path, [{"id": "a", "vector": [1.0, 2.0, 3.0, 4.0]}])
    s = VectorStore(filename=path)
    assert s.dim == 4


def test_load_empty_list_keeps_dim_none(tmp_path):
    path = _write(tmp_path, [])
    s = VectorStore(filename=path)
    assert s._records == {}
    assert s.dim is None


def test_load_corrupt_json_raises(tmp_path):
    path = _write(tmp_path, "{not json")
    with pytest.raises(json.JSONDecodeError):
        VectorStore(filename=path)


def test_load_non_list_top_level_raises(tmp_path):
    path = _write(tmp_path, {"id": "cat", "vector": [1.0]})
    with pytest.raises(ValueError):
        VectorStore(filename=path)


def test_load_rejects_mixed_lengths(tmp_path):
    path = _write(
        tmp_path,
        [
            {"id": "a", "vector": [0.1, 0.2, 0.3]},
            {"id": "b", "vector": [0.1, 0.2]},
        ],
    )
    with pytest.raises(ValueError):
        VectorStore(filename=path)


def test_load_rejects_dim_mismatch(tmp_path):
    path = _write(tmp_path, [{"id": "a", "vector": [0.1, 0.2, 0.3, 0.4]}])
    with pytest.raises(ValueError):
        VectorStore(filename=path, dim=3)


def test_load_rejects_duplicate_ids(tmp_path):
    path = _write(
        tmp_path,
        [
            {"id": "a", "vector": [0.1, 0.2, 0.3]},
            {"id": "a", "vector": [0.4, 0.5, 0.6]},
        ],
    )
    with pytest.raises(ValueError):
        VectorStore(filename=path)


def test_load_rejects_row_without_id(tmp_path):
    path = _write(tmp_path, [{"vector": [0.1, 0.2, 0.3]}])
    with pytest.raises(ValueError):
        VectorStore(filename=path)


def test_load_rejects_row_without_vector(tmp_path):
    path = _write(tmp_path, [{"id": "a"}])
    with pytest.raises(ValueError):
        VectorStore(filename=path)


def test_load_rejects_nested_vector(tmp_path):
    path = _write(tmp_path, [{"id": "a", "vector": [[0.1, 0.2], [0.3, 0.4]]}])
    with pytest.raises(ValueError):
        VectorStore(filename=path)


def test_load_rejects_old_format(tmp_path):
    path = _write(tmp_path, [{"vectors": [0.9, 0.1, 0.2], "meta_data": "cat"}])
    with pytest.raises(ValueError):
        VectorStore(filename=path)


def test_load_failure_leaves_store_unchanged(tmp_path):
    path = _write(
        tmp_path,
        [
            {"id": "a", "vector": [0.1, 0.2, 0.3]},
            {"id": "b", "vector": [0.1, 0.2]},
        ],
    )
    with pytest.raises(ValueError):
        VectorStore(filename=path)


def test_example_store_file_is_valid():
    """The reference file in examples/ must stay loadable and well formed."""
    s = VectorStore(filename=str(EXAMPLE_STORE))

    assert s._records
    assert s.dim is not None
    assert len(s._records) == len(set(s._records))

    for row in s._records.values():
        assert isinstance(row["id"], str) and row["id"]
        assert isinstance(row["vector"], list)
        assert all(isinstance(x, float) for x in row["vector"])
        assert len(row["vector"]) == s.dim
        assert row["document"] is None or isinstance(row["document"], str)
        assert isinstance(row["metadata"], dict)


# Further tests to add:
# - add() without vector or document should raise
# - add() with document but no embed_fn should raise
# - query() with neither vector nor text should raise
# - query() on an empty store returns []
# - top_k larger than n returns n hits
# - top_k <= 0 returns []
# - filter matching no rows returns []
# - filter with a metadata key that some rows lack (skip those rows)
# - update() re-embeds when document changes and vector is omitted
# - add_batch is all-or-nothing if one id duplicates / dim is wrong
# - save() creates parent directories
# - save() does not leave a .tmp file behind
# - loading a missing file starts empty (no crash)
# - vectors round-trip through JSON as lists, not numpy arrays
# - __len__ after delete / batch add
# - metadata defaults to {} when omitted
