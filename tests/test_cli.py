"""Starter tests for the CLI.

Fill in cli.main() then run:
    pytest tests/test_cli.py -v
"""

import json

from simple_vector_search.cli import build_parser, main


def test_parser_add_command():
    args = build_parser().parse_args(
        ["--file", "x.json", "add", "--id", "cat", "--vector", "0.9,0.1,0.2"]
    )
    assert args.command == "add"
    assert args.id == "cat"
    assert args.vector == "0.9,0.1,0.2"


def test_parser_query_defaults():
    args = build_parser().parse_args(["query", "--vector", "1,0,0"])
    assert args.top_k == 3
    assert args.metric == "cosine"


def test_add_then_query_roundtrip(tmp_path):
    path = str(tmp_path / "store.json")
    assert (
        main(
            [
                "--file",
                path,
                "add",
                "--id",
                "cat",
                "--vector",
                "0.9,0.1,0.2",
                "--meta",
                '{"type": "animal"}',
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--file",
                path,
                "add",
                "--id",
                "car",
                "--vector",
                "0.1,0.9,0.3",
            ]
        )
        == 0
    )

    saved = json.loads((tmp_path / "store.json").read_text())
    assert {row["id"] for row in saved} == {"cat", "car"}

    assert main(["--file", path, "query", "--vector", "0.85,0.15,0.15", "--top-k", "1"]) == 0


def test_delete_and_list(tmp_path, capsys):
    path = str(tmp_path / "store.json")
    main(["--file", path, "add", "--id", "cat", "--vector", "1,0,0"])
    assert main(["--file", path, "delete", "--id", "cat"]) == 0
    assert main(["--file", path, "list"]) == 0
    # Once implemented, list should not print "cat".
    # out = capsys.readouterr().out
    # assert "cat" not in out


# Further tests to add:
# - query --filter '{"type": "animal"}'
# - add with --text once CLI embedders exist
# - invalid --vector string ("1,x,2") should return non-zero or raise
# - query with neither --vector nor --text should fail
# - delete missing id: decide between exit 0 and exit 1, then test it
# - `python -m simple_vector_search --help` via subprocess
