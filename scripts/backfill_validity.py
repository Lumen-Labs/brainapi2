#!/usr/bin/env python3
"""Fill missing valid_from / valid_to on an existing brain.

Safe to run twice on a live brain: existing stamps are not rewritten, live
copies are not closed, and a second pass reports zero writes.

    poetry run python scripts/backfill_validity.py --brain-id BRAIN --dry-run
    poetry run python scripts/backfill_validity.py --brain-id BRAIN

Uses DATA_DB / GRAPH_DB and the same connection env as the API (Postgres or
Mongo for chunks; Neo4j or NetworkX for relationships). See
docs/validity-backfill.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _adapters():
    from src.adapters.data import DataAdapter
    from src.adapters.graph import GraphAdapter
    from src.config import config

    data = DataAdapter()
    if config.data_db == "postgresql":
        from src.lib.postgresql.data import get_postgresql_data_client

        data.add_client(get_postgresql_data_client())
    elif config.data_db == "mongo":
        from src.lib.mongo.client import _mongo_client

        data.add_client(_mongo_client)
    else:
        raise SystemExit(f"Unsupported DATA_DB={config.data_db!r}")

    graph = GraphAdapter()
    if config.graph_db == "neo4j":
        from src.lib.neo4j.client import Neo4jClient

        graph.add_client(Neo4jClient())
    elif config.graph_db == "networkx":
        from src.lib.postgresql.networkx_client import get_networkx_graph_client

        graph.add_client(get_networkx_graph_client())
    else:
        raise SystemExit(f"Unsupported GRAPH_DB={config.graph_db!r}")
    return data, graph


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--brain-id", required=True, help="Brain to backfill")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count missing stamps without writing",
    )
    args = parser.parse_args(argv)

    from src.core.saving.validity_backfill import backfill_brain_validity

    data, graph = _adapters()
    summary = backfill_brain_validity(
        data, graph, args.brain_id, dry_run=args.dry_run
    )
    payload = {
        "brain_id": args.brain_id,
        "dry_run": args.dry_run,
        **summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
