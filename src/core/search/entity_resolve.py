from __future__ import annotations

from src.constants.kg import Node
from src.services.kg_agent.main import embeddings_adapter, graph_adapter, vector_store_adapter
from src.utils.vector_search import VectorSearchFacade

vector_search = VectorSearchFacade(vector_store_adapter)


def node_matches_target(node: Node | None, target: str) -> bool:
    wanted = (target or "").strip()
    if node is None or not wanted:
        return False
    uuid = str(getattr(node, "uuid", "") or "").strip()
    if uuid and uuid == wanted:
        return True
    name = str(getattr(node, "name", "") or "").strip()
    return bool(name) and name.casefold() == wanted.casefold()


def resolve_entity_node(target: str, brain_id: str) -> Node | None:
    wanted = (target or "").strip()
    if not wanted:
        return None
    node = graph_adapter.get_by_uuid(wanted, brain_id=brain_id)
    if node_matches_target(node, wanted):
        return node
    try:
        embedding = embeddings_adapter.embed_text(wanted)
        hits = vector_search.search_nodes(
            embedding.embeddings,
            brain_id=brain_id,
        )
    except Exception:
        return None
    for hit in hits:
        metadata = getattr(hit, "metadata", None) or {}
        uuid = metadata.get("uuid")
        if not uuid:
            continue
        candidate = graph_adapter.get_by_uuid(uuid, brain_id=brain_id)
        if node_matches_target(candidate, wanted):
            return candidate
    return None
