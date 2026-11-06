"""Deterministic, lightweight relationship communities for the Explore atlas."""

from collections import Counter
from hashlib import sha256

import networkx as nx


def build_communities(nodes, relationships):
    graph = nx.Graph()
    metadata = {str(n["id"]): n for n in nodes}
    graph.add_nodes_from(sorted(metadata))
    valid_edges = sorted(
        (str(e["fromId"]), str(e["toId"]))
        for e in relationships
        if str(e["fromId"]) in metadata and str(e["toId"]) in metadata
    )
    for source, target in valid_edges:
        if source != target:
            graph.add_edge(
                source,
                target,
                weight=graph.get_edge_data(source, target, {}).get("weight", 0) + 1,
            )
    connected = graph.subgraph([n for n, d in graph.degree() if d]).copy()
    partitions = (
        list(nx.community.louvain_partitions(connected, seed=17, weight="weight"))
        if connected
        else []
    )
    # Louvain yields fine-to-coarse partitions. Walk the hierarchy from its root.
    levels = list(reversed(partitions))
    isolated = sorted(set(graph) - set(connected))
    if not levels:
        levels = [[]]
    if isolated:
        # Keep disconnected entities reachable without a node per isolated entity at overview.
        for level in levels:
            level.append(set(isolated))
    # Bound disconnected-component overviews as well as dense graphs. These
    # containers preserve the Louvain children rather than discarding regions.
    if len(levels[0]) > 200:
        ordered = sorted(levels[0], key=lambda m: min(m))
        width = (len(ordered) + 199) // 200
        levels.insert(
            0,
            [
                set().union(*ordered[i : i + width])
                for i in range(0, len(ordered), width)
            ],
        )
    communities, memberships = [], []
    previous = [(None, set(graph))]
    seen = {}
    for partition in levels:
        current = []
        for members in sorted(partition, key=lambda m: min(m)):
            members = set(members)
            identity = sha256("\0".join(sorted(members)).encode()).hexdigest()[:24]
            if identity in seen:
                current.append((seen[identity], members))
                continue
            parent = next(
                (cid for cid, parent_members in previous if members <= parent_members),
                None,
            )
            representatives = sorted(
                members,
                key=lambda n: (-graph.degree(n), metadata[n].get("name") or "", n),
            )[:3]
            types = Counter(t for n in members for t in metadata[n].get("types", []))
            names = [metadata[n].get("name") or "Unnamed" for n in representatives]
            cid = "group:" + identity
            communities.append(
                {
                    "id": cid,
                    "parentId": parent,
                    "name": " · ".join(names[:2]),
                    "count": len(members),
                    "types": [t for t, _ in types.most_common(3)],
                    "representatives": [metadata[n] for n in representatives],
                }
            )
            memberships.extend((cid, n) for n in sorted(members))
            seen[identity] = cid
            current.append((cid, members))
        previous = current
    return {
        "groups": communities,
        "totals": {"things": len(metadata), "connections": len(valid_edges)},
    }, memberships
