import time

from src.services.explore.communities import build_communities

n = 50000
nodes = [
    {"id": str(i), "name": f"Entity {i}", "types": ["TYPE_" + str(i % 8)]}
    for i in range(n)
]
edges = [
    {"fromId": str(i), "toId": str((i + step) % n)}
    for i in range(n)
    for step in (1, 2, 5, 31)
]
started = time.perf_counter()
payload, members = build_communities(nodes, edges)
roots = {g["id"] for g in payload["groups"] if g["parentId"] is None}
assert len([1 for g, n in members if g in roots]) == 50000
print(
    {
        "seconds": round(time.perf_counter() - started, 3),
        "entities": n,
        "relationships": len(edges),
        "groups": len(payload["groups"]),
        "root_groups": len(roots),
        "memberships": len(members),
    }
)
