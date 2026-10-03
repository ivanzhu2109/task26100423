"""进程内引文链查询库：排名最前的 k 条简单引文链。"""
from __future__ import annotations

import heapq

_LOG2_10 = 3.3219280948873626


class _Cost:
    """精确可信度，值为 num / 10**exp（0 < num <= 10**exp）。

    堆序约定：__lt__ 表示"更优"（可信度更大），配合最小堆实现最优优先。
    """

    __slots__ = ("num", "exp", "_approx")

    def __init__(self, num: int, exp: int) -> None:
        self.num = num
        self.exp = exp
        # log2(value) 落在 [_approx - 1, _approx)，仅作保守预筛，不作判定依据
        self._approx = num.bit_length() - exp * _LOG2_10

    def __mul__(self, other: "_Cost") -> "_Cost":
        return _Cost(self.num * other.num, self.exp + other.exp)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Cost):
            return NotImplemented
        if abs(self._approx - other._approx) > 1:
            return False
        if self.exp >= other.exp:
            return self.num == other.num * 10 ** (self.exp - other.exp)
        return self.num * 10 ** (other.exp - self.exp) == other.num

    def __lt__(self, other: "_Cost") -> bool:
        # "更小"即"更优"：可信度更大者排在堆顶
        if self._approx > other._approx + 1:
            return True
        if self._approx < other._approx - 1:
            return False
        if self.exp >= other.exp:
            return self.num > other.num * 10 ** (self.exp - other.exp)
        return self.num * 10 ** (other.exp - self.exp) > other.num


class CitationGraph:
    """只读引用图。nodes 为整数 ID 可迭代对象，edges 为 (起点, 终点, 强度) 三元组。"""

    def __init__(self, nodes, edges) -> None:
        self._adj = {n: [] for n in nodes}
        self._rev = {n: [] for n in nodes}
        for u, v, w in edges:
            cost = _Cost(w, 1)
            self._adj[u].append((v, cost))
            self._rev[v].append((u, cost))

    def trace(self, source: int, target: int, k: int) -> list[list[int]]:
        """返回从 source 到 target 排名前 k 的引文链（节点 ID 序列）。"""
        h = self._heuristic(target)
        if source not in h:
            return []
        results = []
        hc, he = h[source]
        # 堆项：(键可信度, 键边数, 路径, 路径可信度, 路径节点集合)
        # 键 = (路径可信度 × 到终点最大可信度, 路径边数 + 到终点最少边数, 节点序列)
        heap = [(hc, he, (source,), _Cost(1, 0), frozenset((source,)))]
        while heap and len(results) < k:
            _, _, path, cost, seen = heapq.heappop(heap)
            last = path[-1]
            if last == target:
                results.append(list(path))
                continue
            for nbr, wcost in self._adj[last]:
                if nbr in seen:
                    continue
                hn = h.get(nbr)
                if hn is None:
                    continue  # 该节点无法到达终点
                ncost = cost * wcost
                npath = path + (nbr,)
                heapq.heappush(
                    heap,
                    (ncost * hn[0], len(npath) - 1 + hn[1], npath, ncost, seen | {nbr}),
                )
        return results

    def _heuristic(self, target: int) -> dict:
        """反向 Dijkstra：node -> (到 target 的最大可信度 _Cost, 对应最少边数)。"""
        dist = {target: (_Cost(1, 0), 0)}
        heap = [(_Cost(1, 0), 0, target)]
        while heap:
            cost, edges, node = heapq.heappop(heap)
            cur = dist[node]
            if cost != cur[0] or edges != cur[1]:
                continue  # 过期堆项
            for pred, wcost in self._rev[node]:
                ncost = wcost * cost
                nedges = edges + 1
                old = dist.get(pred)
                if old is None or ncost < old[0] or (ncost == old[0] and nedges < old[1]):
                    dist[pred] = (ncost, nedges)
                    heapq.heappush(heap, (ncost, nedges, pred))
        return dist
