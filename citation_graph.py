"""In-process citation-chain query library.

A CitationGraph is built once from nodes and weighted directed edges and is
read-only afterwards.  ``trace`` returns the top-k citation chains between two
nodes ranked by an exact multiplicative credibility score.
"""

from __future__ import annotations

import heapq
import math

__all__ = ["CitationGraph"]

_LN = (math.log(2.0), math.log(3.0), math.log(5.0), math.log(7.0))
_PRIMES = (2, 3, 5, 7)


class _CostKey:
    """Exact order key for a path cost 10^L / prod(strengths).

    The cost is 2^e2 * 3^e3 * 5^e5 * 7^e7 (smaller is better); only primes
    2, 3, 5, 7 occur because strengths are in 1..10.  Comparisons use a
    cached float approximation and fall back to exact big-integer
    comparison when the floats are too close to decide safely, so the
    resulting order is always the exact mathematical order.
    """

    __slots__ = ("exp", "approx")

    def __init__(self, exp, approx=None):
        self.exp = exp  # tuple (e2, e3, e5, e7)
        if approx is None:
            approx = (exp[0] * _LN[0] + exp[1] * _LN[1]
                      + exp[2] * _LN[2] + exp[3] * _LN[3])
        self.approx = approx

    def add(self, edge_exp):
        """Return the key for self * (one edge cost)."""
        a, b, c, d = self.exp
        w, x, y, z = edge_exp
        return _CostKey(
            (a + w, b + x, c + y, d + z),
            self.approx + w * _LN[0] + x * _LN[1] + y * _LN[2] + z * _LN[3],
        )

    def _exact_cmp(self, other):
        pos = neg = 1
        for p, e in zip(_PRIMES, (self.exp[0] - other.exp[0],
                                  self.exp[1] - other.exp[1],
                                  self.exp[2] - other.exp[2],
                                  self.exp[3] - other.exp[3])):
            if e > 0:
                pos *= p ** e
            elif e < 0:
                neg *= p ** (-e)
        return (pos > neg) - (pos < neg)

    def __lt__(self, other):
        diff = self.approx - other.approx
        scale = max(1.0, abs(self.approx), abs(other.approx))
        if diff < -1e-9 * scale:
            return True
        if diff > 1e-9 * scale:
            return False
        return self._exact_cmp(other) < 0

    def __eq__(self, other):
        if not isinstance(other, _CostKey):
            return NotImplemented
        diff = self.approx - other.approx
        scale = max(1.0, abs(self.approx), abs(other.approx))
        if abs(diff) > 1e-9 * scale:
            return False
        return self._exact_cmp(other) == 0

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __hash__(self):
        return hash(self.exp)


_ZERO_COST = _CostKey((0, 0, 0, 0), 0.0)

# Cost exponents of one edge with strength s: 10/s = 2^(1-a) 3^(-b) 5^(1-c) 7^(-d).
_EDGE_COST_EXP = [None] * 11
for _s in range(1, 11):
    _exp = [1, 0, 1, 0]
    _r = _s
    for _i, _p in enumerate(_PRIMES):
        while _r % _p == 0:
            _r //= _p
            _exp[_i] -= 1
    _EDGE_COST_EXP[_s] = tuple(_exp)


class CitationGraph:
    """Read-only directed citation graph with integer edge strengths 1..10."""

    def __init__(self, nodes, edges):
        """Build the graph.

        nodes: iterable of integer node ids.
        edges: iterable of (source, target, strength) triples, strength in
        1..10, at most one edge per ordered node pair.
        """
        self._adj = {}  # node -> list of (neighbor, strength)
        for n in nodes:
            self._adj[n] = []
        for u, v, s in edges:
            self._adj[u].append((v, s))
        for lst in self._adj.values():
            lst.sort()
        # Heuristic tables, computed lazily per target:
        # node -> (_CostKey min remaining cost, min edges among min-cost paths)
        self._heur_cache = {}

    def _heuristic(self, target):
        """Reverse Dijkstra from target over exact edge costs.

        Returns dict node -> (cost_key, min_edges) for nodes that can reach
        target; nodes absent from the dict cannot reach target.
        """
        cached = self._heur_cache.get(target)
        if cached is not None:
            return cached
        # reverse adjacency
        radj = {n: [] for n in self._adj}
        for u, lst in self._adj.items():
            for v, s in lst:
                radj[v].append((u, s))
        dist = {target: (_ZERO_COST, 0)}
        heap = [(_ZERO_COST, 0, target)]
        while heap:
            cost, edges, v = heapq.heappop(heap)
            if dist[v] != (cost, edges):
                continue
            for u, s in radj[v]:
                ncost = cost.add(_EDGE_COST_EXP[s])
                nedges = edges + 1
                cur = dist.get(u)
                if cur is None or (ncost, nedges) < cur:
                    dist[u] = (ncost, nedges)
                    heapq.heappush(heap, (ncost, nedges, u))
        self._heur_cache[target] = dist
        return dist

    def trace(self, source, target, k):
        """Return the top-k citation chains from source to target.

        Each chain is a list of node ids.  Chains are simple (no repeated
        nodes) and follow edge directions.  Ranking: higher credibility
        (product of strength/10 over edges) first; ties broken by fewer
        edges, then by lexicographic node-id order.
        """
        heur = self._heuristic(target)
        if source not in heur or source not in self._adj or k < 1:
            return []
        results = []
        # Best-first enumeration over simple-path prefixes.  Each heap entry
        # is ordered by (estimated total cost, estimated total edges, path);
        # the estimate adds the exact reverse-Dijkstra heuristic to the exact
        # cost so far, so it never underestimates a completion.  Popping a
        # complete path therefore yields the next chain in exact rank order.
        hcost, hedges = heur[source]
        heap = [(hcost, hedges, (source,), source, _ZERO_COST, 0)]
        while heap and len(results) < k:
            _, _, path, v, gcost, gedges = heapq.heappop(heap)
            if v == target:
                results.append(list(path))
                continue
            visited = set(path)
            for u, s in self._adj[v]:
                if u in visited:
                    continue
                hu = heur.get(u)
                if hu is None:
                    continue  # cannot reach target; prune at push time
                ngcost = gcost.add(_EDGE_COST_EXP[s])
                ngedges = gedges + 1
                est_cost = ngcost.add(hu[0].exp)
                est_edges = ngedges + hu[1]
                heapq.heappush(
                    heap, (est_cost, est_edges, path + (u,), u, ngcost, ngedges)
                )
        return results
