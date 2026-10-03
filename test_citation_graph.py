import random
import resource
import time
import unittest
from fractions import Fraction

from citation_graph import CitationGraph, _CostKey, _EDGE_COST_EXP


def cost_key_of(strengths):
    key = _CostKey((0, 0, 0, 0), 0.0)
    for s in strengths:
        key = key.add(_EDGE_COST_EXP[s])
    return key


def cost_fraction_of(strengths):
    num, den = 1, 1
    for s in strengths:
        num, den = num * 10, den * s
    return Fraction(num, den)


class CostKeyExactnessTest(unittest.TestCase):
    """The internal order key must always agree with exact Fraction math."""

    def test_random_sequences_match_fraction(self):
        rng = random.Random(99)
        for _ in range(2000):
            a = [rng.randint(1, 10) for _ in range(rng.randint(1, 600))]
            b = [rng.randint(1, 10) for _ in range(rng.randint(1, 600))]
            ka, kb = cost_key_of(a), cost_key_of(b)
            fa, fb = cost_fraction_of(a), cost_fraction_of(b)
            self.assertEqual(fa < fb, ka < kb)
            self.assertEqual(fa == fb, ka == kb)
            self.assertEqual(fa <= fb, ka <= kb)

    def test_exact_cmp_matches_fraction_for_huge_exponents(self):
        rng = random.Random(7)
        for _ in range(100):
            ea = tuple(rng.randint(-20000, 20000) for _ in range(4))
            eb = tuple(rng.randint(-20000, 20000) for _ in range(4))

            # exact reference sign of 2^ea 3^ea 5^ea 7^ea - (same for eb),
            # via cross-multiplied big integers (no gcd reduction)
            def signed_parts(exp):
                pos = neg = 1
                for p, e in zip((2, 3, 5, 7), exp):
                    if e >= 0:
                        pos *= p ** e
                    else:
                        neg *= p ** (-e)
                return pos, neg

            pa, na = signed_parts(ea)
            pb, nb = signed_parts(eb)
            diff = pa * nb - pb * na
            expected_sign = (diff > 0) - (diff < 0)
            self.assertEqual(expected_sign,
                             _CostKey(ea)._exact_cmp(_CostKey(eb)))

    def test_near_tie_chains_ranked_exactly(self):
        # prod 4374 = 2*3^7 vs prod 4375 = 5^4*7, both 8 edges:
        # credibilities differ by a factor of 4375/4374; the lexicographically
        # larger chain (huge node ids) must win.
        strengths_a = [2] + [3] * 7
        strengths_b = [5] * 4 + [7] + [1] * 3
        edges = []
        path_a = [0] + list(range(1, 8)) + [100]
        path_b = [0] + list(range(1000, 1007)) + [100]
        for (u, v), s in zip(zip(path_a, path_a[1:]), strengths_a):
            edges.append((u, v, s))
        for (u, v), s in zip(zip(path_b, path_b[1:]), strengths_b):
            edges.append((u, v, s))
        nodes = [0, 100] + list(range(1, 8)) + list(range(1000, 1007))
        graph = CitationGraph(nodes, edges)
        self.assertEqual([path_b, path_a], graph.trace(0, 100, 2))


def oracle_top_k(nodes, edges, source, target, k):
    """Independent brute-force reference: enumerate ALL simple chains, sort
    with exact Fraction arithmetic, return the top-k.  Small graphs only."""
    adj = {n: [] for n in nodes}
    for u, v, s in edges:
        adj[u].append((v, s))
    chains = []

    def dfs(path, on_path, cred):
        u = path[-1]
        if u == target:
            chains.append((cred, len(path) - 1, path))
            return
        for v, s in adj[u]:
            if v not in on_path:
                on_path.add(v)
                dfs(path + [v], on_path, cred * Fraction(s, 10))
                on_path.discard(v)

    on_path = {source}
    dfs([source], on_path, Fraction(1))
    # rank: credibility desc, edge count asc, node-id sequence asc
    chains.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [list(path) for _, _, path in chains[:k]]


class ExhaustiveSmallGraphTest(unittest.TestCase):
    """Compare the full returned sequence against the brute-force oracle."""

    def check(self, nodes, edges, source, target, max_k=20):
        graph = CitationGraph(nodes, edges)
        expected_all = oracle_top_k(nodes, edges, source, target, 10 ** 9)
        for k in range(1, 21):
            got = graph.trace(source, target, k)
            self.assertEqual(expected_all[:k], got, f"k={k}")
        # k prefix property: larger k only extends the same ranking
        for k in range(1, 20):
            self.assertEqual(graph.trace(source, target, k),
                             graph.trace(source, target, k + 1)[:k])

    def test_handcrafted_graph(self):
        # Contains: a strength-10 cycle (2<->5), chains sharing prefixes and
        # suffixes, equal credibility with different lengths
        # ([1,4] cred 1/10 vs [1,2,4] cred 1/10), and equal credibility with
        # equal length ([1,2,4] vs [1,3,4] -> lexicographic order).
        nodes = [1, 2, 3, 4, 5, 6]
        edges = [
            (1, 4, 1),           # cred 1/10, 1 edge
            (1, 2, 5), (2, 4, 2),   # cred 1/10, 2 edges
            (1, 3, 2), (3, 4, 5),   # cred 1/10, 2 edges
            (2, 5, 10), (5, 2, 10), # strength-10 cycle
            (5, 6, 10), (6, 4, 10), # detour via cycle exit, cred unchanged
            (3, 6, 10),
        ]
        self.check(nodes, edges, 1, 4)

    def test_random_small_graphs(self):
        rng = random.Random(20261004)
        for trial in range(30):
            n = rng.randint(2, 8)
            nodes = list(range(n))
            edges = []
            for u in nodes:
                for v in nodes:
                    if u != v and rng.random() < 0.35:
                        edges.append((u, v, rng.randint(1, 10)))
            source, target = 0, n - 1
            if source == target:
                continue
            self.check(nodes, edges, source, target)

    def test_no_chain_returns_empty(self):
        graph = CitationGraph([1, 2, 3], [(1, 2, 5)])
        self.assertEqual([], graph.trace(1, 3, 5))
        self.assertEqual([], graph.trace(2, 1, 5))


class LongChainRankingTest(unittest.TestCase):
    """Two >400-edge chains with credibility below 1e-400, unequal, and the
    correct ranking is the opposite of node-id lexicographic order."""

    def setUp(self):
        edges = []
        # chain A: small ids, 406 edges all strength 1 -> cred 10^-406
        a_nodes = list(range(1, 406))
        path_a = [0] + a_nodes + [10 ** 6]
        for u, v in zip(path_a, path_a[1:]):
            edges.append((u, v, 1))
        # chain B: large ids, 405 edges strength 1 + one edge strength 2
        # -> cred 2*10^-406 (better, but lexicographically larger)
        b_nodes = list(range(10 ** 5, 10 ** 5 + 405))
        path_b = [0] + b_nodes + [10 ** 6]
        strengths = [1] * 405 + [2]
        for (u, v), s in zip(zip(path_b, path_b[1:]), strengths):
            edges.append((u, v, s))
        self.nodes = [0, 10 ** 6] + a_nodes + b_nodes
        self.edges = edges
        self.source, self.target = 0, 10 ** 6
        self.path_a = path_a
        self.path_b = path_b

    def test_exact_ranking_beats_lexicographic(self):
        graph = CitationGraph(self.nodes, self.edges)
        got = graph.trace(self.source, self.target, 2)
        self.assertEqual([self.path_b, self.path_a], got)

    def test_k_selects_prefix_of_same_ranking(self):
        graph = CitationGraph(self.nodes, self.edges)
        full = graph.trace(self.source, self.target, 2)
        for k in (1, 2):
            self.assertEqual(full[:k], graph.trace(self.source, self.target, k))


class DiamondScaleTest(unittest.TestCase):
    """80 series diamonds -> 2^80 equal chains, plus a dead-end cyclic
    region reachable from source but unable to reach target.
    50000 nodes, 200000 edges total.  Query the top 20 exactly."""

    DIAMONDS = 80
    TOTAL_NODES = 50000
    TOTAL_EDGES = 200000

    @classmethod
    def build(cls):
        rng = random.Random(7)
        entries = list(range(cls.DIAMONDS + 1))          # 0..80
        mids_a = [100 + 2 * i for i in range(cls.DIAMONDS)]
        mids_b = [100 + 2 * i + 1 for i in range(cls.DIAMONDS)]
        main_nodes = entries + mids_a + mids_b           # 241 nodes
        dead_nodes = list(range(1000, 1000 + (cls.TOTAL_NODES - len(main_nodes))))
        edges = []
        for i in range(cls.DIAMONDS):
            edges.append((entries[i], mids_a[i], 10))
            edges.append((entries[i], mids_b[i], 10))
            edges.append((mids_a[i], entries[i + 1], 10))
            edges.append((mids_b[i], entries[i + 1], 10))
        # dead region: reachable from source, cyclic, cannot reach target
        edges.append((entries[0], dead_nodes[0], 10))
        edges.append((dead_nodes[0], dead_nodes[1], 5))
        edges.append((dead_nodes[1], dead_nodes[2], 5))
        edges.append((dead_nodes[2], dead_nodes[0], 5))  # explicit cycle
        seen = {(u, v) for u, v, _ in edges}
        while len(edges) < cls.TOTAL_EDGES:
            u = rng.choice(dead_nodes)
            v = rng.choice(dead_nodes)
            if u != v and (u, v) not in seen:
                seen.add((u, v))
                edges.append((u, v, rng.randint(1, 10)))
        nodes = main_nodes + dead_nodes
        assert len(nodes) == cls.TOTAL_NODES
        assert len(edges) == cls.TOTAL_EDGES
        return nodes, edges, entries, mids_a, mids_b

    @staticmethod
    def expected_top20(entries, mids_a, mids_b):
        # All 2^80 chains tie on (credibility 1, 160 edges); ranking is pure
        # lexicographic on node ids.  Chain j differs from chain 0 by taking
        # the b-exit at diamond i exactly when bit (79-i) of j is set.
        expected = []
        for j in range(20):
            path = []
            for i in range(80):
                path.append(entries[i])
                path.append(mids_b[i] if (j >> (79 - i)) & 1 else mids_a[i])
            path.append(entries[80])
            expected.append(path)
        return expected

    def test_top20_exact_order_time_and_memory(self):
        nodes, edges, entries, mids_a, mids_b = self.build()
        graph = CitationGraph(nodes, edges)  # construction not timed
        start = time.perf_counter()
        got = graph.trace(entries[0], entries[-1], 20)
        elapsed = time.perf_counter() - start
        expected = self.expected_top20(entries, mids_a, mids_b)
        self.assertEqual(expected, got)
        peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
        print(f"\n[diamond-scale] query time: {elapsed:.3f}s, "
              f"peak RSS: {peak_mib:.1f} MiB")
        self.assertLess(elapsed, 20.0)
        self.assertLess(peak_mib, 512.0)


if __name__ == "__main__":
    unittest.main()
