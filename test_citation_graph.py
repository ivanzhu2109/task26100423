"""citation_graph 的自动化测试（仅依赖标准库 unittest）。"""
import resource
import random
import sys
import time
import unittest
from fractions import Fraction

from citation_graph import CitationGraph


def brute_force_trace(nodes, edges, source, target, k):
    """独立对照：穷举全部简单链后精确排序。仅用于小图测试。"""
    adj = {}
    for u, v, w in edges:
        adj.setdefault(u, []).append((v, w))
    found = []
    stack = [(source, [source], Fraction(1), {source})]
    while stack:
        node, path, cred, seen = stack.pop()
        if node == target:
            found.append((cred, path))
            continue
        for nbr, w in adj.get(node, []):
            if nbr not in seen:
                stack.append((nbr, path + [nbr], cred * Fraction(w, 10), seen | {nbr}))
    found.sort(key=lambda item: (-item[0], len(item[1]) - 1, item[1]))
    return [path for _, path in found[:k]]


# 小图：含强度10的环 1<->5、共享前后缀的链、
# 可信度相同长度不同（如 (0,4,7) 与 (0,1,3,7)）、
# 可信度和长度都相同（如 (0,1,3,7) 与 (0,2,3,7)、(0,1,6,7) 与 (0,2,6,7)）。
SMALL_NODES = list(range(8))
SMALL_EDGES = [
    (0, 1, 10), (0, 2, 10), (0, 4, 10), (0, 6, 5),
    (1, 3, 10), (1, 5, 10), (1, 6, 5),
    (2, 3, 10), (2, 6, 5),
    (3, 7, 10), (4, 7, 10),
    (5, 1, 10), (5, 2, 10),
    (6, 7, 10),
]
SMALL_EXPECTED = [
    [0, 4, 7],           # cred 1,   2 边
    [0, 1, 3, 7],        # cred 1,   3 边，字典序小
    [0, 2, 3, 7],        # cred 1,   3 边
    [0, 1, 5, 2, 3, 7],  # cred 1,   5 边（经过强度10的环上的边，但不重复节点）
    [0, 6, 7],           # cred 0.5, 2 边
    [0, 1, 6, 7],        # cred 0.5, 3 边，字典序小
    [0, 2, 6, 7],        # cred 0.5, 3 边
    [0, 1, 5, 2, 6, 7],  # cred 0.5, 5 边
]


class SmallGraphTest(unittest.TestCase):
    def setUp(self):
        self.graph = CitationGraph(SMALL_NODES, SMALL_EDGES)

    def test_full_order_matches_expected(self):
        self.assertEqual(self.graph.trace(0, 7, 20), SMALL_EXPECTED)

    def test_matches_brute_force_for_all_k(self):
        for k in range(1, 21):
            with self.subTest(k=k):
                self.assertEqual(
                    self.graph.trace(0, 7, k),
                    brute_force_trace(SMALL_NODES, SMALL_EDGES, 0, 7, k),
                )

    def test_k_is_prefix_of_full_order(self):
        for k in range(1, len(SMALL_EXPECTED) + 1):
            with self.subTest(k=k):
                self.assertEqual(self.graph.trace(0, 7, k), SMALL_EXPECTED[:k])

    def test_no_chain_returns_empty(self):
        self.assertEqual(self.graph.trace(7, 0, 5), [])


def build_long_chains():
    """两条 402 边的链：A 全强度1（cred 1e-402），B 首边强度2（cred 2e-402）。

    两者真实可信度均 < 1e-400 且不相等；节点 ID 字典序 A < B，
    但正确排名 B 在 A 前。
    """
    edges = []
    prev = 0
    for n in range(1000, 1401):  # 链 A：0 -> 1000..1400 -> 1，全强度 1
        edges.append((prev, n, 1))
        prev = n
    edges.append((prev, 1, 1))
    edges.append((0, 2000, 2))   # 链 B：0 -> 2000..2400 -> 1，首边强度 2
    prev = 2000
    for n in range(2001, 2401):
        edges.append((prev, n, 1))
        prev = n
    edges.append((prev, 1, 1))
    nodes = [0, 1] + list(range(1000, 1401)) + list(range(2000, 2401))
    return nodes, edges


class LongChainsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nodes, edges = build_long_chains()
        cls.graph = CitationGraph(nodes, edges)
        cls.path_a = [0] + list(range(1000, 1401)) + [1]
        cls.path_b = [0] + list(range(2000, 2401)) + [1]
        cls.expected = [cls.path_b, cls.path_a]  # 2e-402 优于 1e-402

    def test_ranking_opposes_lexicographic(self):
        self.assertLess(self.path_a, self.path_b)  # 字典序 A 在前
        self.assertEqual(self.graph.trace(0, 1, 20), self.expected)

    def test_k_selects_prefix_of_same_order(self):
        full = self.graph.trace(0, 1, 20)
        for k in (1, 2):
            with self.subTest(k=k):
                self.assertEqual(self.graph.trace(0, 1, k), full[:k])


DEAD_BASE = 1000
DEAD_COUNT = 50000 - 241  # 49759 个死角节点


def build_scale_graph():
    """80 个菱形串联（2^80 条等分链）+ 含环死角区域，共 50000 节点、200000 边。"""
    edges = []
    for i in range(80):  # 菱形 i：d_i -> {u_i, v_i} -> d_{i+1}，全强度 10
        d, dn = i, i + 1
        u, v = 100 + 2 * i, 101 + 2 * i
        edges += [(d, u, 10), (u, dn, 10), (d, v, 10), (v, dn, 10)]
    edges.append((0, DEAD_BASE, 10))  # 起点可达死角区域
    dead_edges = set()
    for j in range(DEAD_COUNT):  # 死角内部含环，且无法到达主链与终点
        for t in ((j + 1) % DEAD_COUNT, (j * 7 + 3) % DEAD_COUNT,
                  (j * 13 + 5) % DEAD_COUNT, (j * 31 + 7) % DEAD_COUNT):
            if t != j:
                dead_edges.add((DEAD_BASE + j, DEAD_BASE + t))
    need = 200000 - (320 + 1) - len(dead_edges)
    j = 0
    while need > 0:  # 补足到恰好 200000 条边
        u = DEAD_BASE + (j % DEAD_COUNT)
        v = DEAD_BASE + ((j * 3 + 11) % DEAD_COUNT)
        if u != v and (u, v) not in dead_edges:
            dead_edges.add((u, v))
            need -= 1
        j += 1
    edges += [(u, v, (n % 10) + 1) for n, (u, v) in enumerate(sorted(dead_edges))]
    nodes = (list(range(81)) + [100 + 2 * i + o for i in range(80) for o in (0, 1)]
             + [DEAD_BASE + j for j in range(DEAD_COUNT)])
    assert len(nodes) == 50000 and len(edges) == 200000
    return nodes, edges


def expected_scale_top20():
    """2^80 条链可信度均为 1、边数均 160，排名即节点序列字典序。

    第 r 名（0 起）在菱形 i 处取 v_i 当且仅当 r 的第 79-i 位为 1。
    """
    paths = []
    for r in range(20):
        path = [0]
        for i in range(80):
            bit = (r >> (79 - i)) & 1
            path.append(100 + 2 * i + bit)
            path.append(i + 1)
        paths.append(path)
    return paths


class ScaleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nodes, edges = build_scale_graph()
        cls.graph = CitationGraph(nodes, edges)  # 构图不计入查询耗时

    def test_top20_exact_order_within_limits(self):
        start = time.perf_counter()
        result = self.graph.trace(0, 80, 20)
        elapsed = time.perf_counter() - start
        self.assertEqual(result, expected_scale_top20())
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = rss if sys.platform == "darwin" else rss * 1024
        print(f"\n[scale] 查询耗时 {elapsed:.3f}s，进程峰值内存 "
              f"{peak_bytes / 2**20:.1f} MiB")
        self.assertLess(elapsed, 20.0)
        self.assertLess(peak_bytes, 512 * 2**20)

    def test_k10_is_prefix_of_k20(self):
        self.assertEqual(self.graph.trace(0, 80, 10), self.graph.trace(0, 80, 20)[:10])


class FuzzTest(unittest.TestCase):
    """随机小图上与穷举对照全量比对，覆盖各种 k。"""

    def test_random_small_graphs(self):
        rng = random.Random(20261004)
        for case in range(60):
            n = rng.randint(2, 9)
            nodes = list(range(n))
            edges = []
            for u in range(n):
                for v in range(n):
                    if u != v and rng.random() < 0.3:
                        edges.append((u, v, rng.randint(1, 10)))
            source, target = 0, n - 1
            graph = CitationGraph(nodes, edges)
            for k in (1, 3, 7, 20):
                with self.subTest(case=case, k=k):
                    self.assertEqual(
                        graph.trace(source, target, k),
                        brute_force_trace(nodes, edges, source, target, k),
                    )


if __name__ == "__main__":
    unittest.main()
