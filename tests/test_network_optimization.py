"""Reference and playback checks for graph algorithms."""

from fractions import Fraction
from itertools import combinations
import unittest

from topics.network_optimization.formatting import format_step
from topics.network_optimization.models import Method, parse_graph, predecessor_path
from topics.network_optimization.solver import run_algorithm


def graph(size, edges, *, directed=False):
    matrix = [[""] * size for _ in range(size)]
    for u, v, cost in edges:
        matrix[u][v] = str(cost)
        if not directed:
            matrix[v][u] = str(cost)
    return parse_graph(tuple(chr(65 + i) for i in range(size)), tuple(map(tuple, matrix)), directed)


def component_count(size, edges):
    groups = [{i} for i in range(size)]
    for u, v in edges:
        a = next(g for g in groups if u in g)
        b = next(g for g in groups if v in g)
        if a != b:
            groups.remove(b)
            a.update(b)
    return len(groups)


def reference_distances(g, start):
    best = [None] * g.size
    stack = [(start, (start,), Fraction(0))]
    while stack:
        u, path, cost = stack.pop()
        if best[u] is None or cost < best[u]:
            best[u] = cost
        for v in g.neighbors(u):
            if v not in path:
                stack.append((v, (*path, v), cost + g.weight(g.edge(u, v))))
    return tuple(best)


class GraphAlgorithmsTests(unittest.TestCase):
    def test_parser_distinguishes_zero_fraction_and_absent_edge(self):
        g = parse_graph((" Zürich ", "Bern", "Basel"), (
            ("0", "0", "∞"), ("0", "", "3/2"), ("", "1,5", "0"),
        ), False)
        self.assertEqual(g.names, ("Zürich", "Bern", "Basel"))
        self.assertEqual(g.edges(), ((0, 1), (1, 2)))
        self.assertEqual(g.weights[0][1], 0)
        self.assertEqual(g.weights[1][2], Fraction(3, 2))
        self.assertIsNone(g.weights[0][2])

    def test_invalid_values_names_dimensions_and_undirected_asymmetry(self):
        cases = (
            (("A", "A"), (("0", ""), ("", "0")), "eindeutig"),
            (("A", ""), (("0", ""), ("", "0")), "Namen"),
            (("A", "B"), (("0", "NaN"), ("", "0")), "Zeile 1, Spalte 2"),
            (("A", "B"), (("0", "1"), ("2", "0")), "symmetrische"),
            (("A", "B"), (("0",), ("2", "0")), "quadratisch"),
            (("A",), (("2",),), "Diagonale"),
        )
        for names, weights, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                parse_graph(names, weights, False)

    def test_dfs_bfs_use_distinct_frontiers_and_bfs_finds_fewest_edges(self):
        g = graph(5, ((0, 1, 30), (0, 2, 1), (1, 3, 1), (3, 2, 1)), directed=True)
        dfs = run_algorithm(g, Method.DFS, 0, 2)
        bfs = run_algorithm(g, Method.BFS, 0, 2)
        self.assertEqual(dfs[-1].state.order, (0, 1, 3, 2))
        self.assertEqual(dfs[-1].state.path, (0, 1, 3, 2))
        self.assertEqual(bfs[-1].state.order, (0, 1, 2, 3))
        self.assertEqual(bfs[-1].state.path, (0, 2))
        self.assertEqual(bfs[1].state.frontier, (1, 2))
        self.assertEqual(dfs[1].state.frontier, (0, 1))
        self.assertEqual(bfs[0].state.order, ())
        self.assertEqual(dfs[0].state.frontier, (0,))
        self.assertIsNone(run_algorithm(g, Method.BFS, 0, 4)[-1].state.path)

    def test_dfs_finishes_each_branch_when_its_nodes_leave_the_stack(self):
        g = graph(6, ((5, 0, 1), (5, 3, 1), (0, 1, 1), (0, 2, 1), (1, 0, 1), (3, 4, 1)),
                  directed=True)
        steps = run_algorithm(g, Method.DFS, 5)
        self.assertEqual(steps[0].state.discovery_order, (5,))
        self.assertEqual(steps[0].state.finishing_order, ())
        for previous, step in zip(steps, steps[1:-1]):
            before, after = previous.state, step.state
            if len(after.frontier) < len(before.frontier):
                self.assertEqual(after.discovery_order, before.discovery_order)
                self.assertEqual(after.finishing_order, (*before.finishing_order, before.frontier[-1]))
            else:
                self.assertEqual(after.discovery_order, (*before.discovery_order, after.frontier[-1]))
                self.assertEqual(after.finishing_order, before.finishing_order)
        self.assertEqual(steps[-1].state.discovery_order, (5, 0, 1, 2, 3, 4))
        self.assertEqual(steps[-1].state.finishing_order, (1, 2, 0, 4, 3, 5))

    def test_bfs_records_discovery_at_enqueue_and_finishing_after_neighbor_scan(self):
        g = graph(6, ((5, 0, 1), (5, 3, 1), (0, 1, 1), (0, 2, 1), (1, 0, 1), (3, 4, 1)),
                  directed=True)
        steps = run_algorithm(g, Method.BFS, 5)
        self.assertEqual(steps[0].state.discovery_order, (5,))
        self.assertEqual(steps[0].state.finishing_order, ())
        self.assertEqual(steps[2].state.discovery_order, (5, 0, 3, 1, 2))
        self.assertEqual(steps[2].state.finishing_order, (5, 0))
        for previous, step in zip(steps, steps[1:-1]):
            before, after = previous.state, step.state
            self.assertEqual(after.finishing_order, (*before.finishing_order, before.frontier[0]))
            self.assertEqual(after.discovery_order, after.finishing_order + after.frontier)
        self.assertEqual(steps[-1].state.discovery_order, (5, 0, 3, 1, 2, 4))
        self.assertEqual(steps[-1].state.finishing_order, (5, 0, 3, 1, 2, 4))

    def test_spanning_methods_match_exhaustive_forests_with_negative_and_equal_weights(self):
        g = graph(7, ((0, 1, -1), (0, 2, 1), (1, 2, 1), (1, 3, 2),
                      (2, 3, 2), (4, 5, 0)))
        edges = g.edges()
        components = component_count(g.size, edges)
        count = g.size - components
        optimum = min(
            sum(g.weight(e) for e in subset)
            for subset in combinations(edges, count)
            if component_count(g.size, subset) == components
        )
        for method in (Method.KRUSKAL, Method.REVERSE_DELETE, Method.PRIM):
            with self.subTest(method=method):
                steps = run_algorithm(g, method, start=6)
                end = steps[-1]
                self.assertEqual(end.state.cost, optimum)
                self.assertEqual(end.state.components, components)
                self.assertEqual(len(end.selected), count)
                self.assertTrue(end.selected.isdisjoint(end.rejected))
                self.assertEqual(end, run_algorithm(g, method, start=6)[-1])
                if method is Method.REVERSE_DELETE:
                    self.assertEqual(steps[0].selected, frozenset(edges))
                    self.assertLessEqual(end.state.cost, 4)
                else:
                    self.assertFalse(steps[0].selected)

    def test_dijkstra_and_floyd_match_independent_path_enumeration(self):
        g = graph(6, ((0, 1, 5), (0, 2, 2), (2, 1, 0), (1, 3, 1), (2, 3, 2),
                      (3, 0, 2), (4, 3, 1)), directed=True)
        floyd = run_algorithm(g, Method.FLOYD, 0, 3)
        self.assertEqual([step.state.via for step in floyd[1:-1]], list(range(g.size)))
        self.assertTrue(any(step.state.changes for step in floyd))
        self.assertEqual(floyd[0].state.distances, g.weights)
        for start in range(g.size):
            with self.subTest(start=start):
                expected = reference_distances(g, start)
                run = run_algorithm(g, Method.DIJKSTRA, start, 3)
                self.assertEqual(run[-1].state.distances, expected)
                self.assertEqual(floyd[-1].state.distances[start], expected)
                self.assertEqual(run[0].state.fixed, ())
        finish = run_algorithm(g, Method.DIJKSTRA, 0, 3)[-1].state
        path = (0, 2, 1, 3)
        self.assertEqual(predecessor_path(finish.parents, 0, 3), path)
        self.assertEqual(floyd[-1].state.path, path)

    def test_floyd_accepts_negative_edges_and_marks_only_affected_negative_cycle_pairs(self):
        dag = graph(4, ((0, 1, 2), (1, 2, -7), (0, 2, 1), (2, 3, 1)), directed=True)
        run = run_algorithm(dag, Method.FLOYD, 0, 3)
        self.assertFalse(run[-1].state.unbounded)
        self.assertEqual(run[-1].state.distances[0], reference_distances(dag, 0))
        self.assertEqual(run[-1].state.path, (0, 1, 2, 3))
        with self.assertRaisesRegex(ValueError, "nichtnegative"):
            run_algorithm(dag, Method.DIJKSTRA)
        with self.assertRaisesRegex(ValueError, "ungerichteten"):
            run_algorithm(dag, Method.PRIM)

        cycle = graph(5, ((0, 1, 1), (1, 2, -4), (2, 1, 1), (2, 3, 2), (4, 3, 6)), directed=True)
        end = run_algorithm(cycle, Method.FLOYD, 0, 3)[-1]
        self.assertEqual(end.state.unbounded, {(i, j) for i in (0, 1, 2) for j in (1, 2, 3)})
        self.assertIsNone(end.state.path)
        self.assertIn("−∞", format_step(cycle, end, 6, 7))
        unaffected = run_algorithm(cycle, Method.FLOYD, 4, 3)[-1].state
        self.assertEqual(unaffected.path, (4, 3))
        self.assertEqual(unaffected.distances[4][3], 6)

    def test_one_node_and_disconnected_empty_graph(self):
        for size in (1, 3):
            g = graph(size, ())
            for method in Method:
                with self.subTest(size=size, method=method):
                    end = run_algorithm(g, method)[-1]
                    self.assertFalse(end.selected)
                    if method.requires_undirected:
                        self.assertEqual(end.state.cost, 0)
                        self.assertEqual(end.state.components, size)
