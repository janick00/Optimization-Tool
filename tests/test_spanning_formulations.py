from fractions import Fraction
from itertools import combinations, product
from unittest import TestCase

from topics.integer_programming.spanning_formulations import (
    LECTURE_GRAPH, LECTURE_PARTITION, TRIANGLE, build_connector_models, choose_partition,
    polytope_three, set_partitions, solve_connector_lp, verify_integer_optimum,
)
from topics.network_optimization.models import Graph


def make_graph(size, edges):
    matrix = [[None] * size for _ in range(size)]
    for i in range(size):
        matrix[i][i] = Fraction(0)
    for a, b, weight in edges:
        matrix[a][b] = matrix[b][a] = Fraction(weight)
    return Graph(tuple(str(i + 1) for i in range(size)), tuple(map(tuple, matrix)))


class SpanningFormulationsTests(TestCase):
    def test_triangle_excludes_the_fractional_lecture_vertex(self):
        models = build_connector_models(make_graph(3, TRIANGLE))
        weak, strong = solve_connector_lp(models), solve_connector_lp(models, strong=True)
        self.assertEqual(weak.point, (Fraction(1, 2),) * 3)
        self.assertEqual(weak.bound, 8)
        self.assertEqual(strong.point, (1, 0, 1))
        self.assertEqual(strong.bound, 9)
        row = choose_partition(models, weak)
        self.assertEqual((row.indices, row.rhs, row.activity(weak.point)), ((0, 1, 2), 2, Fraction(3, 2)))
        optimum = verify_integer_optimum(models)
        self.assertEqual(optimum.connector_cost, strong.bound)
        self.assertEqual(optimum.tree_cost, strong.bound)

    def test_triangle_polytopes_contain_all_four_binary_connectors(self):
        models = build_connector_models(make_graph(3, TRIANGLE))
        weak, strong = polytope_three(models), polytope_three(models, strong=True)
        expected = {(1, 1, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)}
        self.assertEqual(set(strong.vertices), expected)
        self.assertEqual(set(weak.vertices) - expected, {(Fraction(1, 2),) * 3})
        self.assertTrue(weak.faces and strong.faces)

    def test_complete_cut_and_partition_rows_retain_the_same_binary_points(self):
        edges = [(u, v, u + v + 1) for u, v in combinations(range(4), 2)]
        models = build_connector_models(make_graph(4, edges))
        self.assertEqual((len(models.cuts), len(models.partition_rows)), (7, 14))
        for point in product((0, 1), repeat=6):
            satisfies_cuts = all(row.activity(point) >= row.rhs for row in models.cuts)
            satisfies_partitions = all(row.activity(point) >= row.rhs for row in models.partition_rows)
            self.assertEqual(satisfies_cuts, satisfies_partitions)
        weak = solve_connector_lp(models)
        strong = solve_connector_lp(models, strong=True)
        optimum = verify_integer_optimum(models)
        self.assertLessEqual(weak.bound, strong.bound)
        self.assertEqual(strong.bound, optimum.connector_cost)

    def test_lecture_graph_uses_the_partition_from_the_slide(self):
        models = build_connector_models(make_graph(9, LECTURE_GRAPH), illustration=LECTURE_PARTITION)
        self.assertTrue(models.cuts_complete)
        self.assertFalse(models.partitions_complete)
        row = models.additions[0]
        self.assertEqual(row.rhs, 2)
        self.assertEqual({models.graph.edge_name(models.edges[i]).replace(" ", "") for i in row.indices},
                         {"1–2", "1–5", "4–9", "6–9", "6–7", "5–7", "8–9"})
        weak = solve_connector_lp(models)
        strong = solve_connector_lp(models, strong=True)
        optimum = verify_integer_optimum(models)
        self.assertLessEqual(weak.bound, strong.bound)
        self.assertLessEqual(strong.bound, optimum.connector_cost)
        self.assertEqual(optimum.connector_cost, optimum.tree_cost)

    def test_disconnected_and_singleton_inputs(self):
        disconnected = build_connector_models(make_graph(3, ((0, 1, 1),)))
        self.assertEqual(solve_connector_lp(disconnected).status, "infeasible")
        self.assertIsNone(verify_integer_optimum(disconnected))
        self.assertTrue(any(not row.indices for row in disconnected.cuts))
        singleton = build_connector_models(make_graph(1, ()))
        self.assertEqual(solve_connector_lp(singleton).bound, 0)
        self.assertEqual(verify_integer_optimum(singleton).tree, frozenset())

    def test_negative_cycle_differs_from_a_minimum_spanning_tree(self):
        models = build_connector_models(make_graph(3, ((0, 1, -3), (0, 2, -2), (1, 2, -1))))
        optimum = verify_integer_optimum(models)
        self.assertEqual((optimum.connector_cost, optimum.tree_cost), (-6, -5))
        self.assertEqual(solve_connector_lp(models, strong=True).bound, -6)

    def test_large_graph_returns_valid_lower_bounds_when_exact_lp_is_skipped(self):
        graph = make_graph(8, [(u, v, u + v + 1) for u, v in combinations(range(8), 2)])
        models = build_connector_models(graph)
        weak = solve_connector_lp(models)
        strong = solve_connector_lp(models, strong=True, weak_bound=weak.bound)
        self.assertEqual((weak.status, strong.status), ("bound_only", "bound_only"))
        optimum = verify_integer_optimum(models)
        self.assertLessEqual(weak.bound, strong.bound)
        self.assertLessEqual(strong.bound, optimum.connector_cost)

    def test_partition_enumeration_has_bell_counts_and_no_permutations(self):
        for size, count in enumerate((1, 1, 2, 5, 15, 52)):
            partitions = tuple(set_partitions(size))
            self.assertEqual(len(set(partitions)), count)
            self.assertEqual(len(partitions), count)
