from fractions import Fraction
from unittest import TestCase

from topics.integer_programming.hull_examples import ILP_EXAMPLES
from topics.integer_programming.hull_geometry import construct_hull, convex_hull
from topics.integer_programming.integer_hull import compute_integer_hull
from topics.integer_programming.models import Domain
from topics.linear_programming.models import LinearConstraint, LinearProgram, ObjectiveSense, VariableDomain


INTEGER = (Domain.INTEGER, Domain.INTEGER)


def vector(*entries):
    return tuple(map(Fraction, entries))


def problem(rows, objective=(0, 1), sense=ObjectiveSense.MAXIMIZE, sign=VariableDomain.NONNEGATIVE):
    return LinearProgram(
        sense, objective, tuple(LinearConstraint((a, b), "<=", end) for a, b, end in rows),
        ("x1", "x2"), (sign, sign),
    )


def example(name):
    fixture = ILP_EXAMPLES[name]
    return problem(fixture.constraints, fixture.objective, fixture.sense), fixture.domains


class ConvexHullTests(TestCase):
    def test_stepwise_hull_discards_duplicates_and_non_vertices(self):
        expected = (vector(0, 0), vector(3, 0), vector(3, 3), vector(0, 3))
        points = (vector(1, 1), expected[2], vector(1, 0), *expected, vector(2, 2))
        construction = construct_hull(points)
        self.assertEqual(construction.vertices, expected)
        self.assertEqual(construction.chains_at(0), ((), ()))
        lower, upper = construction.chains_at(len(construction.steps))
        self.assertEqual(lower[:-1] + upper[:-1], expected)
        removals = [step for step in construction.steps if step.action == "remove"]
        self.assertTrue(any(step.point == vector(1, 1) and step.turn < 0 for step in removals))
        self.assertTrue(any(step.point == vector(1, 0) and step.turn == 0 for step in removals))

    def test_collinear_points_and_small_inputs(self):
        a, b, c = vector("-3/7", "-6/7"), vector(1, 2), vector(2, 4)
        self.assertEqual(convex_hull([b, a, c, c]), (a, c))
        self.assertEqual(convex_hull([a, a]), (a,))
        self.assertEqual(convex_hull([c, a]), (a, c))
        self.assertEqual(convex_hull([]), ())


class IntegerHullTests(TestCase):
    def test_lecture_triangle_lp_and_integer_hull(self):
        model, domains = example("Lecture: Gomory–Chvátal (p. 29)")
        result = compute_integer_hull(model, domains)
        self.assertEqual(result.status, "complete")
        self.assertEqual(set(result.region.vertices), {
            vector("1/2", 1), vector("5/2", 1), vector("3/2", 3),
        })
        self.assertEqual(result.points, (vector(1, 1), vector(1, 2), vector(2, 1), vector(2, 2)))
        self.assertEqual(set(result.vertices), set(result.points))
        self.assertEqual(result.lp.optimum.point, vector("3/2", 3))
        self.assertEqual(result.lp.optimum.value, 3)
        self.assertEqual(result.discrete_optimum.value, 2)
        self.assertEqual(result.hull_optimum.value, 2)
        self.assertEqual(result.gap, 1)

    def test_minimization_uses_the_lower_lp_bound(self):
        model, domains = example("Fractional minimum")
        result = compute_integer_hull(model, domains)
        self.assertEqual(result.lp.optimum.value, Fraction(3, 2))
        self.assertEqual(result.discrete_optimum.value, 2)
        self.assertEqual(result.hull_optimum.value, 2)
        self.assertEqual(result.gap, Fraction(1, 2))

    def test_binaries_are_bounded_by_one_and_mixed_hull_contains_entire_slices(self):
        model, domains = example("Binary and continuous")
        result = compute_integer_hull(model, domains)
        self.assertEqual(result.region.limits[0], vector(0, "1/2"))
        self.assertEqual(result.segments, ((vector(0, 0), vector(0, "3/2")),))
        self.assertEqual(result.vertices, result.segments[0])
        self.assertEqual(result.discrete_optimum.point, vector(0, "3/2"))
        binary_region = compute_integer_hull(problem(((0, 1, 2),)), domains)
        self.assertEqual(binary_region.region.limits[0], vector(0, 1))

    def test_continuous_variables_yield_the_original_hull(self):
        model, _ = example("Fractional packing")
        result = compute_integer_hull(model, (Domain.CONTINUOUS, Domain.CONTINUOUS))
        self.assertEqual(result.vertices, result.region.vertices)
        self.assertEqual(result.hull_optimum.value, result.lp.optimum.value)
        self.assertEqual(result.gap, 0)

    def test_limits_never_produce_a_partial_hull_or_gap(self):
        model, domains = example("Lecture: Gomory–Chvátal (p. 29)")
        for expected, setting in (("search_truncated", {"search_radius": 1}), ("point_limit", {"max_points": 3})):
            with self.subTest(setting=setting):
                result = compute_integer_hull(model, domains, **setting)
                self.assertEqual(result.status, expected)
                self.assertEqual(result.vertices, ())
                self.assertEqual(result.points, ())
                self.assertIsNone(result.discrete_optimum)
                self.assertIsNone(result.gap)
                self.assertEqual(result.lp.optimum.value, 3)
        self.assertEqual(compute_integer_hull(model, domains, max_points=4).status, "complete")

    def test_empty_region_and_empty_integer_set_are_distinct(self):
        result = compute_integer_hull(problem(((1, 0, -1),)), INTEGER)
        self.assertEqual(result.status, "empty_lp")
        self.assertEqual(result.lp.status, "infeasible")
        single_fractional = problem(((2, 0, 1), (-2, 0, -1), (0, 1, 0)))
        result = compute_integer_hull(single_fractional, INTEGER)
        self.assertEqual(result.region.vertices, (vector("1/2", 0),))
        self.assertEqual(result.status, "empty_discrete")
        diagonal = problem(((2, -2, 1), (-2, 2, -1), (1, 0, 2), (0, 1, 2)))
        result = compute_integer_hull(diagonal, INTEGER)
        self.assertEqual(result.status, "empty_discrete")

    def test_degenerate_segment_and_singleton(self):
        for rows, vertices in (
            (((0, 1, 0), (1, 0, 2)), (vector(0, 0), vector(2, 0))),
            (((0, 1, 0), (1, 0, 0)), (vector(0, 0),)),
        ):
            with self.subTest(rows=rows):
                result = compute_integer_hull(problem(rows), INTEGER)
                self.assertEqual(result.status, "complete")
                self.assertEqual(result.vertices, vertices)

    def test_unbounded_geometry_is_separate_from_an_unbounded_objective(self):
        strip = problem(((0, 1, 2), (0, -1, -1)), sign=VariableDomain.FREE)
        result = compute_integer_hull(strip, INTEGER)
        self.assertEqual(result.status, "unbounded_region")
        self.assertEqual(result.region.vertices, ())
        self.assertEqual(result.lp.optimum.point, vector(0, 2))
        self.assertIsNone(result.discrete_optimum)
        unbounded = compute_integer_hull(problem((), (1, 0)), INTEGER)
        self.assertEqual(unbounded.lp.status, "unbounded")
