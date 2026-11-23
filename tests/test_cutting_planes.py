from fractions import Fraction
from math import floor
from unittest import TestCase

from tests.test_hull_geometry import INTEGER, example, problem, vector
from topics.integer_programming.cutting_planes import CuttingSession
from topics.integer_programming.hull_geometry import dot
from topics.linear_programming.models import ObjectiveSense


class CuttingPlaneTests(TestCase):
    def test_lecture_cut_matches_slide_and_replays(self):
        model, domains = example("Lecture: Gomory–Chvátal (p. 29)")
        session = CuttingSession(model, domains)
        original = session.frame
        self.assertEqual(original.phase, "region")
        self.assertIsNone(original.lp)
        self.assertEqual(original.comparisons, ())
        session.next()
        solved = session.frame
        self.assertEqual(solved.lp.optimum.point, vector("3/2", 3))
        session.next()
        self.assertEqual(session.frame.fractional, (0,))
        session.next()
        preview = session.frame
        candidate = preview.candidate
        self.assertEqual(preview.phase, "preview")
        self.assertEqual(candidate.coefficients, vector(-1, 1))
        self.assertEqual(candidate.combined_bound, Fraction(3, 2))
        self.assertEqual(candidate.bound, 1)
        self.assertEqual(tuple((weight, row.name) for weight, row in candidate.combination),
                         ((Fraction(3, 4), "C1"), (Fraction(1, 4), "C2")))
        self.assertEqual(preview.region, original.region)
        session.next()
        self.assertEqual(session.frame.phase, "applied")
        self.assertIsNone(session.frame.lp)
        self.assertEqual(session.frame.comparisons, solved.comparisons)
        session.next()
        self.assertEqual(session.frame.lp.optimum.value, Fraction(8, 3))
        session.previous()
        session.previous()
        self.assertIs(session.frame, preview)
        session.reset()
        self.assertIs(session.frame, original)
        session.next()
        self.assertEqual(session.frame, solved)

    def test_generated_cuts_have_exact_certificates_and_preserve_feasible_points(self):
        for name in ("Lecture: Gomory–Chvátal (p. 29)", "Fractional packing",
                     "Binary and continuous", "Fractional minimum"):
            with self.subTest(name=name):
                model, domains = example(name)
                session = CuttingSession(model, domains)
                originals = (*session.reference.points, *(p for s in session.reference.segments for p in s))
                seen = 0
                previous_value = None
                for _ in range(100):
                    if not session.can_advance:
                        break
                    session.next()
                    frame = session.frame
                    if frame.phase == "preview":
                        seen += 1
                        cut = frame.candidate
                        self.assertTrue(all(weight >= 0 for weight, _ in cut.combination))
                        self.assertEqual(
                            cut.coefficients, tuple(sum(weight * row.coefficients[i]
                                                        for weight, row in cut.combination) for i in range(2)),
                        )
                        self.assertEqual(cut.combined_bound,
                                         sum(weight * row.bound for weight, row in cut.combination))
                        self.assertEqual(cut.bound, floor(cut.combined_bound))
                        self.assertGreater(dot(cut.coefficients, frame.lp.optimum.point), cut.bound)
                        self.assertTrue(all(dot(cut.coefficients, point) <= cut.bound for point in originals))
                    if frame.phase == "solved":
                        value = frame.lp.optimum.value
                        if previous_value is not None:
                            if model.objective_sense is ObjectiveSense.MAXIMIZE:
                                self.assertLessEqual(value, previous_value)
                            else:
                                self.assertGreaterEqual(value, previous_value)
                        previous_value = value
                self.assertGreaterEqual(seen, 2 if name.startswith("Lecture") else 1)
                self.assertEqual(session.frame.phase, "finished")
                self.assertEqual(session.frame.lp.optimum.value, session.reference.discrete_optimum.value)

    def test_manual_cuts_validate_against_all_feasible_points_and_replace_future(self):
        model, domains = example("Lecture: Gomory–Chvátal (p. 29)")
        session = CuttingSession(model, domains)
        for _ in range(5):
            session.next()
        session.previous()
        session.previous()
        original_preview = session.frame
        with self.assertRaisesRegex(ValueError, r"feasible point"):
            session.add_manual_cut(vector(0, 1), Fraction(1))
        self.assertIs(session.frame, original_preview)
        with self.assertRaisesRegex(ValueError, "does not cut off"):
            session.add_manual_cut(vector(0, 1), Fraction(3))
        session.add_manual_cut(vector(0, 1), Fraction(2))
        self.assertIn("complete integer hull", session.frame.candidate.validation)
        session.next()
        session.next()
        self.assertEqual(session.frame.lp.optimum.value, 2)

    def test_mixed_manual_validation_checks_segment_endpoints(self):
        model, domains = example("Binary and continuous")
        session = CuttingSession(model, domains)
        session.next()
        session.next()
        with self.assertRaisesRegex(ValueError, "3/2"):
            session.add_manual_cut(vector(0, 1), Fraction(1, 2))
        session.add_manual_cut(vector(1, 0), Fraction(0))
        self.assertEqual(session.frame.phase, "preview")

    def test_uses_active_cone_when_short_directions_do_not_separate(self):
        session = CuttingSession(problem(((5, 1, 3), (3, 1, 2)), (8, 2)), INTEGER)
        for _ in range(3):
            session.next()
        cut = session.frame.candidate
        self.assertEqual(session.frame.lp.optimum.point, vector("1/2", "1/2"))
        self.assertGreater(dot(cut.coefficients, session.frame.lp.optimum.point), cut.bound)
        self.assertTrue(all(dot(cut.coefficients, point) <= cut.bound for point in session.reference.points))

    def test_manual_cut_uses_global_proof_when_enumeration_was_limited(self):
        model, domains = example("Lecture: Gomory–Chvátal (p. 29)")
        session = CuttingSession(model, domains, max_points=1)
        session.next()
        session.next()
        session.add_manual_cut(vector(0, 1), Fraction(2))
        self.assertIn("proved by Branch and Bound", session.frame.candidate.validation)

    def test_empty_and_unbounded_relaxations_stop_without_adding_cuts(self):
        for model, status in ((problem(((0, 1, -1),)), "infeasible"), (problem(()), "unbounded")):
            with self.subTest(status=status):
                session = CuttingSession(model, INTEGER)
                session.next()
                self.assertEqual(session.frame.lp.status, status)
                session.next()
                self.assertEqual(session.frame.phase, "finished")
                self.assertFalse(session.frame.cuts)
                self.assertFalse(session.can_advance)
