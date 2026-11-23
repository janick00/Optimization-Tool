"""Examples for the hull workbench in the shared ILP editor's input format."""

from dataclasses import dataclass

from topics.integer_programming.models import Domain
from topics.linear_programming.models import ObjectiveSense


POINT_EXAMPLES = {
    "Points inside a polygon": "0, 0\n1, 3\n3, 4\n5, 2\n4, 0\n1, 1\n2, 2\n3, 2\n3, 4",
    "Collinear points": "-2, -1\n0, 0\n2, 1\n4, 2\n2, 1",
}


@dataclass(frozen=True)
class ILPExample:
    objective: tuple[int, int]
    constraints: tuple[tuple[int, int, int], ...]
    domains: tuple[Domain, Domain] = (Domain.INTEGER, Domain.INTEGER)
    sense: ObjectiveSense = ObjectiveSense.MAXIMIZE


ILP_EXAMPLES = {
    # Chapter 4, page 29. Maximizing x2 visits the vertex separated on the slide.
    "Lecture: Gomory–Chvátal (p. 29)": ILPExample((0, 1), ((-2, 1, 0), (2, 1, 6), (0, -1, -1))),
    "Fractional packing": ILPExample((1, 1), ((2, 1, 7), (1, 2, 7))),
    "Binary and continuous": ILPExample((2, 1), ((2, 2, 3), (2, -1, 0)),
                                        (Domain.BINARY, Domain.CONTINUOUS)),
    "Fractional minimum": ILPExample((1, 1), ((-2, -2, -3), (1, 1, 4)),
                                    sense=ObjectiveSense.MINIMIZE),
}
