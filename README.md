# Optimization Solver

Modular Python desktop application for working through optimization topics. The
application includes **Problem Formulation**, **Simplex Method**, **Integer Linear Programming**, **Unconstrained Optimization**, and **Graphen & Netzwerke**.
Problem Formulation accepts a linear
program and converts it between the four representations from Chapter 3.1:
General Form, Canonical Form, Standard Form, and Inequality Form. The converted
problem and the mathematical transformation steps are shown in the interface.
Further optimization solvers and visualizations will be added topic by topic.

## Requirements

- Python 3.10 or newer
- A virtual environment
- PySide6, NumPy, SymPy, and Matplotlib

## Installation

### Ubuntu

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Start the application

With the virtual environment activated:

```bash
python main.py
```

The first topic is selected automatically. Select a topic in the left sidebar,
or use the menu button in the top-left corner to collapse or expand the sidebar.

## Run tests

```bash
python -m unittest discover -s tests -v
```

The GUI tests use Qt's offscreen platform, so they do not require a display.

## Architecture

```text
.
├── gui/
│   ├── main_window.py       # Main window and navigation
│   ├── styles.py            # Dark theme
│   ├── topic_registry.py    # Central topic configuration
│   └── pages/placeholder_page.py
├── topics/
│   ├── linear_programming/
│   │   ├── page.py          # Linear-programming interface
│   │   ├── models.py        # Problem and conversion data
│   │   ├── formatting.py    # Result rendering
│   │   ├── problem_input.py # Text input for a linear program
│   │   └── form_conversion.py # Variable and constraint transformations
│   └── unconstrained_optimization/
│       ├── page.py          # Qt interface
│       ├── models.py        # Settings and iteration history
│       ├── formatting.py    # Formula and result rendering
│       ├── methods/
│       │   ├── line_search.py # Successive halving and parabola fitting
│       │   ├── newton_methods.py # Newton step and Broyden update
│       │   └── aitken.py    # Fixed-point acceleration
│       ├── objective.py     # Function, gradient, and Hessian
│       ├── solver.py        # Iterations and stopping criteria
│       └── visualization.py # Function and trajectory plots
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

Each implemented topic has a `page.py` for Qt widgets, a `models.py` for data,
and a `formatting.py` for HTML results.

To add a topic, add its page and register a `TopicDefinition` in
`gui/topic_registry.py`. The main window creates the navigation button and
stacked page automatically.

## Problem Formulation input

Select the number of variables and constraints first. The interface then
creates the `x1`, `x2`, ... labels, coefficient fields, one relation dropdown
and one right-hand-side field per constraint, as well as one domain dropdown
per variable. For General Form, relations and domains are editable. For
Canonical, Standard, and Inequality Form they are derived from the selected
objective direction according to the lecture definitions.

The converted result can additionally be displayed as Expanded notation, Row
notation, Column notation, or Matrix notation.

Exercise 3 from the exercise sheet contains binary restrictions
`x_i in {0, 1}`. It is therefore an integer/combinatorial optimization model,
not a pure LP. The current converter intentionally does not relax those binary
restrictions; an integer-programming extension can be added separately.

## Simplex Method

The **Simplex Method** page uses the same linear-program input fields.
**Starting Point** defaults to **Automatic**, which finds a feasible vertex.
Under **Manual**, enter one decimal or fraction per variable and choose
**Solve**. The calculation panel reports feasibility, active rows, their rank,
the selected basis and the objective value at the start. The two-variable
plot shows the start and the simplex path or marks an invalid candidate.

## Integer Linear Programming

The **Integer Linear Programming** page uses nonnegative variables with
Integer, Binary, or Continuous domains. Choose **LP Relaxation** to inspect
the continuous subproblem. Its formula-sheet view shows the LP optimum, checks
your manually entered comparison point, and combines their bounds. Coordinate
edits update the interval and the two-variable plot immediately.

Choose **Branch and Bound** to solve and replay the search. **Solve** displays
the open root. **Next Step** and **Previous Step**
move through the LP solve, integrality check, bounds and branching or pruning.
**Show Final Result** jumps to the last recorded event. An optional feasible
**Initial solution x₀** supplies an incumbent. The node limit counts solved
LP relaxations. Select a visible node to inspect its LP, bounds and simplex
calculations at the current step. With two variables the plot shows the
feasible region and the solutions and branch limits known at that step. The
integer-point overlay appears with the optimality proof. An unresolved
relaxation or node limit leaves the search incomplete.

### Convex Hull / Cutting Planes

Select **Convex Hull / Cutting Planes** inside **Integer Linear Programming**.
Use **Convex Hull from Points** for a list of planar points, **Integer Hull**
to compare the LP relaxation with the hull of the discrete feasible set, or
**Cutting Planes** to tighten the relaxation one step at a time. Enter your
own data or use **Load Example**. The lecture example from Chapter 4, page 29,
uses an objective of maximizing x2 to visit the vertex separated by the cut
−x1 + x2 ≤ 1. Use the plot toolbar to pan or zoom and hover over a marker to
inspect exact coordinates.

- `hull_geometry.convex_hull` builds the two boundary chains using exact
  determinant signs. `construct_hull` records each point decision.
- `integer_hull.compute_integer_hull` solves the relaxation and enumerates
  feasible integer points, or the endpoints of slices for mixed problems.
  The search radius limits discrete coordinates independently of the plot
  range. Unbounded regions, truncated searches and exceeded point limits
  have no enumerated hull or exact integrality gap.
- `cutting_planes.gomory_chvatal_cut` combines active inequalities with
  nonnegative rational weights. The resulting coefficients are integral
  on discrete variables and zero on continuous variables, so rounding the
  right-hand side down preserves every feasible discrete solution. Manual
  cuts require proof over the original discrete set and must separate the
  current LP optimum. **Reset** discards the applied cuts and returns to P.

These modes use the existing Python dependencies.

### Knapsack Problem

Choose **Knapsack Problem** in the ILP method list. Edit the item weight and
value to update its value-to-weight ratio, or load the Chapter 4 example.
**Greedy LP Relaxation** plays the sorted packing, fractional item and the
first feasible binary packing. The calculation view gives the solution in
the order of the item table. **Branch and Bound** shows the same relaxation
at each node, the incumbent, and the branching or pruning decision. In
Depth-First Search the child fixing a variable to one is visited first.
Best-Bound prioritizes the highest child LP bound. The node limit counts
visits. A positive objective tolerance can stop the search with a certified
gap. Set the tolerance to zero for an exact optimality proof.

The mathematical entry points are `knapsack.greedy_relaxation` and
`knapsack.knapsack_branch_and_bound`. The lecture case gives LP bound 32.5,
a feasible binary value of 31 and a proven optimum of 31.
At visit 4 on slide 16, the displayed packing evaluates to 28
(`10 + 9 + 12/3 + 5`). The slide reports 27 for that bound. Both
values lead to pruning by dominance.

### Spanning Tree / ILP Formulations

Choose **Spanning Tree / ILP Formulations**. Enter undirected edges by their
node table numbers, or load a lecture example. The triangle uses the
original weights. The nine-node topology uses added positive example
weights. Use **Next Step** to inspect a cut or partition, compare the LP
relaxations and display the independently checked minimum spanning tree.
The **3D** toggle is available for connected inputs with three edges.

`spanning_formulations.build_connector_models` generates full cut systems
through nine nodes and full partition systems through five nodes. Larger
models use explicitly labelled selections of valid rows. The relaxations
reuse Simplex for up to 16 edge variables. Above that size, or if the solver
reaches its pivot limit, the interface displays a certified lower bound.
`verify_integer_optimum` uses binary enumeration for small graphs or
Prim for larger ones, and checks the tree cost independently with Kruskal.
For nonpositive weights the optimal binary connector may have a different
cost from the minimum spanning tree.

## Graphen & Netzwerke

Edit the node names in the first column and the weights in the adjacency
matrix beside them. The editor supports 1–20 nodes. Empty cells and **∞**
mean no edge. An off-diagonal **0** is an edge of weight zero. The diagonal
is fixed at zero. In undirected mode, changes mirror automatically. When
merging conflicting directed edges, the upper-triangle entry takes precedence.

Choose DFS, BFS, Kruskal, Reverse-Delete, Prim, Dijkstra, or Floyd-Warshall
and select **Berechnen**. Move through the saved steps with the buttons
below the graph or jump to **Endergebnis**. Nodes can be dragged, and the
graph can be panned or zoomed. The three spanning-tree algorithms return a
minimum spanning forest for disconnected graphs. Dijkstra requires
nonnegative weights. Floyd-Warshall marks pairs affected by a negative
cycle with **−∞**.

## Unconstrained Optimization

The **Unconstrained Opt.** page lets you enter a function directly in
SymPy/Python syntax and computes its analytic gradients and Hessian with SymPy.
It supports:

- Gradient descent with successive halving for a requested number of iterations
- Gradient descent with parabola fitting and an inspectable interpolation per iteration
- Newton's method
- Broyden's method

Use **Previous**, **Next**, the iteration field or an iteration number in
the table. The trajectory, calculations and convergence plots follow the
same selection. **Auto-fit step** focuses on the selected transition.
**Fit run** shows the full path and keeps the manual viewport when stepping.
Each plot has pan and zoom controls. For more than two variables the
trajectory projects onto the first two coordinates, while the contours hold
the remaining coordinates at the current point.

The method tab shows the line-search profile and each trial for **Successive
Halving**, the actual profile and its interpolant for **Parabola Fitting**,
the local quadratic model and Hessian classification for **Newton**, or
the inverse matrices and residuals for **Broyden**. Select a trial or
support point from the table or the method plot. **Convergence** shows
the objective, gradient norm and tolerance. The default two-variable
example has a quartic term and gives inspectable iterations with all four
methods. The initial β for Parabola Fitting is 0.25 and is reused in
each iteration.

**Successive Halving** defaults to 10 iterations. The first point within the
gradient tolerance is marked. The run continues until the requested length,
an exactly zero gradient, or a failed line search. Each iteration starts
from the last accepted step size. The **Backtracking** tab shows the complete
β sequence and one row per tested candidate for the current iteration.
Scroll the table for all nine columns. Click a row to highlight its
candidate in **Trajectory** and inspect its substitution and comparison.

Newton and Broyden both take the full step (β = 1). The Broyden view
shows the update actually used between iterations. The terminal row
reports the stopping reason when the run ends before another update.

**Aitken Acceleration** takes a scalar iteration function such as `cos(x)`
and starts at 1 by default. **Number of Points** includes the starting value.
The page calculates the ordinary sequence and estimates its limit from
consecutive triples. Expand a table row for the substitution and residual,
or switch between the iteration-value and residual plots.

### Use the numerical functions in your own script

`topics/unconstrained_optimization/methods` uses NumPy. It accepts numeric gradients and Hessians
from any source. This example uses the app's SymPy parser for the formula
from the MATLAB worksheet.

```python
from topics.unconstrained_optimization.methods import parabola_fitting
from topics.unconstrained_optimization.objective import parse_function

f = parse_function("(x**2 - 2*x*y + x)**2", "x, y")
x = [2, 2]
result = parabola_fitting(f.evaluate, x, f.gradient_at(x), beta=1 / 32)
if result.selected is not None:
    print(result.selected.step_size, result.selected.point, result.selected.value)
```

- `successive_halving(f, x, gradient, beta)` starts with β₀ and halves it until
  the first improvement. This follows the successive-halving formula sheet.
- `parabola_fitting(f, x, gradient, beta)` calculates f₀, f₁ and f₂, computes
  β*, and compares f₁ with f(β*) as shown on the formula sheet.
- `newton_step(gradient, hessian)` and `broyden_step(gradient, inverse_hessian)`
  return displacements to add to the current point. Use
  `broyden_update(inverse_hessian, displacement, gradient_change)` to update the
  inverse Hessian between iterations.
- `iterate_fixed_point(g, start, number_of_points=10)` returns the ordinary
  scalar sequence, its residuals and the Aitken estimates for its triples.
- `aitken_step(G, x)` computes a single componentwise estimate for vector input.

To change the parabolic formula, edit `parabola_minimum()` in
`topics/unconstrained_optimization/methods/line_search.py`.
For convergence rules or iteration limits in the app, see
`topics/unconstrained_optimization/solver.py`.

