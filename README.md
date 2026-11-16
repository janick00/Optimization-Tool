# Optimization Solver

Modular Python desktop application for working through optimization topics. The
application includes **Problem Formulation** and **Unconstrained Optimization**.
Problem Formulation accepts a linear
program and converts it between the four representations from Chapter 3.1:
General Form, Canonical Form, Standard Form, and Inequality Form. The converted
problem and the mathematical transformation steps are shown in the interface.
Further optimization solvers and visualizations will be added topic by topic.

## Requirements

- Python 3.10 or newer
- A virtual environment
- PySide6
- NumPy, SciPy, SymPy, and Matplotlib are listed as prepared project dependencies

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

The first topic is selected automatically. The navigation remains visible on
the left, and selecting a topic changes the page in the main content area.

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
│   └── pages/               # Shared page widgets
├── topics/
│   └── <topic>/
│       ├── page.py          # Topic-specific GUI page
│       ├── solver.py        # Topic-specific calculation layer
│       └── visualization.py # Topic-specific visualization layer
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

To add a topic, create its topic package and page, solver, and visualization
modules, then add one `TopicDefinition` to `gui/topic_registry.py`. The main
window creates the navigation button and stacked page automatically.

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

## Unconstrained Optimization

The **Unconstrained Opt.** page lets you enter a function directly in
SymPy/Python syntax and computes its analytic gradients and Hessian with SymPy.
It supports:

- Gradient descent with successive halving
- Gradient descent with direct parabola fitting from the chosen beta start value
- Newton's method
- Broyden's method

Each run shows the function trajectory, iteration table, line-search trials,
derivatives, and the final result. Enter the variable names and starting point
next to the formula, for example `(x1 - 2)**4 + (x1 - 2*x2)**2` with variables
`x1, x2` and starting point `0, 0`.

