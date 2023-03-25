"""
Tests for solver library
"""
from solver.solver import Solver, ProblemType, ConstraintSign
import numpy as np

# import pytest


def test_problem_1():
    """Minimization test"""

    solver = Solver(
        problem_type=ProblemType.MIN,
        objective_function=np.array([10, 15, 25]),
        constraints_left=np.array(
            [
                [1, 1, 1],
                [1, -2, 0],
                [0, 0, 1],
            ]
        ),
        constraints_right=np.array(
            [
                1000,
                0,
                340,
            ]
        ),
        constraints_signs=np.array(
            [
                ConstraintSign.GREATER_OR_EQUAL,
                ConstraintSign.GREATER_OR_EQUAL,
                ConstraintSign.GREATER_OR_EQUAL,
            ]
        ),
    )
    solution = solver.solve(make_unconstrained_non_negative=True)

    assert solution.fun == 15100.0
    quantities = np.array([660.0, 0.0, 340.0])

    assert np.allclose(solution.x, quantities)


def test_problem_2():
    """Maximization test"""

    solver = Solver(
        problem_type=ProblemType.MAX,
        objective_function=np.array([16, 20.5, 14]),
        constraints_left=np.array(
            [
                [4, 6, 2],
                [3, 8, 6],
                [9, 6, 4],
                [30, 40, 25],
            ]
        ),
        constraints_right=np.array(
            [
                2000,
                2000,
                1440,
                9600,
            ]
        ),
        constraints_signs=np.array(
            [
                ConstraintSign.LESS_OR_EQUAL,
                ConstraintSign.LESS_OR_EQUAL,
                ConstraintSign.LESS_OR_EQUAL,
                ConstraintSign.LESS_OR_EQUAL,
            ]
        ),
    )

    solution = solver.solve()

    assert round(solution.fun, ndigits=2) == -4960
    quantities = np.array([0.0, 160.0, 120.0])

    assert np.allclose(solution.x, quantities)


def test_problem_3():
    """Trailmix cost minimization problem

    Seems to orinate from homework in https://www.cmu.edu/tepper/programs/courses/45751.html

    https://www.chegg.com/homework-help/questions-and-answers/1-26-points-patterson-nut-company-wishes-introduce-packaged-trail-mix-new-product-five-ing-q81058093
    https://homework.study.com/explanation/you-ve-been-hired-by-a-food-service-company-to-create-a-custom-packaged-trail-mix-to-include-in-school-lunches-the-ingredients-for-the-trail-mix-will-include-raisins-grain-chocolate-chips-peanuts-and-almonds-these-cost-respectively-2-50-1.html
    """
    print("Problem #3 (Trail Mix):")
    solver = Solver(
        problem_type=ProblemType.MIN,
        objective_function=np.array([4, 5, 3, 7, 6]),
        constraints_left=np.array(
            [
                [10, 20, 10, 30, 20],
                [5, 7, 4, 9, 2],
                [1, 4, 10, 2, 1],
                [500, 450, 160, 300, 500],
            ]
        ),
        constraints_right=np.array(
            [
                16,
                10,
                15,
                600,
            ]
        ),
        constraints_signs=np.array(
            [
                ConstraintSign.GREATER_OR_EQUAL,
                ConstraintSign.GREATER_OR_EQUAL,
                ConstraintSign.GREATER_OR_EQUAL,
                ConstraintSign.GREATER_OR_EQUAL,
            ]
        ),
    )

    solution = solver.solve(minimum_for_all=0.1)
    assert round(solution.fun, ndigits=2) == 8.04
    quantities = np.array([0.44415274, 0.18090692, 1.35322196, 0.1, 0.1])
    assert np.allclose(solution.x, quantities)


def test_problem_4():
    solver = Solver(
        problem_type=ProblemType.MAX,
        objective_function=np.array([16, 20.5, 14]),
        constraints_left=np.array(
            [
                [4, 6, 2],
                [9, 6, 4],
                [30, 40, 25],
                [3, 8, 6],
            ]
        ),
        constraints_right=np.array(
            [
                2000,
                1440,
                9600,
                1984,
            ]
        ),
        constraints_signs=np.array(
            [
                ConstraintSign.LESS_OR_EQUAL,
                ConstraintSign.LESS_OR_EQUAL,
                ConstraintSign.LESS_OR_EQUAL,
                ConstraintSign.EQUAL,
            ]
        ),
    )

    solution = solver.solve()
    assert round(solution.fun) == -4952
    quantities = np.array([0.0, 176.0, 96.0])
    assert np.allclose(solution.x, quantities)