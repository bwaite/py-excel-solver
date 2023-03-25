"""
Simplified interface to the scipy linprog API which allows it to be used
in a similar way to Excel's Solver
"""
import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog, OptimizeResult
from enum import Enum, StrEnum

# from tabulate import tabulate


class ProblemType(Enum):
    """Type of optimization to perform (maximize or minimize)."""

    MAX = 1
    MIN = 2


class ConstraintSign(StrEnum):
    """Type of contstraints to use."""

    GREATER_OR_EQUAL = ">="
    LESS_OR_EQUAL = "<="
    EQUAL = "="


ALPHABET = "abcdefghijklmnopqrstuvwxyz"


class Solver:
    objective_function: np.ndarray
    constraints_left: np.ndarray
    constraints_right: np.ndarray
    constraints_signs: npt.NDArray[ConstraintSign]
    problem_type: ProblemType

    def __init__(
        self,
        objective_function: np.ndarray,
        constraints_left: np.ndarray,
        constraints_right: np.ndarray,
        constraints_signs: npt.NDArray[ConstraintSign],
        problem_type: ProblemType,
    ):
        if not len(objective_function) == len(constraints_left[0]):
            raise ValueError(
                f"Your objective function has {len(objective_function)} coefficients, but you passed {len(constraints_left[0])} coefficients in your constraints matrix (c_l)."
            )

        if not len(constraints_left) == len(constraints_right) == len(constraints_signs):
            raise ValueError("The lengths of your c_l, c_r, and signs arrays must be equal.")

        self.objective_function = objective_function
        self.constraints_left = constraints_left
        self.constraints_right = constraints_right
        self.constraints_signs = constraints_signs
        self.problem_type = problem_type

    def solve(
        self,
        make_unconstrained_non_negative: bool = True,
        minimum_for_all: int | float = None,
        maximum_for_all: int | float = None,
        bounds: np.ndarray | None = None,
        method: str = "highs",
    ):
        self.print_objective_function(self.objective_function, self.problem_type)

        if not bounds:
            # Default bounds array: a 2d array with len(obj) # of rows, where each row is [0, None]
            # Excel translation: by default, enable 'Make Unconstrained Vars Non-Negative'
            bounds = np.repeat([[0, None]], len(self.objective_function), axis=0)

            # The "[:,x]" used below sets column x of each row to the given scalar
            if not make_unconstrained_non_negative:
                bounds[:, 0] = None

            if minimum_for_all:
                bounds[:, 0] = minimum_for_all

            if maximum_for_all:
                bounds[:, 1] = maximum_for_all

            # Convert bounds 2d array to list of tuples
            bounds = [tuple(x) for x in bounds]

        # Reverse coefficient +/- sign for maximization problem
        if self.problem_type == ProblemType.MAX:
            self.objective_function *= -1

        # Reverse constraint +/- sign where inequality is ">="
        self.constraints_right[self.constraints_signs == ConstraintSign.GREATER_OR_EQUAL] *= -1
        self.constraints_left[self.constraints_signs == ConstraintSign.GREATER_OR_EQUAL] *= -1

        # Delete all equalities from c_l and c_r. Move them to their own arrays
        if ConstraintSign.EQUAL in self.constraints_signs:
            equalities_left = self.constraints_left[self.constraints_signs == ConstraintSign.EQUAL]
            equalities_right = self.constraints_right[self.constraints_signs == ConstraintSign.EQUAL]
            self.constraints_left = np.delete(
                self.constraints_left, np.where(self.constraints_signs == ConstraintSign.EQUAL), 0
            )
            self.constraints_right = np.delete(
                self.constraints_right, np.where(self.constraints_signs == ConstraintSign.EQUAL), 0
            )
        else:
            equalities_left = None
            equalities_right = None

        # Solve linear programming problem
        solution = linprog(
            self.objective_function,
            A_ub=self.constraints_left,
            b_ub=self.constraints_right,
            A_eq=equalities_left,
            b_eq=equalities_right,
            bounds=bounds,
            method=method,
        )

        return solution

    def print_objective_function(self, obj: np.ndarray, problem_type: ProblemType) -> None:
        """ """
        print("------------------------------------------------------")

        if problem_type == ProblemType.MAX:
            print("MAXIMIZE: z = ", end="")
        elif problem_type == ProblemType.MIN:
            print("MINIMIZE: z = ", end="")

        for num, letter in zip(obj, ALPHABET):
            num = round(num, ndigits=2)
            if num < 0:
                display_num = f"- {abs(num):g}"
            else:
                display_num = f"+ {num:g}"
            print(f" {display_num}{letter}", end="")

        print("\n------------------------------------------------------")

    def print_results(self, solution: OptimizeResult) -> None:
        """
        Print results (TODO use tabulate to format table)
        """
        np.set_printoptions(precision=2, suppress=True)
        optimal = round(solution.fun, ndigits=2)
        if optimal % 1 == 0:
            optimal = int(solution.fun)
        # For maximization problem, reverse the sign of optimal value
        optimal = optimal * -1 if self.problem_type == ProblemType.MAX else optimal
        print(f"OPTIMAL VALUE:  {optimal}")
        print("------------------------------------------------------")
        print("QUANTITIES:")

        for num, letter in zip(solution.x, ALPHABET):
            num = round(num, ndigits=5)
            print(f"{letter}:  {num:g}")

        print("------------------------------------------------------")
        # print(f"Iterations: {solution.nit}")
        print(solution.message)
