from csp.sudoku_csp import SudokuCSP
from csp.ac3 import AC3Solver

# True puzzle clues:
# Row 0: . 3 . | 6 . 2 | 9 . .
# Row 1: . 2 . | 5 . 9 | 6 . 3
# Row 2: 6 4 9 | . 3 . | 2 . .
# Row 3: . 9 6 | . . 7 | . 3 2
# Row 4: 2 1 8 | . . . | . . .
# Row 5: . 5 . | 2 . . | . 9 6
# Row 6: . 6 2 | . 1 5 | . . 8
# Row 7: 1 8 . | . . . | 4 2 .
# Row 8: . 7 . | . 2 . | . 6 .

true_grid = [
    0, 3, 0,  6, 0, 2,  9, 0, 0,
    0, 2, 0,  5, 0, 9,  6, 0, 3,
    6, 4, 9,  0, 3, 0,  2, 0, 0,
    0, 9, 6,  0, 0, 7,  0, 3, 2,
    2, 1, 8,  0, 0, 0,  0, 0, 0,
    0, 5, 0,  2, 0, 0,  0, 9, 6,
    0, 6, 2,  0, 1, 5,  0, 0, 8,
    1, 8, 0,  0, 0, 0,  4, 2, 0,
    0, 7, 0,  0, 2, 0,  0, 6, 0
]

csp = SudokuCSP(true_grid)
solver = AC3Solver()
sol = solver.solve(csp)
print("Is the TRUE grid solvable by AC3?", sol is not None)
if sol:
    print("True solution row 4:", [sol[4*9 + c] for c in range(9)])
    print("True solution row 7:", [sol[7*9 + c] for c in range(9)])
    print("True solution row 8:", [sol[8*9 + c] for c in range(9)])
