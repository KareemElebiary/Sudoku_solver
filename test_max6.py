import sys
from pathlib import Path
ROOT_DIR = Path(".").resolve()
sys.path.insert(0, str(ROOT_DIR))

from solve_image import load_model, process_sudoku_image
from integration.pipeline import SudokuPipeline
from eval.metrics import format_grid

model = load_model(ROOT_DIR / "nn" / "digit_cnn.pth")
grid, proba_matrix, _ = process_sudoku_image(ROOT_DIR / "temp_upload.png", model)

print("Decoded grid givens count:", sum(1 for d in grid if d != 0))

pipeline = SudokuPipeline(
    nn_model=model,
    solver_mode="ac3",
    search_mode="astar",
    max_corrections=6,
    use_nn=True
)

res = pipeline.run_from_grid(grid, proba_matrix)
print("Pipeline solvable:", res.solvable)
print("Corrections count:", len(res.correction_log))
for c in res.correction_log:
    print(f"  Cell ({c.row}, {c.col}): {c.read_digit} -> {c.new_digit} (cost: {c.cost:.4f})")
if res.solvable:
    solved_grid = [res.solution.get(i, 0) for i in range(81)]
    print("\nSolved grid:")
    print(format_grid(solved_grid))
