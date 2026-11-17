from solve_image import load_model, process_sudoku_image
from integration.pipeline import SudokuPipeline
from eval.metrics import format_grid
from pathlib import Path
from check_true_grid import true_grid

model = load_model(Path("nn/digit_cnn.pth"))
grid, proba_matrix, _ = process_sudoku_image("temp_upload.png", model)

pipeline = SudokuPipeline(
    nn_model=model,
    solver_mode="ac3",
    search_mode="astar",
    max_corrections=7,
    use_nn=True
)

res = pipeline.run_from_grid(grid, proba_matrix)
print("Pipeline solvable:", res.solvable)
print("Total corrections:", len(res.correction_log))
for c in res.correction_log:
    print(f"  Cell ({c.row}, {c.col}): {c.read_digit} -> {c.new_digit} (cost: {c.cost:.4f})")

if res.solvable:
    solved = [res.solution.get(i, 0) for i in range(81)]
    print("\nRow 4:", solved[4*9 : 4*9+9])
    print("Row 7:", solved[7*9 : 7*9+9])
    print("Row 8:", solved[8*9 : 8*9+9])
