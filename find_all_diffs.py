from pathlib import Path
from solve_image import load_model, process_sudoku_image
from integration.pipeline import SudokuPipeline
from eval.metrics import format_grid

model = load_model(Path("nn/digit_cnn.pth"))
grid, proba_matrix, _ = process_sudoku_image("temp_upload.png", model)

# Count how many differences between decoded grid and true_grid
from check_true_grid import true_grid
diffs = []
for idx in range(81):
    r, c = divmod(idx, 9)
    if grid[idx] != true_grid[idx]:
        diffs.append((r, c, grid[idx], true_grid[idx]))

print(f"Total differences between CNN reading and TRUE clues ({len(diffs)}):")
for r, c, read_d, true_d in diffs:
    print(f"  Cell ({r}, {c}): Read {read_d} vs True {true_d}")
