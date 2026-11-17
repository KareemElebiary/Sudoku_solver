import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from PIL import Image

from data.make_data import (
    load_mnist,
    generate_puzzle,
    build_mosaic,
    inject_hard_digits,
    SEED,
)
from nn.model import DigitCNN

def make_samples():
    print("Loading test data...")
    _, _, _, _, test_x, test_y = load_mnist()
    
    # 1. Standard clean handwritten puzzle (30 clues)
    clue_grid, sol_grid = generate_puzzle(seed=42, num_remove=51) # 30 clues
    img_clean = build_mosaic(clue_grid, test_x, test_y, seed=42)
    
    out_dir = ROOT_DIR / "data" / "sample_puzzles"
    p1 = out_dir / "sample_puzzle_1.png"
    img_clean.save(p1)
    print(f"Saved: {p1}")
    
    # 2. Puzzle with misreads
    device = torch.device("cpu")
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(ROOT_DIR / "nn" / "digit_cnn.pth", map_location=device))
    
    corrupt_grid = inject_hard_digits(
        puzzle=clue_grid,
        grid_images=test_x,
        grid_labels=test_y,
        nn_model=model,
        num_inject=2,
        seed=42
    )
    # Convert negative provenance markers back to positive digit values for visual mosaic
    mosaic_grid = [[abs(d) for d in row] for row in corrupt_grid]
    img_corrupt = build_mosaic(mosaic_grid, test_x, test_y, seed=42)
    p2 = out_dir / "sample_puzzle_with_misread.png"
    img_corrupt.save(p2)
    print(f"Saved: {p2}")
    
    # Also save to Downloads for quick access
    downloads_path = Path("C:/Users/karee/Downloads/sample_sudoku_puzzle.png")
    img_clean.save(downloads_path)
    print(f"Saved clean sample to Downloads: {downloads_path}")

if __name__ == "__main__":
    make_samples()
