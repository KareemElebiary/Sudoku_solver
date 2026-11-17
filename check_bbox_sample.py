from PIL import Image
import numpy as np
from solve_image import find_grid_bounding_box

img = Image.open('data/sample_puzzles/sample_puzzle_1.png').convert("L")
print("Size:", img.size)
print("find_grid_bounding_box returned:", find_grid_bounding_box(img))
