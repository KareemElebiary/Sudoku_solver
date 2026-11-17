from PIL import Image
import numpy as np
import torch
from nn.model import DigitCNN
from solve_image import process_sudoku_image

model = DigitCNN()
model.load_state_dict(torch.load("nn/digit_cnn.pth", map_location="cpu"))
model.eval()

grid, proba_matrix, img = process_sudoku_image("temp_upload.png", model)

# Cell (4, 1) contains '1'
# Cell (7, 0) contains '1'
# Cell (8, 1) contains '7'

for (r, c, name) in [(4, 1, "true 1"), (7, 0, "true 1"), (8, 1, "true 7")]:
    idx = r * 9 + c
    probs = proba_matrix[idx]
    top_indices = np.argsort(-probs)[:4]
    print(f"Cell ({r}, {c}) [{name}]:")
    for rank, d_idx in enumerate(top_indices):
        print(f"   #{rank+1}: Digit {d_idx+1} -> prob = {probs[d_idx]:.4f} (log-cost = {-np.log(max(probs[d_idx], 1e-6)):.4f})")
