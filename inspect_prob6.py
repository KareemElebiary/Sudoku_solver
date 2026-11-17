from PIL import Image
import numpy as np
import torch
from nn.model import DigitCNN
from solve_image import find_grid_bounding_box

model = DigitCNN()
model.load_state_dict(torch.load("nn/digit_cnn.pth", map_location="cpu"))
model.eval()

img_raw = Image.open('temp_upload.png').convert("L")
x0_grid, y0_grid, x1_grid, y1_grid = find_grid_bounding_box(img_raw)
img = img_raw.crop((x0_grid, y0_grid, x1_grid, y1_grid))
img_np = 255.0 - np.array(img, dtype=np.float32)

width, height = img.size
cell_w = width / 9.0
cell_h = height / 9.0

# Cell (1, 6) contains '6'
r, c = 1, 6
x0 = int(c * cell_w + cell_w * 0.18)
y0 = int(r * cell_h + cell_h * 0.18)
x1 = int((c + 1) * cell_w - cell_w * 0.18)
y1 = int((r + 1) * cell_h - cell_h * 0.18)
patch_np = img_np[y0:y1, x0:x1]
patch_img = Image.fromarray(patch_np.astype(np.uint8)).resize((28, 28), Image.Resampling.BILINEAR)
tensor_input = torch.tensor(np.array(patch_img), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

probs = model.predict_proba(tensor_input)[0]
print("Cell (1, 6) true is 6. Predicted probabilities:")
for d in range(1, 10):
    print(f"  Digit {d}: {probs[d-1]:.4f}")
