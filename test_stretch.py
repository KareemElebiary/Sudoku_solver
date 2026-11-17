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

def evaluate_cell(r, c):
    x0 = int(c * cell_w + cell_w * 0.18)
    y0 = int(r * cell_h + cell_h * 0.18)
    x1 = int((c + 1) * cell_w - cell_w * 0.18)
    y1 = int((r + 1) * cell_h - cell_h * 0.18)
    patch_np = img_np[y0:y1, x0:x1]
    
    # Method 1: Raw resize
    p1 = Image.fromarray(patch_np.astype(np.uint8)).resize((28, 28), Image.Resampling.BILINEAR)
    t1 = torch.tensor(np.array(p1), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    probs1 = model.predict_proba(t1)[0]
    
    # Method 2: Contrast stretch & threshold
    # Normalize patch so foreground stroke is full 255
    min_val, max_val = patch_np.min(), patch_np.max()
    if max_val > 50:
        stretched = np.clip((patch_np - min_val) / (max_val - min_val) * 255.0, 0, 255)
        # Apply slight morphological dilation or threshold to simulate MNIST stroke width
        stretched = np.where(stretched > 100, 255.0, stretched * 0.5)
    else:
        stretched = patch_np
        
    p2 = Image.fromarray(stretched.astype(np.uint8)).resize((28, 28), Image.Resampling.BILINEAR)
    t2 = torch.tensor(np.array(p2), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    probs2 = model.predict_proba(t2)[0]
    
    pred1 = int(np.argmax(probs1)) + 1
    pred2 = int(np.argmax(probs2)) + 1
    return pred1, probs1[pred1-1], pred2, probs2[pred2-1], probs2

print("Testing cell (1, 6) [True: 6]:")
p1, c1, p2, c2, probs = evaluate_cell(1, 6)
print(f"  Raw: pred={p1} (conf={c1:.2f}) | Stretched: pred={p2} (conf={c2:.2f})")
print("  Stretched top 3:", sorted([(d+1, probs[d]) for d in range(9)], key=lambda x: -x[1])[:3])

print("\nTesting cell (4, 1) [True: 1]:")
p1, c1, p2, c2, probs = evaluate_cell(4, 1)
print(f"  Raw: pred={p1} (conf={c1:.2f}) | Stretched: pred={p2} (conf={c2:.2f})")
print("  Stretched top 3:", sorted([(d+1, probs[d]) for d in range(9)], key=lambda x: -x[1])[:3])
