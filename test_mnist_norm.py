from PIL import Image, ImageFilter
import numpy as np
import torch
from nn.model import DigitCNN
from solve_image import find_grid_bounding_box
from check_true_grid import true_grid

model = DigitCNN()
model.load_state_dict(torch.load("nn/digit_cnn.pth", map_location="cpu"))
model.eval()

img_raw = Image.open('temp_upload.png').convert("L")
x0_grid, y0_grid, x1_grid, y1_grid = find_grid_bounding_box(img_raw)
img = img_raw.crop((x0_grid, y0_grid, x1_grid, y1_grid))
img_np = 255.0 - np.array(img, dtype=np.float32)

width, height = img.size
cell_w, cell_h = width / 9.0, height / 9.0

def mnist_normalize(patch_np):
    # Threshold out noise / light background
    mask = patch_np > 70
    if mask.sum() < 25:
        return None  # empty
        
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    
    digit_crop = patch_np[y0:y1, x0:x1]
    h, w = digit_crop.shape
    if h < 5 or w < 2:
        return None
        
    # Thicken if thin (max stroke width < 3)
    digit_img = Image.fromarray(np.clip(digit_crop * (255.0 / max(digit_crop.max(), 1)), 0, 255).astype(np.uint8))
    # Slight max filter to thicken strokes like MNIST
    digit_img = digit_img.filter(ImageFilter.MaxFilter(3))
    
    # Scale to fit inside 20x20
    if h > w:
        new_h = 20
        new_w = max(1, int(round(w * 20.0 / h)))
    else:
        new_w = 20
        new_h = max(1, int(round(h * 20.0 / w)))
        
    digit_res = digit_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    digit_arr = np.array(digit_res, dtype=np.float32)
    
    # Place on 28x28 canvas centered by center-of-mass
    canvas = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = digit_arr
    
    # Center of mass adjustment
    cy = np.sum(np.arange(28)[:, None] * canvas) / max(canvas.sum(), 1)
    cx = np.sum(np.arange(28)[None, :] * canvas) / max(canvas.sum(), 1)
    shift_y = int(round(13.5 - cy))
    shift_x = int(round(13.5 - cx))
    
    # Roll / shift
    canvas = np.roll(canvas, shift_y, axis=0)
    canvas = np.roll(canvas, shift_x, axis=1)
    return canvas

correct = 0
total = 0
for idx in range(81):
    r, c = divmod(idx, 9)
    x0 = int(c * cell_w + cell_w * 0.12)
    y0 = int(r * cell_h + cell_h * 0.12)
    x1 = int((c + 1) * cell_w - cell_w * 0.12)
    y1 = int((r + 1) * cell_h - cell_h * 0.12)
    patch_np = img_np[y0:y1, x0:x1]
    
    norm = mnist_normalize(patch_np)
    if norm is None:
        pred = 0
    else:
        t = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        probs = model.predict_proba(t)[0]
        pred = int(np.argmax(probs)) + 1
        
    expected = true_grid[idx]
    if expected != 0:
        total += 1
        if pred == expected:
            correct += 1
        else:
            print(f"Mismatch at ({r}, {c}): Pred={pred}, True={expected}")

print(f"\nAccuracy on clues with MNIST normalization: {correct}/{total} ({correct/total:.1%})")
