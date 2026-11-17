from PIL import Image
import numpy as np

img = Image.open('temp_upload.png').convert("L")
w, h = img.size
cell_w, cell_h = w / 9.0, h / 9.0

def inspect_stroke(r, c, name):
    x0 = int(c * cell_w + cell_w * 0.15)
    y0 = int(r * cell_h + cell_h * 0.15)
    x1 = int((c + 1) * cell_w - cell_w * 0.15)
    y1 = int((r + 1) * cell_h - cell_h * 0.15)
    patch = 255.0 - np.array(img.crop((x0, y0, x1, y1)), dtype=np.float32)
    
    # Binary mask of stroke
    mask = patch > 100
    ys, xs = np.where(mask)
    if len(ys) == 0:
        print(f"Cell ({r}, {c}) {name}: Empty")
        return
        
    top_y = ys.min()
    # Width of stroke at top 20%
    top_mask = mask[top_y : top_y + int((ys.max() - top_y) * 0.25), :]
    top_width = top_mask.any(axis=0).sum()
    
    # Width of stroke in the middle
    mid_y = int((top_y + ys.max()) / 2)
    mid_mask = mask[mid_y - 2 : mid_y + 2, :]
    mid_width = mid_mask.any(axis=0).sum()
    
    print(f"Cell ({r}, {c}) [{name}]: top_width={top_width}, mid_width={mid_width}, ratio={top_width / max(mid_width, 1):.2f}")

inspect_stroke(4, 1, "true 1")
inspect_stroke(7, 0, "true 1")
inspect_stroke(8, 1, "true 7")
