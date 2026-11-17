from PIL import Image
import numpy as np

img = Image.open('temp_upload.png').convert("L")
cropped = img.crop((14, 1, 1065, 1052))
w, h = cropped.size
cell_w = w / 9.0
cell_h = h / 9.0

for r in range(3):
    for c in range(9):
        # Center crop (20% to 80% to avoid grid lines completely)
        x0 = int(c * cell_w + cell_w * 0.20)
        y0 = int(r * cell_h + cell_h * 0.20)
        x1 = int((c + 1) * cell_w - cell_w * 0.20)
        y1 = int((r + 1) * cell_h - cell_h * 0.20)
        patch = np.array(cropped.crop((x0, y0, x1, y1)))
        
        # Invert if white background
        patch_inv = 255.0 - patch
        print(f"Cell ({r},{c}): max_intensity={patch_inv.max():.1f}, std={patch_inv.std():.1f}")
