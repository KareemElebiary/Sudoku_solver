from PIL import Image
import numpy as np

img = Image.open('temp_upload.png').convert("L")
w, h = img.size
cell_w = w / 9.0
cell_h = h / 9.0

# Test inner 60% of cell (0, 0) - which is blank
x0 = int(0 * cell_w + cell_w * 0.20)
y0 = int(0 * cell_h + cell_h * 0.20)
x1 = int(1 * cell_w - cell_w * 0.20)
y1 = int(1 * cell_h - cell_h * 0.20)

patch = img.crop((x0, y0, x1, y1))
arr = np.array(patch)
print("Inner 60% of blank cell (0, 0): min=", arr.min(), "max=", arr.max(), "std=", arr.std())

# Test inner 60% of cell (0, 2) - which contains digit '3'
x0 = int(2 * cell_w + cell_w * 0.20)
y0 = int(0 * cell_h + cell_h * 0.20)
x1 = int(3 * cell_w - cell_w * 0.20)
y1 = int(1 * cell_h - cell_h * 0.20)

patch3 = img.crop((x0, y0, x1, y1))
arr3 = np.array(patch3)
print("Inner 60% of cell (0, 2) with '3': min=", arr3.min(), "max=", arr3.max(), "std=", arr3.std())
