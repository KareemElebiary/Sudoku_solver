from PIL import Image
import numpy as np

img = Image.open('temp_upload.png')
print("Image size:", img.size, "bands:", img.getbands())

# Let's inspect the top-left 1/9th
w, h = img.size
cell_w = w / 9.0
cell_h = h / 9.0

# Crop cell (0, 0)
c00 = img.crop((0, 0, int(cell_w), int(cell_h)))
arr00 = np.array(c00.convert("L"))
print("Cell (0, 0) min, max, mean, var:", arr00.min(), arr00.max(), arr00.mean(), arr00.var())
