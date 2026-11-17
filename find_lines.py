from PIL import Image
import numpy as np

img = Image.open('temp_upload.png').convert("L")
arr = 255.0 - np.array(img, dtype=np.float32)

# Grid lines are long continuous dark lines (now bright lines)
# Horizontal lines have high sum along rows
# Vertical lines have high sum along columns
row_sums = arr.sum(axis=1)
col_sums = arr.sum(axis=0)

import matplotlib.pyplot as plt
# Find peaks in row_sums and col_sums
print("Image dimensions (H, W):", arr.shape)
print("Top 15 row sum peaks:")
peaks_r = np.argsort(row_sums)[-15:]
print(sorted(peaks_r))

print("Top 15 col sum peaks:")
peaks_c = np.argsort(col_sums)[-15:]
print(sorted(peaks_c))
