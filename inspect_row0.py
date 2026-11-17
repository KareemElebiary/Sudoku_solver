from PIL import Image
import numpy as np

img = Image.open('temp_upload.png').convert("L")
arr = 255.0 - np.array(img, dtype=np.float32)

# Look at y from 10 to 120 (Row 0)
row0 = arr[10:120, :]
for c in range(9):
    x0 = int(16 + c * (1062 - 16) / 9.0)
    x1 = int(16 + (c + 1) * (1062 - 16) / 9.0)
    patch = row0[:, x0:x1]
    print(f"Col {c} (x in [{x0}, {x1}]): max={patch.max():.1f}, mean={patch.mean():.1f}")
