from PIL import Image
import numpy as np

img = Image.open('temp_upload.png').convert("L")
arr = np.array(img)
print("Image shape:", arr.shape)
print("Min pixel in entire image:", arr.min(), "Max:", arr.max(), "Mean:", arr.mean())

# Find coordinates of all pixels that are dark (e.g. < 200)
dark_ys, dark_xs = np.where(arr < 200)
print(f"Dark pixels range: X: [{dark_xs.min()}, {dark_xs.max()}], Y: [{dark_ys.min()}, {dark_ys.max()}]")
