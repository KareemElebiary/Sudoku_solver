from PIL import Image
import numpy as np

img1 = np.array(Image.open('temp_upload.png').convert("L"))
img2 = np.array(Image.open('data/sample_puzzles/sample_puzzle_1.png').convert("L"))

diff = np.abs(img1.astype(int) - img2.astype(int))
print("Diff min, max, mean:", diff.min(), diff.max(), diff.mean())
print("Are they identical?", np.array_equal(img1, img2))
