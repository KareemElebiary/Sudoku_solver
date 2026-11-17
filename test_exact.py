from PIL import Image
import numpy as np
import torch
from nn.model import DigitCNN

model = DigitCNN()
model.load_state_dict(torch.load("nn/digit_cnn.pth", map_location="cpu"))
model.eval()

img = Image.open('data/sample_puzzles/sample_puzzle_1.png').convert("L")
print("Image size:", img.size)

# If image is 252x252 (standard MNIST mosaic), slice exactly 28x28 patches with 0 margin!
grid = []
for r in range(9):
    row = []
    for c in range(9):
        patch = img.crop((c * 28, r * 28, (c + 1) * 28, (r + 1) * 28))
        arr = np.array(patch, dtype=np.float32)
        if arr.max() < 40 or arr.var() < 25:
            row.append(0)
        else:
            t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            probs = model.predict_proba(t)[0]
            pred = int(np.argmax(probs)) + 1
            row.append(pred)
    grid.append(row)

print("Exact 28x28 slice result on sample_puzzle_1:")
for r in grid:
    print(r)
