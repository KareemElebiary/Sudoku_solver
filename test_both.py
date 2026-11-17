from PIL import Image
import numpy as np
import torch
from nn.model import DigitCNN
from test_mnist_norm import mnist_normalize

model = DigitCNN()
model.load_state_dict(torch.load("nn/digit_cnn.pth", map_location="cpu"))
model.eval()

img = Image.open('data/sample_puzzles/sample_puzzle_1.png').convert("L")
img_np = np.array(img, dtype=np.float32)

correct = 0
for r in range(9):
    for c in range(9):
        patch = img_np[r*28:(r+1)*28, c*28:(c+1)*28]
        if patch.max() < 40 or patch.var() < 25:
            pred = 0
        else:
            t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            probs = model.predict_proba(t)[0]
            pred = int(np.argmax(probs)) + 1
        if pred != 0:
            correct += 1

print(f"Sample puzzle detected {correct} clues cleanly.")
