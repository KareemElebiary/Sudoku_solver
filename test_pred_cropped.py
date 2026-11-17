from PIL import Image
import numpy as np
import torch
from nn.model import DigitCNN

model = DigitCNN()
model.load_state_dict(torch.load("nn/digit_cnn.pth", map_location="cpu"))
model.eval()

img = Image.open('temp_upload.png').convert("L")
cropped = img.crop((14, 1, 1065, 1052))
w, h = cropped.size
cell_w = w / 9.0
cell_h = h / 9.0

grid = []
for r in range(9):
    row = []
    for c in range(9):
        x0 = int(c * cell_w + cell_w * 0.18)
        y0 = int(r * cell_h + cell_h * 0.18)
        x1 = int((c + 1) * cell_w - cell_w * 0.18)
        y1 = int((r + 1) * cell_h - cell_h * 0.18)
        patch = cropped.crop((x0, y0, x1, y1)).resize((28, 28), Image.Resampling.BILINEAR)
        arr = np.array(patch, dtype=np.float32)
        arr_inv = 255.0 - arr
        
        if arr_inv.max() < 50 or arr_inv.std() < 15:
            row.append(0)
        else:
            t = torch.tensor(arr_inv, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            probs = model.predict_proba(t)[0]
            pred = int(np.argmax(probs)) + 1
            conf = probs.max()
            row.append(pred)
    grid.append(row)

for r in grid:
    print(r)
