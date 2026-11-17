import json
from pathlib import Path

def create_notebook():
    cells = []

    def md(text):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        }

    def code(text):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        }

    # Cell 1: Header
    cells.append(md("""# Handwritten Sudoku: Neural Network Training & Evaluation
This notebook trains, evaluates, and calibrates the perception models for the **Handwritten Sudoku Reader and Solver**:
1. **DigitCNN**: 2-layer Convolutional Neural Network with BatchNorm, Dropout, and Softmax.
2. **LogisticBaseline**: Scikit-Learn Multiclass Logistic Regression baseline.
3. **Calibration & Metrics**: Top-1 Accuracy, Per-Class Accuracy, 9x9 Confusion Matrix, and Expected Calibration Error (ECE).
4. **End-to-End Test**: Runs the trained model directly through an interactive Sudoku puzzle reading & solving step!"""))

    # Cell 2: Setup & Imports
    cells.append(md("""## 1. Setup & Imports
Install dependencies (if running in Colab / fresh environment), configure project paths, set random seeds, and detect the computing device (GPU / CPU)."""))

    # Cell 3: Pip install dependencies
    cells.append(code("""# Install dependencies directly from the notebook (use !pip or %pip)
!pip install torch torchvision numpy scipy matplotlib scikit-learn tqdm pandas pillow"""))

    # Cell 4: Imports & Path setup
    cells.append(code("""import sys
from pathlib import Path

# Add project root to sys.path (supports running from root, notebooks/, or external paths)
candidate_roots = [
    Path(".").resolve(),
    Path(".").resolve().parent,
    Path(r"c:\\Users\\karee\\.gemini\\antigravity\\scratch\\sudoku_project"),
]
PROJECT_ROOT = Path(".").resolve()
for candidate in candidate_roots:
    if (candidate / "nn" / "model.py").is_file():
        PROJECT_ROOT = candidate.resolve()
        break

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from nn.model import DigitCNN, LogisticBaseline
from data.make_data import load_mnist, SEED
from nn.evaluate import expected_calibration_error, per_class_accuracy, plot_confusion_matrix

# Seed everything
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {device}")
print(f"[Seed] Fixed seed: {SEED}")"""))

    # Cell 3: Data Loading
    cells.append(md("""## 2. Load MNIST Dataset (Digits 1–9)
We filter the MNIST dataset to digits 1–9 (excluding digit 0, as blank Sudoku cells are detected via brightness/variance rules).
The 60,000-sample training set is split deterministically into:
- **50,000** training samples
- **10,000** validation samples (4,077 digits 1-9)
- **10,000** untouched test set samples (9,020 digits 1-9)"""))

    cells.append(code("""train_x, train_y, val_x, val_y, test_x, test_y = load_mnist()
print(f"Train samples: {len(train_x)} | Val samples: {len(val_x)} | Test samples: {len(test_x)}")

# Create PyTorch DataLoaders
train_ds = TensorDataset(train_x, train_y)
val_ds = TensorDataset(val_x, val_y)
test_ds = TensorDataset(test_x, test_y)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)"""))

    # Cell 4: Visualise Sample Digits
    cells.append(md("""### Visualize Sample Digits
Display a 3x3 sample gallery of MNIST handwritten digits 1–9."""))

    cells.append(code("""fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    img = test_x[i].squeeze().numpy()
    label = test_y[i].item()
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Label: {label}", fontsize=12)
    ax.axis("off")
plt.tight_layout()
plt.show()"""))

    # Cell 5: Training DigitCNN
    cells.append(md("""## 3. Train DigitCNN
We instantiate the `DigitCNN` model and train it using CrossEntropyLoss and Adam optimizer ($lr=10^{-3}$).
You can adjust the `EPOCHS` parameter below."""))

    cells.append(code("""EPOCHS = 3   # Increase to 5 or 10 if desired

model = DigitCNN().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

print(f"{'Epoch':>5s} | {'Train Loss':>10s} {'Train Acc':>9s} | {'Val Loss':>10s} {'Val Acc':>9s}")
print("-" * 54)

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    tr_loss, tr_correct, tr_total = 0.0, 0, 0
    for images, labels in train_loader:
        images = images.to(device, dtype=torch.float32) / 255.0
        targets = (labels - 1).to(device)   # convert 1-9 to 0-8
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        tr_loss += loss.item() * len(images)
        tr_correct += (logits.argmax(dim=1) == targets).sum().item()
        tr_total += len(images)
        
    tr_loss /= tr_total
    tr_acc = tr_correct / tr_total
    
    # Validate
    model.eval()
    va_loss, va_correct, va_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, dtype=torch.float32) / 255.0
            targets = (labels - 1).to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            
            va_loss += loss.item() * len(images)
            va_correct += (logits.argmax(dim=1) == targets).sum().item()
            va_total += len(images)
            
    va_loss /= va_total
    va_acc = va_correct / va_total
    
    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)
    
    print(f"{epoch:>5d} | {tr_loss:>10.4f} {tr_acc:>9.4f} | {va_loss:>10.4f} {va_acc:>9.4f}")

# Save CNN weights
save_cnn_path = PROJECT_ROOT / "nn" / "digit_cnn.pth"
save_cnn_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), save_cnn_path)
print(f"\\nSaved CNN weights to: {save_cnn_path}")"""))

    # Cell 6: Training Curves Plot
    cells.append(md("""### Plot Training & Validation Curves"""))

    cells.append(code("""plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), history["train_loss"], label="Train Loss", marker='o')
plt.plot(range(1, EPOCHS + 1), history["val_loss"], label="Val Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("CrossEntropy Loss")
plt.title("Loss Curves")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), history["train_acc"], label="Train Acc", marker='o')
plt.plot(range(1, EPOCHS + 1), history["val_acc"], label="Val Acc", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curves")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()"""))

    # Cell 7: Train Logistic Regression Baseline
    cells.append(md("""## 4. Train Logistic Regression Baseline
We fit a multinomial logistic regression baseline on the 784-dimensional flattened pixel features."""))

    cells.append(code("""import pickle

print("Fitting Logistic Regression Baseline...")
x_tr_np = (train_x.float() / 255.0).numpy().reshape(len(train_x), -1)
y_tr_np = train_y.numpy()

baseline = LogisticBaseline(max_iter=300, random_state=42)
baseline.fit(x_tr_np, y_tr_np)

x_va_np = (val_x.float() / 255.0).numpy().reshape(len(val_x), -1)
y_va_np = val_y.numpy()
proba_va = baseline.predict_proba(x_va_np)
preds_va = proba_va.argmax(axis=1) + 1
val_acc_bl = (preds_va == y_va_np).mean()
print(f"Logistic Baseline Validation Accuracy: {val_acc_bl:.4%}")

# Save baseline model
save_bl_path = PROJECT_ROOT / "nn" / "logistic_baseline.pkl"
with open(save_bl_path, "wb") as f:
    pickle.dump(baseline, f)
print(f"Saved Baseline to: {save_bl_path}")"""))

    # Cell 8: Comprehensive Evaluation on Test Set
    cells.append(md("""## 5. Comprehensive Test Set Evaluation
We evaluate both models on the test set:
- **Top-1 Accuracy**
- **Expected Calibration Error (ECE)** (15 equal-frequency bins)
- **Per-Class Accuracy** (digits 1–9)
- **9x9 Confusion Matrix Heatmap**"""))

    cells.append(code("""# 1. Compute CNN predictions & probabilities
model.eval()
cnn_proba_list, cnn_labels_list = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images_f = images.to(device, dtype=torch.float32) / 255.0
        proba = model.predict_proba(images_f)
        cnn_proba_list.append(proba)
        cnn_labels_list.append(labels.numpy() - 1)

cnn_proba = np.concatenate(cnn_proba_list, axis=0)
cnn_labels = np.concatenate(cnn_labels_list, axis=0)
cnn_preds = cnn_proba.argmax(axis=1)

# 2. Compute Baseline predictions & probabilities
x_te_np = (test_x.float() / 255.0).numpy().reshape(len(test_x), -1)
bl_proba = baseline.predict_proba(x_te_np)
bl_labels = test_y.numpy() - 1
bl_preds = bl_proba.argmax(axis=1)

# 3. Overall Metrics & Calibration
cnn_acc = (cnn_preds == cnn_labels).mean()
bl_acc = (bl_preds == bl_labels).mean()
cnn_ece = expected_calibration_error(cnn_proba, cnn_labels, n_bins=15)
bl_ece = expected_calibration_error(bl_proba, bl_labels, n_bins=15)

print("=" * 60)
print(f"DigitCNN Top-1 Accuracy: {cnn_acc:.4%} | ECE (15 bins): {cnn_ece:.4f}")
print(f"Logistic Baseline Acc : {bl_acc:.4%} | ECE (15 bins): {bl_ece:.4f}")
print("=" * 60)

# 4. Per-Class Table
cnn_per_class = per_class_accuracy(cnn_preds, cnn_labels)
bl_per_class = per_class_accuracy(bl_preds, bl_labels)

print(f"{'Digit':>7} | {'CNN Acc':>9} | {'Baseline Acc':>12}")
print("-" * 34)
for d in range(1, 10):
    print(f"{d:>7} | {cnn_per_class[d]:>9.4f} | {bl_per_class[d]:>12.4f}")"""))

    # Cell 9: Confusion Matrix Plot
    cells.append(md("""### Confusion Matrix Heatmap
Inspect which digits the CNN occasionally confuses."""))

    cells.append(code("""try:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(cnn_labels, cnn_preds)
except ImportError:
    cm = np.zeros((9, 9), dtype=int)
    for t, p in zip(cnn_labels, cnn_preds):
        cm[t, p] += 1

fig, ax = plt.subplots(figsize=(7, 6))
cax = ax.imshow(cm, cmap="Blues", interpolation="nearest")
fig.colorbar(cax)

digits = [str(d) for d in range(1, 10)]
ax.set_xticks(range(9))
ax.set_yticks(range(9))
ax.set_xticklabels(digits)
ax.set_yticklabels(digits)
ax.set_xlabel("Predicted Digit", fontsize=11)
ax.set_ylabel("True Digit", fontsize=11)
ax.set_title("DigitCNN Test Set Confusion Matrix (digits 1-9)", fontsize=13)

# Overlay counts
thresh = cm.max() / 2.0
for r in range(9):
    for c in range(9):
        ax.text(c, r, f"{cm[r, c]}",
                ha="center", va="center",
                color="white" if cm[r, c] > thresh else "black",
                fontsize=9)

plt.tight_layout()
plt.show()"""))

    # Cell 10: Interactive Sudoku Reader & Solver Demo
    cells.append(md("""## 6. Upload & Solve Any Sudoku Image
You can pass any image path below (PNG or JPG). The pipeline will:
1. Divide it into 81 cells and auto-detect blank vs clue digits.
2. Read handwritten clue digits with `DigitCNN`.
3. Fix any misread digits using Informed Search (A* / UCS).
4. Solve the entire grid with Arc Consistency (AC-3) + Backtracking.
5. Render a side-by-side visual of the input image, recognized clues, and completed solution!"""))

    cells.append(code("""from solve_image import process_sudoku_image, visualize_solution
from integration.pipeline import SudokuPipeline

# Choose your image path (or use the sample puzzle with misreads):
IMAGE_PATH = PROJECT_ROOT / "data" / "sample_puzzles" / "sample_puzzle_with_misread.png"

# 1. Process image with DigitCNN
grid, proba_matrix, original_img = process_sudoku_image(IMAGE_PATH, model)
clues_count = sum(1 for d in grid if d != 0)
print(f"Detected {clues_count} clue digits and {81 - clues_count} blank cells.")

# 2. Run Search + CSP Pipeline
pipeline = SudokuPipeline(
    nn_model=model,
    solver_mode="ac3",
    search_mode="astar",
    max_corrections=3,
    use_nn=True
)
result = pipeline.run_from_grid(grid, proba_matrix)

# 3. Report Results
if result.solvable:
    print("\\nPUZZLE SOLVED!")
    if result.correction_log:
        print("Corrections applied by Search:")
        for c in result.correction_log:
            print(f"  * Cell ({c.row}, {c.col}): read '{c.read_digit}' -> corrected to '{c.new_digit}' (cost: {c.cost:.4f})")
    print(f"Search Nodes Expanded: {result.search_nodes_expanded} | CSP Backtracks: {result.csp_backtracks}")
    
    # Visualise inline
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_img, cmap="gray")
    axes[0].set_title("Input Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    
    # Solved board
    solved_text = ""
    for r in range(9):
        row_str = " ".join(str(result.solution.get(r * 9 + c, 0)) for c in range(9))
        solved_text += row_str + "\\n"
    axes[1].text(0.5, 0.5, solved_text, fontsize=13, family="monospace", ha="center", va="center")
    axes[1].set_title("Solved Board (AC-3 CSP)", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("Could not solve puzzle (unreadable or invalid).")"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.14.0"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # Save at root and notebooks/
    p1 = Path("C:/Users/karee/.gemini/antigravity/scratch/sudoku_project/Train_Model.ipynb")
    p2 = Path("C:/Users/karee/.gemini/antigravity/scratch/sudoku_project/notebooks/train_and_demo.ipynb")
    
    with open(p1, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
        
    with open(p2, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
        
    print(f"Created: {p1}")
    print(f"Created: {p2}")

if __name__ == "__main__":
    create_notebook()
