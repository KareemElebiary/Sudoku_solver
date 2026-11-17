"""
solve_image.py
==============
End-to-end Sudoku image reader and solver with robust MNIST centering & stroke normalization.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project root setup
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nn.model import DigitCNN
from integration.pipeline import SudokuPipeline, PipelineResult
from eval.metrics import format_grid


def load_model(weights_path: Path) -> DigitCNN:
    """Load trained DigitCNN model weights."""
    model = DigitCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}. Please train the model first.")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def find_grid_bounding_box(img_gray: Image.Image) -> tuple[int, int, int, int]:
    """Auto-detect the outer bounding box of a printed/screenshot Sudoku grid."""
    arr = np.array(img_gray, dtype=np.float32)
    if arr.mean() > 128:
        # Dark lines on light background
        mask = arr < 180
    else:
        # Light lines on dark background
        mask = arr > 75
        
    ys, xs = np.where(mask)
    if len(xs) > 100 and len(ys) > 100:
        x0 = int(np.percentile(xs, 0.2))
        x1 = int(np.percentile(xs, 99.8))
        y0 = int(np.percentile(ys, 0.2))
        y1 = int(np.percentile(ys, 99.8))
        if (x1 - x0) > 150 and (y1 - y0) > 150:
            return (x0, y0, x1, y1)
    return (0, 0, img_gray.size[0], img_gray.size[1])


def mnist_normalize(patch_np: np.ndarray) -> np.ndarray | None:
    """Normalize a raw cell patch to standard MNIST format:
    1. Threshold foreground stroke.
    2. Bounding box crop.
    3. Thicken stroke width for thin digital fonts.
    4. Aspect-ratio preserving resize into 20x20.
    5. Center-of-mass centering on 28x28 canvas.
    """
    mask = patch_np > 70
    if mask.sum() < 25:
        return None  # empty cell
        
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    
    digit_crop = patch_np[y0:y1, x0:x1]
    h, w = digit_crop.shape
    if h < 5 or w < 2:
        return None
        
    digit_img = Image.fromarray(np.clip(digit_crop * (255.0 / max(digit_crop.max(), 1.0)), 0, 255).astype(np.uint8))
    # Thicken strokes slightly to match MNIST handwriting profile
    digit_img = digit_img.filter(ImageFilter.MaxFilter(3))
    
    # Scale to fit inside 20x20
    if h > w:
        new_h = 20
        new_w = max(1, int(round(w * 20.0 / h)))
    else:
        new_w = 20
        new_h = max(1, int(round(h * 20.0 / w)))
        
    digit_res = digit_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    digit_arr = np.array(digit_res, dtype=np.float32)
    
    # Place on 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = digit_arr
    
    # Center-of-mass alignment
    sum_val = max(canvas.sum(), 1.0)
    cy = np.sum(np.arange(28)[:, None] * canvas) / sum_val
    cx = np.sum(np.arange(28)[None, :] * canvas) / sum_val
    shift_y = int(round(13.5 - cy))
    shift_x = int(round(13.5 - cx))
    
    canvas = np.roll(canvas, shift_y, axis=0)
    canvas = np.roll(canvas, shift_x, axis=1)
    return canvas


def process_sudoku_image(
    image_path: Path,
    model: DigitCNN,
    empty_threshold_max: float = 40.0,
    empty_threshold_var: float = 25.0,
) -> tuple[list[int], np.ndarray, Image.Image]:
    """Slice an image into 81 cells and extract digit probabilities."""
    img_raw = Image.open(image_path).convert("L")
    w, h = img_raw.size
    
    is_mnist_mosaic = (w == 252 and h == 252) or (w == h and np.array(img_raw).mean() < 60)
    
    if is_mnist_mosaic:
        img = img_raw
        margin = 0.0
    else:
        x0_grid, y0_grid, x1_grid, y1_grid = find_grid_bounding_box(img_raw)
        img = img_raw.crop((x0_grid, y0_grid, x1_grid, y1_grid))
        margin = 0.12
        
    img_np = np.array(img, dtype=np.float32)
    if img_np.mean() > 120:
        img_np = 255.0 - img_np
        
    width, height = img.size
    cell_w = width / 9.0
    cell_h = height / 9.0
    
    grid = [0] * 81
    proba_matrix = np.full((81, 9), 1.0 / 9.0, dtype=float)
    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
    
    for r in range(9):
        for c in range(9):
            idx = r * 9 + c
            if margin > 0:
                x0 = int(c * cell_w + cell_w * margin)
                y0 = int(r * cell_h + cell_h * margin)
                x1 = int((c + 1) * cell_w - cell_w * margin)
                y1 = int((r + 1) * cell_h - cell_h * margin)
            else:
                x0 = int(c * cell_w)
                y0 = int(r * cell_h)
                x1 = int((c + 1) * cell_w)
                y1 = int((r + 1) * cell_h)
                
            patch_np = img_np[y0:y1, x0:x1]
            
            if is_mnist_mosaic:
                if patch_np.size == 0 or patch_np.max() < empty_threshold_max or patch_np.var() < empty_threshold_var:
                    grid[idx] = 0
                    proba_matrix[idx] = np.full(9, 1.0 / 9.0)
                else:
                    tensor_input = torch.tensor(patch_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) / 255.0
                    probs = model.predict_proba(tensor_input)[0]
                    grid[idx] = int(np.argmax(probs)) + 1
                    proba_matrix[idx] = probs
            else:
                norm_patch = mnist_normalize(patch_np)
                if norm_patch is None:
                    grid[idx] = 0
                    proba_matrix[idx] = np.full(9, 1.0 / 9.0)
                else:
                    tensor_input = torch.tensor(norm_patch, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) / 255.0
                    probs = model.predict_proba(tensor_input)[0]
                    grid[idx] = int(np.argmax(probs)) + 1
                    proba_matrix[idx] = probs
                
    return grid, proba_matrix, img


def visualize_solution(
    original_img: Image.Image,
    initial_grid: list[int],
    result: PipelineResult,
    save_path: Path,
):
    """Save side-by-side visualization of input, recognition/corrections, and solved puzzle."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_img, cmap="gray")
    axes[0].set_title("1. Uploaded Sudoku Image", fontsize=13, fontweight="bold")
    axes[0].axis("off")
    
    def draw_grid_text(ax, grid_data, title, highlight_cells=None, color="black"):
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 9)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold")
        
        for i in range(10):
            lw = 2.0 if i % 3 == 0 else 0.7
            ax.axhline(i, color="gray", lw=lw)
            ax.axvline(i, color="gray", lw=lw)
            
        highlight_dict = highlight_cells or {}
        for r in range(9):
            for c in range(9):
                idx = r * 9 + c
                val = grid_data[idx]
                if val != 0:
                    text_color = highlight_dict.get(idx, color)
                    weight = "bold" if idx in highlight_dict else "normal"
                    ax.text(c + 0.5, r + 0.5, str(val), fontsize=15,
                            ha="center", va="center", color=text_color, fontweight=weight)
                            
    corrections_map = {entry.cell_idx: "crimson" for entry in result.correction_log}
    draw_grid_text(
        axes[1],
        initial_grid,
        f"2. DigitCNN Reading ({len(result.correction_log)} misreads fixed)",
        highlight_cells=corrections_map
    )
    
    if result.solvable and result.solution is not None:
        solved_list = [result.solution.get(i, 0) for i in range(81)]
        orig_clues = {i: "royalblue" for i in range(81) if initial_grid[i] != 0}
        for entry in result.correction_log:
            orig_clues[entry.cell_idx] = "forestgreen"
            
        draw_grid_text(axes[2], solved_list, "3. Fully Solved Puzzle (CSP)", highlight_cells=orig_clues)
    else:
        axes[2].text(0.5, 0.5, "Unreadable / No Solution", fontsize=14, color="red", ha="center")
        axes[2].axis("off")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Upload & solve a handwritten Sudoku puzzle.")
    parser.add_argument(
        "--image",
        type=str,
        default=str(ROOT_DIR / "data" / "sample_puzzles" / "sample_puzzle_1.png"),
        help="Path to Sudoku image file (.png, .jpg)."
    )
    parser.add_argument("--search", type=str, default="astar", choices=["ucs", "greedy", "astar"])
    parser.add_argument("--solver", type=str, default="ac3", choices=["bt", "fc", "ac3"])
    parser.add_argument("--max-corrections", type=int, default=6)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        fallback = Path("C:/Users/karee/Downloads/sample_sudoku_puzzle.png")
        if fallback.exists():
            image_path = fallback
        else:
            print(f"[Error] Image not found: {image_path}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  Handwritten Sudoku Reader & Solver Pipeline")
    print("=" * 60)
    print(f"Input image: {image_path}")
    print(f"Search mode: {args.search.upper()} | CSP solver: {args.solver.upper()}")

    model_path = ROOT_DIR / "nn" / "digit_cnn.pth"
    model = load_model(model_path)

    print("\n[1/3] Reading cells with DigitCNN...")
    initial_grid, proba_matrix, original_img = process_sudoku_image(image_path, model)
    givens_count = sum(1 for d in initial_grid if d != 0)
    print(f"      Detected {givens_count} filled clue cells and {81 - givens_count} blank cells.")
    print("\nDecoded Initial Clue Grid:")
    print(format_grid(initial_grid))

    print("\n[2/3] Checking constraints & running Search + CSP...")
    t0 = time.perf_counter()
    pipeline = SudokuPipeline(
        nn_model=model,
        solver_mode=args.solver,
        search_mode=args.search,
        max_corrections=args.max_corrections,
        use_nn=True
    )
    result = pipeline.run_from_grid(initial_grid, proba_matrix)
    t_elapsed = (time.perf_counter() - t0) * 1000.0

    if result.solvable and result.solution is not None:
        print("\n" + "=" * 60)
        print("  PUZZLE SOLVED SUCCESSFULLY!")
        print("=" * 60)
        if result.correction_log:
            print(f"\nCorrections made by {args.search.upper()} Search:")
            for c in result.correction_log:
                print(f"  * Cell ({c.row}, {c.col}): read '{c.read_digit}' -> corrected to '{c.new_digit}' (cost: {c.cost:.4f})")
        else:
            print("\nNo misreads detected; puzzle was directly consistent!")

        solved_grid = [result.solution.get(i, 0) for i in range(81)]
        print("\nFinal Solved Sudoku Grid:")
        print(format_grid(solved_grid))

        print(f"\nPerformance Metrics:")
        print(f"  * Total Time        : {t_elapsed:.1f} ms")
        print(f"  * Nodes Expanded    : {result.search_nodes_expanded}")
        print(f"  * Nodes Generated   : {result.search_nodes_generated}")
        print(f"  * CSP Backtracks    : {result.csp_backtracks}")
        print(f"  * CSP Assignments   : {result.csp_assignments}")

        out_plot = ROOT_DIR / "solved_sudoku_result.png"
        visualize_solution(original_img, initial_grid, result, out_plot)
        print(f"\n[3/3] Solution visual saved -> {out_plot}")
    else:
        print("\n[Status] Puzzle marked as UNREADABLE (contradiction cannot be resolved within max corrections).")


if __name__ == "__main__":
    main()
