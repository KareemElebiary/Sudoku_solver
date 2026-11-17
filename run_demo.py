"""
run_demo.py
===========
One-command demo for the Handwritten Sudoku Reader & Solver project.

Usage
-----
    python run_demo.py [--puzzle_seed 42] [--misreads 2] [--search ucs] [--solver ac3]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from csp.backtracking import BacktrackingSolver
from csp.forward_checking import ForwardCheckingSolver
from csp.ac3 import AC3Solver
from csp.sudoku_csp import SudokuCSP
from integration.pipeline import SudokuPipeline
from eval.metrics import format_grid


# ===========================================================================
# Self-contained Sudoku generator (same logic as eval/experiments.py)
# ===========================================================================

def _is_valid_placement(grid: list[int], idx: int, digit: int) -> bool:
    """Return True if placing *digit* at *idx* causes no conflict."""
    row, col = divmod(idx, 9)
    br, bc = (row // 3) * 3, (col // 3) * 3
    for c in range(9):
        if grid[row * 9 + c] == digit:
            return False
    for r in range(9):
        if grid[r * 9 + col] == digit:
            return False
    for dr in range(3):
        for dc in range(3):
            if grid[(br + dr) * 9 + (bc + dc)] == digit:
                return False
    return True


def _fill_grid(grid: list[int], rng: random.Random) -> bool:
    """Backtracking filler — fills all 81 cells of *grid* in-place."""
    try:
        idx = grid.index(0)
    except ValueError:
        return True
    digits = list(range(1, 10))
    rng.shuffle(digits)
    for d in digits:
        if _is_valid_placement(grid, idx, d):
            grid[idx] = d
            if _fill_grid(grid, rng):
                return True
            grid[idx] = 0
    return False


def generate_full_solution(seed: int) -> list[int]:
    """Generate a fully filled, valid 9×9 Sudoku grid."""
    rng = random.Random(seed)
    grid = [0] * 81
    _fill_grid(grid, rng)
    return grid


def make_puzzle(solution: list[int], num_givens: int, seed: int) -> list[int]:
    """Create a puzzle by removing cells from a complete solution."""
    rng = random.Random(seed + 1000)
    indices = list(range(81))
    rng.shuffle(indices)
    remove = set(indices[num_givens:])
    return [0 if i in remove else solution[i] for i in range(81)]


def inject_misreads(
    grid: list[int],
    solution: list[int],
    num_misreads: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Inject random misread errors into non-blank cells."""
    rng = random.Random(seed + 2000)
    filled = [i for i in range(81) if grid[i] != 0]
    if num_misreads > len(filled):
        num_misreads = len(filled)
    misread_idx = rng.sample(filled, num_misreads)
    new_grid = list(grid)
    for idx in misread_idx:
        cur = new_grid[idx]
        alts = [d for d in range(1, 10) if d != cur]
        new_grid[idx] = rng.choice(alts)
    return new_grid, misread_idx


def make_proba_matrix(
    grid: list[int],
    solution: list[int],
    confidence: float = 0.85,
    seed: int = 0,
) -> np.ndarray:
    """Build a synthetic (81, 9) probability matrix."""
    rng = np.random.default_rng(seed)
    proba = np.ones((81, 9), dtype=float) / 9.0
    for i in range(81):
        digit = grid[i]
        if digit == 0:
            continue
        proba[i] = (1.0 - confidence) / 8.0
        proba[i, digit - 1] = confidence
    noise = rng.uniform(0, 0.01, size=(81, 9))
    proba += noise
    proba /= proba.sum(axis=1, keepdims=True)
    return proba


# ===========================================================================
# Grid display helpers for plots
# ===========================================================================

def _grid_to_2d(grid: list[int]) -> list[list[str]]:
    """Convert flat grid to 9×9 list of display strings."""
    return [
        [str(grid[r * 9 + c]) if grid[r * 9 + c] != 0 else "." for c in range(9)]
        for r in range(9)
    ]


def _draw_sudoku_grid(ax: plt.Axes, grid: list[int], title: str,
                      highlights: dict[int, str] | None = None) -> None:
    """Draw a 9×9 Sudoku grid on a matplotlib Axes."""
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    if highlights is None:
        highlights = {}

    for r in range(9):
        for c in range(9):
            idx = r * 9 + c
            val = grid[idx]
            display = str(val) if val != 0 else ""

            # Cell background
            color = highlights.get(idx, "white")
            rect = plt.Rectangle((c, 8 - r), 1, 1, facecolor=color,
                                 edgecolor="gray", linewidth=0.5)
            ax.add_patch(rect)

            # Cell text
            ax.text(c + 0.5, 8 - r + 0.5, display,
                    ha="center", va="center", fontsize=10, fontweight="bold")

    # Thick box lines
    for i in range(10):
        lw = 2.5 if i % 3 == 0 else 0.5
        ax.plot([i, i], [0, 9], color="black", linewidth=lw)
        ax.plot([0, 9], [i, i], color="black", linewidth=lw)


# ===========================================================================
# Main demo
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Handwritten Sudoku Reader & Solver — Demo"
    )
    parser.add_argument("--puzzle_seed", type=int, default=42,
                        help="Seed for puzzle generation (default: 42)")
    parser.add_argument("--misreads", type=int, default=2,
                        help="Number of injected misread digits (default: 2)")
    parser.add_argument("--search", type=str, default="ucs",
                        choices=["ucs", "greedy", "astar"],
                        help="Search algorithm (default: ucs)")
    parser.add_argument("--solver", type=str, default="ac3",
                        choices=["bt", "fc", "ac3"],
                        help="CSP solver (default: ac3)")
    parser.add_argument("--num_givens", type=int, default=30,
                        help="Number of clue cells in the puzzle (default: 30)")
    args = parser.parse_args()

    seed = args.puzzle_seed
    n_mis = args.misreads

    # ------------------------------------------------------------------
    # 1. Generate puzzle & inject misreads
    # ------------------------------------------------------------------
    print(f"\n{'=' * 50}")
    print(f"  Sudoku Demo — seed={seed}, misreads={n_mis}")
    print(f"  search={args.search.upper()}, solver={args.solver.upper()}")
    print(f"{'=' * 50}\n")

    solution = generate_full_solution(seed)
    puzzle = make_puzzle(solution, num_givens=args.num_givens, seed=seed)
    misread_grid, misread_indices = inject_misreads(puzzle, solution, n_mis, seed)
    proba = make_proba_matrix(misread_grid, solution, confidence=0.85, seed=seed)

    # Also give the correct digit higher probability for the *true* digits
    # (simulating NN giving correct but lower confidence on the right answer)
    for idx in misread_indices:
        correct_digit = puzzle[idx]
        if correct_digit != 0:
            # The misread digit has high proba, the correct digit has moderate proba
            proba[idx, :] = 0.005
            proba[idx, misread_grid[idx] - 1] = 0.52   # misread (top-1)
            proba[idx, correct_digit - 1] = 0.45       # correct (top-2)
            proba[idx] /= proba[idx].sum()

    print("=== Original (NN Reading — may contain misreads) ===")
    print(format_grid(misread_grid))
    print()

    if misread_indices:
        print("Injected misreads:")
        for idx in misread_indices:
            r, c = divmod(idx, 9)
            print(f"  Cell ({r},{c}): true={puzzle[idx]}, misread={misread_grid[idx]}")
        print()

    # ------------------------------------------------------------------
    # 2. Run the full pipeline
    # ------------------------------------------------------------------
    pipeline = SudokuPipeline(
        nn_model=None,
        solver_mode=args.solver,
        search_mode=args.search,
        max_corrections=3,
        use_nn=False,
    )

    t0 = time.perf_counter()
    result = pipeline.run_from_grid(misread_grid, proba)
    elapsed = (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------
    # 3. Print correction log
    # ------------------------------------------------------------------
    if result.correction_log:
        print("=== Corrections Applied ===")
        for entry in result.correction_log:
            print(f"  Cell ({entry.row},{entry.col}): "
                  f"read={entry.read_digit} -> corrected={entry.new_digit} "
                  f"(cost={entry.cost:.4f})")
        print()

    # ------------------------------------------------------------------
    # 4. Print solution
    # ------------------------------------------------------------------
    if result.solvable and result.solution:
        sol_grid = [result.solution.get(i, 0) for i in range(81)]
        print("=== Solution ===")
        print(format_grid(sol_grid))
    else:
        print("=== No Solution Found: Unreadable ===")
        print("Least-confident cells:")
        confidences = proba.max(axis=1)
        filled = [(i, confidences[i]) for i in range(81) if misread_grid[i] != 0]
        filled.sort(key=lambda x: x[1])
        for idx, conf in filled[:5]:
            r, c = divmod(idx, 9)
            print(f"  Cell ({r},{c}): digit={misread_grid[idx]}, confidence={conf:.3f}")
    print()

    # ------------------------------------------------------------------
    # 5. Print stats
    # ------------------------------------------------------------------
    print(f"Search: {args.search.upper()} | "
          f"Nodes expanded: {result.search_nodes_expanded} | "
          f"Nodes generated: {result.search_nodes_generated}")
    print(f"CSP: {args.solver.upper()} | "
          f"Backtracks: {result.csp_backtracks} | "
          f"Assignments: {result.csp_assignments}")
    print(f"Time: {elapsed:.1f} ms")
    print()

    # ------------------------------------------------------------------
    # 6. CSP comparison on this puzzle (BT vs FC vs AC-3)
    # ------------------------------------------------------------------
    final_grid = result.corrected_grid if result.solvable else misread_grid
    csp_stats = {}
    for mode, SolverClass in [("BT", BacktrackingSolver),
                               ("FC", ForwardCheckingSolver),
                               ("AC3", AC3Solver)]:
        solver = SolverClass()
        csp = SudokuCSP(final_grid)
        solver.solve(csp)
        csp_stats[mode] = {
            "backtracks": solver.backtracks,
            "assignments": solver.assignments,
        }

    print("=== CSP Solver Comparison on this puzzle ===")
    print(f"{'Solver':<8} {'Backtracks':>12} {'Assignments':>13}")
    print("-" * 35)
    for mode in ["BT", "FC", "AC3"]:
        s = csp_stats[mode]
        print(f"{mode:<8} {s['backtracks']:>12} {s['assignments']:>13}")
    print()

    # ------------------------------------------------------------------
    # 7. Generate matplotlib figure with 4 subplots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(f"Sudoku Demo — seed={seed}, {n_mis} misreads, "
                 f"{args.search.upper()}/{args.solver.upper()}", fontsize=14)

    # Subplot 1: Original misread grid
    misread_highlights = {idx: "#ffcccc" for idx in misread_indices}
    _draw_sudoku_grid(axes[0, 0], misread_grid,
                      "Original (NN Reading)", misread_highlights)

    # Subplot 2: Corrected grid
    corrected_highlights = {}
    for entry in result.correction_log:
        corrected_highlights[entry.cell_idx] = "#ccffcc"
    _draw_sudoku_grid(axes[0, 1], result.corrected_grid,
                      "After Correction", corrected_highlights)

    # Subplot 3: Final solution
    if result.solvable and result.solution:
        sol_grid = [result.solution.get(i, 0) for i in range(81)]
        _draw_sudoku_grid(axes[1, 0], sol_grid, "Final Solution")
    else:
        axes[1, 0].text(0.5, 0.5, "No Solution\n(Unreadable)",
                        transform=axes[1, 0].transAxes, ha="center", va="center",
                        fontsize=16, color="red")
        axes[1, 0].set_title("Final Solution", fontsize=12, fontweight="bold")

    # Subplot 4: CSP comparison bar chart
    ax4 = axes[1, 1]
    modes_list = ["BT", "FC", "AC3"]
    bt_counts = [csp_stats[m]["backtracks"] for m in modes_list]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bars = ax4.bar(modes_list, bt_counts, color=colors, edgecolor="black", linewidth=0.5)
    ax4.set_ylabel("Backtracks")
    ax4.set_title("CSP Solver Comparison (Backtracks)", fontsize=12, fontweight="bold")
    ax4.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, bt_counts):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out_path = _PROJECT_ROOT / "demo_output.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Demo figure saved -> {out_path}")


if __name__ == "__main__":
    main()
