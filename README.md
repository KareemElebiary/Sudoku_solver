# Handwritten Sudoku Reader & Solver

A system that reads handwritten digits from a Sudoku grid, detects misread cells using constraint propagation, searches for the cheapest corrections using informed search, and solves the corrected puzzle using CSP techniques.

## Architecture

```
MNIST Images → CNN Classifier → AC-3 Pre-check → UCS/Greedy/A* Search → BT/FC/AC-3 Solver → Solution
```

**Neural Network** classifies each cell's handwritten digit (1–9) with calibrated probabilities.  
**Search** finds the cheapest set of corrections (≤3 cells) to make a broken puzzle solvable.  
**CSP** solves the corrected Sudoku using backtracking with optional Forward Checking and AC-3.

## System Requirements

- Python 3.10+
- pip

## Setup

```bash
cd sudoku_project
pip install -r requirements.txt
```

## Run Commands

### Interactive Web Application (Drag & Drop UI)
```bash
python app.py
# Open http://localhost:8000 in your browser
```

### Solve Any Sudoku Image
```bash
python solve_image.py --image path/to/sudoku.png
```

### Full Integration Demo
```bash
python run_demo.py --puzzle_seed 42 --misreads 2 --search astar --solver ac3
```

### Data Preparation
```bash
python data/make_data.py
```

### Train Neural Network
```bash
python nn/train.py --epochs 10 --seed 42 --device cpu
```

### Evaluate Neural Network
```bash
python nn/evaluate.py
```

### Run Experiments
```bash
python eval/experiments.py
```

### Run Tests
```bash
python -m pytest tests/ -v
```

## Module Descriptions

| Module | Description |
|---|---|
| `data/make_data.py` | Downloads MNIST, generates Sudoku puzzles, builds mosaics, injects hard digits |
| `data/data_dict.md` | Data dictionary for all fields in the dataset |
| `nn/model.py` | CNN digit classifier (`DigitCNN`) and logistic regression baseline |
| `nn/train.py` | Training script with CLI flags for epochs, seed, device |
| `nn/evaluate.py` | Evaluation: accuracy, confusion matrix, ECE, model comparison |
| `search/ucs.py` | Uniform Cost Search for correction finding |
| `search/greedy.py` | Greedy Best-First Search (heuristic: conflict count) |
| `search/astar.py` | A* Search (cost + unresolved conflict groups heuristic) |
| `search/state.py` | Immutable `CorrectionState` dataclass |
| `csp/sudoku_csp.py` | Sudoku CSP formulation (81 vars, domains, constraints) |
| `csp/backtracking.py` | Pure backtracking solver with MRV/LCV |
| `csp/forward_checking.py` | Backtracking + Forward Checking |
| `csp/ac3.py` | AC-3 arc consistency + combined AC-3+FC solver |
| `integration/pipeline.py` | End-to-end pipeline: NN → Search → CSP |
| `eval/experiments.py` | Three experiments with tables and plots |
| `eval/metrics.py` | Recovery rate, ECE, grid formatting, conflict counting |
| `run_demo.py` | One-command demo with visualization |

## Experiments

### Experiment 1: Search Algorithm Comparison
UCS vs Greedy vs A* on puzzles with 0–4 misread cells. Measures nodes expanded and recovery rate.

### Experiment 2: CSP Solver Comparison
BT vs BT+FC vs BT+FC+AC-3 on puzzles with 25, 30, and 35 givens. Measures backtracks, assignments, and time.

### Experiment 3: Correction Mode Comparison
NN-guided correction vs no correction vs random correction. Measures recovery rate across 1–3 misread levels.

All experiments use ≥5 random seeds and report mean ± standard deviation.

## Expected Output

Running `python run_demo.py` produces:
1. Console output showing the original grid, corrections, solution, and statistics
2. `demo_output.png` — 4-panel figure with original grid, corrected grid, solution, and CSP comparison
3. Search trace with nodes expanded/generated
4. CSP stats with backtracks/assignments for BT, FC, and AC-3

## Dataset

- **Source**: [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- **Size**: ~11 MiB (4 gzip files)
- **Split**: 50,000 train / 10,000 validation / 10,000 test (digits 1–9 only, digit 0 dropped)
- **Puzzles**: Generated synthetically via seeded randomized backtracking
