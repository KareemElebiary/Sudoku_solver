"""
app.py  –  Local Web UI for uploading and solving Sudoku images.

Run:
----
    python app.py [--port 8000]

Open:
-----
    http://localhost:8000
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solve_image import load_model, process_sudoku_image, visualize_solution
from integration.pipeline import SudokuPipeline

# Pre-load CNN model
MODEL = load_model(ROOT_DIR / "nn" / "digit_cnn.pth")

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Handwritten Sudoku Reader & Solver</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --border: #334155;
      --clue: #38bdf8;
      --solved: #4ade80;
      --corrected: #f43f5e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 30px 20px;
    }
    h1 {
      font-size: 2.2rem;
      margin-bottom: 8px;
      background: linear-gradient(90deg, #38bdf8, #818cf8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    p.subtitle {
      color: var(--text-muted);
      margin-bottom: 25px;
      font-size: 1.05rem;
    }
    .container {
      width: 100%;
      max-width: 900px;
      background: var(--card);
      border-radius: 14px;
      border: 1px solid var(--border);
      padding: 25px;
      box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
    }
    .upload-zone {
      border: 2px dashed var(--border);
      border-radius: 10px;
      padding: 35px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: rgba(15, 23, 42, 0.4);
      margin-bottom: 20px;
    }
    .upload-zone:hover, .upload-zone.dragover {
      border-color: var(--accent);
      background: rgba(56, 189, 248, 0.05);
    }
    .upload-zone input { display: none; }
    .btn {
      background: var(--accent);
      color: #0f172a;
      font-weight: 600;
      border: none;
      padding: 12px 28px;
      border-radius: 8px;
      font-size: 1rem;
      cursor: pointer;
      transition: background 0.15s ease;
      display: inline-block;
    }
    .btn:hover { background: var(--accent-hover); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-sample {
      background: #334155;
      color: #f8fafc;
      margin-left: 10px;
    }
    .btn-sample:hover { background: #475569; }
    #preview-container {
      display: none;
      margin-bottom: 20px;
      text-align: center;
    }
    #preview-img {
      max-height: 220px;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    #status {
      margin: 15px 0;
      font-size: 1rem;
      color: var(--accent);
      text-align: center;
      display: none;
    }
    .results {
      display: none;
      margin-top: 30px;
      border-top: 1px solid var(--border);
      padding-top: 25px;
    }
    .grid-container {
      display: flex;
      flex-wrap: wrap;
      gap: 30px;
      justify-content: center;
      margin-top: 20px;
    }
    .board-card {
      background: #0f172a;
      border-radius: 10px;
      padding: 15px;
      border: 1px solid var(--border);
      text-align: center;
    }
    .board-card h3 {
      margin-bottom: 12px;
      font-size: 1.1rem;
    }
    .sudoku-grid {
      display: grid;
      grid-template-columns: repeat(9, 36px);
      grid-template-rows: repeat(9, 36px);
      gap: 1px;
      background: #475569;
      border: 3px solid #64748b;
      margin: 0 auto;
    }
    .sudoku-cell {
      background: #1e293b;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.15rem;
      font-weight: bold;
    }
    /* 3x3 block borders */
    .sudoku-cell:nth-child(3n) { border-right: 2px solid #64748b; }
    .sudoku-cell:nth-child(9n) { border-right: none; }
    .row-border { border-bottom: 2px solid #64748b; }
    .clue-digit { color: var(--clue); }
    .solved-digit { color: var(--solved); }
    .corrected-digit { color: var(--corrected); }
    .metrics-bar {
      margin-top: 20px;
      background: #0f172a;
      border-radius: 8px;
      padding: 15px;
      display: flex;
      justify-content: space-around;
      border: 1px solid var(--border);
      font-size: 0.95rem;
    }
    .metric-item span {
      display: block;
      color: var(--accent);
      font-size: 1.3rem;
      font-weight: bold;
    }
    .log-card {
      margin-top: 15px;
      background: #0f172a;
      border-radius: 8px;
      padding: 12px 18px;
      border-left: 4px solid var(--corrected);
      font-family: monospace;
      font-size: 0.95rem;
    }
  </style>
</head>
<body>
  <h1>Sudoku Reader & Solver</h1>
  <p class="subtitle">Neural Network Perception (MNIST) + Informed Search (A*) + CSP (AC-3)</p>

  <div class="container">
    <div class="upload-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
      <p style="font-size: 1.1rem; margin-bottom: 8px;">📷 <strong>Click to upload</strong> or drag & drop a Sudoku image</p>
      <p style="color: var(--text-muted); font-size: 0.9rem;">Supports PNG, JPG, or screenshot mosaics</p>
      <input type="file" id="file-input" accept="image/*">
    </div>

    <div id="preview-container">
      <img id="preview-img" src="" alt="Uploaded Sudoku">
      <div style="margin-top: 15px; display: flex; align-items: center; justify-content: center; gap: 15px; flex-wrap: wrap;">
        <label style="color: var(--text-muted); font-size: 0.95rem;">
          Max Corrections:
          <select id="max-corrections-select" style="background: #0f172a; color: var(--text); border: 1px solid var(--border); padding: 6px 12px; border-radius: 6px; margin-left: 6px;">
            <option value="3">3 (Assignment Default)</option>
            <option value="6" selected>6 (Recommended for Mobile/Web Screenshots)</option>
            <option value="8">8 (High Tolerance)</option>
          </select>
        </label>
        <button class="btn" id="solve-btn" onclick="submitImage()">⚡ Solve This Sudoku</button>
        <button class="btn btn-sample" onclick="loadSample()">🎲 Load Sample Puzzle</button>
      </div>
    </div>

    <div style="text-align: center;" id="initial-controls">
      <button class="btn btn-sample" onclick="loadSample()">🎲 Try With Sample Puzzle</button>
    </div>

    <div id="status">Reading image with DigitCNN and solving...</div>

    <div class="results" id="results-panel">
      <div class="metrics-bar" id="metrics-bar"></div>
      <div id="correction-box"></div>
      
      <div class="grid-container">
        <div class="board-card">
          <h3 style="color: var(--clue);">1. Initial Reading (DigitCNN)</h3>
          <div class="sudoku-grid" id="original-board"></div>
        </div>
        <div class="board-card">
          <h3 style="color: var(--solved);">2. Final Solved Grid (AC-3 CSP)</h3>
          <div class="sudoku-grid" id="solved-board"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let currentImageBase64 = null;

    const fileInput = document.getElementById('file-input');
    const dropZone = document.getElementById('drop-zone');
    const previewContainer = document.getElementById('preview-container');
    const previewImg = document.getElementById('preview-img');
    const statusDiv = document.getElementById('status');
    const resultsPanel = document.getElementById('results-panel');

    fileInput.addEventListener('change', e => {
      const file = e.target.files[0];
      if (file) handleFile(file);
    });

    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    });

    function handleFile(file) {
      const reader = new FileReader();
      reader.onload = e => {
        currentImageBase64 = e.target.result;
        previewImg.src = currentImageBase64;
        previewContainer.style.display = 'block';
        document.getElementById('initial-controls').style.display = 'none';
        resultsPanel.style.display = 'none';
      };
      reader.readAsDataURL(file);
    }

    async function loadSample() {
      statusDiv.style.display = 'block';
      statusDiv.innerText = 'Loading sample puzzle...';
      const res = await fetch('/sample');
      const data = await res.json();
      currentImageBase64 = data.image;
      previewImg.src = currentImageBase64;
      previewContainer.style.display = 'block';
      document.getElementById('initial-controls').style.display = 'none';
      submitImage();
    }

    async function submitImage() {
      if (!currentImageBase64) return;
      statusDiv.style.display = 'block';
      statusDiv.innerText = 'Extracting cells, running DigitCNN perception, search corrections, and CSP solve...';
      document.getElementById('solve-btn').disabled = true;

      const maxCorr = parseInt(document.getElementById('max-corrections-select').value) || 6;

      try {
        const res = await fetch('/solve', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: currentImageBase64, max_corrections: maxCorr })
        });
        const data = await res.json();
        statusDiv.style.display = 'none';
        document.getElementById('solve-btn').disabled = false;

        if (data.error) {
          alert("Error: " + data.error);
          return;
        }

        renderResults(data);
      } catch (err) {
        statusDiv.style.display = 'none';
        document.getElementById('solve-btn').disabled = false;
        alert("Failed to solve: " + err);
      }
    }

    function renderResults(data) {
      resultsPanel.style.display = 'block';
      
      // Metrics
      const mBar = document.getElementById('metrics-bar');
      const statusColor = data.solvable ? 'var(--solved)' : 'var(--corrected)';
      const statusText = data.solvable ? 'Solved ✅' : 'Unreadable / Conflict ❌';
      mBar.innerHTML = `
        <div class="metric-item">Status<span style="color: ${statusColor}; font-size: 1.1rem;">${statusText}</span></div>
        <div class="metric-item">Time<span>${data.time_ms.toFixed(1)} ms</span></div>
        <div class="metric-item">Search Nodes<span>${data.nodes_expanded}</span></div>
        <div class="metric-item">Corrections<span>${data.corrections.length}</span></div>
        <div class="metric-item">CSP Backtracks<span>${data.backtracks}</span></div>
      `;

      // Corrections log
      const corrBox = document.getElementById('correction-box');
      if (data.corrections.length > 0) {
        corrBox.innerHTML = '<div class="log-card"><strong>🛠️ A* Search Corrections:</strong><br>' +
          data.corrections.map(c => `Cell (${c.row}, ${c.col}): read '${c.read_digit}' ➔ corrected to '<strong>${c.new_digit}</strong>' (cost: ${c.cost.toFixed(4)})`).join('<br>') +
          '</div>';
      } else if (!data.solvable) {
        corrBox.innerHTML = '<div class="log-card" style="border-left-color: #f43f5e; background: #2a151b;"><strong>⚠️ Unreadable Puzzle / Contradictions Detected:</strong><br>' +
          'The digit perception found conflicting clues that cannot be resolved within the maximum allowed corrections (max 3). This usually occurs if the uploaded image uses digital/computer vector fonts instead of handwritten digits, or if grid alignment misreads cells.</div>';
      } else {
        corrBox.innerHTML = '';
      }

      // Render original and solved board
      const origMap = new Set(data.corrections.map(c => c.row * 9 + c.col));
      renderBoard('original-board', data.original_grid, (idx, val) => {
        if (val === 0) return '';
        if (origMap.has(idx)) return `<span class="corrected-digit">${val}</span>`;
        return `<span class="clue-digit">${val}</span>`;
      });

      if (!data.solvable || !data.solution_grid || data.solution_grid.length === 0) {
        document.getElementById('solved-board').innerHTML = `
          <div style="grid-column: span 9; padding: 45px 15px; color: #f43f5e; font-weight: bold; font-size: 1.05rem; text-align: center;">
            ❌ No Valid Solution Found<br>
            <span style="color: var(--text-muted); font-size: 0.85rem; font-weight: normal; margin-top: 10px; display: block; line-height: 1.4;">
              The puzzle state contains irreconcilable duplicate conflicts in rows/columns/boxes.<br>
              Try using <strong>"Load Sample Puzzle"</strong> for guaranteed handwritten MNIST grids!
            </span>
          </div>`;
      } else {
        renderBoard('solved-board', data.solution_grid, (idx, val) => {
          if (data.original_grid[idx] !== 0) {
            if (origMap.has(idx)) return `<span class="corrected-digit">${val}</span>`;
            return `<span class="clue-digit">${val}</span>`;
          }
          return `<span class="solved-digit">${val}</span>`;
        });
      }
    }

    function renderBoard(elementId, gridData, formatter) {
      const el = document.getElementById(elementId);
      el.innerHTML = '';
      for (let r = 0; r < 9; r++) {
        for (let c = 0; c < 9; c++) {
          const idx = r * 9 + c;
          const cell = document.createElement('div');
          cell.className = 'sudoku-cell';
          if (r === 2 || r === 5) cell.classList.add('row-border');
          cell.innerHTML = formatter(idx, gridData[idx]);
          el.appendChild(cell);
        }
      }
    }
  </script>
</body>
</html>
"""


class SudokuHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path == "/sample":
            # Return sample puzzle as base64
            sample_path = ROOT_DIR / "data" / "sample_puzzles" / "sample_puzzle_with_misread.png"
            if not sample_path.exists():
                sample_path = ROOT_DIR / "data" / "sample_puzzles" / "sample_puzzle_1.png"
            with open(sample_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"image": f"data:image/png;base64,{b64}"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/solve":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))
            
            # Extract base64 image
            raw_b64 = data["image"]
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(raw_b64)
            
            # Save temporary image to disk
            temp_path = ROOT_DIR / "temp_upload.png"
            with open(temp_path, "wb") as f:
                f.write(img_bytes)

            # Check aspect ratio
            with Image.open(temp_path) as check_img:
                w, h = check_img.size
                aspect = max(w, h) / max(min(w, h), 1)
                if aspect > 1.6:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "error": f"Image aspect ratio is {w}x{h} ({aspect:.1f}:1). A Sudoku puzzle must be roughly square (1:1). It looks like you uploaded the 3-panel result image (solved_sudoku_result.png) instead of a single 9x9 Sudoku puzzle!"
                    }).encode("utf-8"))
                    return
                
            # Process & solve
            import time
            t0 = time.perf_counter()
            grid, proba_matrix, _ = process_sudoku_image(temp_path, MODEL)
            
            max_corrections = int(data.get("max_corrections", 6))
            pipeline = SudokuPipeline(
                nn_model=MODEL,
                solver_mode="ac3",
                search_mode="astar",
                max_corrections=max_corrections,
                use_nn=True
            )
            result = pipeline.run_from_grid(grid, proba_matrix)
            t_elapsed = (time.perf_counter() - t0) * 1000.0
            
            corrections = [
                {
                    "row": c.row,
                    "col": c.col,
                    "read_digit": c.read_digit,
                    "new_digit": c.new_digit,
                    "cost": c.cost
                }
                for c in result.correction_log
            ]
            
            solved_grid = [result.solution.get(i, 0) for i in range(81)] if result.solution else []
            
            response = {
                "solvable": result.solvable,
                "original_grid": grid,
                "solution_grid": solved_grid,
                "corrections": corrections,
                "time_ms": t_elapsed,
                "nodes_expanded": result.search_nodes_expanded,
                "nodes_generated": result.search_nodes_generated,
                "backtracks": result.csp_backtracks,
            }
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))


def run_server(port: int = 8000):
    server = HTTPServer(("localhost", port), SudokuHandler)
    print("\n" + "=" * 60)
    print(f"  Sudoku Web App Running at: http://localhost:{port}")
    print("  Drag & drop or upload your Sudoku image in the browser!")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60 + "\n")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(port=args.port)
