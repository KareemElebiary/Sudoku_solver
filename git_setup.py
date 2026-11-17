import os
import sys
from pathlib import Path
from dulwich.repo import Repo
from dulwich.porcelain import init, add, commit, remote_add, status

PROJECT_DIR = Path("C:/Users/karee/.gemini/antigravity/scratch/sudoku_project").resolve()
os.chdir(PROJECT_DIR)

# 1. Initialize repo if not already
git_dir = PROJECT_DIR / ".git"
if not git_dir.exists():
    repo = init(str(PROJECT_DIR))
    print("Initialized Git repository at:", PROJECT_DIR)
else:
    repo = Repo(str(PROJECT_DIR))
    print("Opened existing Git repository at:", PROJECT_DIR)

# 2. Stage all tracked files
add(repo=repo, paths=".")
print("Staged all files successfully.")

# 3. Create initial commit
commit_msg = b"Initial commit: End-to-end Handwritten Sudoku Reader and Solver (Perception + Search + CSP)"
try:
    c_id = commit(
        repo=repo,
        message=commit_msg,
        author=b"Kareem Elebiary <kareem@example.com>",
        committer=b"Kareem Elebiary <kareem@example.com>"
    )
    print("Committed successfully with commit ID:", c_id.decode("utf-8") if isinstance(c_id, bytes) else c_id)
except Exception as e:
    print("Commit info:", e)

# 4. Set remote origin
remote_url = "https://github.com/KareemElebiary/Sudoku_solver.git"
config = repo.get_config()
config.set((b"remote", b"origin"), b"url", remote_url.encode("utf-8"))
config.set((b"remote", b"origin"), b"fetch", b"+refs/heads/*:refs/remotes/origin/*")
config.write_to_path()
print("Remote 'origin' set to:", remote_url)
