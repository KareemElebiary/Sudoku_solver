import sys
from pathlib import Path
from dulwich.repo import Repo
from dulwich.porcelain import push

PROJECT_DIR = Path("C:/Users/karee/.gemini/antigravity/scratch/sudoku_project").resolve()
repo = Repo(str(PROJECT_DIR))

def do_push(token=None):
    if token:
        url = f"https://KareemElebiary:{token}@github.com/KareemElebiary/Sudoku_solver.git"
    else:
        url = "https://github.com/KareemElebiary/Sudoku_solver.git"
        
    print(f"Pushing to {url.split('@')[-1]}...")
    try:
        # Push to main
        push(repo, url, refspecs=[b"HEAD:refs/heads/main"])
        print("Push completed successfully!")
    except Exception as e:
        print("Push error:", e)

if __name__ == "__main__":
    token = sys.argv[1] if len(sys.argv) > 1 else None
    do_push(token)
