"""Run all four experiments in separate processes."""
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    for name in ('chapter3', 'chapter5', 'chapter6', 'chapter7'):
        subprocess.run([sys.executable, str(root / f'{name}.py'), *sys.argv[1:]], check=True)
