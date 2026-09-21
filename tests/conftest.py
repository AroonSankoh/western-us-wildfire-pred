import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (REPO_ROOT, os.path.join(REPO_ROOT, "scripts"), os.path.join(REPO_ROOT, "training")):
    if path not in sys.path:
        sys.path.insert(0, path)
