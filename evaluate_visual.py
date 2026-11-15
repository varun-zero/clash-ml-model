"""Root wrapper for tools/evaluate_visual.py"""
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from tools.evaluate_visual import *

if __name__ == "__main__":
    from tools.evaluate_visual import main
    main()
