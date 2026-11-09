"""
Root wrapper for train.py
"""
import sys
import os

sys.path.insert(0, os.path.abspath("."))
from cr_alphastar_agent.train import main

if __name__ == "__main__":
    main()
