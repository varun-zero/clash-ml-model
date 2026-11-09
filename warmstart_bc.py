"""
Root wrapper for warmstart_bc.py
"""
import sys
import os

sys.path.insert(0, os.path.abspath("."))
from cr_alphastar_agent.warmstart_bc import main

if __name__ == "__main__":
    main()
