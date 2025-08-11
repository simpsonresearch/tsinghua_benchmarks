#!/usr/bin/env python3
"""
Shortest Path Algorithm Benchmark
Main entry point for the benchmarking application.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui import BenchmarkUI

def main():
    """Main entry point for the application."""
    app = BenchmarkUI()
    app.run()

if __name__ == "__main__":
    main()