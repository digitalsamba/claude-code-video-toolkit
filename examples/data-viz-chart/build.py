#!/usr/bin/env python3
"""Render the bundled chart or --spec chart.json through the shared Python chart renderer."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'tools'))
from data_video import main

if __name__ == '__main__':
    raise SystemExit(main(default_spec=HERE / 'chart.json', default_output=HERE / 'out.mp4', default_brand='digital-samba'))
