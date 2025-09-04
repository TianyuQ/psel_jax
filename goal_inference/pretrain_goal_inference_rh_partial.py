#!/usr/bin/env python3
"""
Goal Inference Network Pretraining on Receding Horizon Trajectories - Partial Observations

This script is a specialized version of pretrain_goal_inference_rh.py that trains
a goal inference network using only position observations (x, y) instead of full
state observations (x, y, vx, vy).

Usage:
    python3 goal_inference/pretrain_goal_inference_rh_partial.py

This script will automatically set obs_input_type="partial" and train a model
that takes only position data as input but still predicts full goal positions.
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the main RH training script
from goal_inference.pretrain_goal_inference_rh import *

# Override the configuration to use partial observations
config.goal_inference.obs_input_type = "partial"

if __name__ == "__main__":
    print("=" * 80)
    print("GOAL INFERENCE NETWORK PRETRAINING - PARTIAL OBSERVATIONS")
    print("=" * 80)
    print("Training with PARTIAL observations (position only: x, y)")
    print("=" * 80)
    
    # The rest of the execution will use the main script with partial observations
    # The config override above ensures obs_input_type="partial" is used throughout
