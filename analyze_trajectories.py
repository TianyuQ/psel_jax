#!/usr/bin/env python3
"""
Quick analysis of generated trajectories to understand the issue.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_trajectory(filename):
    """Analyze a single trajectory file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    init_positions = np.array(data["init_positions"])
    target_positions = np.array(data["target_positions"])
    trajectories = data["trajectories"]
    
    print(f"\nAnalyzing {filename}:")
    print("=" * 50)
    
    for i in range(len(init_positions)):
        agent_key = f"agent_{i}"
        states = np.array(trajectories[agent_key]["states"])
        positions = states[:, :2]  # Extract x, y positions
        
        # Calculate distances
        init_pos = init_positions[i]
        target_pos = target_positions[i]
        final_pos = positions[-1]
        
        # Distance from initial to target
        init_to_target_dist = np.linalg.norm(target_pos - init_pos)
        
        # Distance from final position to target
        final_to_target_dist = np.linalg.norm(target_pos - final_pos)
        
        # Distance traveled
        total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        
        # Progress towards goal (0 = no progress, 1 = reached goal)
        progress = max(0, (init_to_target_dist - final_to_target_dist) / init_to_target_dist)
        
        print(f"Agent {i}:")
        print(f"  Initial pos: ({init_pos[0]:.2f}, {init_pos[1]:.2f})")
        print(f"  Target pos:  ({target_pos[0]:.2f}, {target_pos[1]:.2f})")
        print(f"  Final pos:   ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
        print(f"  Init->Target distance: {init_to_target_dist:.2f}m")
        print(f"  Final->Target distance: {final_to_target_dist:.2f}m")
        print(f"  Total distance traveled: {total_distance:.2f}m")
        print(f"  Progress towards goal: {progress:.1%}")
        
        if progress < 0.1:
            print(f"  ⚠️  WARNING: Agent {i} made very little progress towards goal!")
        elif progress > 0.8:
            print(f"  ✅ Agent {i} reached close to goal!")
        else:
            print(f"  📍 Agent {i} made partial progress.")
        print()

if __name__ == "__main__":
    # Analyze the first few trajectory files
    for i in range(3):
        filename = f"reference_trajectories_4p_test/ref_traj_sample_{i:03d}.json"
        try:
            analyze_trajectory(filename)
        except FileNotFoundError:
            print(f"File {filename} not found")
            break
