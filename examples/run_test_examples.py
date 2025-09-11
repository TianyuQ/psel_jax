#!/usr/bin/env python3
"""
Example script demonstrating the new test structure for receding horizon testing.

This script shows how to run the 4 different test configurations:
1a. prediction_test + true_goals
1b. prediction_test + goal_inference  
2a. planning_test + true_goals
2b. planning_test + goal_inference

Usage:
    python3 examples/run_test_examples.py [test_type] [goal_source]
    
Examples:
    python3 examples/run_test_examples.py prediction_test true_goals
    python3 examples/run_test_examples.py planning_test goal_inference
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config_loader import load_config

def run_test(test_type: str, goal_source: str, use_baseline: bool = True, baseline_mode: str = "Control Barrier Function"):
    """
    Run a specific test configuration.
    
    Args:
        test_type: "prediction_test" or "planning_test"
        goal_source: "true_goals" or "goal_inference"
        use_baseline: Whether to use baseline methods
        baseline_mode: Baseline method to use
    """
    print(f"Running {test_type} with {goal_source} goals...")
    print(f"Using baseline: {use_baseline} ({baseline_mode})")
    print("=" * 60)
    
    # Load config and update test parameters
    config = load_config()
    config.testing.receding_horizon.test_type = test_type
    config.testing.receding_horizon.goal_source = goal_source
    config.testing.receding_horizon.use_baseline = use_baseline
    config.testing.receding_horizon.baseline_mode = baseline_mode
    
    # Run the test
    test_script = project_root / "player_selection_network" / "test_psn_receding_horizon.py"
    
    try:
        result = subprocess.run([
            sys.executable, str(test_script)
        ], cwd=str(project_root), check=True, capture_output=True, text=True)
        
        print("Test completed successfully!")
        print("Output:", result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Test failed with error: {e}")
        print("Error output:", e.stderr)
        return False
    
    return True

def print_usage():
    """Print usage information."""
    print("Usage: python3 examples/run_test_examples.py [test_type] [goal_source]")
    print()
    print("Available test types:")
    print("  prediction_test  - All agents' goals are not known (tests goal inference + player selection)")
    print("  planning_test    - Ego agent's goal is always known (tests player selection only)")
    print()
    print("Available goal sources:")
    print("  true_goals       - Use ground truth goals")
    print("  goal_inference   - Use goal inference model predictions")
    print()
    print("Examples:")
    print("  python3 examples/run_test_examples.py prediction_test true_goals")
    print("  python3 examples/run_test_examples.py planning_test goal_inference")
    print()
    print("Test configurations:")
    print("  1a. prediction_test + true_goals     - Use true goals, test goal inference + player selection")
    print("  1b. prediction_test + goal_inference - Use inferred goals, test goal inference + player selection")
    print("  2a. planning_test + true_goals       - Use true goals, test player selection only")
    print("  2b. planning_test + goal_inference   - Use inferred goals, test player selection only")

def main():
    """Main function."""
    if len(sys.argv) != 3:
        print_usage()
        sys.exit(1)
    
    test_type = sys.argv[1]
    goal_source = sys.argv[2]
    
    # Validate inputs
    valid_test_types = ["prediction_test", "planning_test"]
    valid_goal_sources = ["true_goals", "goal_inference"]
    
    if test_type not in valid_test_types:
        print(f"Error: Invalid test_type '{test_type}'. Must be one of: {valid_test_types}")
        print_usage()
        sys.exit(1)
    
    if goal_source not in valid_goal_sources:
        print(f"Error: Invalid goal_source '{goal_source}'. Must be one of: {valid_goal_sources}")
        print_usage()
        sys.exit(1)
    
    # Run the test
    success = run_test(test_type, goal_source)
    
    if success:
        print("\nTest completed successfully!")
        print(f"Results saved to: baseline_results/{test_type}/N_10/receding_horizon_results_*")
    else:
        print("\nTest failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
