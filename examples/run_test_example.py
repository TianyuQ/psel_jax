#!/usr/bin/env python3
"""
Example script showing how to run the PSN model test.

This script demonstrates different ways to test a trained PSN model.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def find_latest_model():
    """Find the most recent trained model in the log directory."""
    log_dir = Path("log")
    if not log_dir.exists():
        print("No log directory found. Please train a model first.")
        return None
    
    # Find all directories that contain trained models
    model_dirs = []
    for item in log_dir.iterdir():
        if item.is_dir() and (item / "psn_trained_model.pkl").exists():
            model_dirs.append(item)
    
    if not model_dirs:
        print("No trained models found in log directory.")
        return None
    
    # Sort by modification time (most recent first)
    model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_model = model_dirs[0] / "psn_trained_model.pkl"
    
    print(f"Found latest model: {latest_model}")
    return str(latest_model)

def run_basic_test(model_path, num_samples=5):
    """Run a basic test with default settings."""
    print(f"\n{'='*60}")
    print("RUNNING BASIC TEST")
    print(f"{'='*60}")
    
    cmd = [
        "python", "examples/test_trained_psn.py",
        "--model_path", model_path,
        "--num_samples", str(num_samples)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Basic test completed successfully!")
        print("\nOutput:")
        print(result.stdout)
    else:
        print("❌ Basic test failed!")
        print("\nError output:")
        print(result.stderr)

def run_comprehensive_test(model_path, num_samples=10, save_dir="test_results"):
    """Run a comprehensive test with results saving."""
    print(f"\n{'='*60}")
    print("RUNNING COMPREHENSIVE TEST")
    print(f"{'='*60}")
    
    cmd = [
        "python", "examples/test_trained_psn.py",
        "--model_path", model_path,
        "--num_samples", str(num_samples),
        "--save_dir", save_dir
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Comprehensive test completed successfully!")
        print(f"Results saved to: {save_dir}")
        
        # Try to load and display summary of results
        results_file = os.path.join(save_dir, "comprehensive_test_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print("\n📊 TEST RESULTS SUMMARY:")
            print(f"  Samples tested: {results['num_samples_tested']}")
            print(f"  Goal Prediction RMSE: {results['goal_prediction']['mean_rmse']:.4f} ± {results['goal_prediction']['std_rmse']:.4f}")
            print(f"  Trajectory Success Rate: {results['trajectory_similarity']['success_rate']:.2%}")
            print(f"  Agent Selection Sparsity: {results['agent_selection']['mean_sparsity']:.2f} ± {results['agent_selection']['std_sparsity']:.2f}")
        
        print("\nOutput:")
        print(result.stdout)
    else:
        print("❌ Comprehensive test failed!")
        print("\nError output:")
        print(result.stderr)

def run_custom_test(model_path, reference_file, num_samples=5):
    """Run a test with custom reference file."""
    print(f"\n{'='*60}")
    print("RUNNING CUSTOM TEST")
    print(f"{'='*60}")
    
    cmd = [
        "python", "examples/test_trained_psn.py",
        "--model_path", model_path,
        "--reference_file", reference_file,
        "--num_samples", str(num_samples)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Custom test completed successfully!")
        print("\nOutput:")
        print(result.stdout)
    else:
        print("❌ Custom test failed!")
        print("\nError output:")
        print(result.stderr)

def main():
    """Main function to demonstrate different testing approaches."""
    print("PSN Model Testing Examples")
    print("="*60)
    
    # Find the latest trained model
    model_path = find_latest_model()
    if not model_path:
        return
    
    print(f"\nUsing model: {model_path}")
    
    # Example 1: Basic test
    run_basic_test(model_path, num_samples=3)
    
    # Example 2: Comprehensive test with saving
    run_comprehensive_test(model_path, num_samples=5, save_dir="test_results_comprehensive")
    
    # Example 3: Test with different reference data (if available)
    if os.path.exists("reference_trajectories_4p/all_reference_trajectories.json"):
        print(f"\n{'='*60}")
        print("NOTE: 4-player reference data available for testing")
        print("You can test with: --reference_file reference_trajectories_4p/all_reference_trajectories.json")
        print(f"{'='*60}")
    
    print(f"\n{'='*60}")
    print("MANUAL TESTING COMMANDS")
    print(f"{'='*60}")
    print("You can also run tests manually:")
    print(f"\n1. Basic test:")
    print(f"   python examples/test_trained_psn.py --model_path {model_path} --num_samples 5")
    
    print(f"\n2. Comprehensive test with saving:")
    print(f"   python examples/test_trained_psn.py --model_path {model_path} --num_samples 10 --save_dir my_test_results")
    
    print(f"\n3. Test with different reference data:")
    print(f"   python examples/test_trained_psn.py --model_path {model_path} --reference_file reference_trajectories_4p/all_reference_trajectories.json")
    
    print(f"\n4. Help:")
    print(f"   python examples/test_trained_psn.py --help")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
