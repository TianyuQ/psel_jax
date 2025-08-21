#!/usr/bin/env python3
"""
Standalone Goal Prediction Test Script for Receding Horizon

This script tests the pretrained goal inference model on reference trajectory data
in the context of receding horizon planning. It focuses specifically on goal prediction
accuracy without the complexity of full PSN testing.

All parameters are loaded from config.yaml for consistency across scripts.
"""

import argparse
import gc
import glob
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import configuration loader
from config_loader import load_config, get_device_config, setup_jax_config

# Import the goal inference network
from goal_inference.pretrain_goal_inference import GoalInferenceNetwork

# Import Flax for model loading
import flax.serialization
from flax.training import train_state

# Import the correct extract_observation_trajectory function from pretrain_goal_inference
from goal_inference.pretrain_goal_inference import extract_observation_trajectory

# ============================================================================
# LOAD CONFIGURATION AND SETUP
# ============================================================================

# Load configuration from config.yaml
config = load_config()

# Setup JAX configuration
setup_jax_config()

# Get device from configuration
device = get_device_config()
print(f"Using device: {device}")

# Extract parameters from configuration
N_agents = config.game.N_agents
T_observation = config.goal_inference.observation_length
T_total = config.game.T_total
state_dim = config.game.state_dim

if N_agents == 4:
    hidden_dims = config.goal_inference.hidden_dims_4p
elif N_agents == 10:
    hidden_dims = config.goal_inference.hidden_dims_10p
else:
    # Default fallback to 4p dimensions
    hidden_dims = config.goal_inference.hidden_dims_4p
    print(f"Warning: Using 4p hidden dimensions {hidden_dims} for {N_agents} agents")

print(f"Configuration loaded:")
print(f"  N agents: {N_agents}")
print(f"  T observation: {T_observation}")
print(f"  T total: {T_total}")
print(f"  State dim: {state_dim}")

# Constants
SEPARATOR_LINE = "=" * 80


def clear_gpu_memory():
    """Clear GPU memory."""
    if hasattr(jax, 'clear_caches'):
        jax.clear_caches()
    gc.collect()


def load_trained_goal_model(goal_model_path: str) -> Tuple[GoalInferenceNetwork, Any]:
    """
    Load trained goal inference model from file.
    
    Args:
        goal_model_path: Path to the trained goal inference model file
        
    Returns:
        Tuple of (goal_model, goal_trained_state)
    """
    print(f"Loading trained goal inference model from: {goal_model_path}")
    
    # Load the goal inference model bytes
    with open(goal_model_path, 'rb') as f:
        goal_model_bytes = pickle.load(f)
    
    # Create the goal inference model
    goal_model = GoalInferenceNetwork(hidden_dims=hidden_dims)
    
    # Deserialize the goal inference state
    goal_trained_state = flax.serialization.from_bytes(goal_model, goal_model_bytes)
    print("✓ Goal inference model loaded successfully")
    
    return goal_model, goal_trained_state


def extract_reference_goals(sample_data: Dict[str, Any]) -> jnp.ndarray:
    """Extract reference goals from sample data."""
    # Use the target_positions field as in the training data
    return jnp.array(sample_data["target_positions"])  # (N_agents, goal_dim)


def test_goal_prediction(goal_model: GoalInferenceNetwork, 
                         goal_trained_state: Any,
                         sample_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test goal prediction on a single sample using the same approach as training.
    
    Args:
        goal_model: Goal inference model
        goal_trained_state: Trained goal inference model state
        sample_data: Reference trajectory sample
        
    Returns:
        Dictionary containing test results
    """
    # Extract observation trajectory (first 10 steps) as in training
    obs_traj = extract_observation_trajectory(sample_data)
    obs_input = obs_traj.flatten().reshape(1, -1)  # Add batch dimension
    
    # Get goal predictions from pretrained goal inference network
    predicted_goals = goal_model.apply({'params': goal_trained_state['params']}, obs_input, deterministic=True)
    predicted_goals = predicted_goals[0]  # Remove batch dimension
    
    # Extract true goals
    true_goals = extract_reference_goals(sample_data)
    
    # Reshape predicted goals
    predicted_goals = predicted_goals.reshape(N_agents, 2)
    
    # Compute goal prediction error
    # Compute per-agent goal error: err_x² + err_y² for each agent
    goal_diff = predicted_goals - true_goals  # (N_agents, 2)
    per_agent_error = jnp.sum(jnp.square(goal_diff), axis=1)  # (N_agents,) - sum over x,y dimensions
    
    # Take mean over N agents, then compute RMSE
    mean_agent_error = jnp.mean(per_agent_error)  # scalar - mean over agents
    goal_rmse = jnp.sqrt(mean_agent_error)
    
    print(f"      Goals RMSE: {goal_rmse:.4f}")
    
    # Compile results
    results = {
        'goal_prediction_rmse': float(goal_rmse),
        'goal_prediction_error': float(mean_agent_error),  # Use the mean agent error (before sqrt)
        'predicted_goals': predicted_goals.tolist(),
        'true_goals': true_goals.tolist(),
        'mode': 'goal_prediction_test'
    }
    
    return results


def create_goal_map_visualization(sample_data: Dict[str, Any], results: Dict[str, Any], 
                                  sample_id: int, save_dir: str = None) -> None:
    """
    Create a simple, clean map visualization focusing only on goals.
    
    Args:
        sample_data: Reference trajectory sample
        results: Test results dictionary
        sample_id: Sample identifier
        save_dir: Directory to save plots
    """
    # Create a single, focused map plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title(f'Sample {sample_id+1}: Goal Prediction Map\n'
                f'Goal RMSE: {results["goal_prediction_rmse"]:.4f}', fontsize=16)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Color palette for agents
    colors = plt.cm.tab10(np.linspace(0, 1, N_agents))
    
    # Get initial positions from first step
    init_positions = []
    for i in range(N_agents):
        agent_key = f"agent_{i}"
        agent_states = sample_data["trajectories"][agent_key]["states"]
        init_positions.append(np.array(agent_states[0][:2]))
    init_positions = np.array(init_positions)
    
    # Plot initial positions (small circles)
    for j in range(N_agents):
        ax.plot(init_positions[j][0], init_positions[j][1], 'o', 
                color=colors[j], markersize=10, alpha=0.7, label=f'Agent {j} Start')
    
    # Plot true goals (medium squares)
    true_goals = np.array(results['true_goals'])
    for j in range(N_agents):
        ax.plot(true_goals[j][0], true_goals[j][1], 's', 
                color=colors[j], markersize=12, alpha=0.9, label=f'Agent {j} True Goal')
    
    # Plot predicted goals (medium triangles with borders)
    predicted_goals = np.array(results['predicted_goals'])
    for j in range(N_agents):
        ax.plot(predicted_goals[j][0], predicted_goals[j][1], '^', 
                color=colors[j], markersize=12, alpha=0.9, 
                markerfacecolor='none', markeredgewidth=2, 
                label=f'Agent {j} Predicted Goal')
    
    # Plot reference trajectories (thin lines)
    for j in range(N_agents):
        agent_key = f"agent_{j}"
        agent_states = sample_data["trajectories"][agent_key]["states"]
        agent_traj = np.array(agent_states)
        ax.plot(agent_traj[:, 0], agent_traj[:, 1], '-', 
                color=colors[j], alpha=0.4, linewidth=1)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f"goal_map_sample_{sample_id:03d}.png")
        plt.savefig(plot_path, dpi=config.reference_generation.plot_dpi, bbox_inches='tight')
        print(f"  Goal map visualization saved: {plot_path}")
    
    plt.close()


def load_reference_data(reference_file: str) -> List[Dict[str, Any]]:
    """
    Load reference trajectory data from directory.
    
    Args:
        reference_file: Path to directory containing ref_traj_sample_*.json files
        
    Returns:
        List of loaded reference trajectory samples
    """
    print(f"Loading reference data from directory: {reference_file}")
    
    # Find all ref_traj_sample_*.json files in the directory
    pattern = os.path.join(reference_file, "ref_traj_sample_*.json")
    json_files = sorted(glob.glob(pattern))
    
    if not json_files:
        raise FileNotFoundError(f"No ref_traj_sample_*.json files found in directory: {reference_file}")
    
    # Load samples
    reference_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                sample_data = json.load(f)
                reference_data.append(sample_data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue
    
    print(f"Loaded {len(reference_data)} reference samples from {reference_file}")
    print(f"Sample files: {len(json_files)} found, {len(reference_data)} loaded successfully")
    
    return reference_data


def save_detailed_results(output_dir: str, summary_stats: Dict, all_results: List, 
                         goal_model_path: str, reference_file: str, num_samples: int) -> None:
    """Save detailed JSON results file."""
    results_file = os.path.join(output_dir, "goal_prediction_test_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'summary_stats': summary_stats,
            'detailed_results': all_results,
            'test_config': {
                'goal_model_path': goal_model_path,
                'reference_file': reference_file,
                'num_samples': num_samples,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")


def save_summary_report(output_dir: str, summary_stats: Dict, all_results: List,
                       goal_model_path: str, reference_file: str, num_samples: int) -> None:
    """Save human-readable summary report."""
    summary_file = os.path.join(output_dir, "goal_prediction_test_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"{SEPARATOR_LINE}\n")
        f.write("GOAL PREDICTION TEST SUMMARY REPORT\n")
        f.write(f"{SEPARATOR_LINE}\n\n")
        
        # Test Configuration
        f.write("Test Configuration:\n")
        f.write(f"  Goal Model: {goal_model_path}\n")
        f.write(f"  Reference File: {reference_file}\n")
        f.write(f"  Number of Samples: {num_samples}\n")
        f.write(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Performance Metrics
        f.write("Performance Metrics:\n")
        if summary_stats['goal_rmse_values']:
            avg_goal_rmse = np.mean(summary_stats['goal_rmse_values'])
            std_goal_rmse = np.std(summary_stats['goal_rmse_values'])
            f.write(f"  Overall Goal RMSE: {avg_goal_rmse:.4f} ± {std_goal_rmse:.4f}\n")
        f.write(f"  Total Samples: {summary_stats['total_samples']}\n\n")
        
        # Sample Results
        f.write("Sample Results:\n")
        for i, result in enumerate(all_results):
            f.write(f"  Sample {i+1}: Goal RMSE: {result['goal_prediction_rmse']:.4f}\n")
    
    print(f"Summary report saved to: {summary_file}")


def run_goal_prediction_testing(goal_model_path: str, reference_file: str, 
                               output_dir: str = None, num_samples: int = 10) -> Dict[str, Any]:
    """
    Run goal prediction testing on the trained model using the same approach as training.
    
    Args:
        goal_model_path: Path to trained goal inference model
        reference_file: Path to reference trajectory file
        output_dir: Directory to save test results
        num_samples: Number of samples to test
        
    Returns:
        Dictionary containing test results
    """
    print(SEPARATOR_LINE)
    print("GOAL PREDICTION TESTING")
    print(SEPARATOR_LINE)
    
    # Determine output directory
    if output_dir is None:
        # Extract goal model directory and create test results subdirectory
        goal_model_dir = os.path.dirname(goal_model_path)
        output_dir = os.path.join(goal_model_dir, "goal_prediction_test_results")
        print(f"Test results will be saved under goal model directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load goal inference model
    goal_model, goal_trained_state = load_trained_goal_model(goal_model_path)
    
    # Load reference data from directory
    reference_data = load_reference_data(reference_file)
    
    # Limit number of samples for testing
    test_samples = reference_data[:min(num_samples, len(reference_data))]
    print(f"Testing on {len(test_samples)} samples")
    
    # Initialize results storage
    all_results = []
    summary_stats = {
        'goal_rmse_values': [],
        'total_samples': len(test_samples)
    }
    
    # Test each sample
    for i, sample_data in enumerate(test_samples):
        print(f"\nTesting sample {i+1}/{len(test_samples)}...")
        
        # Test goal prediction using the same approach as training
        results = test_goal_prediction(
            goal_model, goal_trained_state, sample_data)
        
        # Store results
        all_results.append(results)
        
        # Update summary statistics
        if results['goal_prediction_rmse'] != float('inf'):
            summary_stats['goal_rmse_values'].append(results['goal_prediction_rmse'])
        
        # Create simple goal map visualization
        create_goal_map_visualization(sample_data, results, i, output_dir)
        
        # Print sample results
        print(f"  Overall Goal RMSE: {results['goal_prediction_rmse']:.4f}")
    
    # Compute summary statistics
    print(f"\n{SEPARATOR_LINE}")
    print("TESTING SUMMARY")
    print(SEPARATOR_LINE)
    
    if summary_stats['goal_rmse_values']:
        avg_goal_rmse = np.mean(summary_stats['goal_rmse_values'])
        std_goal_rmse = np.std(summary_stats['goal_rmse_values'])
        print(f"Overall Goal Prediction RMSE: {avg_goal_rmse:.4f} ± {std_goal_rmse:.4f}")
    
    print(f"Total Samples Tested: {summary_stats['total_samples']}")
    
    # Save results
    save_detailed_results(output_dir, summary_stats, all_results, goal_model_path, reference_file, num_samples)
    save_summary_report(output_dir, summary_stats, all_results, goal_model_path, reference_file, num_samples)
    
    return {
        'summary_stats': summary_stats,
        'detailed_results': all_results,
        'output_dir': output_dir
    }


def main():
    """Main function for running the goal prediction test script."""
    parser = argparse.ArgumentParser(description="Test trained goal inference model")
    parser.add_argument("--goal_model", type=str, required=True,
                       help="Path to trained goal inference model file")
    parser.add_argument("--reference_file", type=str, 
                       default=config.paths.goal_inference_data_dir,
                       help="Path to directory containing ref_traj_sample_*.json files")
    parser.add_argument("--num_samples", type=int, default=config.testing.num_test_samples,
                       help="Number of samples to test")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save test results")
    
    args = parser.parse_args()
    
    # Run goal prediction testing
    results = run_goal_prediction_testing(
        args.goal_model, args.reference_file,
        output_dir=args.output_dir, num_samples=args.num_samples
    )
    
    print(f"\nGoal prediction testing completed successfully!")
    print(f"Results saved to: {results['output_dir']}")
    
    return 0


if __name__ == "__main__":
    exit(main())
