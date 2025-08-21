#!/usr/bin/env python3
"""
Test script for trained PSN model with goal prediction.

This script loads a trained PSN model and tests it on reference trajectory data
to evaluate its performance in both agent selection and goal prediction.

All parameters are loaded from config.yaml for consistency across scripts.
"""

import json
import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Dict, Any, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GPU issues
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import pickle
import os
import sys
from pathlib import Path
import argparse
import signal
import time
import gc
import subprocess

# Add parent directory to path
sys.path.append('..')

# Import configuration loader
from config_loader import load_config, get_device_config, setup_jax_config

# Import the PSN network and related functions
from psn_training_with_reference import (
    PlayerSelectionNetwork, 
    N_agents, T_observation, T_total, state_dim, num_iters,
    extract_observation_trajectory, 
    extract_reference_goals,
    create_masked_game_setup,
    create_loss_functions,
    solve_masked_game
)

# Import Flax for model loading
import flax.serialization


def clear_gpu_memory():
    """Clear GPU memory using nvidia-smi if available."""
    # Try to clear GPU memory using nvidia-smi
    subprocess.run(['nvidia-smi', '--gpu-reset'], 
                  capture_output=True, timeout=10, check=False)
    
    # Also try JAX-specific cleanup
    if hasattr(jax, 'clear_caches'):
        jax.clear_caches()
    
    # Force garbage collection
    gc.collect()


def safe_model_inference(model, trained_state, obs_input, max_retries=3):
    """
    Safely run model inference with retry logic and memory cleanup.
    
    Args:
        model: PSN model
        trained_state: Trained model state
        obs_input: Input data
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (predicted_mask, predicted_goals) or (None, None) if failed
    """
    for attempt in range(max_retries):
        # Clear memory before inference
        clear_gpu_memory()
        
        # Run inference
        predicted_mask, predicted_goals = model.apply({'params': trained_state['params']}, obs_input)
        
        # Convert to numpy immediately to avoid JAX device memory issues
        predicted_mask = np.array(predicted_mask)
        predicted_goals = np.array(predicted_goals)
        
        return predicted_mask, predicted_goals
    
    return None, None


def load_trained_model(model_path: str) -> Tuple[PlayerSelectionNetwork, Any]:
    """
    Load a trained PSN model from file.
    
    Args:
        model_path: Path to the trained model file
        
    Returns:
        Tuple of (model, trained_state)
    """
    print(f"Loading trained model from: {model_path}")
    
    # Load the model bytes
    with open(model_path, 'rb') as f:
        model_bytes = pickle.load(f)
    
    # Create the model
    model = PlayerSelectionNetwork()
    
    # Deserialize the state - use the correct API that matches how it was saved
    # The model was saved using flax.serialization.to_bytes(trained_state)
    # So we need to use from_bytes with the model as the first argument
    trained_state = flax.serialization.from_bytes(model, model_bytes)
    
    print("✓ Model loaded successfully")
    return model, trained_state


def test_model_on_sample(model: PlayerSelectionNetwork, trained_state: Any, 
                        sample_data: Dict[str, Any], ego_agent_id: int = 0, timeout_seconds: int = 30) -> Dict[str, Any]:
    """
    Test the trained model on a single sample with timeout protection.
    
    Args:
        model: PSN model
        trained_state: Trained model state
        sample_data: Reference trajectory sample
        ego_agent_id: ID of the ego agent
        timeout_seconds: Maximum time allowed for computation
        
    Returns:
        Dictionary containing test results
    """
    # Set up timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError("Computation timed out")
    
    # Set signal alarm for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        # Extract observation trajectory
        obs_traj = extract_observation_trajectory(sample_data, ego_agent_id)
        obs_input = obs_traj.flatten().reshape(1, -1)  # Add batch dimension
        
        # Get model predictions
        predicted_mask, predicted_goals = model.apply({'params': trained_state['params']}, obs_input)
        
        # Extract true goals and initial positions
        true_goals = extract_reference_goals(sample_data)
        init_positions = jnp.array(sample_data["init_positions"])
        
        # Reshape predicted goals
        predicted_goals = predicted_goals[0].reshape(N_agents, 2)
        predicted_mask = predicted_mask[0]  # Remove batch dimension
        
        # Compute goal prediction error
        goal_error = jnp.mean(jnp.square(predicted_goals - true_goals))
        goal_rmse = jnp.sqrt(goal_error)
        
        # Compute per-agent goal errors
        per_agent_errors = []
        for i in range(N_agents):
            agent_error = jnp.sqrt(jnp.mean(jnp.square(predicted_goals[i] - true_goals[i])))
            per_agent_errors.append(float(agent_error))
        
        # Analyze mask predictions
        mask_threshold = 0.3
        selected_agents = jnp.where(predicted_mask > mask_threshold)[0]
        num_selected = len(selected_agents)
        mask_sparsity = 1.0 - (num_selected / (N_agents - 1))
        
        # Test masked game solving with predicted goals
        # Create masked game setup
        agents, initial_states, target_positions, mask_values = create_masked_game_setup(
            sample_data, ego_agent_id, predicted_mask, predicted_goals.flatten(), is_training=False)
        
        # Safety check: limit the number of agents to prevent memory issues
        if len(agents) > 5:
            print(f"Warning: Too many agents ({len(agents)}), limiting to 5 for safety")
            agents = agents[:5]
            initial_states = initial_states[:5]
            target_positions = target_positions[:5]
            mask_values = mask_values[:5] if len(mask_values) > 5 else mask_values
        
        # Create loss functions
        loss_functions, linearize_functions, compiled_functions = create_loss_functions(
            agents, mask_values, is_training=False)
        
        # Solve masked game with reduced iterations for safety
        state_trajectories, control_trajectories = solve_masked_game(
            agents, initial_states, target_positions, compiled_functions, mask_values, num_iters=num_iters)
        
        # Extract ego agent trajectory
        ego_traj_masked = state_trajectories[0]
        
        # Debug: Print trajectory lengths
        print(f"  Debug: ego_traj_masked shape: {ego_traj_masked.shape}")
        print(f"  Debug: Expected length from T_obs({T_observation}) to T_total({T_total}): {T_total - T_observation}")
        
        # Ensure the computed trajectory spans the full expected length
        if ego_traj_masked.shape[0] < T_total:
            print(f"  Warning: Computed trajectory length {ego_traj_masked.shape[0]} < expected {T_total}")
            # Pad with the last state if trajectory is too short
            if ego_traj_masked.shape[0] > 0:
                last_state = ego_traj_masked[-1:]
                pad_size = T_total - ego_traj_masked.shape[0]
                padding = jnp.tile(last_state, (pad_size, 1))
                ego_traj_masked = jnp.concatenate([ego_traj_masked, padding], axis=0)
                print(f"  Padded trajectory to length: {ego_traj_masked.shape[0]}")
            else:
                print(f"  Error: Empty computed trajectory, using fallback")
                # Create fallback trajectory
                start_pos = initial_states[0][:2]
                end_pos = target_positions[0]
                fallback_traj = jnp.linspace(start_pos, end_pos, T_total)
                ego_traj_masked = jnp.zeros((T_total, 4))
                ego_traj_masked = ego_traj_masked.at[:, :2].set(fallback_traj)
        
        # Extract reference trajectory for ego agent
        agent_key = f"agent_{ego_agent_id}"
        ref_states = sample_data["trajectories"][agent_key]["states"]
        ego_traj_ref = jnp.array(ref_states)  # Full reference trajectory
        
        print(f"  Debug: Reference trajectory shape: {ego_traj_ref.shape}")
        
        # Compute trajectory similarity using future trajectory (from observation horizon onwards)
        # This matches the training script logic
        future_ref_states = ego_traj_ref[T_observation:]  # Future trajectory after observation
        
        # Limit to the shorter of the two trajectories
        min_length = min(ego_traj_masked.shape[0], future_ref_states.shape[0])
        
        if min_length > 0:
            # Use only the first min_length steps for comparison
            masked_traj_matched = ego_traj_masked[:min_length, :2]  # (min_length, 2) - positions only
            future_ref_matched = future_ref_states[:min_length, :2]  # (min_length, 2) - positions only
            
            trajectory_error = jnp.mean(jnp.square(masked_traj_matched - future_ref_matched))
            trajectory_rmse = jnp.sqrt(trajectory_error)
            game_solved = True
            print(f"  Debug: Trajectory comparison length: {min_length}")
        else:
            trajectory_rmse = float('inf')
            game_solved = False
            print(f"  Error: No valid trajectory comparison possible")
            state_trajectories = None
        
        # Store computed trajectories if available
        computed_trajectories = None
        if state_trajectories is not None:
            computed_trajectories = [traj.tolist() for traj in state_trajectories]
        
        # Clear timeout
        signal.alarm(0)
        
        results = {
            'goal_prediction_rmse': float(goal_rmse),
            'goal_prediction_error': float(goal_error),
            'per_agent_errors': per_agent_errors,
            'mask_sparsity': float(mask_sparsity),
            'num_selected_agents': int(num_selected),
            'selected_agent_ids': [int(i) for i in selected_agents],
            'trajectory_rmse': float(trajectory_rmse),
            'game_solved': game_solved,
            'predicted_mask': predicted_mask.tolist(),
            'predicted_goals': predicted_goals.tolist(),
            'true_goals': true_goals.tolist(),
            'init_positions': init_positions.tolist(),
            'computed_trajectories': computed_trajectories
        }
        
        return results
        
    except TimeoutError:
        print(f"Warning: Computation timed out after {timeout_seconds} seconds")
        signal.alarm(0)  # Clear timeout
        # Return fallback results
        return {
            'goal_prediction_rmse': float('inf'),
            'goal_prediction_error': float('inf'),
            'per_agent_errors': [float('inf')] * N_agents,
            'mask_sparsity': 0.0,
            'num_selected_agents': 0,
            'selected_agent_ids': [],
            'trajectory_rmse': float('inf'),
            'game_solved': False,
            'predicted_mask': [0.0] * (N_agents - 1),
            'predicted_goals': [[0.0, 0.0]] * N_agents,
            'true_goals': [[0.0, 0.0]] * N_agents,
            'init_positions': [[0.0, 0.0]] * N_agents,
            'computed_trajectories': None,
            'error': 'timeout'
        }
        
    except Exception as e:
        print(f"Warning: Unexpected error during testing: {e}")
        signal.alarm(0)  # Clear timeout
        # Return fallback results
        return {
            'goal_prediction_rmse': float('inf'),
            'goal_prediction_error': float('inf'),
            'per_agent_errors': [float('inf')] * N_agents,
            'mask_sparsity': 0.0,
            'num_selected_agents': 0,
            'selected_agent_ids': [],
            'trajectory_rmse': float('inf'),
            'game_solved': False,
            'predicted_mask': [0.0] * (N_agents - 1),
            'predicted_goals': [[0.0, 0.0]] * N_agents,
            'true_goals': [[0.0, 0.0]] * N_agents,
            'init_positions': [[0.0, 0.0]] * N_agents,
            'computed_trajectories': None,
            'error': str(e)
        }


def test_model_on_sample_simple(model: PlayerSelectionNetwork, trained_state: Any, 
                               sample_data: Dict[str, Any], ego_agent_id: int = 0) -> Dict[str, Any]:
    """
    Simple test that only does model inference without game solving to avoid crashes.
    
    Args:
        model: PSN model
        trained_state: Trained model state
        sample_data: Reference trajectory sample
        ego_agent_id: ID of the ego agent
        
    Returns:
        Dictionary containing basic test results
    """
    # Extract observation trajectory
    obs_traj = extract_observation_trajectory(sample_data, ego_agent_id)
    obs_input = obs_traj.flatten().reshape(1, -1)  # Add batch dimension
    
    # Get model predictions with safety measures
    predicted_mask, predicted_goals = safe_model_inference(model, trained_state, obs_input)
    
    if predicted_mask is None or predicted_goals is None:
        print(f"  Warning: Model inference failed")
        return {
            'goal_prediction_rmse': float('inf'),
            'goal_prediction_error': float('inf'),
            'per_agent_errors': [float('inf')] * N_agents,
            'mask_sparsity': 0.0,
            'num_selected_agents': 0,
            'selected_agent_ids': [],
            'trajectory_rmse': float('inf'),
            'game_solved': False,
            'predicted_mask': [0.0] * (N_agents - 1),
            'predicted_goals': [[0.0, 0.0]] * N_agents,
            'true_goals': [[0.0, 0.0]] * N_agents,
            'init_positions': [[0.0, 0.0]] * N_agents,
            'computed_trajectories': None,
            'error': 'inference_failed',
            'mode': 'simple'
        }
    
    # Extract true goals and initial positions
    true_goals = extract_reference_goals(sample_data)
    init_positions = jnp.array(sample_data["init_positions"])
    
    # Reshape predicted goals
    predicted_goals = predicted_goals[0].reshape(N_agents, 2)
    predicted_mask = predicted_mask[0]  # Remove batch dimension
    
    # Compute goal prediction error
    goal_error = jnp.mean(jnp.square(predicted_goals - true_goals))
    goal_rmse = jnp.sqrt(goal_error)
    
    # Compute per-agent goal errors
    per_agent_errors = []
    for i in range(N_agents):
        agent_error = jnp.sqrt(jnp.mean(jnp.square(predicted_goals[i] - true_goals[i])))
        per_agent_errors.append(float(agent_error))
    
    # Analyze mask predictions
    mask_threshold = 0.5
    selected_agents = jnp.where(predicted_mask > mask_threshold)[0]
    num_selected = len(selected_agents)
    mask_sparsity = 1.0 - (num_selected / (N_agents - 1))
    
    results = {
        'goal_prediction_rmse': float(goal_rmse),
        'goal_prediction_error': float(goal_error),
        'per_agent_errors': per_agent_errors,
        'mask_sparsity': float(mask_sparsity),
        'num_selected_agents': int(num_selected),
        'selected_agent_ids': [int(i) for i in selected_agents],
        'trajectory_rmse': float('inf'),  # Not computed in simple mode
        'game_solved': False,  # Not computed in simple mode
        'predicted_mask': predicted_mask.tolist(),
        'predicted_goals': predicted_goals.tolist(),
        'true_goals': true_goals.tolist(),
        'init_positions': init_positions.tolist(),
        'computed_trajectories': None,
        'mode': 'simple'
    }
    
    return results


def visualize_sample_results(sample_data: Dict[str, Any], results: Dict[str, Any], 
                           sample_id: int, save_dir: str = None) -> None:
    """
    Visualize test results for a single sample using CPU-only mode.
    
    Args:
        sample_data: Reference trajectory sample
        results: Test results dictionary
        sample_id: Sample identifier
        save_dir: Directory to save plots
    """
    # Force CPU-only mode for matplotlib to avoid GPU issues
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    # Set matplotlib to use CPU
    plt.ioff()  # Turn off interactive mode
    
    # Extract data
    predicted_mask = np.array(results['predicted_mask'])
    predicted_goals = np.array(results['predicted_goals'])
    true_goals = np.array(results['true_goals'])
    init_positions = np.array(results['init_positions'])
    
    # Create single comprehensive visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title(f'Sample {sample_id+1}: Goals, Trajectories, and Ego Agent Selection\n'
                 f'Goal RMSE: {results["goal_prediction_rmse"]:.4f}, '
                 f'Mask Sparsity: {results["mask_sparsity"]:.2f}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True, alpha=0.3)
    
    # Color palette for agents
    colors = plt.cm.tab10(np.linspace(0, 1, N_agents))
    
    # Plot initial positions
    for j in range(N_agents):
        ax.plot(init_positions[j][0], init_positions[j][1], 'o', 
                 color=colors[j], markersize=10, alpha=0.7, label=f'Agent {j} Start')
    
    # Plot true goals (ground truth goals)
    for j in range(N_agents):
        ax.plot(true_goals[j][0], true_goals[j][1], 's', 
                 color=colors[j], markersize=12, alpha=0.8, label=f'Agent {j} True Goal')
    
    # Plot predicted goals
    for j in range(N_agents):
        ax.plot(predicted_goals[j][0], predicted_goals[j][1], '^', 
                 color=colors[j], markersize=12, alpha=0.8, label=f'Agent {j} Predicted Goal')
    
    # Plot ground truth trajectories for all agents (dashed lines)
    ego_agent_id = 0
    for j in range(N_agents):
        agent_key = f"agent_{j}"
        if agent_key in sample_data["trajectories"]:
            states = np.array(sample_data["trajectories"][agent_key]["states"])
            positions = states[:, :2]  # Extract x, y positions
            
            # All ground truth trajectories are dashed lines
            line_style = '--'
            line_width = 2
            alpha = 0.6
            label_suffix = ' (Ground Truth)'
            
            ax.plot(positions[:, 0], positions[:, 1], 
                    line_style, color=colors[j], linewidth=line_width, alpha=alpha,
                    label=f'Agent {j}{label_suffix}')
    
    # Plot computed trajectory from masked game (only ego agent)
    if results.get('computed_trajectories') is not None:
        computed_trajs = results['computed_trajectories']
        if len(computed_trajs) > 0 and computed_trajs[0] is not None:
            # Only plot ego agent's computed trajectory (index 0)
            # This shows the trajectory computed by solving the masked game with PSN-selected agents
            ego_traj = computed_trajs[0]
            traj_array = np.array(ego_traj)
            if traj_array.size > 0:
                ax.plot(traj_array[:, 0], traj_array[:, 1], 
                        color=colors[ego_agent_id], linewidth=4, linestyle='-', alpha=0.9, 
                        label=f'Agent {ego_agent_id} Computed (Masked Game)')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plot if save_dir is provided
    if save_dir:
        plot_path = os.path.join(save_dir, f"test_results_sample_{sample_id:03d}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Test results plot saved to: {plot_path}")
    
    # Always close the figure to free memory
    plt.close(fig)


def run_comprehensive_test(model_path: str, reference_data: List[Dict[str, Any]], 
                         num_test_samples: int = 10, save_dir: str = None) -> Dict[str, Any]:
    """
    Run comprehensive testing on multiple samples with enhanced safety measures.
    
    Args:
        model_path: Path to the trained model
        reference_data: Reference trajectory data
        num_test_samples: Number of samples to test
        save_dir: Directory to save results
        
    Returns:
        Comprehensive test results
    """
    print(f"Running comprehensive test on {num_test_samples} samples...")
    
    # Load the trained model
    model, trained_state = load_trained_model(model_path)
    
    # Set up save directory
    if save_dir is None:
        save_dir = os.path.dirname(model_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # Sample random indices
    rng = jax.random.PRNGKey(42)
    rng, sample_key = jax.random.split(rng)
    sample_indices = jax.random.choice(sample_key, len(reference_data), shape=(num_test_samples,), replace=False)
    
    # Test results storage
    all_results = []
    goal_errors = []
    trajectory_errors = []
    mask_sparsities = []
    num_selected_list = []
    
    # Test each sample with enhanced safety
    for i, idx in enumerate(sample_indices):
        print(f"\nTesting sample {i+1}/{num_test_samples} (index {idx})...")
        
        # Clear GPU memory before each test
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
        gc.collect()
        
        sample_data = reference_data[idx]
        
        # Test with timeout and enhanced error handling
        results = test_model_on_sample(model, trained_state, sample_data, timeout_seconds=60)
        all_results.append(results)
        
        # Collect metrics
        goal_errors.append(results['goal_prediction_rmse'])
        if results['game_solved']:
            trajectory_errors.append(results['trajectory_rmse'])
        mask_sparsities.append(results['mask_sparsity'])
        num_selected_list.append(results['num_selected_agents'])
        
        # Check if there was an error
        if 'error' in results:
            print(f"  Sample had error: {results['error']}")
            continue
        
        # Visualize results only if successful
        visualize_sample_results(sample_data, results, i, save_dir)
        
        # Print summary for this sample
        print(f"  Goal Prediction RMSE: {results['goal_prediction_rmse']:.4f}")
        print(f"  Mask Sparsity: {results['mask_sparsity']:.2f}")
        print(f"  Selected Agents: {results['num_selected_agents']}")
        if results['game_solved']:
            print(f"  Trajectory RMSE: {results['trajectory_rmse']:.4f}")
        else:
            print(f"  Trajectory: Failed to solve")
        
        # Force garbage collection after each sample
        gc.collect()
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
    
    # Compute aggregate statistics
    goal_errors = np.array(goal_errors)
    trajectory_errors = np.array(trajectory_errors)
    mask_sparsities = np.array(mask_sparsities)
    num_selected_list = np.array(num_selected_list)
    
    comprehensive_results = {
        'num_samples_tested': num_test_samples,
        'goal_prediction': {
            'mean_rmse': float(np.mean(goal_errors)),
            'std_rmse': float(np.std(goal_errors)),
            'min_rmse': float(np.min(goal_errors)),
            'max_rmse': float(np.max(goal_errors)),
            'all_errors': goal_errors.tolist()
        },
        'trajectory_similarity': {
            'mean_rmse': float(np.mean(trajectory_errors)) if len(trajectory_errors) > 0 else float('inf'),
            'std_rmse': float(np.std(trajectory_errors)) if len(trajectory_errors) > 0 else float('inf'),
            'success_rate': len(trajectory_errors) / num_test_samples,
            'all_errors': trajectory_errors.tolist()
        },
        'agent_selection': {
            'mean_sparsity': float(np.mean(mask_sparsities)),
            'std_sparsity': float(np.std(mask_sparsities)),
            'mean_selected': float(np.mean(num_selected_list)),
            'std_selected': float(np.std(num_selected_list)),
            'all_sparsities': mask_sparsities.tolist(),
            'all_selected_counts': num_selected_list.tolist()
        },
        'detailed_results': all_results
    }
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST RESULTS")
    print("="*60)
    print(f"Goal Prediction RMSE: {comprehensive_results['goal_prediction']['mean_rmse']:.4f} ± {comprehensive_results['goal_prediction']['std_rmse']:.4f}")
    print(f"Trajectory Similarity RMSE: {comprehensive_results['trajectory_similarity']['mean_rmse']:.4f} ± {comprehensive_results['trajectory_similarity']['std_rmse']:.4f}")
    print(f"Trajectory Success Rate: {comprehensive_results['trajectory_similarity']['success_rate']:.2%}")
    print(f"Agent Selection Sparsity: {comprehensive_results['agent_selection']['mean_sparsity']:.2f} ± {comprehensive_results['agent_selection']['std_sparsity']:.2f}")
    print(f"Average Selected Agents: {comprehensive_results['agent_selection']['mean_selected']:.1f} ± {comprehensive_results['agent_selection']['std_selected']:.1f}")
    print("="*60)
    
    return comprehensive_results


def run_simple_test(model_path: str, reference_data: List[Dict[str, Any]], 
                    num_test_samples: int = 10, save_dir: str = None) -> Dict[str, Any]:
    """
    Run simple test on multiple samples without game solving.
    
    Args:
        model_path: Path to the trained model
        reference_data: Reference trajectory data
        num_test_samples: Number of samples to test
        save_dir: Directory to save results
        
    Returns:
        Simple test results
    """
    print(f"Running simple test on {num_test_samples} samples...")
    
    # Load the trained model
    model, trained_state = load_trained_model(model_path)
    
    # Set up save directory
    if save_dir is None:
        save_dir = os.path.dirname(model_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # Sample random indices
    rng = jax.random.PRNGKey(42)
    rng, sample_key = jax.random.split(rng)
    sample_indices = jax.random.choice(sample_key, len(reference_data), shape=(num_test_samples,), replace=False)
    
    # Test results storage
    all_results = []
    goal_errors = []
    mask_sparsities = []
    num_selected_list = []
    
    # Test each sample with enhanced safety
    for i, idx in enumerate(sample_indices):
        print(f"\nTesting sample {i+1}/{num_test_samples} (index {idx})...")
        
        # Clear GPU memory before each test
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
        gc.collect()
        
        sample_data = reference_data[idx]
        
        # Test with enhanced error handling
        results = test_model_on_sample_simple(model, trained_state, sample_data)
        all_results.append(results)
        
        # Collect metrics
        goal_errors.append(results['goal_prediction_rmse'])
        mask_sparsities.append(results['mask_sparsity'])
        num_selected_list.append(results['num_selected_agents'])
        
        # Check if there was an error
        if 'error' in results:
            print(f"  Sample had error: {results['error']}")
            continue
        
        # Visualize results only if successful
        visualize_sample_results(sample_data, results, i, save_dir)
        
        # Print summary for this sample
        print(f"  Goal Prediction RMSE: {results['goal_prediction_rmse']:.4f}")
        print(f"  Mask Sparsity: {results['mask_sparsity']:.2f}")
        print(f"  Selected Agents: {results['num_selected_agents']}")
        print(f"  Trajectory: Not computed in simple mode")
        
        # Force garbage collection after each sample
        gc.collect()
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
    
    # Compute aggregate statistics
    goal_errors = np.array(goal_errors)
    mask_sparsities = np.array(mask_sparsities)
    num_selected_list = np.array(num_selected_list)
    
    simple_results = {
        'num_samples_tested': num_test_samples,
        'goal_prediction': {
            'mean_rmse': float(np.mean(goal_errors)),
            'std_rmse': float(np.std(goal_errors)),
            'min_rmse': float(np.min(goal_errors)),
            'max_rmse': float(np.max(goal_errors)),
            'all_errors': goal_errors.tolist()
        },
        'agent_selection': {
            'mean_sparsity': float(np.mean(mask_sparsities)),
            'std_sparsity': float(np.std(mask_sparsities)),
            'mean_selected': float(np.mean(num_selected_list)),
            'std_selected': float(np.std(num_selected_list)),
            'all_sparsities': mask_sparsities.tolist(),
            'all_selected_counts': num_selected_list.tolist()
        },
        'detailed_results': all_results
    }
    
    # Print simple summary
    print("\n" + "="*60)
    print("SIMPLE TEST RESULTS")
    print("="*60)
    print(f"Goal Prediction RMSE: {simple_results['goal_prediction']['mean_rmse']:.4f} ± {simple_results['goal_prediction']['std_rmse']:.4f}")
    print(f"Agent Selection Sparsity: {simple_results['agent_selection']['mean_sparsity']:.2f} ± {simple_results['agent_selection']['std_sparsity']:.2f}")
    print(f"Average Selected Agents: {simple_results['agent_selection']['mean_selected']:.1f} ± {simple_results['agent_selection']['std_selected']:.1f}")
    print("="*60)
    
    return simple_results


def run_minimal_test(model_path: str, reference_data: List[Dict[str, Any]], 
                    num_test_samples: int = 10, save_dir: str = None) -> Dict[str, Any]:
    """
    Run minimal test on multiple samples without visualization or complex operations.
    
    Args:
        model_path: Path to the trained model
        reference_data: Reference trajectory data
        num_test_samples: Number of samples to test
        save_dir: Directory to save results
        
    Returns:
        Minimal test results
    """
    print(f"Running minimal test on {num_test_samples} samples...")
    
    # Load the trained model
    model, trained_state = load_trained_model(model_path)
    
    # Set up save directory
    if save_dir is None:
        save_dir = os.path.dirname(model_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # Sample random indices
    rng = jax.random.PRNGKey(42)
    rng, sample_key = jax.random.split(rng)
    sample_indices = jax.random.choice(sample_key, len(reference_data), shape=(num_test_samples,), replace=False)
    
    # Test results storage
    all_results = []
    goal_errors = []
    mask_sparsities = []
    num_selected_list = []
    
    # Test each sample with minimal operations
    for i, idx in enumerate(sample_indices):
        print(f"\nTesting sample {i+1}/{num_test_samples} (index {idx})...")
        
        # Clear memory before each test
        clear_gpu_memory()
        
        sample_data = reference_data[idx]
        
        # Extract observation trajectory
        obs_traj = extract_observation_trajectory(sample_data, 0)  # ego_agent_id = 0
        obs_input = obs_traj.flatten().reshape(1, -1)
        
        # Get model predictions with safety measures
        predicted_mask, predicted_goals = safe_model_inference(model, trained_state, obs_input)
        
        if predicted_mask is None or predicted_goals is None:
            print(f"  Warning: Model inference failed")
            all_results.append({
                'goal_prediction_rmse': float('inf'),
                'mask_sparsity': 0.0,
                'num_selected_agents': 0,
                'error': 'inference_failed'
            })
            goal_errors.append(float('inf'))
            mask_sparsities.append(0.0)
            num_selected_list.append(0)
            continue
        
        # Extract true goals
        true_goals = extract_reference_goals(sample_data)
        
        # Reshape predicted goals
        predicted_goals = predicted_goals[0].reshape(N_agents, 2)
        predicted_mask = predicted_mask[0]
        
        # Compute basic metrics
        goal_error = jnp.mean(jnp.square(predicted_goals - true_goals))
        goal_rmse = jnp.sqrt(goal_error)
        
        # Analyze mask predictions
        mask_threshold = 0.5
        selected_agents = jnp.where(predicted_mask > mask_threshold)[0]
        num_selected = len(selected_agents)
        mask_sparsity = 1.0 - (num_selected / (N_agents - 1))
        
        # Store results
        results = {
            'goal_prediction_rmse': float(goal_rmse),
            'mask_sparsity': float(mask_sparsity),
            'num_selected_agents': int(num_selected),
            'error': None
        }
        
        all_results.append(results)
        goal_errors.append(float(goal_rmse))
        mask_sparsities.append(float(mask_sparsity))
        num_selected_list.append(int(num_selected))
        
        # Print summary for this sample
        print(f"  Goal Prediction RMSE: {goal_rmse:.4f}")
        print(f"  Mask Sparsity: {mask_sparsity:.2f}")
        print(f"  Selected Agents: {num_selected}")
        
        # Force garbage collection after each sample
        gc.collect()
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
    
    # Compute aggregate statistics
    goal_errors = np.array(goal_errors)
    mask_sparsities = np.array(mask_sparsities)
    num_selected_list = np.array(num_selected_list)
    
    minimal_results = {
        'num_samples_tested': num_test_samples,
        'goal_prediction': {
            'mean_rmse': float(np.mean(goal_errors)),
            'std_rmse': float(np.std(goal_errors)),
            'min_rmse': float(np.min(goal_errors)),
            'max_rmse': float(np.max(goal_errors)),
            'all_errors': goal_errors.tolist()
        },
        'agent_selection': {
            'mean_sparsity': float(np.mean(mask_sparsities)),
            'std_sparsity': float(np.std(mask_sparsities)),
            'mean_selected': float(np.mean(num_selected_list)),
            'std_selected': float(np.std(num_selected_list)),
            'all_sparsities': mask_sparsities.tolist(),
            'all_selected_counts': num_selected_list.tolist()
        },
        'detailed_results': all_results
    }
    
    # Print minimal summary
    print("\n" + "="*60)
    print("MINIMAL TEST RESULTS")
    print("="*60)
    print(f"Goal Prediction RMSE: {minimal_results['goal_prediction']['mean_rmse']:.4f} ± {minimal_results['goal_prediction']['std_rmse']:.4f}")
    print(f"Agent Selection Sparsity: {minimal_results['agent_selection']['mean_sparsity']:.2f} ± {minimal_results['agent_selection']['std_sparsity']:.2f}")
    print(f"Average Selected Agents: {minimal_results['agent_selection']['mean_selected']:.1f} ± {minimal_results['agent_selection']['std_selected']:.1f}")
    print("="*60)
    
    return minimal_results


def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description='Test trained PSN model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file')
    parser.add_argument('--reference_file', type=str, 
                       default='reference_trajectories_4p/all_reference_trajectories.json',
                       help='Path to reference trajectory file')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to test')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save results (optional)')
    parser.add_argument('--simple_mode', action='store_true',
                       help='Use simple mode (no game solving) to avoid crashes')
    parser.add_argument('--cpu_only', action='store_true',
                       help='Force CPU-only mode to avoid GPU crashes')
    parser.add_argument('--minimal_mode', action='store_true',
                       help='Minimal mode: no visualization, just basic metrics')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # Set device based on arguments
    if args.cpu_only:
        print("Forcing CPU-only mode to avoid GPU crashes...")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        jax.config.update('jax_platform_name', 'cpu')
        print("✓ CPU-only mode activated")
    else:
        print(f"Using device: {jax.devices()[0]}")
    
    # Load reference data
    print(f"Loading reference data from: {args.reference_file}")
    with open(args.reference_file, 'r') as f:
        reference_data = json.load(f)
    print(f"Loaded {len(reference_data)} reference samples")
    
    # Set up save directory
    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.model_path)
    print(f"Results will be saved to: {args.save_dir}")
    
    # Run test based on mode
    if args.simple_mode:
        print("Running in SIMPLE MODE (no game solving) to avoid crashes...")
        results = run_simple_test(args.model_path, reference_data, args.num_samples, args.save_dir)
    elif args.minimal_mode:
        print("Running in MINIMAL MODE (no visualization) to isolate segmentation fault...")
        results = run_minimal_test(args.model_path, reference_data, args.num_samples, args.save_dir)
    else:
        print("Running in FULL MODE with game solving...")
        results = run_comprehensive_test(args.model_path, reference_data, args.num_samples, args.save_dir)
    
    # Save results
    results_file = os.path.join(args.save_dir, "test_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nTest results saved to: {results_file}")


if __name__ == "__main__":
    main()

