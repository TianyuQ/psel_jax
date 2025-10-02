#!/usr/bin/env python3
"""
Test PSN with Pedestrian Data

This script tests the Player Selection Network (PSN) using pedestrian trajectory data
from the data_ped directory. It performs prediction_test with true goals only.

Author: Assistant
Date: 2024
"""

import json
import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Dict, Tuple, Any
import time
from pathlib import Path
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration loader and models
from config_loader import load_config, get_device_config, setup_jax_config

# Import model classes
from psn_training_with_pretrained_goals import (
    PlayerSelectionNetwork, GoalInferenceNetwork, load_trained_models, load_pretrained_goal_model
)

# Import baselines
from player_selection_network.baselines import baseline_selection

# Import from the main lqrax module
from lqrax import iLQR

# Import game solving functions from the main test script
from test_psn_receding_horizon import (
    solve_receding_horizon_game, create_agent_setup, create_loss_functions
)

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

# ============================================================================
# DATA CONVERSION FUNCTIONS
# ============================================================================

def convert_pedestrian_data_to_test_format(json_file: str, csv_file: str, sample_id: int) -> Dict[str, Any]:
    """
    Convert pedestrian data from JSON format to the expected test format.
    Everything is extracted from the JSON file: trajectories, initial states, and goals.
    
    Args:
        json_file: Path to JSON file with trajectory data, initial states, and goals
        csv_file: Path to CSV file (not used, kept for compatibility)
        sample_id: Sample ID for this data
    
    Returns:
        Dictionary in the expected test format
    """
    # Load JSON data
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    # Count number of agents from JSON file
    trajectory_keys = [k for k in json_data.keys() if k.endswith(' Trajectory')]
    n_agents = len(trajectory_keys)
    
    print(f"      Found {n_agents} agents in sample {sample_id} (from JSON)")
    
    # Extract trajectories, initial states, and goals for all players
    trajectories = {}
    initial_states = []
    target_positions = []
    
    for i in range(1, n_agents + 1):  # Player 1, Player 2, etc.
        # Extract trajectory
        traj_key = f"Player {i} Trajectory"
        if traj_key in json_data:
            traj_data = json_data[traj_key]
            traj_array = np.array(traj_data)  # Shape: (T, 4) - [x, y, vx, vy]
            
            # Store as agent_{i-1} (0-indexed)
            agent_key = f"agent_{i-1}"
            trajectories[agent_key] = {"states": traj_array.tolist()}
        else:
            print(f"      Warning: Missing trajectory for Player {i}")
            continue
        
        # Extract initial state from JSON
        initial_key = f"Player {i} Initial State"
        if initial_key in json_data:
            initial_data = json_data[initial_key]
            initial_state = [float(initial_data[0]), float(initial_data[1]), 
                           float(initial_data[2]), float(initial_data[3])]
            initial_states.append(initial_state)
        else:
            print(f"      Warning: Missing initial state for Player {i}")
            # Use first trajectory point as initial state
            if len(traj_array) > 0:
                initial_state = traj_array[0].tolist()
                initial_states.append(initial_state)
            else:
                print(f"      Error: No trajectory data for Player {i}")
                continue
        
        # Extract goal from JSON - use state at step 50 as goal
        goal_key = f"Player {i} Goal"
        if goal_key in json_data:
            goal_data = json_data[goal_key]
            # Use step 50 as goal instead of the final goal
            if len(traj_array) > 50:
                target_pos = traj_array[50][:2].tolist()  # Use step 50 position as goal
                target_positions.append(target_pos)
            else:
                # If trajectory is shorter than 50 steps, use the final position
                target_pos = traj_array[-1][:2].tolist()
                target_positions.append(target_pos)
                print(f"      Warning: Player {i} trajectory shorter than 50 steps, using final position as goal")
        else:
            print(f"      Warning: Missing goal for Player {i}")
            # Use step 50 trajectory point as goal
            if len(traj_array) > 50:
                target_pos = traj_array[50][:2].tolist()  # Use step 50 position as goal
                target_positions.append(target_pos)
            elif len(traj_array) > 0:
                target_pos = traj_array[-1][:2].tolist()  # Fallback to final position
                target_positions.append(target_pos)
            else:
                print(f"      Error: No trajectory data for Player {i}")
                continue
    
    # Create sample data in expected format
    sample_data = {
        "sample_id": sample_id,
        "trajectories": trajectories,
        "target_positions": target_positions,
        "initial_states": initial_states,
        "n_agents": n_agents  # Store actual number of agents
    }
    
    return sample_data

def load_pedestrian_data(data_dir: str, num_samples: int = None) -> List[Dict[str, Any]]:
    """
    Load pedestrian data from data_ped directory.
    
    Args:
        data_dir: Path to data_ped directory
        num_samples: Number of samples to load (None for all)
    
    Returns:
        List of sample data dictionaries
    """
    samples = []
    
    # Find all JSON files
    json_files = sorted([f for f in os.listdir(data_dir) if f.startswith('original_trajectories_') and f.endswith('.json')])
    csv_files = sorted([f for f in os.listdir(data_dir) if f.startswith('scenario') and f.endswith('.csv')])
    
    if len(json_files) != len(csv_files):
        raise ValueError(f"Mismatch between JSON files ({len(json_files)}) and CSV files ({len(csv_files)})")
    
    # Limit number of samples if specified
    if num_samples is not None:
        json_files = json_files[:num_samples]
        csv_files = csv_files[:num_samples]
    
    for i, (json_file, csv_file) in enumerate(zip(json_files, csv_files)):
        json_path = os.path.join(data_dir, json_file)
        csv_path = os.path.join(data_dir, csv_file)
        
        print(f"Loading pedestrian data: {json_file} + {csv_file}")
        
        sample_data = convert_pedestrian_data_to_test_format(json_path, csv_path, i)
        samples.append(sample_data)
    
    print(f"Loaded {len(samples)} pedestrian samples")
    return samples

# ============================================================================
# PLAYER SELECTION UTILITIES (from test_psn_receding_horizon.py)
# ============================================================================

def select_agents_by_mask(predicted_mask: jnp.ndarray, 
                         selection_method: str = "threshold",
                         mask_threshold: float = 0.5,
                         rank: int = 3) -> Tuple[jnp.ndarray, int, float]:
    """Select agents based on predicted mask using either threshold or rank method."""
    n_other_agents = len(predicted_mask)
    
    if selection_method == "threshold":
        selected_agents = jnp.where(predicted_mask > mask_threshold)[0]
        num_selected = len(selected_agents)
        mask_sparsity = num_selected / n_other_agents
    elif selection_method == "rank":
        num_to_select = max(0, min(rank - 1, n_other_agents))
        if num_to_select == 0:
            selected_agents = jnp.array([])
        else:
            top_indices = jnp.argsort(predicted_mask)[-num_to_select:]
            selected_agents = top_indices
        num_selected = len(selected_agents)
        mask_sparsity = num_selected / n_other_agents
    else:
        raise ValueError(f"Invalid selection_method: {selection_method}. Must be 'threshold' or 'rank'")
    
    return selected_agents, num_selected, mask_sparsity

# ============================================================================
# AGENT DEFINITIONS (from test_psn_receding_horizon.py)
# ============================================================================

class PointAgent(iLQR):
    """Point mass agent for trajectory optimization."""
    def __init__(self, dt, x_dim, u_dim, Q, R):
        super().__init__(dt, x_dim, u_dim, Q, R)
    
    def dyn(self, xt, ut):
        """Dynamics function for point mass."""
        return jnp.array([
            xt[2],  # dx/dt = vx
            xt[3],  # dy/dt = vy
            ut[0],  # dvx/dt = ax
            ut[1]   # dvy/dt = ay
        ])

# ============================================================================
# METRICS COMPUTATION (from test_psn_receding_horizon.py)
# ============================================================================

def compute_ade_fde(predicted_trajectory: jnp.ndarray, ground_truth_trajectory: jnp.ndarray) -> Tuple[float, float]:
    """Compute Average Displacement Error (ADE) and Final Displacement Error (FDE)."""
    min_length = min(len(predicted_trajectory), len(ground_truth_trajectory))
    pred_traj = predicted_trajectory[:min_length, :2]  # Only position (x, y)
    gt_traj = ground_truth_trajectory[:min_length, :2]  # Only position (x, y)
    
    displacement_errors = jnp.linalg.norm(pred_traj - gt_traj, axis=1)
    ade = jnp.mean(displacement_errors)
    fde = displacement_errors[-1]
    
    return float(ade), float(fde)

def compute_planning_metrics(ego_trajectory: jnp.ndarray, 
                           other_trajectories: List[jnp.ndarray],
                           ego_controls: jnp.ndarray,
                           ego_goals: jnp.ndarray,
                           dt: float) -> Dict[str, float]:
    """Compute planning metrics: navigation cost, safety cost, control cost, trajectory length, and trajectory smoothness."""
    T = len(ego_trajectory)
    
    # Navigation cost: distance to goal
    ego_positions = ego_trajectory[:, :2]  # (T, 2)
    goal_positions = jnp.tile(ego_goals, (T, 1))  # (T, 2)
    navigation_errors = jnp.linalg.norm(ego_positions - goal_positions, axis=1)
    navigation_cost = jnp.sum(navigation_errors) * dt
    
    # Safety cost: collision avoidance with other agents
    collision_weight = config.optimization.collision_weight
    collision_scale = config.optimization.collision_scale
    safety_cost = 0.0
    
    for other_traj in other_trajectories:
        if len(other_traj) >= T:
            other_positions = other_traj[:T, :2]  # (T, 2)
            distances = jnp.linalg.norm(ego_positions - other_positions, axis=1)
            safety_penalties = collision_weight * jnp.exp(-collision_scale * distances)
            safety_cost += jnp.sum(safety_penalties) * dt
    
    # Control cost: control effort
    ctrl_weight = config.optimization.control_weight
    control_magnitudes = jnp.linalg.norm(ego_controls, axis=1)
    control_cost = ctrl_weight * jnp.sum(control_magnitudes) * dt
    
    # Trajectory length: total distance traveled along the trajectory
    if T > 1:
        position_diffs = ego_positions[1:] - ego_positions[:-1]  # (T-1, 2)
        segment_lengths = jnp.linalg.norm(position_diffs, axis=1)  # (T-1,)
        trajectory_length = jnp.sum(segment_lengths)
    else:
        trajectory_length = 0.0
    
    # Trajectory smoothness: measure of trajectory orientation changes
    if T > 2:
        direction_vectors = ego_positions[1:] - ego_positions[:-1]  # (T-1, 2)
        angles = []
        for i in range(len(direction_vectors) - 1):
            v1 = direction_vectors[i]
            v2 = direction_vectors[i + 1]
            v1_norm = jnp.linalg.norm(v1)
            v2_norm = jnp.linalg.norm(v2)
            
            if v1_norm > 1e-8 and v2_norm > 1e-8:
                cos_angle = jnp.dot(v1, v2) / (v1_norm * v2_norm)
                cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
                angle = jnp.arccos(cos_angle)
                angles.append(angle)
            else:
                angles.append(0.0)
        
        if angles:
            trajectory_smoothness = jnp.mean(jnp.array(angles))
        else:
            trajectory_smoothness = 0.0
    else:
        trajectory_smoothness = 0.0
    
    return {
        'navigation_cost': float(navigation_cost),
        'safety_cost': float(safety_cost),
        'control_cost': float(control_cost),
        'trajectory_length': float(trajectory_length),
        'trajectory_smoothness': float(trajectory_smoothness)
    }

# ============================================================================
# UTILITY FUNCTIONS (from test_psn_receding_horizon.py)
# ============================================================================

def extract_observation_trajectory(sample_data: Dict[str, Any], obs_input_type: str = "full", num_agents: int = None) -> jnp.ndarray:
    """Extract observation trajectory (first 10 steps) for all agents."""
    if num_agents is None:
        num_agents = config.game.N_agents
    
    if obs_input_type == "partial":
        output_dim = 2  # Only position (x, y)
    else:  # "full"
        output_dim = 4  # Full state (x, y, vx, vy)
    
    T_observation = config.game.T_observation
    observation_trajectory = jnp.zeros((T_observation, num_agents, output_dim))
    
    for i in range(num_agents):
        agent_key = f"agent_{i}"
        agent_states = sample_data["trajectories"][agent_key]["states"]
        
        if len(agent_states) >= T_observation:
            agent_states_array = jnp.array(agent_states[:T_observation])
        else:
            agent_states_padded = agent_states[:]
            last_state = agent_states[-1] if agent_states else [0.0, 0.0, 0.0, 0.0]
            while len(agent_states_padded) < T_observation:
                agent_states_padded.append(last_state)
            agent_states_array = jnp.array(agent_states_padded[:T_observation])
        
        if obs_input_type == "partial":
            agent_obs = agent_states_array[:, :2]  # (T_observation, 2)
        else:  # "full"
            agent_obs = agent_states_array[:, :4]  # (T_observation, 4)
        
        observation_trajectory = observation_trajectory.at[:, i, :].set(agent_obs)
    
    return observation_trajectory

def extract_reference_goals(sample_data: Dict[str, Any], num_agents: int = None) -> jnp.ndarray:
    """Extract reference goals from sample data."""
    if num_agents is None:
        num_agents = config.game.N_agents
    
    if 'target_positions' in sample_data:
        return jnp.array(sample_data['target_positions'])
    else:
        goals = []
        for agent_idx in range(num_agents):
            agent_key = f"agent_{agent_idx}"
            agent_states = sample_data["trajectories"][agent_key]["states"]
            if len(agent_states) > 0:
                final_state = agent_states[-1]
                goals.append([final_state[0], final_state[1]])
            else:
                goals.append([0.0, 0.0])
        return jnp.array(goals)

# ============================================================================
# METRICS COMPUTATION FUNCTIONS
# ============================================================================

def compute_ade_fde(predicted_trajectory: jnp.ndarray, ground_truth_trajectory: jnp.ndarray) -> Tuple[float, float]:
    """Compute Average Displacement Error (ADE) and Final Displacement Error (FDE)."""
    min_length = min(len(predicted_trajectory), len(ground_truth_trajectory))
    pred_traj = predicted_trajectory[:min_length, :2]  # Only position (x, y)
    gt_traj = ground_truth_trajectory[:min_length, :2]  # Only position (x, y)
    
    displacement_errors = jnp.linalg.norm(pred_traj - gt_traj, axis=1)
    ade = jnp.mean(displacement_errors)
    fde = displacement_errors[-1]
    
    return float(ade), float(fde)

def compute_consistency_metric(receding_horizon_results: List[Dict[str, Any]]) -> float:
    """
    Compute consistency metric from receding horizon results.
    Consistency measures how much the selected agents change between iterations.
    
    Args:
        receding_horizon_results: List of iteration results from receding horizon planning
    
    Returns:
        Consistency score (0 = completely consistent, 1 = completely inconsistent)
    """
    if len(receding_horizon_results) < 2:
        return 0.0
    
    # Extract selected agents for each iteration
    selected_agents_per_iteration = []
    for iteration_result in receding_horizon_results:
        if 'selected_agents' in iteration_result:
            selected_agents = set(iteration_result['selected_agents'])
            selected_agents_per_iteration.append(selected_agents)
    
    if len(selected_agents_per_iteration) < 2:
        return 0.0
    
    # Compute Jaccard distance between consecutive iterations
    jaccard_distances = []
    for i in range(1, len(selected_agents_per_iteration)):
        prev_agents = selected_agents_per_iteration[i-1]
        curr_agents = selected_agents_per_iteration[i]
        
        # Jaccard similarity = intersection / union
        intersection = len(prev_agents & curr_agents)
        union = len(prev_agents | curr_agents)
        
        if union == 0:
            jaccard_similarity = 1.0  # If no agents, consider it perfectly consistent
        else:
            jaccard_similarity = intersection / union
        
        jaccard_distances.append(jaccard_similarity)
    
    # Return average consistency (higher is more consistent)
    return float(np.mean(jaccard_distances))

# ============================================================================
# MAIN TESTING FUNCTION
# ============================================================================

def test_psn_with_pedestrian_data(sample_data: Dict[str, Any],
                                 psn_model: PlayerSelectionNetwork = None,
                                 psn_trained_state: Any = None,
                                 goal_model: GoalInferenceNetwork = None,
                                 goal_trained_state: Any = None) -> Dict[str, Any]:
    """
    Test PSN with pedestrian data using prediction_test with true goals.
    
    Args:
        sample_data: Pedestrian sample data in expected format
        psn_model: Trained PSN model
        psn_trained_state: Trained PSN model state
        goal_model: Trained goal inference model (not used for true goals)
        goal_trained_state: Trained goal inference model state (not used for true goals)
    
    Returns:
        Dictionary containing test results
    """
    print(f"    Testing PSN with pedestrian data sample {sample_data['sample_id']}")
    
    # Extract parameters from configuration
    dt = config.game.dt
    # T_receding_horizon_planning = config.game.T_receding_horizon_planning
    T_receding_horizon_planning = 30
    T_observation = config.game.T_observation
    ego_agent_id = config.game.ego_agent_id
    
    # Get actual number of agents from the data
    n_agents = sample_data.get("n_agents", config.game.N_agents)
    
    # Determine actual trajectory length from the data
    # Use the first agent's trajectory length as reference
    first_agent_key = f"agent_0"
    actual_trajectory_length = len(sample_data["trajectories"][first_agent_key]["states"])
    
    # Calculate receding horizon parameters - use step 50 as goal
    # We need at least T_observation steps for observation, then planning up to step 50
    target_step = 50
    if actual_trajectory_length < target_step:
        print(f"    Warning: Trajectory too short ({actual_trajectory_length} steps), using full trajectory")
        T_total = actual_trajectory_length
        T_receding_horizon_iterations = max(1, T_total - T_observation)
    else:
        # Use step 50 as the target
        T_total = target_step
        T_receding_horizon_iterations = T_total - T_observation
    
    print(f"    Sample trajectory length: {actual_trajectory_length} steps")
    print(f"    Observation period: {T_observation} steps")
    print(f"    Receding horizon iterations: {T_receding_horizon_iterations} steps")
    print(f"    Total steps: {T_total} steps (using step 50 as goal)")
    
    # Initialize results storage
    results = {
        'sample_id': sample_data['sample_id'],
        'ego_agent_id': ego_agent_id,
        'goal_source': 'true_goals',
        'test_type': 'prediction_test',
        'T_observation': T_observation,
        'T_total': T_total,
        'T_receding_horizon_planning': T_receding_horizon_planning,
        'T_receding_horizon_iterations': T_receding_horizon_iterations,
        'receding_horizon_results': [],
        'final_game_state': None,
        'computation_times': [],
        'sample_computation_time': 0.0,
        'prediction_metrics': {},
        'planning_metrics': {}
    }
    
    # Extract true goals for all agents
    true_goals = extract_reference_goals(sample_data, n_agents)
    
    # Initialize game state
    current_game_state = {
        "trajectories": {
            f"agent_{i}": {
                "states": [],
                "controls": []
            }
            for i in range(n_agents)
        }
    }
    
    # Phase 1: Observation period (steps 1-T_observation) - use ground truth trajectories
    print(f"    Phase 1: Observation period (steps 1-{T_observation})")
    for step in range(T_observation):
        for agent_idx in range(n_agents):
            agent_key = f"agent_{agent_idx}"
            agent_states = sample_data["trajectories"][agent_key]["states"]
            
            if step < len(agent_states):
                current_state = agent_states[step]
                current_game_state["trajectories"][agent_key]["states"].append(current_state)
            else:
                last_state = agent_states[-1]
                current_game_state["trajectories"][agent_key]["states"].append(last_state)
    
    # Phase 2: Receding horizon planning with PSN (steps T_observation+1 to T_total)
    print(f"    Phase 2: Receding horizon planning with PSN (steps {T_observation+1} to {T_total})")
    
    sample_start_time = time.time()
    
    # Current states (start with states at end of observation period)
    current_states = []
    for agent_idx in range(n_agents):
        agent_key = f"agent_{agent_idx}"
        agent_states = current_game_state["trajectories"][agent_key]["states"]
        if len(agent_states) > 0:
            current_states.append(jnp.array(agent_states[-1]))
        else:
            sample_states = sample_data["trajectories"][agent_key]["states"]
            current_states.append(jnp.array(sample_states[T_observation - 1] if len(sample_states) >= T_observation else sample_states[-1]))
    
    # Main receding horizon loop
    for iteration in range(T_receding_horizon_iterations):
        # print(f"      Iteration {iteration + 1}/{T_receding_horizon_iterations}")
        
        # Use true goals for all agents (prediction_test with true_goals)
        predicted_goals = true_goals
        
        # Select agents using PSN or baseline method
        if config.testing.receding_horizon.use_baseline:
            # Use baseline method for agent selection
            baseline_mode = config.testing.receding_horizon.baseline_mode
            baseline_param = config.testing.receding_horizon.baseline_parameter
            
            # Prepare trajectory history for baseline selection
            trajectory_history = []
            for agent_idx in range(n_agents):
                agent_key = f"agent_{agent_idx}"
                agent_states = current_game_state["trajectories"][agent_key]["states"]
                if len(agent_states) > 0:
                    trajectory_history.append(np.array(agent_states))
                else:
                    # If no trajectory yet, use current state
                    trajectory_history.append(np.array([current_states[agent_idx]]))
            
            # Prepare control history (dummy for now, baselines may not need it)
            prev_controls = [np.zeros(2) for _ in range(n_agents)]
            
            # Prepare observation input for baseline
            obs_traj = extract_observation_trajectory(sample_data, config.psn.obs_input_type, n_agents)
            if config.psn.obs_input_type == "partial":
                state_dim = 2  # Only position (x, y)
                obs_array = obs_traj[:, :, :2]  # Keep only x, y coordinates
            else:  # full
                state_dim = 4  # Full state (x, y, vx, vy)
                obs_array = obs_traj
            
            # Reshape for baseline input
            obs_input_baseline = obs_array.reshape(1, T_observation, n_agents, state_dim)
            
            # Convert mode to proper case for baselines.py
            if baseline_mode.lower() == "nearest neighbor":
                baseline_mode_proper = "Nearest Neighbor"
            elif baseline_mode.lower() == "distance threshold":
                baseline_mode_proper = "Distance Threshold"
            elif baseline_mode.lower() == "all":
                baseline_mode_proper = "All"
            else:
                baseline_mode_proper = baseline_mode
            
            # Use baseline selection function
            predicted_mask = baseline_selection(
                input_traj=obs_input_baseline,
                trajectory=trajectory_history,
                control=prev_controls,
                mode=baseline_mode_proper,
                sim_step=iteration,
                mode_parameter=baseline_param
            )
            
            # Convert mask to selected agents (baseline returns mask for other agents, excluding ego)
            selected_other_agents = np.where(predicted_mask == 1)[0] + 1  # +1 because mask excludes ego
            selected_agents = np.concatenate([[0], selected_other_agents])  # Include ego agent
            selected_agents = jnp.array(selected_agents)
            num_selected = len(selected_agents)
            mask_sparsity = num_selected / n_agents
            
            # print(f"        Baseline ({baseline_mode}): Selected {num_selected} agents (ego + {num_selected-1} others)")
                
        elif psn_model is not None and psn_trained_state is not None:
            if n_agents != 10:
                print(f"        PSN model trained for 10 agents, but data has {n_agents} agents, using all agents")
                selected_agents = jnp.arange(n_agents)
                num_selected = n_agents
                mask_sparsity = 1.0
                predicted_mask = jnp.ones(n_agents - 1)
            else:
                # Extract observation trajectory for PSN
                obs_traj = extract_observation_trajectory(sample_data, config.psn.obs_input_type, n_agents)
                
                # Convert to PSN input format
                psn_obs_input_type = config.psn.obs_input_type
                if psn_obs_input_type == "partial":
                    state_dim = 2  # Only position (x, y)
                    obs_array = obs_traj[:, :, :2]  # Keep only x, y coordinates
                else:  # full
                    state_dim = 4  # Full state (x, y, vx, vy)
                    obs_array = obs_traj
                
                # The PSN model expects input in (batch_size, T_observation, N_agents, state_dim) format
                obs_input = obs_array.reshape(1, T_observation, n_agents, state_dim)
                
                # Get PSN prediction
                predicted_mask = psn_model.apply({'params': psn_trained_state['params']}, obs_input, deterministic=True)
                predicted_mask = predicted_mask[0]  # Remove batch dimension
                
                # Select agents based on predicted mask
                selection_method = config.testing.receding_horizon.selection_method
                mask_threshold = config.testing.receding_horizon.mask_threshold
                rank = config.testing.receding_horizon.rank
                
                selected_agents, num_selected, mask_sparsity = select_agents_by_mask(
                    predicted_mask, selection_method, mask_threshold, rank)
                
                # print(f"        PSN selected {num_selected} agents out of {n_agents}")
        else:
            # No PSN: use all agents
            selected_agents = jnp.arange(n_agents)
            num_selected = n_agents
            mask_sparsity = 1.0
            predicted_mask = jnp.ones(n_agents - 1)
            print(f"        No PSN model, using all {n_agents} agents")
        
        # Ensure ego agent is always included
        if 0 not in selected_agents:
            selected_agents = jnp.concatenate([jnp.array([0]), selected_agents])
            selected_agents = jnp.unique(selected_agents)
        
        # Filter current states and predicted goals to only include selected agents
        filtered_current_states = [current_states[i] for i in selected_agents]
        filtered_predicted_goals = predicted_goals[selected_agents]
        
        # Solve the receding horizon game using proper iLQGames solver
        try:
            # Create agent setup for the filtered agents
            agents, reference_trajectories = create_agent_setup(filtered_current_states, filtered_predicted_goals)
            
            # Create loss functions
            loss_functions, linearize_loss_functions, compiled_functions = create_loss_functions(
                agents, reference_trajectories)
            
            # Solve the game with filtered agents
            first_controls, full_trajectories, game_time = solve_receding_horizon_game(
                agents, filtered_current_states, filtered_predicted_goals, compiled_functions)
            
            # Map results back to full agent list for compatibility
            full_first_controls = [jnp.zeros(2) for _ in range(n_agents)]
            full_trajectories_expanded = [jnp.zeros((T_receding_horizon_planning, 4)) for _ in range(n_agents)]
            
            for i, agent_idx in enumerate(selected_agents):
                full_first_controls[agent_idx] = first_controls[i]
                full_trajectories_expanded[agent_idx] = full_trajectories[i]
            
            first_controls = full_first_controls
            full_trajectories = full_trajectories_expanded
            
        except Exception as e:
            print(f"        Game solving failed: {e}, using fallback")
            # Fallback: use simple proportional control
            first_controls = []
            for i in range(len(selected_agents)):
                current_state = filtered_current_states[i]
                goal = filtered_predicted_goals[i]
                position_error = goal - current_state[:2]
                control = 0.1 * position_error
                first_controls.append(control)
            
            # Map results back to full agent list
            full_first_controls = [jnp.zeros(2) for _ in range(n_agents)]
            for i, agent_idx in enumerate(selected_agents):
                full_first_controls[agent_idx] = first_controls[i]
            
            full_trajectories = [jnp.zeros((T_receding_horizon_planning, 4)) for _ in range(n_agents)]
            game_time = 0.001
        
        # Store results for this iteration
        iteration_result = {
            'iteration': iteration,
            'step': T_observation + iteration,
            'current_states': [state.tolist() if hasattr(state, 'tolist') else list(state) for state in current_states],
            'predicted_goals': predicted_goals.tolist() if hasattr(predicted_goals, 'tolist') else predicted_goals,
            'true_goals': true_goals.tolist() if hasattr(true_goals, 'tolist') else true_goals,
            'predicted_mask': predicted_mask.tolist() if hasattr(predicted_mask, 'tolist') else predicted_mask,
            'selected_agents': selected_agents.tolist() if hasattr(selected_agents, 'tolist') else list(selected_agents),
            'num_selected': int(num_selected),
            'mask_sparsity': float(mask_sparsity),
            'first_controls': [control.tolist() if hasattr(control, 'tolist') else list(control) for control in full_first_controls],
            'full_trajectories': [traj.tolist() if hasattr(traj, 'tolist') else traj for traj in full_trajectories],
            'game_solving_time': float(game_time)
        }
        
        results['receding_horizon_results'].append(iteration_result)
        results['computation_times'].append(game_time)
        
        # Update current states (simplified dynamics)
        for i in range(n_agents):
            if i == 0:  # Ego agent: apply computed control
                current_state = current_states[i]
                control = full_first_controls[i]
                
                # Simple dynamics update
                new_state = jnp.array([
                    current_state[0] + dt * current_state[2],  # x + dt * vx
                    current_state[1] + dt * current_state[3],  # y + dt * vy
                    current_state[2] + dt * control[0],        # vx + dt * ax
                    current_state[3] + dt * control[1]         # vy + dt * ay
                ])
                
                current_states[i] = new_state
                current_game_state["trajectories"][f"agent_{i}"]["states"].append(new_state.tolist())
            else:  # Other agents: use ground truth
                ref_step = T_observation + iteration
                if ref_step < len(sample_data["trajectories"][f"agent_{i}"]["states"]):
                    ref_state = sample_data["trajectories"][f"agent_{i}"]["states"][ref_step]
                else:
                    ref_state = sample_data["trajectories"][f"agent_{i}"]["states"][-1]
                
                current_states[i] = jnp.array(ref_state)
                current_game_state["trajectories"][f"agent_{i}"]["states"].append(ref_state)
    
    # Store final game state with computed trajectories
    # The current_game_state contains the step-by-step computed states
    # We also want to store the full computed trajectories from the iLQGames solver
    final_game_state = current_game_state.copy()
    
    # Add computed trajectories from the last iteration's full trajectories
    if results['receding_horizon_results']:
        last_iteration = results['receding_horizon_results'][-1]
        if 'full_trajectories' in last_iteration:
            computed_trajectories = {}
            for i in range(n_agents):
                agent_key = f"agent_{i}"
                if i < len(last_iteration['full_trajectories']):
                    computed_trajectories[agent_key] = {
                        'states': last_iteration['full_trajectories'][i]
                    }
                else:
                    # Fallback to current game state if no computed trajectory
                    computed_trajectories[agent_key] = {
                        'states': current_game_state["trajectories"][agent_key]["states"]
                    }
            final_game_state['computed_trajectories'] = computed_trajectories
    
    results['final_game_state'] = final_game_state
    
    # Compute metrics
    if results['receding_horizon_results']:
        # Extract ego trajectory (computed) and ground truth trajectory
        ego_computed_trajectory = jnp.array(current_game_state["trajectories"][f"agent_{ego_agent_id}"]["states"])
        ego_ground_truth_trajectory = jnp.array(sample_data["trajectories"][f"agent_{ego_agent_id}"]["states"])
        
        # Only analyze steps 10-50 (receding horizon planning phase)
        analysis_start_step = T_observation
        analysis_end_step = T_total  # This is now 50
        
        ego_computed_analysis = ego_computed_trajectory[analysis_start_step:analysis_end_step]
        ego_ground_truth_analysis = ego_ground_truth_trajectory[analysis_start_step:analysis_end_step]
        
        # Compute prediction metrics
        ade, fde = compute_ade_fde(ego_computed_analysis, ego_ground_truth_analysis)
        results['prediction_metrics'] = {'ade': ade, 'fde': fde}
        
        # Compute planning metrics (simplified)
        other_ground_truth_trajectories = []
        for i in range(n_agents):
            if i != ego_agent_id:
                other_traj = jnp.array(sample_data["trajectories"][f"agent_{i}"]["states"])
                other_traj_analysis = other_traj[analysis_start_step:analysis_end_step]
                other_ground_truth_trajectories.append(other_traj_analysis)
        
        # Extract ego controls
        ego_controls = []
        for iter_result in results['receding_horizon_results']:
            ego_control = jnp.array(iter_result['first_controls'][ego_agent_id])
            ego_controls.append(ego_control)
        ego_controls = jnp.array(ego_controls)
        
        # Extract ego goals
        ego_goals = jnp.array(true_goals[ego_agent_id])
        
        # Compute planning metrics
        planning_metrics = compute_planning_metrics(
            ego_computed_analysis,
            other_ground_truth_trajectories,
            ego_controls,
            ego_goals,
            dt
        )
        
        results['planning_metrics'] = planning_metrics
        results['mean_computation_time'] = float(np.mean(results['computation_times']))
        
        # Compute consistency metric
        results['consistency_metric'] = compute_consistency_metric(results['receding_horizon_results'])
    else:
        results['prediction_metrics'] = {'ade': float('inf'), 'fde': float('inf')}
        results['planning_metrics'] = {'navigation_cost': float('inf'), 'safety_cost': float('inf'), 'control_cost': float('inf'), 'trajectory_length': float('inf'), 'trajectory_smoothness': float('inf')}
        results['mean_computation_time'] = 0.0
        results['consistency_metric'] = 0.0
    
    sample_end_time = time.time()
    results['sample_computation_time'] = sample_end_time - sample_start_time
    
    print(f"    ✓ Completed PSN testing with pedestrian data")
    print(f"    ✓ Sample computation time: {results['sample_computation_time']:.4f}s")
    
    # Store normalized sample data for visualization
    results['normalized_sample_data'] = sample_data
    
    # Compute metrics
    metrics = {}
    
    # Get ego trajectory and ground truth
    ego_trajectory = results.get('ego_trajectory', [])
    ground_truth_trajectory = results.get('ground_truth_trajectory', [])
    
    if len(ego_trajectory) > 0 and len(ground_truth_trajectory) > 0:
        # Prediction metrics (ADE, FDE)
        ade, fde = compute_ade_fde(jnp.array(ego_trajectory), jnp.array(ground_truth_trajectory))
        metrics['ade'] = ade
        metrics['fde'] = fde
        
        # Planning metrics - use the already computed planning_metrics from results
        if 'planning_metrics' in results:
            metrics.update(results['planning_metrics'])
    
    # Add mask statistics and consistency metric from receding horizon results
    if 'receding_horizon_results' in results and results['receding_horizon_results']:
        # Compute average mask sparsity and num selected agents across all iterations
        mask_sparsities = []
        num_selected_list = []
        for iteration_result in results['receding_horizon_results']:
            mask_sparsities.append(iteration_result.get('mask_sparsity', 0.0))
            num_selected_list.append(iteration_result.get('num_selected', 0))
        
        metrics['mask_sparsity'] = float(np.mean(mask_sparsities)) if mask_sparsities else 0.0
        metrics['num_selected_agents'] = float(np.mean(num_selected_list)) if num_selected_list else 0.0
    else:
        metrics['mask_sparsity'] = 0.0
        metrics['num_selected_agents'] = 0
    
    metrics['consistency_metric'] = results.get('consistency_metric', 0.0)
    
    results['metrics'] = metrics
    
    return results

def save_test_results(results: Dict[str, Any], save_dir: str) -> str:
    """Save test results to JSON file."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    filename = f"psn_ped_test_sample_{results['sample_id']:03d}.json"
    filepath = save_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return str(filepath)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run PSN testing with pedestrian data."""
    print("=" * 80)
    print("PSN TESTING WITH PEDESTRIAN DATA")
    print("=" * 80)
    
    # Configuration
    data_dir = "data_ped"
    num_samples = 6  # Test with all 6 samples
    
    # Create proper output directory structure based on config
    if config.testing.receding_horizon.use_baseline:
        # Baseline results go to baseline_results/ped_test directory
        base_dir = "baseline_results/ped_test"
        test_type = "prediction_test"
        goal_source = "true_goals"
        n_agents = config.game.N_agents
        baseline_mode = config.testing.receding_horizon.baseline_mode.replace(" ", "_").lower()
        baseline_param = config.testing.receding_horizon.baseline_parameter
        
        # Create baseline directory structure: ped_results_{n_agents}_{method}_param_{param}_{goal_suffix}
        goal_suffix = "goal_inference" if goal_source == "goal_inference" else "goal_true"
        output_dir = os.path.join(base_dir, f"ped_results_{n_agents}_{baseline_mode}_param_{baseline_param}_{goal_suffix}")
    else:
        # PSN results go to specific PSN model directory - use same logic as test_psn_receding_horizon.py
        base_dir = "log/goal_true_N_10_T_50_obs_10"
        test_type = "prediction_test"
        goal_source = "true_goals"
        n_agents = config.game.N_agents
        
        # Determine effective number of agents for model selection
        effective_n_agents = 10 if config.game.N_agents > 10 else config.game.N_agents
        
        # Construct PSN model directory name based on config
        obs_input_type = config.psn.obs_input_type
        if test_type == "planning_test":
            model_name = f"psn_gru_{obs_input_type}_planning_true_goals"
        else:  # prediction_test
            model_name = f"psn_gru_{obs_input_type}_true_goals"
        
        psn_dir = f"{model_name}_N_{effective_n_agents}_T_{config.game.T_total}_obs_{config.game.T_observation}_lr_{config.psn.learning_rate}_bs_{config.psn.batch_size}_sigma1_{config.psn.sigma1}_sigma2_{config.psn.sigma2}_epochs_{config.psn.num_epochs}"
        
        # Create PSN directory structure - use same pattern as test_psn_receding_horizon.py
        # Extract method name from PSN model path
        psn_model_name = "psn_best_model"
        
        # Extract full/partial designation from config
        obs_type = config.psn.obs_input_type  # This is "full" or "partial"
        
        # Include selection method in directory name for PSN methods
        selection_method = config.testing.receding_horizon.selection_method
        if selection_method == "threshold":
            method_suffix = f"threshold_{config.testing.receding_horizon.mask_threshold}"
        else:  # rank
            method_suffix = f"rank_{config.testing.receding_horizon.rank}"
        
        # Construct output directory like in test_psn_receding_horizon.py but with ped_results prefix
        output_dir = os.path.join(base_dir, psn_dir, f"ped_results_{n_agents}_{test_type}_goal_true_{obs_type}_{method_suffix}_{psn_model_name}")
    
    # Create the directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Override optimization parameters for faster testing
    print("Using optimized parameters for faster testing:")
    print(f"  - iLQGames iterations: 10 (reduced from {config.optimization.num_iters})")
    print(f"  - Step size: 0.01 (increased from {config.optimization.step_size})")
    print(f"  - Planning horizon: 10 (reduced from {config.game.T_receding_horizon_planning})")
    
    # Store original values
    original_num_iters = config.optimization.num_iters
    original_step_size = config.optimization.step_size
    original_planning_horizon = config.game.T_receding_horizon_planning
    
    # Set faster parameters
    config.optimization.num_iters = 10
    config.optimization.step_size = 0.01
    config.game.T_receding_horizon_planning = 10
    
    # Load PSN model (optional)
    psn_model = None
    psn_trained_state = None
    
    # Try to load PSN model if available
    try:
        # Construct PSN model path dynamically based on config
        effective_n_agents = 10 if config.game.N_agents > 10 else config.game.N_agents
        obs_input_type = config.psn.obs_input_type
        test_type = "prediction_test"
        
        if test_type == "planning_test":
            model_name = f"psn_gru_{obs_input_type}_planning_true_goals"
        else:  # prediction_test
            model_name = f"psn_gru_{obs_input_type}_true_goals"
        
        model_path = f"log/goal_true_N_{effective_n_agents}_T_{config.game.T_total}_obs_{config.game.T_observation}/{model_name}_N_{effective_n_agents}_T_{config.game.T_total}_obs_{config.game.T_observation}_lr_{config.psn.learning_rate}_bs_{config.psn.batch_size}_sigma1_{config.psn.sigma1}_sigma2_{config.psn.sigma2}_epochs_{config.psn.num_epochs}/psn_best_model.pkl"
        
        if os.path.exists(model_path):
            print(f"Loading PSN model from: {model_path}")
            psn_model, psn_trained_state, _, _ = load_trained_models(model_path, None, obs_input_type)
            print("✓ PSN model loaded successfully")
            print(f"Note: PSN model was trained for {effective_n_agents} agents with {obs_input_type} input")
            print("Pedestrian data uses 10 agents from CSV, so PSN should work correctly")
        else:
            print(f"No PSN model found at: {model_path}")
            print("Running without PSN (all agents selected)")
    except Exception as e:
        print(f"Could not load PSN model: {e}")
        print("Running without PSN (all agents selected)")
    
    # Load pedestrian data
    print(f"Loading pedestrian data from: {data_dir}")
    samples = load_pedestrian_data(data_dir, num_samples)
    
    # Test each sample
    all_results = []
    for i, sample_data in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"TESTING SAMPLE {i+1}/{len(samples)}")
        print(f"{'='*60}")
        
        try:
            results = test_psn_with_pedestrian_data(
                sample_data, psn_model, psn_trained_state, None, None)
            
            # Save results
            filepath = save_test_results(results, output_dir)
            print(f"  ✓ Results saved to: {filepath}")
            
            all_results.append(results)
            print(f"  ✓ Sample {i+1} completed successfully")
            
        except Exception as e:
            print(f"  ✗ Sample {i+1} failed: {e}")
            print(f"  Continuing with next sample...")
            continue
    
    # Print summary
    print(f"\n" + "=" * 80)
    print("TESTING COMPLETED")
    print("=" * 80)
    print(f"Successfully tested: {len(all_results)}/{len(samples)} samples")
    print(f"Results saved to: {output_dir}")
    
    # Print average metrics
    if all_results:
        avg_ade = np.mean([r['prediction_metrics']['ade'] for r in all_results])
        avg_fde = np.mean([r['prediction_metrics']['fde'] for r in all_results])
        print(f"Average ADE: {avg_ade:.4f}")
        print(f"Average FDE: {avg_fde:.4f}")
    
    # Create summary file
    summary_path = os.path.join(output_dir, "test_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("PEDESTRIAN DATA PSN TESTING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Configuration:\n")
        f.write(f"  - Data source: {data_dir}\n")
        f.write(f"  - Number of samples: {num_samples}\n")
        f.write(f"  - Goal source: true_goals (step 50)\n")
        f.write(f"  - PSN model: {'Loaded' if psn_model is not None else 'Not used'}\n")
        f.write(f"  - Test type: prediction_test\n\n")
        f.write(f"Results:\n")
        f.write(f"  - Successfully tested: {len(all_results)} samples\n")
        f.write(f"  - Output directory: {output_dir}\n\n")
        if all_results:
            avg_ade = np.mean([r['prediction_metrics']['ade'] for r in all_results])
            avg_fde = np.mean([r['prediction_metrics']['fde'] for r in all_results])
            f.write(f"  - Average ADE: {avg_ade:.4f}\n")
            f.write(f"  - Average FDE: {avg_fde:.4f}\n")
        f.write(f"\nTo create visualizations, run:\n")
        f.write(f"  python player_selection_network/create_gif_ped.py --results_dir {output_dir}\n")
        f.write(f"  python player_selection_network/create_fig_ped.py --results_dir {output_dir}\n")
        f.write(f"  python player_selection_network/test_analysis_ped.py --results_dir {output_dir}\n")
    
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()
