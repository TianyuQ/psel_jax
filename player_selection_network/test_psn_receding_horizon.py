#!/usr/bin/env python3
"""
Test Receding Horizon Planning with Goal Inference and Player Selection Models

This script tests receding horizon planning by applying both the goal inference model
and player selection model at each iteration. It demonstrates how these models
can be integrated into a closed-loop receding horizon control system.

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
import matplotlib.pyplot as plt
from jax import vmap, jit, grad
import pickle
import os

# Import from the main lqrax module
import sys
# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lqrax import iLQR

# Import configuration loader and models
from config_loader import load_config, get_device_config, setup_jax_config

# Import model classes
from psn_training_with_pretrained_goals import (
    PlayerSelectionNetwork, GoalInferenceNetwork, load_trained_models
)
# Import baselines
from player_selection_network.baselines import baseline_selection

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
dt = config.game.dt
T_receding_horizon_planning = config.game.T_receding_horizon_planning  # Planning horizon for each individual game
T_receding_horizon_iterations = config.game.T_receding_horizon_iterations  # Total number of receding horizon iterations
T_total = config.game.T_total  # Total number of time steps in trajectory
T_observation = config.game.T_observation  # Number of steps to observe before solving the game
n_agents = config.game.N_agents
ego_agent_id = config.game.ego_agent_id

# Optimization parameters
num_iters = config.optimization.num_iters
step_size = config.optimization.step_size

print(f"Configuration loaded:")
print(f"  N agents: {n_agents}")
print(f"  Planning horizon: {T_receding_horizon_planning} steps per game")
print(f"  Total receding horizon iterations: {T_receding_horizon_iterations} steps")
print(f"  Total trajectory steps: {T_total}")
print(f"  Observation steps: {T_observation}")
print(f"  dt: {dt}")
print(f"  Optimization: {num_iters} iters, step size: {step_size}")


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

class PointAgent(iLQR):
    """
    Point mass agent for trajectory optimization.
    
    State: [x, y, vx, vy] - position (x,y) and velocity (vx, vy)
    Control: [ax, ay] - acceleration in x and y directions
    
    Dynamics:
        dx/dt = vx
        dy/dt = vy
        dvx/dt = ax
        dvy/dt = ay
    """
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
# METRICS COMPUTATION FUNCTIONS
# ============================================================================

def compute_ade_fde(predicted_trajectory: jnp.ndarray, ground_truth_trajectory: jnp.ndarray) -> Tuple[float, float]:
    """
    Compute Average Displacement Error (ADE) and Final Displacement Error (FDE).
    
    Args:
        predicted_trajectory: Predicted trajectory (T, 2) - position only
        ground_truth_trajectory: Ground truth trajectory (T, 2) - position only
    
    Returns:
        Tuple of (ADE, FDE)
    """
    # Ensure both trajectories have the same length
    min_length = min(len(predicted_trajectory), len(ground_truth_trajectory))
    pred_traj = predicted_trajectory[:min_length, :2]  # Only position (x, y)
    gt_traj = ground_truth_trajectory[:min_length, :2]  # Only position (x, y)
    
    # Compute displacement errors at each time step
    displacement_errors = jnp.linalg.norm(pred_traj - gt_traj, axis=1)
    
    # ADE: average displacement error over all time steps
    ade = jnp.mean(displacement_errors)
    
    # FDE: displacement error at the final time step
    fde = displacement_errors[-1]
    
    return float(ade), float(fde)


def compute_planning_metrics(ego_trajectory: jnp.ndarray, 
                           other_trajectories: List[jnp.ndarray],
                           ego_controls: jnp.ndarray,
                           ego_goals: jnp.ndarray,
                           dt: float) -> Dict[str, float]:
    """
    Compute planning metrics: navigation cost, safety cost, and control cost.
    
    Args:
        ego_trajectory: Ego agent trajectory (T, 4) - [x, y, vx, vy]
        other_trajectories: List of other agent trajectories (T, 4) each
        ego_controls: Ego agent controls (T, 2) - [ax, ay]
        ego_goals: Ego agent goals (2,) - [x, y]
        dt: Time step size
    
    Returns:
        Dictionary with navigation_cost, safety_cost, control_cost
    """
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
            # Exponential penalty for proximity
            safety_penalties = collision_weight * jnp.exp(-collision_scale * distances)
            safety_cost += jnp.sum(safety_penalties) * dt
    
    # Control cost: control effort
    ctrl_weight = config.optimization.control_weight
    control_magnitudes = jnp.linalg.norm(ego_controls, axis=1)
    control_cost = ctrl_weight * jnp.sum(control_magnitudes) * dt
    
    return {
        'navigation_cost': float(navigation_cost),
        'safety_cost': float(safety_cost),
        'control_cost': float(control_cost)
    }


def compute_trajectory_metrics(ego_trajectory: jnp.ndarray,
                             ground_truth_trajectory: jnp.ndarray,
                             other_trajectories: List[jnp.ndarray],
                             ego_controls: jnp.ndarray,
                             ego_goals: jnp.ndarray,
                             dt: float) -> Dict[str, float]:
    """
    Compute all trajectory metrics for a single sample.
    
    Args:
        ego_trajectory: Computed ego trajectory (T, 4)
        ground_truth_trajectory: Ground truth ego trajectory (T, 4)
        other_trajectories: List of other agent ground truth trajectories
        ego_controls: Computed ego controls (T, 2)
        ego_goals: Ego agent goals (2,)
        dt: Time step size
    
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    # Prediction metrics (ADE, FDE)
    if config.testing.receding_horizon.compute_prediction_metrics:
        ade, fde = compute_ade_fde(ego_trajectory, ground_truth_trajectory)
        metrics['ade'] = ade
        metrics['fde'] = fde
    
    # Planning metrics (navigation, safety, control costs)
    if config.testing.receding_horizon.compute_planning_metrics:
        planning_metrics = compute_planning_metrics(
            ego_trajectory, other_trajectories, ego_controls, ego_goals, dt)
        metrics.update(planning_metrics)
    
    return metrics


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_agent_setup(initial_states: List[jnp.ndarray], target_positions: List[jnp.ndarray]) -> tuple:
    """
    Create a set of agents with their initial states and reference trajectories.
    
    Args:
        initial_states: Initial states for each agent
        target_positions: Target positions for each agent
    
    Returns:
        Tuple of (agents, reference_trajectories)
    """
    agents = []
    reference_trajectories = []
    
    # Use the actual number of agents passed in, not the global n_agents
    n_agents_in_game = len(initial_states)
    
    # Cost function weights (same for all agents) - exactly like original ilqgames_example
    Q = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001]))  # State cost weights (position, position, velocity, velocity)
    R = jnp.diag(jnp.array([0.01, 0.01]))               # Control cost weights (ax, ay)
    
    for i in range(n_agents_in_game):
        # Create agent
        agent = PointAgent(dt=dt, x_dim=4, u_dim=2, Q=Q, R=R)
        agents.append(agent)
        
        # Reference trajectory (EXACTLY like reference generation)
        # Create a straight-line reference trajectory from initial position to target
        start_pos = jnp.array(initial_states[i][:2])  # Extract x, y position and convert to array
        target_pos = jnp.array(target_positions[i])   # Convert to array
        
        # Linear interpolation over time steps (exactly like reference generation)
        ref_traj = jnp.linspace(start_pos, target_pos, T_receding_horizon_planning)
        reference_trajectories.append(ref_traj)
    
    return agents, reference_trajectories


def create_loss_functions(agents: list, reference_trajectories: list) -> tuple:
    """
    Create loss functions and their linearizations for all agents.
    
    Args:
        agents: List of agent objects
        reference_trajectories: List of reference trajectories for each agent
    
    Returns:
        Tuple of (loss_functions, linearize_loss_functions, compiled_functions)
    """
    loss_functions = []
    linearize_loss_functions = []
    compiled_functions = []
    
    n_agents_in_game = len(agents)  # Use actual number of agents in this game
    
    for i, agent in enumerate(agents):
        # Create loss function for this agent
        def create_runtime_loss(agent_idx, agent_obj, ref_traj):
            def runtime_loss(xt, ut, ref_xt, other_states):
                # Navigation cost - track reference trajectory (exactly like reference generation)
                nav_loss = jnp.sum(jnp.square(xt[:2] - ref_xt[:2]))
                
                collision_weight = config.optimization.collision_weight
                collision_scale = config.optimization.collision_scale
                ctrl_weight = config.optimization.control_weight
                
                # Collision avoidance costs - exponential penalty for proximity to other agents
                # (exactly like reference generation)
                collision_loss = 0.0
                for other_xt in other_states:
                    collision_loss += collision_weight * jnp.exp(-collision_scale * jnp.sum(jnp.square(xt[:2] - other_xt[:2])))
                
                # Control cost - simplified without velocity scaling
                ctrl_loss = ctrl_weight * jnp.sum(jnp.square(ut))
                
                # Return complete loss including all terms
                return nav_loss + collision_loss + ctrl_loss
            
            return runtime_loss
        
        runtime_loss = create_runtime_loss(i, agent, reference_trajectories[i])
        
        # Create trajectory loss function
        def trajectory_loss(x_traj, u_traj, ref_x_traj, other_x_trajs):
            def single_step_loss(args):
                xt, ut, ref_xt, other_xts = args
                return runtime_loss(xt, ut, ref_xt, other_xts)
            
            loss_array = vmap(single_step_loss)((x_traj, u_traj, ref_x_traj, other_x_trajs))
            return loss_array.sum() * agent.dt
        
        # Create linearization function
        def linearize_loss(x_traj, u_traj, ref_x_traj, other_x_trajs):
            dldx = grad(runtime_loss, argnums=(0))
            dldu = grad(runtime_loss, argnums=(1))
            
            def grad_step(args):
                xt, ut, ref_xt, other_xts = args
                return dldx(xt, ut, ref_xt, other_xts), dldu(xt, ut, ref_xt, other_xts)
            
            grads = vmap(grad_step)((x_traj, u_traj, ref_x_traj, other_x_trajs))
            return grads[0], grads[1]  # a_traj, b_traj
        
        # Compile functions with GPU optimizations
        compiled_loss = jit(trajectory_loss, device=device)
        compiled_linearize = jit(linearize_loss, device=device)
        compiled_linearize_dyn = jit(agent.linearize_dyn, device=device)
        compiled_solve = jit(agent.solve, device=device)
        
        loss_functions.append(trajectory_loss)
        linearize_loss_functions.append(linearize_loss)
        compiled_functions.append({
            'loss': compiled_loss,
            'linearize_loss': compiled_linearize,
            'linearize_dyn': compiled_linearize_dyn,
            'solve': compiled_solve
        })
    
    return loss_functions, linearize_loss_functions, compiled_functions


def solve_ilqgames_iterative(agents: list, 
                            initial_states: list,
                            reference_trajectories: list,
                            compiled_functions: list) -> tuple:
    """
    Solve the iLQGames problem using the original iterative approach.
    
    Args:
        agents: List of agent objects
        initial_states: List of initial states for each agent
        reference_trajectories: List of reference trajectories for each agent
        compiled_functions: List of compiled functions for each agent
    
    Returns:
        Tuple of (final_state_trajectories, final_control_trajectories, total_time)
    """
    start_time = time.time()
    
    # Initialize control trajectories with zeros
    control_trajectories = [jnp.zeros((T_receding_horizon_planning, 2)) for _ in range(len(agents))]
    
    # Track losses for debugging
    total_losses = []
    
    for iter in range(num_iters + 1):
        # Step 1: Linearize dynamics for all agents
        state_trajectories = []
        A_trajectories = []
        B_trajectories = []
        
        for i in range(len(agents)):
            x_traj, A_traj, B_traj = compiled_functions[i]['linearize_dyn'](
                initial_states[i], control_trajectories[i])
            state_trajectories.append(x_traj)
            A_trajectories.append(A_traj)
            B_trajectories.append(B_traj)
        
        # Step 2: Linearize loss functions for all agents
        a_trajectories = []
        b_trajectories = []
        
        for i in range(len(agents)):
            # Create list of other agents' states for this agent
            other_states = [state_trajectories[j] for j in range(len(agents)) if j != i]
            
            a_traj, b_traj = compiled_functions[i]['linearize_loss'](
                state_trajectories[i], control_trajectories[i], reference_trajectories[i], other_states)
            a_trajectories.append(a_traj)
            b_trajectories.append(b_traj)
        
        # Step 3: Solve LQR subproblems for all agents
        control_updates = []
        
        for i in range(len(agents)):
            v_traj, _ = compiled_functions[i]['solve'](
                A_trajectories[i], B_trajectories[i], 
                a_trajectories[i], b_trajectories[i])
            control_updates.append(v_traj)
        
        # Update control trajectories with gradient descent
        for i in range(len(agents)):
            control_trajectories[i] = control_trajectories[i] + step_size * control_updates[i]
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return state_trajectories, control_trajectories, total_time


def solve_ilqgames(agents: list, 
                   initial_states: list,
                   reference_trajectories: list,
                   compiled_functions: list) -> tuple:
    """
    Solve the iLQGames problem for multiple agents using original iterative approach.
    """
    return solve_ilqgames_iterative(agents, initial_states, reference_trajectories, compiled_functions)


def solve_receding_horizon_game(agents: list, 
                               current_states: list, 
                               target_positions: List[jnp.ndarray], 
                               compiled_functions: list) -> tuple:
    """
    Solve a single receding horizon game (50-horizon) and return the first control.
    
    Args:
        agents: List of agent objects
        current_states: Current states for each agent
        target_positions: Target positions for each agent
        compiled_functions: Compiled functions for each agent
    
    Returns:
        Tuple of (first_controls, full_trajectories, total_time)
    """
    start_time = time.time()
    
    # Create reference trajectories from current positions to targets
    current_reference_trajectories = []
    for i in range(len(agents)):
        start_pos = jnp.array(current_states[i][:2])  # Extract x, y position and convert to array
        target_pos = jnp.array(target_positions[i])   # Convert to array
        # Linear interpolation over planning horizon
        ref_traj = jnp.linspace(start_pos, target_pos, T_receding_horizon_planning)
        current_reference_trajectories.append(ref_traj)
    
    # Solve the 50-horizon game
    state_trajectories, control_trajectories, total_time = solve_ilqgames(
        agents, current_states, current_reference_trajectories, compiled_functions)
    
    # Extract the first control from each control trajectory
    first_controls = []
    for i in range(len(agents)):
        if len(control_trajectories[i]) > 0:
            first_control = control_trajectories[i][0]  # First control from the computed trajectory
            first_controls.append(first_control)
        else:
            first_controls.append(jnp.zeros(2))  # Fallback to zero control
    
    return first_controls, state_trajectories, total_time


def normalize_sample_data(sample_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize sample data structure to ensure consistent access.
    
    Handles both reference trajectory files (with "trajectories" key) and 
    receding horizon files (with "receding_horizon_trajectories" key).
    """
    normalized_data = sample_data.copy()
    
    # Check if this is a receding horizon file
    if "receding_horizon_trajectories" in sample_data and "trajectories" not in sample_data:
        # Convert receding horizon format to standard format
        normalized_data["trajectories"] = {}
        
        for agent_key, agent_data in sample_data["receding_horizon_trajectories"].items():
            # Use the actual executed states (50 steps) from receding horizon simulation
            if "states" in agent_data:
                # Use the actual executed receding horizon states (50 steps)
                normalized_data["trajectories"][agent_key] = {"states": agent_data["states"]}
            elif "full_trajectories" in agent_data and len(agent_data["full_trajectories"]) > 0:
                # Fallback: if no states field, use first trajectory (15 steps)
                first_trajectory = agent_data["full_trajectories"][0]
                normalized_data["trajectories"][agent_key] = {"states": first_trajectory}
            else:
                # Fallback: create dummy states
                normalized_data["trajectories"][agent_key] = {"states": [[0.0, 0.0, 0.0, 0.0] for _ in range(T_total)]}
    
    return normalized_data


def extract_observation_trajectory(sample_data: Dict[str, Any]) -> jnp.ndarray:
    """
    Extract observation trajectory (first 10 steps) for all agents.
    This matches the format used in goal inference training.
    
    Args:
        sample_data: Reference trajectory sample
        
    Returns:
        observation_trajectory: Observation trajectory (T_observation, N_agents, state_dim)
    """
    # Normalize data structure first
    normalized_data = normalize_sample_data(sample_data)
    
    # Initialize array to store all agent states
    # Shape: (T_observation, N_agents, state_dim)
    observation_trajectory = jnp.zeros((T_observation, n_agents, 4))  # state_dim = 4
    
    for i in range(n_agents):
        agent_key = f"agent_{i}"
        agent_states = normalized_data["trajectories"][agent_key]["states"]
        # Take first T_observation steps
        if len(agent_states) >= T_observation:
            agent_states_array = jnp.array(agent_states[:T_observation])  # (T_observation, state_dim)
        else:
            # Pad with last state if trajectory is too short
            agent_states_padded = agent_states[:]
            last_state = agent_states[-1] if agent_states else [0.0, 0.0, 0.0, 0.0]
            while len(agent_states_padded) < T_observation:
                agent_states_padded.append(last_state)
            agent_states_array = jnp.array(agent_states_padded[:T_observation])
        
        # Place in the correct position: (T_observation, N_agents, state_dim)
        observation_trajectory = observation_trajectory.at[:, i, :].set(agent_states_array)
    
    return observation_trajectory


def extract_reference_goals(sample_data: Dict[str, Any]) -> jnp.ndarray:
    """Extract reference goals from sample data."""
    # Use the target_positions field directly from the sample data
    if 'target_positions' in sample_data:
        return jnp.array(sample_data['target_positions'])
    else:
        # Fallback: extract from final trajectory positions
        goals = []
        for agent_idx in range(n_agents):
            agent_key = f"agent_{agent_idx}"
            agent_states = normalized_sample_data["trajectories"][agent_key]["states"]
            if len(agent_states) > 0:
                # Use the final position as the goal
                final_state = agent_states[-1]
                goals.append([final_state[0], final_state[1]])
            else:
                goals.append([0.0, 0.0])
        
        return jnp.array(goals)


def test_receding_horizon_with_models(sample_data: Dict[str, Any],
                                      psn_model: PlayerSelectionNetwork = None,
                                      psn_trained_state: Any = None,
                                      goal_model: GoalInferenceNetwork = None,
                                      goal_trained_state: Any = None,
                                      psn_model_path: str = None,
                                      use_baseline: bool = False,
                                      baseline_mode: str = None) -> Dict[str, Any]:

    
    """
    Test receding horizon planning with goal inference and player selection models.
    
    Args:
        sample_data: Reference trajectory sample data (will be normalized)
        psn_model: Trained PSN model (optional if use_baseline=True)
        psn_trained_state: Trained PSN model state (optional if use_baseline=True)
        goal_model: Trained goal inference model (optional if use_baseline=True)
        goal_trained_state: Trained goal inference model state (optional if use_baseline=True)
        psn_model_path: Path to PSN model (for logging)
        use_baseline: Whether to use baseline methods instead of PSN model
        baseline_mode: Baseline method to use (if use_baseline=True)
    
    Returns:
        Dictionary containing test results
    """
    print(f"    Testing receding horizon planning with models on sample {sample_data['sample_id']}")
    
    # Normalize sample data to handle different formats
    normalized_sample_data = normalize_sample_data(sample_data)
    
    # Initialize results storage
    results = {
        'sample_id': sample_data['sample_id'],
        'ego_agent_id': ego_agent_id,
        'goal_source': config.testing.receding_horizon.goal_source,
        'use_baseline': use_baseline,
        'baseline_mode': baseline_mode if use_baseline else None,
        'T_observation': T_observation,
        'T_total': T_total,
        'T_receding_horizon_planning': T_receding_horizon_planning,
        'T_receding_horizon_iterations': T_receding_horizon_iterations,
        'receding_horizon_results': [],
        'final_game_state': None,
        'computation_times': [],  # Per receding horizon step times
        'sample_computation_time': 0.0,  # Total time for entire sample
        'prediction_metrics': {},
        'planning_metrics': {}
    }
    
    # Initialize game state with ground truth trajectories for observation period
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
        # Add ground truth states for all agents
        for agent_idx in range(n_agents):
            agent_key = f"agent_{agent_idx}"
            agent_states = normalized_sample_data["trajectories"][agent_key]["states"]
            
            if step < len(agent_states):
                current_state = agent_states[step]
                current_game_state["trajectories"][agent_key]["states"].append(current_state)
            else:
                # If trajectory is too short, use last state
                last_state = agent_states[-1]
                current_game_state["trajectories"][agent_key]["states"].append(last_state)
    
    # Phase 2: Receding horizon planning with models (steps T_observation+1 to T_total)
    print(f"    Phase 2: Receding horizon planning with models (steps {T_observation+1} to {T_total})")
    print(f"        Initial stabilization: {config.testing.receding_horizon.initial_stabilization_iterations} iterations")
    
    # Start timing the entire receding horizon planning phase
    sample_start_time = time.time()
    
    # Initialize receding horizon trajectories
    receding_horizon_trajectories = [[] for _ in range(n_agents)]
    receding_horizon_states = [[] for _ in range(n_agents)]
    
    # Current states (start with states at end of observation period)
    current_states = []
    for agent_idx in range(n_agents):
        agent_key = f"agent_{agent_idx}"
        agent_states = current_game_state["trajectories"][agent_key]["states"]
        if len(agent_states) > 0:
            current_states.append(jnp.array(agent_states[-1]))
        else:
            # Fallback
            sample_states = normalized_sample_data["trajectories"][agent_key]["states"]
            current_states.append(jnp.array(sample_states[T_observation - 1] if len(sample_states) >= T_observation else sample_states[-1]))
    
    # Main receding horizon loop
    for iteration in range(T_receding_horizon_iterations):
        # Decide whether to use models or default values based on iteration
        if iteration < config.testing.receding_horizon.initial_stabilization_iterations:
            # First N iterations: Use ground truth goals and all agents (mask = 1)
            predicted_goals = extract_reference_goals(normalized_sample_data)
            predicted_mask = jnp.ones(n_agents - 1)  # All 1s for mask (no selection)
            num_selected = n_agents - 1  # All other agents selected
            mask_sparsity = 0.0  # No sparsity
            true_goals = predicted_goals  # Same as predicted for this phase
        else:
            # After N iterations: Use goal inference and player selection models (threshold: {config.testing.receding_horizon.mask_threshold})
            
            # Use goal source configuration to decide between true goals and goal inference
            goal_source = config.testing.receding_horizon.goal_source
            
            if goal_source == "true_goals":
                # Use true goals for PSN testing
                predicted_goals = extract_reference_goals(normalized_sample_data)
            elif goal_source == "goal_inference":
                # Always use first T_observation steps from the original ground truth trajectory
                goal_obs_traj = extract_observation_trajectory(normalized_sample_data)
                
                # Convert to input format for goal inference model
                goal_obs_input = goal_obs_traj.flatten().reshape(1, -1)
                predicted_goals = goal_model.apply({'params': goal_trained_state['params']}, goal_obs_input, deterministic=True)
                predicted_goals = predicted_goals[0].reshape(n_agents, 2)
            else:
                raise ValueError(f"Invalid goal_source: {goal_source}. Must be 'true_goals' or 'goal_inference'")
            
            # Get true goals for comparison
            true_goals = extract_reference_goals(normalized_sample_data)
            
            # Step 2: Infer player selection using current observation
            # Construct observation trajectory using LATEST 10 steps (sliding window)
            # Current total accumulated steps: T_observation + iteration
            current_total_steps = T_observation + iteration
            
            # Get the latest T_observation steps for PSN input
            obs_start_step = max(0, current_total_steps - T_observation)  # Start of observation window
            obs_end_step = current_total_steps  # End of observation window (exclusive)
            
            # PSN uses sliding window: latest T_observation steps
            # - Ego agent (0): accumulated computed trajectory from receding horizon solver
            # - Other agents: ground truth receding horizon trajectory
            
            obs_traj = []
            for obs_step in range(T_observation):
                actual_step = obs_start_step + obs_step
                step_states = []
                
                for agent_idx in range(n_agents):
                    agent_key = f"agent_{agent_idx}"
                    if agent_idx == 0:  # Ego agent: use computed accumulated trajectory
                        agent_states = current_game_state["trajectories"][agent_key]["states"]
                        if actual_step < len(agent_states):
                            step_states.append(agent_states[actual_step])
                        else:
                            # Use last available state if beyond current trajectory
                            last_state = agent_states[-1] if agent_states else [0.0, 0.0, 0.0, 0.0]
                            step_states.append(last_state)
                    else:  # Other agents: use ground truth receding horizon trajectory
                        agent_states = normalized_sample_data["trajectories"][agent_key]["states"]
                        if actual_step < len(agent_states):
                            step_states.append(agent_states[actual_step])
                        else:
                            # Use last state if trajectory is too short
                            last_state = agent_states[-1] if agent_states else [0.0, 0.0, 0.0, 0.0]
                            step_states.append(last_state)
                
                obs_traj.append(step_states)
            
            # Convert to array and reshape to (1, T_observation, n_agents, state_dim)
            obs_array = jnp.array(obs_traj)  # (T_observation, n_agents, state_dim)
            obs_input = obs_array.reshape(1, T_observation, n_agents, 4)  # (1, 10, 4, 4)
        
            # Use baseline or PSN model based on configuration
            if use_baseline:
                # Use baseline method
                mode_parameter = config.testing.receding_horizon.baseline_parameter
                trajectory_history = [np.array(current_game_state["trajectories"][f"agent_{i}"]["states"]) for i in range(n_agents)]
                
                
                if iteration > config.testing.receding_horizon.initial_stabilization_iterations:
                    prev_controls = [np.array(c) for c in results['receding_horizon_results'][-1]['first_controls']]
                else:
                    prev_controls = [np.zeros(2) for _ in range(n_agents)]

                predicted_mask = baseline_selection(
                    input_traj=obs_input,
                    trajectory=trajectory_history,
                    control=prev_controls,
                    mode=baseline_mode,
                    sim_step=iteration,
                    mode_parameter=mode_parameter
                )
            else:
                # Use PSN model
                if psn_model is None or psn_trained_state is None:
                    raise ValueError("PSN model and state must be provided when use_baseline=False")
                predicted_mask = psn_model.apply({'params': psn_trained_state['params']}, obs_input, deterministic=True)
                predicted_mask = predicted_mask[0]  # Remove batch dimension
            
            # Apply threshold to get selected agents
            mask_threshold = config.testing.receding_horizon.mask_threshold
            selected_agents = jnp.where(predicted_mask > mask_threshold)[0]
            num_selected = len(selected_agents)
            
            # Calculate mask sparsity based on configuration
            mask_sparsity = num_selected / (n_agents - 1)
        
        # Step 3: Solve receding horizon game with predicted goals
        # Apply masking: only include agents above threshold (EXACTLY like training script)
        if iteration >= config.testing.receding_horizon.initial_stabilization_iterations and 'predicted_mask' in locals() and predicted_mask is not None:
            # Filter agents and goals based on mask threshold (only after initial stabilization)
            mask_threshold = config.testing.receding_horizon.mask_threshold
            selected_agents = jnp.where(predicted_mask > mask_threshold)[0]
            
            # Ensure ego agent (agent 0) is always included
            if 0 not in selected_agents:
                selected_agents = jnp.concatenate([jnp.array([0]), selected_agents])
                selected_agents = jnp.unique(selected_agents)  # Remove duplicates
            
            # Filter current states and predicted goals to only include selected agents
            filtered_current_states = [current_states[i] for i in selected_agents]
            filtered_predicted_goals = predicted_goals[selected_agents]
            
            # Create agent setup for the filtered agents
            agents, reference_trajectories = create_agent_setup(filtered_current_states, filtered_predicted_goals)
            
            # Create loss functions
            loss_functions, linearize_loss_functions, compiled_functions = create_loss_functions(
                agents, reference_trajectories)
            
            # Solve the game with filtered agents
            first_controls, full_trajectories, game_time = solve_receding_horizon_game(
                agents, filtered_current_states, filtered_predicted_goals, compiled_functions)
            
            # Map results back to full agent list for compatibility with rest of code
            full_first_controls = [jnp.zeros(2) for _ in range(n_agents)]  # Default zero controls
            full_trajectories_expanded = [jnp.zeros((T_receding_horizon_planning, 4)) for _ in range(n_agents)]
            
            for i, agent_idx in enumerate(selected_agents):
                full_first_controls[agent_idx] = first_controls[i]
                full_trajectories_expanded[agent_idx] = full_trajectories[i]
            
            first_controls = full_first_controls
            full_trajectories = full_trajectories_expanded
        else:
            # No masking: use all agents (for initial stabilization or fallback)
            agents, reference_trajectories = create_agent_setup(current_states, predicted_goals)
            
            # Create loss functions
            loss_functions, linearize_loss_functions, compiled_functions = create_loss_functions(
                agents, reference_trajectories)
            
            # Solve the game
            first_controls, full_trajectories, game_time = solve_receding_horizon_game(
                agents, current_states, predicted_goals, compiled_functions)
        
        # Step 4: Store results for this iteration
        iteration_result = {
            'iteration': iteration,
            'step': T_observation + iteration,
            'current_states': [state.tolist() if hasattr(state, 'tolist') else list(state) for state in current_states],
            'predicted_goals': predicted_goals.tolist() if hasattr(predicted_goals, 'tolist') else predicted_goals,
            'true_goals': true_goals.tolist() if hasattr(true_goals, 'tolist') else true_goals,
            'predicted_mask': predicted_mask.tolist() if hasattr(predicted_mask, 'tolist') else predicted_mask,
            'num_selected': int(num_selected),
            'mask_sparsity': float(mask_sparsity),
            'first_controls': [control.tolist() if hasattr(control, 'tolist') else list(control) for control in first_controls],
            'full_trajectories': [traj.tolist() if hasattr(traj, 'tolist') else list(traj) for traj in full_trajectories],
            'game_solving_time': game_time
        }
        
        # Track computation time
        results['computation_times'].append(game_time)
        
        results['receding_horizon_results'].append(iteration_result)
        
        # Step 5: Apply first controls to move agents forward one step
        for i in range(n_agents):
            if i == 0:  # Ego agent: apply computed control and update state
                # Get current state and control
                current_state = current_states[i]
                control = first_controls[i]
                
                # Apply dynamics: x_{t+1} = x_t + dt * f(x_t, u_t)
                new_state = jnp.array([
                    current_state[0] + dt * current_state[2],  # x + dt * vx
                    current_state[1] + dt * current_state[3],  # y + dt * vy
                    current_state[2] + dt * control[0],        # vx + dt * ax
                    current_state[3] + dt * control[1]         # vy + dt * ay
                ])
                
                # Update current state for next iteration
                current_states[i] = new_state
                
                # Store in receding horizon trajectories
                receding_horizon_trajectories[i].append(new_state.tolist())
                receding_horizon_states[i].append(new_state.tolist())
                
                # Add to game state
                current_game_state["trajectories"][f"agent_{i}"]["states"].append(new_state.tolist())
            else:  # Other agents: use reference receding horizon trajectory
                # Get the reference trajectory for this agent at this step
                ref_step = T_observation + iteration
                if ref_step < len(normalized_sample_data["trajectories"][f"agent_{i}"]["states"]):
                    ref_state = normalized_sample_data["trajectories"][f"agent_{i}"]["states"][ref_step]
                else:
                    # If reference trajectory is too short, use last state
                    ref_state = normalized_sample_data["trajectories"][f"agent_{i}"]["states"][-1]
                
                # Update current state for next iteration
                current_states[i] = jnp.array(ref_state)
                
                # Store reference state in receding horizon trajectories
                receding_horizon_trajectories[i].append(ref_state)
                receding_horizon_states[i].append(ref_state)
                
                # Add to game state
                current_game_state["trajectories"][f"agent_{i}"]["states"].append(ref_state)
    
    # Store final game state
    results['final_game_state'] = current_game_state
    
    # Compute trajectory metrics (only for steps 10-50, excluding ground truth observation period)
    if results['receding_horizon_results']:
        # Extract ego trajectory (computed) and ground truth trajectory
        ego_computed_trajectory = jnp.array(current_game_state["trajectories"][f"agent_{ego_agent_id}"]["states"])
        ego_ground_truth_trajectory = jnp.array(normalized_sample_data["trajectories"][f"agent_{ego_agent_id}"]["states"])
        
        # Only analyze steps 10-50 (receding horizon planning phase)
        # Skip first 10 steps which use ground truth data
        analysis_start_step = T_observation  # Step 10
        analysis_end_step = T_total  # Step 50
        
        # Extract trajectories for analysis period only
        ego_computed_analysis = ego_computed_trajectory[analysis_start_step:analysis_end_step]
        ego_ground_truth_analysis = ego_ground_truth_trajectory[analysis_start_step:analysis_end_step]
        
        # Extract other agent ground truth trajectories for analysis period
        other_ground_truth_trajectories = []
        for i in range(n_agents):
            if i != ego_agent_id:
                other_traj = jnp.array(normalized_sample_data["trajectories"][f"agent_{i}"]["states"])
                other_traj_analysis = other_traj[analysis_start_step:analysis_end_step]
                other_ground_truth_trajectories.append(other_traj_analysis)
        
        # Extract ego controls (from receding horizon results) - these correspond to steps 10-50
        ego_controls = []
        for iter_result in results['receding_horizon_results']:
            ego_control = jnp.array(iter_result['first_controls'][ego_agent_id])
            ego_controls.append(ego_control)
        ego_controls = jnp.array(ego_controls)  # (T_receding_horizon_iterations, 2)
        
        # Extract ego goals
        ego_goals = jnp.array(extract_reference_goals(normalized_sample_data)[ego_agent_id])
        
        # Compute trajectory metrics for analysis period only
        trajectory_metrics = compute_trajectory_metrics(
            ego_computed_analysis,
            ego_ground_truth_analysis,
            other_ground_truth_trajectories,
            ego_controls,
            ego_goals,
            dt
        )
        
        # Store metrics
        results['prediction_metrics'] = {k: v for k, v in trajectory_metrics.items() if k in ['ade', 'fde']}
        results['planning_metrics'] = {k: v for k, v in trajectory_metrics.items() if k in ['navigation_cost', 'safety_cost', 'control_cost']}
        
        # Compute summary statistics
        goal_rmse_values = []
        mask_sparsity_values = []
        num_selected_values = []
        
        for iter_result in results['receding_horizon_results']:
            # Goal RMSE
            pred_goals = jnp.array(iter_result['predicted_goals'])
            true_goals = jnp.array(iter_result['true_goals'])
            goal_rmse = jnp.sqrt(jnp.mean(jnp.square(pred_goals - true_goals)))
            goal_rmse_values.append(float(goal_rmse))
            
            # Mask statistics
            mask_sparsity_values.append(iter_result['mask_sparsity'])
            num_selected_values.append(iter_result['num_selected'])
        
        results['goal_rmse'] = float(np.mean(goal_rmse_values))
        results['mask_sparsity'] = float(np.mean(mask_sparsity_values))
        results['num_selected_agents'] = float(np.mean(num_selected_values))
        
        # Compute mean computation time
        results['mean_computation_time'] = float(np.mean(results['computation_times']))
    else:
        results['goal_rmse'] = float('inf')
        results['mask_sparsity'] = 0.0
        results['num_selected_agents'] = 0.0
        results['mean_computation_time'] = 0.0
        results['prediction_metrics'] = {'ade': float('inf'), 'fde': float('inf')}
        results['planning_metrics'] = {'navigation_cost': float('inf'), 'safety_cost': float('inf'), 'control_cost': float('inf')}
    
    # End timing the entire receding horizon planning phase
    sample_end_time = time.time()
    results['sample_computation_time'] = sample_end_time - sample_start_time
    
    print(f"    ✓ Completed receding horizon planning with models")
    print(f"    ✓ Goal RMSE: {results['goal_rmse']:.4f}")
    print(f"    ✓ Mask Sparsity: {results['mask_sparsity']:.2f}")
    print(f"    ✓ Selected Agents: {results['num_selected_agents']:.1f}")
    print(f"    ✓ Mean Computation Time per Receding Horizon Step: {results['mean_computation_time']:.4f}s")
    print(f"    ✓ Total Computation Time per Sample: {results['sample_computation_time']:.4f}s")
    
    # Print prediction metrics (steps 10-50 only)
    if results['prediction_metrics']:
        print(f"    ✓ Prediction Metrics (steps {T_observation}-{T_total}):")
        print(f"        ADE: {results['prediction_metrics'].get('ade', 'N/A'):.4f}")
        print(f"        FDE: {results['prediction_metrics'].get('fde', 'N/A'):.4f}")
    
    # Print planning metrics (steps 10-50 only)
    if results['planning_metrics']:
        print(f"    ✓ Planning Metrics (steps {T_observation}-{T_total}):")
        print(f"        Navigation Cost: {results['planning_metrics'].get('navigation_cost', 'N/A'):.4f}")
        print(f"        Safety Cost: {results['planning_metrics'].get('safety_cost', 'N/A'):.4f}")
        print(f"        Control Cost: {results['planning_metrics'].get('control_cost', 'N/A'):.4f}")
    
    # Store normalized data for GIF creation
    results['normalized_sample_data'] = normalized_sample_data
    
    return results


def save_test_results(results: Dict[str, Any], save_dir: str) -> str:
    """Save test results to JSON file."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    filename = f"receding_horizon_test_sample_{results['sample_id']:03d}.json"
    filepath = save_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return str(filepath)


def create_receding_horizon_gif(sample_data: Dict[str, Any], 
                                results: Dict[str, Any], 
                                sample_id: int, 
                                save_dir: str,
                                normalized_sample_data: Dict[str, Any] = None) -> str:
    """
    Create a GIF visualization of the receding horizon trajectory evolution.
    
    Args:
        sample_data: Reference trajectory sample data
        results: Test results containing receding horizon data
        sample_id: Sample identifier
        save_dir: Directory to save the GIF
        normalized_sample_data: Normalized sample data (if None, will normalize sample_data)
    
    Returns:
        Path to the saved GIF file
    """
    print(f"      Creating trajectory visualization GIF...")
    
    if 'final_game_state' not in results:
        print(f"      Warning: No game state data for sample {sample_id}")
        return ""
    
    game_state = results['final_game_state']
    if not game_state.get('trajectories'):
        print(f"      Warning: Empty game state for sample {sample_id}")
        return ""
    
    # Use normalized data if provided, otherwise normalize the sample data
    if normalized_sample_data is None:
        normalized_sample_data = normalize_sample_data(sample_data)
    
    # Create frames for the GIF showing all trajectory steps
    frames = []
    
    # Color scheme
    ego_color = 'darkblue'
    other_agent_color = 'gray'
    selected_color = 'red'
    goal_prediction_color = 'orange'
    
    # Create exactly T_total frames (steps 1-T_total)
    total_steps = T_total
    
    # Create frames for all steps
    for step in range(total_steps):
        # Create frame for this step
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_aspect('equal')
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        
        # Title with step information
        if step < T_observation:
            step_title = f'Sample {sample_id+1}: Step {step+1}/{T_total}\nPhase 1: Ground Truth Trajectories'
        else:
            step_title = f'Sample {sample_id+1}: Step {step+1}/{T_total}\nPhase 2: Receding Horizon with Models'
        ax.set_title(step_title)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        
        # Plot trajectories up to current step for all agents
        # Visualization logic:
        # - Ego agent: Shows both computed trajectory (from solver) AND ground truth trajectory
        # - Other agents: Only show ground truth trajectories (not from solver)
        for i in range(n_agents):
            agent_key = f"agent_{i}"
            agent_states = game_state["trajectories"][agent_key]["states"]
            
            # Ensure we don't go beyond available trajectory data
            max_available_steps = len(agent_states)
            actual_step = min(step, max_available_steps - 1)
            
            if actual_step >= 0:
                if i == 0:  # Ego agent - plot both ground truth and computed trajectories
                    # Plot ground truth trajectory (black dashed)
                    sample_agent_states = normalized_sample_data["trajectories"][agent_key]["states"]
                    if len(sample_agent_states) > 0:
                        sample_traj = np.array(sample_agent_states[:T_total])
                        ax.plot(sample_traj[:, 0], sample_traj[:, 1], '--', 
                                 color='black', alpha=0.8, linewidth=2, 
                                 label=f'Ego Agent {i} (Ground Truth)')
                    
                    # Plot computed trajectory (blue solid)
                    agent_traj = np.array(agent_states[:actual_step+1])
                    if len(agent_traj) > 0:
                        ax.plot(agent_traj[:, 0], agent_traj[:, 1], '-', 
                                 color='blue', alpha=0.9, linewidth=3, 
                                 label=f'Ego Agent {i} (Computed)')
                        
                        # Force plot limits to include both trajectories
                        all_x = np.concatenate([sample_traj[:, 0], agent_traj[:, 0]])
                        all_y = np.concatenate([sample_traj[:, 1], agent_traj[:, 1]])
                        ax.set_xlim(min(ax.get_xlim()[0], all_x.min() - 0.1),
                                   max(ax.get_xlim()[1], all_x.max() + 0.1))
                        ax.set_ylim(min(ax.get_ylim()[0], all_y.min() - 0.1),
                                   max(ax.get_ylim()[1], all_y.max() + 0.1))
                else:  # Other agents - show ground truth trajectories (not from solver)
                    # Plot ground truth trajectory from sample data
                    sample_agent_states = normalized_sample_data["trajectories"][agent_key]["states"]
                    if len(sample_agent_states) > 0:
                        sample_traj = np.array(sample_agent_states[:T_total])
                        ax.plot(sample_traj[:, 0], sample_traj[:, 1], '-', 
                                 color=other_agent_color, alpha=0.6, linewidth=1, 
                                 label=f'Agent {i} (Ground Truth)')
        
        # Plot current positions with selection coloring
        # Note: Position markers show PSN selection status, but trajectories are as described above
        for i in range(n_agents):
            agent_key = f"agent_{i}"
            agent_states = game_state["trajectories"][agent_key]["states"]
            
            # Ensure we don't go beyond available trajectory data
            max_available_steps = len(agent_states)
            actual_step = min(step, max_available_steps - 1)
            
            if actual_step >= 0:
                current_pos = np.array(agent_states[actual_step][:2])
                
                # Determine if this agent is selected by PSN (for non-ego agents)
                # This is just for visualization coloring - all agents use ground truth trajectories
                is_selected = False
                if i > 0 and step >= T_observation:  # Only check selection after observation phase
                    # Find the iteration result for this step
                    iteration_idx = step - T_observation
                    if iteration_idx < len(results['receding_horizon_results']):
                        iteration_result = results['receding_horizon_results'][iteration_idx]
                        predicted_mask = iteration_result['predicted_mask']
                        mask_threshold = config.testing.receding_horizon.mask_threshold
                        mask_value = predicted_mask[i-1] if i-1 < len(predicted_mask) else 'N/A'
                        is_selected = mask_value > mask_threshold
                        
                if i == 0:  # Ego agent
                    ax.plot(current_pos[0], current_pos[1], 'o', 
                             color=ego_color, markersize=10, alpha=0.8)
                else:  # Other agents - color based on selection
                    if is_selected:
                        ax.plot(current_pos[0], current_pos[1], 'o', 
                                 color=selected_color, markersize=8, alpha=0.8)
                        # Add text label with selection indicator
                        ax.text(current_pos[0] + 0.1, current_pos[1] + 0.1, f'{i}*', 
                                fontsize=12, ha='left', va='bottom', 
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                    else:
                        ax.plot(current_pos[0], current_pos[1], 'o', 
                                 color=other_agent_color, markersize=8, alpha=0.7)
                        # Add text label
                        ax.text(current_pos[0] + 0.1, current_pos[1] + 0.1, f'{i}', 
                                fontsize=12, ha='left', va='bottom', 
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        # Plot goals and goal predictions
        true_goals = extract_reference_goals(normalized_sample_data)
        
        # Plot true goals
        for j in range(n_agents):
            if j == 0:  # Ego agent goal
                ax.plot(true_goals[j][0], true_goals[j][1], 's', 
                         color=ego_color, markersize=12, alpha=0.8, label=f'Ego Agent {j} Goal (True)')
            else:  # Other agent goals
                ax.plot(true_goals[j][0], true_goals[j][1], 's', 
                         color=other_agent_color, markersize=10, alpha=0.6, label=f'Agent {j} Goal (True)')
        
        # Plot predicted goals at current iteration (if available)
        if step >= T_observation:  # Only show predictions after observation phase
            iteration_idx = step - T_observation
            if iteration_idx < len(results['receding_horizon_results']):
                iteration_result = results['receding_horizon_results'][iteration_idx]
                predicted_goals = iteration_result['predicted_goals']
                
                for j in range(n_agents):
                    if j == 0:  # Ego agent predicted goal
                        ax.plot(predicted_goals[j][0], predicted_goals[j][1], '^', 
                                 color=goal_prediction_color, markersize=10, alpha=0.8, 
                                 label=f'Ego Agent {j} Goal (Predicted)')
                    else:  # Other agent predicted goals
                        ax.plot(predicted_goals[j][0], predicted_goals[j][1], '^', 
                                 color=goal_prediction_color, markersize=8, alpha=0.6, 
                                 label=f'Agent {j} Goal (Predicted)')
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Convert plot to image
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        # Use modern buffer_rgba() instead of deprecated tostring_rgb()
        image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (4,))  # RGBA has 4 channels
        # Convert RGBA to RGB by dropping alpha channel
        image = image[:, :, :3]
        frames.append(image)
        
        plt.close()
    
    # Verify we have exactly T_total frames
    expected_frames = T_total
    if len(frames) != expected_frames:
        print(f"      Warning: Expected {expected_frames} frames, but created {len(frames)} frames")
    
    # Save as GIF
    if frames:
        gif_path = os.path.join(save_dir, f"receding_horizon_test_sample_{sample_id:03d}.gif")
        import imageio
        imageio.mimsave(gif_path, frames, 
                       duration=config.testing.receding_horizon.gif_duration, 
                       loop=config.testing.receding_horizon.gif_loop)
        print(f"      GIF saved: {gif_path}")
        print(f"      Created {len(frames)} frames for all {T_total} steps")
        return gif_path
    else:
        print(f"      Warning: No frames created for GIF")
        return ""


def run_receding_horizon_testing(psn_model_path: str = None, 
                                goal_model_path: str = None,
                                reference_file: str = None,
                                output_dir: str = None,
                                num_samples: int = None,
                                use_baseline: bool = None,
                                baseline_mode: str = None) -> List[Dict[str, Any]]:
    """
    Run receding horizon testing with goal inference and player selection models.
    
    Args:
        psn_model_path: Path to trained PSN model (optional if use_baseline=True)
        goal_model_path: Path to trained goal inference model (optional if use_baseline=True)
        reference_file: Path to reference trajectory file (uses config if None)
        output_dir: Directory to save test results (uses config if None)
        num_samples: Number of samples to test (uses config if None)
        use_baseline: Whether to use baseline methods (uses config if None)
        baseline_mode: Baseline method to use (uses config if None)
    
    Returns:
        List of test results for each sample
    """
    print("=" * 80)
    print("RECEDING HORIZON TESTING WITH GOAL INFERENCE AND PLAYER SELECTION MODELS")
    print("=" * 80)
    
    # Load configuration values if not provided
    if reference_file is None:
        reference_file = config.testing.psn_data_dir
    if num_samples is None:
        num_samples = config.testing.receding_horizon.num_samples
    if use_baseline is None:
        use_baseline = config.testing.receding_horizon.use_baseline
    if baseline_mode is None:
        baseline_mode = config.testing.receding_horizon.baseline_mode
    
    # Determine output directory based on goal source and input method if not provided
    if output_dir is None:
        goal_source = config.testing.receding_horizon.goal_source
        if use_baseline:
            # For baseline methods, save to baseline_results directory with method name and goal source
            method_name = baseline_mode.lower().replace(' ', '_').replace('_', '')
            if goal_source == "goal_inference":
                output_dir = f"baseline_results/receding_horizon_results_{method_name}_goal_inference"
            else:
                output_dir = f"baseline_results/receding_horizon_results_{method_name}_goal_true"
        else:
            if goal_source == "true_goals":
                # Extract method name from PSN model path if available
                if psn_model_path:
                    psn_model_name = os.path.basename(psn_model_path).replace('.pkl', '')
                    output_dir = f"receding_horizon_results_goal_true_{psn_model_name}"
                else:
                    output_dir = "receding_horizon_results_goal_true"
            elif goal_source == "goal_inference":
                # Always use first_steps for goal inference
                if psn_model_path:
                    psn_model_name = os.path.basename(psn_model_path).replace('.pkl', '')
                    output_dir = f"receding_horizon_results_goal_inference_{psn_model_name}"
                else:
                    output_dir = "receding_horizon_results_goal_inference"
            else:
                output_dir = "receding_horizon_test_results"
    
    # Load models based on goal source and baseline configuration
    psn_model, psn_trained_state, goal_model, goal_trained_state = None, None, None, None
    
    # Determine goal source
    goal_source = config.testing.receding_horizon.goal_source
    
    if goal_source == "goal_inference":
        # Load goal inference model regardless of baseline/PSN choice
        print(f"Loading goal inference model...")
        if goal_model_path is None:
            raise ValueError("Goal model path must be provided when goal_source='goal_inference'")
        _, _, goal_model, goal_trained_state = load_trained_models(None, goal_model_path)
        print(f"✓ Goal inference model loaded successfully")
    
    if not use_baseline:
        # Load PSN model only when not using baseline
        print(f"Loading PSN model...")
        if psn_model_path is None:
            raise ValueError("PSN model path must be provided when use_baseline=False")
        psn_model, psn_trained_state, _, _ = load_trained_models(psn_model_path, None)
        print(f"✓ PSN model loaded successfully")
    else:
        print(f"Using baseline method: {baseline_mode}")
        print(f"✓ Baseline mode configured")
    
    # Load reference data
    print(f"Loading reference data from: {reference_file}")
    import glob
    
    # Find all receding_horizon_sample_*.json files in the directory
    pattern = os.path.join(reference_file, "receding_horizon_sample_*.json")
    json_files = sorted(glob.glob(pattern))
    
    if not json_files:
        raise FileNotFoundError(f"No receding_horizon_sample_*.json files found in directory: {reference_file}")
    
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
    
    print(f"Loaded {len(reference_data)} reference samples")
    
    # Select samples based on configuration
    if config.testing.receding_horizon.use_later_samples:
        # Use later 128 samples (samples 384-511)
        train_samples = config.training.train_samples  # 384
        start_idx = train_samples  # Start from sample 384
        end_idx = start_idx + num_samples  # End at sample 384 + 128 = 512
        test_samples = reference_data[start_idx:end_idx]
        print(f"Using later samples {start_idx}-{end_idx-1} ({len(test_samples)} samples)")
    else:
        # Use first N samples (original behavior)
        test_samples = reference_data[:min(num_samples, len(reference_data))]
        print(f"Using first {len(test_samples)} samples")
    
    # Test each sample
    all_results = []
    for i, sample_data in enumerate(test_samples):
        print(f"\nTesting sample {i+1}/{len(test_samples)}...")
        
        try:
            # Run receding horizon testing with models
            results = test_receding_horizon_with_models(
                sample_data, psn_model, psn_trained_state, goal_model, goal_trained_state, 
                psn_model_path, use_baseline, baseline_mode)
            
            # Save results
            filepath = save_test_results(results, output_dir)
            print(f"  ✓ Results saved to: {filepath}")
            
            # Create trajectory visualization GIF  
            gif_path = create_receding_horizon_gif(sample_data, results, sample_data['sample_id'], output_dir, results.get('normalized_sample_data'))
            if gif_path:
                print(f"  ✓ GIF visualization created: {gif_path}")
            
            all_results.append(results)
            
        except Exception as e:
            print(f"  ✗ Error testing sample {i}: {str(e)}")
            continue
    
    # Print summary
    print(f"\n" + "=" * 80)
    print("TESTING SUMMARY")
    print("=" * 80)
    print(f"Successfully tested: {len(all_results)}/{len(test_samples)} samples")
    
    if all_results:
        goal_rmse_values = [r['goal_rmse'] for r in all_results if r['goal_rmse'] != float('inf')]
        mask_sparsity_values = [r['mask_sparsity'] for r in all_results]
        num_selected_values = [r['num_selected_agents'] for r in all_results]
        computation_times = [r['mean_computation_time'] for r in all_results if r['mean_computation_time'] > 0]
        
        if goal_rmse_values:
            print(f"Goal Prediction RMSE: {np.mean(goal_rmse_values):.4f} ± {np.std(goal_rmse_values):.4f}")
        print(f"Mask Sparsity: {np.mean(mask_sparsity_values):.3f} ± {np.std(mask_sparsity_values):.3f}")
        print(f"Average Selected Agents: {np.mean(num_selected_values):.2f} ± {np.std(num_selected_values):.2f}")
        if computation_times:
            print(f"Mean Computation Time per Receding Horizon Step: {np.mean(computation_times):.4f}s ± {np.std(computation_times):.4f}s")
        
        # Per-sample computation times
        sample_computation_times = [r['sample_computation_time'] for r in all_results if r['sample_computation_time'] > 0]
        if sample_computation_times:
            print(f"Mean Computation Time per Sample: {np.mean(sample_computation_times):.4f}s ± {np.std(sample_computation_times):.4f}s")
        
        # Print prediction metrics summary (steps 10-50 only)
        if config.testing.receding_horizon.compute_prediction_metrics:
            ade_values = [r['prediction_metrics'].get('ade', float('inf')) for r in all_results if r['prediction_metrics'].get('ade', float('inf')) != float('inf')]
            fde_values = [r['prediction_metrics'].get('fde', float('inf')) for r in all_results if r['prediction_metrics'].get('fde', float('inf')) != float('inf')]
            
            if ade_values:
                print(f"ADE (steps {T_observation}-{T_total}): {np.mean(ade_values):.4f} ± {np.std(ade_values):.4f}")
            if fde_values:
                print(f"FDE (steps {T_observation}-{T_total}): {np.mean(fde_values):.4f} ± {np.std(fde_values):.4f}")
        
        # Print planning metrics summary (steps 10-50 only)
        if config.testing.receding_horizon.compute_planning_metrics:
            nav_cost_values = [r['planning_metrics'].get('navigation_cost', float('inf')) for r in all_results if r['planning_metrics'].get('navigation_cost', float('inf')) != float('inf')]
            safety_cost_values = [r['planning_metrics'].get('safety_cost', float('inf')) for r in all_results if r['planning_metrics'].get('safety_cost', float('inf')) != float('inf')]
            control_cost_values = [r['planning_metrics'].get('control_cost', float('inf')) for r in all_results if r['planning_metrics'].get('control_cost', float('inf')) != float('inf')]
            
            if nav_cost_values:
                print(f"Navigation Cost (steps {T_observation}-{T_total}): {np.mean(nav_cost_values):.4f} ± {np.std(nav_cost_values):.4f}")
            if safety_cost_values:
                print(f"Safety Cost (steps {T_observation}-{T_total}): {np.mean(safety_cost_values):.4f} ± {np.std(safety_cost_values):.4f}")
            if control_cost_values:
                print(f"Control Cost (steps {T_observation}-{T_total}): {np.mean(control_cost_values):.4f} ± {np.std(control_cost_values):.4f}")
    
    print(f"Results saved to: {output_dir}")
    
    return all_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RECEDING HORIZON TESTING WITH GOAL INFERENCE AND PLAYER SELECTION MODELS")
    print("=" * 80)
    
    # Load configuration
    use_baseline = config.testing.receding_horizon.use_baseline
    baseline_mode = config.testing.receding_horizon.baseline_mode
    goal_source = config.testing.receding_horizon.goal_source
    
    # Initialize model paths
    psn_model_path = None
    goal_model_path = None
    
    # Set goal model path if goal source is goal_inference (regardless of baseline mode)
    if goal_source == "goal_inference":
        goal_model_path = f"log/goal_inference_rh_gru_N_{config.game.N_agents}_T_{config.game.T_total}_obs_{config.goal_inference.observation_length}_lr_{config.goal_inference.learning_rate}_bs_{config.goal_inference.batch_size}_goal_loss_weight_{config.goal_inference.goal_loss_weight}_epochs_{config.goal_inference.num_epochs}/goal_inference_rh_best_model.pkl"
        
        # Check if goal model exists
        if not os.path.exists(goal_model_path):
            print(f"Error: Goal inference model not found at: {goal_model_path}")
            print("Please train a goal inference model first using: python3 goal_inference/pretrain_goal_inference.py")
            exit(1)
    
    if not use_baseline:
        # Model paths - PSN from goal_true directory
        # PSN was trained with true goals, so load from goal_true_xxx directory
        psn_model_path = f"log/goal_true_N_{config.game.N_agents}_T_{config.game.T_total}_obs_{config.goal_inference.observation_length}/psn_gru_true_goals_N_{config.game.N_agents}_T_{config.game.T_total}_obs_{config.goal_inference.observation_length}_lr_{config.psn.learning_rate}_bs_{config.psn.batch_size}_sigma1_{config.psn.sigma1}_sigma2_{config.psn.sigma2}_epochs_{config.psn.num_epochs}/psn_best_model.pkl"
        
        # Check if PSN model exists
        if not os.path.exists(psn_model_path):
            print(f"Error: PSN model not found at: {psn_model_path}")
            print("Please train a PSN model first using: python3 player_selection_network/psn_training_with_pretrained_goals.py")
            exit(1)
    
    # Use PSN-specific testing data directory (receding horizon trajectories)
    reference_file = config.testing.psn_data_dir
    
    # Create output directory based on configuration with method name
    if use_baseline:
        # For baseline methods, create hierarchical directory structure
        method_name = baseline_mode.lower().replace(' ', '_').replace('_', '')
        
        # Determine if this is prediction or planning test based on metrics computed
        test_type = "planning"  # Default to planning since we always compute planning metrics
        if config.testing.receding_horizon.compute_prediction_metrics and not config.testing.receding_horizon.compute_planning_metrics:
            test_type = "prediction"
        
        # Create directory structure: baseline_results/test_type/N_agents/method_param_goal_source
        n_agents = config.game.N_agents
        baseline_param = config.testing.receding_horizon.baseline_parameter
        goal_suffix = "goal_inference" if goal_source == "goal_inference" else "goal_true"
        output_dir = f"baseline_results/{test_type}/N_{n_agents}/receding_horizon_results_{method_name}_param_{baseline_param}_{goal_suffix}"
    else:
        # For PSN model, create directory under the PSN model directory with method name
        psn_model_dir = os.path.dirname(psn_model_path)
        
        # Extract method name from PSN model path
        psn_model_name = os.path.basename(psn_model_path).replace('.pkl', '')
        
        if goal_source == "true_goals":
            output_dir = os.path.join(psn_model_dir, f"receding_horizon_results_goal_true_{psn_model_name}")
        elif goal_source == "goal_inference":
            # Always use first_steps for goal inference
            output_dir = os.path.join(psn_model_dir, f"receding_horizon_results_goal_inference_{psn_model_name}")
        else:
            # Fallback to original directory name with method name
            output_dir = os.path.join(psn_model_dir, f"{config.testing.receding_horizon.output_dir}_{psn_model_name}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Use Baseline: {use_baseline}")
    if use_baseline:
        print(f"  Baseline Mode: {baseline_mode}")
        print(f"  Baseline Parameter: {config.testing.receding_horizon.baseline_parameter}")
    else:
        print(f"  PSN Model: {psn_model_path}")
        print(f"  Goal Model: {goal_model_path}")
    print(f"  Reference Data: {reference_file}")
    print(f"  Goal Source: {goal_source}")
    print(f"  Number of Samples: {config.testing.receding_horizon.num_samples}")
    print(f"  Use Later Samples: {config.testing.receding_horizon.use_later_samples}")
    print(f"  Results will be saved to: {output_dir}")
    
    # Run testing
    results = run_receding_horizon_testing(
        psn_model_path=psn_model_path,
        goal_model_path=goal_model_path,
        reference_file=reference_file,
        output_dir=output_dir,
        num_samples=config.testing.receding_horizon.num_samples,
        use_baseline=use_baseline,
        baseline_mode=baseline_mode
    )
    
    # Create summary file explaining the model relationships
    summary_path = os.path.join(output_dir, "test_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Receding Horizon Testing with Integrated Models\n")
        f.write("=" * 60 + "\n\n")
        f.write("Model Configuration:\n")
        if use_baseline:
            f.write(f"  - Baseline Mode: {baseline_mode}\n")
            f.write(f"  - Baseline Parameter: {config.testing.receding_horizon.baseline_parameter}\n")
        else:
            f.write(f"  - Goal Inference Model: {goal_model_path}\n")
            f.write(f"  - PSN Model: {psn_model_path}\n")
        f.write(f"  - Reference Data: {reference_file}\n\n")
        f.write("Test Configuration:\n")
        f.write(f"  - Use Baseline: {use_baseline}\n")
        f.write(f"  - Goal source: {config.testing.receding_horizon.goal_source}\n")
        f.write(f"  - Number of samples: {config.testing.receding_horizon.num_samples}\n")
        f.write(f"  - Use later samples: {config.testing.receding_horizon.use_later_samples}\n")
        f.write(f"  - Receding horizon iterations: {T_receding_horizon_iterations}\n")
        f.write(f"  - Planning horizon per game: {T_receding_horizon_planning}\n")
        f.write(f"  - Total trajectory steps: {T_total}\n")
        f.write(f"  - Observation steps: {T_observation}\n")
        f.write(f"  - Number of agents: {n_agents}\n")
        f.write(f"  - Compute prediction metrics: {config.testing.receding_horizon.compute_prediction_metrics}\n")
        f.write(f"  - Compute planning metrics: {config.testing.receding_horizon.compute_planning_metrics}\n\n")
        f.write("Results:\n")
        f.write(f"  - Successfully tested: {len(results)} samples\n")
        f.write(f"  - Output directory: {output_dir}\n\n")
        
        # Add metrics summary if available (steps 10-50 only)
        if results:
            f.write("Metrics Summary (steps 10-50, receding horizon planning phase only):\n")
            # Prediction metrics
            if config.testing.receding_horizon.compute_prediction_metrics:
                ade_values = [r['prediction_metrics'].get('ade', float('inf')) for r in results if r['prediction_metrics'].get('ade', float('inf')) != float('inf')]
                fde_values = [r['prediction_metrics'].get('fde', float('inf')) for r in results if r['prediction_metrics'].get('fde', float('inf')) != float('inf')]
                if ade_values:
                    f.write(f"  - ADE (steps {T_observation}-{T_total}): {np.mean(ade_values):.4f} ± {np.std(ade_values):.4f}\n")
                if fde_values:
                    f.write(f"  - FDE (steps {T_observation}-{T_total}): {np.mean(fde_values):.4f} ± {np.std(fde_values):.4f}\n")
            
            # Planning metrics
            if config.testing.receding_horizon.compute_planning_metrics:
                nav_cost_values = [r['planning_metrics'].get('navigation_cost', float('inf')) for r in results if r['planning_metrics'].get('navigation_cost', float('inf')) != float('inf')]
                safety_cost_values = [r['planning_metrics'].get('safety_cost', float('inf')) for r in results if r['planning_metrics'].get('safety_cost', float('inf')) != float('inf')]
                control_cost_values = [r['planning_metrics'].get('control_cost', float('inf')) for r in results if r['planning_metrics'].get('control_cost', float('inf')) != float('inf')]
                if nav_cost_values:
                    f.write(f"  - Navigation Cost (steps {T_observation}-{T_total}): {np.mean(nav_cost_values):.4f} ± {np.std(nav_cost_values):.4f}\n")
                if safety_cost_values:
                    f.write(f"  - Safety Cost (steps {T_observation}-{T_total}): {np.mean(safety_cost_values):.4f} ± {np.std(safety_cost_values):.4f}\n")
                if control_cost_values:
                    f.write(f"  - Control Cost (steps {T_observation}-{T_total}): {np.mean(control_cost_values):.4f} ± {np.std(control_cost_values):.4f}\n")
            
            # Computation time
            computation_times = [r['mean_computation_time'] for r in results if r['mean_computation_time'] > 0]
            if computation_times:
                f.write(f"  - Mean Computation Time per Receding Horizon Step: {np.mean(computation_times):.4f}s ± {np.std(computation_times):.4f}s\n")
            
            # Per-sample computation time
            sample_computation_times = [r['sample_computation_time'] for r in results if r['sample_computation_time'] > 0]
            if sample_computation_times:
                f.write(f"  - Mean Computation Time per Sample: {np.mean(sample_computation_times):.4f}s ± {np.std(sample_computation_times):.4f}s\n")
        
        f.write("\nDirectory Structure:\n")
        if not use_baseline:
            f.write(f"  Goal Inference: {os.path.dirname(goal_model_path)}\n")
            f.write(f"  PSN Training: {os.path.dirname(psn_model_path)}\n")
            f.write(f"  Receding Horizon Test: {output_dir}\n")
        else:
            f.write(f"  Baseline Results → {test_type} → N_{n_agents} → {os.path.basename(output_dir)}\n")
            f.write(f"  Method: {baseline_mode}\n")
            f.write(f"  Parameter: {baseline_param}\n")
            f.write(f"  Goal Source: {goal_source}\n")
    
    print(f"\nReceding horizon testing completed!")
    print(f"Generated {len(results)} test results with integrated models.")
    print(f"Results saved to: {output_dir}")
    print(f"Summary file: {summary_path}")
    print(f"\nDirectory organization:")
    if use_baseline:
        print(f"  Baseline Results → {test_type} → N_{n_agents} → {os.path.basename(output_dir)}")
        print(f"  Method: {baseline_mode}")
        print(f"  Parameter: {baseline_param}")
        print(f"  Goal Source: {goal_source}")
    else:
        print(f"  Goal Inference → PSN Training → Receding Horizon Test")
        if goal_model_path and psn_model_path:
            print(f"  {os.path.basename(os.path.dirname(goal_model_path))} → {os.path.basename(os.path.dirname(psn_model_path))} → {os.path.basename(output_dir)}")
        else:
            print(f"  Models → {os.path.basename(output_dir)}")
