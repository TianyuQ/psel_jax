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
                                     psn_model: PlayerSelectionNetwork,
                                     psn_trained_state: Any,
                                     goal_model: GoalInferenceNetwork,
                                     goal_trained_state: Any,
                                     psn_model_path: str = None) -> Dict[str, Any]:
    # TODO: write option here.

    """
    Test receding horizon planning with goal inference and player selection models.
    
    Args:
        sample_data: Reference trajectory sample data (will be normalized)
        psn_model: Trained PSN model
        psn_trained_state: Trained PSN model state
        goal_model: Trained goal inference model
        goal_trained_state: Trained goal inference model state
    
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
        'goal_inference_input_method': getattr(config.testing.receding_horizon, 'goal_inference_input_method', 'first_steps') if config.testing.receding_horizon.goal_source == "goal_inference" else None,
        'T_observation': T_observation,
        'T_total': T_total,
        'T_receding_horizon_planning': T_receding_horizon_planning,
        'T_receding_horizon_iterations': T_receding_horizon_iterations,
        'receding_horizon_results': [],
        'final_game_state': None
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
    print(f"      Initial stabilization: {config.testing.receding_horizon.initial_stabilization_iterations} iterations")
    
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
                # Check input method for goal inference
                input_method = getattr(config.testing.receding_horizon, 'goal_inference_input_method', 'first_steps')
                
                if input_method == "first_steps":
                    # Use first T_observation steps from the original ground truth trajectory
                    goal_obs_traj = extract_observation_trajectory(normalized_sample_data)
                    
                elif input_method == "sliding_window":
                    # Use sliding window: latest T_observation steps from current game state
                    current_total_steps = T_observation + iteration
                    obs_start_step = max(0, current_total_steps - T_observation)  # Start of observation window
                    
                    # Build sliding window observation trajectory (same as PSN input construction)
                    goal_obs_traj = []
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
                        
                        goal_obs_traj.append(step_states)
                    
                    # Convert to array format
                    goal_obs_traj = jnp.array(goal_obs_traj)  # (T_observation, N_agents, state_dim)
                    
                else:
                    raise ValueError(f"Invalid goal_inference_input_method: {input_method}. Must be 'first_steps' or 'sliding_window'")
                
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
            
            # To Eric, how to select players:
            # selected players is the vector of selected players' index.
            # selected_agents = Baseline_Selection(obs_input, masked_threhold, desired_number_of_agents, option)

            predicted_mask = psn_model.apply({'params': psn_trained_state['params']}, obs_input, deterministic=True)
            predicted_mask = predicted_mask[0]  # Remove batch dimension
            
            # Apply threshold to get selected agents
            mask_threshold = config.testing.receding_horizon.mask_threshold
            selected_agents = jnp.where(predicted_mask > mask_threshold)[0]
            num_selected = len(selected_agents)
            
            # Calculate mask sparsity based on configuration
            if config.testing.receding_horizon.mask_sparsity_calculation == "fraction":
                mask_sparsity = num_selected / n_agents
            else:  # "ratio" - original calculation
                mask_sparsity = 1.0 - (num_selected / (n_agents - 1))
        
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
    
    # Compute summary statistics
    if results['receding_horizon_results']:
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
    else:
        results['goal_rmse'] = float('inf')
        results['mask_sparsity'] = 0.0
        results['num_selected_agents'] = 0.0
    
    print(f"    ✓ Completed receding horizon planning with models")
    print(f"    ✓ Goal RMSE: {results['goal_rmse']:.4f}")
    print(f"    ✓ Mask Sparsity: {results['mask_sparsity']:.2f}")
    print(f"    ✓ Selected Agents: {results['num_selected_agents']:.1f}")
    
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


def run_receding_horizon_testing(psn_model_path: str, 
                                goal_model_path: str,
                                reference_file: str,
                                output_dir: str = None,
                                num_samples: int = 5) -> List[Dict[str, Any]]:
    """
    Run receding horizon testing with goal inference and player selection models.
    
    Args:
        psn_model_path: Path to trained PSN model
        goal_model_path: Path to trained goal inference model
        reference_file: Path to reference trajectory file
        output_dir: Directory to save test results
        num_samples: Number of samples to test
    
    Returns:
        List of test results for each sample
    """
    print("=" * 80)
    print("RECEDING HORIZON TESTING WITH GOAL INFERENCE AND PLAYER SELECTION MODELS")
    print("=" * 80)
    
    # Determine output directory based on goal source and input method if not provided
    if output_dir is None:
        goal_source = config.testing.receding_horizon.goal_source
        if goal_source == "true_goals":
            output_dir = "receding_horizon_results_goal_true"
        elif goal_source == "goal_inference":
            # Check input method for goal inference
            input_method = getattr(config.testing.receding_horizon, 'goal_inference_input_method', 'first_steps')
            if input_method == "sliding_window":
                output_dir = "receding_horizon_results_goal_inference_sliding_window"
            else:  # first_steps
                output_dir = "receding_horizon_results_goal_inference"
        else:
            output_dir = "receding_horizon_test_results"
    
    # Load models
    print(f"Loading trained models...")
    psn_model, psn_trained_state, goal_model, goal_trained_state = load_trained_models(
        psn_model_path, goal_model_path)
    
    print(f"✓ Models loaded successfully")
    
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
    
    # Limit number of samples for testing
    test_samples = reference_data[:min(num_samples, len(reference_data))]
    print(f"Testing on {len(test_samples)} samples")
    
    # Test each sample
    all_results = []
    for i, sample_data in enumerate(test_samples):
        print(f"\nTesting sample {i+1}/{len(test_samples)}...")
        
        try:
            # Run receding horizon testing with models
            results = test_receding_horizon_with_models(
                sample_data, psn_model, psn_trained_state, goal_model, goal_trained_state, psn_model_path)
            
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
        
        if goal_rmse_values:
            print(f"Goal Prediction RMSE: {np.mean(goal_rmse_values):.4f} ± {np.std(goal_rmse_values):.4f}")
        print(f"Mask Sparsity: {np.mean(mask_sparsity_values):.3f} ± {np.std(mask_sparsity_values):.3f}")
        print(f"Average Selected Agents: {np.mean(num_selected_values):.2f} ± {np.std(num_selected_values):.2f}")
    
    print(f"Results saved to: {output_dir}")
    
    return all_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RECEDING HORIZON TESTING WITH GOAL INFERENCE AND PLAYER SELECTION MODELS")
    print("=" * 80)
    
    # Model paths - PSN from goal_true directory, Goal inference from goal_inference_gru directory
    # PSN was trained with true goals, so load from goal_true_xxx directory
    psn_model_path = f"log/goal_true_N_{config.game.N_agents}_T_{config.game.T_total}_obs_{config.goal_inference.observation_length}/psn_gru_true_goals_N_{config.game.N_agents}_T_{config.game.T_total}_obs_{config.goal_inference.observation_length}_lr_{config.psn.learning_rate}_bs_{config.psn.batch_size}_sigma1_{config.psn.sigma1}_sigma2_{config.psn.sigma2}_epochs_{config.psn.num_epochs}/psn_best_model.pkl"
    
    # Goal inference model from goal_inference_gru_xxx directory
    goal_model_path = f"log/goal_inference_rh_gru_N_{config.game.N_agents}_T_{config.game.T_total}_obs_{config.goal_inference.observation_length}_lr_{config.goal_inference.learning_rate}_bs_{config.goal_inference.batch_size}_goal_loss_weight_{config.goal_inference.goal_loss_weight}_epochs_{config.goal_inference.num_epochs}/goal_inference_rh_best_model.pkl"
    
    # Check if models exist
    if psn_model_path is None or not os.path.exists(psn_model_path):
        print(f"Error: PSN model not found at: {psn_model_path}")
        print("Please train a PSN model first using: python3 player_selection_network/psn_training_with_pretrained_goals.py")
        exit(1)
    
    if goal_model_path is None or not os.path.exists(goal_model_path):
        print(f"Error: Goal inference model not found at: {goal_model_path}")
        print("Please train a goal inference model first using: python3 goal_inference/pretrain_goal_inference.py")
        exit(1)
    
    # Use PSN-specific testing data directory (receding horizon trajectories)
    reference_file = config.testing.psn_data_dir
    
    # Create output directory under the PSN model directory based on goal source and input method
    psn_model_dir = os.path.dirname(psn_model_path)
    goal_source = config.testing.receding_horizon.goal_source
    
    if goal_source == "true_goals":
        output_dir = os.path.join(psn_model_dir, "receding_horizon_results_goal_true")
    elif goal_source == "goal_inference":
        # Check input method for goal inference
        input_method = getattr(config.testing.receding_horizon, 'goal_inference_input_method', 'first_steps')
        if input_method == "sliding_window":
            output_dir = os.path.join(psn_model_dir, "receding_horizon_results_goal_inference_sliding_window")
        else:  # first_steps
            output_dir = os.path.join(psn_model_dir, "receding_horizon_results_goal_inference")
    else:
        # Fallback to original directory name
        output_dir = os.path.join(psn_model_dir, config.testing.receding_horizon.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Using models:")
    print(f"  PSN Model: {psn_model_path}")
    print(f"  Goal Model: {goal_model_path}")
    print(f"  Reference Data: {reference_file}")
    print(f"  Goal Source: {goal_source}")
    if goal_source == "goal_inference":
        input_method = getattr(config.testing.receding_horizon, 'goal_inference_input_method', 'first_steps')
        print(f"  Goal Inference Input Method: {input_method}")
    print(f"  Results will be saved to: {output_dir}")
    
    # Run testing
    results = run_receding_horizon_testing(
        psn_model_path=psn_model_path,
        goal_model_path=goal_model_path,
        reference_file=reference_file,
        output_dir=output_dir,
        num_samples=config.testing.receding_horizon.num_samples
    )
    
    # Create summary file explaining the model relationships
    summary_path = os.path.join(output_dir, "test_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Receding Horizon Testing with Integrated Models\n")
        f.write("=" * 60 + "\n\n")
        f.write("Model Configuration:\n")
        f.write(f"  - Goal Inference Model: {goal_model_path}\n")
        f.write(f"  - PSN Model: {psn_model_path}\n")
        f.write(f"  - Reference Data: {reference_file}\n\n")
        f.write("Test Configuration:\n")
        f.write(f"  - Goal source: {config.testing.receding_horizon.goal_source}\n")
        if config.testing.receding_horizon.goal_source == "goal_inference":
            input_method = getattr(config.testing.receding_horizon, 'goal_inference_input_method', 'first_steps')
            f.write(f"  - Goal inference input method: {input_method}\n")
        f.write(f"  - Number of samples: {config.testing.receding_horizon.num_samples}\n")
        f.write(f"  - Receding horizon iterations: {T_receding_horizon_iterations}\n")
        f.write(f"  - Planning horizon per game: {T_receding_horizon_planning}\n")
        f.write(f"  - Total trajectory steps: {T_total}\n")
        f.write(f"  - Observation steps: {T_observation}\n")
        f.write(f"  - Number of agents: {n_agents}\n\n")
        f.write("Results:\n")
        f.write(f"  - Successfully tested: {len(results)} samples\n")
        f.write(f"  - Output directory: {output_dir}\n\n")
        f.write("Directory Structure:\n")
        f.write(f"  Goal Inference: {os.path.dirname(goal_model_path)}\n")
        f.write(f"  PSN Training: {psn_model_dir}\n")
        f.write(f"  Receding Horizon Test: {output_dir}\n")
    
    print(f"\nReceding horizon testing completed!")
    print(f"Generated {len(results)} test results with integrated models.")
    print(f"Results saved to: {output_dir}")
    print(f"Summary file: {summary_path}")
    print(f"\nDirectory organization:")
    print(f"  Goal Inference → PSN Training → Receding Horizon Test")
    print(f"  {os.path.basename(os.path.dirname(goal_model_path))} → {os.path.basename(psn_model_dir)} → {os.path.basename(output_dir)}")
