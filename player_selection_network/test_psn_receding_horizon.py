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
    
    # Cost function weights (same for all agents) - exactly like original ilqgames_example
    Q = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001]))  # State cost weights (position, position, velocity, velocity)
    R = jnp.diag(jnp.array([0.01, 0.01]))               # Control cost weights (ax, ay)
    
    for i in range(n_agents):
        # Create agent
        agent = PointAgent(dt=dt, x_dim=4, u_dim=2, Q=Q, R=R)
        agents.append(agent)
        
        # Reference trajectory (simple linear interpolation like original example)
        # Create a straight-line reference trajectory from initial position to target
        start_pos = jnp.array(initial_states[i][:2])  # Extract x, y position and convert to array
        target_pos = jnp.array(target_positions[i])   # target_positions[i] is already [x, y]
        
        # Create straight-line reference trajectory from initial position to target
        
        # Linear interpolation over time steps (exactly like original ilqgames_example)
        ref_traj = jnp.linspace(start_pos, target_pos, T_receding_horizon_planning)
        reference_trajectories.append(ref_traj)
    
    return agents, reference_trajectories


def create_loss_functions(agents: list, reference_trajectories: list) -> tuple:
    """
    Create loss functions for each agent based on their reference trajectories.
    
    Args:
        agents: List of agent objects
        reference_trajectories: Reference trajectories for each agent
    
    Returns:
        Tuple of (loss_functions, linearize_functions, compiled_functions)
    """
    loss_functions = []
    linearize_functions = []
    compiled_functions = []
    
    for i, agent in enumerate(agents):
        ref_traj = reference_trajectories[i]
        
        def create_runtime_loss(agent_idx, ref_traj):
            def runtime_loss(xt, ut, ref_xt, other_states):
                # Navigation cost: follow reference trajectory
                nav_loss = jnp.sum(jnp.square(xt[:2] - ref_xt[:2]))
                
                # Collision avoidance cost
                collision_loss = 0.0
                for other_state in other_states:
                    if other_state is not None:
                        distance_squared = jnp.sum(jnp.square(xt[:2] - other_state[:2]))
                        collision_loss += 10.0 * jnp.exp(-5.0 * distance_squared)
                
                # Control cost
                control_loss = jnp.sum(jnp.square(ut))
                
                return nav_loss + collision_loss + 0.1 * control_loss
            
            return runtime_loss
        
        def trajectory_loss(x_traj, u_traj, ref_x_traj, other_x_trajs):
            """Compute total trajectory loss"""
            total_loss = 0.0
            for t in range(len(x_traj)):
                # Get reference point for this timestep
                ref_xt = ref_x_traj[t] if t < len(ref_x_traj) else ref_x_traj[-1]
                
                # Get other agents' states at this timestep
                other_states = []
                for other_x_traj in other_x_trajs:
                    if other_x_traj is not None and t < len(other_x_traj):
                        other_states.append(other_x_traj[t])
                
                # Compute single step loss
                step_loss = create_runtime_loss(i, ref_traj)(x_traj[t], u_traj[t], ref_xt, other_states)
                total_loss += step_loss
            
            return total_loss
        
        def linearize_loss(x_traj, u_traj, ref_x_traj, other_x_trajs):
            """Compute gradients of trajectory loss w.r.t. states and controls"""
            # Define single step loss
            def single_step_loss(xt, ut, ref_xt, other_states):
                return create_runtime_loss(i, ref_traj)(xt, ut, ref_xt, other_states)
            
            # Compute gradients for each timestep
            dldx = grad(single_step_loss, argnums=0)
            dldu = grad(single_step_loss, argnums=1)
            
            def grad_step(args):
                xt, ut, ref_xt, other_xts = args
                return dldx(xt, ut, ref_xt, other_xts), dldu(xt, ut, ref_xt, other_xts)
            
            # Prepare arguments for vmap
            ref_x_expanded = []
            other_x_expanded = []
            
            for t in range(len(x_traj)):
                # Get reference point for this timestep
                ref_xt = ref_x_traj[t] if t < len(ref_x_traj) else ref_x_traj[-1]
                ref_x_expanded.append(ref_xt)
                
                # Get other agents' states at this timestep
                other_states = []
                for other_x_traj in other_x_trajs:
                    if other_x_traj is not None and t < len(other_x_traj):
                        other_states.append(other_x_traj[t])
                    else:
                        other_states.append(jnp.zeros(4))  # Placeholder state
                other_x_expanded.append(jnp.array(other_states))
            
            ref_x_expanded = jnp.array(ref_x_expanded)
            other_x_expanded = jnp.array(other_x_expanded)
            
            grads = vmap(grad_step)((x_traj, u_traj, ref_x_expanded, other_x_expanded))
            return grads[0], grads[1]  # a_traj, b_traj
        
        loss_functions.append(trajectory_loss)
        linearize_functions.append(linearize_loss)
        
        # Compile the functions for efficiency
        compiled_functions.append({
            'loss': jit(trajectory_loss),
            'linearize': jit(linearize_loss)
        })
    
    return loss_functions, linearize_functions, compiled_functions


def solve_receding_horizon_game(agents: list, 
                               current_states: list, 
                               target_positions: List[jnp.ndarray], 
                               compiled_functions: list) -> tuple:
    """
    Solve a single receding horizon game using the proper iLQR approach.
    
    Args:
        agents: List of agent objects
        current_states: Current states of all agents
        target_positions: Target positions for all agents
        compiled_functions: Compiled loss functions for each agent
    
    Returns:
        Tuple of (first_controls, full_trajectories, game_time)
    """
    start_time = time.time()
    
    # Create agent setup with current states
    initial_states = []
    for i in range(n_agents):
        if i < len(current_states):
            initial_states.append(jnp.array(current_states[i]))
        else:
            # Fallback to zero state if not available
            initial_states.append(jnp.array([0.0, 0.0, 0.0, 0.0]))
    
    # Create reference trajectories for current planning horizon
    current_reference_trajectories = []
    for i in range(n_agents):
        start_pos = jnp.array(initial_states[i][:2])  # Convert to array
        target_pos = jnp.array(target_positions[i])   # target_positions[i] is already [x, y]
        # Linear interpolation over planning horizon
        ref_traj = jnp.linspace(start_pos, target_pos, T_receding_horizon_planning)
        current_reference_trajectories.append(ref_traj)
    
    # Initialize control trajectories with zeros (like the training script)
    control_trajectories = [jnp.zeros((T_receding_horizon_planning, 2)) for _ in range(n_agents)]
    
    # Main optimization loop (following the pattern from generate_receding_horizon_trajectories.py)
    for iter in range(num_iters):
        # Step 1: Linearize dynamics for all agents using agent.linearize_dyn
        state_trajectories = []
        A_trajectories = []
        B_trajectories = []
        
        for i in range(n_agents):
            # Use the agent's linearize_dyn method (like in the reference files)
            x_traj, A_traj, B_traj = agents[i].linearize_dyn(initial_states[i], control_trajectories[i])
            state_trajectories.append(x_traj)
            A_trajectories.append(A_traj)
            B_trajectories.append(B_traj)
        
        # Step 2: Linearize loss functions for all agents
        a_trajectories = []
        b_trajectories = []
        
        for i in range(n_agents):
            # Create list of other agents' states for this agent
            other_states = [state_trajectories[j] for j in range(n_agents) if j != i]
            
            # Use the compiled linearize function from the loss functions
            a_traj, b_traj = compiled_functions[i]['linearize'](
                state_trajectories[i], control_trajectories[i], 
                current_reference_trajectories[i], other_states)
            a_trajectories.append(a_traj)
            b_trajectories.append(b_traj)
        
        # Step 3: Solve LQR subproblems for all agents using agent.solve
        control_updates = []
        
        for i in range(n_agents):
            # Use the agent's built-in solve method (like in the training script)
            v_traj, _ = agents[i].solve(
                A_trajectories[i], B_trajectories[i], 
                a_trajectories[i], b_trajectories[i])
            control_updates.append(v_traj)
        
        # Step 4: Update control trajectories with gradient descent
        for i in range(n_agents):
            control_trajectories[i] += step_size * control_updates[i]
    
    # Extract first controls (what will actually be applied)
    first_controls = [control_trajectories[i][0] for i in range(n_agents)]
    
    # Get final state trajectories for return
    final_state_trajectories = []
    for i in range(n_agents):
        x_traj, _, _ = agents[i].linearize_dyn(initial_states[i], control_trajectories[i])
        final_state_trajectories.append(x_traj)
    
    end_time = time.time()
    game_time = end_time - start_time
    
    return first_controls, final_state_trajectories, game_time


def extract_observation_trajectory(sample_data: Dict[str, Any]) -> jnp.ndarray:
    """Extract observation trajectory for goal inference."""
    obs_traj = []
    for step in range(T_observation):
        step_states = []
        for agent_idx in range(n_agents):
            agent_key = f"agent_{agent_idx}"
            agent_states = sample_data["trajectories"][agent_key]["states"]
            if step < len(agent_states):
                step_states.append(agent_states[step])
            else:
                # If trajectory is too short, use last state
                last_state = agent_states[-1] if agent_states else [0.0, 0.0, 0.0, 0.0]
                step_states.append(last_state)
        obs_traj.append(step_states)
    
    return jnp.array(obs_traj)


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
            agent_states = sample_data["trajectories"][agent_key]["states"]
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
                                     goal_trained_state: Any) -> Dict[str, Any]:
    """
    Test receding horizon planning with goal inference and player selection models.
    
    Args:
        sample_data: Reference trajectory sample data
        psn_model: Trained PSN model
        psn_trained_state: Trained PSN model state
        goal_model: Trained goal inference model
        goal_trained_state: Trained goal inference model state
    
    Returns:
        Dictionary containing test results
    """
    print(f"    Testing receding horizon planning with models on sample {sample_data['sample_id']}")
    
    # Initialize results storage
    results = {
        'sample_id': sample_data['sample_id'],
        'ego_agent_id': ego_agent_id,
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
            agent_states = sample_data["trajectories"][agent_key]["states"]
            
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
            current_states.append(agent_states[-1])
        else:
            # Fallback
            sample_states = sample_data["trajectories"][agent_key]["states"]
            current_states.append(sample_states[T_observation - 1] if len(sample_states) >= T_observation else sample_states[-1])
    
    # Main receding horizon loop
    for iteration in range(T_receding_horizon_iterations):        
        # Decide whether to use models or default values based on iteration
        if iteration < config.testing.receding_horizon.initial_stabilization_iterations:
            # First N iterations: Use ground truth goals and all agents (mask = 1)
            predicted_goals = extract_reference_goals(sample_data)
            predicted_mask = jnp.ones(n_agents - 1)  # All 1s for mask (no selection)
            num_selected = n_agents - 1  # All other agents selected
            mask_sparsity = 0.0  # No sparsity
            true_goals = predicted_goals  # Same as predicted for this phase
        else:
            # After N iterations: Use goal inference and player selection models (threshold: {config.testing.receding_horizon.mask_threshold})
            # Step 1: Infer goals using the observation trajectory
            goal_obs_traj = extract_observation_trajectory(sample_data)
            goal_obs_input = goal_obs_traj.flatten().reshape(1, -1)
            predicted_goals = goal_model.apply({'params': goal_trained_state['params']}, goal_obs_input, deterministic=True)
            predicted_goals = predicted_goals[0].reshape(n_agents, 2)
            
            # Get true goals for comparison
            true_goals = extract_reference_goals(sample_data)
            
            # Step 2: Infer player selection using current observation
            # Construct observation trajectory in the correct format (T_observation, n_agents, state_dim)
            obs_traj = []
            for step in range(T_observation):
                step_states = []
                for agent_idx in range(n_agents):
                    agent_key = f"agent_{agent_idx}"
                    if agent_idx == 0:  # Ego agent: use computed accumulated trajectory
                        agent_states = current_game_state["trajectories"][agent_key]["states"]
                    else:  # Other agents: use ground truth trajectory from reference
                        agent_states = sample_data["trajectories"][agent_key]["states"]
                    
                    if step < len(agent_states):
                        step_states.append(agent_states[step])
                    else:
                        # If trajectory is too short, use last state
                        last_state = agent_states[-1] if agent_states else [0.0, 0.0, 0.0, 0.0]
                        step_states.append(last_state)
                obs_traj.append(step_states)
            
            # Convert to array and reshape to (1, T_observation, n_agents, state_dim)
            obs_array = jnp.array(obs_traj)  # (T_observation, n_agents, state_dim)
            obs_input = obs_array.reshape(1, T_observation, n_agents, 4)  # (1, 10, 4, 4)
            
            predicted_mask = psn_model.apply({'params': psn_trained_state['params']}, obs_input)
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
        # Create agent setup for the current iteration
        agents, reference_trajectories = create_agent_setup(current_states, predicted_goals)
        
        # Create loss functions
        loss_functions, linearize_functions, compiled_functions = create_loss_functions(
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
                if ref_step < len(sample_data["trajectories"][f"agent_{i}"]["states"]):
                    ref_state = sample_data["trajectories"][f"agent_{i}"]["states"][ref_step]
                else:
                    # If reference trajectory is too short, use last state
                    ref_state = sample_data["trajectories"][f"agent_{i}"]["states"][-1]
                
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
                                save_dir: str) -> str:
    """
    Create a GIF visualization of the receding horizon trajectory evolution.
    
    Args:
        sample_data: Reference trajectory sample data
        results: Test results containing receding horizon data
        sample_id: Sample identifier
        save_dir: Directory to save the GIF
    
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
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        
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
        for i in range(n_agents):
            agent_key = f"agent_{i}"
            agent_states = game_state["trajectories"][agent_key]["states"]
            
            # Ensure we don't go beyond available trajectory data
            max_available_steps = len(agent_states)
            actual_step = min(step, max_available_steps - 1)
            
            if actual_step >= 0:
                if i == 0:  # Ego agent - plot both ground truth and computed trajectories
                    # Plot ground truth trajectory (black dashed)
                    sample_agent_states = sample_data["trajectories"][agent_key]["states"]
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
                else:  # Other agents - show reference receding horizon trajectories
                    # Plot reference trajectory from sample data (this is the receding horizon reference)
                    sample_agent_states = sample_data["trajectories"][agent_key]["states"]
                    if len(sample_agent_states) > 0:
                        sample_traj = np.array(sample_agent_states[:T_total])
                        ax.plot(sample_traj[:, 0], sample_traj[:, 1], '-', 
                                 color=other_agent_color, alpha=0.6, linewidth=1, 
                                 label=f'Agent {i} (Reference Receding Horizon)')
        
        # Plot current positions with selection coloring
        for i in range(n_agents):
            agent_key = f"agent_{i}"
            agent_states = game_state["trajectories"][agent_key]["states"]
            
            # Ensure we don't go beyond available trajectory data
            max_available_steps = len(agent_states)
            actual_step = min(step, max_available_steps - 1)
            
            if actual_step >= 0:
                current_pos = np.array(agent_states[actual_step][:2])
                
                # Determine if this agent is selected (for non-ego agents)
                is_selected = False
                if i > 0 and step >= T_observation:  # Only check selection after observation phase
                    # Find the iteration result for this step
                    iteration_idx = step - T_observation
                    if iteration_idx < len(results['receding_horizon_results']):
                        iteration_result = results['receding_horizon_results'][iteration_idx]
                        predicted_mask = iteration_result['predicted_mask']
                        if i-1 < len(predicted_mask) and predicted_mask[i-1] > 0.05:  # i-1 because mask excludes ego agent
                            is_selected = True
                
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
        true_goals = extract_reference_goals(sample_data)
        
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
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
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
    
    # Determine output directory
    if output_dir is None:
        output_dir = "receding_horizon_test_results"
    
    # Load models
    print(f"Loading trained models...")
    psn_model, psn_trained_state, goal_model, goal_trained_state = load_trained_models(
        psn_model_path, goal_model_path)
    
    print(f"✓ Models loaded successfully")
    
    # Load reference data
    print(f"Loading reference data from: {reference_file}")
    import glob
    
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
                sample_data, psn_model, psn_trained_state, goal_model, goal_trained_state)
            
            # Save results
            filepath = save_test_results(results, output_dir)
            print(f"  ✓ Results saved to: {filepath}")
            
            # Create trajectory visualization GIF
            gif_path = create_receding_horizon_gif(sample_data, results, sample_data['sample_id'], output_dir)
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
    
    # Model paths from config
    psn_model_path = config.testing.psn_model
    goal_model_path = config.testing.goal_inference_model
    reference_file = "reference_trajectories_4p"
    
    print(f"Using models:")
    print(f"  PSN Model: {psn_model_path}")
    print(f"  Goal Model: {goal_model_path}")
    print(f"  Reference Data: {reference_file}")
    
    # Run testing
    results = run_receding_horizon_testing(
        psn_model_path=psn_model_path,
        goal_model_path=goal_model_path,
        reference_file=reference_file,
        output_dir=config.testing.receding_horizon.output_dir,
        num_samples=config.testing.receding_horizon.num_samples
    )
    
    print(f"\nReceding horizon testing completed!")
    print(f"Generated {len(results)} test results with integrated models.")
