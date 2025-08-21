#!/usr/bin/env python3
"""
Test script for trained PSN model with pretrained goal inference.

This script loads a trained PSN model and tests it on reference trajectory data
using single masked game planning with a simplified approach:

KEY FEATURES:
- Player selection starts from step 11 onwards
- Observations come from previously solved games (not ground truth trajectories)
- Each receding horizon step uses the solved trajectory from the previous step
- Generates GIF visualizations showing the evolution of player selection
- Creates a realistic closed-loop testing scenario

GOAL INFERENCE APPROACH:
- Steps 1-10: Everyone solves the full game with ground truth goals
- Steps 11-50: Infer goals using FIRST 10 steps of trajectory (as in training), pick players, solve smaller game
- Ego agent acts first control step from computed trajectory, other agents follow ground truth
- This creates proper receding horizon behavior where each step builds on the previous solution
- Goal inference uses the same approach as goal_prediction_test (first 10 steps, not last 10)

VISUALIZATION IMPROVEMENTS:
- Consistent color scheme: ego agent (blue), other agents (gray), selected agents (red)
- Text labels to differentiate non-ego agents
- Ego agent trajectory is fully animated (no special marker needed)
- GIF shows all 50 steps with computed vs ground truth trajectories
- Selected agents change color to red when selected

MODIFIED: Removed simple and full mode tests, only receding horizon test remains.
Player selection starts from step 11 and uses solved game trajectories for observations.
No timeout mechanism - allows computation to complete naturally.
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

import time
from PIL import Image
import imageio

# Import the PSN network and related functions
from psn_training_with_pretrained_goals import (
    PlayerSelectionNetwork, 
    create_loss_functions,
    solve_masked_game,
    extract_ego_reference_trajectory,
    similarity_loss,
    PointAgent
)

# Import the goal inference network
from goal_inference.pretrain_goal_inference import GoalInferenceNetwork, extract_observation_trajectory

# Import Flax for model loading
import flax.serialization
from flax.training import train_state
import optax

# Add parent directory to path for imports and config loading
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from config_loader import load_config

# Load configuration
config = load_config()

# Extract configuration constants
N_agents = config.game.N_agents
T_observation = config.goal_inference.observation_length
T_total = config.game.T_total
state_dim = config.game.state_dim
num_iters = config.optimization.num_iters
dt = config.game.dt
ego_agent_id = config.game.ego_agent_id





def create_masked_game_setup_with_adaptive_threshold(sample_data: dict, ego_agent_id: int,
                                                   predicted_mask: jnp.ndarray, predicted_goals: jnp.ndarray, 
                                                   is_training: bool = True) -> tuple:
    """
    Create masked game setup with adaptive threshold for testing.
    
    This version uses a lower threshold (0.05) appropriate for the learned mask distribution
    instead of the hardcoded 0.5 threshold from training.
    """
    import jax.numpy as jnp
    
    if is_training:
        # Training mode: include all agents but use mask values for continuous weighting
        # (same as original function)
        # Reshape predicted goals to (N_agents, 2) format
        predicted_goals_reshaped = predicted_goals.reshape(N_agents, 2)
        
        agents = []
        initial_states = []
        target_positions = []
        
        # Cost function weights
        Q = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001]))
        R = jnp.diag(jnp.array([0.01, 0.01]))
        
        for i in range(N_agents):
            agent = PointAgent(dt=dt, x_dim=4, u_dim=2, Q=Q, R=R)
            agents.append(agent)
            
            agent_key = f"agent_{i}"
            agent_states = sample_data["trajectories"][agent_key]["states"]
            pos_2d = jnp.array(agent_states[0][:2])
            initial_state = jnp.array([pos_2d[0], pos_2d[1], 0.0, 0.0])
            initial_states.append(initial_state)
            
            # Use predicted goal position for this agent (both x and y coordinates)
            target_positions.extend([predicted_goals_reshaped[i, 0], predicted_goals_reshaped[i, 1]])
        
        return agents, initial_states, jnp.array(target_positions), predicted_mask
        
    else:
        # Testing mode: Use adaptive threshold based on learned mask distribution
        # Network was trained for continuous multiplication, outputs values ~0.005-0.08
        mask_threshold = 0.05  # Adaptive threshold for learned distribution
        
        selected_agents = [ego_agent_id]
        mask_values = []
        
        for i in range(N_agents - 1):
            # Map mask index to actual agent ID (skip ego agent)
            agent_id = i if i < ego_agent_id else i + 1
            if predicted_mask[i] > mask_threshold:  # Use adaptive threshold
                selected_agents.append(agent_id)
                mask_values.append(predicted_mask[i])
        
        # Reshape predicted goals to (N_agents, 2) format
        predicted_goals_reshaped = predicted_goals.reshape(N_agents, 2)
        
        # Create agents and get their initial states
        agents = []
        initial_states = []
        selected_targets = []
        
        # Cost function weights (same for all agents)
        Q = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001]))
        R = jnp.diag(jnp.array([0.01, 0.01]))
        
        for agent_id in selected_agents:
            agent = PointAgent(dt=dt, x_dim=4, u_dim=2, Q=Q, R=R)
            agents.append(agent)
            
            agent_key = f"agent_{agent_id}"
            agent_states = sample_data["trajectories"][agent_key]["states"]
            pos_2d = jnp.array(agent_states[0][:2])
            initial_state = jnp.array([pos_2d[0], pos_2d[1], 0.0, 0.0])
            initial_states.append(initial_state)
            
            # Use predicted goal position for this agent (both x and y coordinates)
            selected_targets.extend([predicted_goals_reshaped[agent_id, 0], predicted_goals_reshaped[agent_id, 1]])
        
        return agents, initial_states, jnp.array(selected_targets), jnp.array(mask_values)








def load_trained_models(psn_model_path: str, goal_model_path: str) -> Tuple[PlayerSelectionNetwork, Any, GoalInferenceNetwork, Any]:
    """
    Load trained PSN and goal inference models from files.
    
    Args:
        psn_model_path: Path to the trained PSN model file
        goal_model_path: Path to the trained goal inference model file
        
    Returns:
        Tuple of (psn_model, psn_trained_state, goal_model, goal_trained_state)
    """
    print(f"Loading trained PSN model from: {psn_model_path}")
    
    # Load the PSN model bytes
    with open(psn_model_path, 'rb') as f:
        psn_model_bytes = pickle.load(f)
    
    # Create the PSN model
    psn_model = PlayerSelectionNetwork()
    
    # Deserialize the PSN state
    psn_trained_state = flax.serialization.from_bytes(psn_model, psn_model_bytes)
    print("✓ PSN model loaded successfully")
    
    print(f"Loading trained goal inference model from: {goal_model_path}")
    
    # Load the goal inference model bytes
    with open(goal_model_path, 'rb') as f:
        goal_model_bytes = pickle.load(f)
    
    # Create the goal inference model
    goal_model = GoalInferenceNetwork()
    
    # Deserialize the goal inference state
    goal_trained_state = flax.serialization.from_bytes(goal_model, goal_model_bytes)
    print("✓ Goal inference model loaded successfully")
    
    return psn_model, psn_trained_state, goal_model, goal_trained_state


def extract_reference_goals(sample_data: Dict[str, Any]) -> jnp.ndarray:
    """Extract reference goals from sample data."""
    # Use the target_positions field as in the training data
    return jnp.array(sample_data["target_positions"])  # (N_agents, goal_dim)





# Full mode test function removed - only receding horizon test remains.


# Simple mode test function removed - only receding horizon test is used


def create_simple_trajectory_gif(sample_data: Dict[str, Any], results: Dict[str, Any], 
                               sample_id: int, save_dir: str = None) -> None:
    """
    Create GIF visualization for single masked game results.
    
    Args:
        sample_data: Reference trajectory sample
        results: Test results dictionary with game state data
        sample_id: Sample identifier
        save_dir: Directory to save GIF
    """
    # Force CPU-only mode for matplotlib to avoid GPU issues
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    if 'final_game_state' not in results:
        print(f"  Warning: No game state data for sample {sample_id}")
        return
    
    game_state = results['final_game_state']
    if not game_state.get('trajectories'):
        print(f"  Warning: Empty game state for sample {sample_id}")
        return
    
    # Create frames for the GIF showing all 50 steps
    frames = []
    
    # Use same color for all non-ego agents, different color for ego agent
    ego_color = 'darkblue'
    other_agent_color = 'gray'
    
    # Create exactly 50 frames (steps 1-50)
    total_steps = 50
    
    # Create frames for all steps
    for step in range(total_steps):
        # Create frame for this step
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_aspect('equal')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        
        # Title with step information
        step_title = f'Sample {sample_id+1}: Step {step+1}/50 (Single Game)'
        if step < 10:
            step_title += '\nPhase 1: Ground Truth Trajectories'
        else:
            step_title += '\nPhase 2: Ego Agent Computed, Others Ground Truth'
        ax.set_title(step_title)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        

        
        # Create frame for steps with receding horizon results
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_aspect('equal')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(f'Sample {sample_id+1}: Step {step+1}/50 (Single Game)\n'
                    f'Selected Agents: {results["num_selected_agents"]}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        
        # Define colors for this frame
        ego_color = 'darkblue'
        other_agent_color = 'gray'
        selected_color = 'red'
        
        # Extract data for this step
        predicted_mask = np.array(results['predicted_mask'])
        predicted_goals = results.get('predicted_goals')
        true_goals = np.array(results['true_goals'])
        
        # Get agent positions at this step
        init_positions = []
        for i in range(N_agents):
            agent_key = f"agent_{i}"
            agent_states = sample_data["trajectories"][agent_key]["states"]
            # Ensure we don't go beyond available trajectory data
            max_available_steps = len(agent_states)
            actual_step = min(step, max_available_steps - 1)
            init_positions.append(np.array(agent_states[actual_step][:2]))
        init_positions = np.array(init_positions)
        
        # Plot trajectories up to current step for all agents
        for i in range(N_agents):
            agent_key = f"agent_{i}"
            agent_states = sample_data["trajectories"][agent_key]["states"]
            # Ensure we don't go beyond available trajectory data
            max_available_steps = len(agent_states)
            actual_step = min(step, max_available_steps - 1)
            agent_traj = np.array(agent_states[:actual_step+1])
            
            if len(agent_traj) > 0:
                if i == 0:  # Ego agent - plot both ground truth and computed trajectories
                    # Plot ground truth trajectory (black dashed)
                    ax.plot(agent_traj[:, 0], agent_traj[:, 1], '--', 
                             color='black', alpha=0.8, linewidth=2, 
                             label=f'Ego Agent {i} (Ground Truth)', zorder=8)
                    
                    # Get computed trajectory from game state
                    game_agent_states = game_state["trajectories"][agent_key]["states"]
                    if len(game_agent_states) > 0:
                        game_traj = np.array(game_agent_states[:actual_step+1])
                        
                        # Plot computed trajectory (blue solid)
                        ax.plot(game_traj[:, 0], game_traj[:, 1], '-', 
                                 color='blue', alpha=0.9, linewidth=3, 
                                 label=f'Ego Agent {i} (Computed)', zorder=10)
                        
                        # Force plot limits to include both trajectories
                        all_x = np.concatenate([agent_traj[:, 0], game_traj[:, 0]])
                        all_y = np.concatenate([agent_traj[:, 1], game_traj[:, 1]])
                        ax.set_xlim(min(ax.get_xlim()[0], all_x.min() - 0.1),
                                   max(ax.get_xlim()[1], all_x.max() + 0.1))
                        ax.set_ylim(min(ax.get_ylim()[0], all_y.min() - 0.1),
                                   max(ax.get_ylim()[1], all_y.max() + 0.1))
                else:  # Other agents - always show only ground truth trajectories
                    # Check if this agent is selected for color coding
                    mask_idx = i - 1  # Mask indices are 0 to N-2 for agents 1 to N-1
                    is_selected = (mask_idx < len(predicted_mask) and predicted_mask[mask_idx] > 0.05)
                    
                    if is_selected:
                        # Selected agents: show ground truth trajectory with selected color
                        ax.plot(agent_traj[:, 0], agent_traj[:, 1], '-', 
                                 color=selected_color, alpha=0.8, linewidth=2, label=f'Agent {i} (Selected)')
                    else:
                        # Non-selected agents: show ground truth trajectory with normal color
                        ax.plot(agent_traj[:, 0], agent_traj[:, 1], '-', 
                                 color=other_agent_color, alpha=0.6, linewidth=1, label=f'Agent {i} (Not Selected)')
        
        # Plot current positions
        for i in range(N_agents):
            if i == 0:  # Ego agent
                ax.plot(init_positions[i][0], init_positions[i][1], 'o', 
                         color='darkblue', markersize=10, alpha=0.8)
            else:  # Other agents
                # Check if this agent is selected
                mask_idx = i - 1  # Mask indices are 0 to N-2 for agents 1 to N-1
                if mask_idx < len(predicted_mask) and predicted_mask[mask_idx] > 0.05:
                    # Agent is selected - use selected color
                    ax.plot(init_positions[i][0], init_positions[i][1], 'o', 
                             color=selected_color, markersize=12, alpha=0.9, 
                             markeredgecolor='black', markeredgewidth=2)
                else:
                    # Agent is not selected - use normal color
                    ax.plot(init_positions[i][0], init_positions[i][1], 'o', 
                             color=other_agent_color, markersize=8, alpha=0.7)
                
                # Add text label to differentiate agents
                ax.text(init_positions[i][0] + 0.1, init_positions[i][1] + 0.1, f'{i}', 
                        fontsize=12, ha='left', va='bottom', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Plot true goals
        for j in range(N_agents):
            if j == 0:  # Ego agent goal
                ax.plot(true_goals[j][0], true_goals[j][1], 's', 
                         color='darkblue', markersize=12, alpha=0.8, label=f'Ego Agent {j} True Goal')
            else:  # Other agent goals
                ax.plot(true_goals[j][0], true_goals[j][1], 's', 
                         color=other_agent_color, markersize=10, alpha=0.6, label=f'Agent {j} True Goal')
        
        # Plot goals
        true_goals = np.array(results['true_goals'])
        for j in range(N_agents):
            if j == 0:  # Ego agent goal
                ax.plot(true_goals[j][0], true_goals[j][1], 's', 
                         color=ego_color, markersize=12, alpha=0.8, label=f'Ego Agent {j} Goal')
            else:  # Other agent goals
                ax.plot(true_goals[j][0], true_goals[j][1], 's', 
                         color=other_agent_color, markersize=10, alpha=0.6, label=f'Agent {j} Goal')
        
        # Plot predicted goals only if they exist and are valid
        if predicted_goals is not None and len(predicted_goals) > 0:
            try:
                predicted_goals_array = np.array(predicted_goals)
                if predicted_goals_array.ndim == 2 and predicted_goals_array.shape[0] >= N_agents:
                    for j in range(N_agents):
                        if j == 0:  # Ego agent predicted goal
                            ax.plot(predicted_goals_array[j][0], predicted_goals_array[j][1], '^', 
                                     color=ego_color, markersize=10, alpha=0.6, label=f'Ego Agent {j} Predicted Goal')
                        else:  # Other agent predicted goals
                            ax.plot(predicted_goals_array[j][0], predicted_goals_array[j][1], '^', 
                                     color=other_agent_color, markersize=8, alpha=0.5, label=f'Agent {j} Predicted Goal')
            except (IndexError, ValueError) as e:
                print(f"      Warning: Could not plot predicted goals for step {step}: {e}")
        
        # ADD TEXT BOXES FOR NON-EGO AGENTS' TRUE AND PREDICTED GOALS
        if predicted_goals is not None and len(predicted_goals) > 0:
            try:
                predicted_goals_array = np.array(predicted_goals)
                if predicted_goals_array.ndim == 2 and predicted_goals_array.shape[0] >= N_agents:
                    # Add simple labels for non-ego agents near their goals
                    for j in range(1, N_agents):  # Skip ego agent (j=0)
                        true_goal = true_goals[j]
                        pred_goal = predicted_goals_array[j]
                        
                        # Label near TRUE goal
                        ax.text(true_goal[0] + 0.1, true_goal[1] + 0.1, f'{j}', 
                                fontsize=14, ha='left', va='bottom', 
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                                transform=ax.transData)
                        
                        # Label near PREDICTED goal
                        ax.text(pred_goal[0] + 0.1, pred_goal[1] + 0.1, f'{j}', 
                                fontsize=14, ha='left', va='bottom', 
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8),
                                transform=ax.transData)
                        
            except (IndexError, ValueError) as e:
                print(f"      Warning: Could not add goal labels for step {step}: {e}")
        
        # Add legend only
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Convert plot to image
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(canvas.get_width_height()[::-1] + (3,))
        
        frames.append(image)
        plt.close()
        # print(f"      Created frame for step {step+1}/50 (with player selection)")
    
    # Verify we have exactly 50 frames (steps 1-50)
    expected_frames = 50
    if len(frames) != expected_frames:
        print(f"      Warning: Expected {expected_frames} frames, but created {len(frames)} frames")
    
    # Save as GIF
    if frames:
        gif_path = os.path.join(save_dir, f"single_game_sample_{sample_id:03d}.gif")
        imageio.mimsave(gif_path, frames, duration=0.5, loop=0)  # 0.5 second per frame, loop infinitely
        print(f"  GIF saved: {gif_path}")
        print(f"  Created {len(frames)} frames for all 50 steps (single masked game)")
    






def update_game_state_with_solved_trajectories(current_game_state: Dict[str, Any], 
                                             state_trajectories: List[jnp.ndarray], 
                                             current_step: int, 
                                             horizon_steps: int, 
                                             ego_agent_id: int = 0) -> Dict[str, Any]:
    """
    Update the game state with solved trajectories for the next receding horizon step.
    
    Args:
        current_game_state: Current game state (modified from original sample data)
        state_trajectories: List of solved state trajectories for all agents
        current_step: Current time step
        horizon_steps: Number of steps to look ahead
        ego_agent_id: ID of the ego agent
        
    Returns:
        Updated game state with solved trajectories
    """
    # Create a deep copy to avoid modifying the original
    updated_game_state = current_game_state.copy()
    
    # Update trajectories for each agent based on solved game
    for agent_idx, trajectory in enumerate(state_trajectories):
        if trajectory is not None and len(trajectory) > 0:
            agent_key = f"agent_{agent_idx}"
            
            # Ensure the agent exists in the game state
            if agent_key not in updated_game_state["trajectories"]:
                updated_game_state["trajectories"][agent_key] = {"states": []}
            
            # Get the current trajectory length
            current_traj_length = len(updated_game_state["trajectories"][agent_key]["states"])
            
            # Update the trajectory from current_step onwards with solved trajectory
            # Keep the history up to current_step, then replace with solved trajectory
            if current_traj_length > current_step:
                # Truncate at current_step and append solved trajectory
                updated_game_state["trajectories"][agent_key]["states"] = (
                    updated_game_state["trajectories"][agent_key]["states"][:current_step] + 
                    trajectory.tolist()
                )
            else:
                # Extend with solved trajectory
                updated_game_state["trajectories"][agent_key]["states"].extend(trajectory.tolist())
    
    return updated_game_state


def test_model_on_sample_single_game(model: PlayerSelectionNetwork, trained_state: Any, 
                                   goal_model: GoalInferenceNetwork, goal_trained_state: Any,
                                   sample_data: Dict[str, Any], ego_agent_id: int = 0) -> Dict[str, Any]:
    """
    Test the trained model on a single sample using a simple single masked game approach.
    
    This function implements a simplified testing approach:
    1. Steps 1-10: Use ground truth trajectories for all agents (observation period)
    2. Steps 11-50: Ego agent follows computed trajectory from single masked game, others follow ground truth
    
    Args:
        model: PSN model (outputs N-1 mask values for player selection)
        trained_state: Trained PSN model state
        goal_model: Goal inference model
        goal_trained_state: Trained goal inference model state
        sample_data: Reference trajectory sample
        ego_agent_id: ID of the ego agent
        
    Returns:
        Dictionary containing test results
    """
    print(f"    Testing PSN model on sample {sample_data['sample_id']} using single masked game approach")
    
    # Constants
    T_observation = 10
    T_total = 50
    horizon_steps = T_total - T_observation  # 40 steps
    
    # Initialize game state with ground truth trajectories for observation period
    current_game_state = {
        "trajectories": {
            f"agent_{i}": {
                "states": [],
                "controls": []
            }
            for i in range(N_agents)
        }
    }
    
    # Phase 1: Observation period (steps 1-10) - use ground truth trajectories
    print(f"    Phase 1: Observation period (steps 1-10)")
    for step in range(T_observation):
        print(f"      Step {step + 1}/{T_observation}")
        
        # Add ground truth states for all agents
        for agent_idx in range(N_agents):
            agent_key = f"agent_{agent_idx}"
            agent_states = sample_data["trajectories"][agent_key]["states"]
            
            if step < len(agent_states):
                current_state = agent_states[step]
                current_game_state["trajectories"][agent_key]["states"].append(current_state)
            else:
                # If trajectory is too short, use last state
                last_state = agent_states[-1]
                current_game_state["trajectories"][agent_key]["states"].append(last_state)
    
    # Phase 2: Single masked game solving (steps 11-50)
    print(f"    Phase 2: Single masked game solving (steps 11-50)")
    
    # Get the final observation state (step 10) for goal inference
    final_observation_states = []
    for agent_idx in range(N_agents):
        agent_key = f"agent_{agent_idx}"
        agent_states = current_game_state["trajectories"][agent_key]["states"]
        if len(agent_states) > 0:
            final_observation_states.append(agent_states[-1])
        else:
            # Fallback to sample data
            sample_states = sample_data["trajectories"][agent_key]["states"]
            final_observation_states.append(sample_states[9] if len(sample_states) > 9 else sample_states[-1])
    
    # Infer goals using the final observation state
    print(f"      Inferring goals from final observation state...")
    goal_obs_traj = extract_observation_trajectory(sample_data)
    goal_obs_input = goal_obs_traj.flatten().reshape(1, -1)
    predicted_goals = goal_model.apply({'params': goal_trained_state['params']}, goal_obs_input, deterministic=True)
    predicted_goals = predicted_goals[0].reshape(N_agents, 2)
    
    # Get true goals for comparison
    true_goals = extract_reference_goals(sample_data)
    
    # Infer player selection using the final observation state
    print(f"      Inferring player selection from final observation state...")
    
    # Construct observation trajectory in the correct format (T_observation, N_agents, state_dim)
    obs_traj = []
    for step in range(T_observation):
        step_states = []
        for agent_idx in range(N_agents):
            agent_key = f"agent_{agent_idx}"
            agent_states = current_game_state["trajectories"][agent_key]["states"]
            if step < len(agent_states):
                step_states.append(agent_states[step])
            else:
                # If trajectory is too short, use last state
                last_state = agent_states[-1] if agent_states else [0.0, 0.0, 0.0, 0.0]
                step_states.append(last_state)
        obs_traj.append(step_states)
    
    # Convert to array and reshape to (1, T_observation, N_agents, state_dim)
    obs_array = jnp.array(obs_traj)  # (T_observation, N_agents, state_dim)
    obs_input = obs_array.reshape(1, T_observation, N_agents, state_dim)  # (1, 10, 4, 4)
    
    predicted_mask = model.apply({'params': trained_state['params']}, obs_input)
    predicted_mask = predicted_mask[0]  # Remove batch dimension
    
    # Apply threshold to get selected agents
    mask_threshold = 0.05  # Lower threshold based on observed mask value range
    selected_agents = jnp.where(predicted_mask > mask_threshold)[0]
    num_selected = len(selected_agents)
    mask_sparsity = 1.0 - (num_selected / (N_agents - 1))
    
    # Store selected agents for this step
    all_selected_agents = set()
    all_selected_agents.update([int(i) for i in selected_agents])
    
    # Create masked game setup with predicted goals and selected agents
    # Use the states at step 10 (end of observation period) as initial states for the masked game
    initial_states_at_step_10 = []
    for agent_idx in range(N_agents):
        agent_key = f"agent_{agent_idx}"
        agent_states = current_game_state["trajectories"][agent_key]["states"]
        if len(agent_states) >= T_observation:
            final_state = agent_states[T_observation - 1]  # Step 10 (0-indexed: step 9)
            initial_states_at_step_10.append(final_state)
        else:
            # Fallback
            sample_states = sample_data["trajectories"][agent_key]["states"]
            initial_states_at_step_10.append(sample_states[T_observation - 1] if len(sample_states) >= T_observation else sample_states[-1])
    
    agents, _, target_positions, mask_values = create_masked_game_setup_with_adaptive_threshold(
        sample_data, ego_agent_id, predicted_mask, predicted_goals.flatten(), is_training=False)
    
    # Override the initial states with the correct states from step 10
    initial_states = []
    for i, state in enumerate(initial_states_at_step_10):
        if i < len(agents):  # Only include agents that are in the masked game
            initial_states.append(jnp.array(state))
    
    # Safety check: limit the number of agents to prevent memory issues
    if len(agents) > 5:
        print(f"      Warning: Too many agents ({len(agents)}), limiting to 5 for safety")
        agents = agents[:5]
        initial_states = initial_states[:5]
        target_positions = target_positions[:5]
        mask_values = mask_values[:5] if len(mask_values) > 5 else mask_values
    
    # Extract reference trajectories for navigation (like ilqgames_example.py)
    reference_trajectories = []
    for agent_id in selected_agents:
        agent_key = f"agent_{agent_id}"
        agent_states = sample_data["trajectories"][agent_key]["states"]
        # Get the future trajectory from observation step onwards
        future_traj = agent_states[T_observation:T_total]
        if len(future_traj) < horizon_steps:
            # Pad with last state if trajectory is too short
            last_state = future_traj[-1] if future_traj else agent_states[-1]
            padding = [last_state] * (horizon_steps - len(future_traj))
            future_traj.extend(padding)
        reference_trajectories.append(jnp.array(future_traj))
    
    # Create loss functions
    loss_functions, linearize_functions, compiled_functions = create_loss_functions(
        agents, mask_values, is_training=False, reference_trajectories=reference_trajectories)
    
    # Solve the masked game with horizon T_total - T_observation (40 steps)
    print(f"      Solving masked game with horizon {horizon_steps} steps...")
    state_trajectories, control_trajectories = solve_masked_game(
        agents, initial_states, target_positions, compiled_functions, mask_values, num_iters=50,
        reference_trajectories=reference_trajectories)
    
    # Extract trajectories for all agents
    all_agent_trajectories = []
    for agent_idx in range(len(state_trajectories)):
        agent_traj = state_trajectories[agent_idx]
        
        # Ensure the computed trajectory spans the expected length
        if agent_traj.shape[0] < horizon_steps:
            if agent_traj.shape[0] > 0:
                last_state = agent_traj[-1:]
                pad_size = horizon_steps - agent_traj.shape[0]
                padding = jnp.tile(last_state, (pad_size, 1))
                agent_traj = jnp.concatenate([agent_traj, padding], axis=0)
            else:
                # Create fallback trajectory
                start_pos = initial_states[agent_idx][:2]
                end_pos = target_positions[agent_idx]
                fallback_traj = jnp.linspace(start_pos, end_pos, horizon_steps)
                agent_traj = jnp.zeros((horizon_steps, 4))
                agent_traj = agent_traj.at[:, :2].set(fallback_traj)
        
        all_agent_trajectories.append(agent_traj)
    
    # Complete the game state: Ego agent follows computed trajectory, others follow ground truth
    print(f"      Completing game state with computed and ground truth trajectories...")
    
    # Add ego agent's computed trajectory (steps 11-50)
    ego_key = f"agent_{ego_agent_id}"
    if ego_agent_id < len(all_agent_trajectories) and all_agent_trajectories[ego_agent_id] is not None:
        ego_traj = all_agent_trajectories[ego_agent_id]
        for step_idx in range(horizon_steps):
            if step_idx < len(ego_traj):
                next_state = ego_traj[step_idx]
                current_game_state["trajectories"][ego_key]["states"].append(next_state.tolist())
            else:
                # Fallback to last state if trajectory is too short
                last_state = ego_traj[-1] if len(ego_traj) > 0 else initial_states[ego_agent_id]
                current_game_state["trajectories"][ego_key]["states"].append(last_state.tolist())
    
    # Add other agents' ground truth trajectories (steps 11-50)
    for agent_idx in range(N_agents):
        if agent_idx != ego_agent_id:
            agent_key = f"agent_{agent_idx}"
            agent_states = sample_data["trajectories"][agent_key]["states"]
            
            for step_idx in range(horizon_steps):
                step_in_sample = T_observation + step_idx
                if step_in_sample < len(agent_states):
                    next_state = agent_states[step_in_sample]
                    current_game_state["trajectories"][agent_key]["states"].append(next_state)
                else:
                    # If no more ground truth, use last state
                    last_state = agent_states[-1]
                    current_game_state["trajectories"][agent_key]["states"].append(last_state)
    
    # Verify we have exactly 50 steps in total
    total_steps = len(current_game_state["trajectories"][f"agent_{ego_agent_id}"]["states"])
    if total_steps != T_total:
        print(f"    Warning: Expected {T_total} steps total, but got {total_steps} steps")
        print(f"    Steps 1-10: {T_observation}")
        print(f"    Steps 11-50: {total_steps - T_observation}")
    else:
        print(f"    Successfully created results for all {T_total} steps")
    
    # Compute trajectory similarity for ego agent
    trajectory_rmse = float('inf')
    ego_key = f"agent_{ego_agent_id}"
    
    if len(current_game_state["trajectories"][ego_key]["states"]) >= T_total:
        # Get computed trajectory (steps 11-50)
        computed_traj = current_game_state["trajectories"][ego_key]["states"][T_observation:]
        computed_traj = jnp.array(computed_traj)
        
        # Get reference trajectory (steps 11-50)
        ref_traj = sample_data["trajectories"][ego_key]["states"][T_observation:T_total]
        ref_traj = jnp.array(ref_traj)
        
        # Ensure trajectories have the same length
        min_length = min(len(computed_traj), len(ref_traj))
        if min_length > 0:
            computed_subset = computed_traj[:min_length, :2]  # Only positions
            ref_subset = ref_traj[:min_length, :2]  # Only positions
            
            # Compute RMSE
            position_diff = computed_subset - ref_subset
            distances = jnp.linalg.norm(position_diff, axis=-1)
            trajectory_rmse = float(jnp.mean(distances))
            print(f"    Trajectory RMSE (steps 11-50): {trajectory_rmse:.4f}")
        else:
            print(f"    Warning: No valid trajectory comparison possible")
    else:
        print(f"    Warning: Incomplete trajectory generated")
    
    # Compute goal prediction error
    goal_error = jnp.mean(jnp.square(predicted_goals - true_goals))
    goal_rmse = jnp.sqrt(goal_error)
    
    print(f"      Goals RMSE: {goal_rmse:.4f}, Mask Sparsity: {mask_sparsity:.2f}, Selected: {num_selected}")
    
    # Prepare final results
    final_results = {
        'sample_id': sample_data['sample_id'],
        'ego_agent_id': ego_agent_id,
        'T_observation': T_observation,
        'T_total': T_total,
        'goal_rmse': float(goal_rmse),
        'trajectory_rmse': float(trajectory_rmse),
        'mask_sparsity': float(mask_sparsity),
        'num_selected_agents': int(num_selected),
        'selected_agents': list(all_selected_agents),
        'final_game_state': current_game_state,
        'true_goals': true_goals.tolist(),
        'predicted_goals': predicted_goals.tolist(),
        'predicted_mask': predicted_mask.tolist()  # Add the predicted mask for GIF generation
    }
    
    return final_results


def run_comprehensive_testing(psn_model_path: str, goal_model_path: str, 
                            reference_file: str, output_dir: str = None,
                            num_samples: int = 10, ego_agent_id: int = 0,
                            test_mode: str = "single_masked_game") -> Dict[str, Any]:
    """
    Run comprehensive testing on the trained model using single masked game planning.
    
    The single masked game testing uses a simplified approach where:
    - First 10 steps use ground truth trajectories for all agents
    - Player selection and goal inference happen at step 10
    - A single masked game is solved for the remaining 40 steps
    - Ego agent follows computed trajectory, others follow ground truth
    
    Args:
        psn_model_path: Path to trained PSN model
        goal_model_path: Path to trained goal inference model
        reference_file: Path to reference trajectory file
        output_dir: Directory to save test results (if None, uses PSN model's log directory)
        num_samples: Number of samples to test
        ego_agent_id: ID of the ego agent
        test_mode: Always "single_masked_game" for single masked game planning
        
    Returns:
        Dictionary containing comprehensive test results
    """
    print("=" * 80)
    print("COMPREHENSIVE MODEL TESTING")
    print("=" * 80)
    
    # Determine output directory
    if output_dir is None:
        # Extract PSN model directory and create test results subdirectory
        psn_model_dir = os.path.dirname(psn_model_path)
        output_dir = os.path.join(psn_model_dir, "test_results")
        print(f"Test results will be saved under PSN model directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    psn_model, psn_trained_state, goal_model, goal_trained_state = load_trained_models(
        psn_model_path, goal_model_path)
    
    # Load reference data from directory
    print(f"Loading reference data from directory: {reference_file}")
    
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
    
    print(f"Loaded {len(reference_data)} reference samples from {reference_file}")
    print(f"Sample files: {len(json_files)} found, {len(reference_data)} loaded successfully")
    
    # Limit number of samples for testing
    test_samples = reference_data[:min(num_samples, len(reference_data))]
    print(f"Testing on {len(test_samples)} samples")
    print(f"Test mode: {test_mode}")
    
    # Initialize results storage
    all_results = []
    summary_stats = {
        'goal_rmse_values': [],
        'mask_sparsity_values': [],
        'num_selected_agents_values': [],
        'trajectory_rmse_values': [],
        'game_solved_count': 0,
        'total_samples': len(test_samples)
    }
    
    # Test each sample
    for i, sample_data in enumerate(test_samples):
        print(f"\nTesting sample {i+1}/{len(test_samples)}...")
        
        # Use single masked game testing
        results = test_model_on_sample_single_game(
            psn_model, psn_trained_state, goal_model, goal_trained_state,
            sample_data, ego_agent_id)
        
        # Store results
        all_results.append(results)
        
        # Update summary statistics
        # (Removed: if results['goal_prediction_rmse'] != float('inf'):)
        summary_stats['goal_rmse_values'].append(results['goal_rmse'])
        summary_stats['mask_sparsity_values'].append(results['mask_sparsity'])
        summary_stats['num_selected_agents_values'].append(results['num_selected_agents'])
        summary_stats['trajectory_rmse_values'].append(results['trajectory_rmse'])
        if results.get('game_solved', False):
            summary_stats['game_solved_count'] += 1
        
        # Create GIF visualization for receding horizon results
        create_simple_trajectory_gif(sample_data, results, i, output_dir)
        

        
        # Print sample results
        print(f"  Goal RMSE: {results['goal_rmse']:.4f}")
        print(f"  Mask Sparsity: {results['mask_sparsity']:.2f}")
        print(f"  Selected Agents: {results['num_selected_agents']}")
        print(f"  Trajectory RMSE: {results['trajectory_rmse']:.4f}")
    
    # Compute summary statistics
    print("\n" + "=" * 80)
    print("TESTING SUMMARY")
    print("=" * 80)
    
    if summary_stats['goal_rmse_values']:
        avg_goal_rmse = np.mean(summary_stats['goal_rmse_values'])
        std_goal_rmse = np.std(summary_stats['goal_rmse_values'])
        print(f"Goal Prediction RMSE: {avg_goal_rmse:.4f} ± {std_goal_rmse:.4f}")
    
    if summary_stats['mask_sparsity_values']:
        avg_mask_sparsity = np.mean(summary_stats['mask_sparsity_values'])
        std_mask_sparsity = np.std(summary_stats['mask_sparsity_values'])
        print(f"Mask Sparsity: {avg_mask_sparsity:.3f} ± {std_mask_sparsity:.3f}")
    
    if summary_stats['num_selected_agents_values']:
        avg_selected = np.mean(summary_stats['num_selected_agents_values'])
        std_selected = np.std(summary_stats['num_selected_agents_values'])
        print(f"Average Selected Agents: {avg_selected:.2f} ± {std_selected:.2f}")
    
    if summary_stats['trajectory_rmse_values']:
        avg_traj_rmse = np.mean(summary_stats['trajectory_rmse_values'])
        std_traj_rmse = np.std(summary_stats['trajectory_rmse_values'])
        print(f"Trajectory RMSE: {avg_traj_rmse:.4f} ± {std_traj_rmse:.4f}")
    
    game_solved_rate = summary_stats['game_solved_count'] / summary_stats['total_samples']
    print(f"Game Solving Success Rate: {game_solved_rate:.2%} ({summary_stats['game_solved_count']}/{summary_stats['total_samples']})")
    
    # Save comprehensive results
    results_file = os.path.join(output_dir, "comprehensive_test_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'summary_stats': summary_stats,
            'detailed_results': all_results,
            'test_config': {
                'psn_model_path': psn_model_path,
                'goal_model_path': goal_model_path,
                'reference_file': reference_file,
                'num_samples': num_samples,
                'ego_agent_id': ego_agent_id,
                'test_mode': test_mode,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Create summary report
    summary_file = os.path.join(output_dir, "test_summary_report.txt")
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PSN MODEL TESTING SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test Configuration:\n")
        f.write(f"  PSN Model: {psn_model_path}\n")
        f.write(f"  Goal Model: {goal_model_path}\n")
        f.write(f"  Reference File: {reference_file}\n")
        f.write(f"  Test Mode: single_masked_game (50-step approach)\n")
        f.write(f"  Phase 1 (Steps 1-10): Ground truth trajectories for all agents\n")
        f.write(f"  Phase 2 (Steps 11-50): Single masked game solving with horizon 40 steps\n")
        f.write(f"  GIF Output: Steps 1-50 (50 frames) showing complete trajectory\n")
        f.write(f"  Number of Samples: {num_samples}\n")
        f.write(f"  Ego Agent ID: {ego_agent_id}\n")
        f.write(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Performance Metrics:\n")
        if summary_stats['goal_rmse_values']:
            avg_goal_rmse = np.mean(summary_stats['goal_rmse_values'])
            std_goal_rmse = np.std(summary_stats['goal_rmse_values'])
            f.write(f"  Goal Prediction RMSE: {avg_goal_rmse:.4f} ± {std_goal_rmse:.4f}\n")
        
        if summary_stats['mask_sparsity_values']:
            avg_mask_sparsity = np.mean(summary_stats['mask_sparsity_values'])
            std_mask_sparsity = np.std(summary_stats['mask_sparsity_values'])
            f.write(f"  Mask Sparsity: {avg_mask_sparsity:.3f} ± {std_mask_sparsity:.3f}\n")
        
        if summary_stats['num_selected_agents_values']:
            avg_selected = np.mean(summary_stats['num_selected_agents_values'])
            std_selected = np.std(summary_stats['num_selected_agents_values'])
            f.write(f"  Average Selected Agents: {avg_selected:.2f} ± {std_selected:.2f}\n")
        
        if summary_stats['trajectory_rmse_values']:
            avg_traj_rmse = np.mean(summary_stats['trajectory_rmse_values'])
            std_traj_rmse = np.std(summary_stats['trajectory_rmse_values'])
            f.write(f"  Trajectory RMSE: {avg_traj_rmse:.4f} ± {std_traj_rmse:.4f}\n")
        
        game_solved_rate = summary_stats['game_solved_count'] / summary_stats['total_samples']
        f.write(f"  Game Solving Success Rate: {game_solved_rate:.2%} ({summary_stats['game_solved_count']}/{summary_stats['total_samples']})\n\n")
        
        f.write("Sample Results:\n")
        for i, result in enumerate(all_results):
            f.write(f"  Sample {i+1}:\n")
            f.write(f"    Goal RMSE: {result['goal_rmse']:.4f}\n")
            f.write(f"    Mask Sparsity: {result['mask_sparsity']:.3f}\n")
            f.write(f"    Selected Agents: {result['num_selected_agents']}\n")
            f.write(f"    Trajectory RMSE: {result['trajectory_rmse']:.4f}\n")
            if 'error' in result:
                f.write(f"    Error: {result['error']}\n")
            f.write("\n")
    
    print(f"Summary report saved to: {summary_file}")
    
    return {
        'summary_stats': summary_stats,
        'detailed_results': all_results,
        'output_dir': output_dir
    }


def main():
    """Main function for running the test script."""
    # Use config values for model paths and parameters
    psn_model = config.testing.psn_model
    goal_model = config.testing.goal_inference_model
    reference_file = config.testing.test_data_dir
    num_samples = config.testing.num_test_samples
    ego_agent_id = config.game.ego_agent_id  # Use config value
    test_mode = "single_masked_game"  # Default test mode
    
    print("=" * 80)
    print("PSN MODEL TESTING WITH PRETRAINED GOAL INFERENCE")
    print("=" * 80)
    print("This script tests the trained PSN model using single masked game planning:")
    print("- Steps 1-10: Solve full game with ground truth goals")
    print("- Steps 11-50: Infer goals using FIRST 10 steps from original data, pick players, solve smaller game")
    print("- Goal inference always uses first 10 steps (not current game state)")
    print("- Ego agent acts first control step, others follow ground truth")
    print("- GIF shows all 50 steps with computed vs ground truth trajectories")
    print("- Uses consistent color scheme: ego agent (blue), others (gray), selected (red)")
    print("=" * 80)
    print(f"Using config values:")
    print(f"  PSN Model: {psn_model}")
    print(f"  Goal Model: {goal_model}")
    print(f"  Reference Data: {reference_file}")
    print(f"  Number of Samples: {num_samples}")
    print("=" * 80)
    
    # Run comprehensive testing (output_dir will be automatically determined)
    results = run_comprehensive_testing(
        psn_model, goal_model, reference_file,
        output_dir=None,  # Will be automatically set to PSN model's log directory
        num_samples=num_samples, ego_agent_id=ego_agent_id, 
        test_mode=test_mode
    )
    
    print(f"\nTesting completed successfully!")
    print(f"Results saved to: {results['output_dir']}")
    print(f"Generated {num_samples} GIFs with 50 frames each")
    print(f"Each GIF shows: trajectory evolution, player selection, and goal inference")
    
    return 0


if __name__ == "__main__":
    exit(main())
