#!/usr/bin/env python3
"""
Goal Inference Network Pretraining on Receding Horizon Trajectories

This script pretrains a goal inference network using receding horizon trajectories 
with sliding window inputs. The network learns to predict goal positions from any 
T_observation-length window of trajectory data (e.g., steps 1-10, 21-30, etc.).

Key improvements:
- Uses realistic receding horizon trajectory data
- Supports sliding window inputs from any time segment
- More robust to different trajectory dynamics
- Better generalization for real-world deployment

Author: Assistant
Date: 2024
"""

import json
import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Dict, Tuple, Any, Optional
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GPU issues
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import pickle
from tqdm import tqdm
import os
from datetime import datetime
import gc
from torch.utils.tensorboard import SummaryWriter
import torch.utils.tensorboard as tb
import random
import glob

# JAX/Flax imports
import flax.linen as nn
import optax
from flax.training import train_state
import flax.serialization

# Import from the main lqrax module
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from lqrax import iLQR
from config_loader import load_config, setup_jax_config, create_log_dir


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

# Load configuration
config = load_config()

# Setup JAX configuration
setup_jax_config()

# Extract parameters from config
# Game parameters
N_agents = config.game.N_agents
T_observation = config.game.T_observation
T_total = config.game.T_total
state_dim = config.game.state_dim
goal_dim = 2  # Goal dimension (x, y)

# Training parameters
num_epochs = config.goal_inference.num_epochs
learning_rate = config.goal_inference.learning_rate
batch_size = config.goal_inference.batch_size
goal_loss_weight = config.goal_inference.goal_loss_weight

# Network architecture parameters
if N_agents == 4:
    hidden_dims = config.goal_inference.hidden_dims_4p
elif N_agents == 10:
    hidden_dims = config.goal_inference.hidden_dims_10p
else:
    hidden_dims = config.goal_inference.hidden_dims_4p
    print(f"Warning: Using 4p hidden dimensions {hidden_dims} for {N_agents} agents")

gru_hidden_size = config.goal_inference.gru_hidden_size
dropout_rate = config.goal_inference.dropout_rate

# Finetuning parameters
finetune_with_game_solving = getattr(config.goal_inference, 'finetune_with_game_solving', True)  # Default to True
finetune_start_epoch_ratio = getattr(config.goal_inference, 'finetune_start_epoch_ratio', 0.5)
finetune_learning_rate = getattr(config.goal_inference, 'finetune_learning_rate', 0.0005)
similarity_loss_weight = getattr(config.goal_inference, 'similarity_loss_weight', 0.5)
freeze_gru_layers = getattr(config.goal_inference, 'freeze_gru_layers', True)
freeze_early_mlp_layers = getattr(config.goal_inference, 'freeze_early_mlp_layers', True)

# Game solving parameters (always needed for finetuning)
dt = config.game.dt
T_receding_horizon_planning = config.game.T_receding_horizon_planning
num_iters = config.optimization.num_iters
step_size = config.optimization.step_size

# Data splitting
validation_split = 0.25

# File paths - use receding horizon trajectories for training
rh_data_dir = f"receding_horizon_trajectories_{N_agents}p"

# Random seed
random_seed = config.training.seed

# Testing parameters
num_eval_samples = config.testing.num_test_samples
num_vis_samples = min(5, num_eval_samples)

# Plot parameters
plot_dpi = config.reference_generation.plot_dpi
plot_format = config.reference_generation.plot_format

# Environment parameters
def get_boundary_size(n_agents):
    """Get boundary size based on number of agents."""
    if n_agents <= 4:
        return 2.5
    else:
        return 3.5

boundary_size = get_boundary_size(N_agents)

# Network activation function
activation_function = config.goal_inference.activation

def get_activation_fn(activation_name):
    """Get activation function by name."""
    if activation_name == "relu":
        return nn.relu
    elif activation_name == "tanh":
        return nn.tanh
    elif activation_name == "swish":
        return lambda x: x * nn.sigmoid(x)
    else:
        print(f"Warning: Unknown activation function '{activation_name}', using ReLU")
        return nn.relu

activation_fn = get_activation_fn(activation_function)

# Device selection
gpu_devices = jax.devices("gpu")
if gpu_devices:
    device = gpu_devices[0]
    print(f"Using GPU: {device}")
    print("JAX platform configured via config")
    
    # Test GPU functionality
    test_array = jax.random.normal(jax.random.PRNGKey(0), (10, 10))
    test_result = jnp.linalg.inv(test_array)
    print("GPU matrix operations working correctly")
else:
    raise RuntimeError("No GPU devices found")


# ============================================================================
# GOAL INFERENCE NETWORK DEFINITION
# ============================================================================

class GoalInferenceNetwork(nn.Module):
    """Goal inference network using GRU for temporal sequence processing."""
    
    hidden_dims: List[int]
    gru_hidden_size: int = 64
    dropout_rate: float = 0.3
    goal_output_dim: int = N_agents * goal_dim
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        """
        Forward pass of the goal inference network.
        
        Args:
            x: Input observations (batch_size, T_observation * N_agents * state_dim)
            deterministic: Whether to use deterministic mode (no dropout)
            
        Returns:
            Predicted goals (batch_size, N_agents * goal_dim)
        """
        batch_size = x.shape[0]
        
        # Reshape to separate time steps and agents
        x = x.reshape(batch_size, T_observation, N_agents, state_dim)
        
        # Process each agent's trajectory through shared GRU
        agent_features = []
        
        # Create shared GRU cell
        gru_cell = nn.GRUCell(features=self.gru_hidden_size, name='shared_gru')
        
        for agent_idx in range(N_agents):
            # Extract this agent's trajectory: (batch_size, T_observation, state_dim)
            agent_traj = x[:, :, agent_idx, :]
            
            # Initialize hidden state for this agent
            hidden = jnp.zeros((batch_size, self.gru_hidden_size))
            
            # Process trajectory step by step
            for t in range(T_observation):
                step_input = agent_traj[:, t, :]  # (batch_size, state_dim)
                hidden, _ = gru_cell(hidden, step_input)
            
            # Use final hidden state as agent feature
            agent_features.append(hidden)
        
        # Combine all agent features
        combined_features = jnp.concatenate(agent_features, axis=-1)  # (batch_size, N_agents * gru_hidden_size)
        
        # Apply dropout
        combined_features = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(combined_features)
        
        # MLP layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            combined_features = nn.Dense(hidden_dim, name=f'dense_{i}')(combined_features)
            combined_features = activation_fn(combined_features)
            combined_features = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(combined_features)
        
        # Final output layer for goals
        goals = nn.Dense(self.goal_output_dim, name='goal_output')(combined_features)
        
        return goals


# ============================================================================
# POINT AGENT CLASS FOR GAME SOLVING (only if finetuning enabled)
# ============================================================================

if finetune_with_game_solving:
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


    def create_agent_setup_for_finetuning(initial_states: List[jnp.ndarray], target_positions: List[jnp.ndarray]) -> tuple:
        """
        Create a set of agents with their initial states and reference trajectories for finetuning.
        Uses exact same setup as original trajectory generation.
        
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
        
        for i in range(N_agents):
            # Create agent
            agent = PointAgent(dt=dt, x_dim=4, u_dim=2, Q=Q, R=R)
            agents.append(agent)
            
            # Reference trajectory (simple linear interpolation like original example)
            # Create a straight-line reference trajectory from initial position to target
            start_pos = initial_states[i][:2]  # Extract x, y position
            target_pos = target_positions[i]
            
            # Linear interpolation over time steps (exactly like original ilqgames_example)
            ref_traj = jnp.linspace(start_pos, target_pos, T_receding_horizon_planning)
            reference_trajectories.append(ref_traj)
        
        return agents, reference_trajectories


    def create_loss_functions_for_finetuning(agents: list, reference_trajectories: list) -> tuple:
        """
        Create loss functions and their linearizations for all agents for finetuning.
        Uses exact same setup as original trajectory generation.
        
        Args:
            agents: List of agent objects
            reference_trajectories: List of reference trajectories for each agent
        
        Returns:
            Tuple of (loss_functions, linearize_loss_functions, compiled_functions)
        """
        from jax import vmap, jit, grad
        
        loss_functions = []
        linearize_loss_functions = []
        compiled_functions = []
        
        for i, agent in enumerate(agents):
            # Create loss function for this agent
            def create_runtime_loss(agent_idx, agent_obj, ref_traj):
                def runtime_loss(xt, ut, ref_xt, other_states):
                    # Navigation cost - track reference trajectory (exactly like original)
                    nav_loss = jnp.sum(jnp.square(xt[:2] - ref_xt[:2]))
                    
                    collision_weight = config.optimization.collision_weight
                    collision_scale = config.optimization.collision_scale
                    ctrl_weight = config.optimization.control_weight
                    
                    # Collision avoidance costs - exponential penalty for proximity to other agents
                    # (exactly like original ilqgames_example)
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
            
            # Compile functions (basic compilation for finetuning efficiency)
            compiled_loss = jit(trajectory_loss)
            compiled_linearize = jit(linearize_loss)
            compiled_linearize_dyn = jit(agent.linearize_dyn)
            compiled_solve = jit(agent.solve)
            
            loss_functions.append(trajectory_loss)
            linearize_loss_functions.append(trajectory_loss)
            compiled_functions.append({
                'loss': compiled_loss,
                'linearize_loss': compiled_linearize,
                'linearize_dyn': compiled_linearize_dyn,
                'solve': compiled_solve
            })
        
        return loss_functions, linearize_loss_functions, compiled_functions


    def solve_ilqgames_iterative_for_finetuning(agents: list, 
                                initial_states: list,
                                reference_trajectories: list,
                                compiled_functions: list) -> tuple:
        """
        Solve the iLQGames problem using the original iterative approach for finetuning.
        Uses exact same methodology as original trajectory generation.
        
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
        control_trajectories = [jnp.zeros((T_receding_horizon_planning, 2)) for _ in range(N_agents)]
        
        for iter in range(num_iters + 1):
            # Step 1: Linearize dynamics for all agents
            state_trajectories = []
            A_trajectories = []
            B_trajectories = []
            
            for i in range(N_agents):
                x_traj, A_traj, B_traj = compiled_functions[i]['linearize_dyn'](
                    initial_states[i], control_trajectories[i])
                state_trajectories.append(x_traj)
                A_trajectories.append(A_traj)
                B_trajectories.append(B_traj)
            
            # Step 2: Linearize loss functions for all agents
            a_trajectories = []
            b_trajectories = []
            
            for i in range(N_agents):
                # Create list of other agents' states for this agent
                other_states = [state_trajectories[j] for j in range(N_agents) if j != i]
                
                a_traj, b_traj = compiled_functions[i]['linearize_loss'](
                    state_trajectories[i], control_trajectories[i], reference_trajectories[i], other_states)
                a_trajectories.append(a_traj)
                b_trajectories.append(b_traj)
            
            # Step 3: Solve LQR subproblems for all agents
            control_updates = []
            
            for i in range(N_agents):
                v_traj, _ = compiled_functions[i]['solve'](
                    A_trajectories[i], B_trajectories[i], 
                    a_trajectories[i], b_trajectories[i])
                control_updates.append(v_traj)
            
            # Update control trajectories with gradient descent
            for i in range(N_agents):
                control_trajectories[i] += step_size * control_updates[i]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return state_trajectories, control_trajectories, total_time


    def solve_ilqgames_for_finetuning(agents: list, 
                       initial_states: list,
                       reference_trajectories: list,
                       compiled_functions: list) -> tuple:
        """
        Solve the iLQGames problem for multiple agents using original iterative approach for finetuning.
        """
        return solve_ilqgames_iterative_for_finetuning(agents, initial_states, reference_trajectories, compiled_functions)


    def solve_receding_horizon_game_for_finetuning(agents: list, 
                                   current_states: list,
                                   target_positions: List[jnp.ndarray],
                                   compiled_functions: list) -> tuple:
        """
        Solve a single receding horizon game (T_receding_horizon_planning-horizon) and return the first control.
        Uses exact same methodology as original trajectory generation.
        
        Args:
            agents: List of agent objects
            current_states: Current states for each agent
            target_positions: Target positions for each agent
            compiled_functions: Compiled functions for each agent
        
        Returns:
            Tuple of (first_controls, full_trajectories, total_time)
        """
        # Create reference trajectories from current positions to targets
        current_reference_trajectories = []
        for i in range(N_agents):
            start_pos = current_states[i][:2]  # Extract x, y position
            target_pos = target_positions[i]
            # Linear interpolation over planning horizon
            ref_traj = jnp.linspace(start_pos, target_pos, T_receding_horizon_planning)
            current_reference_trajectories.append(ref_traj)
        
        # Solve the T_receding_horizon_planning-horizon game
        state_trajectories, control_trajectories, total_time = solve_ilqgames_for_finetuning(
            agents, current_states, current_reference_trajectories, compiled_functions)
        
        # Extract the first control from each control trajectory
        first_controls = []
        for i in range(N_agents):
            if len(control_trajectories[i]) > 0:
                first_control = control_trajectories[i][0]  # First control from the computed trajectory
                first_controls.append(first_control)
            else:
                first_controls.append(jnp.zeros(2))  # Fallback to zero control
        
        return first_controls, state_trajectories, total_time


# ============================================================================
# EXACT RECEDING HORIZON TRAJECTORY GENERATION (only if finetuning enabled)
# ============================================================================

if finetune_with_game_solving:
    def generate_receding_horizon_trajectory_for_finetuning(initial_states: list,
                                           target_positions: List[jnp.ndarray]) -> List[jnp.ndarray]:
        """
        Generate receding horizon trajectory by solving T_receding_horizon_planning-horizon games and applying first controls.
        Uses EXACT same methodology as original trajectory generation.
        
        Receding horizon process:
        1. At each iteration, solve a game with horizon T_receding_horizon_planning (e.g., 50 steps)
        2. Extract only the first control/step from the solution
        3. Apply the first control to move agents forward one step
        4. Repeat for T_observation iterations (for finetuning efficiency)
        
        Args:
            initial_states: Initial states for each agent
            target_positions: Target positions for each agent
        
        Returns:
            List of receding horizon state trajectories for each agent
        """
        start_time = time.time()
        
        # Create agent setup using exact same method as original
        agents, reference_trajectories = create_agent_setup_for_finetuning(initial_states, target_positions)
        
        # Create loss functions using exact same method as original
        loss_functions, linearize_functions, compiled_functions = create_loss_functions_for_finetuning(
            agents, reference_trajectories)
        
        # Initialize receding horizon trajectories as JAX arrays
        all_states = []  # Will collect states for all agents and all timesteps
        
        # Current states (start with initial states)
        current_states = [jnp.array(state) for state in initial_states]
        
        # Reduced iterations for finetuning efficiency (match observation window)
        max_rh_iterations = T_observation
        
        # Collect states for each timestep
        timestep_states = []
        
        for step in range(max_rh_iterations):
            
            # Store the current states before applying controls
            current_step_states = jnp.stack(current_states)  # Shape: (N_agents, state_dim)
            timestep_states.append(current_step_states)
            
            # Solve the current T_receding_horizon_planning-horizon game from current states
            # This solves a game looking ahead T_receding_horizon_planning steps (e.g., 50 steps)
            first_controls, full_trajectories, game_time = solve_receding_horizon_game_for_finetuning(
                agents, current_states, target_positions, compiled_functions)
            
            # Apply the first control to move agents forward one step
            new_states = []
            for i in range(N_agents):
                # Get current state and control
                current_state = current_states[i]
                control = first_controls[i]
                
                # Apply dynamics: x_{t+1} = x_t + dt * f(x_t, u_t)
                # For point mass: dx/dt = vx, dy/dt = vy, dvx/dt = ax, dvy/dt = ay
                new_state = jnp.array([
                    current_state[0] + dt * current_state[2],  # x + dt * vx
                    current_state[1] + dt * current_state[3],  # y + dt * vy
                    current_state[2] + dt * control[0],        # vx + dt * ax
                    current_state[3] + dt * control[1]         # vy + dt * ay
                ])
                
                new_states.append(new_state)
            
            # Update current states for next iteration
            current_states = new_states
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Convert to final format: list of trajectories per agent
        # timestep_states is a list of (N_agents, state_dim) arrays
        all_timesteps = jnp.stack(timestep_states)  # Shape: (T_observation, N_agents, state_dim)
        
        final_trajectories = []
        for i in range(N_agents):
            agent_trajectory = all_timesteps[:, i, :]  # Shape: (T_observation, state_dim)
            final_trajectories.append(agent_trajectory)
        
        return final_trajectories


# ============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# ============================================================================

def load_receding_horizon_trajectories_with_sliding_window(directory: str, 
                                                         max_files: Optional[int] = None,
                                                         random_seed: int = 42) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        """
        Forward pass of the goal inference network.
        
        Args:
            x: Input observations (batch_size, T_observation * N_agents * state_dim)
            deterministic: Whether to use deterministic mode (no dropout)
            
        Returns:
            Predicted goals (batch_size, N_agents * goal_dim)
        """
        batch_size = x.shape[0]
        
        # Reshape to separate time steps and agents
        x = x.reshape(batch_size, T_observation, N_agents, state_dim)
        
        # Process each agent's trajectory through shared GRU
        agent_features = []
        
        for agent_idx in range(N_agents):
            # Extract trajectory for this agent: (batch_size, T_observation, state_dim)
            agent_traj = x[:, :, agent_idx, :]
            
            # Use simple GRU cell with manual scanning for compatibility
            gru_cell = nn.GRUCell(features=self.gru_hidden_size, name=f'gru_agent_{agent_idx}')
            
            # Initialize hidden state
            init_hidden = jnp.zeros((batch_size, self.gru_hidden_size))
            
            # Process sequence step by step
            hidden = init_hidden
            for t in range(T_observation):
                hidden, _ = gru_cell(hidden, agent_traj[:, t, :])
            
            # Use final hidden state as agent representation
            agent_features.append(hidden)
        
        # Concatenate all agent features: (batch_size, N_agents * gru_hidden_size)
        x = jnp.concatenate(agent_features, axis=1)
        
        # Apply MLP head for goal prediction
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(features=hidden_dim, name=f'goal_head_{i}')(x)
            x = activation_fn(x)
            if i < len(self.hidden_dims) - 1:
                x = nn.Dropout(rate=self.dropout_rate, name=f'goal_dropout_{i}')(x, deterministic=deterministic)
        
        # Goal prediction output
        goals = nn.Dense(features=self.goal_output_dim, name='goal_output')(x)
        
        return goals


# ============================================================================
# LOSS FUNCTIONS (same as original)
# ============================================================================

def goal_prediction_loss(predicted_goals: jnp.ndarray, true_goals: jnp.ndarray) -> jnp.ndarray:
    """
    Goal prediction loss using Huber loss for robustness.
    """
    batch_size = predicted_goals.shape[0]
    predicted_goals_reshaped = predicted_goals.reshape(batch_size, N_agents, goal_dim)
    true_goals_reshaped = true_goals.reshape(batch_size, N_agents, goal_dim)
    
    goal_diff = predicted_goals_reshaped - true_goals_reshaped
    
    # Huber loss parameters
    delta = 1.0
    
    abs_diff = jnp.abs(goal_diff)
    quadratic = jnp.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    huber_loss_per_dim = 0.5 * quadratic**2 + delta * linear
    
    # Sum over x,y dimensions for each agent
    per_agent_error = jnp.sum(huber_loss_per_dim, axis=2)
    
    # Take mean over N agents for each sample, then mean over all samples
    mean_per_sample = jnp.mean(per_agent_error, axis=1)
    total_loss = jnp.mean(mean_per_sample)
    
    return total_loss


# ============================================================================
# RECEDING HORIZON TRAJECTORY LOADING (DIRECT 1:1 MAPPING)
# ============================================================================

def normalize_receding_horizon_data(rh_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize receding horizon data to standard format.
    """
    normalized_data = rh_data.copy()
    
    if "receding_horizon_trajectories" in rh_data and "trajectories" not in rh_data:
        normalized_data["trajectories"] = {}
        
        for agent_key, agent_data in rh_data["receding_horizon_trajectories"].items():
            if "states" in agent_data:
                # Use the actual executed states from receding horizon
                normalized_data["trajectories"][agent_key] = {"states": agent_data["states"]}
            else:
                # Fallback
                normalized_data["trajectories"][agent_key] = {"states": [[0.0, 0.0, 0.0, 0.0] for _ in range(T_total)]}
    
    return normalized_data

def load_receding_horizon_trajectories(data_dir: str, validation_split: float = 0.25) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load receding horizon trajectories and split into training/validation sets.
    
    Args:
        data_dir: Path to directory containing receding_horizon_sample_*.json files
        validation_split: Fraction of data to use for validation
        
    Returns:
        training_data: List of training samples
        validation_data: List of validation samples
    """
    # Find all receding_horizon_sample_*.json files
    pattern = os.path.join(data_dir, "receding_horizon_sample_*.json")
    json_files = sorted(glob.glob(pattern))
    
    if not json_files:
        raise FileNotFoundError(f"No receding_horizon_sample_*.json files found in directory: {data_dir}")
    
    # Load all samples
    rh_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                sample_data = json.load(f)
                # Normalize the data structure
                normalized_data = normalize_receding_horizon_data(sample_data)
                rh_data.append(normalized_data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue
    
    total_samples = len(rh_data)
    validation_size = int(total_samples * validation_split)
    training_size = total_samples - validation_size
    
    # Shuffle data for random split
    random.shuffle(rh_data)
    
    training_data = rh_data[:training_size]
    validation_data = rh_data[training_size:]
    
    print(f"Loaded {total_samples} receding horizon trajectory samples")
    print(f"Training samples: {len(training_data)}")
    print(f"Validation samples: {len(validation_data)}")
    
    return training_data, validation_data

def extract_first_steps_trajectory(sample_data: Dict[str, Any]) -> jnp.ndarray:
    """
    Extract the first T_observation steps from receding horizon trajectory.
    
    Args:
        sample_data: Trajectory sample data
        
    Returns:
        observation_trajectory: (T_observation, N_agents, state_dim)
    """
    # Initialize array to store all agent states
    observation_trajectory = jnp.zeros((T_observation, N_agents, state_dim))
    
    for i in range(N_agents):
        agent_key = f"agent_{i}"
        states = sample_data["trajectories"][agent_key]["states"]
        
        # Extract first T_observation steps
        if len(states) >= T_observation:
            agent_states = jnp.array(states[:T_observation])
        else:
            # Pad with last state if trajectory is shorter than T_observation
            agent_states_padded = states[:]
            last_state = states[-1] if states else [0.0, 0.0, 0.0, 0.0]
            while len(agent_states_padded) < T_observation:
                agent_states_padded.append(last_state)
            agent_states = jnp.array(agent_states_padded)
        
        # Place in the correct position: (T_observation, N_agents, state_dim)
        observation_trajectory = observation_trajectory.at[:, i, :].set(agent_states)
    
    return observation_trajectory

def extract_reference_goals(sample_data: Dict[str, Any]) -> jnp.ndarray:
    """Extract goal positions for all agents."""
    return jnp.array(sample_data["target_positions"])

def generate_direct_samples(rh_data: List[Dict[str, Any]]) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Generate direct 1:1 samples from receding horizon trajectories.
    Uses only the first T_observation steps as input.
    
    Args:
        rh_data: List of receding horizon trajectory samples
        
    Returns:
        List of (observation_window, goals) tuples (1 per trajectory)
    """
    direct_samples = []
    
    for sample_data in rh_data:
        try:
            # Extract first T_observation steps as input
            obs_window = extract_first_steps_trajectory(sample_data)
            # Extract goals
            goals = extract_reference_goals(sample_data)
            
            direct_samples.append((obs_window, goals))
        except Exception as e:
            print(f"Warning: Failed to extract sample {sample_data.get('sample_id', 'unknown')}: {e}")
            continue
    
    print(f"Generated {len(direct_samples)} direct samples from {len(rh_data)} trajectories (1:1 mapping)")
    
    return direct_samples


# ============================================================================
# FINETUNING FUNCTIONS
# ============================================================================

def similarity_loss_simple(predicted_trajectories: List[jnp.ndarray], 
                           ground_truth_trajectories: List[jnp.ndarray]) -> jnp.ndarray:
    """
    Compute similarity loss between predicted and ground truth RH trajectories.
    """
    if len(predicted_trajectories) != len(ground_truth_trajectories):
        print(f"WARNING: Trajectory count mismatch: {len(predicted_trajectories)} vs {len(ground_truth_trajectories)}")
    
    total_loss = 0.0
    agents_processed = 0
    
    for pred_traj, gt_traj in zip(predicted_trajectories, ground_truth_trajectories):
        if pred_traj.ndim != 2 or gt_traj.ndim != 2:
            continue
        if pred_traj.shape[1] < 2 or gt_traj.shape[1] < 2:
            continue
            
        # Position similarity (most important)
        pred_pos = pred_traj[:, :2]  # Extract x, y positions
        gt_pos = gt_traj[:, :2]      # Extract x, y positions
        
        # Ensure same length by truncating to shorter trajectory
        min_len = min(pred_pos.shape[0], gt_pos.shape[0])
        if min_len == 0:
            continue
            
        pred_pos = pred_pos[:min_len]
        gt_pos = gt_pos[:min_len]
        
        # Mean squared error for positions
        position_loss = jnp.mean((pred_pos - gt_pos) ** 2)
        
        # Velocity similarity (if available)
        velocity_loss = 0.0
        if pred_traj.shape[1] >= 4 and gt_traj.shape[1] >= 4:
            pred_vel = pred_traj[:min_len, 2:4]
            gt_vel = gt_traj[:min_len, 2:4]
            velocity_loss = jnp.mean((pred_vel - gt_vel) ** 2)
        
        # Weighted combination
        agent_loss = position_loss + 0.1 * velocity_loss
        total_loss += agent_loss
        agents_processed += 1
    
    if agents_processed == 0:
        return jnp.array(0.0)
    
    return total_loss / agents_processed


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def goal_prediction_loss(predicted_goals: jnp.ndarray, true_goals: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate goal prediction loss using MSE.
    
    Args:
        predicted_goals: Predicted goals (batch_size, N_agents * goal_dim)
        true_goals: True goals (batch_size, N_agents * goal_dim)
        
    Returns:
        MSE loss between predicted and true goals
    """
    return jnp.mean((predicted_goals - true_goals) ** 2)


def create_train_state(model: nn.Module, learning_rate: float) -> train_state.TrainState:
    """Create training state for the model."""
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=5e-4
    )
    
    # Create dummy input for initialization
    input_shape = (batch_size, T_observation * N_agents * state_dim)
    dummy_input = jnp.ones(input_shape)
    
    # Initialize model parameters
    rng = jax.random.PRNGKey(random_seed)
    variables = model.init(rng, dummy_input)
    params = variables['params']
    
    # Create train state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    return state

def train_step(state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray],
               goal_loss_weight: float = 1.0, rng: jnp.ndarray = None) -> Tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray]:
    """Single training step."""
    observations, true_goals = batch
    
    def loss_fn(params):
        predicted_goals = state.apply_fn({'params': params}, observations, rngs={'dropout': rng}, deterministic=False)
        goal_loss_val = goal_prediction_loss(predicted_goals, true_goals)
        total_loss_val = goal_loss_weight * goal_loss_val
        return total_loss_val, goal_loss_val
    
    # Compute gradients
    (loss, goal_loss_val), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    return state, loss, goal_loss_val


def finetune_train_step_rh(state: train_state.TrainState, 
                          batch: Tuple[jnp.ndarray, jnp.ndarray],
                          raw_trajectory_data: List[Dict[str, Any]],
                          goal_loss_weight: float = 1.0,
                          similarity_loss_weight: float = 0.5,
                          rng: jnp.ndarray = None) -> Tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Finetuning training step that uses predicted goals to solve receding horizon games.
    
    This step:
    1. Predicts goals from observations using the current model
    2. Uses predicted goals to solve the exact same receding horizon game as trajectory generation
    3. Compares the resulting trajectories with ground truth for similarity loss
    4. Combines goal prediction loss with trajectory similarity loss
    
    Args:
        state: Current training state
        batch: Batch of (observations, true_goals)
        raw_trajectory_data: Raw trajectory data for extracting initial states
        goal_loss_weight: Weight for goal prediction loss
        similarity_loss_weight: Weight for trajectory similarity loss
        rng: Random key for dropout
        
    Returns:
        Tuple of (updated_state, total_loss, goal_loss, similarity_loss)
    """
    observations, true_goals = batch
    batch_size_actual = observations.shape[0]
    
    def loss_fn(params):
        # 1. Predict goals using current model
        predicted_goals = state.apply_fn({'params': params}, observations, rngs={'dropout': rng}, deterministic=False)
        
        # 2. Calculate standard goal prediction loss
        goal_loss_val = goal_prediction_loss(predicted_goals, true_goals)
        
        # 3. Calculate trajectory similarity loss using predicted goals
        similarity_loss_val = jnp.array(0.0)
        
        if finetune_with_game_solving:
            # Sample a subset of the batch for game solving (for efficiency)
            max_game_solving_samples = min(2, batch_size_actual)  # Reduce to 2 for easier debugging
            
            similarity_losses = []
            
            for i in range(max_game_solving_samples):
                try:
                    
                    # Extract predicted goals for this sample
                    sample_predicted_goals = predicted_goals[i].reshape(N_agents, goal_dim)  # (N_agents, 2)
                    
                    # Get corresponding initial state from observations
                    # Observations are (T_observation * N_agents * state_dim,)
                    obs_reshaped = observations[i].reshape(T_observation, N_agents, state_dim)
                    initial_states = [obs_reshaped[0, agent_idx, :] for agent_idx in range(N_agents)]  # First timestep
                    
                    # Convert predicted goals to target positions
                    target_positions = [sample_predicted_goals[agent_idx, :] for agent_idx in range(N_agents)]
                    
                    # 4. Solve receding horizon game using predicted goals
                    predicted_trajectories = generate_receding_horizon_trajectory_for_finetuning(
                        initial_states, target_positions)
                    
                    # 5. Get ground truth trajectory for comparison
                    # Use a random sample from raw trajectory data for ground truth
                    if len(raw_trajectory_data) > 0:
                        gt_sample_idx = i % len(raw_trajectory_data)
                        gt_sample = raw_trajectory_data[gt_sample_idx]
                        
                        # Extract ground truth trajectories
                        gt_trajectories = []
                        for agent_idx in range(N_agents):
                            agent_key = f"agent_{agent_idx}"
                            if agent_key in gt_sample["receding_horizon_trajectories"]:
                                gt_states = jnp.array(gt_sample["receding_horizon_trajectories"][agent_key]["states"])
                                gt_trajectories.append(gt_states)
                            else:
                                # Fallback: use initial states repeated
                                fallback_traj = jnp.tile(initial_states[agent_idx].reshape(1, -1), (T_observation, 1))
                                gt_trajectories.append(fallback_traj)
                        
                        # 6. Calculate similarity loss between predicted and ground truth trajectories
                        traj_similarity = similarity_loss_simple(predicted_trajectories, gt_trajectories)
                        similarity_losses.append(traj_similarity)
                        
                except Exception as e:
                    # If game solving fails, use zero similarity loss for this sample
                    print(f"ERROR: Game solving failed for sample {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    similarity_losses.append(jnp.array(0.0))
            
            # Average similarity loss across samples
            if similarity_losses:
                similarity_loss_val = jnp.mean(jnp.array(similarity_losses))
            else:
                similarity_loss_val = jnp.array(0.0)
        
        # 7. Combine losses
        total_loss_val = goal_loss_weight * goal_loss_val + similarity_loss_weight * similarity_loss_val
        
        return total_loss_val, (goal_loss_val, similarity_loss_val)
    
    # Compute gradients
    (loss, (goal_loss_val, similarity_loss_val)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    return state, loss, goal_loss_val, similarity_loss_val


def prepare_batch(batch_data: List[Tuple[jnp.ndarray, jnp.ndarray]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Prepare a batch of sliding window data for training.
    
    Args:
        batch_data: List of (observation_window, goals) tuples
        
    Returns:
        observations: Batch of observations (batch_size, T_observation * N_agents * state_dim)
        true_goals: Batch of true goals (batch_size, N_agents * goal_dim)
    """
    batch_obs = []
    batch_true_goals = []
    
    for obs_window, goals in batch_data:
        batch_obs.append(obs_window.flatten())  # Flatten to 1D
        batch_true_goals.append(goals.flatten())  # Flatten to (N_agents * goal_dim)
    
    # Pad batch if necessary
    if len(batch_obs) < batch_size:
        pad_size = batch_size - len(batch_obs)
        # Pad observations
        obs_pad = jnp.zeros((pad_size, T_observation * N_agents * state_dim))
        batch_obs.extend([obs_pad[i] for i in range(pad_size)])
        # Pad true goals
        true_goals_pad = jnp.zeros((pad_size, N_agents * goal_dim))
        batch_true_goals.extend([true_goals_pad[i] for i in range(pad_size)])
    
    # Convert to JAX arrays
    batch_obs = jnp.stack(batch_obs)
    batch_true_goals = jnp.stack(batch_true_goals)
    
    return batch_obs, batch_true_goals

def evaluate_epoch(goal_model: GoalInferenceNetwork, 
                  state: train_state.TrainState, 
                  validation_samples: List[Tuple[jnp.ndarray, jnp.ndarray]], 
                  batch_size: int) -> float:
    """Evaluate the model on validation data for one epoch."""
    val_losses = []
    
    # Create batches for validation
    num_batches = (len(validation_samples) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(validation_samples))
        batch_data = validation_samples[start_idx:end_idx]
        
        # Prepare batch
        observations, true_goals = prepare_batch(batch_data)
        
        # Forward pass (deterministic, no dropout)
        predicted_goals = state.apply_fn({'params': state.params}, observations, deterministic=True)
        
        # Compute loss
        goal_loss = goal_prediction_loss(predicted_goals, true_goals)
        val_losses.append(float(goal_loss))
    
    return np.mean(val_losses)

def train_goal_inference_network_rh(goal_model: GoalInferenceNetwork, 
                                   training_samples: List[Tuple[jnp.ndarray, jnp.ndarray]],
                                   validation_samples: List[Tuple[jnp.ndarray, jnp.ndarray]],
                                   num_epochs: int = 50, 
                                   learning_rate: float = 0.001,
                                   batch_size: int = 32,
                                   goal_loss_weight: float = 1.0) -> Tuple[List[float], train_state.TrainState, str, float, int]:
    """Train the goal inference network with receding horizon sliding window data."""
    print(f"Training Goal Inference Network on Receding Horizon data...")
    print(f"Training samples: {len(training_samples)}, Validation samples: {len(validation_samples)}")
    print(f"Parameters: epochs={num_epochs}, lr={learning_rate}, batch_size={batch_size}")
    print(f"Observation window length: {T_observation}, Total trajectory length: {T_total}")
    
    # Create log directory with RH suffix to distinguish from reference trajectory training
    log_dir = create_log_dir("goal_inference_rh", config)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create training state
    state = create_train_state(goal_model, learning_rate)
    
    # Training loop
    losses = []
    best_loss = float('inf')
    best_epoch = 0
    
    # Initialize random key for training
    rng = jax.random.PRNGKey(random_seed)
    
    # Progress bar for epochs
    progress_bar = tqdm(range(num_epochs), desc="Training Progress")
    
    for epoch in progress_bar:
        epoch_losses = []
        
        # Shuffle training data for each epoch
        random.shuffle(training_samples)
        
        # Create batches
        num_batches = (len(training_samples) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(training_samples))
            batch_data = training_samples[start_idx:end_idx]
            
            # Prepare batch
            observations, true_goals = prepare_batch(batch_data)
            
            # Split random key for this step
            rng, step_key = jax.random.split(rng)
            
            # Training step
            state, loss, goal_loss = train_step(state, (observations, true_goals), goal_loss_weight, step_key)
            epoch_losses.append(float(loss))
        
        # Calculate average loss for the epoch
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        # Validation
        val_loss = evaluate_epoch(goal_model, state, validation_samples, batch_size)
        
        # Update progress bar
        progress_bar.set_postfix({'Avg Loss': f'{avg_loss:.4f}', 'Val Loss': f'{val_loss:.4f}'})
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Training', avg_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            save_model(state, log_dir, "goal_inference_rh_best_model.pkl")
            print(f"New best model found at epoch {best_epoch} with validation loss: {best_loss:.4f}")
    
    # Close TensorBoard writer
    writer.close()
    
    # Save final model
    save_model(state, log_dir, "goal_inference_rh_final_model.pkl")
    
    # Save training configuration
    save_training_config(log_dir, num_epochs, learning_rate, batch_size, goal_loss_weight)
    
    print(f"\nTraining completed!")
    print(f"Log directory: {log_dir}")
    print(f"Best model saved to: {log_dir}/goal_inference_rh_best_model.pkl")
    print(f"Final training loss: {losses[-1]:.4f}")
    print(f"Best validation loss: {best_loss:.4f} (achieved at epoch {best_epoch})")
    
    return losses, state, log_dir, best_loss, best_epoch

def save_model(state: train_state.TrainState, log_dir: str, filename: str):
    """Save model to file."""
    model_bytes = flax.serialization.to_bytes(state)
    model_path = os.path.join(log_dir, filename)
    with open(model_path, 'wb') as f:
        pickle.dump(model_bytes, f)

def save_training_config(log_dir: str, num_epochs: int, learning_rate: float, batch_size: int, goal_loss_weight: float):
    """Save training configuration to JSON file."""
    training_results = {
        'training_data_type': 'receding_horizon_trajectories',
        'sliding_window_approach': True,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'goal_loss_weight': goal_loss_weight,
        'N_agents': N_agents,
        'T_observation': T_observation,
        'T_total': T_total,
        'state_dim': state_dim,
        'goal_dim': goal_dim,
        'validation_split': validation_split,
        'gru_hidden_size': gru_hidden_size,
        'hidden_dims': hidden_dims,
        'dropout_rate': dropout_rate
    }
    
    training_results_path = os.path.join(log_dir, 'training_results.json')
    with open(training_results_path, 'w') as f:
        json.dump(training_results, f, indent=2)


def train_goal_inference_network_rh_with_finetuning(
    model: GoalInferenceNetwork,
    training_data: List[Tuple[jnp.ndarray, jnp.ndarray]],
    validation_data: List[Tuple[jnp.ndarray, jnp.ndarray]],
    raw_trajectory_data: List[Dict[str, Any]],
    num_epochs: int = 20,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    goal_loss_weight: float = 1.0
) -> Tuple[List[float], train_state.TrainState, str, float, int]:
    """
    Train goal inference network with optional finetuning using true RH game solving.
    """
    # Create log directory
    log_dir = create_log_dir("goal_inference_rh_gru_finetune", config)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Create training state
    state = create_train_state(model, learning_rate)
    
    # Calculate finetuning start epoch
    finetune_start_epoch = int(num_epochs * finetune_start_epoch_ratio) if finetune_with_game_solving else num_epochs + 1
    
    # Training tracking
    losses = []
    best_loss = float('inf')
    best_epoch = 0
    best_state = None
    
    print(f"\\nStarting training for {num_epochs} epochs...")
    print(f"Training data: {len(training_data)} samples")
    print(f"Validation data: {len(validation_data)} samples")
    if finetune_with_game_solving:
        print(f"Finetuning will start at epoch {finetune_start_epoch}")
    
    for epoch in range(num_epochs):
        # Determine if we're in finetuning phase
        is_finetuning = finetune_with_game_solving and epoch >= finetune_start_epoch
        
        # Switch learning rate for finetuning
        if is_finetuning and epoch == finetune_start_epoch:
            print(f"\\n🔧 Switching to finetuning mode at epoch {epoch}!")
            # Reduce learning rate for finetuning
            new_optimizer = optax.adamw(learning_rate=finetune_learning_rate, weight_decay=5e-4)
            state = state.replace(tx=new_optimizer, opt_state=new_optimizer.init(state.params))
        
        # Shuffle training data
        epoch_key = jax.random.fold_in(jax.random.PRNGKey(random_seed), epoch)
        shuffled_indices = jax.random.permutation(epoch_key, len(training_data))
        shuffled_training_data = [training_data[i] for i in shuffled_indices]
        
        # Training phase
        epoch_losses = []
        epoch_goal_losses = []
        epoch_similarity_losses = []
        
        # Process batches
        batch_indices = list(range(0, len(shuffled_training_data), batch_size))
        
        if is_finetuning:
            # Add progress bar for finetuning samples
            batch_progress = tqdm(batch_indices, 
                                desc=f"Finetuning Epoch {epoch} - {len(shuffled_training_data)} samples",
                                leave=False,
                                unit="batch",
                                postfix={'samples_processed': 0})
        else:
            batch_progress = batch_indices
        
        for i in batch_progress:
            batch_data = shuffled_training_data[i:i + batch_size]
            batch = prepare_batch(batch_data)
            
            step_key = jax.random.fold_in(epoch_key, i // batch_size)
            
            if is_finetuning:
                try:
                    # Use finetuning with game solving
                    state, loss, goal_loss, similarity_loss = finetune_train_step_rh(
                        state, batch, raw_trajectory_data, goal_loss_weight, similarity_loss_weight, step_key
                    )

                    epoch_losses.append(float(loss))
                    epoch_goal_losses.append(float(goal_loss))
                    epoch_similarity_losses.append(float(similarity_loss))
                    
                    # Update progress bar with samples processed
                    samples_processed = min(i + batch_size, len(shuffled_training_data))
                    if hasattr(batch_progress, 'set_postfix'):
                        batch_progress.set_postfix({'samples_processed': f"{samples_processed}/{len(shuffled_training_data)}"})
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    # Fallback to regular training
                    state, loss, goal_loss = train_step(state, batch, goal_loss_weight, step_key)
                    epoch_losses.append(float(loss))
                    epoch_goal_losses.append(float(goal_loss))
                    epoch_similarity_losses.append(0.0)  # Zero similarity loss on fallback
            else:
                # Regular training
                state, loss, goal_loss = train_step(state, batch, goal_loss_weight, step_key)
                epoch_losses.append(float(loss))
                epoch_goal_losses.append(float(goal_loss))
        
        # Calculate epoch averages
        avg_loss = np.mean(epoch_losses)
        avg_goal_loss = np.mean(epoch_goal_losses)
        avg_similarity_loss = np.mean(epoch_similarity_losses) if epoch_similarity_losses else 0.0
        
        losses.append(avg_loss)
        
        # Validation
        val_losses = []
        for i in range(0, len(validation_data), batch_size):
            val_batch_data = validation_data[i:i + batch_size]
            val_batch = prepare_batch(val_batch_data)
            
            observations, true_goals = val_batch
            val_predicted_goals = model.apply({'params': state.params}, observations, deterministic=True)
            val_goal_loss = goal_prediction_loss(val_predicted_goals, true_goals)
            val_losses.append(float(val_goal_loss))
        
        avg_val_loss = np.mean(val_losses)
        
        # Track best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_epoch = epoch
            best_state = state
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Training', avg_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('GoalLoss/Train', avg_goal_loss, epoch)
        
        if is_finetuning:
            writer.add_scalar('SimilarityLoss/Train', avg_similarity_loss, epoch)
            writer.add_scalar('FinetuningMode', 1.0, epoch)
            print(f"Epoch {epoch:3d} [FINETUNE]: Loss={avg_loss:.4f} (Goal={avg_goal_loss:.4f}, Sim={avg_similarity_loss:.4f}), Val={avg_val_loss:.4f}")
        else:
            writer.add_scalar('FinetuningMode', 0.0, epoch)
            print(f"Epoch {epoch:3d} [REGULAR ]: Loss={avg_loss:.4f} (Goal={avg_goal_loss:.4f}), Val={avg_val_loss:.4f}")
    
    # Save best model
    if best_state is not None:
        model_path = os.path.join(log_dir, "goal_inference_rh_best_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'params': best_state.params,
                'model_config': {
                    'hidden_dims': hidden_dims,
                    'gru_hidden_size': gru_hidden_size,
                    'dropout_rate': dropout_rate
                }
            }, f)
    
    # Close TensorBoard writer
    writer.close()
    
    return losses, best_state if best_state is not None else state, log_dir, best_loss, best_epoch


# ============================================================================
# EVALUATION AND VISUALIZATION FUNCTIONS
# ============================================================================

def evaluate_goal_inference_model_rh(model: nn.Module, trained_state: train_state.TrainState, 
                                    validation_samples: List[Tuple[jnp.ndarray, jnp.ndarray]], 
                                    num_samples: int = 20) -> Dict[str, float]:
    """Evaluate the trained goal inference model on goal prediction accuracy."""
    print(f"\nEvaluating Goal Inference model on {num_samples} sliding window samples...")
    
    goal_prediction_errors = []
    
    # Sample random indices
    rng = jax.random.PRNGKey(random_seed)
    rng, sample_key = jax.random.split(rng)
    sample_indices = jax.random.choice(sample_key, len(validation_samples), shape=(num_samples,), replace=False)
    
    for idx in sample_indices:
        obs_window, true_goals = validation_samples[idx]
        
        # Add batch dimension and flatten
        obs_input = obs_window.flatten().reshape(1, -1)
        
        # Get model predictions
        predicted_goals = trained_state.apply_fn({'params': trained_state.params}, obs_input, deterministic=True)
        
        # Compute goal prediction error
        goal_error = jnp.mean(jnp.square(predicted_goals[0] - true_goals.flatten()))
        goal_prediction_errors.append(float(goal_error))
    
    # Compute metrics
    avg_goal_error = np.mean(goal_prediction_errors)
    std_goal_error = np.std(goal_prediction_errors)
    
    metrics = {
        'avg_goal_prediction_error': avg_goal_error,
        'std_goal_prediction_error': std_goal_error,
        'goal_prediction_rmse': np.sqrt(avg_goal_error)
    }
    
    print(f"Goal Prediction RMSE: {metrics['goal_prediction_rmse']:.4f}")
    print(f"Goal Prediction Error (mean ± std): {avg_goal_error:.4f} ± {std_goal_error:.4f}")
    
    return metrics


# ============================================================================
# TRAINING FUNCTION WITH FINETUNING
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Goal Inference Network Pretraining on Receding Horizon Trajectories")
    print("=" * 60)
    print(f"Configuration loaded from: config.yaml")
    print(f"Game parameters: N_agents={N_agents}, T_observation={T_observation}, T_total={T_total}")
    print(f"Training parameters: epochs={num_epochs}, lr={learning_rate}, batch_size={batch_size}")
    print(f"Network parameters: hidden_dims={hidden_dims}, gru_hidden_size={gru_hidden_size}, dropout_rate={dropout_rate}")
    print(f"Finetuning parameters: finetune_with_game_solving={finetune_with_game_solving}, similarity_loss_weight={similarity_loss_weight}")
    print("=" * 60)
    
    # Load receding horizon trajectories
    print(f"Loading receding horizon trajectories from directory: {rh_data_dir}...")
    training_data, validation_data = load_receding_horizon_trajectories(rh_data_dir, validation_split)
    
    # Generate direct 1:1 samples (first 10 steps → goals)
    print(f"Generating direct samples (1:1 mapping)...")
    training_samples = generate_direct_samples(training_data)
    validation_samples = generate_direct_samples(validation_data)
    
    print(f"Final dataset sizes:")
    print(f"  Training samples: {len(training_samples)}")
    print(f"  Validation samples: {len(validation_samples)}")
    
    # Create goal inference model
    goal_model = GoalInferenceNetwork(
        hidden_dims=hidden_dims,
        gru_hidden_size=gru_hidden_size,
        dropout_rate=dropout_rate
    )
    
    # Train the goal inference network with finetuning capability
    if finetune_with_game_solving:
        print("Training Goal Inference Network with RH Finetuning...")
        print(f"Finetuning will start at epoch {int(num_epochs * finetune_start_epoch_ratio)} ({finetune_start_epoch_ratio*100:.0f}% of training)")
    else:
        print("Training Goal Inference Network on Receding Horizon data (standard training)...")
    
    # Call the finetuning-capable training function
    losses, trained_state, log_dir, best_loss, best_epoch = train_goal_inference_network_rh_with_finetuning(
        goal_model, training_samples, validation_samples, training_data,
        num_epochs=num_epochs, 
        learning_rate=learning_rate,
        batch_size=batch_size, 
        goal_loss_weight=goal_loss_weight
    )
    
    # Save training results
    training_results = {
        'training_data_type': 'receding_horizon_trajectories',
        'sliding_window_approach': False,
        'direct_mapping_approach': True,
        'finetune_with_game_solving': finetune_with_game_solving,
        'N_agents': N_agents,
        'T_observation': T_observation,
        'T_total': T_total,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'goal_loss_weight': goal_loss_weight,
        'num_epochs': num_epochs,
        'final_loss': float(losses[-1]),
        'best_loss': float(best_loss),
        'best_epoch': int(best_epoch),
        'timestamp': datetime.now().isoformat(),
        'config_source': 'config.yaml'
    }
    config_path = os.path.join(log_dir, "training_results.json")
    with open(config_path, 'w') as f:
        json.dump(training_results, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Log directory: {log_dir}")
    print(f"Best model saved to: {log_dir}/goal_inference_rh_best_model.pkl")
    print(f"Training results saved to: {config_path}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Best loss: {best_loss:.4f} (achieved at epoch {best_epoch})")
    
    # Evaluate the trained model
    evaluation_metrics = evaluate_goal_inference_model_rh(goal_model, trained_state, validation_samples, num_samples=num_eval_samples)
    
    # Save evaluation metrics
    metrics_path = os.path.join(log_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    print(f"Evaluation metrics saved to: {metrics_path}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Goal Inference Training Loss (Receding Horizon Data)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plot_path = os.path.join(log_dir, f"training_loss.{plot_format}")
    plt.savefig(plot_path, dpi=plot_dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Training loss plot saved to: {plot_path}")
    print(f"\nTo view TensorBoard logs, run: tensorboard --logdir={log_dir}")
    print(f"\nThis model can now be used for PSN testing with receding horizon trajectories!")
    print(f"The direct mapping approach (first 10 steps → goals) provides clean 1:1 training.")
    if finetune_with_game_solving:
        print(f"Finetuning with true receding horizon game solving enables trajectory-aware goal inference!")
