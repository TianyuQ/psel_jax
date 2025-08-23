#!/usr/bin/env python3
"""
PSN Training with Pretrained Goal Inference Network

This script trains the Player Selection Network (PSN) using a pretrained
goal inference network. The PSN learns to select important agents while
the goal inference network provides accurate goal predictions.

Stage 2 of the two-stage training approach:
1. Stage 1: Pretrain Goal Inference Network (completed)
2. Stage 2: Train PSN using pretrained goals (this script)

Log Organization:
- PSN training logs are organized under the specific goal inference model directory
- Structure: log/goal_inference_*/psn_pretrained_goals_*/
- This allows easy comparison between different goal inference models

DEBUG MODE:
- Using small dataset for faster debugging
- Game solving is preserved - masks affect game dynamics
- Focus on fixing gradient flow in the game-solving pipeline

Author: Assistant
Date: 2024
"""

# Set JAX environment variables to help with GPU issues
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import json
import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Dict, Tuple, Any, Optional
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
import pickle
from tqdm import tqdm
import os
from datetime import datetime
import gc
from torch.utils.tensorboard import SummaryWriter
import torch.utils.tensorboard as tb

# JAX/Flax imports
import flax.linen as nn
import optax
from flax.training import train_state
import flax.serialization

# Import from the main lqrax module
import sys
import os
# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from lqrax import iLQR  # Now needed for PointAgent
from config_loader import load_config

# Import the goal inference network
# Copy the necessary classes and functions directly to avoid import issues


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
    
    def dyn(self, x, u):
        """Dynamics function for point mass."""
        return jnp.array([
            x[2],  # dx/dt = vx
            x[3],  # dy/dt = vy
            u[0],  # dvx/dt = ax
            u[1]   # dvy/dt = ay
        ])

# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================

# Load configuration from config.yaml
config = load_config()

# Game parameters
N_agents = config.game.N_agents
dt = config.game.dt
T_total = config.game.T_total
T_observation = config.goal_inference.observation_length
T_reference = config.game.T_total
state_dim = config.game.state_dim
control_dim = config.game.control_dim

# PSN training parameters
num_epochs = config.psn.num_epochs
learning_rate = config.psn.learning_rate
batch_size = config.psn.batch_size
sigma1 = config.psn.sigma1  # Final mask sparsity weight (will gradually increase from 0)
sigma2 = config.psn.sigma2  # Binary loss weight

# Game solving parameters
num_iters = config.optimization.num_iters

# Reference trajectory parameters
# Use dataset directory for loading individual sample files
# This directory contains ref_traj_sample_*.json files
reference_dir = config.paths.reference_data_dir

# Pretrained goal inference model path
# Use the template to generate the path based on current config parameters
pretrained_goal_model_path = config.testing.goal_inference_model_template.format(
        N_agents=config.game.N_agents,
        T_total=config.game.T_total,
        T_observation=config.goal_inference.observation_length,
        learning_rate=config.goal_inference.learning_rate,
        batch_size=config.goal_inference.batch_size,
        goal_loss_weight=config.goal_inference.goal_loss_weight,
        num_epochs=config.goal_inference.num_epochs
    )

# Set hidden dimensions based on number of agents for both networks
if N_agents == 4:
    goal_inference_hidden_dims = config.goal_inference.hidden_dims_4p
    psn_hidden_dims = config.psn.hidden_dims_4p
elif N_agents == 10:
    goal_inference_hidden_dims = config.goal_inference.hidden_dims_10p
    psn_hidden_dims = config.psn.hidden_dims_10p
else:
    # Default fallback to 4p dimensions
    goal_inference_hidden_dims = config.goal_inference.hidden_dims_4p
    psn_hidden_dims = config.psn.hidden_dims_4p
    print(f"Warning: Using 4p hidden dimensions for {N_agents} agents")

# Device selection - Use config preference
if config.device.preferred_device == "gpu":
    gpu_devices = jax.devices("gpu")
    if gpu_devices:
        device = gpu_devices[0]
        print(f"Using GPU: {device}")
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
        print("JAX platform forced to: gpu")
        
        # Test GPU functionality with a simple operation
        test_array = jax.random.normal(jax.random.PRNGKey(config.training.seed), (10, 10))
        test_result = jnp.linalg.inv(test_array)
        print("GPU matrix operations working correctly")
    else:
        if config.device.preferred_device == "gpu":
            raise RuntimeError("No GPU devices found, but GPU is preferred")
        else:
            print("No GPU devices found, falling back to CPU")
            device = jax.devices("cpu")[0]
            print(f"Using CPU: {device}")
else:
    device = jax.devices("cpu")[0]
    print(f"Using CPU: {device}")

# ============================================================================
# GOAL INFERENCE NETWORK DEFINITION
# ============================================================================

class GoalInferenceNetwork(nn.Module):
    """Goal inference network for predicting agent goals from observation trajectories."""
    
    hidden_dims: List[int]
    goal_output_dim: int = N_agents * 2  # goal_dim = 2 for (x, y)
    
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
        
        # Average over time steps to get a summary representation
        x = jnp.mean(x, axis=1)  # (batch_size, N_agents, state_dim)
        
        # Flatten all agent states
        x = x.reshape(batch_size, N_agents * state_dim)
        
        # Feature extraction layers using the full hidden_dims list
        x = nn.Dense(features=self.hidden_dims[0])(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=config.goal_inference.dropout_rate)(x, deterministic=deterministic)
        
        x = nn.Dense(features=self.hidden_dims[1])(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=config.goal_inference.dropout_rate)(x, deterministic=deterministic)
        
        x = nn.Dense(features=self.hidden_dims[2])(x)
        x = nn.relu(x)
        
        # Goal prediction output
        goals = nn.Dense(features=self.goal_output_dim)(x)
        # No activation for goals - they can be any real values
        
        return goals

def extract_observation_trajectory(sample_data: Dict[str, Any]) -> jnp.ndarray:
    """
    Extract observation trajectory (first 10 steps) for all agents.
    
    Args:
        sample_data: Reference trajectory sample
        
    Returns:
        observation_trajectory: Observation trajectory (T_observation, N_agents, state_dim)
    """
    # Initialize array to store all agent states
    # Shape: (T_observation, N_agents, state_dim)
    observation_trajectory = jnp.zeros((T_observation, N_agents, state_dim))
    
    for i in range(N_agents):
        agent_key = f"agent_{i}"
        states = sample_data["trajectories"][agent_key]["states"]
        # Take first T_observation steps
        agent_states = jnp.array(states[:T_observation])  # (T_observation, state_dim)
        # Place in the correct position: (T_observation, N_agents, state_dim)
        observation_trajectory = observation_trajectory.at[:, i, :].set(agent_states)
    
    return observation_trajectory

# ============================================================================
# PSN NETWORK DEFINITIONS (GOAL-FREE VERSION)
# ============================================================================
class PlayerSelectionNetwork(nn.Module):
    """
    Player Selection Network (PSN) that learns to select important agents.
    
    Input: First 10 steps of all agents' trajectories (T_observation * N_agents * state_dim)
    Output: Binary mask for selecting other agents (excluding ego agent)
    """
    
    hidden_dims: List[int]
    mask_output_dim: int = N_agents - 1  # Mask for other agents (excluding ego)
    
    @nn.compact
    def __call__(self, x):
        """
        Forward pass of PSN.
        
        Args:
            x: Input tensor of shape (batch_size, T_observation * N_agents * state_dim)
            
        Returns:
            mask: Binary mask of shape (batch_size, N_agents - 1)
        """
        # Reshape input to (batch_size, T_observation, N_agents, state_dim)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, T_observation, N_agents, state_dim)
        
        # Average over time steps to get a summary representation
        x = jnp.mean(x, axis=1)  # (batch_size, N_agents, state_dim)
        
        # Flatten all agent states
        x = x.reshape(batch_size, N_agents * state_dim)
        
        # Feature extraction layers using the full hidden_dims list
        x = nn.Dense(features=self.hidden_dims[0])(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dims[1])(x)
        x = nn.relu(x)
        
        # Mask output
        mask = nn.Dense(features=self.mask_output_dim)(x)
        mask = nn.sigmoid(mask)  # Binary mask
        
        return mask

# ============================================================================
# GAME SOLVING FUNCTIONS
# ============================================================================

def create_masked_game_setup(sample_data: Dict[str, Any], ego_agent_id: int,
                           predicted_mask: jnp.ndarray, predicted_goals: jnp.ndarray, 
                           is_training: bool = True) -> Tuple[List, List, jnp.ndarray, jnp.ndarray]:
    """
    Create masked game setup based on predicted mask and goals.
    
    During training: ALL agents are included, but only agent 0 (ego agent) uses mask values
    for mutual costs. Other agents always consider full mutual costs with all agents.
    During runtime: Only selected agents are included based on binary mask threshold.
    
    Args:
        sample_data: Reference trajectory sample
        ego_agent_id: ID of the ego agent
        predicted_mask: Predicted mask from PSN (N_agents - 1)
        predicted_goals: Predicted goals from pretrained network (N_agents * 2)
        is_training: Whether this is for training (masked mutual costs) or testing (agent selection)
        
    Returns:
        agents: List of selected agents
        initial_states: List of initial states for selected agents
        target_positions: Target positions for selected agents
        mask_values: Mask values for the game
    """
    # Parse predicted goals
    predicted_goals = predicted_goals.reshape(-1, 2)  # (N_agents, 2)
    
    if is_training:
        # Training time: Include ALL agents, mask values multiply the mutual costs
        # This implements the equation: \tilde{c}_{s,k}^i = \sum_{j=1}^N m^{ij}c_k^{ij}
        agents = []
        initial_states = []
        target_positions = []
        
        # Cost function weights (same for all agents) - matching original ilqgames_example.py
        Q = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001]))  # State cost weights [x, y, vx, vy]
        R = jnp.diag(jnp.array([0.01, 0.01]))               # Control cost weights [ax, ay]
        
        # Get initial positions from sample data
        original_positions = []
        for agent_id in range(N_agents):
            agent_key = f"agent_{agent_id}"
            agent_states = sample_data["trajectories"][agent_key]["states"]
            pos_2d = jnp.array(agent_states[0][:2])  # [x, y]
            original_positions.append(pos_2d)
        
        # Create conflicting scenario: agents start in corners and cross paths
        # Similar to ilqgames_example.py setup with crossing trajectories
        
        # Define 4 starting positions in corners of a square
        start_positions = [
            jnp.array([-1.0, -1.0]),  # Agent 0: bottom-left
            jnp.array([1.0, -1.0]),   # Agent 1: bottom-right  
            jnp.array([1.0, 1.0]),    # Agent 2: top-right
            jnp.array([-1.0, 1.0])    # Agent 3: top-left
        ]
        
        # Define 4 goal positions: each agent goes to opposite corner (crossing paths)
        goal_positions = [
            jnp.array([1.0, 1.0]),    # Agent 0: to top-right
            jnp.array([-1.0, 1.0]),   # Agent 1: to top-left
            jnp.array([-1.0, -1.0]),  # Agent 2: to bottom-left  
            jnp.array([1.0, -1.0])    # Agent 3: to bottom-right
        ]
        
        for agent_id in range(N_agents):
            # Create agent
            agent = PointAgent(dt=dt, x_dim=4, u_dim=2, Q=Q, R=R)
            agents.append(agent)
            
            # Set initial state: [x, y, vx, vy] starting from zero velocity
            start_pos = start_positions[agent_id]
            initial_state = jnp.array([start_pos[0], start_pos[1], 0.0, 0.0])
            initial_states.append(initial_state)
            
            # Set goal position (crossing path scenario)
            target_positions.append(goal_positions[agent_id])
        
        # For training, only agent 0 (ego agent) uses the mask
        # Other agents have full interaction with all agents (no masking)
        # Simply return the mask values directly - they will only be used by agent 0
        return agents, initial_states, jnp.array(target_positions), predicted_mask
        
    else:
        # Runtime: Only include selected agents based on binary mask threshold
        # This implements the equation: m_{ij} > m_th -> 1, otherwise 0
        selected_agents = [ego_agent_id]
        mask_values = []
        
        for i in range(N_agents - 1):
            # Map mask index to actual agent ID (skip ego agent)
            agent_id = i if i < ego_agent_id else i + 1
            if predicted_mask[i] > config.psn.mask_threshold:  # Threshold for selection
                selected_agents.append(agent_id)
                mask_values.append(predicted_mask[i])
        
        # Create agents and get their initial states
        agents = []
        initial_states = []
        selected_targets = []
        
        # Cost function weights (same for all agents) - matching original ilqgames_example.py
        Q = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001]))  # State cost weights [x, y, vx, vy]
        R = jnp.diag(jnp.array([0.01, 0.01]))               # Control cost weights [ax, ay]
        
        for agent_id in selected_agents:
            # Create agent
            agent = PointAgent(dt=dt, x_dim=4, u_dim=2, Q=Q, R=R)
            agents.append(agent)
            
            # Initial state - convert 2D position to 4D state [x, y, vx, vy]
            agent_key = f"agent_{agent_id}"
            agent_states = sample_data["trajectories"][agent_key]["states"]
            pos_2d = jnp.array(agent_states[0][:2])  # [x, y]
            initial_state = jnp.array([pos_2d[0], pos_2d[1], 0.0, 0.0])  # [x, y, vx, vy]
            initial_states.append(initial_state)
            
            # Use predicted goal position for this agent
            selected_targets.append(predicted_goals[agent_id])
        
        return agents, initial_states, jnp.array(selected_targets), jnp.array(mask_values)

def create_loss_functions(agents: list, mask_values=None, is_training: bool = True, reference_trajectories: list = None) -> tuple:
    """
    Create loss functions and their linearizations for all agents.
    
    During training: mask_values is the original mask array (only used by agent 0/ego agent)
    During runtime: mask_values is a list of mask values for selected agents
    
    Args:
        agents: List of agent objects
        mask_values: Mask array (training) or mask list (runtime)
        is_training: Whether this is for training or runtime
        
    Returns:
        Tuple of (loss_functions, linearize_loss_functions, compiled_functions)
    """
    loss_functions = []
    linearize_loss_functions = []
    compiled_functions = []
    
    for i, agent in enumerate(agents):
        # Create loss function for this agent
        def create_runtime_loss(agent_idx, agent_obj, is_training=False):
            def runtime_loss(xt, ut, goal_pos, other_states, mask_values, ref_traj=None):
                # Navigation cost - follow reference trajectory if available, otherwise go to goal
                if ref_traj is not None and len(ref_traj) > 0:
                    # Use reference trajectory for navigation (like ilqgames_example.py)
                    nav_loss = jnp.sum(jnp.square(xt[:2] - ref_traj[:2]))
                else:
                    # Fallback to goal-based navigation
                    nav_loss = jnp.sum(jnp.square(xt[:2] - goal_pos[:2]))
                
                # Collision avoidance costs with mask-based filtering
                collision_loss = 0.0
                if len(other_states) > 0 and mask_values is not None:
                    if is_training:
                        # Training: Only agent 0 (ego agent) uses the mask
                        # Other agents have full interaction with ALL agents (including ego agent)
                        other_positions = jnp.stack([other_xt[:2] for other_xt in other_states])
                        distances_squared = jnp.sum(jnp.square(xt[:2] - other_positions), axis=1)
                        
                        # Base collision cost: using config values for consistency
                        collision_weight = config.optimization.collision_weight
                        collision_scale = config.optimization.collision_scale
                        base_collision = collision_weight * jnp.exp(-collision_scale * distances_squared)
                        
                        if agent_idx == 0:  # Ego agent (agent 0)
                            # Apply mask values to ego agent's interactions with other agents
                            # mask_values contains [m_ego_1, m_ego_2, ..., m_ego_N-1]
                            # other_states order: [agent_1, agent_2, ..., agent_N-1] (excluding ego)
                            masked_collision = base_collision * mask_values[:len(other_states)]
                        else:
                            # Other agents: full interaction with ALL agents (including ego agent)
                            # This means they consider collision costs with agent 0 and all other agents
                            # No masking applied - they always consider full mutual costs
                            masked_collision = base_collision
                        
                        collision_loss = jnp.sum(masked_collision)
                    else:
                        # Runtime: Use simple mask values for selected agents
                        other_positions = jnp.stack([other_xt[:2] for other_xt in other_states])
                        distances_squared = jnp.sum(jnp.square(xt[:2] - other_positions), axis=1)
                        collision_weight = config.optimization.collision_weight
                        collision_scale = config.optimization.collision_scale
                        base_collision = collision_weight * jnp.exp(-collision_scale * distances_squared)
                        masked_collision = base_collision * mask_values[:len(other_states)]
                        collision_loss = jnp.sum(masked_collision) / (len(agents) - 1)
                
                # Control cost - using config values for consistency
                ctrl_weight = config.optimization.control_weight
                ctrl_loss = ctrl_weight * jnp.sum(jnp.square(ut))
                
                return nav_loss + collision_loss + ctrl_loss
            
            return runtime_loss
        
        # Get reference trajectory for this agent if available
        ref_traj = None
        if reference_trajectories is not None and i < len(reference_trajectories):
            ref_traj = reference_trajectories[i]
        
        runtime_loss = create_runtime_loss(i, agent, is_training)
        
        # Create trajectory loss function
        def trajectory_loss(x_traj, u_traj, goal_pos, other_x_trajs, mask_values):
            def single_step_loss(args):
                xt, ut, other_xts = args
                return runtime_loss(xt, ut, goal_pos, other_xts, mask_values, ref_traj)
            
            loss_array = jax.vmap(single_step_loss)((x_traj, u_traj, other_x_trajs))
            return loss_array.sum() * agent.dt
        
        # Create linearization function
        def linearize_loss(x_traj, u_traj, goal_pos, other_x_trajs, mask_values):
            dldx = jax.grad(runtime_loss, argnums=(0))
            dldu = jax.grad(runtime_loss, argnums=(1))
            
            def grad_step(args):
                xt, ut, other_xts = args
                return dldx(xt, ut, goal_pos, other_xts, mask_values, ref_traj), dldu(xt, ut, goal_pos, other_xts, mask_values, ref_traj)
            
            grads = jax.vmap(grad_step)((x_traj, u_traj, other_x_trajs))
            return grads[0], grads[1]  # a_traj, b_traj
        
        # Compile functions
        compiled_loss = jax.jit(trajectory_loss)
        compiled_linearize = jax.jit(linearize_loss)
        compiled_linearize_dyn = jax.jit(agent.linearize_dyn)
        compiled_solve = jax.jit(agent.solve)
        
        loss_functions.append(trajectory_loss)
        linearize_loss_functions.append(linearize_loss)
        compiled_functions.append({
            'loss': compiled_loss,
            'linearize_loss': compiled_linearize,
            'linearize_dyn': compiled_linearize_dyn,
            'solve': compiled_solve
        })
    
    return loss_functions, linearize_loss_functions, compiled_functions

def solve_masked_game_differentiable(agents: list, initial_states: list, target_positions: jnp.ndarray,
                                   mask_values: jnp.ndarray = None, num_iters: int = 10, 
                                   reference_trajectories: list = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve the masked game using a fully differentiable approach.
    
    This version ensures proper gradient flow through the entire game-solving pipeline
    by using JAX-compatible operations throughout.
    
    Args:
        agents: List of selected agents (used for configuration only)
        initial_states: List of initial states for selected agents
        target_positions: Target positions for selected agents
        mask_values: Mask values for collision avoidance
        num_iters: Number of optimization iterations
        
    Returns:
        Tuple of (state_trajectories, control_trajectories)
    """
    n_selected = len(agents)
    
    # Convert lists to JAX arrays for better performance and differentiability
    initial_states_array = jnp.stack([jnp.array(s) for s in initial_states])  # (n_selected, 4)
    
    # Create goal trajectories (goal position repeated for all time steps)
    # Each agent should try to reach its goal at every time step, not follow a linear path
    goal_trajectories = jnp.stack([
        jnp.tile(target_positions[i], (T_total, 1))
        for i in range(n_selected)
    ])  # (n_selected, T_total, 2)
    
    # Initialize control trajectories
    initial_controls = jnp.zeros((n_selected, T_total, 2))
    
    # Optimization parameters (following ilqgames_example.py pattern)
    step_size = config.optimization.step_size  # Conservative step size similar to original
    
    # Use reasonable number of iterations for convergence
    # training_iters = min(num_iters, config.optimization.num_iters)  # Reasonable iterations for stable convergence
    training_iters = config.optimization.num_iters
    
    # Agent configuration (cost matrices) - matching original ilqgames_example.py
    Q = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001]))  # State cost weights [x, y, vx, vy]
    R = jnp.diag(jnp.array([0.01, 0.01]))               # Control cost weights [ax, ay]
    
    def dynamics_function(x, u):
        """Point mass dynamics: [x, y, vx, vy] with controls [ax, ay]"""
        return jnp.array([
            x[2],  # dx/dt = vx
            x[3],  # dy/dt = vy  
            u[0],  # dvx/dt = ax
            u[1]   # dvy/dt = ay
        ])
    
    def integrate_dynamics(x0, u_traj):
        """Integrate dynamics forward using Euler integration"""
        def step_fn(x, u):
            x_next = x + dt * dynamics_function(x, u)
            return x_next, x_next
        
        _, x_traj = jax.lax.scan(step_fn, x0, u_traj)
        return x_traj
    
    def linearize_dynamics_at_trajectory(x0, u_traj):
        """Linearize dynamics around a trajectory"""
        x_traj = integrate_dynamics(x0, u_traj)
        
        # Compute Jacobians A = df/dx and B = df/du
        def compute_jacobians(x, u):
            A = jax.jacfwd(dynamics_function, argnums=0)(x, u)
            B = jax.jacfwd(dynamics_function, argnums=1)(x, u)
            return A, B
        
        A_traj, B_traj = jax.vmap(compute_jacobians)(x_traj, u_traj)
        return x_traj, A_traj, B_traj
    
    def compute_cost_gradients(agent_idx, x_traj, u_traj, goal_traj, all_x_trajs, mask_values, ref_traj=None):
        """Compute cost gradients for a single agent"""
        def single_step_cost(x, u, goal_x, other_xs, ref_x=None):
            # Navigation cost - use reference trajectory if available, otherwise go to goal
            if ref_x is not None:
                nav_cost = jnp.sum(jnp.square(x[:2] - ref_x[:2]))  # Follow reference trajectory
            else:
                nav_cost = jnp.sum(jnp.square(x[:2] - goal_x[:2]))  # Go to goal position
            
            # Collision avoidance cost with masking (following ilqgames_example.py pattern)
            collision_cost = 0.0
            if len(other_xs) > 0 and mask_values is not None:
                distances_sq = jnp.sum(jnp.square(x[:2] - other_xs[:, :2]), axis=1)
                # Use collision penalty similar to ilqgames_example.py
                collision_weight = config.optimization.collision_weight
                collision_scale = config.optimization.collision_scale
                base_collision = collision_weight * jnp.exp(-collision_scale * distances_sq)  # Using config values
                
                if agent_idx == 0:  # Ego agent uses mask
                    # Apply mask values to collision costs - this is the key differentiable part
                    num_others = len(other_xs)
                    if num_others > 0:
                        mask_vals_truncated = mask_values[:num_others]
                        masked_collision = base_collision * mask_vals_truncated
                        collision_cost = jnp.sum(masked_collision)
                else:  # Other agents: full interaction
                    collision_cost = jnp.sum(base_collision)
            
            # Control cost (following ilqgames_example.py pattern)
            ctrl_cost = config.optimization.control_weight * jnp.sum(jnp.square(u))
            
            return nav_cost + collision_cost + ctrl_cost
        
        # Get other agents' trajectories (excluding current agent)
        other_indices = jnp.array([i for i in range(n_selected) if i != agent_idx])
        
        if len(other_indices) > 0:
            other_x_trajs = all_x_trajs[other_indices]
        else:
            other_x_trajs = jnp.zeros((0, T_total, 4))
        
        # Compute gradients w.r.t. state and control
        if len(other_indices) > 0:
            other_x_transposed = other_x_trajs.transpose(1, 0, 2)
        else:
            other_x_transposed = jnp.zeros((T_total, 0, 4))
            
        # Create reference trajectory array for this agent
        if ref_traj is not None:
            ref_traj_array = ref_traj[:T_total]  # Ensure it matches T_total length
            if ref_traj_array.shape[0] < T_total:
                # Pad with last state if trajectory is too short
                last_state = ref_traj_array[-1] if ref_traj_array.shape[0] > 0 else jnp.zeros(4)
                padding = jnp.tile(last_state, (T_total - ref_traj_array.shape[0], 1))
                ref_traj_array = jnp.concatenate([ref_traj_array, padding], axis=0)
        else:
            ref_traj_array = jnp.zeros((T_total, 4))
        
        a_traj = jax.vmap(jax.grad(single_step_cost, argnums=0))(
            x_traj, u_traj, goal_traj, other_x_transposed, ref_traj_array)
        b_traj = jax.vmap(jax.grad(single_step_cost, argnums=1))(
            x_traj, u_traj, goal_traj, other_x_transposed, ref_traj_array)
        
        return a_traj, b_traj
    
    def solve_lqr_subproblem(A_traj, B_traj, a_traj, b_traj, agent):
        """Solve LQR subproblem using proper iLQR solve method like ilqgames_example.py"""
        # Use the agent's built-in solve method which implements the Riccati equation
        v_traj, z_traj = agent.solve(A_traj, B_traj, a_traj, b_traj)
        return v_traj
    
    def optimization_step(carry, _):
        """Single optimization step - fully differentiable"""
        control_trajectories = carry
        
        # Step 1: Linearize dynamics for all agents
        x_trajs = []
        A_trajs = []
        B_trajs = []
        
        for i in range(n_selected):
            x_traj, A_traj, B_traj = linearize_dynamics_at_trajectory(
                initial_states_array[i], control_trajectories[i])
            x_trajs.append(x_traj)
            A_trajs.append(A_traj)
            B_trajs.append(B_traj)
        
        x_trajs = jnp.stack(x_trajs)  # (n_selected, T_total, 4)
        A_trajs = jnp.stack(A_trajs)  # (n_selected, T_total, 4, 4)
        B_trajs = jnp.stack(B_trajs)  # (n_selected, T_total, 4, 2)
        
        # Step 2: Compute cost gradients for all agents
        a_trajs = []
        b_trajs = []
        
        for i in range(n_selected):
            # Get reference trajectory for this agent if available
            ref_traj = None
            if reference_trajectories is not None and i < len(reference_trajectories):
                ref_traj = reference_trajectories[i]
            
            a_traj, b_traj = compute_cost_gradients(
                i, x_trajs[i], control_trajectories[i], 
                goal_trajectories[i], x_trajs, mask_values, ref_traj)
            a_trajs.append(a_traj)
            b_trajs.append(b_traj)
        
        a_trajs = jnp.stack(a_trajs)  # (n_selected, T_total, 4)
        b_trajs = jnp.stack(b_trajs)  # (n_selected, T_total, 2)
        
        # Step 3: Solve LQR subproblems for all agents (like ilqgames_example.py)
        control_updates = []
        for i in range(n_selected):
            v_traj = solve_lqr_subproblem(A_trajs[i], B_trajs[i], a_trajs[i], b_trajs[i], agents[i])
            control_updates.append(v_traj)
        
        control_updates = jnp.stack(control_updates)  # (n_selected, T_total, 2)
        
        # Step 4: Update control trajectories
        new_control_trajectories = control_trajectories + step_size * control_updates
        
        return new_control_trajectories, x_trajs
    
    # Use JAX scan for differentiable optimization
    final_carry, scan_outputs = jax.lax.scan(
        optimization_step, initial_controls, None, length=training_iters)
    
    # Get final results
    final_control_trajectories = final_carry  # Final control trajectories
    final_state_trajectories = scan_outputs[-1]  # Last state trajectories
    
    # Convert back to list format for compatibility
    final_state_list = [final_state_trajectories[i] for i in range(n_selected)]
    final_control_list = [final_control_trajectories[i] for i in range(n_selected)]
    
    return final_state_list, final_control_list


def solve_masked_game(agents: list, initial_states: list, target_positions: jnp.ndarray,
                     compiled_functions: list, mask_values: jnp.ndarray = None, num_iters: int = 10, 
                     reference_trajectories: list = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Wrapper function that calls the differentiable game solver.
    
    This maintains compatibility with the existing API while using the new
    fully differentiable implementation.
    
    Args:
        agents: List of agent objects
        initial_states: Initial states for each agent
        target_positions: Target positions for each agent
        compiled_functions: Compiled functions for loss computation
        mask_values: Mask values for agent selection
        num_iters: Number of iterations for solver
        reference_trajectories: Reference trajectories for navigation (optional)
    """
    return solve_masked_game_differentiable(agents, initial_states, target_positions, mask_values, num_iters, reference_trajectories)

def extract_ego_reference_trajectory(sample_data: Dict[str, Any], ego_agent_id: int) -> jnp.ndarray:
    """
    Extract reference trajectory for the ego agent.
    
    Args:
        sample_data: Reference trajectory sample
        ego_agent_id: ID of the ego agent
        
    Returns:
        Reference trajectory for ego agent (T_reference, state_dim)
    """
    ego_key = f"agent_{ego_agent_id}"
    ego_states = sample_data["trajectories"][ego_key]["states"]
    return jnp.array(ego_states)

def similarity_loss(pred_traj: jnp.ndarray, target_traj: jnp.ndarray) -> jnp.ndarray:
    """
    Similarity loss: compare predicted trajectory with FUTURE target trajectory.
    
    The PSN observes the first T_observation steps and predicts the trajectory
    for the remaining time. This loss compares the predicted trajectory with
    the actual future trajectory (from T_observation onwards).
    
    Args:
        pred_traj: Predicted trajectory from masked game (T_total, state_dim)
        target_traj: Full reference trajectory (T_reference, state_dim)
        
    Returns:
        Similarity loss value comparing predicted vs future reference trajectory
    """
    # Extract positions (first 2 dimensions) for comparison
    pred_positions = pred_traj[:, :2]  # (T_total, 2)
    target_positions = target_traj[:, :2]  # (T_reference, 2)
    
    # Use future trajectory from observation horizon onwards
    # We want to compare with the trajectory AFTER the observation period
    future_target_positions = target_positions[T_observation:]  # (T_reference - T_observation, 2)
    
    # Limit to the shorter of the two trajectories
    min_length = min(pred_positions.shape[0], future_target_positions.shape[0])
    
    if min_length == 0:
        return jnp.array(100.0)  # High loss if no valid comparison possible
    
    # Use only the first min_length steps for comparison
    pred_positions_matched = pred_positions[:min_length]  # (min_length, 2)
    target_positions_matched = future_target_positions[:min_length]  # (min_length, 2)
    
    # Compute position-wise distance
    position_diff = pred_positions_matched - target_positions_matched
    distances = jnp.linalg.norm(position_diff, axis=-1)  # (min_length,)
    
    # Return mean distance
    return jnp.mean(distances)

def compute_similarity_loss_from_masked_game(sample_data: Dict[str, Any], ego_agent_id: int,
                                           predicted_mask: jnp.ndarray, predicted_goals: jnp.ndarray) -> jnp.ndarray:
    """
    Compute similarity loss by solving masked game and comparing with reference.
    
    Args:
        sample_data: Reference trajectory sample
        ego_agent_id: ID of the ego agent
        predicted_mask: Predicted mask from PSN (N_agents - 1)
        predicted_goals: Predicted goals from pretrained network (N_agents * 2)
        
    Returns:
        Similarity loss value
    """
    print(f"        Creating masked game setup...")
    # Create masked game setup
    agents, initial_states, target_positions, mask_values = create_masked_game_setup(
        sample_data, ego_agent_id, predicted_mask, predicted_goals, is_training=True)
    
    print(f"        Creating loss functions...")
    # Create loss functions with mask values
    loss_functions, linearize_functions, compiled_functions = create_loss_functions(
        agents, mask_values, is_training=True)
    
    print(f"        Solving masked game with {num_iters} iterations...")
    # Solve masked game
    state_trajectories, control_trajectories = solve_masked_game(
        agents, initial_states, target_positions, compiled_functions, mask_values, num_iters=num_iters)
    
    print(f"        Game solved. Extracting trajectories...")
    # Extract ego agent trajectory from masked game
    ego_traj_masked = state_trajectories[0]  # Ego agent is always first
    
    # Extract reference trajectory for ego agent
    ego_traj_ref = extract_ego_reference_trajectory(sample_data, ego_agent_id)
    
    print(f"        Computing similarity loss between predicted and reference trajectories...")
    # Compute similarity loss
    similarity_val = similarity_loss(ego_traj_masked, ego_traj_ref)
    print(f"        Similarity loss computed: {float(similarity_val):.6f}")
    
    return similarity_val

def _observations_to_initial_states(obs_row: jnp.ndarray) -> jnp.ndarray:
    """Convert a flattened observations row into initial 4D states for each agent.
    obs_row: shape (T_observation * N_agents * state_dim,)
    returns: (N_agents, 4)
    """
    traj = obs_row.reshape(T_observation, N_agents, state_dim)
    first = traj[0]  # (N_agents, state_dim)
    # Use position and velocity from the first observed state
    pos = first[:, :2]
    vel = first[:, 2:4]
    return jnp.concatenate([pos, vel], axis=-1)

def compute_similarity_loss_from_arrays(agents: list,
                                        initial_states: jnp.ndarray,
                                        predicted_goals_row: jnp.ndarray,
                                        predicted_mask_row: jnp.ndarray,
                                        ref_ego_traj: jnp.ndarray) -> jnp.ndarray:
    """Fully-JAX similarity loss using arrays only (no Python dict access).
    - initial_states: (N_agents, 4)
    - predicted_goals_row: (N_agents * 2,) or (N_agents, 2)
    - predicted_mask_row: (N_agents - 1,)
    - ref_ego_traj: (T_reference, state_dim)
    """
    targets = predicted_goals_row.reshape(N_agents, 2)
    mask_values = predicted_mask_row

    # Use the new differentiable game solver directly
    state_trajectories, _ = solve_masked_game_differentiable(
        agents,
        [initial_states[i] for i in range(N_agents)],
        targets,
        mask_values,
        num_iters=num_iters,
    )

    ego_traj_masked = state_trajectories[0]
    return similarity_loss(ego_traj_masked, ref_ego_traj)

def compute_batch_similarity_loss(predicted_masks: jnp.ndarray,
                                  predicted_goals: jnp.ndarray,
                                  batch_data: List[Dict[str, Any]],
                                  ego_agent_id: int = 0,
                                  similarity_pbar: tqdm = None) -> jnp.ndarray:
    """
    Compute similarity loss for a batch of samples by solving masked games.
    
    Args:
        predicted_masks: Predicted masks from PSN (batch_size, N_agents - 1)
        predicted_goals: Predicted goals from pretrained network (batch_size, N_agents * 2)
        batch_data: List of reference trajectory samples
        ego_agent_id: ID of the ego agent
        similarity_pbar: Optional progress bar for similarity loss computation
        
    Returns:
        Average similarity loss for the batch
    """
    # Build shared agents once (purely static configuration) - matching original ilqgames_example.py
    Q = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001]))  # State cost weights [x, y, vx, vy]
    R = jnp.diag(jnp.array([0.01, 0.01]))               # Control cost weights [ax, ay]
    shared_agents = [PointAgent(dt=dt, x_dim=4, u_dim=2, Q=Q, R=R) for _ in range(N_agents)]

    # To force a JAX path, construct needed arrays from batch_data using helper
    # Extract observations and ref trajectories via existing helpers
    observations, _, ref_trajs = prepare_batch_for_training(batch_data)

    def per_sample(i):
        mask = predicted_masks[i]
        goals = predicted_goals[i]
        obs_row = observations[i]
        ref_ego = ref_trajs[i]
        init_states = _observations_to_initial_states(obs_row)
        return compute_similarity_loss_from_arrays(shared_agents, init_states, goals, mask, ref_ego)

    valid_bs = min(predicted_masks.shape[0], observations.shape[0])
    losses = jax.vmap(per_sample)(jnp.arange(valid_bs))
    return jnp.mean(losses)

def prepare_batch_for_training(batch_data: List[Dict[str, Any]]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Prepare batch data for training.
    
    Args:
        batch_data: List of reference trajectory samples
        
    Returns:
        observations: Batch of observations (batch_size, T_observation * N_agents * state_dim)
        masks: Batch of masks (batch_size, N_agents - 1)
        reference_trajectories: Batch of reference trajectories (batch_size, T_reference, state_dim)
    """
    batch_obs = []
    batch_masks = []
    batch_ref_traj = []
    
    for sample_data in batch_data:
        ego_agent_id = 0
        
        # Extract observation trajectory
        obs_traj = extract_observation_trajectory(sample_data)
        batch_obs.append(obs_traj.flatten())
        
        # Extract ego reference trajectory
        ego_ref_traj = extract_ego_reference_trajectory(sample_data, ego_agent_id)
        if ego_ref_traj.shape[0] != T_reference:
            if ego_ref_traj.shape[0] < T_reference:
                pad_size = T_reference - ego_ref_traj.shape[0]
                last_state = ego_ref_traj[-1:]
                padding = jnp.tile(last_state, (pad_size, 1))
                ego_ref_traj = jnp.concatenate([ego_ref_traj, padding], axis=0)
            else:
                ego_ref_traj = ego_ref_traj[:T_reference]
        batch_ref_traj.append(ego_ref_traj)
        
        # For validation, we don't need masks since PSN will predict them
        # For training, this will be replaced by PSN prediction anyway
        # So we can just use zeros as placeholders
        placeholder_mask = jnp.zeros((N_agents - 1,))
        batch_masks.append(placeholder_mask)
    
    # Pad batch if necessary
    if len(batch_obs) < batch_size:
        pad_size = batch_size - len(batch_obs)
        obs_pad = jnp.zeros((pad_size, T_observation * N_agents * state_dim))
        batch_obs.extend([obs_pad[i] for i in range(pad_size)])
        mask_pad = jnp.zeros((pad_size, N_agents - 1))
        batch_masks.extend([mask_pad[i] for i in range(pad_size)])
        ref_pad = jnp.zeros((pad_size, T_reference, state_dim))
        batch_ref_traj.extend([ref_pad[i] for i in range(pad_size)])
    
    # Convert to JAX arrays
    batch_obs = jnp.stack(batch_obs)
    batch_masks = jnp.stack(batch_masks)
    batch_ref_traj = jnp.stack(batch_ref_traj)
    
    return batch_obs, batch_masks, batch_ref_traj

def prepare_batch_for_training_with_progress(batch_data: List[Dict[str, Any]], sample_pbar: tqdm) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Prepare batch data for training with a progress bar for samples.
    
    Args:
        batch_data: List of reference trajectory samples
        sample_pbar: tqdm progress bar for samples
        
    Returns:
        observations: Batch of observations (batch_size, T_observation * N_agents * state_dim)
        masks: Batch of masks (batch_size, N_agents - 1)
        reference_trajectories: Batch of reference trajectories (batch_size, T_reference, state_dim)
    """
    batch_obs = []
    batch_masks = []
    batch_ref_traj = []
    
    for i, sample_data in enumerate(batch_data):
        ego_agent_id = 0
        
        # Extract observation trajectory
        obs_traj = extract_observation_trajectory(sample_data)
        batch_obs.append(obs_traj.flatten())
        
        # Extract ego reference trajectory
        ego_ref_traj = extract_ego_reference_trajectory(sample_data, ego_agent_id)
        if ego_ref_traj.shape[0] != T_reference:
            if ego_ref_traj.shape[0] < T_reference:
                pad_size = T_reference - ego_ref_traj.shape[0]
                last_state = ego_ref_traj[-1:]
                padding = jnp.tile(last_state, (pad_size, 1))
                ego_ref_traj = jnp.concatenate([ego_ref_traj, padding], axis=0)
            else:
                ego_ref_traj = ego_ref_traj[:T_reference]
        batch_ref_traj.append(ego_ref_traj)
        
        # For training, this will be replaced by PSN prediction anyway
        # So we can just use zeros as placeholders
        placeholder_mask = jnp.zeros((N_agents - 1,))
        batch_masks.append(placeholder_mask)
        
        # Update sample progress bar
        sample_pbar.set_postfix({'Sample': f'{i+1}/{len(batch_data)}'})
        sample_pbar.update(1)
    
    # Pad batch if necessary
    if len(batch_obs) < batch_size:
        pad_size = batch_size - len(batch_obs)
        obs_pad = jnp.zeros((pad_size, T_observation * N_agents * state_dim))
        batch_obs.extend([obs_pad[i] for i in range(pad_size)])
        mask_pad = jnp.zeros((pad_size, N_agents - 1))
        batch_masks.extend([mask_pad[i] for i in range(pad_size)])
        ref_pad = jnp.zeros((pad_size, T_reference, state_dim))
        batch_ref_traj.extend([ref_pad[i] for i in range(pad_size)])
    
    # Convert to JAX arrays
    batch_obs = jnp.stack(batch_obs)
    batch_masks = jnp.stack(batch_masks)
    batch_ref_traj = jnp.stack(batch_ref_traj)
    
    return batch_obs, batch_masks, batch_ref_traj

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def binary_loss(mask: jnp.ndarray) -> jnp.ndarray:
    """Binary loss: encourages mask values to be close to 0 or 1."""
    binary_penalty = mask * (1 - mask)
    return jnp.mean(binary_penalty)

def mask_sparsity_loss(mask: jnp.ndarray) -> jnp.ndarray:
    """Mask sparsity loss: encourages fewer agents to be selected."""
    return jnp.mean(mask)

def total_loss(mask: jnp.ndarray, binary_loss_val: jnp.ndarray, 
               sparsity_loss_val: jnp.ndarray, similarity_loss_val: jnp.ndarray,
               sigma1: float = 0.1, sigma2: float = 1.0) -> jnp.ndarray:
    """Total loss combining all components."""
    return similarity_loss_val + sigma1 * sparsity_loss_val + sigma2 * binary_loss_val

# ============================================================================
# GOAL INFERENCE INTEGRATION
# ============================================================================

def load_pretrained_goal_model(model_path: str) -> Tuple[GoalInferenceNetwork, train_state.TrainState]:
    """Load the pretrained goal inference model."""
    print(f"Loading pretrained goal inference model from: {model_path}")
    
    # Create model instance
    goal_model = GoalInferenceNetwork(hidden_dims=goal_inference_hidden_dims)
    
    # Load trained parameters
    with open(model_path, 'rb') as f:
        model_bytes = pickle.load(f)
    
    # Recreate train state
    optimizer = optax.adam(config.goal_inference.learning_rate)  # Dummy optimizer for inference
    input_shape = (1, T_observation * N_agents * state_dim)
    dummy_state = create_train_state(goal_model, optimizer, input_shape, jax.random.PRNGKey(config.training.seed))
    
    # Load parameters
    trained_state = dummy_state.replace(params=flax.serialization.from_bytes(dummy_state, model_bytes).params)
    
    print("Pretrained goal inference model loaded successfully")
    return goal_model, trained_state

def predict_goals_with_pretrained_model(goal_model: GoalInferenceNetwork, 
                                       trained_state: train_state.TrainState,
                                       observations: jnp.ndarray, rng: jnp.ndarray = None) -> jnp.ndarray:
    """Use pretrained goal inference model to predict goals."""
    if rng is not None:
        # Training mode: use dropout with random key
        return trained_state.apply_fn({'params': trained_state.params}, observations, rngs={'dropout': rng}, deterministic=False)
    else:
        # Evaluation mode: deterministic (no dropout)
        return trained_state.apply_fn({'params': trained_state.params}, observations, deterministic=True)

# ============================================================================
# REFERENCE TRAJECTORY LOADING
# ============================================================================

def load_reference_trajectories(data_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load reference trajectories from directory containing individual JSON files and split into training and validation sets."""
    import glob
    
    # Find all ref_traj_sample_*.json files in the directory
    pattern = os.path.join(data_dir, "ref_traj_sample_*.json")
    json_files = sorted(glob.glob(pattern))
    
    if not json_files:
        raise FileNotFoundError(f"No ref_traj_sample_*.json files found in directory: {data_dir}")
    
    # Load all samples
    reference_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                sample_data = json.load(f)
                reference_data.append(sample_data)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue
    
    print(f"Loaded {len(reference_data)} reference trajectory samples from {data_dir}")
    print(f"Sample files: {len(json_files)} found, {len(reference_data)} loaded successfully")
    
    # Split data: 75% training, 25% validation
    total_samples = len(reference_data)
    train_size = int(0.75 * total_samples)
    val_size = total_samples - train_size
    
    # Shuffle data for random split
    import random
    random.seed(config.training.seed)  # For reproducibility
    shuffled_data = reference_data.copy()
    random.shuffle(shuffled_data)
    
    # Split into training and validation
    training_data = shuffled_data[:train_size]
    validation_data = shuffled_data[train_size:]
    
    train_percentage = int((1 - 0.25) * 100)  # 75% for training
    val_percentage = int(0.25 * 100)  # 25% for validation
    print(f"Training samples: {len(training_data)} ({train_percentage}%)")
    print(f"Validation samples: {len(validation_data)} ({val_percentage}%)")
    
    return training_data, validation_data

# ============================================================================
# MASKED GAME FUNCTIONS
# ============================================================================

# The create_masked_game_setup, create_loss_functions, solve_masked_game,
# extract_ego_reference_trajectory, similarity_loss, compute_similarity_loss_from_masked_game,
# and compute_batch_similarity_loss functions are now defined above.

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_train_state(model: nn.Module, optimizer: optax.GradientTransformation, 
                      input_shape: Tuple[int, ...], rng: jnp.ndarray) -> train_state.TrainState:
    """Create training state for the model."""
    dummy_input = jnp.ones(input_shape)
    variables = model.init(rng, dummy_input)
    params = variables['params']
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    return state

def train_step(state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
               batch_data: List[Dict[str, Any]], goal_model: GoalInferenceNetwork,
               goal_trained_state: train_state.TrainState, sigma1: float, sigma2: float,
               rng: jnp.ndarray = None, similarity_pbar: tqdm = None) -> Tuple[train_state.TrainState, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Single training step with game-solving loss function.
    
    Uses differentiable iLQR-based game solver for proper gradient flow.
    """
    observations, masks, reference_trajectories = batch
    
    def loss_fn(params):
        # Get predicted mask from PSN
        predicted_masks = state.apply_fn({'params': params}, observations)
        
        # Get predicted goals from pretrained goal inference network
        predicted_goals = predict_goals_with_pretrained_model(goal_model, goal_trained_state, observations, rng)
        
        # 1. Binary loss: encourages mask values to be close to 0 or 1
        binary_loss_val = binary_loss(predicted_masks)
        
        # 2. Sparsity loss: encourages fewer agents to be selected
        sparsity_loss_val = mask_sparsity_loss(predicted_masks)
        
        # 3. Game-solving similarity loss: solve masked game and compute trajectory similarity
        similarity_loss_val = compute_batch_similarity_loss(predicted_masks, predicted_goals, batch_data, similarity_pbar=similarity_pbar)
        
        # Total loss combining all components
        total_loss_val = total_loss(predicted_masks, binary_loss_val, 
                                  sparsity_loss_val, similarity_loss_val,
                                  sigma1, sigma2)
        
        return total_loss_val, (binary_loss_val, sparsity_loss_val, similarity_loss_val)
    
    # Compute gradients using JAX automatic differentiation
    (loss, loss_components), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Apply gradient clipping to prevent gradient explosion
    grad_norm = jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(jnp.square(y)), grads, initializer=0.0)
    grad_norm = jnp.sqrt(grad_norm)
    
    max_grad_norm = config.debug.gradient_clip_value
    if grad_norm > max_grad_norm:
        scale = max_grad_norm / grad_norm
        grads = jax.tree_map(lambda g: g * scale, grads)
    
    # Apply gradients using the optimizer
    state = state.apply_gradients(grads=grads)
    
    return state, loss, loss_components

def validation_step(state: train_state.TrainState, validation_data: List[Dict[str, Any]],
                   goal_model: GoalInferenceNetwork, goal_trained_state: train_state.TrainState,
                   batch_size: int = 32, ego_agent_id: int = 0) -> Tuple[float, float, float, List[jnp.ndarray]]:
    """
    Perform validation on the validation dataset.
    
    Args:
        state: Current train state
        validation_data: Validation dataset
        goal_model: Pretrained goal inference network
        goal_trained_state: Trained state of goal inference network
        batch_size: Batch size for validation
        ego_agent_id: ID of the ego agent
        
    Returns:
        Tuple of (average validation loss, average sparsity loss, average similarity loss, list of predicted masks)
    """
    val_losses = []
    val_sparsity_losses = []
    val_similarity_losses = []
    all_predicted_masks = []
    
    # Create validation batches
    num_val_batches = (len(validation_data) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_val_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(validation_data))
        batch_data = validation_data[start_idx:end_idx]
        
        # Prepare batch for validation
        observations, masks, reference_trajectories = prepare_batch_for_training(batch_data)
        
        # Get predicted mask from PSN (no gradients needed for validation)
        predicted_masks = state.apply_fn({'params': state.params}, observations)
        
        # Store masks for analysis
        all_predicted_masks.append(predicted_masks)
        
        # Get predicted goals from pretrained goal inference network
        predicted_goals = predict_goals_with_pretrained_model(goal_model, goal_trained_state, observations)
        
        # Compute validation loss components
        binary_loss_val = binary_loss(predicted_masks)
        sparsity_loss_val = mask_sparsity_loss(predicted_masks)
        
        # Compute similarity loss using game solving
        similarity_loss_val = compute_batch_similarity_loss(predicted_masks, predicted_goals, batch_data, ego_agent_id)
        
        # Total validation loss
        total_loss_val = total_loss(predicted_masks, binary_loss_val, 
                                  sparsity_loss_val, similarity_loss_val,
                                  sigma1, sigma2)  # No regularization during validation

        val_losses.append(float(total_loss_val))
        val_sparsity_losses.append(float(sparsity_loss_val))
        val_similarity_losses.append(float(similarity_loss_val))
    
    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_val_sparsity = sum(val_sparsity_losses) / len(val_sparsity_losses)
    avg_val_similarity = sum(val_similarity_losses) / len(val_similarity_losses)

    return avg_val_loss, avg_val_sparsity, avg_val_similarity, all_predicted_masks

# ============================================================================
# DEBUG FUNCTIONS
# ============================================================================

def test_gradient_flow(state: train_state.TrainState, observations: jnp.ndarray, 
                      goal_model: GoalInferenceNetwork, goal_trained_state: train_state.TrainState,
                      reference_trajectories: jnp.ndarray) -> bool:
    """
    Test if gradients are flowing correctly through the game-solving loss function.
    
    Args:
        state: Current train state
        observations: Input observations
        goal_model: Goal inference model
        goal_trained_state: Trained goal inference state
        reference_trajectories: Reference trajectories for loss computation
        
    Returns:
        True if gradients are non-zero, False otherwise
    """
    print("\n" + "="*60)
    print("TESTING GRADIENT FLOW WITH DIFFERENTIABLE GAME SOLVER")
    print("="*60)
    
    def test_loss_fn(params):
        # Get predicted mask from PSN
        predicted_masks = state.apply_fn({'params': params}, observations)
        
        # Get predicted goals from pretrained goal inference network
        predicted_goals = predict_goals_with_pretrained_model(goal_model, goal_trained_state, observations)
        
        # Test the actual game-solving loss components
        binary_loss_val = binary_loss(predicted_masks)
        sparsity_loss_val = mask_sparsity_loss(predicted_masks)
        
        # Create a complete test batch_data for game solving
        # This will test if the new differentiable game-solving pipeline produces gradients
        test_trajectories = {}
        for agent_id in range(N_agents):
            agent_key = f"agent_{agent_id}"
            # Create dummy trajectory data for testing
            dummy_states = jnp.zeros((T_reference, state_dim))
            dummy_states = dummy_states.at[:, :2].set(reference_trajectories[0, :T_reference, :2])  # Use reference positions
            test_trajectories[agent_key] = {"states": dummy_states}
        
        test_batch_data = [{"trajectories": test_trajectories}]
        
        # Test game-solving similarity loss using the new differentiable approach
        similarity_loss_val = compute_batch_similarity_loss(predicted_masks, predicted_goals, test_batch_data)
        
        total_loss_val = binary_loss_val + sparsity_loss_val + similarity_loss_val
        
        return total_loss_val, (binary_loss_val, sparsity_loss_val, similarity_loss_val)
    
    # Compute gradients
    (loss, loss_components), grads = jax.value_and_grad(test_loss_fn, has_aux=True)(state.params)
    
    # Check gradient norm
    grad_norm = jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(jnp.square(y)), grads, initializer=0.0)
    grad_norm = jnp.sqrt(grad_norm)
    
    print(f"Test loss: {float(loss):.6f}")
    print(f"Loss components:")
    print(f"  Binary: {float(loss_components[0]):.6f}")
    print(f"  Sparsity: {float(loss_components[1]):.6f}")
    print(f"  Game-Solving Similarity: {float(loss_components[2]):.6f}")
    print(f"Gradient norm: {float(grad_norm):.8f}")
    
    if grad_norm > 1e-8:
        print("✓ GRADIENTS ARE FLOWING CORRECTLY WITH DIFFERENTIABLE GAME SOLVER!")
        print("✓ The new implementation should enable proper backpropagation through masks.")
        return True
    else:
        print("✗ GRADIENTS ARE STILL ZERO - NEED FURTHER DEBUGGING!")
        print("The issue might be in the differentiable game solver implementation.")
        return False

# ============================================================================
# ENHANCED TRAINING FUNCTIONS
# ============================================================================

def train_psn_with_pretrained_goals(model: nn.Module, training_data: List[Dict[str, Any]], 
                                   validation_data: List[Dict[str, Any]], goal_model: GoalInferenceNetwork, 
                                   goal_trained_state: train_state.TrainState,
                                   num_epochs: int = 30, learning_rate: float = 1e-3,
                                   sigma1: float = 0.1, sigma2: float = 0.0, batch_size: int = 32, 
                                   rng: jnp.ndarray = None) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], train_state.TrainState, str, float, int]:
    """Train PSN using pretrained goal inference network with validation."""
    if rng is None:
        rng = jax.random.PRNGKey(config.training.seed)
    
    # Create log directory and TensorBoard writer
    # Extract goal inference model name from the path
    goal_model_dir = os.path.dirname(pretrained_goal_model_path)
    goal_model_name = os.path.basename(goal_model_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = f"psn_pretrained_goals_N_{N_agents}_T_{T_total}_obs_{T_observation}_lr_{learning_rate}_bs_{batch_size}_sigma1_{sigma1}_sigma2_{sigma2}_epochs_{num_epochs}"
    
    # Place PSN logs under the goal inference model directory
    log_dir = os.path.join(goal_model_dir, config_name)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"PSN training logs will be saved under goal inference model directory: {goal_model_dir}")
    print(f"PSN specific log directory: {log_dir}")
    
    # Initialize TensorBoard writer
    writer = tb.SummaryWriter(log_dir)
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    
    # Create train state
    input_shape = (batch_size, T_observation * N_agents * state_dim)
    state = create_train_state(model, optimizer, input_shape, rng)
    
    # Test gradient flow before training starts
    print("\nTesting gradient flow before training...")
    test_observations = jnp.ones((1, T_observation * N_agents * state_dim))  # Dummy observations
    test_references = jnp.ones((1, T_reference, state_dim))  # Dummy references
    gradient_test_passed = test_gradient_flow(state, test_observations, goal_model, goal_trained_state, test_references)
    
    if not gradient_test_passed:
        print("WARNING: Gradient test failed! Training may not work properly.")
        print("Continuing anyway to see what happens...")
    else:
        print("Gradient test passed! Training should work correctly.")
    
    training_losses = []
    validation_losses = []
    # Track individual loss components over epochs
    sparsity_losses = []
    similarity_losses = []
    validation_sparsity_losses = []
    validation_similarity_losses = []
    best_loss = float('inf')
    best_state = None
    best_epoch = 0
    
    # Main training loop
    print(f"Starting PSN training with pretrained goals...")
    print(f"Goal inference model: {pretrained_goal_model_path}")
    print(f"Training parameters: epochs={num_epochs}, lr={learning_rate}, batch_size={batch_size}")
    print(f"Loss weights: σ1={sigma1}, σ2={sigma2}")
    print(f"Training data: {len(training_data)} samples")
    print(f"Validation data: {len(validation_data)} samples")
    print(f"Device: {jax.devices()[0]}")
    print("-" * 80)
    
    # Main training progress bar
    total_steps = num_epochs * ((len(training_data) + batch_size - 1) // batch_size)
    training_pbar = tqdm(total=total_steps, desc="Training Progress", position=0)
    
    for epoch in range(num_epochs):
        # Calculate current sigma1 value (linear schedule from 0 to final value)
        # current_sigma1 = (epoch / (num_epochs - 1)) * sigma1 if num_epochs > 1 else sigma1
        current_sigma1 = sigma1
        
        epoch_losses = []
        epoch_sparsity_losses = []
        epoch_similarity_losses = []
        # Create batches
        num_batches = (len(training_data) + batch_size - 1) // batch_size
        
        # Progress bar for batches within each epoch
        batch_pbar = tqdm(range(num_batches), 
                         desc=f"Epoch {epoch+1}/{num_epochs} - Batches", 
                   position=1, leave=False)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(training_data))
            batch_data = training_data[start_idx:end_idx]
            
            # Progress bar for samples within each batch (for data preparation)
            sample_pbar = tqdm(range(len(batch_data)), 
                              desc=f"Batch {batch_idx+1}/{num_batches} - Preparing data", 
                              position=2, leave=False)
            
            # Prepare batch for training with progress bar
            observations, masks, reference_trajectories = prepare_batch_for_training_with_progress(batch_data, sample_pbar)
            
            # Close sample progress bar after preparation
            sample_pbar.close()
            
            # Progress bar for similarity loss computation (game solving)
            similarity_pbar = tqdm(range(len(batch_data)), 
                                 desc=f"Batch {batch_idx+1}/{num_batches} - Game solving", 
                                 position=2, leave=False)
            
            # Split random key for this step
            rng, step_key = jax.random.split(rng)
            
            # Training step with game solving
            state, loss, (binary_loss_val, sparsity_loss_val, similarity_loss_val) = train_step(
                state, (observations, masks, reference_trajectories), batch_data, 
                goal_model, goal_trained_state, current_sigma1, sigma2, step_key, similarity_pbar)
            
            epoch_losses.append(float(loss))
            epoch_sparsity_losses.append(float(sparsity_loss_val))
            epoch_similarity_losses.append(float(similarity_loss_val))
            
            # Close similarity progress bar
            similarity_pbar.close()
            
            # Update batch progress bar
            batch_pbar.set_postfix({
                'Loss': f'{float(loss):.4f}',
                'σ1': f'{current_sigma1:.3f}',
                'Binary': f'{float(binary_loss_val):.4f}',
                'Sparsity': f'{float(sparsity_loss_val):.4f}',
                'Similarity': f'{float(similarity_loss_val):.4f}'
            })
            batch_pbar.update(1)
            
            # Update main training progress bar
            training_pbar.set_postfix({
                'Epoch': f'{epoch+1}/{num_epochs}',
                'Batch': f'{batch_idx+1}/{num_batches}',
                'Loss': f'{float(loss):.4f}',
                'σ1': f'{current_sigma1:.3f}'
            })
            training_pbar.update(1)
        # Close batch progress bar
        batch_pbar.close()
        
        # Perform validation
        val_loss, val_sparsity_loss, val_similarity_loss, val_masks = validation_step(
            state, validation_data, goal_model, goal_trained_state, batch_size)
        validation_losses.append(val_loss)
        validation_sparsity_losses.append(val_sparsity_loss)
        validation_similarity_losses.append(val_similarity_loss)
        
        # Calculate average loss for the epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_sparsity_loss = sum(epoch_sparsity_losses) / len(epoch_sparsity_losses)
        avg_similarity_loss = sum(epoch_similarity_losses) / len(epoch_similarity_losses)
        training_losses.append(avg_loss)
        sparsity_losses.append(avg_sparsity_loss)
        similarity_losses.append(avg_similarity_loss)
        
        # Track best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            
            # Save best model using proper JAX/Flax serialization
            best_model_path = os.path.join(log_dir, "psn_best_model.pkl")
            model_bytes = flax.serialization.to_bytes(state)
            with open(best_model_path, 'wb') as f:
                pickle.dump(model_bytes, f)
            print(f"\nNew best model found at epoch {epoch + 1} with validation loss: {best_loss:.4f} (σ1: {current_sigma1:.3f})")
            print(f"Best model saved to: {best_model_path}")
        
        # Update main progress bar
        training_pbar.set_postfix({
            'Epoch': f'{epoch+1}/{num_epochs}',
            'Train Loss': f'{avg_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'σ1': f'{current_sigma1:.3f}'
        })
        
        # Log epoch-level metrics to TensorBoard
        writer.add_scalar('Loss/Epoch/Training', avg_loss, epoch)
        writer.add_scalar('Loss/Epoch/Validation', val_loss, epoch)
        writer.add_scalar('Loss/Epoch/Best', best_loss, epoch)
        writer.add_scalar('Loss/Epoch/Sparsity', avg_sparsity_loss, epoch)
        writer.add_scalar('Loss/Epoch/Similarity', avg_similarity_loss, epoch)
        writer.add_scalar('Loss/Epoch/Validation_Sparsity', val_sparsity_loss, epoch)
        writer.add_scalar('Loss/Epoch/Validation_Similarity', val_similarity_loss, epoch)
        writer.add_scalar('Hyperparameters/Sigma1', current_sigma1, epoch)
        writer.add_scalar('Training/EpochProgress', (epoch + 1) / num_epochs, epoch)
        
        # Training progress metrics
        writer.add_scalar('Progress/EpochProgress', (epoch + 1) / num_epochs, epoch)
        writer.add_scalar('Progress/LossImprovement', float(best_loss - val_loss), epoch)
        
        # Add text summary for hyperparameters
        if epoch == 0:
            writer.add_text('Hyperparameters', f"""
            - Number of agents: {N_agents}
            - Total game steps: {T_total}
            - Observation steps: {T_observation}
            - Learning rate: {learning_rate}
            - Batch size: {batch_size}
            - Sigma1 (final): {sigma1}
            - Sigma2: {sigma2}
            - Total epochs: {num_epochs}
            - Using pretrained goal inference: True
            - Goal inference model: {pretrained_goal_model_path}
            - Validation split: 25% (hardcoded for now)
            """, epoch)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val: {best_loss:.4f} (epoch {best_epoch+1}), σ1: {current_sigma1:.3f}")
        
        # Clean up memory after each epoch
        gc.collect()
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
    
    # Close progress bar
    training_pbar.close()
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"\nTraining completed! Best model found at epoch {best_epoch + 1} with loss: {best_loss:.4f}")
    
    return training_losses, validation_losses, sparsity_losses, similarity_losses, validation_sparsity_losses, validation_similarity_losses, state, log_dir, best_loss, best_epoch

# ============================================================================
# MODEL LOADING UTILITIES
# ============================================================================

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
    psn_model = PlayerSelectionNetwork(hidden_dims=psn_hidden_dims)
    
    # Deserialize the PSN state
    psn_trained_state = flax.serialization.from_bytes(psn_model, psn_model_bytes)
    print("✓ PSN model loaded successfully")
    
    print(f"Loading trained goal inference model from: {goal_model_path}")
    
    # Load the goal inference model bytes
    with open(goal_model_path, 'rb') as f:
        goal_model_bytes = pickle.load(f)
    
    # Create the goal inference model
    goal_model = GoalInferenceNetwork(hidden_dims=goal_inference_hidden_dims)
    
    # Deserialize the goal inference state
    goal_trained_state = flax.serialization.from_bytes(goal_model, goal_model_bytes)
    print("✓ Goal inference model loaded successfully")
    
    return psn_model, psn_trained_state, goal_model, goal_trained_state

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PSN Training with Pretrained Goal Inference Network")
    print("=" * 80)

    # Load reference trajectories
    print(f"Loading reference trajectories from directory: {reference_dir}")
    training_data, validation_data = load_reference_trajectories(reference_dir)
    
    # Load pretrained goal inference model
    goal_model, goal_trained_state = load_pretrained_goal_model(pretrained_goal_model_path)
    
    # Create PSN model
    psn_model = PlayerSelectionNetwork(hidden_dims=psn_hidden_dims)
    
    # Train PSN with pretrained goals
    training_losses, validation_losses, sparsity_losses, similarity_losses, validation_sparsity_losses, validation_similarity_losses, trained_state, log_dir, best_loss, best_epoch = train_psn_with_pretrained_goals(
        psn_model, training_data, validation_data, goal_model, goal_trained_state,
        num_epochs=num_epochs, learning_rate=learning_rate,
        sigma1=sigma1, sigma2=sigma2, batch_size=batch_size
    )
    
    # Save final model
    final_model_path = os.path.join(log_dir, "psn_final_model.pkl")
    final_model_bytes = flax.serialization.to_bytes(trained_state)
    with open(final_model_path, 'wb') as f:
        pickle.dump(final_model_bytes, f)
    
    # Save training configuration
    training_config = {
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'sigma1': sigma1,
        'sigma2': sigma2,
        'N_agents': N_agents,
        'T_total': T_total,
        'T_observation': T_observation,
        'state_dim': state_dim,
        'control_dim': control_dim,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'goal_inference_model_path': pretrained_goal_model_path,
        'goal_inference_model_dir': os.path.dirname(pretrained_goal_model_path)
    }
    
    config_path = os.path.join(log_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    # Create a summary file showing the relationship
    summary_path = os.path.join(log_dir, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("PSN Training with Pretrained Goal Inference Network\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Goal Inference Model: {pretrained_goal_model_path}\n")
        f.write(f"Goal Inference Model Directory: {os.path.dirname(pretrained_goal_model_path)}\n")
        f.write(f"PSN Training Directory: {log_dir}\n\n")
        f.write(f"Training Parameters:\n")
        f.write(f"  - Epochs: {num_epochs}\n")
        f.write(f"  - Learning Rate: {learning_rate}\n")
        f.write(f"  - Batch Size: {batch_size}\n")
        f.write(f"  - Sigma1: {sigma1}\n")
        f.write(f"  - Sigma2: {sigma2}\n")
        f.write(f"  - N_agents: {N_agents}\n")
        f.write(f"  - T_total: {T_total}\n")
        f.write(f"  - T_observation: {T_observation}\n\n")
        f.write(f"Results:\n")
        f.write(f"  - Best Validation Loss: {best_loss:.6f}\n")
        f.write(f"  - Best Epoch: {best_epoch + 1}\n")
        f.write(f"  - Final Training Loss: {training_losses[-1]:.6f}\n")
        f.write(f"  - Final Validation Loss: {validation_losses[-1]:.6f}\n\n")
        f.write(f"Files:\n")
        f.write(f"  - Best Model: psn_best_model.pkl\n")
        f.write(f"  - Final Model: psn_final_model.pkl\n")
        f.write(f"  - Config: training_config.json\n")
        f.write(f"  - Summary: training_summary.txt\n")

        f.write(f"  - TensorBoard Logs: events.out.tfevents.*\n")
    
    # Print final results
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Goal inference model directory: {os.path.dirname(pretrained_goal_model_path)}")
    print(f"PSN log directory: {log_dir}")
    print(f"Best model saved to: {os.path.join(log_dir, 'psn_best_model.pkl')}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Config saved to: {config_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"\nFinal training loss: {training_losses[-1]:.4f}")
    print(f"Final validation loss: {validation_losses[-1]:.4f}")
    print(f"Best validation loss: {best_loss:.4f} (achieved at epoch {best_epoch + 1})")
    print(f"Training progress plot saved to: {os.path.join(log_dir, 'training_loss.png')}")
    print("\nTo view TensorBoard logs, run:")
    print(f"tensorboard --logdir={log_dir}")
    print(f"\nNote: PSN logs are organized under the goal inference model directory for better organization.")
    print(f"Directory structure: {os.path.dirname(pretrained_goal_model_path)}/ → {os.path.basename(log_dir)}/")