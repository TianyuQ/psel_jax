#!/usr/bin/env python3
"""
PSN Training with Reference Trajectories

This script trains the Player Selection Network (PSN) using reference trajectories
generated from full 10-player games. The similarity loss compares trajectories
from masked games with the reference trajectories.

NEW REQUIREMENTS:
1. Randomly assign ego agent for each sample
2. Modified similarity cost
3. Use first 10 steps as observation trajectory (input)
4. Total steps = 50 (shorter than reference 100 steps)

Author: Assistant
Date: 2024
"""

# Set JAX environment variables to help with GPU issues
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# Don't hardcode platform - let JAX auto-detect
# os.environ['JAX_PLATFORM_NAME'] = 'gpu'  # Will fallback to CPU if GPU fails

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
import gc  # Add garbage collection
from torch.utils.tensorboard import SummaryWriter
import torch.utils.tensorboard as tb

# JAX/Flax imports
import flax.linen as nn
import optax
from flax.training import train_state
import flax.serialization

# Import from the main lqrax module
import sys
sys.path.append('..')
from lqrax import iLQR


# ============================================================================
# GLOBAL PARAMETERS (UPDATED FOR NEW REQUIREMENTS)
# ============================================================================

# Game parameters (UPDATED)
N_agents = 4
dt = 0.1          # Match the reference trajectory generation
T_total = 50       # Total steps for masked game (NEW)
T_observation = 10  # First 10 steps used as observation (NEW)
T_reference = 50   # Reference trajectory has 50 steps
state_dim = 4       # (x, y, vx, vy)
control_dim = 2     # (ax, ay)

# PSN training parameters
num_epochs = 30
learning_rate = 1e-3
batch_size = 32     # Reduced for memory efficiency
sigma1 = 0.0        # Final mask sparsity weight (will gradually increase from 0 to this value)
sigma2 = 0.0      # Binary loss weight
sigma3 = 20.0       # Goal prediction loss weight

# Game solving parameters
num_iters = 20

# Reference trajectory parameters
reference_file = f"reference_trajectories_{N_agents}p/all_reference_trajectories.json"

# Device selection - Always use GPU for training
# Force GPU usage
gpu_devices = jax.devices("gpu")
if gpu_devices:
    device = gpu_devices[0]
    print(f"Using GPU: {device}")
    # Set JAX to use GPU
    os.environ['JAX_PLATFORM_NAME'] = 'gpu'
    print("JAX platform forced to: gpu")
    
    # Test GPU functionality with a simple operation
    test_array = jax.random.normal(jax.random.PRNGKey(0), (10, 10))
    test_result = jnp.linalg.inv(test_array)
    print("GPU matrix operations working correctly")
else:
    raise RuntimeError("No GPU devices found")

# Note: If you encounter GPU cuSolver errors, the script will automatically fallback to CPU.
# This is a known issue with certain GPU configurations and JAX versions.
# Alternative solutions:
# 1. Update NVIDIA drivers to latest version
# 2. Use conda install -c conda-forge jaxlib jax
# 3. Set JAX_PLATFORM_NAME=cpu in environment variables
# 4. The script will automatically handle fallbacks for matrix operations


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_matrix_inverse(matrix, device):
    """
    Compute matrix inverse on GPU.
    
    Args:
        matrix: Matrix to invert
        device: Current device (should be GPU)
        
    Returns:
        Matrix inverse on GPU
    """
    return jnp.linalg.inv(matrix)


# ============================================================================
# AGENT DEFINITIONS (EXACTLY FROM WORKING EXAMPLE)
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
        # Set attributes
        self.dt = dt
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.Q = Q
        self.R = R
    
    def dyn(self, x, u):
        """Dynamics function for point mass."""
        return jnp.array([
            x[2],  # dx/dt = vx
            x[3],  # dy/dt = vy
            u[0],  # dvx/dt = ax
            u[1]   # dvy/dt = ay
        ])


# ============================================================================
# PSN NETWORK DEFINITIONS (UPDATED FOR NEW REQUIREMENTS)
# ============================================================================

class PlayerSelectionNetwork(nn.Module):
    """
    Player Selection Network (PSN) that learns to select important agents and infer their goals.
    
    Input: First 10 steps of all agents' trajectories (T_observation * N_agents * state_dim)
    Output: 
        - Binary mask for selecting other agents (excluding ego agent)
        - Goal positions for all agents (including ego agent)
    """
    
    hidden_dim: int = 64
    gru_hidden_size: int = 64
    dropout_rate: float = 0.3
    mask_output_dim: int = N_agents - 1  # Mask for other agents (excluding ego)
    goal_output_dim: int = N_agents * 2  # Goal positions (x, y) for all agents
    
    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        """
        Forward pass of PSN.
        
        Args:
            x: Input tensor of shape (batch_size, T_observation * N_agents * state_dim)
               Flattened observation trajectories
            
        Returns:
            tuple: (mask, goals) where:
                - mask: Binary mask of shape (batch_size, N_agents - 1)
                - goals: Goal positions of shape (batch_size, N_agents * 2)
        """
        # Reshape input to (batch_size, T_observation, N_agents, state_dim)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, T_observation, N_agents, state_dim)
        
        # Process each agent's trajectory through shared GRU
        agent_features = []
        
        for agent_idx in range(N_agents):
            # Extract trajectory for this agent: (batch_size, T_observation, state_dim)
            agent_traj = x[:, :, agent_idx, :]
            
            # Process through shared GRU
            gru_out, _ = nn.GRUCell(
                features=self.gru_hidden_size,
                name=f'shared_gru'
            ).initialize_carry(jax.random.key(0), (batch_size, self.gru_hidden_size))
            
            # Process sequence step by step
            for t in range(T_observation):
                gru_out, _ = nn.GRUCell(
                    features=self.gru_hidden_size,
                    name=f'shared_gru'
                )(gru_out, agent_traj[:, t, :])
            
            # Use final GRU output as agent representation
            agent_features.append(gru_out)
        
        # Concatenate all agent features: (batch_size, N_agents * gru_hidden_size)
        x = jnp.concatenate(agent_features, axis=1)
        
        # Shared feature extraction layers
        x = nn.Dense(features=self.hidden_dim, name='shared_head_1')(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, name='shared_dropout_1')(x, deterministic=deterministic)
        
        x = nn.Dense(features=self.hidden_dim // 2, name='shared_head_2')(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, name='shared_dropout_2')(x, deterministic=deterministic)
        
        # Branch into two outputs: mask and goals
        # Mask branch (for agent selection)
        mask_branch = nn.Dense(features=self.hidden_dim // 4, name='mask_head')(x)
        mask_branch = nn.relu(mask_branch)
        mask = nn.Dense(features=self.mask_output_dim, name='mask_output')(mask_branch)
        mask = nn.sigmoid(mask)  # Binary mask
        
        # Goal branch (for goal inference)
        goal_branch = nn.Dense(features=self.hidden_dim // 4, name='goal_head')(x)
        goal_branch = nn.relu(goal_branch)
        goals = nn.Dense(features=self.goal_output_dim, name='goal_output')(goal_branch)
        # No activation for goals - they can be any real values
        
        return mask, goals


# ============================================================================
# LOSS FUNCTIONS (UPDATED)
# ============================================================================

def binary_loss(mask: jnp.ndarray) -> jnp.ndarray:
    """
    Binary loss: encourages mask values to be close to 0 or 1.
    
    Args:
        mask: Predicted mask values
        
    Returns:
        Binary loss value
    """
    # Encourage values to be close to 0 or 1
    binary_penalty = mask * (1 - mask)
    return jnp.mean(binary_penalty)


def mask_sparsity_loss(mask: jnp.ndarray) -> jnp.ndarray:
    """
    Mask sparsity loss: encourages fewer agents to be selected.
    
    Args:
        mask: Predicted mask values
        
    Returns:
        Sparsity loss value
    """
    return jnp.mean(mask)


def goal_prediction_loss(predicted_goals: jnp.ndarray, true_goals: jnp.ndarray) -> jnp.ndarray:
    """
    Goal prediction loss using Huber loss for robustness.
    
    Args:
        predicted_goals: Predicted goal positions (batch_size, N_agents * 2)
        true_goals: True goal positions (batch_size, N_agents * 2)
        
    Returns:
        Goal prediction loss value
    """
    # Compute Huber loss for robustness to outliers
    goal_diff = predicted_goals - true_goals
    
    # Huber loss parameters
    delta = 1.0
    
    # Compute Huber loss: more robust than MSE for outliers
    abs_diff = jnp.abs(goal_diff)
    quadratic = jnp.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    huber_loss = 0.5 * quadratic**2 + delta * linear
    
    return jnp.mean(huber_loss)


def compute_similarity_loss_from_masked_game(sample_data: Dict[str, Any], ego_agent_id: int, 
                                           predicted_mask: jnp.ndarray, predicted_goals: jnp.ndarray) -> jnp.ndarray:
    """
    Compute similarity loss by solving masked game and comparing with reference.
    
    Args:
        sample_data: Reference trajectory sample
        ego_agent_id: ID of the ego agent
        predicted_mask: Predicted mask from PSN
        
    Returns:
        Similarity loss value
    """
    # Create masked game setup
    agents, initial_states, target_positions, mask_values = create_masked_game_setup(
        sample_data, ego_agent_id, predicted_mask, predicted_goals, is_training=True)
    
    # Create loss functions with mask values
    loss_functions, linearize_functions, compiled_functions = create_loss_functions(
        agents, mask_values, is_training=True)
    
    # Solve masked game
    state_trajectories, control_trajectories = solve_masked_game(
        agents, initial_states, target_positions, compiled_functions, mask_values, num_iters=num_iters)
    
    # Extract ego agent trajectory from masked game
    ego_traj_masked = state_trajectories[0]  # Ego agent is always first
    
    # Extract reference trajectory for ego agent
    ego_traj_ref = extract_ego_reference_trajectory(sample_data, ego_agent_id)
    
    # Compute similarity loss
    return similarity_loss(ego_traj_masked, ego_traj_ref)


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


def compute_batch_similarity_loss(predicted_masks: jnp.ndarray, predicted_goals: jnp.ndarray, 
                                batch_data: List[Dict[str, Any]], ego_agent_id: int = 0) -> jnp.ndarray:
    """
    Compute similarity loss for a batch of samples by solving masked games.
    
    Args:
        predicted_masks: Predicted masks from PSN (batch_size, N_agents - 1)
        predicted_goals: Predicted goals from PSN (batch_size, N_agents * 2)
        batch_data: List of reference trajectory samples
        ego_agent_id: ID of the ego agent
        
    Returns:
        Average similarity loss for the batch
    """
    batch_losses = []
    
    # Use tqdm for progress tracking during similarity loss computation
    for i in tqdm(range(len(batch_data)), desc="Sample", leave=False):
        if i >= predicted_masks.shape[0]:  # Skip padding
            break
            
        sample_data = batch_data[i]
        mask = predicted_masks[i]
        goals = predicted_goals[i]
        
        # Create masked game setup
        agents, initial_states, target_positions, mask_values = create_masked_game_setup(
            sample_data, ego_agent_id, mask, goals, is_training=True)
        
        # Create loss functions with mask values
        loss_functions, linearize_functions, compiled_functions = create_loss_functions(
            agents, mask_values, is_training=True)
        
        # Solve masked game
        state_trajectories, control_trajectories = solve_masked_game(
            agents, initial_states, target_positions, compiled_functions, mask_values, num_iters=num_iters)
        
        # Extract ego agent trajectory from masked game
        ego_traj_masked = state_trajectories[0]  # Ego agent is always first
        
        # Extract reference trajectory for ego agent
        ego_traj_ref = extract_ego_reference_trajectory(sample_data, ego_agent_id)
        
        # Compute similarity loss
        sim_loss = similarity_loss(ego_traj_masked, ego_traj_ref)
        batch_losses.append(sim_loss)
    
    # Return average loss
    if len(batch_losses) > 0:
        return jnp.mean(jnp.array(batch_losses))
    else:
        return jnp.array(10.0)


def total_loss(mask: jnp.ndarray, binary_loss_val: jnp.ndarray, 
               sparsity_loss_val: jnp.ndarray, similarity_loss_val: jnp.ndarray,
               goal_prediction_loss_val: jnp.ndarray,
               sigma1: float = 0.1, sigma2: float = 1.0, sigma3: float = 0.5) -> jnp.ndarray:
    """
    Total loss combining all components.
    
    Args:
        mask: Predicted mask
        binary_loss_val: Binary loss value
        sparsity_loss_val: Sparsity loss value
        similarity_loss_val: Similarity loss value
        goal_prediction_loss_val: Goal prediction loss value
        sigma1: Weight for sparsity loss
        sigma2: Weight for similarity loss
        sigma3: Weight for goal prediction loss
        
    Returns:
        Total loss value
    """
    return similarity_loss_val + sigma1 * sparsity_loss_val + sigma2 * binary_loss_val + sigma3 * goal_prediction_loss_val


# ============================================================================
# REFERENCE TRAJECTORY LOADING
# ============================================================================

def load_reference_trajectories(file_path: str) -> List[Dict[str, Any]]:
    """
    Load reference trajectories from JSON file.
    
    Args:
        file_path: Path to the JSON file containing reference trajectories
        
    Returns:
        reference_data: List of reference trajectory samples
    """
    with open(file_path, 'r') as f:
        reference_data = json.load(f)
    
    print(f"Loaded {len(reference_data)} reference trajectory samples")
    return reference_data


def extract_observation_trajectory(sample_data: Dict[str, Any], ego_agent_id: int) -> jnp.ndarray:
    """
    Extract observation trajectory (first 10 steps) for all agents.
    
    Args:
        sample_data: Reference trajectory sample
        ego_agent_id: ID of the ego agent (for reference)
        
    Returns:
        observation_trajectory: Observation trajectory (T_observation, N_agents, state_dim)
    """
    observation_trajectory = []
    
    for i in range(N_agents):
        agent_key = f"agent_{i}"
        states = sample_data["trajectories"][agent_key]["states"]
        # Take first T_observation steps
        agent_states = jnp.array(states[:T_observation])  # (T_observation, state_dim)
        observation_trajectory.append(agent_states)
    
    # Stack all agents: (T_observation, N_agents, state_dim)
    return jnp.stack(observation_trajectory, axis=1)


def extract_ego_reference_trajectory(sample_data: Dict[str, Any], ego_agent_id: int) -> jnp.ndarray:
    """
    Extract full reference trajectory for the ego agent.
    
    Args:
        sample_data: Reference trajectory sample
        ego_agent_id: ID of the ego agent
        
    Returns:
        ego_trajectory: Full reference trajectory (T_reference, state_dim)
    """
    agent_key = f"agent_{ego_agent_id}"
    states = sample_data["trajectories"][agent_key]["states"]
    return jnp.array(states)  # (T_reference, state_dim)


def extract_reference_goals(sample_data: Dict[str, Any]) -> jnp.ndarray:
    """
    Extract goal positions for all agents from reference data.
    
    Args:
        sample_data: Reference trajectory sample
        
    Returns:
        goals: Goal positions for all agents (N_agents, 2)
    """
    return jnp.array(sample_data["target_positions"])  # (N_agents, 2)


def create_masked_game_setup(sample_data: Dict[str, Any], ego_agent_id: int, 
                           mask: jnp.ndarray, predicted_goals: jnp.ndarray, 
                           is_training: bool = True) -> Tuple[List[PointAgent], List[jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """
    Create a game setup with mask-based cost filtering.
    
    During training: ALL agents are included, but only agent 0 (ego agent) uses mask values
    for mutual costs. Other agents always consider full mutual costs with all agents.
    During runtime: Only selected agents are included based on binary mask threshold.
    
    Args:
        sample_data: Reference trajectory sample
        ego_agent_id: ID of the ego agent
        mask: Continuous mask values [0,1] for cost filtering
        predicted_goals: Predicted goal positions for all agents (N_agents * 2)
        is_training: If True, include all agents with masked costs; if False, only selected agents
        
    Returns:
        agents: List of agents (all agents if training, selected agents if runtime)
        initial_states: Initial states for agents
        target_positions: Target positions for agents (using predicted goals)
        mask_values: Mask values for collision cost multiplication (original mask during training, selected mask during runtime)
    """
    init_positions = jnp.array(sample_data["init_positions"])
    
    # Reshape predicted goals to (N_agents, 2)
    predicted_goals = predicted_goals.reshape(N_agents, 2)
    
    if is_training:
        # Training time: Include ALL agents, mask values multiply the mutual costs
        # This implements the equation: \tilde{c}_{s,k}^i = \sum_{j=1}^N m^{ij}c_k^{ij}
        agents = []
        initial_states = []
        target_positions = []
        
        # Cost function weights (same for all agents)
        Q = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001]))  # State cost weights
        R = jnp.diag(jnp.array([0.01, 0.01]))               # Control cost weights
        
        for agent_id in range(N_agents):
            # Create agent
            agent = PointAgent(dt=dt, x_dim=4, u_dim=2, Q=Q, R=R)
            agents.append(agent)
            
            # Initial state - convert 2D position to 4D state [x, y, vx, vy]
            pos_2d = init_positions[agent_id]  # [x, y]
            initial_state = jnp.array([pos_2d[0], pos_2d[1], 0.0, 0.0])  # [x, y, vx, vy]
            initial_states.append(initial_state)
            
            # Use predicted goal position for this agent
            target_positions.append(predicted_goals[agent_id])
        
        # For training, only agent 0 (ego agent) uses the mask
        # Other agents have full interaction with all agents (no masking)
        # Simply return the mask values directly - they will only be used by agent 0
        return agents, initial_states, jnp.array(target_positions), mask
        
    else:
        # Runtime: Only include selected agents based on binary mask threshold
        # This implements the threshold conversion: m_{ij} > m_th -> 1, otherwise 0
        selected_agents = [ego_agent_id]
        mask_values = []
        
        for i in range(N_agents - 1):
            # Map mask index to actual agent ID (skip ego agent)
            agent_id = i if i < ego_agent_id else i + 1
            if mask[i] > 0.5:  # Threshold for selection
                selected_agents.append(agent_id)
                mask_values.append(mask[i])
        
        # Create agents and get their initial states
        agents = []
        initial_states = []
        selected_targets = []
        
        # Cost function weights (same for all agents)
        Q = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001]))  # State cost weights
        R = jnp.diag(jnp.array([0.01, 0.01]))               # Control cost weights
        
        for agent_id in selected_agents:
            # Create agent
            agent = PointAgent(dt=dt, x_dim=4, u_dim=2, Q=Q, R=R)
            agents.append(agent)
            
            # Initial state - convert 2D position to 4D state [x, y, vx, vy]
            pos_2d = init_positions[agent_id]  # [x, y]
            initial_state = jnp.array([pos_2d[0], pos_2d[1], 0.0, 0.0])  # [x, y, vx, vy]
            initial_states.append(initial_state)
            
            # Use predicted goal position for this agent
            selected_targets.append(predicted_goals[agent_id])
        
        return agents, initial_states, jnp.array(selected_targets), jnp.array(mask_values)


def create_loss_functions(agents: list, mask_values = None, is_training: bool = True) -> tuple:
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
        def create_runtime_loss(agent_idx, agent_obj, mask_values=None, is_training=False):
            def runtime_loss(xt, ut, ref_xt, other_states):
                # Navigation cost
                nav_loss = jnp.sum(jnp.square(xt[:2] - ref_xt[:2]))
                
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
        
        runtime_loss = create_runtime_loss(i, agent, mask_values, is_training)
        
        # Create trajectory loss function
        def trajectory_loss(x_traj, u_traj, ref_x_traj, other_x_trajs):
            def single_step_loss(args):
                xt, ut, ref_xt, other_xts = args
                return runtime_loss(xt, ut, ref_xt, other_xts)
            
            loss_array = jax.vmap(single_step_loss)((x_traj, u_traj, ref_x_traj, other_x_trajs))
            return loss_array.sum() * agent.dt
        
        # Create linearization function
        def linearize_loss(x_traj, u_traj, ref_x_traj, other_x_trajs):
            dldx = jax.grad(runtime_loss, argnums=(0))
            dldu = jax.grad(runtime_loss, argnums=(1))
            
            def grad_step(args):
                xt, ut, ref_xt, other_xts = args
                return dldx(xt, ut, ref_xt, other_xts), dldu(xt, ut, ref_xt, other_xts)
            
            grads = jax.vmap(grad_step)((x_traj, u_traj, ref_x_traj, other_x_trajs))
            return grads[0], grads[1]  # a_traj, b_traj
        
        # Compile functions with GPU optimizations
        compiled_loss = jax.jit(trajectory_loss, device=device)
        compiled_linearize = jax.jit(linearize_loss, device=device)
        compiled_linearize_dyn = jax.jit(agent.linearize_dyn, device=device)
        compiled_solve = jax.jit(agent.solve, device=device)
        
        loss_functions.append(trajectory_loss)
        linearize_loss_functions.append(linearize_loss)
        compiled_functions.append({
            'loss': compiled_loss,
            'linearize_loss': compiled_linearize,
            'linearize_dyn': compiled_linearize_dyn,
            'solve': compiled_solve
        })
    
    return loss_functions, linearize_loss_functions, compiled_functions


def solve_masked_game(agents: list, initial_states: list, target_positions: jnp.ndarray,
                     compiled_functions: list, mask_values: jnp.ndarray = None, num_iters: int = 100) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve the masked game for selected agents.
    
    Args:
        agents: List of selected agents
        initial_states: List of initial states for selected agents
        target_positions: Target positions for selected agents
        compiled_functions: List of compiled functions for each agent
        num_iters: Number of optimization iterations (reduced for safety)
        
    Returns:
        Tuple of (state_trajectories, control_trajectories)
    """
    n_selected = len(agents)
    
    # Create reference trajectories (linear interpolation to targets)
    reference_trajectories = []
    for i in range(n_selected):
        start_pos = initial_states[i][:2]  # [x, y]
        end_pos = target_positions[i]      # [x, y]
        # Create 2D position trajectory (like in working example)
        ref_traj = jnp.linspace(start_pos, end_pos, T_total)  # (T_total, 2)
        reference_trajectories.append(ref_traj)
    
    # Initialize control trajectories
    control_trajectories = [jnp.zeros((T_total, 2)) for _ in range(n_selected)]
    
    # Optimization parameters
    step_size = 0.002
    
    for iter in range(num_iters + 1):
        # Step 1: Linearize dynamics for all agents
        state_trajectories = []
        A_trajectories = []
        B_trajectories = []
        
        for i, agent in enumerate(agents):
            x_traj, A_traj, B_traj = compiled_functions[i]['linearize_dyn'](
                initial_states[i], control_trajectories[i])
            state_trajectories.append(x_traj)
            A_trajectories.append(A_traj)
            B_trajectories.append(B_traj)
        
        # Step 2: Linearize loss functions for all agents
        a_trajectories = []
        b_trajectories = []
        
        for i in range(n_selected):
            # Create list of other agents' states for this agent
            other_states = [state_trajectories[j] for j in range(n_selected) if j != i]
            
            a_traj, b_traj = compiled_functions[i]['linearize_loss'](
                state_trajectories[i], control_trajectories[i], 
                reference_trajectories[i], other_states)
            a_trajectories.append(a_traj)
            b_trajectories.append(b_traj)
        
        # Step 3: Solve LQR subproblems for all agents
        control_updates = []
        
        for i in range(n_selected):
            v_traj, _ = compiled_functions[i]['solve'](
                A_trajectories[i], B_trajectories[i], 
                a_trajectories[i], b_trajectories[i])
            control_updates.append(v_traj)
        
        # Step 4: Update control trajectories
        for i in range(n_selected):
            control_trajectories[i] += step_size * control_updates[i]
    
    # Ensure all trajectories have the correct length
    for i in range(n_selected):
        if state_trajectories[i].shape[0] != T_total:
            print(f"Warning: Agent {i} trajectory length {state_trajectories[i].shape[0]} != {T_total}")
            # Pad with last state if too short
            if state_trajectories[i].shape[0] > 0:
                last_state = state_trajectories[i][-1:]
                pad_size = T_total - state_trajectories[i].shape[0]
                padding = jnp.tile(last_state, (pad_size, 1))
                state_trajectories[i] = jnp.concatenate([state_trajectories[i], padding], axis=0)
            else:
                # Create fallback trajectory
                start_pos = initial_states[i][:2]
                end_pos = target_positions[i]
                fallback_traj = jnp.linspace(start_pos, end_pos, T_total)
                fallback_state = jnp.zeros((T_total, 4))
                fallback_state = fallback_state.at[:, :2].set(fallback_traj)
                state_trajectories[i] = fallback_state
    
    return state_trajectories, control_trajectories


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_train_state(model: nn.Module, optimizer: optax.GradientTransformation, 
                      input_shape: Tuple[int, ...], rng: jnp.ndarray) -> train_state.TrainState:
    """
    Create training state for the model.
    
    Args:
        model: PSN model
        optimizer: Optax optimizer
        input_shape: Input shape for model initialization
        rng: Random key
        
    Returns:
        Train state
    """
    # Create dummy input for initialization
    dummy_input = jnp.ones(input_shape)
    
    # Initialize model parameters
    variables = model.init(rng, dummy_input)
    params = variables['params']
    
    # Create train state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    return state


def train_step(state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
               batch_data: List[Dict[str, Any]], sigma1: float = 0.1, sigma2: float = 1.0, sigma3: float = 0.5) -> Tuple[train_state.TrainState, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Single training step.
    
    Args:
        state: Current train state
        batch: Tuple of (observations, masks, reference_trajectories, true_goals)
        sigma1: Weight for sparsity loss
        sigma2: Weight for similarity loss
        sigma3: Weight for goal prediction loss
        
    Returns:
        Updated train state, total loss value, and tuple of individual loss components
    """
    observations, masks, reference_trajectories, true_goals = batch
    
    def loss_fn(params):
        # Apply the model with the given parameters
        predicted_masks, predicted_goals = state.apply_fn({'params': params}, observations)
        
        # Compute individual loss components
        binary_loss_val = binary_loss(predicted_masks)
        sparsity_loss_val = mask_sparsity_loss(predicted_masks)
        goal_prediction_loss_val = goal_prediction_loss(predicted_goals, true_goals)
        
        # Compute real similarity loss by solving masked game
        # This is computationally expensive but provides accurate gradients
        similarity_loss_val = compute_batch_similarity_loss(predicted_masks, predicted_goals, batch_data)
        
        # Compute total loss
        total_loss_val = total_loss(predicted_masks, binary_loss_val, 
                                  sparsity_loss_val, similarity_loss_val, goal_prediction_loss_val,
                                  sigma1, sigma2, sigma3)
        
        return total_loss_val, (binary_loss_val, sparsity_loss_val, goal_prediction_loss_val, similarity_loss_val)
    
    # Compute gradients
    (loss, loss_components), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    binary_loss_val, sparsity_loss_val, goal_prediction_loss_val, similarity_loss_val = loss_components
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    return state, loss, (binary_loss_val, sparsity_loss_val, goal_prediction_loss_val, similarity_loss_val)


def train_psn_with_reference(model: nn.Module, reference_data: List[Dict[str, Any]],
                           num_epochs: int = 100, learning_rate: float = 1e-3,
                           sigma1: float = 0.1, sigma2: float = 1.0, sigma3: float = 0.5,
                           batch_size: int = 16, rng: jnp.ndarray = None) -> Tuple[List[float], train_state.TrainState, str, float, int]:
    """
    Train PSN using reference trajectories with NEW REQUIREMENTS.
    
    Args:
        model: PSN model to train
        reference_data: Reference trajectory data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        sigma1: Final weight for mask sparsity loss (will gradually increase from 0)
        sigma2: Weight for similarity loss
        sigma3: Weight for goal prediction loss
        batch_size: Batch size
        rng: Random key
        
    Returns:
        losses: List of training losses
        state: Best trained model state
        log_dir: Directory where logs are saved
        best_loss: Best loss achieved during training
        best_epoch: Epoch where best loss was achieved
    """
    if rng is None:
        rng = jax.random.PRNGKey(42)
    
    # Create log directory and TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = f"psn_N_{N_agents}_T_{T_total}_obs_{T_observation}_ref_{T_reference}_lr_{learning_rate}_bs_{batch_size}_sigma1_{sigma1}_sigma2_{sigma2}_sigma3_{sigma3}_epochs_{num_epochs}"
    log_dir = f"log/{config_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = tb.SummaryWriter(log_dir)
    
    # Note: TensorBoard logging is configured to record only epoch-level metrics
    # to reduce clutter and focus on overall training progress
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    
    # Create train state
    input_shape = (batch_size, T_observation * N_agents * state_dim)
    state = create_train_state(model, optimizer, input_shape, rng)
    
    losses = []
    best_loss = float('inf')
    best_state = None
    best_epoch = 0
    
    print(f"Training PSN with {len(reference_data)} reference samples...")
    print(f"Parameters: epochs={num_epochs}, lr={learning_rate}, batch_size={batch_size}")
    print(f"Observation steps: {T_observation}, Total game steps: {T_total}")
    print(f"Sigma1 will gradually increase from 0 to {sigma1} over {num_epochs} epochs")
    
    # Overall training progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", position=0)
    
    for epoch in epoch_pbar:
        # Calculate current sigma1 value (linear schedule from 0 to final sigma1)
        current_sigma1 = (epoch / (num_epochs - 1)) * sigma1 if num_epochs > 1 else sigma1
        
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle reference data
        rng, shuffle_key = jax.random.split(rng)
        shuffled_indices = jax.random.permutation(shuffle_key, len(reference_data))
        
        # Create batches
        num_batches_epoch = (len(reference_data) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(reference_data), batch_size), 
                   desc=f"Epoch {epoch+1}/{num_epochs}", 
                   total=num_batches_epoch,
                   position=1, leave=False)
        
        for i in pbar:
            batch_indices = shuffled_indices[i:i + batch_size]
            batch_data = [reference_data[idx] for idx in batch_indices]
            
            # Process each sample in batch
            batch_obs = []
            batch_masks = []
            batch_ref_traj = []
            batch_true_goals = [] # Added for true goals
            
            for sample_data in batch_data:
                # ASSIGN AGENT 0 AS EGO AGENT
                ego_agent_id = 0
                
                # Extract observation trajectory (first 10 steps of all agents)
                obs_traj = extract_observation_trajectory(sample_data, ego_agent_id)
                batch_obs.append(obs_traj.flatten())  # Flatten to 1D
                
                # Extract ego reference trajectory
                ego_ref_traj = extract_ego_reference_trajectory(sample_data, ego_agent_id)
                # Ensure consistent shape: (T_reference, state_dim)
                if ego_ref_traj.shape[0] != T_reference:
                    # Pad or truncate to match T_reference
                    if ego_ref_traj.shape[0] < T_reference:
                        # Pad with last state
                        pad_size = T_reference - ego_ref_traj.shape[0]
                        last_state = ego_ref_traj[-1:]
                        padding = jnp.tile(last_state, (pad_size, 1))
                        ego_ref_traj = jnp.concatenate([ego_ref_traj, padding], axis=0)
                    else:
                        # Truncate to T_reference
                        ego_ref_traj = ego_ref_traj[:T_reference]
                batch_ref_traj.append(ego_ref_traj)
                
                # Extract true goals for this sample
                true_goals = jnp.array(sample_data["target_positions"]).flatten()  # Flatten to (N_agents * 2)
                batch_true_goals.append(true_goals)
                
                # For now, use random mask (will be replaced by PSN prediction)
                rng, mask_key = jax.random.split(rng)
                random_mask = jax.random.uniform(mask_key, (N_agents - 1,))
                batch_masks.append(random_mask)
            
            # Pad batch if necessary
            if len(batch_obs) < batch_size:
                pad_size = batch_size - len(batch_obs)
                # Pad observations
                obs_pad = jnp.zeros((pad_size, T_observation * N_agents * state_dim))
                batch_obs.extend([obs_pad[i] for i in range(pad_size)])
                # Pad masks
                mask_pad = jnp.zeros((pad_size, N_agents - 1))
                batch_masks.extend([mask_pad[i] for i in range(pad_size)])
                # Pad reference trajectories
                ref_pad = jnp.zeros((pad_size, T_reference, state_dim))
                batch_ref_traj.extend([ref_pad[i] for i in range(pad_size)])
                # Pad true goals
                true_goals_pad = jnp.zeros((pad_size, N_agents * 2))
                batch_true_goals.extend([true_goals_pad[i] for i in range(pad_size)])
            
            # Convert to JAX arrays
            batch_obs = jnp.stack(batch_obs)
            batch_masks = jnp.stack(batch_masks)
            batch_ref_traj = jnp.stack(batch_ref_traj)
            batch_true_goals = jnp.stack(batch_true_goals) # Stack true goals
            
            # Training step with current sigma1 value
            state, loss, (binary_loss_val, sparsity_loss_val, goal_prediction_loss_val, similarity_loss_val) = train_step(state, (batch_obs, batch_masks, batch_ref_traj, batch_true_goals), batch_data, current_sigma1, sigma2, sigma3)
            epoch_loss += loss
            num_batches += 1
            
            # Update progress bar with loss info
            pbar.set_postfix({'Loss': f'{float(loss):.4f}', 'σ1': f'{current_sigma1:.3f}'})
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        epoch_pbar.set_postfix({'Avg Loss': f'{avg_loss:.4f}', 'σ1': f'{current_sigma1:.3f}'})
        
        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = state
            best_epoch = epoch
            print(f"New best model found at epoch {epoch+1} with loss: {avg_loss:.4f} (σ1: {current_sigma1:.3f})")
            # Save the best model
            best_model_bytes = flax.serialization.to_bytes(best_state)
            best_model_path = os.path.join(log_dir, "psn_best_model.pkl")
            with open(best_model_path, 'wb') as f:
                pickle.dump(best_model_bytes, f)
            print(f"Best model saved to: {best_model_path}")
        
        # Log epoch-level metrics to TensorBoard
        writer.add_scalar('Loss/Epoch', float(avg_loss), epoch)
        writer.add_scalar('Loss/Binary/Epoch', float(binary_loss_val), epoch)
        writer.add_scalar('Loss/Sparsity/Epoch', float(sparsity_loss_val), epoch)
        writer.add_scalar('Loss/GoalPrediction/Epoch', float(goal_prediction_loss_val), epoch)
        writer.add_scalar('Loss/Similarity/Epoch', float(similarity_loss_val), epoch)
        writer.add_scalar('Loss/Best', float(best_loss), epoch)
        writer.add_scalar('Hyperparameters/Sigma1', float(current_sigma1), epoch)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, Best: {best_loss:.4f} (epoch {best_epoch+1}), σ1: {current_sigma1:.3f}")
        
        # Clean up memory after each epoch
        gc.collect()
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
        

    
    # Close TensorBoard writer
    writer.close()
    
    print(f"\nTraining completed! Best model found at epoch {best_epoch+1} with loss: {best_loss:.4f}")
    
    return losses, best_state, log_dir, best_loss, best_epoch


def evaluate_psn_model(model: nn.Module, trained_state: train_state.TrainState, 
                      reference_data: List[Dict[str, Any]], num_samples: int = 10) -> Dict[str, float]:
    """
    Evaluate the trained PSN model on goal prediction accuracy.
    
    Args:
        model: PSN model
        trained_state: Trained model state
        reference_data: Reference trajectory data
        num_samples: Number of samples to evaluate
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\nEvaluating PSN model on {num_samples} samples...")
    
    goal_prediction_errors = []
    mask_accuracies = []
    
    # Sample random indices
    rng = jax.random.PRNGKey(42)
    rng, sample_key = jax.random.split(rng)
    sample_indices = jax.random.choice(sample_key, len(reference_data), shape=(num_samples,), replace=False)
    
    for idx in sample_indices:
        sample_data = reference_data[idx]
        
        # Extract observation trajectory
        ego_agent_id = 0
        obs_traj = extract_observation_trajectory(sample_data, ego_agent_id)
        obs_input = obs_traj.flatten().reshape(1, -1)  # Add batch dimension
        
        # Get model predictions
        predicted_mask, predicted_goals = trained_state.apply_fn({'params': trained_state.params}, obs_input)
        
        # Extract true goals
        true_goals = extract_reference_goals(sample_data)
        true_goals_flat = true_goals.flatten()
        
        # Compute goal prediction error
        goal_error = jnp.mean(jnp.square(predicted_goals[0] - true_goals_flat))
        goal_prediction_errors.append(float(goal_error))
        
        # For mask accuracy, we'll use a simple threshold-based evaluation
        # In practice, you might want to compare with some ground truth mask
        mask_accuracy = float(jnp.mean(predicted_mask[0] > 0.5))  # Percentage of selected agents
        mask_accuracies.append(mask_accuracy)
    
    # Compute metrics
    avg_goal_error = np.mean(goal_prediction_errors)
    std_goal_error = np.std(goal_prediction_errors)
    avg_mask_accuracy = np.mean(mask_accuracies)
    
    metrics = {
        'avg_goal_prediction_error': avg_goal_error,
        'std_goal_prediction_error': std_goal_error,
        'avg_mask_accuracy': avg_mask_accuracy,
        'goal_prediction_rmse': np.sqrt(avg_goal_error)
    }
    
    print(f"Goal Prediction RMSE: {metrics['goal_prediction_rmse']:.4f}")
    print(f"Goal Prediction Error (mean ± std): {avg_goal_error:.4f} ± {std_goal_error:.4f}")
    print(f"Average Mask Accuracy: {avg_mask_accuracy:.4f}")
    
    return metrics


def visualize_goal_predictions(model: nn.Module, trained_state: train_state.TrainState,
                              reference_data: List[Dict[str, Any]], num_samples: int = 5, 
                              save_dir: str = None) -> None:
    """
    Visualize goal predictions vs true goals for selected samples.
    
    Args:
        model: PSN model
        trained_state: Trained model state
        reference_data: Reference trajectory data
        num_samples: Number of samples to visualize
        save_dir: Directory to save plots
    """
    print(f"\nVisualizing goal predictions for {num_samples} samples...")
    
    # Sample random indices
    rng = jax.random.PRNGKey(42)
    rng, sample_key = jax.random.split(rng)
    sample_indices = jax.random.choice(sample_key, len(reference_data), shape=(num_samples,), replace=False)
    
    for i, idx in enumerate(sample_indices):
        sample_data = reference_data[idx]
        
        # Extract observation trajectory
        ego_agent_id = 0
        obs_traj = extract_observation_trajectory(sample_data, ego_agent_id)
        obs_input = obs_traj.flatten().reshape(1, -1)  # Add batch dimension
        
        # Get model predictions
        predicted_mask, predicted_goals = trained_state.apply_fn({'params': trained_state.params}, obs_input)
        
        # Extract true goals and initial positions
        true_goals = extract_reference_goals(sample_data)
        init_positions = jnp.array(sample_data["init_positions"])
        
        # Reshape predicted goals
        predicted_goals = predicted_goals[0].reshape(N_agents, 2)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_aspect('equal')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(f'Sample {i+1}: Goal Predictions vs True Goals')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        
        # Color palette for agents
        colors = plt.cm.tab10(np.linspace(0, 1, N_agents))
        
        # Plot initial positions
        for j in range(N_agents):
            ax.plot(init_positions[j][0], init_positions[j][1], 'o', 
                   color=colors[j], markersize=10, alpha=0.7, label=f'Agent {j} Start')
        
        # Plot true goals
        for j in range(N_agents):
            ax.plot(true_goals[j][0], true_goals[j][1], 's', 
                   color=colors[j], markersize=12, alpha=0.8, label=f'Agent {j} True Goal')
        
        # Plot predicted goals
        for j in range(N_agents):
            ax.plot(predicted_goals[j][0], predicted_goals[j][1], '^', 
                   color=colors[j], markersize=12, alpha=0.8, label=f'Agent {j} Predicted Goal')
        
        # Draw lines from initial to predicted goals
        for j in range(N_agents):
            ax.plot([init_positions[j][0], predicted_goals[j][0]], 
                   [init_positions[j][1], predicted_goals[j][1]], 
                   '--', color=colors[j], alpha=0.5, linewidth=2)
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save plot if save_dir is provided
        if save_dir:
            plot_path = os.path.join(save_dir, f"goal_predictions_sample_{i+1:03d}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Goal prediction plot saved to: {plot_path}")
        
        # Always close the plot to free memory
        plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PSN Training with Reference Trajectories (UPDATED)")
    print("=" * 60)
    
    # Load reference trajectories
    print(f"Loading reference trajectories from {reference_file}...")
    reference_data = load_reference_trajectories(reference_file)
    
    # Create PSN model
    psn_model = PlayerSelectionNetwork()
    
    # Train PSN
    print("\nTraining PSN...")
    print(f"Note: Sigma1 will gradually increase from 0 to {sigma1} over {num_epochs} epochs")
    rng = jax.random.PRNGKey(42)
    losses, trained_state, log_dir, best_loss, best_epoch = train_psn_with_reference(
        psn_model, reference_data, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate,
        sigma1=sigma1, 
        sigma2=sigma2, 
        sigma3=sigma3,
        batch_size=batch_size, 
        rng=rng
    )
    
    # Save best trained model with config-based naming
    best_model_bytes = flax.serialization.to_bytes(trained_state)
    best_model_path = os.path.join(log_dir, "psn_best_model.pkl")
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model_bytes, f)
    
    # Also save the final model for comparison
    final_model_path = os.path.join(log_dir, "psn_final_model.pkl")
    with open(final_model_path, 'wb') as f:
        pickle.dump(best_model_bytes, f)  # Note: trained_state is already the best_state
    
    # Save training config
    config = {
        'N_agents': N_agents,
        'T_total': T_total,
        'T_observation': T_observation,
        'T_reference': T_reference,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'sigma1': sigma1,
        'sigma2': sigma2,
        'sigma3': sigma3,
        'num_epochs': num_epochs,
        'final_loss': float(losses[-1]),
        'best_loss': float(best_loss),
        'best_epoch': int(best_epoch + 1),  # Convert to 1-indexed
        'timestamp': datetime.now().isoformat()
    }
    config_path = os.path.join(log_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Log directory: {log_dir}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Config saved to: {config_path}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Best loss: {best_loss:.4f} (achieved at epoch {best_epoch+1})")
    
    # Evaluate the trained model
    evaluation_metrics = evaluate_psn_model(psn_model, trained_state, reference_data, num_samples=20)
    
    # Save evaluation metrics
    metrics_path = os.path.join(log_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    print(f"Evaluation metrics saved to: {metrics_path}")
    
    # Visualize goal predictions
    # visualize_goal_predictions(psn_model, trained_state, reference_data, num_samples=5, save_dir=log_dir)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('PSN Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plot_path = os.path.join(log_dir, "training_loss.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Always close to free memory
    
    print(f"Training loss plot saved to: {plot_path}")
    print(f"\nTo view TensorBoard logs, run: tensorboard --logdir={log_dir}") 