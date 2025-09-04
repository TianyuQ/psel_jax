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
# GOAL INFERENCE NETWORK DEFINITION (same as original)
# ============================================================================

class GoalInferenceNetwork(nn.Module):
    """Goal inference network using GRU for temporal sequence processing."""
    
    hidden_dims: List[int]
    gru_hidden_size: int = 64
    dropout_rate: float = 0.3
    goal_output_dim: int = N_agents * goal_dim
    obs_input_type: str = "full"  # "full" or "partial"
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        """
        Forward pass of the goal inference network.
        
        Args:
            x: Input observations 
                - If obs_input_type="full": (batch_size, T_observation * N_agents * 4)
                - If obs_input_type="partial": (batch_size, T_observation * N_agents * 2)
            deterministic: Whether to use deterministic mode (no dropout)
            
        Returns:
            Predicted goals (batch_size, N_agents * goal_dim)
        """
        batch_size = x.shape[0]
        
        # Determine input dimension based on observation type
        input_dim = 2 if self.obs_input_type == "partial" else 4
        
        # Reshape to separate time steps and agents
        x = x.reshape(batch_size, T_observation, N_agents, input_dim)
        
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
# RECEDING HORIZON TRAJECTORY LOADING WITH SLIDING WINDOWS
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
        validation_split: Fraction of data to use for validation (ignored, using fixed split)
        
    Returns:
        training_data: List of training samples (first 384 samples)
        validation_data: List of validation samples (later 128 samples)
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
    train_size = config.training.train_samples
    val_size = config.training.test_samples
    
    # Check if we have enough samples
    if total_samples < train_size + val_size:
        print(f"Warning: Only {total_samples} samples available, but need {train_size + val_size} samples")
        print(f"Using all available samples: {total_samples} for training, 0 for validation")
        training_data = rh_data
        validation_data = []
    else:
        # Use first 384 samples for training, later 128 samples for testing
        # Sort by sample_id to ensure consistent ordering
        rh_data.sort(key=lambda x: x.get('sample_id', 0))
        
        training_data = rh_data[:train_size]
        validation_data = rh_data[train_size:train_size + val_size]
    
    print(f"Loaded {total_samples} receding horizon trajectory samples")
    print(f"Training samples: {len(training_data)} (first {len(training_data)} samples)")
    print(f"Validation samples: {len(validation_data)} (samples {len(training_data)} to {len(training_data) + len(validation_data) - 1})")
    
    return training_data, validation_data

def extract_sliding_window_trajectories(sample_data: Dict[str, Any], window_start: int, obs_input_type: str = "full") -> jnp.ndarray:
    """
    Extract a sliding window of trajectory data from any position.
    
    Args:
        sample_data: Trajectory sample data
        window_start: Starting step for the sliding window (0-indexed)
        obs_input_type: Observation input type ["full", "partial"]
        
    Returns:
        observation_trajectory: (T_observation, N_agents, output_dim)
            - If obs_input_type="full": output_dim = 4 (x, y, vx, vy)
            - If obs_input_type="partial": output_dim = 2 (x, y)
    """
    # Determine output dimension based on observation type
    output_dim = 2 if obs_input_type == "partial" else 4
    
    # Initialize array to store all agent states
    observation_trajectory = jnp.zeros((T_observation, N_agents, output_dim))
    
    for i in range(N_agents):
        agent_key = f"agent_{i}"
        states = sample_data["trajectories"][agent_key]["states"]
        
        # Extract T_observation steps starting from window_start
        window_end = window_start + T_observation
        
        if window_end <= len(states):
            # Normal case: full window fits within trajectory
            agent_states = jnp.array(states[window_start:window_end])
        else:
            # Edge case: pad with last state if window extends beyond trajectory
            available_states = states[window_start:] if window_start < len(states) else []
            if available_states:
                agent_states_padded = available_states[:]
                last_state = available_states[-1]
                while len(agent_states_padded) < T_observation:
                    agent_states_padded.append(last_state)
                agent_states = jnp.array(agent_states_padded)
            else:
                # If window_start is beyond trajectory, use last state
                last_state = states[-1] if states else [0.0, 0.0, 0.0, 0.0]
                agent_states = jnp.array([last_state for _ in range(T_observation)])
        
        # Extract only position data (x, y) if partial observations
        if obs_input_type == "partial":
            agent_states = agent_states[:, :2]  # Only x, y coordinates
        
        # Place in the correct position
        observation_trajectory = observation_trajectory.at[:, i, :].set(agent_states)
    
    return observation_trajectory

def extract_reference_goals(sample_data: Dict[str, Any]) -> jnp.ndarray:
    """Extract goal positions for all agents."""
    return jnp.array(sample_data["target_positions"])

def generate_sliding_window_samples(rh_data: List[Dict[str, Any]], samples_per_trajectory: int = 5, obs_input_type: str = "full") -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Generate multiple sliding window samples from each receding horizon trajectory.
    
    Args:
        rh_data: List of receding horizon trajectory samples
        samples_per_trajectory: Number of sliding windows to extract per trajectory
        obs_input_type: Observation input type ["full", "partial"]
        
    Returns:
        List of (observation_window, goals) tuples
    """
    sliding_samples = []
    
    for sample_data in rh_data:
        goals = extract_reference_goals(sample_data)
        
        # Calculate maximum valid starting positions for sliding windows
        # Assume trajectory length is T_total (50 steps)
        max_start = max(0, T_total - T_observation)  # 50 - 10 = 40, so starts 0-40
        
        if max_start <= 0:
            # If trajectory is too short, only use starting from 0
            window_starts = [0]
        else:
            # Generate evenly spaced window starting positions
            if samples_per_trajectory == 1:
                window_starts = [0]  # Just use the beginning
            else:
                window_starts = np.linspace(0, max_start, samples_per_trajectory, dtype=int)
        
        for window_start in window_starts:
            try:
                obs_window = extract_sliding_window_trajectories(sample_data, window_start, obs_input_type)
                sliding_samples.append((obs_window, goals))
            except Exception as e:
                print(f"Warning: Failed to extract window starting at step {window_start}: {e}")
                continue
    
    print(f"Generated {len(sliding_samples)} sliding window samples from {len(rh_data)} trajectories")
    print(f"Average {len(sliding_samples)/len(rh_data):.1f} windows per trajectory")
    
    return sliding_samples


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_train_state(model: nn.Module, learning_rate: float, obs_input_type: str = "full") -> train_state.TrainState:
    """Create training state for the model."""
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=5e-4
    )
    
    # Determine input dimension based on observation type
    obs_dim = 2 if obs_input_type == "partial" else 4
    
    # Create dummy input for initialization
    input_shape = (batch_size, T_observation * N_agents * obs_dim)
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
                                   goal_loss_weight: float = 1.0,
                                   obs_input_type: str = "full") -> Tuple[List[float], train_state.TrainState, str, float, int]:
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
    state = create_train_state(goal_model, learning_rate, obs_input_type)
    
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
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Goal Inference Network Pretraining on Receding Horizon Trajectories")
    print("=" * 60)
    print(f"Configuration loaded from: config.yaml")
    print(f"Game parameters: N_agents={N_agents}, T_observation={T_observation}, T_total={T_total}")
    print(f"Training parameters: epochs={num_epochs}, lr={learning_rate}, batch_size={batch_size}")
    print(f"Network parameters: hidden_dims={hidden_dims}, gru_hidden_size={gru_hidden_size}, dropout_rate={dropout_rate}")
    print(f"Observation type: {config.goal_inference.obs_input_type}")
    print("=" * 60)
    
    # Load receding horizon trajectories
    print(f"Loading receding horizon trajectories from directory: {rh_data_dir}...")
    training_data, validation_data = load_receding_horizon_trajectories(rh_data_dir, validation_split)
    
    # Generate sliding window samples
    print(f"Generating sliding window samples...")
    samples_per_trajectory = 5  # Extract 5 different windows per trajectory
    obs_input_type = config.goal_inference.obs_input_type
    training_samples = generate_sliding_window_samples(training_data, samples_per_trajectory, obs_input_type)
    validation_samples = generate_sliding_window_samples(validation_data, samples_per_trajectory, obs_input_type)
    
    print(f"Final dataset sizes:")
    print(f"  Training sliding windows: {len(training_samples)}")
    print(f"  Validation sliding windows: {len(validation_samples)}")
    
    # Create goal inference model
    goal_model = GoalInferenceNetwork(
        hidden_dims=hidden_dims,
        gru_hidden_size=gru_hidden_size,
        dropout_rate=dropout_rate,
        obs_input_type=obs_input_type
    )
    
    # Train the goal inference network
    print("Training Goal Inference Network on Receding Horizon data...")
    losses, trained_state, log_dir, best_loss, best_epoch = train_goal_inference_network_rh(
        goal_model, training_samples, validation_samples, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate,
        batch_size=batch_size, 
        goal_loss_weight=goal_loss_weight,
        obs_input_type=obs_input_type
    )
    
    # Save training results
    training_results = {
        'training_data_type': 'receding_horizon_trajectories',
        'sliding_window_approach': True,
        'N_agents': N_agents,
        'T_observation': T_observation,
        'T_total': T_total,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'goal_loss_weight': goal_loss_weight,
        'num_epochs': num_epochs,
        'samples_per_trajectory': samples_per_trajectory,
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
    print(f"The sliding window approach makes it robust to different trajectory segments.")
