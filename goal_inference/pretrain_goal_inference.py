#!/usr/bin/env python3
"""
Goal Inference Network Pretraining

This script pretrains a goal inference network for all agents using reference trajectories.
The network learns to predict goal positions from observation trajectories (first 10 steps).

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
T_reference = config.game.T_total
state_dim = config.game.state_dim
goal_dim = 2  # Goal dimension (x, y) - not in config, keeping as constant

# Training parameters
num_epochs = config.goal_inference.num_epochs
learning_rate = config.goal_inference.learning_rate
batch_size = config.goal_inference.batch_size
goal_loss_weight = config.goal_inference.goal_loss_weight

# Network architecture parameters
# Set hidden dimensions based on number of agents from config
if N_agents == 4:
    hidden_dims = config.goal_inference.hidden_dims_4p
elif N_agents == 10:
    hidden_dims = config.goal_inference.hidden_dims_10p
else:
    # Default fallback to 4p dimensions
    hidden_dims = config.goal_inference.hidden_dims_4p
    print(f"Warning: Using 4p hidden dimensions {hidden_dims} for {N_agents} agents")

hidden_dim = hidden_dims[0]  # Use first hidden dimension
dropout_rate = config.goal_inference.dropout_rate

# Data splitting
validation_split = 0.25  # 1/4 for validation - not in config, keeping as constant

# File paths - use training data directory for training
reference_dir = config.training.data_dir

# Random seed
random_seed = config.training.seed

# Testing parameters
num_eval_samples = config.testing.num_test_samples
num_vis_samples = min(5, num_eval_samples)  # Number of samples for visualization

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

# Device selection - Always use GPU for training
# Force GPU usage
gpu_devices = jax.devices("gpu")
if gpu_devices:
    device = gpu_devices[0]
    print(f"Using GPU: {device}")
    # JAX platform is configured via setup_jax_config()
    print("JAX platform configured via config")
    
    # Test GPU functionality with a simple operation
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
            if i < len(self.hidden_dims) - 1:  # Don't apply dropout to last layer
                x = nn.Dropout(rate=self.dropout_rate, name=f'goal_dropout_{i}')(x, deterministic=deterministic)
        
        # Goal prediction output
        goals = nn.Dense(features=self.goal_output_dim, name='goal_output')(x)
        # No activation for goals - they can be any real values
        
        return goals


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def goal_prediction_loss(predicted_goals: jnp.ndarray, true_goals: jnp.ndarray) -> jnp.ndarray:
    """
    Goal prediction loss using Huber loss for robustness.
    
    Args:
        predicted_goals: Predicted goal positions (batch_size, N_agents * goal_dim)
        true_goals: True goal positions (batch_size, N_agents * goal_dim)
        
    Returns:
        Goal prediction loss value
    """
    # Reshape to (batch_size, N_agents, goal_dim) for per-agent computation
    batch_size = predicted_goals.shape[0]
    predicted_goals_reshaped = predicted_goals.reshape(batch_size, N_agents, goal_dim)
    true_goals_reshaped = true_goals.reshape(batch_size, N_agents, goal_dim)
    
    # Compute Huber loss for each agent's goal prediction
    goal_diff = predicted_goals_reshaped - true_goals_reshaped  # (batch_size, N_agents, goal_dim)
    
    # Huber loss parameters
    delta = 1.0
    
    # Compute Huber loss: more robust than MSE for outliers
    abs_diff = jnp.abs(goal_diff)
    quadratic = jnp.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    huber_loss_per_dim = 0.5 * quadratic**2 + delta * linear
    
    # Sum over x,y dimensions for each agent
    per_agent_error = jnp.sum(huber_loss_per_dim, axis=2)  # (batch_size, N_agents)
    
    # Take mean over N agents for each sample, then mean over all samples
    mean_per_sample = jnp.mean(per_agent_error, axis=1)  # (batch_size,) - mean over agents
    total_loss = jnp.mean(mean_per_sample)  # scalar - mean over samples
    
    return total_loss


# ============================================================================
# REFERENCE TRAJECTORY LOADING
# ============================================================================

def load_and_split_reference_trajectories(data_dir: str, validation_split: float = 0.25) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load reference trajectories from directory containing individual JSON files and split into training/validation sets.
    
    Args:
        data_dir: Path to directory containing ref_traj_sample_*.json files
        validation_split: Fraction of data to use for validation (ignored, using fixed split)
        
    Returns:
        training_data: List of training samples (first 384 samples)
        validation_data: List of validation samples (later 128 samples)
    """
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
    
    total_samples = len(reference_data)
    train_size = config.training.train_samples
    val_size = config.training.test_samples
    
    # Check if we have enough samples
    if total_samples < train_size + val_size:
        print(f"Warning: Only {total_samples} samples available, but need {train_size + val_size} samples")
        print(f"Using all available samples: {total_samples} for training, 0 for validation")
        training_data = reference_data
        validation_data = []
    else:
        # Use first 384 samples for training, later 128 samples for testing
        # Sort by sample_id to ensure consistent ordering
        reference_data.sort(key=lambda x: x.get('sample_id', 0))
        
        training_data = reference_data[:train_size]
        validation_data = reference_data[train_size:train_size + val_size]
    
    print(f"Loaded {total_samples} reference trajectory samples")
    print(f"Training samples: {len(training_data)} (first {len(training_data)} samples)")
    print(f"Validation samples: {len(validation_data)} (samples {len(training_data)} to {len(training_data) + len(validation_data) - 1})")
    
    return training_data, validation_data


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


def extract_reference_goals(sample_data: Dict[str, Any]) -> jnp.ndarray:
    """
    Extract goal positions for all agents from reference data.
    
    Args:
        sample_data: Reference trajectory sample
        
    Returns:
        goals: Goal positions for all agents (N_agents, goal_dim)
    """
    return jnp.array(sample_data["target_positions"])  # (N_agents, goal_dim)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_train_state(model: nn.Module, learning_rate: float) -> train_state.TrainState:
    """
    Create training state for the model.
    
    Args:
        model: Goal inference model
        learning_rate: Learning rate for optimization
        
    Returns:
        Train state
    """
    # Create optimizer with weight decay (AdamW)
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=5e-4  # L2 regularization to prevent overfitting
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
    """
    Single training step.
    
    Args:
        state: Current train state
        batch: Tuple of (observations, true_goals)
        goal_loss_weight: Weight for goal prediction loss
        rng: Random key for dropout operations
        
    Returns:
        Updated train state, total loss value, and goal prediction loss value
    """
    observations, true_goals = batch
    
    def loss_fn(params):
        # Apply the model with the given parameters and random key for dropout
        predicted_goals = state.apply_fn({'params': params}, observations, rngs={'dropout': rng}, deterministic=False)
        
        # Compute goal prediction loss
        goal_loss_val = goal_prediction_loss(predicted_goals, true_goals)
        
        # Total loss
        total_loss_val = goal_loss_weight * goal_loss_val
        
        return total_loss_val, goal_loss_val
    
    # Compute gradients
    (loss, goal_loss_val), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    return state, loss, goal_loss_val


def prepare_batch(batch_data: List[Dict[str, Any]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Prepare a batch of data for training.
    
    Args:
        batch_data: List of data samples
        
    Returns:
        observations: Batch of observations (batch_size, T_observation * N_agents * state_dim)
        true_goals: Batch of true goals (batch_size, N_agents * goal_dim)
    """
    batch_obs = []
    batch_true_goals = []
    
    for sample_data in batch_data:
        # Extract observation trajectory (first 10 steps of all agents)
        obs_traj = extract_observation_trajectory(sample_data)
        batch_obs.append(obs_traj.flatten())  # Flatten to 1D
        
        # Extract true goals for this sample
        true_goals = extract_reference_goals(sample_data).flatten()  # Flatten to (N_agents * goal_dim)
        batch_true_goals.append(true_goals)
    
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
                  validation_data: List[Dict[str, Any]], 
                  batch_size: int) -> float:
    """
    Evaluate the model on validation data for one epoch.
    
    Args:
        goal_model: Goal inference network
        state: Current model state
        validation_data: Validation data samples
        batch_size: Batch size for evaluation
        
    Returns:
        Average validation loss
    """
    val_losses = []
    
    # Create batches for validation
    num_batches = (len(validation_data) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(validation_data))
        batch_data = validation_data[start_idx:end_idx]
        
        # Prepare batch
        observations, true_goals = prepare_batch(batch_data)
        
        # Forward pass (deterministic, no dropout)
        predicted_goals = state.apply_fn({'params': state.params}, observations, deterministic=True)
        
        # Compute loss
        goal_loss = goal_prediction_loss(predicted_goals, true_goals)
        val_losses.append(float(goal_loss))
    
    return np.mean(val_losses)


def train_goal_inference_network(goal_model: GoalInferenceNetwork, 
                                training_data: List[Dict[str, Any]],
                                validation_data: List[Dict[str, Any]],
                                num_epochs: int = 50, 
                                learning_rate: float = 0.001,
                                batch_size: int = 32,
                                goal_loss_weight: float = 1.0) -> Tuple[List[float], train_state.TrainState, str, float, int]:
    """
    Train the goal inference network with training and validation.
    
    Args:
        goal_model: Goal inference network to train
        training_data: Training data samples
        validation_data: Validation data samples
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        batch_size: Batch size for training
        goal_loss_weight: Weight for goal prediction loss
        
    Returns:
        losses: List of training losses per epoch
        trained_state: Final trained state
        log_dir: Directory where logs are saved
        best_loss: Best loss achieved during training
        best_epoch: Epoch where best loss was achieved
    """
    print(f"Training Goal Inference Network with {len(training_data)} training samples and {len(validation_data)} validation samples...")
    print(f"Parameters: epochs={num_epochs}, lr={learning_rate}, batch_size={batch_size}")
    print(f"Observation steps: {T_observation}, Reference steps: {T_reference}")
    
    # Create log directory
    log_dir = create_log_dir("goal_inference", config)
    
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
        random.shuffle(training_data)
        
        # Create batches
        num_batches = (len(training_data) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(training_data))
            batch_data = training_data[start_idx:end_idx]
            
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
        val_loss = evaluate_epoch(goal_model, state, validation_data, batch_size)
        
        # Update progress bar
        progress_bar.set_postfix({'Avg Loss': f'{avg_loss:.4f}', 'Val Loss': f'{val_loss:.4f}'})
        
        # Log to TensorBoard (only goal prediction loss over epochs)
        writer.add_scalar('Loss/Training', avg_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            save_model(state, log_dir, "goal_inference_best_model.pkl")
            print(f"New best model found at epoch {best_epoch} with validation loss: {best_loss:.4f}")
            print(f"Best model saved to: {log_dir}/goal_inference_best_model.pkl")
        

    
    # Close TensorBoard writer
    writer.close()
    
    # Save final model
    save_model(state, log_dir, "goal_inference_final_model.pkl")
    
    # Save training configuration
    save_training_config(log_dir, num_epochs, learning_rate, batch_size, goal_loss_weight)
    
    print(f"\nTraining completed!")
    print(f"Log directory: {log_dir}")
    print(f"Best model saved to: {log_dir}/goal_inference_best_model.pkl")
    print(f"Training results saved to: {log_dir}/training_results.json")
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
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'goal_loss_weight': goal_loss_weight,
        'N_agents': N_agents,
        'T_observation': T_observation,
        'T_reference': T_reference,
        'state_dim': state_dim,
        'goal_dim': goal_dim,
        'validation_split': validation_split
    }
    
    training_results_path = os.path.join(log_dir, 'training_results.json')
    with open(training_results_path, 'w') as f:
        json.dump(training_results, f, indent=2)



def evaluate_goal_inference_model(model: nn.Module, trained_state: train_state.TrainState, 
                                reference_data: List[Dict[str, Any]], num_samples: int = 20) -> Dict[str, float]:
    """
    Evaluate the trained goal inference model on goal prediction accuracy.
    
    Args:
        model: Goal inference model
        trained_state: Trained model state
        reference_data: Reference trajectory data
        num_samples: Number of samples to evaluate
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\nEvaluating Goal Inference model on {num_samples} samples...")
    
    goal_prediction_errors = []
    
    # Sample random indices
    rng = jax.random.PRNGKey(random_seed)
    rng, sample_key = jax.random.split(rng)
    sample_indices = jax.random.choice(sample_key, len(reference_data), shape=(num_samples,), replace=False)
    
    for idx in sample_indices:
        sample_data = reference_data[idx]
        
        # Extract observation trajectory
        obs_traj = extract_observation_trajectory(sample_data)
        obs_input = obs_traj.flatten().reshape(1, -1)  # Add batch dimension
        
        # Get model predictions
        predicted_goals = trained_state.apply_fn({'params': trained_state.params}, obs_input, deterministic=True)
        
        # Extract true goals
        true_goals = extract_reference_goals(sample_data).flatten()
        
        # Compute goal prediction error
        goal_error = jnp.mean(jnp.square(predicted_goals[0] - true_goals))
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


def visualize_goal_predictions(model: nn.Module, trained_state: train_state.TrainState,
                              reference_data: List[Dict[str, Any]], num_samples: int = 5, 
                              save_dir: str = None) -> None:
    """
    Visualize goal predictions vs true goals for selected samples.
    
    Args:
        model: Goal inference model
        trained_state: Trained model state
        reference_data: Reference trajectory data
        num_samples: Number of samples to visualize
        save_dir: Directory to save plots
    """
    print(f"\nVisualizing goal predictions for {num_samples} samples...")
    
    # Sample random indices
    rng = jax.random.PRNGKey(random_seed)
    rng, sample_key = jax.random.split(rng)
    sample_indices = jax.random.choice(sample_key, len(reference_data), shape=(num_samples,), replace=False)
    
    for i, idx in enumerate(sample_indices):
        sample_data = reference_data[idx]
        
        # Extract observation trajectory
        obs_traj = extract_observation_trajectory(sample_data)
        obs_input = obs_traj.flatten().reshape(1, -1)  # Add batch dimension
        
        # Get model predictions
        predicted_goals = trained_state.apply_fn({'params': trained_state.params}, obs_input, deterministic=True)
        
        # Extract true goals and initial positions
        true_goals = extract_reference_goals(sample_data)
        init_positions = jnp.array(sample_data["init_positions"])
        
        # Reshape predicted goals
        predicted_goals = predicted_goals[0].reshape(N_agents, goal_dim)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_aspect('equal')
        ax.set_xlim(-boundary_size, boundary_size)
        ax.set_ylim(-boundary_size, boundary_size)
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
            plot_path = os.path.join(save_dir, f"goal_predictions_sample_{i+1:03d}.{plot_format}")
            plt.savefig(plot_path, dpi=plot_dpi, bbox_inches='tight')
            print(f"Goal prediction plot saved to: {plot_path}")
        
        # Always close the plot to free memory
        plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Goal Inference Network Pretraining")
    print("=" * 60)
    print(f"Configuration loaded from: config.yaml")
    print(f"Game parameters: N_agents={N_agents}, T_observation={T_observation}, T_reference={T_reference}")
    print(f"Training parameters: epochs={num_epochs}, lr={learning_rate}, batch_size={batch_size}")
    print(f"Network parameters: hidden_dim={hidden_dim}, dropout_rate={dropout_rate}")
    print("=" * 60)
    
    # Load reference trajectories
    print(f"Loading reference trajectories from directory: {reference_dir}...")
    training_data, validation_data = load_and_split_reference_trajectories(reference_dir, validation_split)
    
    # Create goal inference model with appropriate hidden dimensions
    goal_model = GoalInferenceNetwork(hidden_dims=hidden_dims)
    
    # Train the goal inference network
    print("Training Goal Inference Network...")
    losses, trained_state, log_dir, best_loss, best_epoch = train_goal_inference_network(
        goal_model, training_data, validation_data, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate,
        batch_size=batch_size, 
        goal_loss_weight=goal_loss_weight
    )
    
    # Save best trained model
    best_model_bytes = flax.serialization.to_bytes(trained_state)
    best_model_path = os.path.join(log_dir, "goal_inference_best_model.pkl")
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model_bytes, f)
    
    # Save training config with additional training results
    training_results = {
        'N_agents': N_agents,
        'T_observation': T_observation,
        'T_reference': T_reference,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'goal_loss_weight': goal_loss_weight,
        'num_epochs': num_epochs,
        'final_loss': float(losses[-1]),
        'best_loss': float(best_loss),
        'best_epoch': int(best_epoch + 1),
        'timestamp': datetime.now().isoformat(),
        'config_source': 'config.yaml'
    }
    config_path = os.path.join(log_dir, "training_results.json")
    with open(config_path, 'w') as f:
        json.dump(training_results, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Log directory: {log_dir}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training results saved to: {config_path}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Best loss: {best_loss:.4f} (achieved at epoch {best_epoch+1})")
    
    # Evaluate the trained model
    evaluation_metrics = evaluate_goal_inference_model(goal_model, trained_state, validation_data, num_samples=num_eval_samples)
    
    # Save evaluation metrics
    metrics_path = os.path.join(log_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    print(f"Evaluation metrics saved to: {metrics_path}")
    
    # Visualize goal predictions
    visualize_goal_predictions(goal_model, trained_state, validation_data, num_samples=num_vis_samples, save_dir=log_dir)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Goal Inference Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plot_path = os.path.join(log_dir, f"training_loss.{plot_format}")
    plt.savefig(plot_path, dpi=plot_dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Training loss plot saved to: {plot_path}")
    print(f"\nTo view TensorBoard logs, run: tensorboard --logdir={log_dir}")
