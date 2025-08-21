#!/usr/bin/env python3
"""
Player Selection Network (PSN) Training Script using JAX/Flax

This script implements the training of a Player Selection Network that learns to
select which agents are important for the ego agent to consider during planning.
Based on the mathematical formulation from the paper.

The network outputs a continuous mask M^i ∈ [0,1]^(N-1) that determines which
agents the ego agent should consider in its masked Nash game.

All parameters are loaded from config.yaml for consistency across scripts.
"""

import jax 
import jax.numpy as jnp 
from jax import vmap, jit, grad, random as jax_random
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import optax
import flax.linen as nn
from flax.training import train_state
import jax.random as random
import sys
sys.path.append('..')

from lqrax import iLQR

# Import configuration loader
from config_loader import load_config, get_device_config, setup_jax_config, create_log_dir

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
N_agents = config.game.N_agents
ego_agent_id = config.game.ego_agent_id
K_observation = config.goal_inference.observation_length
T_planning = config.game.T_planning
dt = config.game.dt

# Training parameters
batch_size = config.psn.batch_size
num_epochs = config.psn.num_epochs
learning_rate = config.psn.learning_rate
sigma1 = config.psn.sigma1
sigma2 = config.psn.sigma2
m_threshold = config.psn.mask_threshold

print(f"Configuration loaded:")
print(f"  N agents: {N_agents}, Ego agent: {ego_agent_id}")
print(f"  Observation length: {K_observation}, Planning horizon: {T_planning}")
print(f"  Training: {num_epochs} epochs, batch size: {batch_size}")
print(f"  Loss weights: σ1={sigma1}, σ2={sigma2}")

# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

class PointAgent(iLQR):
    """
    Point mass agent for trajectory optimization.
    
    State: [x, y, vx, vy] - position (x,y) and velocity (vx, vy)
    Control: [ax, ay] - acceleration in x and y directions
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
# PLAYER SELECTION NETWORK
# ============================================================================

class PlayerSelectionNetwork(nn.Module):
    """
    Player Selection Network (PSN) that learns to select important agents.
    
    Two variants:
    - PSN-Full: Input is all agents' past states x_{0:t}
    - PSN-Partial: Input is partial observation h(x_k)
    """
    
    hidden_dim: int = 128
    output_dim: int = N_agents - 1
    
    @nn.compact
    def __call__(self, x):
        """
        Forward pass of the network.
        
        Args:
            x: Input array of shape (batch_size, input_dim)
            
        Returns:
            mask: Continuous mask array of shape (batch_size, N_agents - 1)
        """
        # Encoder
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim // 2)(x)
        x = nn.relu(x)
        
        # Mask predictor
        x = nn.Dense(self.hidden_dim // 4)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        mask = nn.sigmoid(x)  # Output in [0,1]
        
        return mask


class PSNFull(PlayerSelectionNetwork):
    """
    PSN-Full: Takes all agents' past states as input.
    
    Input: x_{0:t} for all agents
    Output: Mask M^i for ego agent
    """
    
    def __init__(self, K_obs: int = K_observation):
        # Input: K_obs * N_agents * state_dim
        input_dim = K_obs * N_agents * 4  # 4 for [x, y, vx, vy]
        super().__init__(input_dim)


class PSNPartial(PlayerSelectionNetwork):
    """
    PSN-Partial: Takes partial observation as input.
    
    Input: h(x_k) for all agents (e.g., only positions)
    Output: Mask M^i for ego agent
    """
    
    def __init__(self, K_obs: int = K_observation):
        # Input: K_obs * N_agents * 2 (only x, y positions)
        input_dim = K_obs * N_agents * 2
        super().__init__(input_dim)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def binary_loss(mask: jnp.ndarray) -> jnp.ndarray:
    """
    Binary loss to encourage mask values to converge to 0 or 1.
    
    L_Binary = sum_j |0.5 - |0.5 - m^ij||
    
    Args:
        mask: Continuous mask array of shape (batch_size, N_agents - 1)
        
    Returns:
        loss: Binary loss value
    """
    return jnp.sum(jnp.abs(0.5 - jnp.abs(0.5 - mask)))


def mask_sparsity_loss(mask: jnp.ndarray) -> jnp.ndarray:
    """
    Mask sparsity loss to encourage considering fewer agents.
    
    L_Mask = ||M^i||_1 / N
    
    Args:
        mask: Continuous mask array of shape (batch_size, N_agents - 1)
        
    Returns:
        loss: Sparsity loss value
    """
    return jnp.sum(mask) / N_agents


def similarity_loss(pred_traj: jnp.ndarray, target_traj: jnp.ndarray) -> jnp.ndarray:
    """
    Similarity loss to preserve trajectory performance.
    
    L_Sim = sum_k ||p_k^i - h(x_k^i)||_2
    
    Args:
        pred_traj: Predicted trajectory from masked game
        target_traj: Target trajectory from full game
        
    Returns:
        loss: Similarity loss value
    """
    return jnp.sum(jnp.linalg.norm(pred_traj - target_traj, axis=-1))


def total_loss(mask: jnp.ndarray, binary_loss_val: jnp.ndarray, 
               sparsity_loss_val: jnp.ndarray, similarity_loss_val: jnp.ndarray,
               sigma1: float = 0.1, sigma2: float = 1.0) -> jnp.ndarray:
    """
    Total loss function for PSN training.
    
    L = L_Binary + sigma1 * L_Mask + sigma2 * L_Sim
    
    Args:
        mask: Continuous mask array
        binary_loss_val: Binary loss value
        sparsity_loss_val: Sparsity loss value
        similarity_loss_val: Similarity loss value
        sigma1: Weight for mask sparsity loss
        sigma2: Weight for similarity loss
        
    Returns:
        total_loss: Combined loss value
    """
    return binary_loss_val + sigma1 * sparsity_loss_val + sigma2 * similarity_loss_val


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_trajectory_data(num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data with agent trajectories.
    
    Args:
        num_samples: Number of training samples to generate
        
    Returns:
        observations: Observation history of shape (num_samples, K_obs, N_agents, state_dim)
        full_trajectories: Full game trajectories of shape (num_samples, T_planning, N_agents, state_dim)
    """
    observations = []
    full_trajectories = []
    
    for _ in range(num_samples):
        # Generate random initial positions
        init_positions = []
        for i in range(N_agents):
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
            vx = np.random.uniform(-0.5, 0.5)
            vy = np.random.uniform(-0.5, 0.5)
            init_positions.append([x, y, vx, vy])
        
        # Generate observation history
        obs_history = []
        current_states = np.array(init_positions)
        
        for k in range(K_observation):
            # Simple dynamics for data generation
            next_states = current_states.copy()
            next_states[:, 0] += current_states[:, 2] * dt  # x += vx * dt
            next_states[:, 1] += current_states[:, 3] * dt  # y += vy * dt
            next_states[:, 2] += np.random.normal(0, 0.1, N_agents)  # Add noise to velocities
            next_states[:, 3] += np.random.normal(0, 0.1, N_agents)
            
            obs_history.append(current_states.copy())
            current_states = next_states
        
        # Generate full game trajectory (simplified)
        full_traj = []
        for t in range(T_planning):
            # Simple trajectory generation
            traj_states = current_states.copy()
            traj_states[:, 0] += np.random.normal(0, 0.05, N_agents)
            traj_states[:, 1] += np.random.normal(0, 0.05, N_agents)
            full_traj.append(traj_states.copy())
        
        observations.append(np.array(obs_history))
        full_trajectories.append(np.array(full_traj))
    
    return np.array(observations), np.array(full_trajectories)


def create_training_data(num_samples: int, network_type: str = "full") -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create training data for PSN training.
    
    Args:
        num_samples: Number of training samples
        network_type: "full" or "partial"
        
    Returns:
        observations: Observation data of shape (num_samples, input_dim)
        full_trajectories: Full game trajectories
    """
    observations = []
    full_trajectories = []
    
    for _ in range(num_samples):
        # Generate random initial positions
        init_positions = []
        for i in range(N_agents):
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
            vx = np.random.uniform(-0.5, 0.5)
            vy = np.random.uniform(-0.5, 0.5)
            init_positions.append([x, y, vx, vy])
        
        # Generate observation history
        obs_history = []
        current_states = np.array(init_positions)
        
        for k in range(K_observation):
            # Simple dynamics for data generation
            next_states = current_states.copy()
            next_states[:, 0] += current_states[:, 2] * dt  # x += vx * dt
            next_states[:, 1] += current_states[:, 3] * dt  # y += vy * dt
            next_states[:, 2] += np.random.normal(0, 0.1, N_agents)  # Add noise to velocities
            next_states[:, 3] += np.random.normal(0, 0.1, N_agents)
            
            obs_history.append(current_states.copy())
            current_states = next_states
        
        # Generate full game trajectory (simplified)
        full_traj = []
        for t in range(T_planning):
            # Simple trajectory generation
            traj_states = current_states.copy()
            traj_states[:, 0] += np.random.normal(0, 0.05, N_agents)
            traj_states[:, 1] += np.random.normal(0, 0.05, N_agents)
            full_traj.append(traj_states.copy())
        
        # Prepare input data
        obs_array = np.array(obs_history)
        if network_type == "full":
            # PSN-Full: Use all state information
            input_data = obs_array.reshape(-1)  # Flatten observation history
        else:
            # PSN-Partial: Use only position information
            input_data = obs_array[:, :, :2].reshape(-1)  # Only x, y positions
        
        observations.append(input_data)
        full_trajectories.append(np.array(full_traj))
    
    return jnp.array(observations), jnp.array(full_trajectories)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_train_state(model: nn.Module, optimizer: optax.GradientTransformation, 
                      input_shape: Tuple[int, ...], rng: jnp.ndarray) -> train_state.TrainState:
    """
    Create initial train state for the model.
    
    Args:
        model: Flax model
        optimizer: Optax optimizer
        input_shape: Shape of input data
        rng: Random key
        
    Returns:
        train_state: Initial train state
    """
    variables = model.init(rng, jnp.ones(input_shape))
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
    )


@jax.jit
def train_step(state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray],
               sigma1: float = 0.1, sigma2: float = 1.0) -> Tuple[train_state.TrainState, jnp.ndarray]:
    """
    Perform a single training step.
    
    Args:
        state: Current train state
        batch: Training batch (observations, full_trajectories)
        sigma1: Weight for mask sparsity loss
        sigma2: Weight for similarity loss
        
    Returns:
        new_state: Updated train state
        loss: Loss value
    """
    observations, full_trajectories = batch
    
    def loss_fn(params):
        # Apply the model with the given parameters
        mask = state.apply_fn({'params': params}, observations)
        
        # Compute losses
        binary_loss_val = binary_loss(mask)
        sparsity_loss_val = mask_sparsity_loss(mask)
        
        # Simplified similarity loss (placeholder)
        similarity_loss_val = jnp.array(0.0)
        
        # Total loss
        total_loss_val = total_loss(mask, binary_loss_val, sparsity_loss_val, 
                                  similarity_loss_val, sigma1, sigma2)
        
        return total_loss_val, mask
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, mask), grads = grad_fn(state.params)
    
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


@jax.jit
def eval_step(state: train_state.TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray],
              sigma1: float = 0.1, sigma2: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform a single evaluation step.
    
    Args:
        state: Current train state
        batch: Evaluation batch (observations, full_trajectories)
        sigma1: Weight for mask sparsity loss
        sigma2: Weight for similarity loss
        
    Returns:
        loss: Loss value
        mask: Predicted mask
    """
    observations, full_trajectories = batch
    
    mask = state.apply_fn({'params': state.params}, observations)
    
    # Compute losses
    binary_loss_val = binary_loss(mask)
    sparsity_loss_val = mask_sparsity_loss(mask)
    similarity_loss_val = jnp.array(0.0)
    
    total_loss_val = total_loss(mask, binary_loss_val, sparsity_loss_val, 
                              similarity_loss_val, sigma1, sigma2)
    
    return total_loss_val, mask


def train_psn(model: nn.Module, train_data: Tuple[jnp.ndarray, jnp.ndarray],
              num_epochs: int = 100, learning_rate: float = 1e-3,
              sigma1: float = 0.1, sigma2: float = 1.0, 
              batch_size: int = 32, rng: jnp.ndarray = None) -> Tuple[List[float], train_state.TrainState]:
    """
    Train the Player Selection Network.
    
    Args:
        model: PSN model to train
        train_data: Training data (observations, full_trajectories)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        sigma1: Weight for mask sparsity loss
        sigma2: Weight for similarity loss
        batch_size: Batch size for training
        rng: Random key for initialization
        
    Returns:
        losses: List of training losses
        final_state: Final train state
    """
    observations, full_trajectories = train_data
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    
    # Initialize model
    input_shape = (batch_size, observations.shape[1])
    state = create_train_state(model, optimizer, input_shape, rng)
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Create batches
        rng, subkey = random.split(rng)
        indices = random.permutation(subkey, len(observations))
        for i in range(0, len(observations), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_obs = observations[batch_indices]
            batch_traj = full_trajectories[batch_indices]
            
            # Pad batch if necessary
            if len(batch_obs) < batch_size:
                pad_size = batch_size - len(batch_obs)
                batch_obs = jnp.pad(batch_obs, ((0, pad_size), (0, 0)), mode='edge')
                batch_traj = jnp.pad(batch_traj, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode='edge')
            
            # Training step
            state, loss = train_step(state, (batch_obs, batch_traj), sigma1, sigma2)
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return losses, state


def evaluate_psn(state: train_state.TrainState, test_data: Tuple[jnp.ndarray, jnp.ndarray],
                sigma1: float = 0.1, sigma2: float = 1.0, 
                batch_size: int = 32) -> Tuple[float, jnp.ndarray]:
    """
    Evaluate the trained PSN model.
    
    Args:
        state: Trained model state
        test_data: Test data (observations, full_trajectories)
        sigma1: Weight for mask sparsity loss
        sigma2: Weight for similarity loss
        batch_size: Batch size for evaluation
        
    Returns:
        avg_loss: Average test loss
        masks: Predicted masks
    """
    observations, full_trajectories = test_data
    
    total_loss = 0.0
    all_masks = []
    num_batches = 0
    
    # Create batches
    for i in range(0, len(observations), batch_size):
        batch_obs = observations[i:i + batch_size]
        batch_traj = full_trajectories[i:i + batch_size]
        
        # Pad batch if necessary
        if len(batch_obs) < batch_size:
            pad_size = batch_size - len(batch_obs)
            batch_obs = jnp.pad(batch_obs, ((0, pad_size), (0, 0)), mode='edge')
            batch_traj = jnp.pad(batch_traj, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode='edge')
        
        # Evaluation step
        loss, mask = eval_step(state, (batch_obs, batch_traj), sigma1, sigma2)
        total_loss += loss
        all_masks.append(mask)
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    masks = jnp.concatenate(all_masks, axis=0)
    
    return avg_loss, masks


def convert_to_binary_mask(mask: jnp.ndarray, threshold: float = 0.5) -> jnp.ndarray:
    """
    Convert continuous mask to binary mask.
    
    Args:
        mask: Continuous mask array
        threshold: Threshold for conversion
        
    Returns:
        binary_mask: Binary mask array
    """
    return (mask > threshold).astype(int)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_loss(losses: List[float], save_path: str = "psn_training_loss.png"):
    """
    Plot training loss over epochs.
    
    Args:
        losses: List of training losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("PSN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_mask_distribution(masks: jnp.ndarray, save_path: str = "psn_mask_distribution.png"):
    """
    Plot distribution of mask values.
    
    Args:
        masks: Predicted masks array
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Convert to numpy for plotting
    masks_np = np.array(masks)
    
    # Plot histogram of mask values
    plt.subplot(2, 2, 1)
    plt.hist(masks_np.flatten(), bins=50, alpha=0.7)
    plt.title("Distribution of Mask Values")
    plt.xlabel("Mask Value")
    plt.ylabel("Frequency")
    
    # Plot average mask per agent
    plt.subplot(2, 2, 2)
    avg_masks = np.mean(masks_np, axis=0)
    plt.bar(range(len(avg_masks)), avg_masks)
    plt.title("Average Mask Value per Agent")
    plt.xlabel("Agent ID")
    plt.ylabel("Average Mask Value")
    
    # Plot binary mask conversion
    plt.subplot(2, 2, 3)
    binary_masks = convert_to_binary_mask(masks)
    binary_counts = np.sum(binary_masks, axis=0)
    plt.bar(range(len(binary_counts)), binary_counts)
    plt.title("Number of Times Each Agent is Selected")
    plt.xlabel("Agent ID")
    plt.ylabel("Selection Count")
    
    # Plot sparsity over time
    plt.subplot(2, 2, 4)
    sparsity = np.mean(masks_np, axis=1)
    plt.plot(sparsity)
    plt.title("Mask Sparsity Over Samples")
    plt.xlabel("Sample")
    plt.ylabel("Average Mask Value")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Player Selection Network (PSN) Training")
    print("=" * 60)
    print(f"Number of agents: {N_agents}")
    print(f"Ego agent ID: {ego_agent_id}")
    print(f"Observation history: {K_observation} steps")
    print(f"Planning horizon: {T_planning} steps")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Set random seeds
    np.random.seed(42)
    # JAX uses PRNGKey instead of seed
    rng = random.PRNGKey(42)
    
    # Generate training data
    print("Generating training data...")
    num_train_samples = 1000
    num_test_samples = 200
    
    train_obs_full, train_traj_full = create_training_data(num_train_samples, "full")
    test_obs_full, test_traj_full = create_training_data(num_test_samples, "full")
    
    train_obs_partial, train_traj_partial = create_training_data(num_train_samples, "partial")
    test_obs_partial, test_traj_partial = create_training_data(num_test_samples, "partial")
    
    # Train PSN-Full
    print("\nTraining PSN-Full...")
    psn_full = PlayerSelectionNetwork()
    rng, rng_full = random.split(rng)
    full_losses, full_state = train_psn(psn_full, (train_obs_full, train_traj_full), 
                                       num_epochs, learning_rate, sigma1, sigma2, batch_size, rng_full)
    
    # Evaluate PSN-Full
    print("\nEvaluating PSN-Full...")
    test_loss, test_masks = evaluate_psn(full_state, (test_obs_full, test_traj_full), 
                                        sigma1, sigma2, batch_size)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Train PSN-Partial
    print("\nTraining PSN-Partial...")
    psn_partial = PlayerSelectionNetwork()
    rng, rng_partial = random.split(rng)
    partial_losses, partial_state = train_psn(psn_partial, (train_obs_partial, train_traj_partial), 
                                             num_epochs, learning_rate, sigma1, sigma2, batch_size, rng_partial)
    
    # Evaluate PSN-Partial
    print("\nEvaluating PSN-Partial...")
    test_loss_partial, test_masks_partial = evaluate_psn(partial_state, (test_obs_partial, test_traj_partial), 
                                                        sigma1, sigma2, batch_size)
    print(f"Test Loss (Partial): {test_loss_partial:.4f}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_training_loss(full_losses, "psn_full_training_loss.png")
    plot_training_loss(partial_losses, "psn_partial_training_loss.png")
    plot_mask_distribution(test_masks, "psn_full_mask_distribution.png")
    plot_mask_distribution(test_masks_partial, "psn_partial_mask_distribution.png")
    
    # Save models using JAX/Flax
    from flax.serialization import to_bytes, from_bytes
    import pickle
    
    # Save full model
    full_model_bytes = to_bytes(full_state.params)
    with open("psn_full_model.pkl", "wb") as f:
        pickle.dump(full_model_bytes, f)
    
    # Save partial model
    partial_model_bytes = to_bytes(partial_state.params)
    with open("psn_partial_model.pkl", "wb") as f:
        pickle.dump(partial_model_bytes, f)
    
    print("\nTraining completed!")
    print("Models saved as 'psn_full_model.pkl' and 'psn_partial_model.pkl'")
    print("Plots saved as PNG files") 