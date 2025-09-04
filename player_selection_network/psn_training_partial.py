#!/usr/bin/env python3
"""
Player Selection Network Training with Partial Observations (Position Only)

This script trains a PSN using only position observations (x, y) instead of the 
full state (x, y, vx, vy). This creates a more lightweight model that may be 
more robust to velocity noise and easier to deploy in real-world scenarios.

The network architecture remains the same, but the input dimensions are reduced
from 4 to 2 per agent, making it more efficient and potentially more generalizable.

Author: Assistant
Date: 2024
"""

import sys
import os
# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main PSN training script
from player_selection_network.psn_training_with_pretrained_goals import (
    load_config, get_device_config, setup_jax_config,
    PlayerSelectionNetwork, load_reference_trajectories,
    train_psn_with_pretrained_goals, load_pretrained_goal_model
)

# Import configuration loader
from config_loader import load_config

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PLAYER SELECTION NETWORK TRAINING WITH PARTIAL OBSERVATIONS")
    print("=" * 80)
    print("Training with position-only observations (x, y) instead of full state (x, y, vx, vy)")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Setup JAX configuration
    setup_jax_config()
    
    # Get device from configuration
    device = get_device_config()
    print(f"Using device: {device}")
    
    # Extract parameters from configuration
    N_agents = config.game.N_agents
    T_observation = config.goal_inference.observation_length
    T_total = config.game.T_total
    state_dim = config.game.state_dim
    control_dim = config.game.control_dim
    
    # Training parameters
    num_epochs = config.psn.num_epochs
    learning_rate = config.psn.learning_rate
    batch_size = config.psn.batch_size
    sigma1 = config.psn.sigma1
    sigma2 = config.psn.sigma2
    
    # Network architecture parameters
    psn_hidden_dims = config.psn.hidden_dims_4p if N_agents == 4 else config.psn.hidden_dims_10p
    
    # Data parameters
    reference_dir = config.training.data_dir
    
    # Goal inference model path (for pretrained goals)
    pretrained_goal_model_path = f"log/goal_inference_gru_N_{N_agents}_T_{T_total}_obs_{T_observation}_lr_{config.goal_inference.learning_rate}_bs_{config.goal_inference.batch_size}_goal_loss_weight_{config.goal_inference.goal_loss_weight}_epochs_{config.goal_inference.num_epochs}/goal_inference_best_model.pkl"
    
    print(f"Configuration loaded from: config.yaml")
    print(f"Game parameters: N_agents={N_agents}, T_observation={T_observation}, T_total={T_total}")
    print(f"Training parameters: epochs={num_epochs}, lr={learning_rate}, batch_size={batch_size}")
    print(f"Loss weights: σ1={sigma1}, σ2={sigma2}")
    print(f"Network parameters: hidden_dims={psn_hidden_dims}")
    print(f"Observation type: PARTIAL (position only - x, y)")
    print("=" * 80)
    
    # Load reference trajectories
    print(f"Loading reference trajectories from directory: {reference_dir}")
    training_data, validation_data = load_reference_trajectories(reference_dir)
    
    # Check if we should use true goals or predicted goals
    use_true_goals = config.psn.use_true_goals
    
    if use_true_goals:
        print("Training PSN with TRUE goals (no goal inference model needed)")
        goal_model = None
        goal_trained_state = None
    else:
        print("Training PSN with PREDICTED goals from goal inference model")
        # Load pretrained goal inference model
        goal_model, goal_trained_state = load_pretrained_goal_model(pretrained_goal_model_path)
    
    # Create PSN model with partial observations
    psn_model = PlayerSelectionNetwork(
        hidden_dims=psn_hidden_dims,
        obs_input_type="partial"  # Use partial observations (position only)
    )
    
    # Train PSN with appropriate goal source
    print(f"Observation input type: partial (position only)")
    training_losses, validation_losses, binary_losses, sparsity_losses, similarity_losses, validation_binary_losses, validation_sparsity_losses, validation_similarity_losses, trained_state, log_dir, best_loss, best_epoch = train_psn_with_pretrained_goals(
        psn_model, training_data, validation_data, goal_model, goal_trained_state,
        num_epochs=num_epochs, learning_rate=learning_rate,
        sigma1=sigma1, sigma2=sigma2, batch_size=batch_size,
        use_true_goals=use_true_goals, obs_input_type="partial"  # Use partial observations
    )
    
    # Save final model
    import pickle
    import flax.serialization
    final_model_path = os.path.join(log_dir, "psn_partial_final_model.pkl")
    final_model_bytes = flax.serialization.to_bytes(trained_state)
    with open(final_model_path, 'wb') as f:
        pickle.dump(final_model_bytes, f)
    
    # Save training configuration
    import json
    from datetime import datetime
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
        'final_training_loss': float(training_losses[-1]),
        'final_validation_loss': float(validation_losses[-1]),
        'best_validation_loss': float(best_loss),
        'best_epoch': int(best_epoch + 1),
        'timestamp': datetime.now().isoformat(),
        'config_source': 'config.yaml',
        'obs_input_type': 'partial',
        'description': 'PSN model trained with position-only observations (x, y)',
        'use_true_goals': use_true_goals
    }
    config_path = os.path.join(log_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Log directory: {log_dir}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training configuration saved to: {config_path}")
    print(f"Final training loss: {training_losses[-1]:.4f}")
    print(f"Final validation loss: {validation_losses[-1]:.4f}")
    print(f"Best validation loss: {best_loss:.4f} (achieved at epoch {best_epoch + 1})")
    print(f"\nModel trained with PARTIAL observations (position only: x, y)")
    print(f"This model can be used for player selection when only position data is available.")
