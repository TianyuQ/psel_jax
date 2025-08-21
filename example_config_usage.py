#!/usr/bin/env python3
"""
Example Script: How to Use the Configuration System

This script demonstrates how to use the centralized configuration system
across different components of the PSN project.
"""

from config_loader import load_config, get_device_config, setup_jax_config, create_log_dir, get_data_paths
import jax
import jax.numpy as jnp


def main():
    """Demonstrate configuration usage."""
    print("=" * 60)
    print("PSN Project Configuration System Demo")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    print("✓ Configuration loaded successfully!")
    
    # Setup JAX according to configuration
    setup_jax_config()
    device = get_device_config()
    print(f"✓ JAX setup complete. Using device: {device}")
    
    # Access game parameters
    print(f"\n📊 Game Parameters:")
    print(f"  • Number of agents: {config.game.N_agents}")
    print(f"  • Time steps: {config.game.T_steps}")
    print(f"  • Time step size: {config.game.dt}")
    print(f"  • State dimension: {config.game.state_dim}")
    print(f"  • Environment radius: {config.game.radius}")
    
    # Access optimization parameters
    print(f"\n⚙️  Optimization Parameters:")
    print(f"  • iLQGames iterations: {config.optimization.num_iters}")
    print(f"  • Step size: {config.optimization.step_size}")
    print(f"  • Navigation weight: {config.optimization.navigation_weight}")
    print(f"  • Collision weight: {config.optimization.collision_weight}")
    print(f"  • Control weight: {config.optimization.control_weight}")
    
    # Access PSN training parameters
    print(f"\n🧠 PSN Training Parameters:")
    print(f"  • Learning rate: {config.psn.learning_rate}")
    print(f"  • Batch size: {config.psn.batch_size}")
    print(f"  • Number of epochs: {config.psn.num_epochs}")
    print(f"  • Hidden dimensions: {config.psn.hidden_dims}")
    print(f"  • Loss weights (σ1, σ2): {config.psn.sigma1}, {config.psn.sigma2}")
    
    # Access goal inference parameters
    print(f"\n🎯 Goal Inference Parameters:")
    print(f"  • Learning rate: {config.goal_inference.learning_rate}")
    print(f"  • Observation length: {config.goal_inference.observation_length}")
    print(f"  • Hidden dimensions: {config.goal_inference.hidden_dims}")
    print(f"  • Number of epochs: {config.goal_inference.num_epochs}")
    
    # Demonstrate path management
    print(f"\n📁 Path Management:")
    paths = get_data_paths(config)
    for name, path in paths.items():
        print(f"  • {name}: {path}")
    
    # Demonstrate log directory creation
    print(f"\n📝 Log Directory Creation:")
    goal_log_dir = create_log_dir("goal_inference", config)
    psn_log_dir = create_log_dir("psn", config)
    ref_log_dir = create_log_dir("reference_generation", config)
    
    print(f"  • Goal inference logs: {goal_log_dir}")
    print(f"  • PSN training logs: {psn_log_dir}")
    print(f"  • Reference generation logs: {ref_log_dir}")
    
    # Demonstrate accessing nested configuration with defaults
    print(f"\n🔧 Configuration Access with Defaults:")
    debug_mode = config.get('debug.debug_mode', False)
    verbose = config.get('debug.verbose_logging', False)
    memory_profiling = config.get('scalability.memory_profiling', True)
    
    print(f"  • Debug mode: {debug_mode}")
    print(f"  • Verbose logging: {verbose}")
    print(f"  • Memory profiling: {memory_profiling}")
    
    # Example of using configuration in a typical training setup
    print(f"\n🚀 Example Training Setup:")
    print("Setting up training with configuration...")
    
    # Extract key parameters
    n_agents = config.game.N_agents
    batch_size = config.psn.batch_size
    learning_rate = config.psn.learning_rate
    
    # Create a simple cost matrix using config weights
    nav_weight = config.optimization.navigation_weight
    ctrl_weight = config.optimization.control_weight
    Q = nav_weight * jnp.eye(config.game.state_dim)
    R = ctrl_weight * jnp.eye(config.game.control_dim)
    
    print(f"  • Cost matrices created: Q shape {Q.shape}, R shape {R.shape}")
    print(f"  • Ready for {n_agents}-agent training with batch size {batch_size}")
    
    print(f"\n✅ Configuration system demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
