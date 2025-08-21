#!/usr/bin/env python3
"""
Generate Reference Trajectories for PSN Training

This script solves full N-player iLQGames to generate reference trajectories
that will be used for computing similarity loss in PSN training.

All parameters are loaded from config.yaml for consistency across scripts.

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

# Import from the main lqrax module
import sys
import os
# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
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
dt = config.game.dt
tsteps = config.game.T_total
n_agents = config.game.N_agents

# Optimization parameters
num_iters = config.optimization.num_iters
step_size = config.optimization.step_size

# Data generation parameters
num_samples = config.reference_generation.num_samples
save_dir = Path(config.reference_generation.save_dir)
save_dir.mkdir(exist_ok=True)

print(f"Configuration loaded:")
print(f"  N agents: {n_agents}")
print(f"  Time steps: {tsteps}, dt: {dt}")
print(f"  Optimization: {num_iters} iters, step size: {step_size}")
print(f"  Generating {num_samples} samples")


# ============================================================================
# AGENT DEFINITIONS (EXACTLY FROM ilqgames_n10_trajectory_plot.py)
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
# UTILITY FUNCTIONS (EXACTLY FROM ilqgames_n10_trajectory_plot.py)
# ============================================================================

def generate_random_positions(n_agents: int, boundary_size: float = None) -> tuple:
    """
    Generate random initial positions and target positions for n agents.
    
    Args:
        n_agents: Number of agents
        boundary_size: Size of the square boundary for positioning. If None, auto-determined based on n_agents.
    
    Returns:
        Tuple of (initial_positions, target_positions)
    """
    # Determine boundary size based on number of agents if not specified
    if boundary_size is None:
        if n_agents <= 4:
            boundary_size = 2.5  # -2.5m to 2.5m for 4 agents
        else:
            boundary_size = 3.5  # -3.5m to 3.5m for 10+ agents
    
    # Generate random positions within the square boundary
    initial_positions = []
    target_positions = []
    
    for i in range(n_agents):
        # Initial position - random within boundary with some margin from edges
        margin = 0.2  # Keep agents slightly away from boundary edges
        x1 = np.random.uniform(-boundary_size + margin, boundary_size - margin)
        y1 = np.random.uniform(-boundary_size + margin, boundary_size - margin)
        initial_positions.append([x1, y1, 0.0, 0.0])  # [x, y, vx, vy]
        
        # Target position - random within boundary, ensuring some minimum distance from initial
        min_distance = 1.0  # Minimum distance between initial and target
        max_attempts = 50
        
        for attempt in range(max_attempts):
            x2 = np.random.uniform(-boundary_size + margin, boundary_size - margin)
            y2 = np.random.uniform(-boundary_size + margin, boundary_size - margin)
            
            # Check distance from initial position
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance >= min_distance:
                break
        
        target_positions.append([x2, y2])
    
    return jnp.array(initial_positions), jnp.array(target_positions), boundary_size


def create_agent_setup() -> tuple:
    """
    Create a set of agents with their initial states and reference trajectories.
    
    Returns:
        Tuple of (agents, initial_states, reference_trajectories, target_positions)
    """
    agents = []
    initial_states = []
    reference_trajectories = []
    
    # Generate random positions using appropriate boundary size for agent count
    init_positions, target_positions, boundary_size = generate_random_positions(n_agents)
    
    # Cost function weights (same for all agents) - exactly like original ilqgames_example
    Q = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001]))  # State cost weights (position, position, velocity, velocity)
    R = jnp.diag(jnp.array([0.01, 0.01]))               # Control cost weights (ax, ay)
    
    for i in range(n_agents):
        # Create agent
        agent = PointAgent(dt=dt, x_dim=4, u_dim=2, Q=Q, R=R)
        agents.append(agent)
        
        # Initial state
        initial_states.append(init_positions[i])
        
        # Reference trajectory (simple linear interpolation like original example)
        # Create a straight-line reference trajectory from initial position to target
        start_pos = init_positions[i][:2]  # Extract x, y position
        target_pos = target_positions[i]
        
        # Linear interpolation over time steps (exactly like original ilqgames_example)
        ref_traj = jnp.linspace(start_pos, target_pos, tsteps)
        reference_trajectories.append(ref_traj)
    
    return agents, initial_states, reference_trajectories, target_positions, boundary_size


def create_loss_functions(agents: list) -> tuple:
    """
    Create loss functions and their linearizations for all agents.
    
    Args:
        agents: List of agent objects
    
    Returns:
        Tuple of (loss_functions, linearize_loss_functions, compiled_functions)
    """
    loss_functions = []
    linearize_loss_functions = []
    compiled_functions = []
    
    for i, agent in enumerate(agents):
        # Create loss function for this agent
        def create_runtime_loss(agent_idx, agent_obj):
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
        
        runtime_loss = create_runtime_loss(i, agent)
        
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
        
        # Compile functions with GPU optimizations
        compiled_loss = jit(trajectory_loss, device=device)
        compiled_linearize = jit(linearize_loss, device=device)
        compiled_linearize_dyn = jit(agent.linearize_dyn, device=device)
        compiled_solve = jit(agent.solve, device=device)
        
        loss_functions.append(trajectory_loss)
        linearize_loss_functions.append(linearize_loss)
        compiled_functions.append({
            'loss': compiled_loss,
            'linearize_loss': compiled_linearize,
            'linearize_dyn': compiled_linearize_dyn,
            'solve': compiled_solve
        })
    
    return loss_functions, linearize_loss_functions, compiled_functions


def solve_ilqgames_iterative(agents: list, 
                            initial_states: list,
                            reference_trajectories: list,
                            compiled_functions: list) -> tuple:
    """
    Solve the iLQGames problem using the original iterative approach.
    
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
    control_trajectories = [jnp.zeros((tsteps, 2)) for _ in range(n_agents)]
    
    for iter in range(num_iters + 1):
        # Step 1: Linearize dynamics for all agents
        state_trajectories = []
        A_trajectories = []
        B_trajectories = []
        
        for i in range(n_agents):
            x_traj, A_traj, B_traj = compiled_functions[i]['linearize_dyn'](
                initial_states[i], control_trajectories[i])
            state_trajectories.append(x_traj)
            A_trajectories.append(A_traj)
            B_trajectories.append(B_traj)
        
        # Step 2: Linearize loss functions for all agents
        a_trajectories = []
        b_trajectories = []
        
        for i in range(n_agents):
            # Create list of other agents' states for this agent
            other_states = [state_trajectories[j] for j in range(n_agents) if j != i]
            
            a_traj, b_traj = compiled_functions[i]['linearize_loss'](
                state_trajectories[i], control_trajectories[i], 
                reference_trajectories[i], other_states)
            a_trajectories.append(a_traj)
            b_trajectories.append(b_traj)
        
        # Step 3: Solve LQR subproblems for all agents
        control_updates = []
        
        for i in range(n_agents):
            v_traj, _ = compiled_functions[i]['solve'](
                A_trajectories[i], B_trajectories[i], 
                a_trajectories[i], b_trajectories[i])
            control_updates.append(v_traj)
        
        # Update control trajectories with gradient descent
        for i in range(n_agents):
            control_trajectories[i] += step_size * control_updates[i]
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return state_trajectories, control_trajectories, total_time


# Use the original iterative approach for compatibility and correctness
def solve_ilqgames(agents: list, 
                   initial_states: list,
                   reference_trajectories: list,
                   compiled_functions: list) -> tuple:
    """
    Solve the iLQGames problem for multiple agents using original iterative approach.
    """
    return solve_ilqgames_iterative(agents, initial_states, reference_trajectories, compiled_functions)


def save_trajectory_sample(sample_id: int, init_positions: jnp.ndarray, 
                          target_positions: jnp.ndarray, state_trajectories: List[jnp.ndarray],
                          control_trajectories: List[jnp.ndarray]) -> Dict[str, Any]:
    """
    Save a trajectory sample to a dictionary format.
    
    Args:
        sample_id: Sample identifier
        init_positions: Initial positions (n_agents, 2)
        target_positions: Target positions (n_agents, 2)
        state_trajectories: List of state trajectories for each agent
        control_trajectories: List of control trajectories for each agent
        
    Returns:
        sample_data: Dictionary containing the trajectory data
    """
    sample_data = {
        "sample_id": sample_id,
        "init_positions": init_positions.tolist(),  # Convert to list for JSON serialization
        "target_positions": target_positions.tolist(),  # Convert to list for JSON serialization
        "trajectories": {
            f"agent_{i}": {
                "states": state_trajectories[i].tolist(),  # (tsteps, state_dim)
                "controls": control_trajectories[i].tolist()  # (tsteps, control_dim)
            }
            for i in range(n_agents)
        },
        "metadata": {
            "n_agents": n_agents,
            "tsteps": tsteps,
            "dt": dt,
            "state_dim": 4,
            "control_dim": 2
        }
    }
    
    return sample_data


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_sample_trajectories(sample_data: Dict[str, Any], boundary_size: float, save_path: str = None):
    """
    Plot trajectories for a single sample with proper boundary scaling.
    
    Args:
        sample_data: Sample data dictionary
        boundary_size: Size of the environment boundary for consistent scaling
        save_path: Path to save the plot (optional)
    """
    init_positions = np.array(sample_data["init_positions"])
    target_positions = np.array(sample_data["target_positions"])
    trajectories = sample_data["trajectories"]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # Square figure for equal aspect
    
    # Plot trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    for i in range(n_agents):
        agent_key = f"agent_{i}"
        states = np.array(trajectories[agent_key]["states"])
        
        # Extract positions
        positions = states[:, :2]  # (tsteps, 2)
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], 
               color=colors[i], linewidth=2, label=f'Agent {i}', alpha=0.8)
        
        # Plot start and end points
        ax.scatter(init_positions[i, 0], init_positions[i, 1], 
                  color=colors[i], s=120, marker='o', edgecolors='black', 
                  linewidth=2, label=f'Start {i}' if i == 0 else "")
        ax.scatter(target_positions[i, 0], target_positions[i, 1], 
                  color=colors[i], s=120, marker='*', edgecolors='black',
                  linewidth=2, label=f'Goal {i}' if i == 0 else "")
    
    # Set axis limits to show full boundary with small margin
    margin = 0.1
    ax.set_xlim(-boundary_size - margin, boundary_size + margin)
    ax.set_ylim(-boundary_size - margin, boundary_size + margin)
    
    # Draw boundary rectangle
    boundary_rect = plt.Rectangle((-boundary_size, -boundary_size), 
                                 2*boundary_size, 2*boundary_size,
                                 fill=False, edgecolor='gray', linewidth=2, 
                                 linestyle='--', alpha=0.5)
    ax.add_patch(boundary_rect)
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Agent Trajectories (N={n_agents}, Boundary=±{boundary_size}m)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"Plot saved to: {save_path}")
    
    # Don't show the plot during generation
    plt.close()


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_reference_trajectories(num_samples: int, save_dir: str = "reference_trajectories") -> List[Dict[str, Any]]:
    """
    Generate reference trajectories by solving full N-player games.
    
    Args:
        num_samples: Number of trajectory samples to generate
        save_dir: Directory to save individual JSON files
        
    Returns:
        all_samples: List of trajectory samples
    """
    all_samples = []
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    print(f"Generating {num_samples} reference trajectory samples...")
    print(f"Game parameters: N={n_agents}, T={tsteps}, dt={dt}")
    print(f"Saving to directory: {save_path}")
    print("=" * 60)
    
    for sample_id in range(num_samples):
        if sample_id % 10 == 0 or sample_id < 5:  # Report every 10 samples + first 5
            print(f"Generating sample {sample_id + 1}/{num_samples}...")
        start_time = time.time()
        
        # Create agent setup
        agents, initial_states, reference_trajectories, target_positions, boundary_size = create_agent_setup()
        
        # Create loss functions (not used in efficient solver but kept for compatibility)
        loss_functions, linearize_functions, compiled_functions = create_loss_functions(agents)
        
        # Solve iLQGames using efficient JAX implementation
        state_trajectories, control_trajectories, total_time = solve_ilqgames(
            agents, initial_states, reference_trajectories, compiled_functions)
        
        # Save sample
        sample_data = save_trajectory_sample(
            sample_id, 
            jnp.array([initial_states[i][:2] for i in range(n_agents)]),  # Extract positions
            target_positions, 
            state_trajectories, 
            control_trajectories
        )
        all_samples.append(sample_data)
        
        # Save individual JSON file
        json_filename = f"ref_traj_sample_{sample_id:03d}.json"
        json_path = save_path / json_filename
        with open(json_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        # Create and save trajectory plot
        plot_filename = f"ref_traj_sample_{sample_id:03d}.png"
        plot_path = save_path / plot_filename
        plot_sample_trajectories(sample_data, boundary_size, str(plot_path))
        
        elapsed_time = time.time() - start_time
    
    print(f"\nGenerated {len(all_samples)} samples successfully!")
    print(f"Individual files saved to: {save_path}")
    print(f"Use directory scanning to load samples for training/testing.")
    
    return all_samples


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Reference Trajectory Generation for PSN Training")
    print("=" * 60)
    
    # Generate reference trajectories
    all_samples = generate_reference_trajectories(num_samples, save_dir)
    
    # Plot a few examples (already done during generation)
    print("\nReference trajectory generation completed!")
    print(f"Generated {len(all_samples)} samples with {n_agents} agents each.")
    print("Use these trajectories for PSN training similarity loss computation.") 