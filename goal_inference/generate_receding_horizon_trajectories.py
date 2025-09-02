#!/usr/bin/env python3
"""
Generate Receding Horizon Trajectories for All Agents

This script solves receding horizon games to generate trajectories that show
how agents move when solving 50-horizon games and taking only the first step
at each iteration.

It uses the same starting positions and goal positions from the reference trajectories
to ensure consistency.

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
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from lqrax import iLQR

# Import configuration loader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config, get_device_config, setup_jax_config


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
T_receding_horizon_planning = config.game.T_receding_horizon_planning  # Planning horizon for each individual game (50)
T_receding_horizon_iterations = config.game.T_receding_horizon_iterations          # Total number of receding horizon iterations (100)
n_agents = config.game.N_agents
ego_agent_id = config.game.ego_agent_id

# Optimization parameters
num_iters = config.optimization.num_iters
step_size = config.optimization.step_size

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
    
    def dyn(self, xt, ut):
        """Dynamics function for point mass."""
        return jnp.array([
            xt[2],  # dx/dt = vx
            xt[3],  # dy/dt = vy
            ut[0],  # dvx/dt = ax
            ut[1]   # dvy/dt = ay
        ])


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_agent_setup(initial_states: List[jnp.ndarray], target_positions: List[jnp.ndarray]) -> tuple:
    """
    Create a set of agents with their initial states and reference trajectories.
    
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
    
    for i in range(n_agents):
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


def create_loss_functions(agents: list, reference_trajectories: list) -> tuple:
    """
    Create loss functions and their linearizations for all agents.
    
    Args:
        agents: List of agent objects
        reference_trajectories: List of reference trajectories for each agent
    
    Returns:
        Tuple of (loss_functions, linearize_loss_functions, compiled_functions)
    """
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
        
        # Compile functions with GPU optimizations
        compiled_loss = jit(trajectory_loss, device=device)
        compiled_linearize = jit(linearize_loss, device=device)
        compiled_linearize_dyn = jit(agent.linearize_dyn, device=device)
        compiled_solve = jit(agent.solve, device=device)
        
        loss_functions.append(trajectory_loss)
        linearize_loss_functions.append(trajectory_loss)
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
    control_trajectories = [jnp.zeros((T_receding_horizon_planning, 2)) for _ in range(n_agents)]
    
    # Track losses for debugging
    total_losses = []
    
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
                state_trajectories[i], control_trajectories[i], reference_trajectories[i], other_states)
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


def solve_ilqgames(agents: list, 
                   initial_states: list,
                   reference_trajectories: list,
                   compiled_functions: list) -> tuple:
    """
    Solve the iLQGames problem for multiple agents using original iterative approach.
    """
    return solve_ilqgames_iterative(agents, initial_states, reference_trajectories, compiled_functions)


def solve_receding_horizon_game(agents: list, 
                               current_states: list,
                               target_positions: List[jnp.ndarray],
                               compiled_functions: list) -> tuple:
    """
    Solve a single receding horizon game (50-horizon) and return the first control.
    
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
    for i in range(n_agents):
        start_pos = current_states[i][:2]  # Extract x, y position
        target_pos = target_positions[i]
        # Linear interpolation over planning horizon
        ref_traj = jnp.linspace(start_pos, target_pos, T_receding_horizon_planning)
        current_reference_trajectories.append(ref_traj)
    
    # Solve the 50-horizon game
    state_trajectories, control_trajectories, total_time = solve_ilqgames(
        agents, current_states, current_reference_trajectories, compiled_functions)
    
    # Extract the first control from each control trajectory
    first_controls = []
    for i in range(n_agents):
        if len(control_trajectories[i]) > 0:
            first_control = control_trajectories[i][0]  # First control from the computed trajectory
            first_controls.append(first_control)
        else:
            first_controls.append(jnp.zeros(2))  # Fallback to zero control
    
    return first_controls, state_trajectories, total_time


def generate_receding_horizon_trajectory(agents: list,
                                       initial_states: list,
                                       target_positions: List[jnp.ndarray],
                                       compiled_functions: list) -> tuple:
    """
    Generate receding horizon trajectory by solving T_receding_horizon_planning-horizon games and applying first controls.
    
    Receding horizon process:
    1. At each iteration, solve a game with horizon T_receding_horizon_planning (e.g., 50 steps)
    2. Extract only the first control/step from the solution
    3. Apply the first control to move agents forward one step
    4. Repeat for T_real iterations (e.g., 100 iterations)
    
    Args:
        agents: List of agent objects
        initial_states: Initial states for each agent
        target_positions: Target positions for each agent
        compiled_functions: Compiled functions for each agent
    
    Returns:
        Tuple of (receding_horizon_trajectories, receding_horizon_states, total_time)
    """
    start_time = time.time()
    
    # Initialize receding horizon trajectories
    receding_horizon_trajectories = [[] for _ in range(n_agents)]  # Full trajectories at each step
    receding_horizon_states = [[] for _ in range(n_agents)]        # States after applying first controls
    
    # Current states (start with initial states)
    current_states = [state.copy() for state in initial_states]
    
    for step in range(T_receding_horizon_iterations):
        
        # Solve the current T_planning-horizon game from current states
        # This solves a game looking ahead T_planning steps (e.g., 50 steps)
        first_controls, full_trajectories, game_time = solve_receding_horizon_game(
            agents, current_states, target_positions, compiled_functions)
        
        # Store the full trajectories for this step
        for i in range(n_agents):
            receding_horizon_trajectories[i].append(full_trajectories[i].tolist())
        
        # Store the current states before applying controls
        for i in range(n_agents):
            receding_horizon_states[i].append(current_states[i].tolist())
        
        # Apply the first control to move agents forward one step
        for i in range(n_agents):
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
            
            # Update current state for next iteration
            current_states[i] = new_state
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return receding_horizon_trajectories, receding_horizon_states, total_time


def save_receding_horizon_sample(sample_id: int, 
                                init_positions: jnp.ndarray, 
                                target_positions: jnp.ndarray, 
                                receding_horizon_trajectories: List[List],
                                receding_horizon_states: List[List]) -> Dict[str, Any]:
    """
    Save a receding horizon trajectory sample to a dictionary format.
    
    Args:
        sample_id: Sample identifier
        init_positions: Initial positions (n_agents, 2)
        target_positions: Target positions (n_agents, 2)
        receding_horizon_trajectories: List of full trajectories for each agent at each step
        receding_horizon_states: List of states for each agent at each step
        
    Returns:
        sample_data: Dictionary containing the receding horizon trajectory data
    """
    sample_data = {
        "sample_id": sample_id,
        "init_positions": init_positions.tolist(),
        "target_positions": target_positions.tolist(),
        "receding_horizon_trajectories": {
            f"agent_{i}": {
                "full_trajectories": receding_horizon_trajectories[i],  # List of 50-step trajectories
                "states": receding_horizon_states[i]                    # List of states at each step
            }
            for i in range(n_agents)
        },
        "metadata": {
            "n_agents": n_agents,
            "T_receding_horizon_iterations": T_receding_horizon_iterations,
            "T_receding_horizon_planning": T_receding_horizon_planning,
            "dt": dt,
            "state_dim": 4,
            "control_dim": 2,
            "num_iters": num_iters,
            "step_size": step_size
        }
    }
    
    return sample_data


def plot_receding_horizon_sample(sample_data: Dict[str, Any], boundary_size: float, save_path: str = None):
    """
    Plot receding horizon trajectories for a single sample with proper boundary scaling.
    
    Args:
        sample_data: Sample data dictionary
        boundary_size: Size of the environment boundary for consistent scaling
        save_path: Path to save the plot (optional)
    """
    init_positions = np.array(sample_data["init_positions"])
    target_positions = np.array(sample_data["target_positions"])
    receding_horizon_data = sample_data["receding_horizon_trajectories"]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # Square figure for equal aspect
    
    # Plot trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    for i in range(n_agents):
        agent_key = f"agent_{i}"
        states = np.array(receding_horizon_data[agent_key]["states"])
        
        # Extract positions from states
        positions = states[:, :2]  # (T_steps, 2)
        
        # Plot accumulated receding horizon trajectory
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
    ax.set_title(f'Receding Horizon Trajectories (N={n_agents}, Boundary=±{boundary_size}m)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Don't show the plot during generation
    plt.close()


def get_boundary_size(n_agents: int) -> float:
    """
    Determine boundary size based on number of agents.
    
    Args:
        n_agents: Number of agents
        
    Returns:
        boundary_size: Size of the square boundary
    """
    if n_agents <= 4:
        return 2.5  # -2.5m to 2.5m for 4 agents
    else:
        return 3.5  # -3.5m to 3.5m for 10+ agents


def load_reference_trajectory_sample(sample_id: int, reference_dir: str) -> tuple:
    """
    Load a reference trajectory sample to get initial positions and target positions.
    
    Args:
        sample_id: Sample identifier
        reference_dir: Directory containing reference trajectories
        
    Returns:
        Tuple of (initial_positions, target_positions, boundary_size)
    """
    reference_path = Path(reference_dir)
    json_filename = f"ref_traj_sample_{sample_id:03d}.json"
    json_path = reference_path / json_filename
    
    if not json_path.exists():
        raise FileNotFoundError(f"Reference trajectory file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        sample_data = json.load(f)
    
    # Extract initial positions and target positions
    init_positions = np.array(sample_data["init_positions"])
    target_positions = np.array(sample_data["target_positions"])
    
    # Determine boundary size based on number of agents
    n_agents = len(init_positions)
    if n_agents <= 4:
        boundary_size = 2.5  # -2.5m to 2.5m for 4 agents
    else:
        boundary_size = 3.5  # -3.5m to 3.5m for 10+ agents
    
    return init_positions, target_positions, boundary_size


def generate_receding_horizon_trajectories(num_samples: int, 
                                         save_dir: str = None) -> List[Dict[str, Any]]:
    """
    Generate receding horizon trajectories by solving receding horizon games.
    
    Args:
        num_samples: Number of trajectory samples to generate
        save_dir: Directory to save receding horizon trajectories
        
    Returns:
        all_samples: List of receding horizon trajectory samples
    """
    all_samples = []
    
    # Set default save directory based on number of agents if not provided
    if save_dir is None:
        save_dir = config.testing.data_dir
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    print(f"Generating {num_samples} receding horizon trajectory samples...")
    print(f"Game parameters: N={n_agents}, Planning horizon={T_receding_horizon_planning}, Total iterations={T_receding_horizon_iterations}, dt={dt}")
    print(f"Optimization parameters: num_iters={num_iters}, step_size={step_size}")
    print(f"Saving to directory: {save_path}")
    print("=" * 80)
    
    for sample_id in range(num_samples):
        if sample_id % 10 == 0 or sample_id < 5:  # Report every 10 samples + first 5
            print(f"Generating sample {sample_id + 1}/{num_samples}...")
        
        start_time = time.time()
        
        try:
            # Load reference trajectory sample to get initial positions and target positions
            # Use training data directory to load reference trajectories for generation  
            reference_dir = config.training.data_dir
            init_positions, target_positions, boundary_size = load_reference_trajectory_sample(sample_id, reference_dir)
            
            # Convert to initial states (add zero velocities)
            initial_states = []
            for pos in init_positions:
                initial_states.append(jnp.array([pos[0], pos[1], 0.0, 0.0]))
            
            # Create agent setup
            agents, reference_trajectories = create_agent_setup(initial_states, target_positions)
            
            # Create loss functions
            loss_functions, linearize_functions, compiled_functions = create_loss_functions(
                agents, reference_trajectories)
            
            # Generate receding horizon trajectories
            receding_horizon_trajectories, receding_horizon_states, total_time = generate_receding_horizon_trajectory(
                agents, initial_states, target_positions, compiled_functions)
            
            # Save sample
            sample_data = save_receding_horizon_sample(
                sample_id, 
                init_positions,
                target_positions, 
                receding_horizon_trajectories, 
                receding_horizon_states
            )
            all_samples.append(sample_data)
            
            # Save individual JSON file
            json_filename = f"receding_horizon_sample_{sample_id:03d}.json"
            json_path = save_path / json_filename
            with open(json_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
            
            # Create and save trajectory plot
            plot_filename = f"receding_horizon_sample_{sample_id:03d}.png"
            plot_path = save_path / plot_filename
            boundary_size = get_boundary_size(n_agents)
            plot_receding_horizon_sample(sample_data, boundary_size, str(plot_path))
        
        except Exception as e:
            print(f"  ✗ Error generating sample {sample_id}: {str(e)}")
            continue
    
    print(f"\nGenerated {len(all_samples)} receding horizon trajectory samples successfully!")
    print(f"Individual files saved to: {save_path}")
    
    return all_samples


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Receding Horizon Trajectory Generation for All Agents")
    print("=" * 80)
    
    # Number of samples to generate (use same as reference generation)
    num_samples = config.reference_generation.num_samples
    
    # Generate receding horizon trajectories
    all_samples = generate_receding_horizon_trajectories(num_samples)
    
    print("\nReceding horizon trajectory generation completed!")
    print(f"Generated {len(all_samples)} samples with {n_agents} agents each.")
    print("Each sample contains:")
    print(f"  - {T_receding_horizon_iterations} receding horizon iterations")
    print(f"  - {T_receding_horizon_planning}-horizon game solving at each iteration")
    print(f"  - First step extraction and state updates")
    print("Use these trajectories for analyzing receding horizon behavior.")
