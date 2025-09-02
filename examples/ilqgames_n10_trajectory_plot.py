#!/usr/bin/env python3
"""
iLQGames N=10 Trajectory Plot

This script solves the iLQGames problem for exactly 10 agents using reciprocal
collision avoidance (1/(x^i-x^j)²) and plots the resulting trajectories.
"""

import jax 
import jax.numpy as jnp 
from jax import vmap, jit, grad
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import random

from lqrax import iLQR

# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================

# Time discretization parameters
dt = 0.05          # Time step size (seconds)
tsteps = 100       # Number of time steps
n_agents = 10      # Fixed number of agents

# Device selection - use GPU if available, otherwise CPU
device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]
print(f"Using device: {device}")

# Optimization parameters
num_iters = 200
step_size = 0.002

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

def generate_random_positions(n_agents: int, radius: float = 2.0) -> tuple:
    """
    Generate random initial positions and target positions for n agents.
    
    Args:
        n_agents: Number of agents
        radius: Radius of the circular area for positioning
    
    Returns:
        Tuple of (initial_positions, target_positions)
    """
    # Generate random angles for initial positions
    angles = np.linspace(0, 2*np.pi, n_agents, endpoint=False)
    np.random.shuffle(angles)
    
    # Generate random angles for target positions (different from initial)
    target_angles = np.linspace(0, 2*np.pi, n_agents, endpoint=False)
    np.random.shuffle(target_angles)
    
    # Add some randomness to avoid perfect symmetry
    initial_positions = []
    target_positions = []
    
    for i in range(n_agents):
        # Initial position
        r1 = radius * (0.5 + 0.5 * np.random.random())
        x1 = r1 * np.cos(angles[i]) + 0.1 * np.random.randn()
        y1 = r1 * np.sin(angles[i]) + 0.1 * np.random.randn()
        initial_positions.append([x1, y1, 0.0, 0.0])  # [x, y, vx, vy]
        
        # Target position (opposite side)
        r2 = radius * (0.5 + 0.5 * np.random.random())
        x2 = r2 * np.cos(target_angles[i]) + 0.1 * np.random.randn()
        y2 = r2 * np.sin(target_angles[i]) + 0.1 * np.random.randn()
        target_positions.append([x2, y2])
    
    return jnp.array(initial_positions), jnp.array(target_positions)


def create_agent_setup() -> tuple:
    """
    Create a set of agents with their initial states and reference trajectories.
    
    Returns:
        Tuple of (agents, initial_states, reference_trajectories, target_positions)
    """
    agents = []
    initial_states = []
    reference_trajectories = []
    
    # Generate random positions
    init_positions, target_positions = generate_random_positions(n_agents)
    
    # Cost function weights (same for all agents)
    Q = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001]))  # State cost weights
    R = jnp.diag(jnp.array([0.01, 0.01]))               # Control cost weights
    
    for i in range(n_agents):
        # Create agent
        agent = PointAgent(dt=dt, x_dim=4, u_dim=2, Q=Q, R=R)
        agents.append(agent)
        
        # Initial state
        initial_states.append(init_positions[i])
        
        # Reference trajectory (linear interpolation to target)
        start_pos = init_positions[i][:2]
        end_pos = target_positions[i]
        # Create proper reference trajectory with correct number of points
        ref_traj = jnp.linspace(start_pos, end_pos, tsteps)
        reference_trajectories.append(ref_traj)
    
    return agents, initial_states, reference_trajectories, target_positions


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
                # Navigation cost
                nav_loss = jnp.sum(jnp.square(xt[:2] - ref_xt[:2]))
                
                # Collision avoidance costs - exponential form
                collision_loss = 0.0
                if len(other_states) > 0:
                    # Stack other states for vectorized computation
                    other_positions = jnp.stack([other_xt[:2] for other_xt in other_states])
                    distances_squared = jnp.sum(jnp.square(xt[:2] - other_positions), axis=1)
                    # Use exponential form: exp(-5 * distance²)
                    collision_loss = jnp.sum(10.0 * jnp.exp(-5.0 * distances_squared)) / (n_agents - 1)
                    # collision_loss = 0.0
                
                # Control cost
                ctrl_loss = 0.1 * jnp.sum(jnp.square(ut * jnp.array([1.0, 0.5])))
                
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


def solve_ilqgames(agents: list, 
                   initial_states: list,
                   reference_trajectories: list,
                   compiled_functions: list) -> tuple:
    """
    Solve the iLQGames problem for multiple agents.
    
    Args:
        agents: List of agent objects
        initial_states: List of initial states for each agent
        reference_trajectories: List of reference trajectories for each agent
        compiled_functions: List of compiled functions for each agent
    
    Returns:
        Tuple of (final_state_trajectories, final_control_trajectories, total_time)
    """
    # Initialize control trajectories
    control_trajectories = [jnp.zeros((tsteps, 2)) for _ in range(n_agents)]
    
    # Track optimization progress
    total_losses = []
    
    start_time = time.time()
    
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
        
        # Step 4: Update control trajectories
        if iter % 20 == 0:
            # Compute total loss
            total_loss = 0.0
            for i in range(n_agents):
                other_states = [state_trajectories[j] for j in range(n_agents) if j != i]
                agent_loss = compiled_functions[i]['loss'](
                    state_trajectories[i], control_trajectories[i],
                    reference_trajectories[i], other_states)
                total_loss += agent_loss
            
            total_losses.append(total_loss)
            print(f'Iteration {iter:3d}/{num_iters} | Total Loss: {total_loss:8.3f}')
        
        # Update controls
        for i in range(n_agents):
            control_trajectories[i] += step_size * control_updates[i]
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return state_trajectories, control_trajectories, total_time


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_trajectories(state_trajectories: list, reference_trajectories: list, 
                     initial_states: list, target_positions: list):
    """
    Plot the optimized trajectories for all agents.
    
    Args:
        state_trajectories: List of state trajectories for each agent
        reference_trajectories: List of reference trajectories for each agent
        initial_states: List of initial states for each agent
        target_positions: List of target positions for each agent
    """
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Color palette for agents
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    # Set up the plot
    ax.set_aspect('equal')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_title(f'iLQGames N={n_agents} - Optimized Trajectories\n(Exponential Collision Avoidance)')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True, alpha=0.3)
    
    # Plot initial positions
    for i in range(n_agents):
        ax.plot(initial_states[i][0], initial_states[i][1], 'o', 
               color=colors[i], markersize=12, label=f'Agent {i+1} Start')
    
    # Plot target positions
    for i in range(n_agents):
        ax.plot(target_positions[i][0], target_positions[i][1], 's', 
               color=colors[i], markersize=12, label=f'Agent {i+1} Target')
    
    # Add legend entries for line styles
    ax.plot([], [], '--', color='gray', linewidth=2, alpha=0.5, label='Reference Trajectories')
    ax.plot([], [], '-', color='gray', linewidth=3, alpha=0.8, label='Optimized Trajectories')
    
    # Plot reference trajectories (dashed lines)
    for i in range(n_agents):
        x_ref = reference_trajectories[i][:, 0]
        y_ref = reference_trajectories[i][:, 1]
        ax.plot(x_ref, y_ref, '--', color=colors[i], linewidth=2, alpha=0.5, 
               label=f'Agent {i+1} Reference' if i == 0 else "")
    
    # Plot optimized trajectories (solid lines)
    for i in range(n_agents):
        x_traj = state_trajectories[i][:, 0]
        y_traj = state_trajectories[i][:, 1]
        ax.plot(x_traj, y_traj, '-', color=colors[i], linewidth=3, alpha=0.8,
               label=f'Agent {i+1} Optimized' if i == 0 else "")
    
    # Plot final positions
    for i in range(n_agents):
        final_pos = state_trajectories[i][-1, :2]
        ax.plot(final_pos[0], final_pos[1], '^', 
               color=colors[i], markersize=14, alpha=0.8)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('ilqgames_n10_trajectories.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_animation(state_trajectories: list, reference_trajectories: list,
                   initial_states: list, target_positions: list):
    """
    Create an animation of the optimized trajectories.
    
    Args:
        state_trajectories: List of state trajectories for each agent
        reference_trajectories: List of reference trajectories for each agent
        initial_states: List of initial states for each agent
        target_positions: List of target positions for each agent
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=120, tight_layout=True)
    
    # Color palette for agents
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    def update(t):
        ax.cla()
        ax.set_aspect('equal')
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_title(f'iLQGames N={n_agents} - Time Step {t}/{tsteps}')
        ax.grid(True, alpha=0.3)
        
        # Plot target positions
        for i in range(n_agents):
            ax.plot(target_positions[i][0], target_positions[i][1], 
                   's', markersize=15, color=colors[i], alpha=0.5, 
                   label=f'Target {i+1}' if t == 0 else "")
        
        # Plot trajectories and current positions
        for i in range(n_agents):
            # Plot trajectory history
            x_traj = state_trajectories[i][:t+1, 0]
            y_traj = state_trajectories[i][:t+1, 1]
            ax.plot(x_traj, y_traj, '-', color=colors[i], linewidth=3, alpha=0.6)
            
            # Plot current position
            if t < len(state_trajectories[i]):
                current_pos = state_trajectories[i][t, :2]
                ax.plot(current_pos[0], current_pos[1], 'o', 
                       color=colors[i], markersize=12, alpha=0.8)
        
        if t == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        return []
    
    ani = animation.FuncAnimation(fig, update, frames=tsteps, interval=50, repeat=True)
    return ani


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    print("=" * 60)
    print(f"iLQGames N={n_agents} Trajectory Optimization")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Number of agents: {n_agents}")
    print(f"Time steps: {tsteps}")
    print(f"Optimization iterations: {num_iters}")
    print(f"Collision avoidance: Exponential form (exp(-5*distance²))")
    print("=" * 60)
    
    # Create agents and setup
    print("Creating agent setup...")
    agents, initial_states, reference_trajectories, target_positions = create_agent_setup()
    
    # Create loss functions
    print("Creating loss functions...")
    loss_functions, linearize_functions, compiled_functions = create_loss_functions(agents)
    
    # Solve the optimization problem
    print("Starting optimization...")
    state_trajectories, control_trajectories, total_time = solve_ilqgames(
        agents, initial_states, reference_trajectories, compiled_functions)
    
    print("=" * 60)
    print("Optimization completed!")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average time per iteration: {total_time/num_iters:.4f} seconds")
    print("=" * 60)
    
    # Plot results
    print("Creating trajectory plots...")
    plot_trajectories(state_trajectories, reference_trajectories, 
                     initial_states, target_positions)
    
    # Create animation
    print("Creating animation...")
    ani = create_animation(state_trajectories, reference_trajectories,
                         initial_states, target_positions)
    plt.show()
    
    # Save animation (optional)
    ani.save('ilqgames_n10_animation.gif', writer='pillow')
    
    print("Results saved to 'ilqgames_n10_trajectories.png'")
    print("Animation saved to 'ilqgames_n10_animation.gif'")