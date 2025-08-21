#!/usr/bin/env python3
"""
iLQGames Scalability Test

This script tests the computational time for solving iLQGames with varying numbers
of players (2-30) to analyze the algorithm's scalability. It creates different
numbers of agents with random initial positions and targets, then measures the
time required to solve the multi-agent optimization problem.

The test helps understand:
1. How computation time scales with the number of agents
2. Memory usage patterns
3. Convergence behavior with different numbers of agents
4. Practical limits of the algorithm
"""

import jax 
import jax.numpy as jnp 
from jax import vmap, jit, grad
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass

from lqrax import iLQR

# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================

# Time discretization parameters
dt = 0.05          # Time step size (seconds)
tsteps = 50        # Reduced for faster testing

# Device selection - use GPU if available, otherwise CPU
device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]
print(f"Using device: {device}")

# Test parameters
min_players = 2
max_players = 30
num_trials = 3     # Number of trials per player count for averaging
num_iters = 50     # Reduced iterations for faster testing

# ============================================================================
# AGENT DEFINITIONS (Simplified for scalability testing)
# ============================================================================

class PointAgent(iLQR):
    """
    Simplified point mass agent for scalability testing.
    
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

def generate_random_positions(n_agents: int, radius: float = 2.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
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


def create_agent_setup(n_agents: int) -> Tuple[List[PointAgent], List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Create a set of agents with their initial states and reference trajectories.
    
    Args:
        n_agents: Number of agents to create
    
    Returns:
        Tuple of (agents, initial_states, reference_trajectories)
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
        ref_traj = jnp.linspace(start_pos, end_pos, tsteps+1)[1:]
        reference_trajectories.append(ref_traj)
    
    return agents, initial_states, reference_trajectories


def create_loss_functions(agents: List[PointAgent]) -> Tuple[List, List, List]:
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
                
                # Collision avoidance costs - vectorized for GPU efficiency
                collision_loss = 0.0
                if len(other_states) > 0:
                    # Stack other states for vectorized computation
                    other_positions = jnp.stack([other_xt[:2] for other_xt in other_states])
                    distances = jnp.sum(jnp.square(xt[:2] - other_positions), axis=1)
                    collision_loss = jnp.sum(10.0 * jnp.exp(-5.0 * distances))
                
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


def solve_ilqgames(agents: List[PointAgent], 
                   initial_states: List[jnp.ndarray],
                   reference_trajectories: List[jnp.ndarray],
                   compiled_functions: List[Dict],
                   num_iters: int = 50,
                   step_size: float = 0.002) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], float]:
    """
    Solve the iLQGames problem for multiple agents.
    
    Args:
        agents: List of agent objects
        initial_states: List of initial states for each agent
        reference_trajectories: List of reference trajectories for each agent
        compiled_functions: List of compiled functions for each agent
        num_iters: Number of optimization iterations
        step_size: Gradient descent step size
    
    Returns:
        Tuple of (final_state_trajectories, final_control_trajectories, total_time)
    """
    n_agents = len(agents)
    
    # Initialize control trajectories
    control_trajectories = [jnp.zeros((tsteps, 2)) for _ in range(n_agents)]
    
    # Track optimization progress
    total_losses = []
    
    start_time = time.time()
    
    # GPU optimization: batch operations where possible
    for iter in range(num_iters + 1):
        # Step 1: Linearize dynamics for all agents (can be parallelized)
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
        if iter % 10 == 0:
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
# SCALABILITY TESTING
# ============================================================================

@dataclass
class TestResult:
    """Data class to store test results."""
    n_agents: int
    total_time: float
    avg_time_per_iter: float
    final_loss: float
    memory_usage: float = 0.0  # Placeholder for memory tracking


def run_scalability_test() -> List[TestResult]:
    """
    Run scalability test for different numbers of agents.
    
    Returns:
        List of TestResult objects with timing and performance data
    """
    results = []
    
    print("Starting iLQGames Scalability Test")
    print("=" * 60)
    print(f"Testing {min_players} to {max_players} agents")
    print(f"Number of trials per test: {num_trials}")
    print(f"Number of iterations per trial: {num_iters}")
    print(f"Device: {device}")
    print("=" * 60)
    
    for n_agents in range(min_players, max_players + 1):
        print(f"\nTesting with {n_agents} agents...")
        
        trial_times = []
        trial_losses = []
        
        for trial in range(num_trials):
            print(f"  Trial {trial + 1}/{num_trials}")
            
            # Create agents and setup
            agents, initial_states, reference_trajectories = create_agent_setup(n_agents)
            loss_functions, linearize_functions, compiled_functions = create_loss_functions(agents)
            
            # Solve the problem
            state_trajectories, control_trajectories, total_time = solve_ilqgames(
                agents, initial_states, reference_trajectories, compiled_functions,
                num_iters=num_iters
            )
            
            # Compute final loss
            final_loss = 0.0
            for i in range(n_agents):
                other_states = [state_trajectories[j] for j in range(n_agents) if j != i]
                agent_loss = compiled_functions[i]['loss'](
                    state_trajectories[i], control_trajectories[i],
                    reference_trajectories[i], other_states)
                final_loss += agent_loss
            
            trial_times.append(total_time)
            trial_losses.append(final_loss)
            
            print(f"    Time: {total_time:.3f}s | Final Loss: {final_loss:.3f}")
        
        # Compute averages
        avg_time = np.mean(trial_times)
        avg_loss = np.mean(trial_losses)
        avg_time_per_iter = avg_time / num_iters
        
        result = TestResult(
            n_agents=n_agents,
            total_time=avg_time,
            avg_time_per_iter=avg_time_per_iter,
            final_loss=avg_loss
        )
        results.append(result)
        
        print(f"  Average Time: {avg_time:.3f}s | Avg Time/Iter: {avg_time_per_iter:.4f}s")
        print(f"  Average Final Loss: {avg_loss:.3f}")
    
    return results


def plot_results(results: List[TestResult]):
    """
    Plot the scalability test results.
    
    Args:
        results: List of TestResult objects
    """
    n_agents_list = [r.n_agents for r in results]
    total_times = [r.total_time for r in results]
    avg_times_per_iter = [r.avg_time_per_iter for r in results]
    final_losses = [r.final_loss for r in results]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Total time vs number of agents
    ax1.plot(n_agents_list, total_times, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Agents')
    ax1.set_ylabel('Total Time (seconds)')
    ax1.set_title('Total Computation Time vs Number of Agents')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time per iteration vs number of agents
    ax2.plot(n_agents_list, avg_times_per_iter, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Number of Agents')
    ax2.set_ylabel('Time per Iteration (seconds)')
    ax2.set_title('Average Time per Iteration vs Number of Agents')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final loss vs number of agents
    ax3.plot(n_agents_list, final_losses, 'go-', linewidth=2, markersize=6)
    ax3.set_xlabel('Number of Agents')
    ax3.set_ylabel('Final Loss')
    ax3.set_title('Final Loss vs Number of Agents')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Log-log plot for complexity analysis
    ax4.loglog(n_agents_list, total_times, 'mo-', linewidth=2, markersize=6, label='Actual')
    
    # Fit theoretical complexity curves
    n_agents_array = np.array(n_agents_list)
    times_array = np.array(total_times)
    
    # O(n²) fit
    coeff_quad = np.polyfit(np.log(n_agents_array), np.log(times_array), 1)
    theoretical_quad = np.exp(coeff_quad[1]) * n_agents_array**coeff_quad[0]
    ax4.loglog(n_agents_list, theoretical_quad, '--', color='red', 
               linewidth=2, label=f'O(n^{coeff_quad[0]:.1f})')
    
    # O(n³) reference
    coeff_cubic = np.polyfit(np.log(n_agents_array), np.log(times_array), 1)
    theoretical_cubic = np.exp(coeff_cubic[1]) * n_agents_array**3
    ax4.loglog(n_agents_list, theoretical_cubic, ':', color='blue', 
               linewidth=2, label='O(n³) reference')
    
    ax4.set_xlabel('Number of Agents')
    ax4.set_ylabel('Total Time (seconds)')
    ax4.set_title('Complexity Analysis (Log-Log)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ilqgames_scalability_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_summary(results: List[TestResult]):
    """
    Print a summary of the scalability test results.
    
    Args:
        results: List of TestResult objects
    """
    print("\n" + "=" * 60)
    print("SCALABILITY TEST SUMMARY")
    print("=" * 60)
    
    print(f"{'Agents':<8} {'Time(s)':<10} {'Time/Iter(s)':<15} {'Final Loss':<12}")
    print("-" * 50)
    
    for result in results:
        print(f"{result.n_agents:<8} {result.total_time:<10.3f} {result.avg_time_per_iter:<15.4f} {result.final_loss:<12.3f}")
    
    # Analyze scaling
    if len(results) >= 2:
        first_time = results[0].total_time
        last_time = results[-1].total_time
        first_agents = results[0].n_agents
        last_agents = results[-1].n_agents
        
        scaling_factor = (last_time / first_time) / ((last_agents / first_agents) ** 2)
        
        print("\n" + "-" * 50)
        print(f"Scaling Analysis:")
        print(f"  Time increased by factor: {last_time/first_time:.1f}x")
        print(f"  Agents increased by factor: {last_agents/first_agents:.1f}x")
        print(f"  Scaling factor (time/n²): {scaling_factor:.2f}")
        
        if scaling_factor < 1.5:
            print("  Conclusion: Algorithm scales better than O(n²)")
        elif scaling_factor < 3.0:
            print("  Conclusion: Algorithm scales approximately O(n²)")
        else:
            print("  Conclusion: Algorithm scales worse than O(n²)")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Print GPU/CPU information
    print("=" * 60)
    print("iLQGames GPU Scalability Test")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print("=" * 60)
    
    # Run the scalability test
    results = run_scalability_test()
    
    # Plot results
    plot_results(results)
    
    # Print summary
    print_summary(results)
    
    # Save results to file
    with open('ilqgames_scalability_results.txt', 'w') as f:
        f.write("iLQGames Scalability Test Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Test Parameters:\n")
        f.write(f"  Min Players: {min_players}\n")
        f.write(f"  Max Players: {max_players}\n")
        f.write(f"  Trials per test: {num_trials}\n")
        f.write(f"  Iterations per trial: {num_iters}\n")
        f.write(f"  Time steps: {tsteps}\n")
        f.write(f"  Time step size: {dt}\n")
        f.write(f"  Device: {device}\n\n")
        
        f.write("Results:\n")
        f.write(f"{'Agents':<8} {'Time(s)':<10} {'Time/Iter(s)':<15} {'Final Loss':<12}\n")
        f.write("-" * 50 + "\n")
        
        for result in results:
            f.write(f"{result.n_agents:<8} {result.total_time:<10.3f} {result.avg_time_per_iter:<15.4f} {result.final_loss:<12.3f}\n")
    
    print(f"\nResults saved to 'ilqgames_scalability_results.txt'")
    print(f"Plot saved to 'ilqgames_scalability_results.png'") 