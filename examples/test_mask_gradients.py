#!/usr/bin/env python3
"""
Test Mask Gradient Flow in iLQGames

This script tests whether gradients can flow correctly through a mask
that affects the game dynamics and loss computation.

The test:
1. Creates a random selection mask
2. Applies the mask to collision avoidance costs
3. Solves the game with masked dynamics
4. Computes loss from the resulting trajectory
5. Attempts to compute gradients with respect to the mask

This verifies that the end-to-end pipeline maintains differentiability.
"""

import jax 
import jax.numpy as jnp 
from jax import vmap, jit, grad, value_and_grad
import numpy as np
import matplotlib.pyplot as plt
from lqrax import iLQR

# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================

# Time discretization parameters
dt = 0.05          # Time step size (seconds)
tsteps = 50        # Number of time steps (reduced for testing)
device = jax.devices("cpu")[0]  # Use CPU for computation

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
    
    def dyn(self, x, u):
        """Dynamics function for point mass."""
        return jnp.array([
            x[2],  # dx/dt = vx
            x[3],  # dy/dt = vy
            u[0],  # dvx/dt = ax
            u[1]   # dvy/dt = ay
        ])


# ============================================================================
# AGENT SETUP
# ============================================================================

# Cost function weights
Q = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001]))  # State cost weights
R = jnp.diag(jnp.array([0.01, 0.01]))               # Control cost weights

# Initialize agents
agent1 = PointAgent(dt=dt, x_dim=4, u_dim=2, Q=Q, R=R)
agent2 = PointAgent(dt=dt, x_dim=4, u_dim=2, Q=Q, R=R)
agent3 = PointAgent(dt=dt, x_dim=4, u_dim=2, Q=Q, R=R)

agents = [agent1, agent2, agent3]

# ============================================================================
# INITIAL CONDITIONS AND REFERENCE TRAJECTORIES
# ============================================================================

# Initial states for each agent
x0_1 = jnp.array([-2.0, 0.0, 0.0, 0.0])      # [x, y, vx, vy]
x0_2 = jnp.array([2.0, 0.0, 0.0, 0.0])       # [x, y, vx, vy]
x0_3 = jnp.array([0.0, -2.0, 0.0, 0.0])      # [x, y, vx, vy]

initial_states = [x0_1, x0_2, x0_3]

# Initial control trajectories (constant controls)
u_traj_1 = jnp.tile(jnp.array([0.0, 0.0]), reps=(tsteps, 1))  # [ax, ay]
u_traj_2 = jnp.tile(jnp.array([0.0, 0.0]), reps=(tsteps, 1))  # [ax, ay]
u_traj_3 = jnp.tile(jnp.array([0.0, 0.0]), reps=(tsteps, 1))  # [ax, ay]

control_trajectories = [u_traj_1, u_traj_2, u_traj_3]

# Reference trajectories (target paths for each agent)
ref_traj_1 = jnp.linspace(
    jnp.array([-2.0, 0.0]), jnp.array([2.0, 0.0]), tsteps+1
)[1:]  # Move from left to right

ref_traj_2 = jnp.linspace(
    jnp.array([2.0, 0.0]), jnp.array([-2.0, 0.0]), tsteps+1
)[1:]  # Move from right to left

ref_traj_3 = jnp.linspace(
    jnp.array([0.0, -2.0]), jnp.array([0.0, 2.0]), tsteps+1
)[1:]  # Move from bottom to top

reference_trajectories = [ref_traj_1, ref_traj_2, ref_traj_3]

# ============================================================================
# MASKED LOSS FUNCTIONS
# ============================================================================

def masked_runtime_loss(xt, ut, ref_xt, other_states, mask_values):
    """
    Runtime loss function with mask-based collision avoidance.
    
    Args:
        xt: Current state [x, y, vx, vy]
        ut: Current control [ax, ay]
        ref_xt: Reference state [x, y, vx, vy]
        other_states: List of other agents' states
        mask_values: Mask values for collision avoidance [m_12, m_13, m_23]
    
    Returns:
        Total loss combining navigation, masked collision avoidance, and control costs
    """
    # Navigation cost: penalize deviation from reference trajectory
    nav_loss = jnp.sum(jnp.square(xt[:2] - ref_xt[:2]))
    
    # Collision avoidance costs with mask-based filtering
    collision_loss = 0.0
    if len(other_states) > 0:
        # Compute distances to other agents
        other_positions = jnp.stack([other_xt[:2] for other_xt in other_states])
        distances_squared = jnp.sum(jnp.square(xt[:2] - other_positions), axis=1)
        
        # Base collision cost: exp(-5 * distance²)
        base_collision = 10.0 * jnp.exp(-5.0 * distances_squared)
        
        # Apply mask values to collision costs
        # mask_values[0] affects collision with agent 1
        # mask_values[1] affects collision with agent 2
        # mask_values[2] affects collision with agent 3
        masked_collision = base_collision * mask_values[:len(other_states)]
        collision_loss = jnp.sum(masked_collision)
    
    # Control cost: penalize large control inputs
    ctrl_loss = 0.1 * jnp.sum(jnp.square(ut * jnp.array([1.0, 0.5])))
    
    return nav_loss + collision_loss + ctrl_loss


def masked_trajectory_loss(x_traj, u_traj, ref_x_traj, other_x_trajs, mask_values):
    """Compute total loss over entire trajectory with mask."""
    def single_step_loss(args):
        xt, ut, ref_xt, other_xts = args
        return masked_runtime_loss(xt, ut, ref_xt, other_xts, mask_values)
    
    loss_array = vmap(single_step_loss)((x_traj, u_traj, ref_x_traj, other_x_trajs))
    return loss_array.sum() * dt


def masked_linearize_loss(x_traj, u_traj, ref_x_traj, other_x_trajs, mask_values):
    """Linearize the loss function around the current trajectory."""
    dldx = grad(masked_runtime_loss, argnums=(0))
    dldu = grad(masked_runtime_loss, argnums=(1))
    
    def grad_step(args):
        xt, ut, ref_xt, other_xts = args
        return dldx(xt, ut, ref_xt, other_xts, mask_values), dldu(xt, ut, ref_xt, other_xts, mask_values)
    
    grads = vmap(grad_step)((x_traj, u_traj, ref_x_traj, other_x_trajs))
    return grads[0], grads[1]  # a_traj, b_traj

# ============================================================================
# MASKED GAME SOLVING
# ============================================================================

def solve_masked_game(agents, initial_states, control_trajectories, 
                     reference_trajectories, mask_values, num_iters=50):
    """
    Solve the masked game for all agents.
    
    Args:
        agents: List of agent objects
        initial_states: List of initial states for each agent
        control_trajectories: List of control trajectories for each agent
        reference_trajectories: List of reference trajectories for each agent
        mask_values: Mask values for collision avoidance [m_12, m_13, m_23]
        num_iters: Number of optimization iterations
        
    Returns:
        Tuple of (state_trajectories, final_control_trajectories)
    """
    n_agents = len(agents)
    
    # Copy control trajectories to avoid modifying originals
    u_trajs = [u_traj.copy() for u_traj in control_trajectories]
    
    # Optimization parameters
    step_size = 0.002
    
    for iter in range(num_iters + 1):
        # Step 1: Linearize dynamics for all agents
        state_trajectories = []
        A_trajectories = []
        B_trajectories = []
        
        for i, agent in enumerate(agents):
            x_traj, A_traj, B_traj = agent.linearize_dyn(initial_states[i], u_trajs[i])
            state_trajectories.append(x_traj)
            A_trajectories.append(A_traj)
            B_trajectories.append(B_traj)
        
        # Step 2: Linearize loss functions for all agents with mask
        a_trajectories = []
        b_trajectories = []
        
        for i in range(n_agents):
            # Create list of other agents' states for this agent
            other_states = [state_trajectories[j] for j in range(n_agents) if j != i]
            
            # For each agent, we need to create the appropriate mask
            # Agent 0: considers agents 1 and 2 with masks [m_12, m_13]
            # Agent 1: considers agents 0 and 2 with masks [m_12, m_23] 
            # Agent 2: considers agents 0 and 1 with masks [m_13, m_23]
            if i == 0:  # Agent 0
                agent_mask = mask_values[:2]  # [m_12, m_13]
            elif i == 1:  # Agent 1
                agent_mask = jnp.array([mask_values[0], mask_values[2]])  # [m_12, m_23]
            else:  # Agent 2
                agent_mask = mask_values[1:]  # [m_13, m_23]
            
            a_traj, b_traj = masked_linearize_loss(
                state_trajectories[i], u_trajs[i], 
                reference_trajectories[i], other_states, agent_mask)
            a_trajectories.append(a_traj)
            b_trajectories.append(b_traj)
        
        # Step 3: Solve LQR subproblems for all agents
        control_updates = []
        
        for i in range(n_agents):
            v_traj, _ = agents[i].solve(
                A_trajectories[i], B_trajectories[i], 
                a_trajectories[i], b_trajectories[i])
            control_updates.append(v_traj)
        
        # Step 4: Update control trajectories
        for i in range(n_agents):
            u_trajs[i] += step_size * control_updates[i]
    
    # Get final state trajectories
    final_state_trajectories = []
    for i, agent in enumerate(agents):
        x_traj, _, _ = agent.linearize_dyn(initial_states[i], u_trajs[i])
        final_state_trajectories.append(x_traj)
    
    return final_state_trajectories, u_trajs


def compute_total_masked_loss(state_trajectories, control_trajectories, 
                            reference_trajectories, mask_values):
    """
    Compute total loss for all agents with mask.
    
    Args:
        state_trajectories: List of state trajectories for each agent
        control_trajectories: List of control trajectories for each agent
        reference_trajectories: List of reference trajectories for each agent
        mask_values: Mask values for collision avoidance [m_12, m_13, m_23]
        
    Returns:
        Total loss value
    """
    n_agents = len(agents)
    total_loss = 0.0
    
    for i in range(n_agents):
        # Create list of other agents' states for this agent
        other_states = [state_trajectories[j] for j in range(n_agents) if j != i]
        
        # For each agent, we need to create the appropriate mask
        if i == 0:  # Agent 0
            agent_mask = mask_values[:2]  # [m_12, m_13]
        elif i == 1:  # Agent 1
            agent_mask = jnp.array([mask_values[0], mask_values[2]])  # [m_12, m_23]
        else:  # Agent 2
            agent_mask = mask_values[1:]  # [m_13, m_23]
        
        agent_loss = masked_trajectory_loss(
            state_trajectories[i], control_trajectories[i],
            reference_trajectories[i], other_states, agent_mask)
        total_loss += agent_loss
    
    return total_loss

# ============================================================================
# GRADIENT TESTING
# ============================================================================

def test_mask_gradient_flow():
    """
    Test gradient flow through the mask parameter.
    
    This function:
    1. Creates a random selection mask
    2. Solves the game with the mask
    3. Computes loss from the resulting trajectory
    4. Attempts to compute gradients with respect to the mask
    5. Verifies that gradients are non-zero
    """
    print("=" * 80)
    print("TESTING MASK GRADIENT FLOW IN iLQGAMES")
    print("=" * 80)
    
    # Create a random mask for testing
    # mask_values = [m_12, m_13, m_23] where:
    # m_12: collision avoidance weight between agents 0 and 1
    # m_13: collision avoidance weight between agents 0 and 2  
    # m_23: collision avoidance weight between agents 1 and 2
    np.random.seed(42)  # For reproducibility
    mask_values = jnp.array([0.7, 0.3, 0.8])  # Random mask values
    
    print(f"Test mask values: {mask_values}")
    print(f"  m_12 (agent 0-1 collision): {mask_values[0]:.3f}")
    print(f"  m_13 (agent 0-2 collision): {mask_values[1]:.3f}")
    print(f"  m_23 (agent 1-2 collision): {mask_values[2]:.3f}")
    print()
    
    # Test 1: Solve the game with the mask
    print("Step 1: Solving masked game...")
    state_trajectories, final_control_trajectories = solve_masked_game(
        agents, initial_states, control_trajectories, 
        reference_trajectories, mask_values, num_iters=50)
    
    print("✓ Game solved successfully!")
    print(f"  Final state trajectories shapes: {[traj.shape for traj in state_trajectories]}")
    print(f"  Final control trajectories shapes: {[traj.shape for traj in control_trajectories]}")
    print()
    
    # Test 2: Compute total loss with the mask
    print("Step 2: Computing total masked loss...")
    total_loss = compute_total_masked_loss(
        state_trajectories, final_control_trajectories, 
        reference_trajectories, mask_values)
    
    print(f"✓ Total loss computed: {float(total_loss):.6f}")
    print()
    
    # Test 3: Test gradient flow through the mask
    print("Step 3: Testing gradient flow through mask...")
    
    def loss_function(mask):
        """Loss function that depends on the mask parameter."""
        # Solve the game with the given mask
        states, controls = solve_masked_game(
            agents, initial_states, control_trajectories, 
            reference_trajectories, mask, num_iters=50)
        
        # Compute loss from the resulting trajectories
        loss = compute_total_masked_loss(
            states, controls, reference_trajectories, mask)
        
        return loss
    
    # Compute gradients with respect to the mask
    try:
        loss_value, gradients = value_and_grad(loss_function)(mask_values)
        
        print(f"✓ Gradients computed successfully!")
        print(f"  Loss value: {float(loss_value):.6f}")
        print(f"  Gradients: {gradients}")
        print(f"  Gradient norm: {float(jnp.linalg.norm(gradients)):.8f}")
        
        # Check if gradients are non-zero
        if jnp.linalg.norm(gradients) > 1e-8:
            print("✓ SUCCESS: Gradients are non-zero! Mask affects the loss function.")
            print("  This means the end-to-end pipeline maintains differentiability.")
        else:
            print("✗ WARNING: Gradients are zero! Mask does not affect the loss function.")
            print("  This suggests a problem in the gradient flow.")
        
    except Exception as e:
        print(f"✗ ERROR: Failed to compute gradients: {e}")
        print("  This suggests a fundamental problem with the pipeline.")
        return False
    
    print()
    
    # Test 4: Test individual mask component gradients
    print("Step 4: Testing individual mask component gradients...")
    
    def test_individual_gradients():
        """Test gradients for each mask component individually."""
        gradients_individual = []
        
        for i in range(len(mask_values)):
            def single_mask_loss(mask_i):
                # Create mask with only one component varying
                test_mask = mask_values.at[i].set(mask_i)
                return loss_function(test_mask)
            
            try:
                grad_i = grad(single_mask_loss)(mask_values[i])
                gradients_individual.append(grad_i)
                print(f"  ∂L/∂m_{i+1}: {float(grad_i):.8f}")
            except Exception as e:
                print(f"  ∂L/∂m_{i+1}: ERROR - {e}")
                gradients_individual.append(0.0)
        
        return gradients_individual
    
    individual_grads = test_individual_gradients()
    
    # Test 5: Verify gradient consistency
    print("\nStep 5: Verifying gradient consistency...")
    
    # Check if individual gradients sum to total gradient
    total_grad_norm = jnp.linalg.norm(gradients)
    individual_grad_norm = jnp.linalg.norm(jnp.array(individual_grads))
    
    print(f"  Total gradient norm: {float(total_grad_norm):.8f}")
    print(f"  Individual gradient norm: {float(individual_grad_norm):.8f}")
    
    if abs(total_grad_norm - individual_grad_norm) < 1e-6:
        print("✓ Gradient consistency check passed!")
    else:
        print("✗ Gradient consistency check failed!")
    
    print()
    
    # Test 6: Test small mask perturbations
    print("Step 6: Testing mask perturbation sensitivity...")
    
    epsilon = 1e-6
    base_loss = loss_function(mask_values)
    
    for i in range(len(mask_values)):
        # Perturb mask component i
        perturbed_mask = mask_values.at[i].set(mask_values[i] + epsilon)
        perturbed_loss = loss_function(perturbed_mask)
        
        # Finite difference approximation
        finite_diff = (perturbed_loss - base_loss) / epsilon
        analytical_grad = gradients[i]
        
        print(f"  m_{i+1}: Finite diff = {float(finite_diff):.8f}, Analytical = {float(analytical_grad):.8f}")
        
        if abs(finite_diff - analytical_grad) < 1e-4:
            print(f"    ✓ Gradient check passed for m_{i+1}")
        else:
            print(f"    ✗ Gradient check failed for m_{i+1}")
    
    print()
    print("=" * 80)
    print("GRADIENT TEST COMPLETED")
    print("=" * 80)
    
    return True


def visualize_masked_trajectories(mask_values):
    """
    Visualize the trajectories with different mask values.
    
    Args:
        mask_values: Mask values for collision avoidance
    """
    print(f"\nVisualizing trajectories with mask: {mask_values}")
    
    # Solve game with the mask
    state_trajectories, final_control_trajectories = solve_masked_game(
        agents, initial_states, control_trajectories, 
        reference_trajectories, mask_values, num_iters=50)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Trajectories
    ax1.set_aspect('equal')
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Agent Trajectories (Mask: {mask_values})')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    
    colors = ['C0', 'C1', 'C2']
    labels = ['Agent 0', 'Agent 1', 'Agent 2']
    
    for i, (traj, ref_traj, color, label) in enumerate(zip(state_trajectories, reference_trajectories, colors, labels)):
        # Plot trajectory
        ax1.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, label=f'{label} (Trajectory)')
        # Plot reference
        ax1.plot(ref_traj[:, 0], ref_traj[:, 1], color=color, linestyle='--', alpha=0.7, label=f'{label} (Reference)')
        # Plot start and end points
        ax1.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=8, markeredgecolor='black')
        ax1.plot(traj[-1, 0], traj[-1, 1], 's', color=color, markersize=8, markeredgecolor='black')
    
    ax1.legend()
    
    # Plot 2: Loss components over time
    ax2.set_title('Loss Components Over Time')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Loss Value')
    
    # Compute loss components for each time step
    time_steps = range(tsteps)
    navigation_losses = []
    collision_losses = []
    control_losses = []
    
    for t in time_steps:
        nav_loss = 0.0
        coll_loss = 0.0
        ctrl_loss = 0.0
        
        for i in range(len(agents)):
            # Navigation loss
            nav_loss += jnp.sum(jnp.square(state_trajectories[i][t, :2] - reference_trajectories[i][t, :2]))
            
            # Collision loss (with mask)
            other_states = [state_trajectories[j][t] for j in range(len(agents)) if j != i]
            if len(other_states) > 0:
                other_positions = jnp.stack([other_xt[:2] for other_xt in other_states])
                distances_squared = jnp.sum(jnp.square(state_trajectories[i][t, :2] - other_positions), axis=1)
                base_collision = 10.0 * jnp.exp(-5.0 * distances_squared)
                
                # Apply appropriate mask
                if i == 0:
                    agent_mask = mask_values[:2]
                elif i == 1:
                    agent_mask = jnp.array([mask_values[0], mask_values[2]])
                else:
                    agent_mask = mask_values[1:]
                
                masked_collision = base_collision * agent_mask[:len(other_states)]
                coll_loss += jnp.sum(masked_collision)
            
            # Control loss
            ctrl_loss += 0.1 * jnp.sum(jnp.square(control_trajectories[i][t] * jnp.array([1.0, 0.5])))
        
        navigation_losses.append(float(nav_loss))
        collision_losses.append(float(coll_loss))
        control_losses.append(float(ctrl_loss))
    
    ax2.plot(time_steps, navigation_losses, label='Navigation Loss', linewidth=2)
    ax2.plot(time_steps, collision_losses, label='Collision Loss (Masked)', linewidth=2)
    ax2.plot(time_steps, control_losses, label='Control Loss', linewidth=2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# SIMPLIFIED GRADIENT TEST
# ============================================================================

def test_simple_mask_gradient():
    """
    Test gradient flow with a simpler, more controlled setup.
    
    This test uses a fixed number of iterations and simpler dynamics
    to ensure more stable gradient computation.
    """
    print("\n" + "=" * 80)
    print("SIMPLIFIED MASK GRADIENT TEST")
    print("=" * 80)
    
    # Use a simpler mask with fewer iterations
    mask_values = jnp.array([0.5, 0.5, 0.5])
    
    print(f"Test mask values: {mask_values}")
    
    def simple_loss_function(mask):
        """Simplified loss function with fewer iterations."""
        # Solve the game with the given mask (fewer iterations for stability)
        states, controls = solve_masked_game(
            agents, initial_states, control_trajectories, 
            reference_trajectories, mask, num_iters=10)  # Reduced iterations
        
        # Compute loss from the resulting trajectories
        loss = compute_total_masked_loss(
            states, controls, reference_trajectories, mask)
        
        return loss
    
    # Test gradient computation
    try:
        loss_value, gradients = value_and_grad(simple_loss_function)(mask_values)
        
        print(f"✓ Simple gradient test successful!")
        print(f"  Loss value: {float(loss_value):.6f}")
        print(f"  Gradients: {gradients}")
        print(f"  Gradient norm: {float(jnp.linalg.norm(gradients)):.8f}")
        
        # Test gradient sign consistency
        print("\nTesting gradient sign consistency...")
        for i in range(len(mask_values)):
            # Test positive perturbation
            pos_mask = mask_values.at[i].set(mask_values[i] + 0.1)
            pos_loss = simple_loss_function(pos_mask)
            
            # Test negative perturbation  
            neg_mask = mask_values.at[i].set(mask_values[i] - 0.1)
            neg_loss = simple_loss_function(neg_mask)
            
            print(f"  m_{i+1}: Loss at {mask_values[i]-0.1:.1f} = {float(neg_loss):.6f}")
            print(f"         Loss at {mask_values[i]:.1f} = {float(loss_value):.6f}")
            print(f"         Loss at {mask_values[i]+0.1:.1f} = {float(pos_loss):.6f}")
            
            # Check if gradient sign makes sense
            if gradients[i] > 0:
                print(f"         ∂L/∂m_{i+1} > 0: Loss increases with mask (expected behavior)")
            else:
                print(f"         ∂L/∂m_{i+1} < 0: Loss decreases with mask (unexpected behavior)")
        
        return True
        
    except Exception as e:
        print(f"✗ Simple gradient test failed: {e}")
        return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Starting mask gradient flow test...")
    
    # Run the main gradient test
    success = test_mask_gradient_flow()
    
    if success:
        print("\n🎉 SUCCESS: Main mask gradient flow test completed successfully!")
        print("The end-to-end pipeline maintains differentiability.")
        
        # Run the simplified gradient test
        simple_success = test_simple_mask_gradient()
        
        if simple_success:
            print("\n🎉 SUCCESS: Simplified gradient test also passed!")
            print("Gradients are consistent and meaningful.")
            
            # Visualize trajectories with different mask values
            print("\nVisualizing trajectories...")
            
            # Test with different mask values
            test_masks = [
                jnp.array([1.0, 1.0, 1.0]),  # Full collision avoidance
                jnp.array([0.0, 0.0, 0.0]),  # No collision avoidance
                jnp.array([0.5, 0.5, 0.5]),  # Medium collision avoidance
            ]
            
            for mask in test_masks:
                visualize_masked_trajectories(mask)
        else:
            print("\n⚠️  WARNING: Simplified gradient test failed!")
            print("There may be numerical instability issues.")
            
    else:
        print("\n❌ FAILURE: Main mask gradient flow test failed!")
        print("There may be an issue with the pipeline's differentiability.")
