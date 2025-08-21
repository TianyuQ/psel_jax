#!/usr/bin/env python3
"""
Multi-agent iLQGames Example

This script demonstrates the implementation of Iterative Linear-Quadratic Games (iLQGames)
for multi-agent trajectory optimization. The example involves three different agents:
1. A differential-drive vehicle
2. A second-order point mass (pedestrian)
3. A bicycle model

Reference: Fridovich-Keil, David, et al. "Efficient iterative linear-quadratic 
approximations for nonlinear multi-player general-sum differential games." 
2020 IEEE international conference on robotics and automation (ICRA).

The algorithm iteratively linearizes the dynamics and cost functions around the 
current trajectory, then solves the resulting linear-quadratic game to find 
descent directions for all agents simultaneously.
"""

import jax 
import jax.numpy as jnp 
from jax import vmap, jit, grad
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from lqrax import iLQR

# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================

# Time discretization parameters
dt = 0.05          # Time step size (seconds)
tsteps = 100       # Number of time steps
device = jax.devices("cpu")[0]  # Use CPU for computation

# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

class DiffdriveAgent(iLQR):
    """
    Differential-drive vehicle agent.
    
    State: [x, y, theta] - position (x,y) and orientation theta
    Control: [v, omega] - linear velocity and angular velocity
    
    Dynamics: 
        dx/dt = v * cos(theta)
        dy/dt = v * sin(theta) 
        dtheta/dt = omega
    """
    def __init__(self, dt, x_dim, u_dim, Q, R):
        super().__init__(dt, x_dim, u_dim, Q, R)
    
    def dyn(self, xt, ut):
        """Dynamics function for differential-drive vehicle."""
        return jnp.array([
            ut[0] * jnp.cos(xt[2]),  # dx/dt = v * cos(theta)
            ut[0] * jnp.sin(xt[2]),  # dy/dt = v * sin(theta)
            ut[1]                     # dtheta/dt = omega
        ])


class PointAgent(iLQR):
    """
    Second-order point mass agent (e.g., pedestrian).
    
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


class BicycleAgent(iLQR):
    """
    Bicycle model agent.
    
    State: [x, y, theta] - position (x,y) and orientation theta
    Control: [v, delta] - velocity and steering angle
    
    Dynamics:
        dx/dt = v * cos(theta)
        dy/dt = v * sin(theta)
        dtheta/dt = v * tan(delta) / L
    """
    def __init__(self, dt, x_dim, u_dim, Q, R):
        super().__init__(dt, x_dim, u_dim, Q, R)

    def dyn(self, xt, ut):
        """Dynamics function for bicycle model."""
        L = 0.03  # Wheelbase length
        x, y, theta = xt
        v, delta = ut
        dx = v * jnp.cos(theta)
        dy = v * jnp.sin(theta)
        dtheta = v * jnp.tan(delta) / L
        return jnp.array([dx, dy, dtheta])


# ============================================================================
# AGENT 1: DIFFERENTIAL-DRIVE VEHICLE SETUP
# ============================================================================

# Cost function weights for differential-drive agent
Q_diffdrive = jnp.diag(jnp.array([0.1, 0.1, 0.01]))  # State cost weights
R_diffdrive = jnp.diag(jnp.array([1.0, 0.01]))        # Control cost weights

# Initialize the differential-drive agent
diffdrive_ilqgames = DiffdriveAgent(dt=dt, x_dim=3, u_dim=2, Q=Q_diffdrive, R=R_diffdrive)


def diffdrive_runtime_loss(xt, ut, ref_xt, other_xt1, other_xt2):
    """
    Runtime loss function for differential-drive agent.
    
    Args:
        xt: Current state [x, y, theta]
        ut: Current control [v, omega]
        ref_xt: Reference state
        other_xt1, other_xt2: States of other agents for collision avoidance
    
    Returns:
        Total loss combining navigation, collision avoidance, and control costs
    """
    # Navigation cost: penalize deviation from reference trajectory
    nav_loss = jnp.sum(jnp.square(xt[:2] - ref_xt[:2]))
    
    # Collision avoidance costs: exponential penalty for proximity to other agents
    collision_loss1 = 10.0 * jnp.exp(-5.0 * jnp.sum(jnp.square(xt[:2] - other_xt1[:2])))
    collision_loss2 = 10.0 * jnp.exp(-5.0 * jnp.sum(jnp.square(xt[:2] - other_xt2[:2])))
    
    # Control cost: penalize large control inputs
    ctrl_loss = 0.1 * jnp.sum(jnp.square(ut * jnp.array([1.0, 0.01])))
    
    return nav_loss + collision_loss1 + collision_loss2 + ctrl_loss


def diffdrive_loss(x_traj, u_traj, ref_x_traj, other_x_traj1, other_x_traj2):
    """Compute total loss over entire trajectory."""
    runtime_loss_array = vmap(diffdrive_runtime_loss, in_axes=(0, 0, 0, 0, 0))(
        x_traj, u_traj, ref_x_traj, other_x_traj1, other_x_traj2)
    return runtime_loss_array.sum() * diffdrive_ilqgames.dt


def diffdrive_linearize_loss(x_traj, u_traj, ref_x_traj, other_x_traj1, other_x_traj2):
    """
    Linearize the loss function around the current trajectory.
    
    Returns:
        a_traj: Gradient of loss with respect to state
        b_traj: Gradient of loss with respect to control
    """
    dldx = grad(diffdrive_runtime_loss, argnums=(0))  # Gradient w.r.t. state
    dldu = grad(diffdrive_runtime_loss, argnums=(1))  # Gradient w.r.t. control
    
    a_traj = vmap(dldx, in_axes=(0, 0, 0, 0, 0))(
        x_traj, u_traj, ref_x_traj, other_x_traj1, other_x_traj2)
    b_traj = vmap(dldu, in_axes=(0, 0, 0, 0, 0))(
        x_traj, u_traj, ref_x_traj, other_x_traj1, other_x_traj2)
    
    return a_traj, b_traj


# Compile functions for efficiency
diffdrive_linearize_dyn = jit(diffdrive_ilqgames.linearize_dyn, device=device)
diffdrive_solve_ilqr = jit(diffdrive_ilqgames.solve, device=device)
diffdrive_loss = jit(diffdrive_loss, device=device)
diffdrive_linearize_loss = jit(diffdrive_linearize_loss, device=device)


# ============================================================================
# AGENT 2: POINT MASS SETUP
# ============================================================================

# Cost function weights for point mass agent
Q_point = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.001]))  # State cost weights
R_point = jnp.diag(jnp.array([0.01, 0.01]))               # Control cost weights

# Initialize the point mass agent
point_ilqgames = PointAgent(dt=dt, x_dim=4, u_dim=2, Q=Q_point, R=R_point)


def point_runtime_loss(xt, ut, ref_xt, other_xt1, other_xt2):
    """Runtime loss function for point mass agent."""
    nav_loss = jnp.sum(jnp.square(xt[:2] - ref_xt[:2]))
    collision_loss1 = 10.0 * jnp.exp(-5.0 * jnp.sum(jnp.square(xt[:2] - other_xt1[:2])))
    collision_loss2 = 10.0 * jnp.exp(-5.0 * jnp.sum(jnp.square(xt[:2] - other_xt2[:2])))
    ctrl_loss = 0.1 * jnp.sum(jnp.square(ut * jnp.array([1.0, 0.5])))
    return nav_loss + collision_loss1 + collision_loss2 + ctrl_loss


def point_loss(x_traj, u_traj, ref_x_traj, other_x_traj1, other_x_traj2):
    """Compute total loss over entire trajectory."""
    runtime_loss_array = vmap(point_runtime_loss, in_axes=(0, 0, 0, 0, 0))(
        x_traj, u_traj, ref_x_traj, other_x_traj1, other_x_traj2)
    return runtime_loss_array.sum() * point_ilqgames.dt


def point_linearize_loss(x_traj, u_traj, ref_x_traj, other_x_traj1, other_x_traj2):
    """Linearize the loss function around the current trajectory."""
    dldx = grad(point_runtime_loss, argnums=(0))
    dldu = grad(point_runtime_loss, argnums=(1))
    a_traj = vmap(dldx, in_axes=(0, 0, 0, 0, 0))(
        x_traj, u_traj, ref_x_traj, other_x_traj1, other_x_traj2)
    b_traj = vmap(dldu, in_axes=(0, 0, 0, 0, 0))(
        x_traj, u_traj, ref_x_traj, other_x_traj1, other_x_traj2)
    return a_traj, b_traj


# Compile functions for efficiency
point_linearize_dyn = jit(point_ilqgames.linearize_dyn, device=device)
point_solve_ilqr = jit(point_ilqgames.solve, device=device)
point_loss = jit(point_loss, device=device)
point_linearize_loss = jit(point_linearize_loss, device=device)


# ============================================================================
# AGENT 3: BICYCLE MODEL SETUP
# ============================================================================

# Cost function weights for bicycle agent
Q_bicycle = jnp.diag(jnp.array([0.1, 0.1, 0.01]))  # State cost weights
R_bicycle = jnp.diag(jnp.array([1.0, 0.1]))         # Control cost weights

# Initialize the bicycle agent
bicycle_ilqgames = BicycleAgent(dt=dt, x_dim=3, u_dim=2, Q=Q_bicycle, R=R_bicycle)


def bicycle_runtime_loss(xt, ut, ref_xt, other_xt1, other_xt2):
    """Runtime loss function for bicycle agent."""
    nav_loss = jnp.sum(jnp.square(xt[:2] - ref_xt[:2]))
    collision_loss1 = 10.0 * jnp.exp(-5.0 * jnp.sum(jnp.square(xt[:2] - other_xt1[:2])))
    collision_loss2 = 10.0 * jnp.exp(-5.0 * jnp.sum(jnp.square(xt[:2] - other_xt2[:2])))
    ctrl_loss = 0.1 * jnp.sum(jnp.square(ut * jnp.array([1.0, 0.01])))
    return nav_loss + collision_loss1 + collision_loss2 + ctrl_loss


def bicycle_loss(x_traj, u_traj, ref_x_traj, other_x_traj1, other_x_traj2):
    """Compute total loss over entire trajectory."""
    runtime_loss_array = vmap(bicycle_runtime_loss, in_axes=(0, 0, 0, 0, 0))(
        x_traj, u_traj, ref_x_traj, other_x_traj1, other_x_traj2)
    return runtime_loss_array.sum() * bicycle_ilqgames.dt


def bicycle_linearize_loss(x_traj, u_traj, ref_x_traj, other_x_traj1, other_x_traj2):
    """Linearize the loss function around the current trajectory."""
    dldx = grad(bicycle_runtime_loss, argnums=(0))
    dldu = grad(bicycle_runtime_loss, argnums=(1))
    a_traj = vmap(dldx, in_axes=(0, 0, 0, 0, 0))(
        x_traj, u_traj, ref_x_traj, other_x_traj1, other_x_traj2)
    b_traj = vmap(dldu, in_axes=(0, 0, 0, 0, 0))(
        x_traj, u_traj, ref_x_traj, other_x_traj1, other_x_traj2)
    return a_traj, b_traj


# Compile functions for efficiency
bicycle_linearize_dyn = jit(bicycle_ilqgames.linearize_dyn, device=device)
bicycle_solve_ilqr = jit(bicycle_ilqgames.solve, device=device)
bicycle_loss = jit(bicycle_loss, device=device)
bicycle_linearize_loss = jit(bicycle_linearize_loss, device=device)


# ============================================================================
# INITIAL CONDITIONS AND REFERENCE TRAJECTORIES
# ============================================================================

# Initial states for each agent
diffdrive_x0 = jnp.array([-2.0, -0.1, 0.0])      # [x, y, theta]
point_x0 = jnp.array([2.0, 0.1, -0.8, 0.0])      # [x, y, vx, vy]
bicycle_x0 = jnp.array([-0.2, -2.0, jnp.pi/2.0]) # [x, y, theta]

# Initial control trajectories (constant controls)
diffdrive_u_traj = jnp.tile(jnp.array([0.8, 0.0]), reps=(tsteps, 1))  # [v, omega]
point_u_traj = jnp.zeros((tsteps, 2))                                  # [ax, ay]
bicycle_u_traj = jnp.tile(jnp.array([0.5, 0.0]), reps=(tsteps, 1))   # [v, delta]

# Reference trajectories (target paths for each agent)
diffdrive_ref_traj = jnp.linspace(
    jnp.array([-2.0, 0.0]), jnp.array([2.0, 0.0]), tsteps+1
)[1:]  # Move from left to right

point_ref_traj = jnp.linspace(
    jnp.array([2.0, 0.0]), jnp.array([-2.0, 0.0]), tsteps+1
)[1:]  # Move from right to left

bicycle_ref_traj = jnp.linspace(
    jnp.array([0.0, -2.0]), jnp.array([0.0, 2.0]), tsteps+1
)[1:]  # Move from bottom to top


# ============================================================================
# iLQGames ITERATIVE OPTIMIZATION
# ============================================================================

print("Starting iLQGames optimization...")
print("=" * 60)

num_iters = 200
step_size = 0.002

for iter in range(num_iters + 1):
    # Step 1: Linearize dynamics at current trajectory
    # This computes the Jacobians A and B for each agent
    diffdrive_x_traj, diffdrive_A_traj, diffdrive_B_traj = \
        diffdrive_linearize_dyn(diffdrive_x0, diffdrive_u_traj)
    point_x_traj, point_A_traj, point_B_traj = \
        point_linearize_dyn(point_x0, point_u_traj)
    bicycle_x_traj, bicycle_A_traj, bicycle_B_traj = \
        bicycle_linearize_dyn(bicycle_x0, bicycle_u_traj)
    
    # Step 2: Linearize loss functions at current trajectory
    # This computes gradients of loss w.r.t. state and control
    diffdrive_a_traj, diffdrive_b_traj = \
        diffdrive_linearize_loss(
            diffdrive_x_traj, diffdrive_u_traj, diffdrive_ref_traj, 
            point_x_traj, bicycle_x_traj)
    point_a_traj, point_b_traj = \
        point_linearize_loss(
            point_x_traj, point_u_traj, point_ref_traj, 
            diffdrive_x_traj, bicycle_x_traj)
    bicycle_a_traj, bicycle_b_traj = \
        bicycle_linearize_loss(
            bicycle_x_traj, bicycle_u_traj, bicycle_ref_traj, 
            diffdrive_x_traj, point_x_traj)
    
    # Step 3: Solve linear-quadratic subproblems to find descent directions
    # Each agent solves its own LQR problem given the current linearization
    diffdrive_v_traj, _ = diffdrive_solve_ilqr(
        diffdrive_A_traj, diffdrive_B_traj, diffdrive_a_traj, diffdrive_b_traj)
    point_v_traj, _ = point_solve_ilqr(
        point_A_traj, point_B_traj, point_a_traj, point_b_traj)
    bicycle_v_traj, _ = bicycle_solve_ilqr(
        bicycle_A_traj, bicycle_B_traj, bicycle_a_traj, bicycle_b_traj)
    
    # Step 4: Update control trajectories using gradient descent
    if iter % int(num_iters/10) == 0:
        # Compute and display current loss values
        diffdrive_loss_val = diffdrive_loss(
            diffdrive_x_traj, diffdrive_u_traj, diffdrive_ref_traj, 
            point_x_traj, bicycle_x_traj)
        point_loss_val = point_loss(
            point_x_traj, point_u_traj, point_ref_traj, 
            diffdrive_x_traj, bicycle_x_traj)
        bicycle_loss_val = bicycle_loss(
            bicycle_x_traj, bicycle_u_traj, bicycle_ref_traj, 
            diffdrive_x_traj, point_x_traj)
        print(
            f'iter[{iter:3d}/{num_iters}] | diffdrive loss: {diffdrive_loss_val:5.2f} | '
            f'point loss: {point_loss_val:5.2f} | bicycle loss: {bicycle_loss_val:5.2f}')
        
    # Update control trajectories with scaled descent directions
    diffdrive_u_traj += step_size * diffdrive_v_traj
    point_u_traj += step_size * point_v_traj
    bicycle_u_traj += step_size * bicycle_v_traj

print("=" * 60)
print("Optimization completed!")


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_animation():
    """Create an animation of the optimized trajectories."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=120, tight_layout=True)
    
    def update(t):
        ax.cla()
        ax.set_aspect('equal')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.axis('off')
        
        # Plot target positions
        ax.plot(diffdrive_ref_traj[-1, 0], diffdrive_ref_traj[-1, 1], 
                linestyle='', marker='X', markersize=20, color='C0', alpha=0.5, label='DiffDrive Target')
        ax.plot(point_ref_traj[-1, 0], point_ref_traj[-1, 1], 
                linestyle='', marker='X', markersize=20, color='C1', alpha=0.5, label='Point Mass Target')
        ax.plot(bicycle_ref_traj[-1, 0], bicycle_ref_traj[-1, 1], 
                linestyle='', marker='X', markersize=20, color='C2', alpha=0.5, label='Bicycle Target')
        
        # Plot differential-drive vehicle
        diffdrive_xt = diffdrive_x_traj[t]
        diffdrive_theta = diffdrive_xt[2]
        diffdrive_angle = np.rad2deg(diffdrive_theta)
        ax.plot(diffdrive_x_traj[:t, 0], diffdrive_x_traj[:t, 1],
                linestyle='-', linewidth=5, color='C0', alpha=0.5)
        ax.plot(diffdrive_xt[0], diffdrive_xt[1], linestyle='', 
                marker=(4, 0, diffdrive_angle+45), markersize=30, color='C0', label='DiffDrive')
        ax.plot(diffdrive_xt[0]+np.cos(diffdrive_theta)*0.32, 
                diffdrive_xt[1]+np.sin(diffdrive_theta)*0.32, 
                linestyle='', marker=(3, 0, diffdrive_angle+30), markersize=15, color='C0')
        
        # Plot point mass
        point_xt = point_x_traj[t]
        point_theta = np.arctan2(point_xt[3], point_xt[2])
        point_angle = np.rad2deg(point_theta)
        ax.plot(point_x_traj[:t, 0], point_x_traj[:t, 1],
                linestyle='-', linewidth=5, color='C1', alpha=0.5)
        ax.plot(point_xt[0], point_xt[1], linestyle='',
                marker='o', markersize=25, color='C1', label='Point Mass')
        ax.plot(point_xt[0]+np.cos(point_theta)*0.36, 
                point_xt[1]+np.sin(point_theta)*0.36, 
                linestyle='', marker=(3, 0, point_angle+30), markersize=15, color='C1')
        
        # Plot bicycle
        bicycle_xt = bicycle_x_traj[t]
        bicycle_theta = bicycle_xt[2]
        bicycle_angle = np.rad2deg(bicycle_theta)
        ax.plot(bicycle_x_traj[:t, 0], bicycle_x_traj[:t, 1],
                linestyle='-', linewidth=5, color='C2', alpha=0.5)
        ax.plot(bicycle_xt[0], bicycle_xt[1], linestyle='', 
                marker=(4, 0, bicycle_angle+45), markersize=30, color='C2', label='Bicycle')
        ax.plot(bicycle_xt[0]+np.cos(bicycle_theta)*0.33, 
                bicycle_xt[1]+np.sin(bicycle_theta)*0.33, 
                linestyle='', marker=(3, 0, bicycle_angle+30), markersize=15, color='C2')
        
        ax.set_title(f'Multi-Agent iLQGames - Time Step {t}/{tsteps}')
        ax.legend()
        return []
    
    ani = animation.FuncAnimation(fig, update, frames=tsteps, interval=50, repeat=True)
    return ani

if __name__ == "__main__":
    # Create and display the animation
    print("\nCreating animation...")
    ani = create_animation()
    plt.show()
    
    # Save the animation as a GIF (optional)
    # ani.save('ilqgames_animation.gif', writer='pillow') 