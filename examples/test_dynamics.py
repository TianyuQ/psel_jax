#!/usr/bin/env python3
"""
Test dynamics function to debug the shape issue.
"""

import jax
import jax.numpy as jnp
from lqrax import iLQR

class PointAgent(iLQR):
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

# Test the dynamics
agent = PointAgent(dt=0.05, x_dim=4, u_dim=2, Q=jnp.eye(4), R=jnp.eye(2))

# Test state and control
xt = jnp.array([1.0, 2.0, 0.5, 0.3])  # [x, y, vx, vy]
ut = jnp.array([0.1, 0.2])  # [ax, ay]

print("Testing dynamics function...")
print(f"xt shape: {xt.shape}, xt: {xt}")
print(f"ut shape: {ut.shape}, ut: {ut}")

# Test dynamics
dx = agent.dyn(xt, ut)
print(f"dx shape: {dx.shape}, dx: {dx}")

# Test dyn_step
print("\nTesting dyn_step...")
xt_new, _ = agent.dyn_step(xt, ut)
print(f"xt_new shape: {xt_new.shape}, xt_new: {xt_new}") 