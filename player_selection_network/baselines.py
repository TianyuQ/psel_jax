import numpy as np
from typing import List, Any

def baseline_selection(
    input_traj: Any,
    trajectory: List[np.ndarray],
    control: List[np.ndarray],
    mode: str,
    sim_step: int,
    mode_parameter: float
) -> np.ndarray:
    """
    Computes a communication mask based on different modes of interaction.

    This function determines which other players the ego player (player 0)
    should consider, based on the specified mode.

    Args:
        input_traj: Input trajectory for neural network models.
        trajectory: A list where each element is a numpy array representing
                    the state history of a player.
        control: A list where each element is the control input for a player.
        mode: The string specifying the computation method.
        sim_step: The current simulation step.
        mode_parameter: A parameter whose meaning depends on the selected mode
                        (e.g., a distance threshold, number of neighbors).

    Returns:
        A numpy array of shape (N-1,) with binary values (0 or 1) indicating
        whether to consider each of the other N-1 players.

    Raises:
        ValueError: If an unknown mode is provided.
    """
    N = len(trajectory)
    mask = np.zeros(N - 1)

    if mode == "All":
        mask = np.ones(N - 1)

    elif mode == "Distance Threshold":
        # Ensure we have enough trajectory data
        if len(trajectory[0]) < 4:
            # If trajectory is too short, use all agents
            mask = np.ones(N - 1)
        else:
            ego_pos = trajectory[0][-1, :2]  # Last state, first 2 elements (x, y)
            other_pos = np.array([traj[-1, :2] for traj in trajectory[1:]])  # Last state, first 2 elements
            distances = np.linalg.norm(ego_pos - other_pos, axis=1)
            mask = (distances <= mode_parameter).astype(int)

    elif mode == "Nearest Neighbor":
        # Ensure we have enough trajectory data
        if len(trajectory[0]) < 4:
            # If trajectory is too short, use all agents
            mask = np.ones(N - 1)
        else:
            ego_pos = trajectory[0][-1, :2]  # Last state, first 2 elements (x, y)
            other_pos = np.array([traj[-1, :2] for traj in trajectory[1:]])  # Last state, first 2 elements
            distances = np.linalg.norm(ego_pos - other_pos, axis=1)
            # Get indices that would sort the distances array (smallest to largest)
            ranked_indices = np.argsort(distances)
            # Select the nearest `mode_parameter - 1` neighbors
            num_neighbors = max(0, min(int(mode_parameter) - 1, len(ranked_indices)))
            top_indices = ranked_indices[:num_neighbors]
            
            mask[top_indices] = 1
        
    elif mode == "Jacobian":
        if sim_step == 1:
            mask = baseline_selection(input_traj, trajectory, control, "Nearest Neighbor", sim_step, mode_parameter)
        else:
            # Ensure we have enough trajectory and control data
            if len(trajectory[0]) < 4 or len(control) < N:
                # If data is insufficient, use all agents
                mask = np.ones(N - 1)
            else:
                delta_t = 0.1
                norm_costs = np.zeros(N - 1)
                ego_state = trajectory[0][-1, :]  # Last state, all elements
                for i in range(N - 1):
                    player_id = i + 1
                    if len(trajectory[player_id]) < 4 or len(control[player_id]) < 2:
                        # Skip this agent if data is insufficient
                        norm_costs[i] = 0.0
                        continue
                        
                    state_diff = ego_state - trajectory[player_id][-1, :]  # Last state, all elements
                    
                    delta_px = (state_diff[0] + delta_t * state_diff[2]) ** 2
                    delta_py = (state_diff[1] + delta_t * state_diff[3]) ** 2
                    delta_vx = (state_diff[2] + delta_t * control[player_id][0]) ** 2
                    delta_vy = (state_diff[3] + delta_t * control[player_id][1]) ** 2

                    D = delta_px + delta_py + delta_vx + delta_vy
                    if D > 1e-10:  # Avoid division by zero

                        future_vel_diff_x = state_diff[2] + delta_t * control[player_id][0]
                        future_vel_diff_y = state_diff[3] + delta_t * control[player_id][1]
                        
                        # New Jacobian for cost = exp(-D) where D is the squared norm
                        exp_term = np.exp(-D)
                        J1 = 2 * delta_t * future_vel_diff_x * exp_term
                        J2 = 2 * delta_t * future_vel_diff_y * exp_term
                        norm_costs[i] = np.linalg.norm([J1, J2])

                    else:
                        norm_costs[i] = 0.0
                
                ranked_indices = np.argsort(norm_costs)[::-1]
                num_neighbors = max(0, min(int(mode_parameter) - 1, len(ranked_indices)))
                top_indices = ranked_indices[:num_neighbors]
                mask[top_indices] = 1
            
    elif mode == "Hessian":
        if sim_step == 1:
            mask = baseline_selection(input_traj, trajectory, control, "Nearest Neighbor", sim_step, mode_parameter)
        else:
            # Ensure we have enough trajectory and control data
            if len(trajectory[0]) < 4 or len(control) < N:
                # If data is insufficient, use all agents
                mask = np.ones(N - 1)
            else:
                delta_t = 0.1
                norm_costs = np.zeros(N - 1)
                ego_state = trajectory[0][-1, :]  # Last state, all elements
                for i in range(N-1):
                    player_id = i + 1
                    if len(trajectory[player_id]) < 4 or len(control[player_id]) < 2:
                        # Skip this agent if data is insufficient
                        norm_costs[i] = 0.0
                        continue
                        
                    state_diff = ego_state - trajectory[player_id][-1, :]  # Last state, all elements

                    delta_px = (state_diff[0] + delta_t * state_diff[2]) ** 2
                    delta_py = (state_diff[1] + delta_t * state_diff[3]) ** 2
                    delta_vx = (state_diff[2] + delta_t * control[player_id][0]) ** 2
                    delta_vy = (state_diff[3] + delta_t * control[player_id][1]) ** 2

                    D = delta_px + delta_py + delta_vx + delta_vy
                    if D > 1e-10:  # Avoid division by zero

                        future_vel_diff_x = state_diff[2] + delta_t * control[player_id][0]
                        future_vel_diff_y = state_diff[3] + delta_t * control[player_id][1]

                        # New Hessian for cost = exp(-D) where D is the squared norm
                        exp_term = np.exp(-D)
                        common_factor = 2 * delta_t**2 * exp_term

                        H11 = common_factor * (2 * future_vel_diff_x**2 - 1)
                        H22 = common_factor * (2 * future_vel_diff_y**2 - 1)
                        H12 = 2 * common_factor * future_vel_diff_x * future_vel_diff_y
                        
                        hessian_matrix = np.array([[H11, H12], [H12, H22]])
                        norm_costs[i] = np.linalg.norm(hessian_matrix) # Frobenius norm

                    else:
                        norm_costs[i] = 0.0
                
                ranked_indices = np.argsort(norm_costs)[::-1]
                num_neighbors = max(0, min(int(mode_parameter) - 1, len(ranked_indices)))
                top_indices = ranked_indices[:num_neighbors]
                mask[top_indices] = 1

    elif mode == "Cost Evolution":
        if sim_step == 1:
            mask = baseline_selection(input_traj, trajectory, control, "Nearest Neighbor", sim_step, mode_parameter)
        else:
            # Ensure we have enough trajectory data (need at least 2 previous states)
            if len(trajectory[0]) < 8:
                # If trajectory is too short, use all agents
                mask = np.ones(N - 1)
            else:
                mu = 1.0
                cost_evolution_values = np.zeros(N - 1)
                for i in range(N - 1):
                    player_id = i + 1
                    if len(trajectory[player_id]) < 8:
                        # Skip this agent if data is insufficient
                        cost_evolution_values[i] = 0.0
                        continue
                        
                    # Current state difference
                    state_diff = trajectory[0][-1, :2] - trajectory[player_id][-1, :2]  # Last state, position only
                    D = np.sum(state_diff**2)
                    # Previous state difference
                    state_diff_prev = trajectory[0][-2, :2] - trajectory[player_id][-2, :2]  # Second to last state, position only
                    D_prev = np.sum(state_diff_prev**2)
                    
                    if D > 1e-10 and D_prev > 1e-10:  # Avoid division by zero
                        cost_evolution_values[i] = mu*np.exp(-D) - mu*np.exp(-D_prev)
                    else:
                        cost_evolution_values[i] = 0.0
                
                ranked_indices = np.argsort(cost_evolution_values)[::-1]
                num_neighbors = max(0, min(int(mode_parameter) - 1, len(ranked_indices)))
                top_indices = ranked_indices[:num_neighbors]
                mask[top_indices] = 1

    elif mode == "Barrier Function":
        # Ensure we have enough trajectory data
        if len(trajectory[0]) < 4:
            # If trajectory is too short, use all agents
            mask = np.ones(N - 1)
        else:
            bf_values = np.zeros(N - 1)
            R = 0.5
            kappa = 5.0
            for i in range(N - 1):
                player_id = i + 1
                if len(trajectory[player_id]) < 4:
                    # Skip this agent if data is insufficient
                    bf_values[i] = 0.0
                    continue
                    
                pos_diff = trajectory[0][-1, :2] - trajectory[player_id][-1, :2]  # Last state, position only
                vel_diff = trajectory[0][-1, 2:] - trajectory[player_id][-1, 2:]  # Last state, velocity only
                
                h = np.sum(pos_diff**2) - R**2
                h_dot = 2 * np.dot(pos_diff, vel_diff)
                bf_values[i] = h_dot + kappa * h
            
            # Rank from smallest to largest (small value = more dangerous)
            ranked_indices = np.argsort(bf_values)
            num_neighbors = max(0, min(int(mode_parameter) - 1, len(ranked_indices)))
            top_indices = ranked_indices[:num_neighbors]
            mask[top_indices] = 1
        
    elif mode == "Control Barrier Function":
        if sim_step == 1:
            mask = baseline_selection(input_traj, trajectory, control, "Nearest Neighbor", sim_step, mode_parameter)
        else:
            # Ensure we have enough trajectory and control data
            if len(trajectory[0]) < 4 or len(control) < N:
                # If data is insufficient, use all agents
                mask = np.ones(N - 1)
            else:
                cbf_values = np.zeros(N - 1)
                R = 0.5
                kappa = 5.0
                for i in range(N-1):
                    player_id = i + 1
                    if len(trajectory[player_id]) < 4 or len(control[player_id]) < 2:
                        # Skip this agent if data is insufficient
                        cbf_values[i] = 0.0
                        continue
                        
                    pos_diff = trajectory[0][-1, :2] - trajectory[player_id][-1, :2]  # Last state, position only
                    vel_diff = trajectory[0][-1, 2:] - trajectory[player_id][-1, 2:]  # Last state, velocity only
                    accel_diff = control[0] - control[player_id]

                    h = np.sum(pos_diff**2) - R**2
                    h_dot = 2 * np.dot(pos_diff, vel_diff)
                    h_ddot = 2 * (np.dot(vel_diff, vel_diff) + np.dot(pos_diff, accel_diff))
                    cbf_values[i] = h_ddot + 2 * kappa * h_dot + kappa**2 * h
                
                # Rank from smallest to largest (small value = more dangerous)
                ranked_indices = np.argsort(cbf_values)
                num_neighbors = max(0, min(int(mode_parameter) - 1, len(ranked_indices)))
                top_indices = ranked_indices[:num_neighbors]
                mask[top_indices] = 1

    else:
        raise ValueError(f"Invalid mode: {mode}")

    return mask