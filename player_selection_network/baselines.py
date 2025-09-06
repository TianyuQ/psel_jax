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
        ego_pos = trajectory[0][-4:-2]
        other_pos = np.array([traj[-4:-2] for traj in trajectory[1:]])
        distances = np.linalg.norm(ego_pos - other_pos, axis=1)
        mask = (distances <= mode_parameter).astype(int)

    elif mode == "Nearest Neighbor":
        ego_pos = trajectory[0][-4:-2]
        other_pos = np.array([traj[-4:-2] for traj in trajectory[1:]])
        distances = np.linalg.norm(ego_pos - other_pos, axis=1)
        # Get indices that would sort the distances array (smallest to largest)
        ranked_indices = np.argsort(distances)
        # Select the nearest `mode_parameter - 1` neighbors
        top_indices = ranked_indices[:int(mode_parameter) - 1]
        mask[top_indices] = 1
        
    elif mode == "Jacobian":
        if sim_step == 1:
            mask = baseline_selection(input_traj, trajectory, control, "Nearest Neighbor", sim_step, mode_parameter)
        else:
            delta_t = 0.1
            norm_costs = np.zeros(N - 1)
            ego_state = trajectory[0][-4:]
            for i in range(N - 1):
                player_id = i + 1
                state_diff = ego_state - trajectory[player_id][-4:]
                
                delta_px = (state_diff[0] + delta_t * state_diff[2]) ** 2
                delta_py = (state_diff[1] + delta_t * state_diff[3]) ** 2
                delta_vx = (state_diff[2] + delta_t * control[player_id][0]) ** 2
                delta_vy = (state_diff[3] + delta_t * control[player_id][1]) ** 2

                D = delta_px + delta_py + delta_vx + delta_vy
                J1 = (1 / (D**2)) * 2 * delta_vx * delta_t
                J2 = (1 / (D**2)) * 2 * delta_vy * delta_t
                norm_costs[i] = np.linalg.norm([J1, J2])
            
            ranked_indices = np.argsort(norm_costs)[::-1]
            top_indices = ranked_indices[:int(mode_parameter) - 1]
            mask[top_indices] = 1
            
    elif mode == "Hessian":
        if sim_step == 1:
            mask = baseline_selection(input_traj, trajectory, control, "Nearest Neighbor", sim_step, mode_parameter)
        else:
            delta_t = 0.1
            norm_costs = np.zeros(N - 1)
            ego_state = trajectory[0][-4:]
            for i in range(N-1):
                player_id = i + 1
                state_diff = ego_state - trajectory[player_id][-4:]

                delta_px = (state_diff[0] + delta_t * state_diff[2]) ** 2
                delta_py = (state_diff[1] + delta_t * state_diff[3]) ** 2
                delta_vx = (state_diff[2] + delta_t * control[player_id][0]) ** 2
                delta_vy = (state_diff[3] + delta_t * control[player_id][1]) ** 2

                D = delta_px + delta_py + delta_vx + delta_vy
                H11 = 2 * delta_t**2 / D**3 * (4*delta_vx**2 - D)
                H12 = 8 * delta_t**2 / D**3 * delta_vx * delta_vy
                H22 = 2 * delta_t**2 / D**3 * (4*delta_vy**2 - D)
                hessian_matrix = np.array([[H11, H12], [H12, H22]])
                norm_costs[i] = np.linalg.norm(hessian_matrix) # Frobenius norm
            
            ranked_indices = np.argsort(norm_costs)[::-1]
            top_indices = ranked_indices[:int(mode_parameter) - 1]
            mask[top_indices] = 1

    elif mode == "Cost Evolution":
        if sim_step == 1:
            mask = baseline_selection(input_traj, trajectory, control, "Nearest Neighbor", sim_step, mode_parameter)
        else:
            mu = 1.0
            cost_evolution_values = np.zeros(N - 1)
            for i in range(N - 1):
                player_id = i + 1
                # Current state difference
                state_diff = trajectory[0][-4:-2] - trajectory[player_id][-4:-2]
                D = np.sum(state_diff**2)
                # Previous state difference
                state_diff_prev = trajectory[0][-8:-6] - trajectory[player_id][-8:-6]
                D_prev = np.sum(state_diff_prev**2)
                
                cost_evolution_values[i] = (mu / D) - (mu / D_prev)
            
            ranked_indices = np.argsort(cost_evolution_values)[::-1]
            top_indices = ranked_indices[:int(mode_parameter) - 1]
            mask[top_indices] = 1

    elif mode == "Barrier Function":
        bf_values = np.zeros(N - 1)
        R = 0.5
        kappa = 5.0
        for i in range(N - 1):
            player_id = i + 1
            pos_diff = trajectory[0][-4:-2] - trajectory[player_id][-4:-2]
            vel_diff = trajectory[0][-2:] - trajectory[player_id][-2:]
            
            h = np.sum(pos_diff**2) - R**2
            h_dot = 2 * np.dot(pos_diff, vel_diff)
            bf_values[i] = h_dot + kappa * h
        
        # Rank from smallest to largest (small value = more dangerous)
        ranked_indices = np.argsort(bf_values)
        top_indices = ranked_indices[:int(mode_parameter) - 1]
        mask[top_indices] = 1
        
    elif mode == "Control Barrier Function":
        if sim_step == 1:
            mask = baseline_selection(input_traj, trajectory, control, "Nearest Neighbor", sim_step, mode_parameter)
        else:
            cbf_values = np.zeros(N - 1)
            R = 0.5
            kappa = 5.0
            for i in range(N-1):
                player_id = i + 1
                pos_diff = trajectory[0][-4:-2] - trajectory[player_id][-4:-2]
                vel_diff = trajectory[0][-2:] - trajectory[player_id][-2:]
                accel_diff = control[0] - control[player_id]

                h = np.sum(pos_diff**2) - R**2
                h_dot = 2 * np.dot(pos_diff, vel_diff)
                h_ddot = 2 * (np.dot(vel_diff, vel_diff) + np.dot(pos_diff, accel_diff))
                cbf_values[i] = h_ddot + 2 * kappa * h_dot + kappa**2 * h
            
            # Rank from smallest to largest (small value = more dangerous)
            ranked_indices = np.argsort(cbf_values)
            top_indices = ranked_indices[:int(mode_parameter) - 1]
            mask[top_indices] = 1

    else:
        raise ValueError(f"Invalid mode: {mode}")

    return mask