#!/usr/bin/env python3

import json
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from PIL import Image
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import load_config

def select_agents_by_mask(predicted_mask, selection_method, mask_threshold, rank):
    """Select agents based on predicted mask using threshold or rank method."""
    if selection_method == "threshold":
        selected_mask_indices = jnp.where(predicted_mask > mask_threshold)[0]
        selected_agents = selected_mask_indices + 1  # Convert to agent numbers 1-9
        num_selected = len(selected_agents)
    elif selection_method == "rank":
        num_other_agents = rank - 1
        if num_other_agents > 0:
            top_indices = jnp.argsort(predicted_mask)[-num_other_agents:]
            selected_mask_indices = top_indices
            selected_agents = selected_mask_indices + 1
        else:
            selected_mask_indices = jnp.array([])
            selected_agents = jnp.array([])
        num_selected = len(selected_agents)
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    return selected_mask_indices, selected_agents, num_selected

def create_static_figure(json_file_path, output_dir=None):
    """
    Creates a static multi-panel figure from the 10th, 20th, 30th, 40th, and 50th frames.
    """
    print(f"Creating static figure for: {json_file_path}")

    # Load data
    config = load_config()
    with open(json_file_path, 'r') as f:
        results = json.load(f)

    T_observation = results['T_observation']
    T_total = results['T_total']
    n_agents = 10
    normalized_sample_data = results['normalized_sample_data']

    # --- 2. Define Plot Parameters ---
    snapshot_frame_indices = [9, 19, 29, 39] # Corresponds to 10th, 20th, 30th, 40th frames
    full_trajectory_end_frame = 49 # Corresponds to 50th frame
    
    if T_total <= full_trajectory_end_frame:
        print(f"Warning: T_total ({T_total}) is not large enough for the required frames. Skipping figure generation.")
        return

    # PSN configurations to plot as rows
    psn_configs = [
        {'label': 'PSN-Full-th[0.5]', 'method': 'threshold', 'value': 0.5, 'rank': None},
        {'label': 'PSN-Full-Rank[2]', 'method': 'rank', 'value': None, 'rank': 2},
        {'label': 'All', 'method': 'all', 'value': None, 'rank': n_agents}
    ]
    
    num_rows = len(psn_configs)
    num_cols = len(snapshot_frame_indices) + 1 # +1 for Full Trajectory

    # Colors from the original script
    ego_color = 'darkblue'
    other_agent_color = 'gray'
    selected_color = 'red'

    # --- 3. Create Subplot Grid ---
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3.5), squeeze=False)
    
    # --- 4. Main Plotting Loop ---
    for row_idx, psn_config in enumerate(psn_configs):
        for col_idx in range(num_cols):
            ax = axes[row_idx, col_idx]
            ax.set_aspect('equal')
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-3.5, 3.5)
            ax.set_xticks([])
            ax.set_yticks([])

            # Determine the step and plot type
            is_full_trajectory_plot = (col_idx == num_cols - 1)
            step = full_trajectory_end_frame if is_full_trajectory_plot else snapshot_frame_indices[col_idx]

            # Get selected agents for this step and PSN config
            selected_agents_np = np.array([])
            if step >= T_observation:
                iteration_idx = step - T_observation
                if iteration_idx < len(results['receding_horizon_results']):
                    iteration_result = results['receding_horizon_results'][iteration_idx]
                    predicted_mask = jnp.array(iteration_result['predicted_mask'])
                    
                    if psn_config['method'] == 'all':
                        selected_agents_np = np.arange(1, n_agents)
                    else:
                        _, selected_agents, _ = select_agents_by_mask(
                            predicted_mask, psn_config['method'], psn_config['value'], psn_config['rank'])
                        selected_agents_np = np.array(selected_agents)

            # --- Plotting Logic ---
            if is_full_trajectory_plot:
                ax.set_title("Full Trajectory")
                for i in range(n_agents):
                    agent_key = f"agent_{i}"
                    if agent_key in normalized_sample_data["trajectories"]:
                        sample_agent_states = normalized_sample_data["trajectories"][agent_key]["states"]
                        if len(sample_agent_states) > 0:
                            traj = np.array(sample_agent_states[:step + 1])
                            color = other_agent_color
                            linewidth = 1.5
                            if i == 0:
                                color = ego_color
                                linewidth = 2.0
                            elif i in selected_agents_np:
                                color = selected_color
                            
                            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=linewidth, alpha=0.7)

                            final_pos = traj[-1, :2]
                            marker = '*' if i == 0 else 'o'
                            size = 120 if i == 0 else 60
                            ax.scatter(final_pos[0], final_pos[1], color=color, marker=marker, s=size, zorder=5, edgecolors='black', linewidth=0.5)
                            ax.text(final_pos[0] + 0.1, final_pos[1], str(i), fontsize=8, ha='left', va='center')
            else:
                ax.set_title(f"t = {int((step+1)/10)}s")
                for i in range(n_agents):
                    agent_key = f"agent_{i}"
                    if agent_key in normalized_sample_data["trajectories"]:
                        sample_agent_states = normalized_sample_data["trajectories"][agent_key]["states"]
                        if len(sample_agent_states) > step:
                            current_pos = np.array(sample_agent_states[step][:2])
                            color = other_agent_color
                            if i == 0:
                                color = ego_color
                            elif i in selected_agents_np:
                                color = selected_color
                            
                            marker = '*' if i == 0 else 'o'
                            size = 150 if i == 0 else 80
                            ax.scatter(current_pos[0], current_pos[1], color=color, marker=marker, s=size, zorder=5, edgecolors='black', linewidth=0.5)
                            ax.text(current_pos[0] + 0.1, current_pos[1], str(i), fontsize=8, ha='left', va='center')
            
            if col_idx == 0:
                ax.set_ylabel(psn_config['label'], fontsize=12, fontweight='bold')

    # --- 5. Finalize and Save ---
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', label='Ego Agent', markerfacecolor=ego_color, markersize=14),
        Line2D([0], [0], marker='o', color='w', label='Selected Agent', markerfacecolor=selected_color, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Other Agent', markerfacecolor=other_agent_color, markersize=10)
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=12)

    plt.tight_layout(rect=[0.02, 0, 0.98, 0.94])

    if output_dir is None:
        output_path = json_file_path.replace('.json', '_figure.png')
    else:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_figure.png")
    
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✓ Static figure saved successfully to: {output_path}")

def create_single_frame(step, results, normalized_sample_data, config, T_observation, T_total, n_agents, sample_id):
    """Create a single frame using the working logic from plot_all_frames.py"""
    
    # Color scheme
    ego_color = 'darkblue'
    other_agent_color = 'gray'
    selected_color = 'red'
    goal_prediction_color = 'orange'
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    
    # Determine phase and title
    if step < T_observation:
        phase = "Ground Truth Trajectories"
        step_title = f'Step {step+1}/{T_total} - {phase}'
    else:
        phase = "Receding Horizon with Models"
        iteration_idx = step - T_observation
        step_title = f'Step {step+1}/{T_total} - {phase} (Iteration {iteration_idx})'
    
    # Get selection info for this step
    selected_agents_np = np.array([0])  # Default: only ego agent
    if step >= T_observation:
        iteration_idx = step - T_observation
        if iteration_idx < len(results['receding_horizon_results']):
            iteration_result = results['receding_horizon_results'][iteration_idx]
            predicted_mask = jnp.array(iteration_result['predicted_mask'])
            
            # Apply selection logic
            selection_method = config.testing.receding_horizon.selection_method
            mask_threshold = config.testing.receding_horizon.mask_threshold
            rank = config.testing.receding_horizon.rank
            
            _, selected_agents, _ = select_agents_by_mask(
                predicted_mask, selection_method, mask_threshold, rank)
            
            # Include ego agent in selected agents
            selected_agents_np = np.array(selected_agents)
            if selected_agents_np.size == 0:
                selected_agents_np = np.array([0])
            elif 0 not in selected_agents_np:
                ego_array = np.array([0])
                if selected_agents_np.ndim == 0:
                    selected_agents_np = np.array([selected_agents_np])
                selected_agents_np = np.concatenate([ego_array, selected_agents_np])
                selected_agents_np = np.unique(selected_agents_np)

    ax.set_title(step_title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True, alpha=0.3)
    
    # Plot trajectories for all agents (gradually accumulated)
    for i in range(n_agents):
        agent_key = f"agent_{i}"
        if agent_key in normalized_sample_data["trajectories"]:
            sample_agent_states = normalized_sample_data["trajectories"][agent_key]["states"]
            if len(sample_agent_states) > 0:
                sample_traj = np.array(sample_agent_states[:step+1])
                is_selected = i in selected_agents_np
                
                if i == 0:
                    if len(sample_traj) > 1:
                        ax.plot(sample_traj[:, 0], sample_traj[:, 1], '--', 
                                 color='black', alpha=0.8, linewidth=2, 
                                 label=f'Ego Agent {i} (Ground Truth)')
                    if step >= T_observation and agent_key in results['final_game_state']['trajectories']:
                        agent_states = results['final_game_state']['trajectories'][agent_key]['states']
                        if len(agent_states) > 0:
                            agent_traj = np.array(agent_states[:step+1])
                            if len(agent_traj) > 1:
                                ax.plot(agent_traj[:, 0], agent_traj[:, 1], '-', 
                                         color='blue', alpha=0.9, linewidth=3, 
                                         label=f'Ego Agent {i} (Computed)')
                else:
                    if len(sample_traj) > 1:
                        if step >= T_observation and is_selected:
                            ax.plot(sample_traj[:, 0], sample_traj[:, 1], '-', 
                                     color=selected_color, alpha=0.8, linewidth=2, 
                                     label=f'Agent {i} (Selected)')
                        else:
                            ax.plot(sample_traj[:, 0], sample_traj[:, 1], '--', 
                                     color=other_agent_color, alpha=0.4, linewidth=1, 
                                     label=f'Agent {i} (Not Selected)')
    
    # Plot current positions
    for i in range(n_agents):
        agent_key = f"agent_{i}"
        is_selected = i in selected_agents_np
        
        if i == 0:
            if step >= T_observation and agent_key in results['final_game_state']['trajectories']:
                agent_states = results['final_game_state']['trajectories'][agent_key]['states']
                if len(agent_states) > 0 and step < len(agent_states):
                    current_pos = np.array(agent_states[step][:2])
                else:
                    agent_states = normalized_sample_data["trajectories"][agent_key]["states"]
                    current_pos = np.array(agent_states[step][:2])
            else:
                agent_states = normalized_sample_data["trajectories"][agent_key]["states"]
                current_pos = np.array(agent_states[step][:2])
            
            ax.plot(current_pos[0], current_pos[1], 'o', 
                     color=ego_color, markersize=10, alpha=0.8)
        else:
            if agent_key in normalized_sample_data["trajectories"]:
                agent_states = normalized_sample_data["trajectories"][agent_key]["states"]
                if len(agent_states) > 0 and step < len(agent_states):
                    current_pos = np.array(agent_states[step][:2])
                    
                    if step >= T_observation and is_selected:
                        ax.plot(current_pos[0], current_pos[1], 'o', 
                                 color=selected_color, markersize=8, alpha=0.8)
                        ax.text(current_pos[0] + 0.1, current_pos[1] + 0.1, f'{i}*', 
                                fontsize=10, ha='left', va='bottom', 
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                    else:
                        ax.plot(current_pos[0], current_pos[1], 'o', 
                                 color=other_agent_color, markersize=6, alpha=0.6)
                        ax.text(current_pos[0] + 0.1, current_pos[1] + 0.1, f'{i}', 
                                fontsize=10, ha='left', va='bottom', 
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Plot goals
    if 'target_positions' in normalized_sample_data:
        target_positions = normalized_sample_data['target_positions']
        for i in range(min(n_agents, len(target_positions))):
            goal_pos = np.array(target_positions[i][:2])
            if i == 0:
                ax.plot(goal_pos[0], goal_pos[1], 's', 
                         color=ego_color, markersize=12, alpha=0.8, 
                         label=f'Ego Agent {i} Goal (True)')
            else:
                ax.plot(goal_pos[0], goal_pos[1], 's', 
                         color=other_agent_color, markersize=8, alpha=0.6, 
                         label=f'Agent {i} Goal (True)')
    
    # Plot predicted goals
    if step >= T_observation:
        iteration_idx = step - T_observation
        if iteration_idx < len(results['receding_horizon_results']):
            iteration_result = results['receding_horizon_results'][iteration_idx]
            predicted_goals = iteration_result['predicted_goals']
            for i in range(n_agents):
                if i == 0:
                    ax.plot(predicted_goals[i][0], predicted_goals[i][1], '^', 
                             color=goal_prediction_color, markersize=10, alpha=0.8, 
                             label=f'Ego Agent {i} Goal (Predicted)')
                else:
                    ax.plot(predicted_goals[i][0], predicted_goals[i][1], '^', 
                             color=goal_prediction_color, markersize=8, alpha=0.6, 
                             label=f'Agent {i} Goal (Predicted)')
    
    plt.tight_layout()
    
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = Image.fromarray(buf)
    
    plt.close(fig)
    return img

def create_gif_from_frames(json_file_path, output_dir=None, fps=10):
    """Create GIF by generating all frames individually and combining them"""
    
    config = load_config()
    with open(json_file_path, 'r') as f:
        results = json.load(f)
    
    sample_id = results['sample_id']
    T_observation = results['T_observation']
    T_total = results['T_total']
    n_agents = 10
    normalized_sample_data = results['normalized_sample_data']
    
    if output_dir is None:
        output_path = json_file_path.replace('.json', '.gif')
    else:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.gif")
    
    print(f"Creating GIF with {T_total} frames...")
    
    frames = []
    for step in range(T_total):
        print(f"Generating frame {step + 1}/{T_total}")
        frame = create_single_frame(step, results, normalized_sample_data, config, 
                                   T_observation, T_total, n_agents, sample_id)
        frames.append(frame)
    
    print(f"Saving GIF to: {output_path}")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000//fps,
        loop=0
    )
    print(f"✓ GIF created successfully!")

def find_latest_test_run():
    """Find the most recent test run directory and return all JSON files in it"""
    log_dir = "log"
    test_dirs = []
    
    if not os.path.isdir(log_dir):
        return []
        
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith('.json') and 'receding_horizon_test_sample' in file:
                test_dirs.append(os.path.dirname(os.path.join(root, file)))
    
    if not test_dirs:
        return []
    
    unique_dirs = list(set(test_dirs))
    unique_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    latest_dir = unique_dirs[0]
    print(f"Found latest test run directory: {latest_dir}")
    
    json_files = []
    for file in os.listdir(latest_dir):
        if file.endswith('.json') and 'receding_horizon_test_sample' in file:
            json_files.append(os.path.join(latest_dir, file))
    
    json_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return json_files

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create visualizations from receding horizon test results.')
    parser.add_argument('--json_file', help='Path to a specific JSON file (if not provided, will process all samples from latest test run)')
    parser.add_argument('--output_dir', help='Output directory for visualizations (default: same as JSON files)')
    parser.add_argument('--mode', choices=['gif', 'figure'], default='gif', help='Type of visualization to create: an animated GIF or a static multi-panel figure.')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for GIF mode')
    
    args = parser.parse_args()

    # Determine which function to call based on mode
    if args.mode == 'gif':
        process_func = lambda json_path, out_dir: create_gif_from_frames(json_path, out_dir, args.fps)
        print("Mode: Creating GIFs")
    else: # mode is 'figure'
        process_func = create_static_figure
        print("Mode: Creating static figures")

    try:
        if args.json_file is not None:
            if not os.path.exists(args.json_file):
                print(f"Error: JSON file not found: {args.json_file}")
                sys.exit(1)
            print(f"Processing single JSON file: {args.json_file}")
            process_func(args.json_file, args.output_dir)
        else:
            print("Processing all samples from the latest test run...")
            json_files = find_latest_test_run()
            if not json_files:
                print("No JSON files found to process in the 'log' directory.")
                return

            print(f"Found {len(json_files)} samples to process")
            
            for i, json_file in enumerate(json_files):
                print(f"\n--- Processing sample {i+1}/{len(json_files)}: {os.path.basename(json_file)} ---")
                try:
                    process_func(json_file, args.output_dir)
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
                    continue
            
            print(f"\n✓ Completed processing {len(json_files)} samples")
            
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()