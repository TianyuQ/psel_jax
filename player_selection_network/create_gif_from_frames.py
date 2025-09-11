#!/usr/bin/env python3

import json
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    
    # Add selection info to title
    # if step >= T_observation:
    #     selected_list = list(selected_agents_np)
    #     step_title += f'\nSelected agents: {selected_list}'
    
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
                # Only plot trajectory up to current step (gradual accumulation)
                sample_traj = np.array(sample_agent_states[:step+1])
                
                is_selected = i in selected_agents_np
                
                if i == 0:  # Ego agent
                    # Plot ground truth trajectory (black dashed) - only up to current step
                    if len(sample_traj) > 1:  # Need at least 2 points to plot
                        ax.plot(sample_traj[:, 0], sample_traj[:, 1], '--', 
                                 color='black', alpha=0.8, linewidth=2, 
                                 label=f'Ego Agent {i} (Ground Truth)')
                    
                    # Plot computed trajectory (blue solid) if available - only up to current step
                    if step >= T_observation and agent_key in results['final_game_state']['trajectories']:
                        agent_states = results['final_game_state']['trajectories'][agent_key]['states']
                        if len(agent_states) > 0:
                            agent_traj = np.array(agent_states[:step+1])
                            if len(agent_traj) > 1:  # Need at least 2 points to plot
                                ax.plot(agent_traj[:, 0], agent_traj[:, 1], '-', 
                                         color='blue', alpha=0.9, linewidth=3, 
                                         label=f'Ego Agent {i} (Computed)')
                else:  # Other agents
                    if len(sample_traj) > 1:  # Need at least 2 points to plot
                        if step >= T_observation and is_selected:
                            # Selected agent: solid line, red color (only during selection period)
                            ax.plot(sample_traj[:, 0], sample_traj[:, 1], '-', 
                                     color=selected_color, alpha=0.8, linewidth=2, 
                                     label=f'Agent {i} (Selected)')
                        else:
                            # Non-selected agent: dashed line, gray color
                            ax.plot(sample_traj[:, 0], sample_traj[:, 1], '--', 
                                     color=other_agent_color, alpha=0.4, linewidth=1, 
                                     label=f'Agent {i} (Not Selected)')
    
    # Plot current positions
    for i in range(n_agents):
        agent_key = f"agent_{i}"
        is_selected = i in selected_agents_np
        
        if i == 0:
            # Ego agent: use computed trajectory from game state if available
            if step >= T_observation and agent_key in results['final_game_state']['trajectories']:
                agent_states = results['final_game_state']['trajectories'][agent_key]['states']
                if len(agent_states) > 0 and step < len(agent_states):
                    current_pos = np.array(agent_states[step][:2])
                else:
                    # Fallback to ground truth
                    agent_states = normalized_sample_data["trajectories"][agent_key]["states"]
                    current_pos = np.array(agent_states[step][:2])
            else:
                # Use ground truth during observation phase
                agent_states = normalized_sample_data["trajectories"][agent_key]["states"]
                current_pos = np.array(agent_states[step][:2])
            
            ax.plot(current_pos[0], current_pos[1], 'o', 
                     color=ego_color, markersize=10, alpha=0.8)
        else:
            # Other agents: use ground truth trajectories
            if agent_key in normalized_sample_data["trajectories"]:
                agent_states = normalized_sample_data["trajectories"][agent_key]["states"]
                if len(agent_states) > 0 and step < len(agent_states):
                    current_pos = np.array(agent_states[step][:2])
                    
                    if step >= T_observation and is_selected:
                        ax.plot(current_pos[0], current_pos[1], 'o', 
                                 color=selected_color, markersize=8, alpha=0.8)
                        # Add text label with selection indicator
                        ax.text(current_pos[0] + 0.1, current_pos[1] + 0.1, f'{i}*', 
                                fontsize=10, ha='left', va='bottom', 
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                    else:
                        ax.plot(current_pos[0], current_pos[1], 'o', 
                                 color=other_agent_color, markersize=6, alpha=0.6)
                        # Add text label
                        ax.text(current_pos[0] + 0.1, current_pos[1] + 0.1, f'{i}', 
                                fontsize=10, ha='left', va='bottom', 
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Plot goals (all gray, no red coloring)
    if 'target_positions' in normalized_sample_data:
        target_positions = normalized_sample_data['target_positions']
        for i in range(min(n_agents, len(target_positions))):
            goal_pos = np.array(target_positions[i][:2])
            
            if i == 0:  # Ego agent goal
                ax.plot(goal_pos[0], goal_pos[1], 's', 
                         color=ego_color, markersize=12, alpha=0.8, 
                         label=f'Ego Agent {i} Goal (True)')
            else:  # Other agent goals - always gray
                ax.plot(goal_pos[0], goal_pos[1], 's', 
                         color=other_agent_color, markersize=8, alpha=0.6, 
                         label=f'Agent {i} Goal (True)')
    
    # Plot predicted goals if available
    if step >= T_observation:
        iteration_idx = step - T_observation
        if iteration_idx < len(results['receding_horizon_results']):
            iteration_result = results['receding_horizon_results'][iteration_idx]
            predicted_goals = iteration_result['predicted_goals']
            
            for i in range(n_agents):
                if i == 0:  # Ego agent predicted goal
                    ax.plot(predicted_goals[i][0], predicted_goals[i][1], '^', 
                             color=goal_prediction_color, markersize=10, alpha=0.8, 
                             label=f'Ego Agent {i} Goal (Predicted)')
                else:  # Other agent predicted goals
                    ax.plot(predicted_goals[i][0], predicted_goals[i][1], '^', 
                             color=goal_prediction_color, markersize=8, alpha=0.6, 
                             label=f'Agent {i} Goal (Predicted)')
    
    # No legend for any frames to keep it clean
    # (removed the legend code completely)
    
    plt.tight_layout()
    
    # Convert to PIL Image
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = Image.fromarray(buf)
    
    plt.close(fig)
    return img

def create_gif_from_frames(json_file_path, output_dir=None, fps=10):
    """Create GIF by generating all frames individually and combining them"""
    
    # Load config
    config = load_config()
    
    # Load results
    with open(json_file_path, 'r') as f:
        results = json.load(f)
    
    # Extract parameters
    sample_id = results['sample_id']
    T_observation = results['T_observation']
    T_total = results['T_total']
    n_agents = 10
    
    # Get the normalized sample data from the test results
    normalized_sample_data = results['normalized_sample_data']
    
    # Set up output path
    if output_dir is None:
        # Use the same directory as the JSON file
        output_path = json_file_path.replace('.json', '.gif')
    else:
        # Use the provided output directory
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.gif")
    
    print(f"Creating GIF with {T_total} frames...")
    
    # Generate all frames
    frames = []
    for step in range(T_total):
        print(f"Generating frame {step + 1}/{T_total}")
        frame = create_single_frame(step, results, normalized_sample_data, config, 
                                   T_observation, T_total, n_agents, sample_id)
        frames.append(frame)
    
    # Save as GIF
    print(f"Saving GIF to: {output_path}")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000//fps,  # Convert fps to duration in ms
        loop=0
    )
    print(f"✓ GIF created successfully!")

def find_latest_test_run():
    """Find the most recent test run directory and return all JSON files in it"""
    log_dir = "log"
    test_dirs = []
    
    # Find all test result directories
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith('.json') and 'receding_horizon_test_sample' in file:
                test_dirs.append(os.path.dirname(os.path.join(root, file)))
    
    if not test_dirs:
        print("Error: No receding horizon test JSON files found in log directory")
        sys.exit(1)
    
    # Get unique directories and sort by modification time
    unique_dirs = list(set(test_dirs))
    unique_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Get the most recent test run directory
    latest_dir = unique_dirs[0]
    print(f"Found latest test run directory: {latest_dir}")
    
    # Find all JSON files in this directory
    json_files = []
    for file in os.listdir(latest_dir):
        if file.endswith('.json') and 'receding_horizon_test_sample' in file:
            json_files.append(os.path.join(latest_dir, file))
    
    # Sort by sample number
    json_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    return json_files

def find_latest_json_file():
    """Find the most recent receding horizon test JSON file (for backward compatibility)"""
    json_files = find_latest_test_run()
    return json_files[0] if json_files else None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create GIFs from receding horizon test results using frame-by-frame approach')
    parser.add_argument('--json_file', help='Path to a specific JSON file (if not provided, will process all samples from latest test run)')
    parser.add_argument('--output_dir', help='Output directory for GIFs (default: same as JSON files)')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for GIF')
    parser.add_argument('--all_samples', action='store_true', help='Process all samples from the latest test run (default behavior)')
    
    args = parser.parse_args()
    
    try:
        if args.json_file is not None:
            # Process single JSON file
            if not os.path.exists(args.json_file):
                print(f"Error: JSON file not found: {args.json_file}")
                sys.exit(1)
            print(f"Processing single JSON file: {args.json_file}")
            create_gif_from_frames(args.json_file, args.output_dir, args.fps)
        else:
            # Process all samples from latest test run
            print("Processing all samples from the latest test run...")
            json_files = find_latest_test_run()
            print(f"Found {len(json_files)} samples to process")
            
            for i, json_file in enumerate(json_files):
                print(f"\n--- Processing sample {i+1}/{len(json_files)}: {os.path.basename(json_file)} ---")
                try:
                    create_gif_from_frames(json_file, args.output_dir, args.fps)
                except Exception as e:
                    print(f"Error creating GIF for {json_file}: {e}")
                    continue
            
            print(f"\n✓ Completed processing {len(json_files)} samples")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()