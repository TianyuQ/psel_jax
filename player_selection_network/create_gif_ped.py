#!/usr/bin/env python3
"""
Create GIF from Pedestrian Data Test Results

This script creates animated GIFs from pedestrian data test results, showing
the trajectory evolution over time with agent selection and goal prediction.

Author: Assistant
Date: 2024
"""

import json
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import sys
import os
import glob
from typing import Dict, Any, List, Tuple
from pathlib import Path

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


def create_single_frame(step, results, sample_data, config, T_observation, T_total, n_agents, sample_id):
    """Create a single frame for the pedestrian data visualization."""
    
    # Color scheme
    ego_color = 'darkblue'
    other_agent_color = 'gray'
    selected_color = 'red'
    goal_prediction_color = 'orange'
    goal_true_color = 'green'
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_aspect('equal')
    
    # Calculate bounds from all trajectory data
    all_positions = []
    for agent_idx in range(n_agents):
        agent_key = f"agent_{agent_idx}"
        if agent_key in sample_data["trajectories"]:
            agent_states = sample_data["trajectories"][agent_key]["states"]
            for state in agent_states:
                if len(state) >= 2:
                    all_positions.append(state[:2])
    
    if all_positions:
        all_positions = np.array(all_positions)
        x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
        y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
        
        # Add padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    else:
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
    
    # Get selected agents for this step
    selected_agents_np = np.array([0])  # Default: only ego agent
    if step >= T_observation and 'receding_horizon_results' in results:
        iteration_idx = step - T_observation
        if iteration_idx < len(results['receding_horizon_results']):
            iteration_result = results['receding_horizon_results'][iteration_idx]
            if 'selected_agents' in iteration_result:
                selected_agents_np = np.array(iteration_result['selected_agents'])
                if selected_agents_np.size == 0:
                    selected_agents_np = np.array([0])
                elif 0 not in selected_agents_np:
                    ego_array = np.array([0])
                    if selected_agents_np.ndim == 0:
                        selected_agents_np = np.array([selected_agents_np])
                    selected_agents_np = np.concatenate([ego_array, selected_agents_np])
                    selected_agents_np = np.unique(selected_agents_np)

    # Plot trajectories up to current step
    for agent_idx in range(n_agents):
        agent_key = f"agent_{agent_idx}"
        if agent_key in sample_data["trajectories"]:
            agent_states = sample_data["trajectories"][agent_key]["states"]
            
            # Plot trajectory up to current step
            if step > 0 and len(agent_states) > step:
                traj_positions = np.array(agent_states[:step+1])[:, :2]
                is_selected = agent_idx in selected_agents_np
                
                if agent_idx == 0:  # Ego agent
                    # Plot ground truth trajectory (black dashed)
                    ax.plot(traj_positions[:, 0], traj_positions[:, 1], '--', 
                           color='black', alpha=0.8, linewidth=2, 
                           label='Ego Agent (Ground Truth)' if step == 0 else "")
                    
                    # Plot computed trajectory (blue solid) if available
                    if step >= T_observation and 'final_game_state' in results:
                        if agent_key in results['final_game_state']['trajectories']:
                            computed_states = results['final_game_state']['trajectories'][agent_key]['states']
                            if len(computed_states) > step:
                                computed_traj = np.array(computed_states[:step+1])[:, :2]
                                ax.plot(computed_traj[:, 0], computed_traj[:, 1], '-', 
                                       color='blue', alpha=0.9, linewidth=3, 
                                       label='Ego Agent (Computed)' if step == T_observation else "")
                else:  # Other agents
                    if step >= T_observation and is_selected:
                        # Selected agent: solid line, red color
                        ax.plot(traj_positions[:, 0], traj_positions[:, 1], '-', 
                               color=selected_color, alpha=0.8, linewidth=2, 
                               label=f'Agent {agent_idx} (Selected)' if step == T_observation else "")
                    else:
                        # Non-selected agent: dashed line, gray color
                        ax.plot(traj_positions[:, 0], traj_positions[:, 1], '--', 
                               color=other_agent_color, alpha=0.4, linewidth=1, 
                               label=f'Agent {agent_idx} (Not Selected)' if step == T_observation else "")
    
    # Plot current positions
    for agent_idx in range(n_agents):
        agent_key = f"agent_{agent_idx}"
        if agent_key in sample_data["trajectories"]:
            agent_states = sample_data["trajectories"][agent_key]["states"]
            
            if len(agent_states) > step:
                pos = agent_states[step][:2]  # x, y position
                vel = agent_states[step][2:4] if len(agent_states[step]) > 3 else [0, 0]  # vx, vy
                is_selected = agent_idx in selected_agents_np
                
                if agent_idx == 0:  # Ego agent
                    # Plot ground truth position
                    ax.scatter(pos[0], pos[1], c='black', s=100, marker='o', 
                              edgecolors='black', linewidth=2, zorder=5, alpha=0.8)
                    
                    # Plot computed position if available
                    if step >= T_observation and 'final_game_state' in results:
                        if agent_key in results['final_game_state']['trajectories']:
                            computed_states = results['final_game_state']['trajectories'][agent_key]['states']
                            if len(computed_states) > step:
                                computed_pos = np.array(computed_states[step][:2])
                                ax.scatter(computed_pos[0], computed_pos[1], c='blue', s=100, marker='s', 
                                          edgecolors='blue', linewidth=2, zorder=6, alpha=0.9)
                    
                    # Velocity arrow
                    if np.linalg.norm(vel) > 0.1:
                        ax.arrow(pos[0], pos[1], vel[0]*0.5, vel[1]*0.5, 
                                head_width=0.2, head_length=0.2, fc='black', ec='black')
                else:  # Other agents
                    if step >= T_observation and is_selected:
                        # Selected agent: red marker with larger size
                        ax.scatter(pos[0], pos[1], c=selected_color, s=80, marker='o', 
                                  edgecolors='black', linewidth=2, zorder=4, alpha=0.8)
                        # Add text label with selection indicator
                        ax.text(pos[0] + 0.1, pos[1] + 0.1, f'{agent_idx}*', 
                                fontsize=10, ha='left', va='bottom', 
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                    else:
                        # Non-selected agent: gray marker
                        ax.scatter(pos[0], pos[1], c=other_agent_color, s=60, marker='o', 
                                  edgecolors='black', linewidth=1, zorder=4, alpha=0.6)
                        # Add text label
                        ax.text(pos[0] + 0.1, pos[1] + 0.1, f'{agent_idx}', 
                                fontsize=10, ha='left', va='bottom', 
                                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                    
                    # Velocity arrow
                    if np.linalg.norm(vel) > 0.1:
                        arrow_color = selected_color if (step >= T_observation and is_selected) else other_agent_color
                        ax.arrow(pos[0], pos[1], vel[0]*0.3, vel[1]*0.3, 
                                head_width=0.15, head_length=0.15, fc=arrow_color, ec=arrow_color)
    
    # Plot goals
    if 'target_positions' in sample_data:
        goals = sample_data['target_positions']
        for i, goal in enumerate(goals):
            if i == 0:  # Ego agent goal
                ax.scatter(goal[0], goal[1], c=goal_true_color, s=150, marker='*', 
                          edgecolors='black', linewidth=2, zorder=6, label='Goals' if step == 0 else "")
            else:  # Other agent goals
                ax.scatter(goal[0], goal[1], c=goal_true_color, s=100, marker='*', 
                          edgecolors='black', linewidth=1, zorder=6)
    
    # Add agent selection information if available
    if step >= T_observation and 'receding_horizon_results' in results:
        # Find the corresponding iteration result
        iteration = step - T_observation
        if iteration < len(results['receding_horizon_results']):
            iter_result = results['receding_horizon_results'][iteration]
            
            # Show selected agents
            if 'predicted_mask' in iter_result and 'num_selected' in iter_result:
                num_selected = iter_result['num_selected']
                mask_sparsity = iter_result.get('mask_sparsity', 0.0)
                
                # Add text box with selection info
                textstr = f'Step: {step}\nSelected: {num_selected}/{n_agents-1}\nSparsity: {mask_sparsity:.2f}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)
    
    # Set title and labels
    ax.set_title(f'Pedestrian Data Sample {sample_id} - Step {step}/{T_total-1}', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    if step == 0:
        ax.legend(loc='upper right', fontsize=10)
    
    return fig


def create_gif_from_pedestrian_results(json_file_path, output_dir=None, fps=5):
    """
    Create GIF from pedestrian data test results.
    
    Args:
        json_file_path: Path to the test result JSON file
        output_dir: Directory to save the GIF (default: same as JSON file)
        fps: Frames per second for the GIF
    """
    print(f"Creating GIF for pedestrian data: {json_file_path}")
    
    # Load data
    config = load_config()
    with open(json_file_path, 'r') as f:
        results = json.load(f)
    
    # Extract parameters
    T_observation = results['T_observation']
    T_total = results['T_total']
    n_agents = results.get('n_agents', 10)
    sample_id = results['sample_id']
    
    # Get sample data from results
    sample_data = results.get('normalized_sample_data', {})
    if not sample_data:
        print("Warning: No normalized sample data found in results")
        return
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(json_file_path)
    
    # Create frames directory
    frames_dir = os.path.join(output_dir, f"frames_sample_{sample_id:03d}")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Create frames
    print(f"Creating {T_total} frames...")
    frame_files = []
    
    for step in range(T_total):
        fig = create_single_frame(step, results, sample_data, config, T_observation, T_total, n_agents, sample_id)
        
        # Save frame
        frame_file = os.path.join(frames_dir, f"frame_{step:03d}.png")
        fig.savefig(frame_file, dpi=100, bbox_inches='tight')
        frame_files.append(frame_file)
        
        plt.close(fig)
        
        if step % 10 == 0:
            print(f"  Created frame {step}/{T_total-1}")
    
    # Create GIF
    print("Creating GIF...")
    gif_file = os.path.join(output_dir, f"pedestrian_sample_{sample_id:03d}.gif")
    
    # Load images and create GIF
    images = []
    for frame_file in frame_files:
        img = Image.open(frame_file)
        images.append(img)
    
    # Save GIF
    images[0].save(
        gif_file,
        save_all=True,
        append_images=images[1:],
        duration=1000//fps,  # Duration in milliseconds
        loop=0
    )
    
    print(f"GIF saved to: {gif_file}")
    
    # Clean up frame files
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(frames_dir)
    
    return gif_file


def create_all_gifs(results_dir, output_dir=None, fps=5):
    """
    Create GIFs for all pedestrian data test results.
    
    Args:
        results_dir: Directory containing test result JSON files
        output_dir: Directory to save GIFs (default: same as results_dir)
        fps: Frames per second for the GIFs
    """
    if output_dir is None:
        output_dir = results_dir
    
    # Find all test result files
    pattern = os.path.join(results_dir, "psn_ped_test_sample_*.json")
    json_files = sorted(glob.glob(pattern))
    
    if not json_files:
        print(f"No psn_ped_test_sample_*.json files found in {results_dir}")
        return
    
    print(f"Found {len(json_files)} test result files")
    
    gif_files = []
    for json_file in json_files:
        try:
            gif_file = create_gif_from_pedestrian_results(json_file, output_dir, fps)
            gif_files.append(gif_file)
        except Exception as e:
            print(f"Error creating GIF for {json_file}: {e}")
    
    print(f"\nCreated {len(gif_files)} GIFs:")
    for gif_file in gif_files:
        print(f"  {gif_file}")


def find_pedestrian_test_directories(base_dir: str = None) -> List[Dict[str, str]]:
    """
    Find all pedestrian test result directories following the new structure.
    
    Args:
        base_dir: Base directory to search (if None, searches both baseline and PSN directories)
    
    Returns:
        List of dictionaries with test configuration info and directory paths
    """
    test_configs = []
    
    if base_dir is None:
        # Search both baseline and PSN directories
        search_dirs = ["baseline_results", "log/goal_true_N_10_T_50_obs_10"]
    else:
        search_dirs = [base_dir]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        # Look for baseline results
        if search_dir == "baseline_results":
            # Look for ped_test subdirectory first
            ped_test_dir = os.path.join(search_dir, "ped_test")
            if os.path.exists(ped_test_dir):
                for ped_dir in os.listdir(ped_test_dir):
                    ped_path = os.path.join(ped_test_dir, ped_dir)
                    if os.path.isdir(ped_path) and ped_dir.startswith("ped_results_"):
                        # Check if it contains test result files
                        json_files = glob.glob(os.path.join(ped_path, "psn_ped_test_sample_*.json"))
                        if json_files:
                            test_configs.append({
                                'name': f'baseline_{ped_dir}',
                                'directory': ped_path,
                                'psn_model': f'baseline_{ped_dir}',
                                'sample_count': len(json_files),
                                'test_type': 'baseline'
                            })
            else:
                # Fallback: look directly in baseline_results
                for ped_dir in os.listdir(search_dir):
                    ped_path = os.path.join(search_dir, ped_dir)
                    if os.path.isdir(ped_path) and ped_dir.startswith("ped_results_"):
                        # Check if it contains test result files
                        json_files = glob.glob(os.path.join(ped_path, "psn_ped_test_sample_*.json"))
                        if json_files:
                            test_configs.append({
                                'name': f'baseline_{ped_dir}',
                                'directory': ped_path,
                                'psn_model': f'baseline_{ped_dir}',
                                'sample_count': len(json_files),
                                'test_type': 'baseline'
                            })
        
        # Look for PSN results
        else:
            for psn_dir in os.listdir(search_dir):
                psn_path = os.path.join(search_dir, psn_dir)
                if os.path.isdir(psn_path) and ('psn_gru' in psn_dir or 'psn_' in psn_dir):
                    # Look for ped_results_* directories
                    for result_dir in os.listdir(psn_path):
                        result_path = os.path.join(psn_path, result_dir)
                        if os.path.isdir(result_path) and result_dir.startswith('ped_results_'):
                            # Check if it contains test result files
                            json_files = glob.glob(os.path.join(result_path, "psn_ped_test_sample_*.json"))
                            if json_files:
                                test_configs.append({
                                    'name': f'psn_{result_dir}',
                                    'directory': result_path,
                                    'psn_model': psn_dir,
                                    'sample_count': len(json_files),
                                    'test_type': 'psn'
                                })
    
    return test_configs


def create_output_directory_from_config():
    """Create output directory based on config.yaml settings."""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_loader import load_config
    config = load_config()
    
    if config.testing.receding_horizon.use_baseline:
        # Baseline results directory
        base_dir = "baseline_results"
        goal_source = "true_goals"
        n_agents = config.game.N_agents
        baseline_mode = config.testing.receding_horizon.baseline_mode.replace(" ", "_").lower()
        baseline_param = config.testing.receding_horizon.baseline_parameter
        
        # Directory structure: ped_test/ped_results_{n_agents}_{method}_param_{param}_{goal_suffix}
        goal_suffix = "goal_inference" if goal_source == "goal_inference" else "goal_true"
        output_dir = os.path.join(base_dir, "ped_test", f"ped_results_{n_agents}_{baseline_mode}_param_{baseline_param}_{goal_suffix}")
    else:
        # PSN results directory
        base_dir = "log/goal_true_N_10_T_50_obs_10"
        psn_dir = "psn_gru_partial_true_goals_N_10_T_50_obs_10_lr_0.002_bs_32_sigma1_0.075_sigma2_0.075_epochs_100"
        goal_source = "true_goals"
        n_agents = config.game.N_agents
        test_type = config.testing.receding_horizon.test_type
        baseline_mode = "psn"
        
        # Directory structure: ped_results_{n_agents}_{test_type}_goal_true_{obs_type}_{method_suffix}_{psn_model_name}
        goal_suffix = "goal_inference" if goal_source == "goal_inference" else "goal_true"
        obs_type = "partial"  # Since we're using partial PSN model
        
        # Get selection method and parameters from config
        selection_method = config.testing.receding_horizon.selection_method
        if selection_method == "rank":
            rank = config.testing.receding_horizon.rank
            method_suffix = f"rank_{rank}"
        else:  # threshold
            mask_threshold = config.testing.receding_horizon.mask_threshold
            method_suffix = f"threshold_{mask_threshold}"
        
        psn_model_name = "psn_best_model"  # Use the actual model name from the directory
        output_dir = os.path.join(base_dir, psn_dir, f"ped_results_{n_agents}_{test_type}_goal_true_{obs_type}_{method_suffix}_{psn_model_name}")
    
    return output_dir

def main():
    """Main function."""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='Create GIFs from pedestrian data test results')
    parser.add_argument('--results_dir', type=str, 
                       default=None,
                       help='Base directory containing test result directories (if None, uses config-based directory)')
    parser.add_argument('--config_name', type=str, default=None,
                       help='Specific configuration name to create GIFs for (if None, uses first found)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save GIFs (default: same as results_dir)')
    parser.add_argument('--fps', type=int, default=5,
                       help='Frames per second for the GIFs')
    parser.add_argument('--sample_id', type=int, default=None,
                       help='Create GIF for specific sample ID only')
    parser.add_argument('--list_configs', action='store_true',
                       help='List available configurations and exit')
    
    args = parser.parse_args()
    
    # Use config-based directory by default
    if args.results_dir is None:
        args.results_dir = create_output_directory_from_config()
        print(f"Using config-based directory: {args.results_dir}")
    
    # Find test configurations
    test_configs = find_pedestrian_test_directories(args.results_dir)
    
    if not test_configs:
        # Check if this is a direct results directory with test files
        if os.path.exists(args.results_dir) and any(f.startswith('psn_ped_test_sample_') for f in os.listdir(args.results_dir)):
            # This is a direct results directory, create a single config
            test_configs = [{
                'name': 'config_based',
                'directory': args.results_dir,
                'psn_model': 'config_based',
                'sample_count': len([f for f in os.listdir(args.results_dir) if f.startswith('psn_ped_test_sample_')]),
                'test_type': 'psn'
            }]
        else:
            print(f"No pedestrian test configurations found in {args.results_dir}")
            return
    
    if args.list_configs:
        print("Available configurations:")
        for i, config in enumerate(test_configs):
            print(f"  {i+1}. {config['name']} ({config['sample_count']} samples)")
            print(f"     Directory: {config['directory']}")
        return
    
    # Select configuration
    if args.config_name:
        target_config = None
        for config in test_configs:
            if config['name'] == args.config_name:
                target_config = config
                break
        if target_config is None:
            print(f"Configuration '{args.config_name}' not found.")
            print(f"Available configurations: {[c['name'] for c in test_configs]}")
            return
    else:
        target_config = test_configs[0]  # Use first found configuration
        print(f"Using configuration: {target_config['name']}")
    
    results_dir = target_config['directory']
    
    if args.sample_id is not None:
        # Create GIF for specific sample
        json_file = os.path.join(results_dir, f"psn_ped_test_sample_{args.sample_id:03d}.json")
        if os.path.exists(json_file):
            create_gif_from_pedestrian_results(json_file, args.output_dir, args.fps)
        else:
            print(f"Sample {args.sample_id} not found: {json_file}")
    else:
        # Create GIFs for all samples
        create_all_gifs(results_dir, args.output_dir, args.fps)


if __name__ == "__main__":
    main()
