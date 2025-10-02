#!/usr/bin/env python3
"""
Create Static Figures from Pedestrian Data Test Results

This script creates static multi-panel figures from pedestrian data test results,
showing key moments in the trajectory evolution with agent selection and goal prediction.

Author: Assistant
Date: 2024
"""

import json
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import os
import glob
from pathlib import Path
from typing import Dict, Any, List, Tuple

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
    Creates a static multi-panel figure from key moments in the pedestrian data trajectory.
    Shows observation period, early planning, mid planning, and final planning stages.
    """
    print(f"Creating static figure for pedestrian data: {json_file_path}")

    # Load data
    config = load_config()
    with open(json_file_path, 'r') as f:
        results = json.load(f)

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

    # Select key time steps for visualization
    key_steps = [
        T_observation - 1,  # End of observation period
        T_observation + (T_total - T_observation) // 4,  # 25% through planning
        T_observation + (T_total - T_observation) // 2,  # 50% through planning
        T_observation + 3 * (T_total - T_observation) // 4,  # 75% through planning
        T_total - 1  # Final step
    ]
    
    # Ensure steps are within bounds
    key_steps = [min(step, T_total - 1) for step in key_steps]
    key_steps = list(dict.fromkeys(key_steps))  # Remove duplicates while preserving order
    
    # Color scheme
    ego_color = 'darkblue'
    other_agent_color = 'gray'
    selected_color = 'red'
    goal_true_color = 'green'
    trajectory_color = 'lightblue'
    
    # Create figure with subplots
    n_panels = len(key_steps)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
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
        x_lim = [x_min - x_padding, x_max + x_padding]
        y_lim = [y_min - y_padding, y_max + y_padding]
    else:
        x_lim = [-5, 5]
        y_lim = [-5, 5]
    
    for i, step in enumerate(key_steps):
        if i >= len(axes):
            break
            
        ax = axes[i]
        ax.set_aspect('equal')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        
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

        # Plot complete trajectories up to current step
        for agent_idx in range(n_agents):
            agent_key = f"agent_{agent_idx}"
            if agent_key in sample_data["trajectories"]:
                agent_states = sample_data["trajectories"][agent_key]["states"]
                
                if len(agent_states) > step:
                    # Plot trajectory up to current step
                    traj_positions = np.array(agent_states[:step+1])[:, :2]
                    is_selected = agent_idx in selected_agents_np
                    
                    if agent_idx == 0:  # Ego agent
                        # Plot ground truth trajectory (black dashed)
                        ax.plot(traj_positions[:, 0], traj_positions[:, 1], '--', 
                               color='black', alpha=0.8, linewidth=2, 
                               label='Ego Agent (Ground Truth)' if i == 0 else "")
                        
                        # Plot computed trajectory (blue solid) if available
                        if step >= T_observation and 'final_game_state' in results:
                            if agent_key in results['final_game_state']['trajectories']:
                                computed_states = results['final_game_state']['trajectories'][agent_key]['states']
                                if len(computed_states) > step:
                                    computed_traj = np.array(computed_states[:step+1])[:, :2]
                                    ax.plot(computed_traj[:, 0], computed_traj[:, 1], '-', 
                                           color='blue', alpha=0.9, linewidth=3, 
                                           label='Ego Agent (Computed)' if i == 0 else "")
                    else:  # Other agents
                        if step >= T_observation and is_selected:
                            # Selected agent: solid line, red color
                            ax.plot(traj_positions[:, 0], traj_positions[:, 1], '-', 
                                   color=selected_color, alpha=0.8, linewidth=2, 
                                   label=f'Agent {agent_idx} (Selected)' if i == 0 else "")
                        else:
                            # Non-selected agent: dashed line, gray color
                            ax.plot(traj_positions[:, 0], traj_positions[:, 1], '--', 
                                   color=other_agent_color, alpha=0.4, linewidth=1, 
                                   label=f'Agent {agent_idx} (Not Selected)' if i == 0 else "")
        
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
                        ax.scatter(pos[0], pos[1], c='black', s=150, marker='o', 
                                  edgecolors='black', linewidth=2, zorder=5, alpha=0.8)
                        
                        # Plot computed position if available
                        if step >= T_observation and 'final_game_state' in results:
                            if agent_key in results['final_game_state']['trajectories']:
                                computed_states = results['final_game_state']['trajectories'][agent_key]['states']
                                if len(computed_states) > step:
                                    computed_pos = np.array(computed_states[step][:2])
                                    ax.scatter(computed_pos[0], computed_pos[1], c='blue', s=150, marker='s', 
                                              edgecolors='blue', linewidth=2, zorder=6, alpha=0.9)
                        
                        # Velocity arrow
                        if np.linalg.norm(vel) > 0.1:
                            ax.arrow(pos[0], pos[1], vel[0]*0.5, vel[1]*0.5, 
                                    head_width=0.3, head_length=0.3, fc='black', ec='black', linewidth=2)
                    else:  # Other agents
                        if step >= T_observation and is_selected:
                            # Selected agent: red marker with larger size
                            ax.scatter(pos[0], pos[1], c=selected_color, s=100, marker='o', 
                                      edgecolors='black', linewidth=2, zorder=4, alpha=0.8)
                            # Add text label with selection indicator
                            ax.text(pos[0] + 0.1, pos[1] + 0.1, f'{agent_idx}*', 
                                    fontsize=10, ha='left', va='bottom', 
                                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                        else:
                            # Non-selected agent: gray marker
                            ax.scatter(pos[0], pos[1], c=other_agent_color, s=80, marker='o', 
                                      edgecolors='black', linewidth=1, zorder=4, alpha=0.6)
                            # Add text label
                            ax.text(pos[0] + 0.1, pos[1] + 0.1, f'{agent_idx}', 
                                    fontsize=10, ha='left', va='bottom', 
                                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                        
                        # Velocity arrow
                        if np.linalg.norm(vel) > 0.1:
                            arrow_color = selected_color if (step >= T_observation and is_selected) else other_agent_color
                            ax.arrow(pos[0], pos[1], vel[0]*0.3, vel[1]*0.3, 
                                    head_width=0.2, head_length=0.2, fc=arrow_color, ec=arrow_color)
        
        # Plot goals
        if 'target_positions' in sample_data:
            goals = sample_data['target_positions']
            for j, goal in enumerate(goals):
                if j == 0:  # Ego agent goal
                    ax.scatter(goal[0], goal[1], c=goal_true_color, s=200, marker='*', 
                              edgecolors='black', linewidth=2, zorder=6, label='Goals' if i == 0 else "")
                else:  # Other agent goals
                    ax.scatter(goal[0], goal[1], c=goal_true_color, s=120, marker='*', 
                              edgecolors='black', linewidth=1, zorder=6)
        
        # Add agent selection information if available
        if step >= T_observation and 'receding_horizon_results' in results:
            iteration = step - T_observation
            if iteration < len(results['receding_horizon_results']):
                iter_result = results['receding_horizon_results'][iteration]
                
                if 'predicted_mask' in iter_result and 'num_selected' in iter_result:
                    num_selected = iter_result['num_selected']
                    mask_sparsity = iter_result.get('mask_sparsity', 0.0)
                    
                    # Add text box with selection info
                    textstr = f'Selected: {num_selected}/{n_agents-1}\nSparsity: {mask_sparsity:.2f}'
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                           verticalalignment='top', bbox=props)
        
        # Set title and labels
        if step < T_observation:
            phase = "Observation"
        else:
            phase = "Planning"
        
        ax.set_title(f'{phase} - Step {step}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position (m)', fontsize=10)
        ax.set_ylabel('Y Position (m)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    # Hide unused subplots
    for i in range(len(key_steps), len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'Pedestrian Data Sample {sample_id} - Trajectory Evolution\n'
                f'Total Steps: {T_total}, Agents: {n_agents}, Observation: {T_observation} steps', 
                fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save figure
    output_file = os.path.join(output_dir, f"pedestrian_sample_{sample_id:03d}_static.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Static figure saved to: {output_file}")
    
    plt.close(fig)
    return output_file


def create_summary_figure(results_dir, output_dir=None):
    """
    Create a summary figure showing all pedestrian samples side by side.
    """
    print(f"Creating summary figure for all pedestrian samples in {results_dir}")
    
    # Find all test result files
    pattern = os.path.join(results_dir, "psn_ped_test_sample_*.json")
    json_files = sorted(glob.glob(pattern))
    
    if not json_files:
        print(f"No psn_ped_test_sample_*.json files found in {results_dir}")
        return
    
    # Load all results
    all_results = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            results = json.load(f)
            all_results.append(results)
    
    # Calculate global bounds from all samples
    all_positions = []
    for results in all_results:
        sample_data = results.get('normalized_sample_data', {})
        if sample_data:
            n_agents = results.get('n_agents', 10)
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
        global_x_lim = [x_min - x_padding, x_max + x_padding]
        global_y_lim = [y_min - y_padding, y_max + y_padding]
    else:
        global_x_lim = [-5, 5]
        global_y_lim = [-5, 5]
    
    # Create figure
    n_samples = len(all_results)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, results in enumerate(all_results):
        if i >= len(axes):
            break
            
        ax = axes[i]
        sample_id = results['sample_id']
        T_total = results['T_total']
        n_agents = results.get('n_agents', 10)
        
        # Set consistent bounds for all subplots
        ax.set_xlim(global_x_lim)
        ax.set_ylim(global_y_lim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Get sample data
        sample_data = results.get('normalized_sample_data', {})
        if not sample_data:
            continue
        
        # Plot final trajectories for all agents
        for agent_idx in range(n_agents):
            agent_key = f"agent_{agent_idx}"
            if agent_key in sample_data["trajectories"]:
                agent_states = sample_data["trajectories"][agent_key]["states"]
                if len(agent_states) > 0:
                    traj_positions = np.array(agent_states)[:, :2]
                    if agent_idx == 0:  # Ego agent
                        # Plot ground truth trajectory (black dashed)
                        ax.plot(traj_positions[:, 0], traj_positions[:, 1], '--', 
                               color='black', linewidth=2, alpha=0.8, 
                               label='Ego (Ground Truth)' if i == 0 else "")
                        
                        # Plot computed trajectory (blue solid) if available
                        if 'final_game_state' in results:
                            if agent_key in results['final_game_state']['trajectories']:
                                computed_states = results['final_game_state']['trajectories'][agent_key]['states']
                                if len(computed_states) > 0:
                                    computed_traj = np.array(computed_states)[:, :2]
                                    ax.plot(computed_traj[:, 0], computed_traj[:, 1], '-', 
                                           color='blue', linewidth=3, alpha=0.9, 
                                           label='Ego (Computed)' if i == 0 else "")
                    else:  # Other agents
                        ax.plot(traj_positions[:, 0], traj_positions[:, 1], 
                               color='gray', linewidth=1.5, alpha=0.6)
        
        # Plot final positions
        for agent_idx in range(n_agents):
            agent_key = f"agent_{agent_idx}"
            if agent_key in sample_data["trajectories"]:
                agent_states = sample_data["trajectories"][agent_key]["states"]
                if len(agent_states) > 0:
                    pos = agent_states[-1][:2]
                    if agent_idx == 0:  # Ego agent
                        # Plot ground truth final position
                        ax.scatter(pos[0], pos[1], c='black', s=100, marker='o', 
                                  edgecolors='black', linewidth=2, zorder=5, alpha=0.8)
                        
                        # Plot computed final position if available
                        if 'final_game_state' in results:
                            if agent_key in results['final_game_state']['trajectories']:
                                computed_states = results['final_game_state']['trajectories'][agent_key]['states']
                                if len(computed_states) > 0:
                                    computed_pos = np.array(computed_states[-1][:2])
                                    ax.scatter(computed_pos[0], computed_pos[1], c='blue', s=100, marker='s', 
                                              edgecolors='blue', linewidth=2, zorder=6, alpha=0.9)
                    else:  # Other agents
                        ax.scatter(pos[0], pos[1], c='gray', s=60, marker='o', 
                                  edgecolors='black', linewidth=1, zorder=4)
        
        # Plot goals
        if 'target_positions' in sample_data:
            goals = sample_data['target_positions']
            for j, goal in enumerate(goals):
                ax.scatter(goal[0], goal[1], c='green', s=100, marker='*', 
                          edgecolors='black', linewidth=1, zorder=6)
        
        # Set title and labels
        ax.set_title(f'Sample {sample_id}\n{T_total} steps, {n_agents} agents', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Position (m)', fontsize=10)
        ax.set_ylabel('Y Position (m)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'Pedestrian Data Test Results Summary\n{n_samples} samples', 
                fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save figure
    if output_dir is None:
        output_dir = results_dir
    
    output_file = os.path.join(output_dir, "pedestrian_summary.png")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Summary figure saved to: {output_file}")
    
    plt.close(fig)
    return output_file


def create_all_figures(results_dir, output_dir=None):
    """
    Create static figures for all pedestrian data test results.
    
    Args:
        results_dir: Directory containing test result JSON files
        output_dir: Directory to save figures (default: same as results_dir)
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
    
    figure_files = []
    for json_file in json_files:
        try:
            fig_file = create_static_figure(json_file, output_dir)
            figure_files.append(fig_file)
        except Exception as e:
            print(f"Error creating figure for {json_file}: {e}")
    
    # Create summary figure
    try:
        summary_file = create_summary_figure(results_dir, output_dir)
        figure_files.append(summary_file)
    except Exception as e:
        print(f"Error creating summary figure: {e}")
    
    print(f"\nCreated {len(figure_files)} figures:")
    for fig_file in figure_files:
        print(f"  {fig_file}")


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
        baseline_mode = "psn"
        
        # Directory structure: ped_results_{n_agents}_{test_type}_goal_true_{obs_type}_{method_suffix}_{psn_model_name}
        goal_suffix = "goal_inference" if goal_source == "goal_inference" else "goal_true"
        obs_type = "partial"  # Since we're using partial PSN model
        method_suffix = "threshold_0.5"  # Default threshold for PSN selection
        psn_model_name = "psn_gru_partial_true_goals_N_10_T_50_obs_10_lr_0.002_bs_32_sigma1_0.075_sigma2_0.075_epochs_100"
        output_dir = os.path.join(base_dir, psn_dir, f"ped_results_{n_agents}_{test_type}_goal_true_{obs_type}_{method_suffix}_{psn_model_name}")
    
    return output_dir


def main():
    """Main function."""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='Create static figures from pedestrian data test results')
    parser.add_argument('--results_dir', type=str, 
                       default=None,
                       help='Base directory containing test result directories (if None, uses config-based directory)')
    parser.add_argument('--config_name', type=str, default=None,
                       help='Specific configuration name to create figures for (if None, uses first found)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save figures (default: same as results_dir)')
    parser.add_argument('--sample_id', type=int, default=None,
                       help='Create figure for specific sample ID only')
    parser.add_argument('--summary_only', action='store_true',
                       help='Create only the summary figure')
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
        # Create figure for specific sample
        json_file = os.path.join(results_dir, f"psn_ped_test_sample_{args.sample_id:03d}.json")
        if os.path.exists(json_file):
            create_static_figure(json_file, args.output_dir)
        else:
            print(f"Sample {args.sample_id} not found: {json_file}")
    elif args.summary_only:
        # Create only summary figure
        create_summary_figure(results_dir, args.output_dir)
    else:
        # Create figures for all samples
        create_all_figures(results_dir, args.output_dir)


if __name__ == "__main__":
    main()
