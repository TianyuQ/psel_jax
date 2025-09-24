#!/usr/bin/env python3
"""
Test Analysis Script for Receding Horizon Planning Results

This script reads JSON files from test results and provides comprehensive statistics analysis.
It extracts the same statistics that are computed in the test_psn_receding_horizon.py script.
The script can automatically determine the directory to analyze based on config.yaml settings.

Author: Assistant
Date: 2024
"""

import json
import numpy as np
import os
import glob
import argparse
import yaml
from typing import List, Dict, Any, Tuple
from pathlib import Path
from scipy import stats


def compute_minimum_distance_ego_others(ego_trajectory: np.ndarray, 
                                      other_trajectories: List[np.ndarray]) -> float:
    """
    Compute the minimum distance between ego player and other agents over the entire trajectory.
    
    Args:
        ego_trajectory: Ego agent trajectory (T, 4) - [x, y, vx, vy]
        other_trajectories: List of other agent trajectories (T, 4) each
    
    Returns:
        Minimum distance between ego and any other agent over the trajectory
    """
    if not other_trajectories:
        return float('inf')
    
    T = len(ego_trajectory)
    ego_positions = ego_trajectory[:, :2]  # (T, 2) - only position coordinates
    
    min_distances = []
    for other_traj in other_trajectories:
        if len(other_traj) >= T:
            other_positions = other_traj[:T, :2]  # (T, 2) - only position coordinates
            # Compute distances at each time step
            distances = np.linalg.norm(ego_positions - other_positions, axis=1)
            # Find minimum distance for this agent
            min_distances.append(np.min(distances))
    
    # Return the overall minimum distance across all agents
    return float(np.min(min_distances)) if min_distances else float('inf')


def compute_bootstrap_statistics(values: List[float], 
                                n_bootstrap: int = 1000, 
                                confidence_level: float = 0.95) -> Dict[str, float]:
    """
    Compute bootstrapped statistics for a list of values.
    
    Args:
        values: List of values to bootstrap
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for confidence intervals
    
    Returns:
        Dictionary with bootstrapped statistics
    """
    if not values or len(values) == 0:
        return {
            'mean': float('inf'),
            'std': 0.0,
            'bootstrap_mean': float('inf'),
            'bootstrap_std': 0.0,
            'bootstrap_std_std': 0.0,
            'ci_lower': float('inf'),
            'ci_upper': float('inf'),
            'count': 0
        }
    
    values = np.array(values)
    n_samples = len(values)
    
    # Original statistics
    original_mean = np.mean(values)
    original_std = np.std(values)
    
    # Bootstrap sampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(values, size=n_samples, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Bootstrap statistics
    bootstrap_mean = np.mean(bootstrap_means)
    bootstrap_std = np.std(bootstrap_means)
    bootstrap_std_std = np.std([np.std(np.random.choice(values, size=n_samples, replace=True)) 
                               for _ in range(100)])  # Bootstrap of std
    
    # Confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return {
        'mean': float(original_mean),
        'std': float(original_std),
        'bootstrap_mean': float(bootstrap_mean),
        'bootstrap_std': float(bootstrap_std),
        'bootstrap_std_std': float(bootstrap_std_std),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'count': n_samples
    }


def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_psn_result_directory(config: Dict[str, Any]) -> str:
    """
    Find the PSN result directory using the same logic as the test script.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to the PSN result directory
    """
    import glob
    
    # Extract parameters from config
    n_agents = config['game']['N_agents']
    test_type = config['testing']['receding_horizon']['test_type']
    goal_source = config['testing']['receding_horizon']['goal_source']
    selection_method = config['testing']['receding_horizon']['selection_method']
    obs_input_type = config['psn']['obs_input_type']
    
    # Determine effective number of agents (20 agents use 10-agent model)
    effective_n_agents = 10 if n_agents == 20 else n_agents
    
    # Construct PSN model directory path using the same logic as test script
    if test_type == "planning_test":
        model_name = f"psn_gru_{obs_input_type}_planning_true_goals"
    else:  # prediction_test
        model_name = f"psn_gru_{obs_input_type}_true_goals"
    
    psn_model_path = f"log/goal_true_N_{effective_n_agents}_T_{config['game']['T_total']}_obs_{config['goal_inference']['observation_length']}/{model_name}_N_{effective_n_agents}_T_{config['game']['T_total']}_obs_{config['goal_inference']['observation_length']}_lr_{config['psn']['learning_rate']}_bs_{config['psn']['batch_size']}_sigma1_{config['psn']['sigma1']}_sigma2_{config['psn']['sigma2']}_epochs_{config['psn']['num_epochs']}/psn_best_model.pkl"
    
    # Get the PSN model directory
    psn_model_dir = os.path.dirname(psn_model_path)
    
    # Check if PSN model directory exists
    if not os.path.exists(psn_model_dir):
        raise FileNotFoundError(f"PSN model directory not found: {psn_model_dir}")
    
    # Determine method suffix based on selection method
    if selection_method == "threshold":
        method_suffix = f"threshold_{config['testing']['receding_horizon']['mask_threshold']}"
    else:  # rank
        method_suffix = f"rank_{config['testing']['receding_horizon']['rank']}"
    
    # Determine PSN model name
    psn_model_name = "psn_best_model"
    
    # Construct output directory using the same logic as test script
    if goal_source == "true_goals":
        output_dir = os.path.join(psn_model_dir, f"receding_horizon_results_{n_agents}_{test_type}_goal_true_{obs_input_type}_{method_suffix}_{psn_model_name}")
    elif goal_source == "goal_inference":
        output_dir = os.path.join(psn_model_dir, f"receding_horizon_results_{n_agents}_{test_type}_goal_inference_{obs_input_type}_{method_suffix}_{psn_model_name}")
    else:
        # Fallback
        output_dir = os.path.join(psn_model_dir, f"receding_horizon_results_{n_agents}_{test_type}_{obs_input_type}_{method_suffix}_{psn_model_name}")
    
    # Check if the directory exists
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"PSN result directory not found: {output_dir}")
    
    return output_dir


def determine_analysis_directory(config: Dict[str, Any], 
                                test_type: str = None, 
                                goal_source: str = None,
                                use_baseline: bool = None,
                                baseline_mode: str = None) -> str:
    """
    Determine the analysis directory based on configuration.
    
    Args:
        config: Configuration dictionary
        test_type: Override test type from config
        goal_source: Override goal source from config  
        use_baseline: Override baseline setting from config
        baseline_mode: Override baseline mode from config
        
    Returns:
        Path to analysis directory
    """
    # Get configuration values
    if test_type is None:
        test_type = config['testing']['receding_horizon']['test_type']
    if goal_source is None:
        goal_source = config['testing']['receding_horizon']['goal_source']
    if use_baseline is None:
        use_baseline = config['testing']['receding_horizon']['use_baseline']
    if baseline_mode is None:
        baseline_mode = config['testing']['receding_horizon']['baseline_mode']
    
    n_agents = config['game']['N_agents']
    baseline_param = config['testing']['receding_horizon']['baseline_parameter']
    
    if use_baseline:
        # For baseline methods, create hierarchical directory structure
        method_name = baseline_mode.lower().replace(' ', '_').replace('_', '')
        goal_suffix = "goal_inference" if goal_source == "goal_inference" else "goal_true"
        analysis_dir = f"baseline_results/{test_type}/N_{n_agents}/receding_horizon_results_{n_agents}_{method_name}_param_{baseline_param}_{goal_suffix}"
    else:
        # For PSN methods, search for the actual result directory
        # PSN results are stored under the model training directory in log/
        analysis_dir = find_psn_result_directory(config)
    
    return analysis_dir


def load_json_files(directory: str) -> List[Dict[str, Any]]:
    """
    Load all JSON files from the specified directory.
    
    Args:
        directory: Path to directory containing JSON files
        
    Returns:
        List of loaded JSON data
    """
    # Find all JSON files in the directory
    pattern = os.path.join(directory, "*.json")
    json_files = sorted(glob.glob(pattern))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in directory: {directory}")
    
    print(f"Found {len(json_files)} JSON files in {directory}")
    
    # Load all JSON files
    all_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                all_data.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(all_data)} JSON files")
    return all_data


def compute_sample_statistics(sample_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute statistics for a single sample.
    
    Args:
        sample_data: Single sample JSON data
        
    Returns:
        Dictionary of computed statistics
    """
    stats = {}
    
    # Basic sample information
    stats['sample_id'] = sample_data.get('sample_id', 0)
    stats['use_baseline'] = sample_data.get('use_baseline', False)
    stats['baseline_mode'] = sample_data.get('baseline_mode', None)
    stats['goal_source'] = sample_data.get('goal_source', 'unknown')
    stats['test_type'] = 'prediction_test' if sample_data.get('goal_source') == 'true_goals' else 'planning_test'
    
    # Extract receding horizon results
    receding_horizon_results = sample_data.get('receding_horizon_results', [])
    
    if not receding_horizon_results:
        # Return default values if no results
        return {
            'sample_id': stats['sample_id'],
            'use_baseline': stats['use_baseline'],
            'baseline_mode': stats['baseline_mode'],
            'goal_source': stats['goal_source'],
            'test_type': stats['test_type'],
            'goal_rmse': float('inf'),
            'mask_sparsity': 0.0,
            'num_selected_agents': 0.0,
            'consistency_metric': 0.0,
            'mean_computation_time': 0.0,
            'sample_computation_time': 0.0,
            'prediction_metrics': {'ade': float('inf'), 'fde': float('inf')},
            'planning_metrics': {
                'navigation_cost': float('inf'),
                'safety_cost': float('inf'),
                'control_cost': float('inf'),
                'trajectory_length': float('inf'),
                'trajectory_smoothness': float('inf')
            }
        }
    
    # Compute goal RMSE from receding horizon results
    goal_rmse_values = []
    mask_sparsity_values = []
    num_selected_values = []
    
    for iter_result in receding_horizon_results:
        # Goal RMSE
        pred_goals = np.array(iter_result.get('predicted_goals', []))
        true_goals = np.array(iter_result.get('true_goals', []))
        
        if len(pred_goals) > 0 and len(true_goals) > 0:
            # Handle shape mismatch
            if pred_goals.shape != true_goals.shape:
                min_agents = min(pred_goals.shape[0], true_goals.shape[0])
                pred_goals = pred_goals[:min_agents]
                true_goals = true_goals[:min_agents]
            
            goal_rmse = np.sqrt(np.mean(np.square(pred_goals - true_goals)))
            goal_rmse_values.append(float(goal_rmse))
        
        # Mask statistics
        mask_sparsity_values.append(iter_result.get('mask_sparsity', 0.0))
        num_selected_values.append(iter_result.get('num_selected', 0))
    
    # Compute mean statistics
    stats['goal_rmse'] = float(np.mean(goal_rmse_values)) if goal_rmse_values else float('inf')
    stats['mask_sparsity'] = float(np.mean(mask_sparsity_values)) if mask_sparsity_values else 0.0
    stats['num_selected_agents'] = float(np.mean(num_selected_values)) if num_selected_values else 0.0
    
    # Extract other metrics from the sample data
    stats['consistency_metric'] = sample_data.get('consistency_metric', 0.0)
    stats['mean_computation_time'] = sample_data.get('mean_computation_time', 0.0)
    stats['sample_computation_time'] = sample_data.get('sample_computation_time', 0.0)
    
    # Extract prediction and planning metrics
    stats['prediction_metrics'] = sample_data.get('prediction_metrics', {'ade': float('inf'), 'fde': float('inf')})
    stats['planning_metrics'] = sample_data.get('planning_metrics', {
        'navigation_cost': float('inf'),
        'safety_cost': float('inf'),
        'control_cost': float('inf'),
        'trajectory_length': float('inf'),
        'trajectory_smoothness': float('inf')
    })
    
    # Compute minimum distance for planning tests
    if stats['test_type'] == 'planning_test':
        # Extract ego trajectory and other agent trajectories from final game state
        final_game_state = sample_data.get('final_game_state', {})
        if final_game_state and 'trajectories' in final_game_state:
            # Get ego agent trajectory (agent_0)
            ego_trajectory = np.array(final_game_state['trajectories'].get('agent_0', {}).get('states', []))
            
            # Get other agent trajectories
            other_trajectories = []
            for agent_key, agent_data in final_game_state['trajectories'].items():
                if agent_key != 'agent_0':  # Skip ego agent
                    other_traj = np.array(agent_data.get('states', []))
                    if len(other_traj) > 0:
                        other_trajectories.append(other_traj)
            
            # Compute minimum distance
            if len(ego_trajectory) > 0 and other_trajectories:
                min_distance = compute_minimum_distance_ego_others(ego_trajectory, other_trajectories)
                stats['min_distance_ego_others'] = min_distance
            else:
                stats['min_distance_ego_others'] = float('inf')
        else:
            stats['min_distance_ego_others'] = float('inf')
    else:
        # For prediction tests, set to inf (not applicable)
        stats['min_distance_ego_others'] = float('inf')
    
    return stats


def compute_aggregate_statistics(all_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate statistics across all samples with bootstrap statistics.
    
    Args:
        all_stats: List of sample statistics
        
    Returns:
        Dictionary of aggregate statistics with bootstrap confidence intervals
    """
    if not all_stats:
        return {}
    
    # Extract values for aggregation
    goal_rmse_values = [s['goal_rmse'] for s in all_stats if s['goal_rmse'] != float('inf')]
    mask_sparsity_values = [s['mask_sparsity'] for s in all_stats]
    num_selected_values = [s['num_selected_agents'] for s in all_stats]
    consistency_values = [s['consistency_metric'] for s in all_stats]
    computation_times = [s['mean_computation_time'] for s in all_stats if s['mean_computation_time'] > 0]
    sample_computation_times = [s['sample_computation_time'] for s in all_stats if s['sample_computation_time'] > 0]
    
    # Minimum distance values (only for planning tests)
    min_distance_values = [s.get('min_distance_ego_others', float('inf')) for s in all_stats 
                          if s.get('min_distance_ego_others', float('inf')) != float('inf')]
    
    # Prediction metrics
    ade_values = [s['prediction_metrics'].get('ade', float('inf')) for s in all_stats 
                  if s['prediction_metrics'].get('ade', float('inf')) != float('inf')]
    fde_values = [s['prediction_metrics'].get('fde', float('inf')) for s in all_stats 
                  if s['prediction_metrics'].get('fde', float('inf')) != float('inf')]
    
    # Planning metrics
    nav_cost_values = [s['planning_metrics'].get('navigation_cost', float('inf')) for s in all_stats 
                       if s['planning_metrics'].get('navigation_cost', float('inf')) != float('inf')]
    safety_cost_values = [s['planning_metrics'].get('safety_cost', float('inf')) for s in all_stats 
                          if s['planning_metrics'].get('safety_cost', float('inf')) != float('inf')]
    control_cost_values = [s['planning_metrics'].get('control_cost', float('inf')) for s in all_stats 
                           if s['planning_metrics'].get('control_cost', float('inf')) != float('inf')]
    trajectory_length_values = [s['planning_metrics'].get('trajectory_length', float('inf')) for s in all_stats 
                                if s['planning_metrics'].get('trajectory_length', float('inf')) != float('inf')]
    trajectory_smoothness_values = [s['planning_metrics'].get('trajectory_smoothness', float('inf')) for s in all_stats 
                                    if s['planning_metrics'].get('trajectory_smoothness', float('inf')) != float('inf')]
    
    # Compute bootstrap statistics for all metrics
    aggregate = {
        'num_samples': len(all_stats),
        'goal_rmse': compute_bootstrap_statistics(goal_rmse_values),
        'mask_sparsity': compute_bootstrap_statistics(mask_sparsity_values),
        'num_selected_agents': compute_bootstrap_statistics(num_selected_values),
        'consistency_metric': compute_bootstrap_statistics(consistency_values),
        'mean_computation_time': compute_bootstrap_statistics(computation_times),
        'sample_computation_time': compute_bootstrap_statistics(sample_computation_times),
        'min_distance_ego_others': compute_bootstrap_statistics(min_distance_values),
        'prediction_metrics': {
            'ade': compute_bootstrap_statistics(ade_values),
            'fde': compute_bootstrap_statistics(fde_values)
        },
        'planning_metrics': {
            'navigation_cost': compute_bootstrap_statistics(nav_cost_values),
            'safety_cost': compute_bootstrap_statistics(safety_cost_values),
            'control_cost': compute_bootstrap_statistics(control_cost_values),
            'trajectory_length': compute_bootstrap_statistics(trajectory_length_values),
            'trajectory_smoothness': compute_bootstrap_statistics(trajectory_smoothness_values)
        }
    }
    
    return aggregate


def print_summary_statistics(aggregate_stats: Dict[str, Any], directory: str):
    """
    Print formatted summary statistics with bootstrap confidence intervals.
    
    Args:
        aggregate_stats: Aggregate statistics dictionary
        directory: Source directory for context
    """
    print("=" * 80)
    print("TEST ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Source Directory: {directory}")
    print(f"Number of Samples: {aggregate_stats['num_samples']}")
    print()
    
    def print_metric_with_bootstrap(name: str, stats: Dict[str, float], unit: str = ""):
        """Helper function to print metric with bootstrap statistics."""
        if stats['count'] > 0:
            print(f"{name}: {stats['mean']:.4f} ± {stats['std']:.4f} {unit}")
            print(f"  Bootstrap: {stats['bootstrap_mean']:.4f} ± {stats['bootstrap_std']:.4f} {unit}")
            print(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] {unit}")
        else:
            print(f"{name}: N/A (no valid samples)")
    
    # Basic statistics
    print("Basic Statistics:")
    print_metric_with_bootstrap("Goal Prediction RMSE", aggregate_stats['goal_rmse'])
    print_metric_with_bootstrap("Mask Sparsity", aggregate_stats['mask_sparsity'])
    print_metric_with_bootstrap("Average Selected Agents", aggregate_stats['num_selected_agents'])
    print_metric_with_bootstrap("Consistency Metric", aggregate_stats['consistency_metric'])
    print_metric_with_bootstrap("Mean Computation Time per Receding Horizon Step", 
                               aggregate_stats['mean_computation_time'], "s")
    print_metric_with_bootstrap("Mean Computation Time per Sample", 
                               aggregate_stats['sample_computation_time'], "s")
    
    # Minimum distance for planning tests
    if aggregate_stats['min_distance_ego_others']['count'] > 0:
        print()
        print("Planning Test Specific Metrics:")
        print_metric_with_bootstrap("Minimum Distance (Ego to Others)", 
                                   aggregate_stats['min_distance_ego_others'], "units")
    
    print()
    
    # Prediction metrics
    if aggregate_stats['prediction_metrics']['ade']['count'] > 0:
        print("Prediction Metrics (steps 10-50, receding horizon planning phase only):")
        print_metric_with_bootstrap("  ADE", aggregate_stats['prediction_metrics']['ade'])
        print_metric_with_bootstrap("  FDE", aggregate_stats['prediction_metrics']['fde'])
        print()
    
    # Planning metrics
    if aggregate_stats['planning_metrics']['navigation_cost']['count'] > 0:
        print("Planning Metrics (steps 10-50, receding horizon planning phase only):")
        print_metric_with_bootstrap("  Navigation Cost", aggregate_stats['planning_metrics']['navigation_cost'])
        print_metric_with_bootstrap("  Safety Cost", aggregate_stats['planning_metrics']['safety_cost'])
        print_metric_with_bootstrap("  Control Cost", aggregate_stats['planning_metrics']['control_cost'])
        print_metric_with_bootstrap("  Trajectory Length", aggregate_stats['planning_metrics']['trajectory_length'])
        print_metric_with_bootstrap("  Trajectory Smoothness", aggregate_stats['planning_metrics']['trajectory_smoothness'])
        print()


def save_detailed_results(aggregate_stats: Dict[str, Any], all_stats: List[Dict[str, Any]], 
                         output_file: str, directory: str):
    """
    Save detailed results to a JSON file.
    
    Args:
        aggregate_stats: Aggregate statistics
        all_stats: Individual sample statistics
        output_file: Output file path
        directory: Source directory
    """
    results = {
        'source_directory': directory,
        'analysis_timestamp': str(np.datetime64('now')),
        'aggregate_statistics': aggregate_stats,
        'individual_sample_statistics': all_stats
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Detailed results saved to: {output_file}")


def create_test_summary(aggregate_stats: Dict[str, Any], config: Dict[str, Any], 
                       directory: str, output_file: str = "test_summary.txt"):
    """
    Create a test_summary.txt file similar to the one generated by the test script.
    
    Args:
        aggregate_stats: Aggregate statistics
        config: Configuration dictionary
        directory: Source directory
        output_file: Output filename (default: test_summary.txt)
    """
    summary_path = os.path.join(directory, output_file)
    
    with open(summary_path, 'w') as f:
        f.write("Receding Horizon Testing with Integrated Models\n")
        f.write("=" * 60 + "\n\n")
        f.write("Test Configuration:\n")
        f.write(f"  - Test Type: {config['testing']['receding_horizon']['test_type']}\n")
        f.write(f"  - Goal Source: {config['testing']['receding_horizon']['goal_source']}\n")
        f.write(f"  - Use Baseline: {config['testing']['receding_horizon']['use_baseline']}\n")
        if config['testing']['receding_horizon']['use_baseline']:
            f.write(f"  - Baseline Mode: {config['testing']['receding_horizon']['baseline_mode']}\n")
            f.write(f"  - Baseline Parameter: {config['testing']['receding_horizon']['baseline_parameter']}\n")
        f.write(f"  - Number of agents: {config['game']['N_agents']}\n")
        f.write(f"  - Receding horizon iterations: {config['game']['T_receding_horizon_iterations']}\n")
        f.write(f"  - Planning horizon per game: {config['game']['T_receding_horizon_planning']}\n")
        f.write(f"  - Total trajectory steps: {config['game']['T_total']}\n")
        f.write(f"  - Observation steps: {config['game']['T_observation']}\n")
        f.write(f"  - Compute prediction metrics: {config['testing']['receding_horizon']['compute_prediction_metrics']}\n")
        f.write(f"  - Compute planning metrics: {config['testing']['receding_horizon']['compute_planning_metrics']}\n\n")
        f.write("Results:\n")
        f.write(f"  - Successfully tested: {aggregate_stats['num_samples']} samples\n")
        f.write(f"  - Output directory: {directory}\n\n")
        
        # Add metrics summary if available (steps 10-50 only)
        if aggregate_stats['num_samples'] > 0:
            f.write("Metrics Summary (steps 10-50, receding horizon planning phase only):\n")
            f.write("(All statistics include bootstrap confidence intervals)\n\n")
            
            # Basic statistics
            f.write("Basic Statistics:\n")
            if aggregate_stats['goal_rmse']['count'] > 0:
                f.write(f"  - Goal Prediction RMSE: {aggregate_stats['goal_rmse']['mean']:.4f} ± {aggregate_stats['goal_rmse']['std']:.4f}\n")
                f.write(f"    Bootstrap: {aggregate_stats['goal_rmse']['bootstrap_mean']:.4f} ± {aggregate_stats['goal_rmse']['bootstrap_std']:.4f}\n")
                f.write(f"    95% CI: [{aggregate_stats['goal_rmse']['ci_lower']:.4f}, {aggregate_stats['goal_rmse']['ci_upper']:.4f}]\n")
            
            f.write(f"  - Mask Sparsity: {aggregate_stats['mask_sparsity']['mean']:.3f} ± {aggregate_stats['mask_sparsity']['std']:.3f}\n")
            f.write(f"    Bootstrap: {aggregate_stats['mask_sparsity']['bootstrap_mean']:.3f} ± {aggregate_stats['mask_sparsity']['bootstrap_std']:.3f}\n")
            f.write(f"    95% CI: [{aggregate_stats['mask_sparsity']['ci_lower']:.3f}, {aggregate_stats['mask_sparsity']['ci_upper']:.3f}]\n")
            
            f.write(f"  - Average Selected Agents: {aggregate_stats['num_selected_agents']['mean']:.2f} ± {aggregate_stats['num_selected_agents']['std']:.2f}\n")
            f.write(f"    Bootstrap: {aggregate_stats['num_selected_agents']['bootstrap_mean']:.2f} ± {aggregate_stats['num_selected_agents']['bootstrap_std']:.2f}\n")
            f.write(f"    95% CI: [{aggregate_stats['num_selected_agents']['ci_lower']:.2f}, {aggregate_stats['num_selected_agents']['ci_upper']:.2f}]\n")
            
            f.write(f"  - Consistency Metric: {aggregate_stats['consistency_metric']['mean']:.4f} ± {aggregate_stats['consistency_metric']['std']:.4f}\n")
            f.write(f"    Bootstrap: {aggregate_stats['consistency_metric']['bootstrap_mean']:.4f} ± {aggregate_stats['consistency_metric']['bootstrap_std']:.4f}\n")
            f.write(f"    95% CI: [{aggregate_stats['consistency_metric']['ci_lower']:.4f}, {aggregate_stats['consistency_metric']['ci_upper']:.4f}]\n")
            
            if aggregate_stats['mean_computation_time']['count'] > 0:
                f.write(f"  - Mean Computation Time per Receding Horizon Step: {aggregate_stats['mean_computation_time']['mean']:.4f}s ± {aggregate_stats['mean_computation_time']['std']:.4f}s\n")
                f.write(f"    Bootstrap: {aggregate_stats['mean_computation_time']['bootstrap_mean']:.4f}s ± {aggregate_stats['mean_computation_time']['bootstrap_std']:.4f}s\n")
                f.write(f"    95% CI: [{aggregate_stats['mean_computation_time']['ci_lower']:.4f}, {aggregate_stats['mean_computation_time']['ci_upper']:.4f}]s\n")
            
            if aggregate_stats['sample_computation_time']['count'] > 0:
                f.write(f"  - Mean Computation Time per Sample: {aggregate_stats['sample_computation_time']['mean']:.4f}s ± {aggregate_stats['sample_computation_time']['std']:.4f}s\n")
                f.write(f"    Bootstrap: {aggregate_stats['sample_computation_time']['bootstrap_mean']:.4f}s ± {aggregate_stats['sample_computation_time']['bootstrap_std']:.4f}s\n")
                f.write(f"    95% CI: [{aggregate_stats['sample_computation_time']['ci_lower']:.4f}, {aggregate_stats['sample_computation_time']['ci_upper']:.4f}]s\n")
            
            # Minimum distance for planning tests
            if aggregate_stats['min_distance_ego_others']['count'] > 0:
                f.write("\nPlanning Test Specific Metrics:\n")
                f.write(f"  - Minimum Distance (Ego to Others): {aggregate_stats['min_distance_ego_others']['mean']:.4f} ± {aggregate_stats['min_distance_ego_others']['std']:.4f} units\n")
                f.write(f"    Bootstrap: {aggregate_stats['min_distance_ego_others']['bootstrap_mean']:.4f} ± {aggregate_stats['min_distance_ego_others']['bootstrap_std']:.4f} units\n")
                f.write(f"    95% CI: [{aggregate_stats['min_distance_ego_others']['ci_lower']:.4f}, {aggregate_stats['min_distance_ego_others']['ci_upper']:.4f}] units\n")
            
            # Prediction metrics
            if config['testing']['receding_horizon']['compute_prediction_metrics']:
                f.write("\nPrediction Metrics:\n")
                if aggregate_stats['prediction_metrics']['ade']['count'] > 0:
                    f.write(f"  - ADE (steps {config['game']['T_observation']}-{config['game']['T_total']}): {aggregate_stats['prediction_metrics']['ade']['mean']:.4f} ± {aggregate_stats['prediction_metrics']['ade']['std']:.4f}\n")
                    f.write(f"    Bootstrap: {aggregate_stats['prediction_metrics']['ade']['bootstrap_mean']:.4f} ± {aggregate_stats['prediction_metrics']['ade']['bootstrap_std']:.4f}\n")
                    f.write(f"    95% CI: [{aggregate_stats['prediction_metrics']['ade']['ci_lower']:.4f}, {aggregate_stats['prediction_metrics']['ade']['ci_upper']:.4f}]\n")
                if aggregate_stats['prediction_metrics']['fde']['count'] > 0:
                    f.write(f"  - FDE (steps {config['game']['T_observation']}-{config['game']['T_total']}): {aggregate_stats['prediction_metrics']['fde']['mean']:.4f} ± {aggregate_stats['prediction_metrics']['fde']['std']:.4f}\n")
                    f.write(f"    Bootstrap: {aggregate_stats['prediction_metrics']['fde']['bootstrap_mean']:.4f} ± {aggregate_stats['prediction_metrics']['fde']['bootstrap_std']:.4f}\n")
                    f.write(f"    95% CI: [{aggregate_stats['prediction_metrics']['fde']['ci_lower']:.4f}, {aggregate_stats['prediction_metrics']['fde']['ci_upper']:.4f}]\n")
            
            # Planning metrics
            if config['testing']['receding_horizon']['compute_planning_metrics']:
                f.write("\nPlanning Metrics:\n")
                if aggregate_stats['planning_metrics']['navigation_cost']['count'] > 0:
                    f.write(f"  - Navigation Cost (steps {config['game']['T_observation']}-{config['game']['T_total']}): {aggregate_stats['planning_metrics']['navigation_cost']['mean']:.4f} ± {aggregate_stats['planning_metrics']['navigation_cost']['std']:.4f}\n")
                    f.write(f"    Bootstrap: {aggregate_stats['planning_metrics']['navigation_cost']['bootstrap_mean']:.4f} ± {aggregate_stats['planning_metrics']['navigation_cost']['bootstrap_std']:.4f}\n")
                    f.write(f"    95% CI: [{aggregate_stats['planning_metrics']['navigation_cost']['ci_lower']:.4f}, {aggregate_stats['planning_metrics']['navigation_cost']['ci_upper']:.4f}]\n")
                if aggregate_stats['planning_metrics']['safety_cost']['count'] > 0:
                    f.write(f"  - Safety Cost (steps {config['game']['T_observation']}-{config['game']['T_total']}): {aggregate_stats['planning_metrics']['safety_cost']['mean']:.4f} ± {aggregate_stats['planning_metrics']['safety_cost']['std']:.4f}\n")
                    f.write(f"    Bootstrap: {aggregate_stats['planning_metrics']['safety_cost']['bootstrap_mean']:.4f} ± {aggregate_stats['planning_metrics']['safety_cost']['bootstrap_std']:.4f}\n")
                    f.write(f"    95% CI: [{aggregate_stats['planning_metrics']['safety_cost']['ci_lower']:.4f}, {aggregate_stats['planning_metrics']['safety_cost']['ci_upper']:.4f}]\n")
                if aggregate_stats['planning_metrics']['control_cost']['count'] > 0:
                    f.write(f"  - Control Cost (steps {config['game']['T_observation']}-{config['game']['T_total']}): {aggregate_stats['planning_metrics']['control_cost']['mean']:.4f} ± {aggregate_stats['planning_metrics']['control_cost']['std']:.4f}\n")
                    f.write(f"    Bootstrap: {aggregate_stats['planning_metrics']['control_cost']['bootstrap_mean']:.4f} ± {aggregate_stats['planning_metrics']['control_cost']['bootstrap_std']:.4f}\n")
                    f.write(f"    95% CI: [{aggregate_stats['planning_metrics']['control_cost']['ci_lower']:.4f}, {aggregate_stats['planning_metrics']['control_cost']['ci_upper']:.4f}]\n")
                if aggregate_stats['planning_metrics']['trajectory_length']['count'] > 0:
                    f.write(f"  - Trajectory Length (steps {config['game']['T_observation']}-{config['game']['T_total']}): {aggregate_stats['planning_metrics']['trajectory_length']['mean']:.4f} ± {aggregate_stats['planning_metrics']['trajectory_length']['std']:.4f}\n")
                    f.write(f"    Bootstrap: {aggregate_stats['planning_metrics']['trajectory_length']['bootstrap_mean']:.4f} ± {aggregate_stats['planning_metrics']['trajectory_length']['bootstrap_std']:.4f}\n")
                    f.write(f"    95% CI: [{aggregate_stats['planning_metrics']['trajectory_length']['ci_lower']:.4f}, {aggregate_stats['planning_metrics']['trajectory_length']['ci_upper']:.4f}]\n")
                if aggregate_stats['planning_metrics']['trajectory_smoothness']['count'] > 0:
                    f.write(f"  - Trajectory Smoothness (steps {config['game']['T_observation']}-{config['game']['T_total']}): {aggregate_stats['planning_metrics']['trajectory_smoothness']['mean']:.4f} ± {aggregate_stats['planning_metrics']['trajectory_smoothness']['std']:.4f}\n")
                    f.write(f"    Bootstrap: {aggregate_stats['planning_metrics']['trajectory_smoothness']['bootstrap_mean']:.4f} ± {aggregate_stats['planning_metrics']['trajectory_smoothness']['bootstrap_std']:.4f}\n")
                    f.write(f"    95% CI: [{aggregate_stats['planning_metrics']['trajectory_smoothness']['ci_lower']:.4f}, {aggregate_stats['planning_metrics']['trajectory_smoothness']['ci_upper']:.4f}]\n")
        
        f.write("\nDirectory Structure:\n")
        if config['testing']['receding_horizon']['use_baseline']:
            f.write(f"  Baseline Results → {config['testing']['receding_horizon']['test_type']} → N_{config['game']['N_agents']} → {os.path.basename(directory)}\n")
            f.write(f"  Method: {config['testing']['receding_horizon']['baseline_mode']}\n")
            f.write(f"  Parameter: {config['testing']['receding_horizon']['baseline_parameter']}\n")
            f.write(f"  Goal Source: {config['testing']['receding_horizon']['goal_source']}\n")
        else:
            f.write(f"  PSN Results → {os.path.basename(directory)}\n")
            f.write(f"  Test Type: {config['testing']['receding_horizon']['test_type']}\n")
            f.write(f"  Goal Source: {config['testing']['receding_horizon']['goal_source']}\n")
    
    print(f"Test summary saved to: {summary_path}")


def main():
    """Main function to run the test analysis."""
    parser = argparse.ArgumentParser(description='Analyze receding horizon test results from JSON files')
    parser.add_argument('--directory', '-d', help='Directory containing JSON test result files (overrides config)')
    parser.add_argument('--config', '-c', default='config.yaml', help='Configuration file path')
    parser.add_argument('--output', '-o', help='Output file for detailed results (JSON format)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--no-summary', action='store_true', help='Skip creating test_summary.txt')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print("Loading configuration...")
        config = load_config(args.config)
        
        # Determine analysis directory
        if args.directory:
            analysis_dir = args.directory
            print(f"Using specified directory: {analysis_dir}")
        else:
            analysis_dir = determine_analysis_directory(config)
            print(f"Using config-determined directory: {analysis_dir}")
        
        # Validate directory
        if not os.path.isdir(analysis_dir):
            print(f"Error: Directory '{analysis_dir}' does not exist")
            print("Available baseline result directories:")
            if os.path.exists("baseline_results"):
                for root, dirs, files in os.walk("baseline_results"):
                    for d in dirs:
                        if "receding_horizon_results" in d:
                            print(f"  {os.path.join(root, d)}")
            return 1
        
        # Load JSON files
        print(f"Loading JSON files from {analysis_dir}...")
        all_data = load_json_files(analysis_dir)
        
        if not all_data:
            print("Error: No valid JSON files found")
            return 1
        
        # Compute statistics for each sample
        print("Computing statistics for each sample...")
        all_stats = []
        for i, data in enumerate(all_data):
            if args.verbose:
                print(f"  Processing sample {i+1}/{len(all_data)}...")
            stats = compute_sample_statistics(data)
            all_stats.append(stats)
        
        # Compute aggregate statistics
        print("Computing aggregate statistics...")
        aggregate_stats = compute_aggregate_statistics(all_stats)
        
        # Print summary
        print_summary_statistics(aggregate_stats, analysis_dir)
        
        # Create test_summary.txt file
        if not args.no_summary:
            create_test_summary(aggregate_stats, config, analysis_dir)
        
        # Save detailed results if requested
        if args.output:
            save_detailed_results(aggregate_stats, all_stats, args.output, analysis_dir)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
