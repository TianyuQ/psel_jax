#!/usr/bin/env python3
"""
Test Analysis Script for Pedestrian Data Results

This script reads JSON files from pedestrian data test results and provides 
comprehensive statistics analysis for the PSN testing with pedestrian trajectories.

Author: Assistant
Date: 2024
"""

import json
import numpy as np
import os
import glob
import argparse
from typing import List, Dict, Any, Tuple
from pathlib import Path
from scipy import stats
from datetime import datetime


def load_test_results(results_dir: str) -> List[Dict[str, Any]]:
    """
    Load all test result JSON files from the specified directory.
    
    Args:
        results_dir: Directory containing test result JSON files
    
    Returns:
        List of test result dictionaries
    """
    pattern = os.path.join(results_dir, "psn_ped_test_sample_*.json")
    json_files = sorted(glob.glob(pattern))
    
    if not json_files:
        raise FileNotFoundError(f"No psn_ped_test_sample_*.json files found in directory: {results_dir}")
    
    results = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            result = json.load(f)
            results.append(result)
    
    print(f"Loaded {len(results)} test results from {results_dir}")
    return results


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
    bootstrap_std_std = np.std([np.std(np.random.choice(values, size=n_samples, replace=True)) for _ in range(100)])
    
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
        'count': len(values)
    }


def analyze_prediction_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Analyze prediction metrics (ADE, FDE) across all samples.
    
    Args:
        results: List of test result dictionaries
    
    Returns:
        Dictionary with prediction metrics statistics
    """
    ade_values = []
    fde_values = []
    
    for result in results:
        if 'prediction_metrics' in result:
            pred_metrics = result['prediction_metrics']
            if 'ade' in pred_metrics and pred_metrics['ade'] != float('inf'):
                ade_values.append(pred_metrics['ade'])
            if 'fde' in pred_metrics and pred_metrics['fde'] != float('inf'):
                fde_values.append(pred_metrics['fde'])
    
    return {
        'ade': compute_bootstrap_statistics(ade_values),
        'fde': compute_bootstrap_statistics(fde_values)
    }


def analyze_planning_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Analyze planning metrics across all samples.
    
    Args:
        results: List of test result dictionaries
    
    Returns:
        Dictionary with planning metrics statistics
    """
    metrics = ['navigation_cost', 'safety_cost', 'control_cost', 'trajectory_length', 'trajectory_smoothness']
    planning_stats = {}
    
    for metric in metrics:
        values = []
        for result in results:
            if 'planning_metrics' in result:
                plan_metrics = result['planning_metrics']
                if metric in plan_metrics and plan_metrics[metric] != float('inf'):
                    values.append(plan_metrics[metric])
        
        planning_stats[metric] = compute_bootstrap_statistics(values)
    
    return planning_stats


def analyze_computation_times(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Analyze computation times across all samples.
    
    Args:
        results: List of test result dictionaries
    
    Returns:
        Dictionary with computation time statistics
    """
    sample_times = []
    mean_iteration_times = []
    
    for result in results:
        if 'sample_computation_time' in result:
            sample_times.append(result['sample_computation_time'])
        
        if 'mean_computation_time' in result:
            mean_iteration_times.append(result['mean_computation_time'])
    
    return {
        'sample_time': compute_bootstrap_statistics(sample_times),
        'iteration_time': compute_bootstrap_statistics(mean_iteration_times)
    }


def analyze_consistency_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Analyze consistency metrics across all samples.
    
    Args:
        results: List of test result dictionaries
    
    Returns:
        Dictionary with consistency metrics statistics
    """
    consistency_values = []
    
    for result in results:
        if 'consistency_metric' in result:
            consistency_values.append(result['consistency_metric'])
        elif 'metrics' in result and 'consistency_metric' in result['metrics']:
            consistency_values.append(result['metrics']['consistency_metric'])
    
    return compute_bootstrap_statistics(consistency_values)


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
    
    T_ego = len(ego_trajectory)
    ego_positions = ego_trajectory[:, :2]  # (T_ego, 2) - only position coordinates
    
    min_distances = []
    for other_traj in other_trajectories:
        T_other = len(other_traj)
        # Use the minimum length to ensure both trajectories have the same length
        T_min = min(T_ego, T_other)
        if T_min > 0:
            ego_pos_trimmed = ego_positions[:T_min, :]  # (T_min, 2)
            other_positions = other_traj[:T_min, :2]    # (T_min, 2) - only position coordinates
            # Compute distances at each time step
            distances = np.linalg.norm(ego_pos_trimmed - other_positions, axis=1)
            # Find minimum distance for this agent
            min_distances.append(np.min(distances))
    
    # Return the overall minimum distance across all agents
    return float(np.min(min_distances)) if min_distances else float('inf')


def analyze_minimum_distance_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Analyze minimum distance metrics across all samples.
    
    Args:
        results: List of test result dictionaries
    
    Returns:
        Dictionary with minimum distance metrics statistics
    """
    min_distance_values = []
    
    for result in results:
        try:
            # Extract ego trajectory from final game state (computed trajectory)
            final_game_state = result.get('final_game_state', {})
            if final_game_state and 'trajectories' in final_game_state:
                # Get ego agent trajectory (agent_0) - this is the computed trajectory
                ego_trajectory = np.array(final_game_state['trajectories'].get('agent_0', {}).get('states', []))
                
                # Get other agent trajectories from original sample data
                other_trajectories = []
                normalized_data = result.get('normalized_sample_data', {})
                original_trajectories = normalized_data.get('trajectories', {})
                
                # Extract trajectories for all other agents from original data
                for agent_key, agent_data in original_trajectories.items():
                    if agent_key != 'agent_0':  # Skip ego agent
                        other_traj = np.array(agent_data.get('states', []))
                        if len(other_traj) > 0:
                            other_trajectories.append(other_traj)
                
                # Compute minimum distance
                if len(ego_trajectory) > 0 and other_trajectories:
                    min_distance = compute_minimum_distance_ego_others(ego_trajectory, other_trajectories)
                    if min_distance != float('inf'):
                        min_distance_values.append(min_distance)
        except Exception as e:
            print(f"Warning: Could not compute minimum distance for sample {result.get('sample_id', 'unknown')}: {e}")
            continue
    
    return compute_bootstrap_statistics(min_distance_values)


def analyze_trajectory_characteristics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze trajectory characteristics across all samples.
    
    Args:
        results: List of test result dictionaries
    
    Returns:
        Dictionary with trajectory characteristics
    """
    trajectory_lengths = []
    n_agents_list = []
    receding_horizon_iterations = []
    
    for result in results:
        if 'T_total' in result:
            trajectory_lengths.append(result['T_total'])
        
        if 'n_agents' in result:
            n_agents_list.append(result['n_agents'])
        elif 'T_receding_horizon_iterations' in result:
            # Estimate from receding horizon iterations
            n_agents_list.append(10)  # Pedestrian data has 10 agents
        
        if 'T_receding_horizon_iterations' in result:
            receding_horizon_iterations.append(result['T_receding_horizon_iterations'])
    
    return {
        'trajectory_lengths': {
            'values': trajectory_lengths,
            'mean': float(np.mean(trajectory_lengths)) if trajectory_lengths else 0.0,
            'std': float(np.std(trajectory_lengths)) if trajectory_lengths else 0.0,
            'min': float(np.min(trajectory_lengths)) if trajectory_lengths else 0.0,
            'max': float(np.max(trajectory_lengths)) if trajectory_lengths else 0.0
        },
        'n_agents': {
            'values': n_agents_list,
            'unique': list(set(n_agents_list)) if n_agents_list else []
        },
        'receding_horizon_iterations': {
            'values': receding_horizon_iterations,
            'mean': float(np.mean(receding_horizon_iterations)) if receding_horizon_iterations else 0.0,
            'std': float(np.std(receding_horizon_iterations)) if receding_horizon_iterations else 0.0
        }
    }


def print_configuration_summary(analysis_results: Dict[str, Any]):
    """Print summary for configuration analysis results."""
    print("\n" + "="*80)
    print("CONFIGURATION ANALYSIS SUMMARY")
    print("="*80)
    
    for config_name, config_data in analysis_results.items():
        print(f"\nConfiguration: {config_name}")
        print(f"Sample count: {config_data['sample_count']}")
        print(f"Results directory: {config_data['results_directory']}")
        
        if 'statistics' in config_data:
            stats = config_data['statistics']
            print("\nMetrics Summary:")
            print("-" * 40)
            
            # Prediction metrics
            if 'prediction_metrics' in stats:
                pred_metrics = stats['prediction_metrics']
                if 'ade' in pred_metrics and pred_metrics['ade']['count'] > 0:
                    ade = pred_metrics['ade']
                    print(f"ADE:  {ade['mean']:.4f} ± {ade['std']:.4f} (Bootstrap: {ade['bootstrap_mean']:.4f} ± {ade['bootstrap_std']:.4f})")
                
                if 'fde' in pred_metrics and pred_metrics['fde']['count'] > 0:
                    fde = pred_metrics['fde']
                    print(f"FDE:  {fde['mean']:.4f} ± {fde['std']:.4f} (Bootstrap: {fde['bootstrap_mean']:.4f} ± {fde['bootstrap_std']:.4f})")
            
            # Planning metrics
            if 'planning_metrics' in stats:
                plan_metrics = stats['planning_metrics']
                if 'navigation_cost' in plan_metrics and plan_metrics['navigation_cost']['count'] > 0:
                    nav = plan_metrics['navigation_cost']
                    print(f"Nav Cost:  {nav['mean']:.4f} ± {nav['std']:.4f} (Bootstrap: {nav['bootstrap_mean']:.4f} ± {nav['bootstrap_std']:.4f})")
                
                if 'safety_cost' in plan_metrics and plan_metrics['safety_cost']['count'] > 0:
                    safety = plan_metrics['safety_cost']
                    print(f"Safety Cost:  {safety['mean']:.4f} ± {safety['std']:.4f} (Bootstrap: {safety['bootstrap_mean']:.4f} ± {safety['bootstrap_std']:.4f})")
                
                if 'control_cost' in plan_metrics and plan_metrics['control_cost']['count'] > 0:
                    control = plan_metrics['control_cost']
                    print(f"Control Cost:  {control['mean']:.4f} ± {control['std']:.4f} (Bootstrap: {control['bootstrap_mean']:.4f} ± {control['bootstrap_std']:.4f})")
            
            # Basic statistics
            if 'mask_sparsity' in stats and stats['mask_sparsity']['count'] > 0:
                mask = stats['mask_sparsity']
                print(f"Mask Sparsity:  {mask['mean']:.3f} ± {mask['std']:.3f} (Bootstrap: {mask['bootstrap_mean']:.3f} ± {mask['bootstrap_std']:.3f})")
            
            if 'num_selected_agents' in stats and stats['num_selected_agents']['count'] > 0:
                agents = stats['num_selected_agents']
                print(f"Selected Agents:  {agents['mean']:.2f} ± {agents['std']:.2f} (Bootstrap: {agents['bootstrap_mean']:.2f} ± {agents['bootstrap_std']:.2f})")
            
            if 'consistency_metric' in stats and stats['consistency_metric']['count'] > 0:
                consistency = stats['consistency_metric']
                print(f"Consistency:  {consistency['mean']:.4f} ± {consistency['std']:.4f} (Bootstrap: {consistency['bootstrap_mean']:.4f} ± {consistency['bootstrap_std']:.4f})")
            
            if 'min_distance_ego_others' in stats and stats['min_distance_ego_others']['count'] > 0:
                min_dist = stats['min_distance_ego_others']
                print(f"Min Distance:  {min_dist['mean']:.4f} ± {min_dist['std']:.4f} units (Bootstrap: {min_dist['bootstrap_mean']:.4f} ± {min_dist['bootstrap_std']:.4f})")
            
            if 'planning_metrics' in stats and 'trajectory_smoothness' in stats['planning_metrics'] and stats['planning_metrics']['trajectory_smoothness']['count'] > 0:
                smoothness = stats['planning_metrics']['trajectory_smoothness']
                print(f"Trajectory Smoothness:  {smoothness['mean']:.6f} ± {smoothness['std']:.6f} (Bootstrap: {smoothness['bootstrap_mean']:.6f} ± {smoothness['bootstrap_std']:.6f})")
            
            if 'computation_time' in stats and stats['computation_time']['count'] > 0:
                comp = stats['computation_time']
                print(f"Computation Time:  {comp['mean']:.4f}s ± {comp['std']:.4f}s (Bootstrap: {comp['bootstrap_mean']:.4f}s ± {comp['bootstrap_std']:.4f}s)")

def print_analysis_summary(results: List[Dict[str, Any]], 
                          prediction_stats: Dict[str, Dict[str, float]],
                          planning_stats: Dict[str, Dict[str, float]],
                          computation_stats: Dict[str, float],
                          trajectory_chars: Dict[str, Any]):
    """
    Print a comprehensive analysis summary.
    
    Args:
        results: List of test result dictionaries
        prediction_stats: Prediction metrics statistics
        planning_stats: Planning metrics statistics
        computation_stats: Computation time statistics
        trajectory_chars: Trajectory characteristics
    """
    print("=" * 80)
    print("PEDESTRIAN DATA TEST ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Basic information
    print(f"\nDataset Information:")
    print(f"  Total samples: {len(results)}")
    print(f"  Trajectory lengths: {trajectory_chars['trajectory_lengths']['min']:.0f} - {trajectory_chars['trajectory_lengths']['max']:.0f} steps")
    print(f"  Mean trajectory length: {trajectory_chars['trajectory_lengths']['mean']:.1f} ± {trajectory_chars['trajectory_lengths']['std']:.1f} steps")
    print(f"  Number of agents: {trajectory_chars['n_agents']['unique']}")
    print(f"  Mean receding horizon iterations: {trajectory_chars['receding_horizon_iterations']['mean']:.1f} ± {trajectory_chars['receding_horizon_iterations']['std']:.1f}")
    
    # Prediction metrics
    print(f"\nPrediction Metrics (ADE/FDE):")
    for metric, stats in prediction_stats.items():
        if stats['count'] > 0:
            print(f"  {metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
            print(f"    Bootstrap: {stats['bootstrap_mean']:.4f} ± {stats['bootstrap_std']:.4f}")
            print(f"    95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
        else:
            print(f"  {metric.upper()}: No valid data")
    
    # Planning metrics
    print(f"\nPlanning Metrics:")
    for metric, stats in planning_stats.items():
        if stats['count'] > 0:
            print(f"  {metric.replace('_', ' ').title()}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
        else:
            print(f"  {metric.replace('_', ' ').title()}: No valid data")
    
    # Computation times
    print(f"\nComputation Times:")
    if 'sample_time' in computation_stats and computation_stats['sample_time']['count'] > 0:
        sample_stats = computation_stats['sample_time']
        print(f"  Sample time: {sample_stats['mean']:.4f} ± {sample_stats['std']:.4f} seconds (n={sample_stats['count']})")
    
    if 'iteration_time' in computation_stats and computation_stats['iteration_time']['count'] > 0:
        iter_stats = computation_stats['iteration_time']
        print(f"  Iteration time: {iter_stats['mean']:.6f} ± {iter_stats['std']:.6f} seconds (n={iter_stats['count']})")
    
    # Sample-by-sample breakdown
    print(f"\nSample-by-Sample Results:")
    for i, result in enumerate(results):
        sample_id = result.get('sample_id', i)
        traj_length = result.get('T_total', 'Unknown')
        n_agents = result.get('n_agents', 'Unknown')
        
        print(f"  Sample {sample_id}: {traj_length} steps, {n_agents} agents")
        
        if 'prediction_metrics' in result:
            pred = result['prediction_metrics']
            ade = pred.get('ade', 'N/A')
            fde = pred.get('fde', 'N/A')
            print(f"    ADE: {ade:.4f}, FDE: {fde:.4f}")
        
        if 'planning_metrics' in result:
            plan = result['planning_metrics']
            nav_cost = plan.get('navigation_cost', 'N/A')
            safety_cost = plan.get('safety_cost', 'N/A')
            print(f"    Navigation Cost: {nav_cost:.4f}, Safety Cost: {safety_cost:.4f}")


def save_analysis_results(analysis_results: Dict[str, Any], output_file: str):
    """
    Save analysis results to a JSON file.
    
    Args:
        analysis_results: Dictionary containing all analysis results
        output_file: Path to output JSON file
    """
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    analysis_results_converted = convert_numpy_types(analysis_results)
    
    with open(output_file, 'w') as f:
        json.dump(analysis_results_converted, f, indent=2)
    
    print(f"\nAnalysis results saved to: {output_file}")


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
        search_dirs = ["baseline_results/ped_test", "log/goal_true_N_10_T_50_obs_10"]
    else:
        search_dirs = [base_dir]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        # Look for baseline results
        if search_dir == "baseline_results/ped_test":
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


def compute_bootstrap_statistics(values: List[float], n_bootstrap: int = 1000) -> Dict[str, float]:
    """Compute bootstrap statistics for a list of values."""
    if not values:
        return {'count': 0, 'mean': 0.0, 'std': 0.0, 'bootstrap_mean': 0.0, 'bootstrap_std': 0.0, 
                'ci_lower': 0.0, 'ci_upper': 0.0}
    
    values = np.array(values)
    n = len(values)
    
    # Basic statistics
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    # Bootstrap sampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    bootstrap_mean = np.mean(bootstrap_means)
    bootstrap_std = np.std(bootstrap_means)
    
    # 95% confidence interval
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)
    
    return {
        'count': n,
        'mean': float(mean_val),
        'std': float(std_val),
        'bootstrap_mean': float(bootstrap_mean),
        'bootstrap_std': float(bootstrap_std),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper)
    }

def compute_aggregate_statistics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate statistics from test samples with bootstrap confidence intervals."""
    if not samples:
        return {'num_samples': 0}
    
    # Extract metrics from all samples
    ade_values = []
    fde_values = []
    nav_cost_values = []
    safety_cost_values = []
    control_cost_values = []
    trajectory_lengths = []
    computation_times = []
    mask_sparsity_values = []
    num_selected_agents_values = []
    consistency_values = []
    min_distance_values = []
    trajectory_smoothness_values = []
    
    for sample in samples:
        # Check for prediction metrics
        if 'prediction_metrics' in sample:
            pred_metrics = sample['prediction_metrics']
            if 'ade' in pred_metrics and pred_metrics['ade'] != float('inf'):
                ade_values.append(pred_metrics['ade'])
            if 'fde' in pred_metrics and pred_metrics['fde'] != float('inf'):
                fde_values.append(pred_metrics['fde'])
        
        # Check for planning metrics
        if 'planning_metrics' in sample:
            plan_metrics = sample['planning_metrics']
            if 'navigation_cost' in plan_metrics and plan_metrics['navigation_cost'] != float('inf'):
                nav_cost_values.append(plan_metrics['navigation_cost'])
            if 'safety_cost' in plan_metrics and plan_metrics['safety_cost'] != float('inf'):
                safety_cost_values.append(plan_metrics['safety_cost'])
            if 'control_cost' in plan_metrics and plan_metrics['control_cost'] != float('inf'):
                control_cost_values.append(plan_metrics['control_cost'])
            if 'trajectory_length' in plan_metrics and plan_metrics['trajectory_length'] != float('inf'):
                trajectory_lengths.append(plan_metrics['trajectory_length'])
            if 'trajectory_smoothness' in plan_metrics and plan_metrics['trajectory_smoothness'] != float('inf'):
                trajectory_smoothness_values.append(plan_metrics['trajectory_smoothness'])
        
        # Compute trajectory smoothness if not in planning_metrics
        if 'planning_metrics' not in sample or 'trajectory_smoothness' not in sample.get('planning_metrics', {}):
            try:
                # Extract ego trajectory from final game state to compute smoothness
                final_game_state = sample.get('final_game_state', {})
                if final_game_state and 'trajectories' in final_game_state:
                    ego_trajectory = np.array(final_game_state['trajectories'].get('agent_0', {}).get('states', []))
                    if len(ego_trajectory) >= 3:
                        # Compute trajectory smoothness based on acceleration changes (jerk)
                        velocities = ego_trajectory[:, 2:4]  # (T, 2) - vx, vy
                        accelerations = np.diff(velocities, axis=0)  # (T-1, 2)
                        if len(accelerations) > 1:
                            jerk = np.diff(accelerations, axis=0)  # (T-2, 2)
                            # Return mean squared jerk as smoothness metric
                            smoothness = float(np.mean(jerk ** 2))
                            trajectory_smoothness_values.append(smoothness)
            except Exception:
                # Skip if computation fails
                pass
        
        # Check for computation time
        if 'mean_computation_time' in sample and sample['mean_computation_time'] > 0:
            computation_times.append(sample['mean_computation_time'])
        
        # Extract metrics from receding horizon results (same logic as test_analysis.py)
        if 'receding_horizon_results' in sample and sample['receding_horizon_results']:
            rh_results = sample['receding_horizon_results']
            
            # Compute average mask sparsity and num selected agents across all iterations
            mask_sparsities = []
            num_selected_list = []
            
            for iteration_result in rh_results:
                # Mask statistics
                mask_sparsities.append(iteration_result.get('mask_sparsity', 0.0))
                num_selected_list.append(iteration_result.get('num_selected', 0))
            
            if mask_sparsities:
                mask_sparsity_values.append(np.mean(mask_sparsities))
            if num_selected_list:
                num_selected_agents_values.append(np.mean(num_selected_list))
        
        # Check for consistency metric in top-level data, or compute from receding horizon results
        if 'consistency_metric' in sample:
            consistency_values.append(sample['consistency_metric'])
        elif 'receding_horizon_results' in sample and sample['receding_horizon_results']:
            # Compute consistency metric from receding horizon results
            rh_results = sample['receding_horizon_results']
            if len(rh_results) >= 2:
                # Extract selected agents for each iteration
                selected_agents_per_iteration = []
                for iteration_result in rh_results:
                    if 'selected_agents' in iteration_result:
                        selected_agents = set(iteration_result['selected_agents'])
                        selected_agents_per_iteration.append(selected_agents)
                
                if len(selected_agents_per_iteration) >= 2:
                    # Compute Jaccard distance between consecutive iterations
                    jaccard_distances = []
                    for i in range(1, len(selected_agents_per_iteration)):
                        prev_agents = selected_agents_per_iteration[i-1]
                        curr_agents = selected_agents_per_iteration[i]
                        
                        # Jaccard similarity = intersection / union
                        intersection = len(prev_agents & curr_agents)
                        union = len(prev_agents | curr_agents)
                        
                        if union == 0:
                            jaccard_similarity = 1.0  # If no agents, consider it perfectly consistent
                        else:
                            jaccard_similarity = intersection / union
                        
                        jaccard_distances.append(jaccard_similarity)
                    
                    # Return average consistency (higher is more consistent)
                    consistency_score = float(np.mean(jaccard_distances))
                    consistency_values.append(consistency_score)
        
        # Compute minimum distance metric
        try:
            # Extract ego trajectory from final game state (computed trajectory)
            final_game_state = sample.get('final_game_state', {})
            if final_game_state and 'trajectories' in final_game_state:
                # Get ego agent trajectory (agent_0) - this is the computed trajectory
                ego_trajectory = np.array(final_game_state['trajectories'].get('agent_0', {}).get('states', []))
                
                # Get other agent trajectories from original sample data
                other_trajectories = []
                normalized_data = sample.get('normalized_sample_data', {})
                original_trajectories = normalized_data.get('trajectories', {})
                
                # Extract trajectories for all other agents from original data
                for agent_key, agent_data in original_trajectories.items():
                    if agent_key != 'agent_0':  # Skip ego agent
                        other_traj = np.array(agent_data.get('states', []))
                        if len(other_traj) > 0:
                            other_trajectories.append(other_traj)
                
                # Compute minimum distance
                if len(ego_trajectory) > 0 and other_trajectories:
                    min_distance = compute_minimum_distance_ego_others(ego_trajectory, other_trajectories)
                    if min_distance != float('inf'):
                        min_distance_values.append(min_distance)
        except Exception:
            # Skip if computation fails
            pass
        
        # Also check nested metrics field for backward compatibility
        if 'metrics' in sample:
            metrics = sample['metrics']
            if 'mask_sparsity' in metrics:
                mask_sparsity_values.append(metrics['mask_sparsity'])
            if 'num_selected_agents' in metrics:
                num_selected_agents_values.append(metrics['num_selected_agents'])
            if 'consistency_metric' in metrics:
                consistency_values.append(metrics['consistency_metric'])
    
    # Compute aggregate statistics with bootstrap
    aggregate = {
        'num_samples': len(samples),
        'prediction_metrics': {
            'ade': compute_bootstrap_statistics(ade_values),
            'fde': compute_bootstrap_statistics(fde_values)
        },
        'planning_metrics': {
            'navigation_cost': compute_bootstrap_statistics(nav_cost_values),
            'safety_cost': compute_bootstrap_statistics(safety_cost_values),
            'control_cost': compute_bootstrap_statistics(control_cost_values),
            'trajectory_length': compute_bootstrap_statistics(trajectory_lengths),
            'trajectory_smoothness': compute_bootstrap_statistics(trajectory_smoothness_values)
        },
        'mask_sparsity': compute_bootstrap_statistics(mask_sparsity_values),
        'num_selected_agents': compute_bootstrap_statistics(num_selected_agents_values),
        'consistency_metric': compute_bootstrap_statistics(consistency_values),
        'min_distance_ego_others': compute_bootstrap_statistics(min_distance_values),
        'computation_time': compute_bootstrap_statistics(computation_times)
    }
    
    return aggregate

def create_test_summary(aggregate_stats: Dict[str, Any], config: Dict[str, Any], 
                       directory: str, output_file: str = "test_summary.txt"):
    """Create a test_summary.txt file similar to the original test script."""
    summary_path = os.path.join(directory, output_file)
    
    with open(summary_path, 'w') as f:
        f.write("Receding Horizon Testing with Integrated Models\n")
        f.write("=" * 60 + "\n\n")
        f.write("Test Configuration:\n")
        f.write(f"  - Test Type: prediction_test\n")
        f.write(f"  - Goal Source: true_goals\n")
        f.write(f"  - Use Baseline: {config.testing.receding_horizon.use_baseline}\n")
        if config.testing.receding_horizon.use_baseline:
            f.write(f"  - Baseline Mode: {config.testing.receding_horizon.baseline_mode}\n")
            f.write(f"  - Baseline Parameter: {config.testing.receding_horizon.baseline_parameter}\n")
        f.write(f"  - Number of agents: {config.game.N_agents}\n")
        f.write(f"  - Receding horizon iterations: 50\n")
        f.write(f"  - Planning horizon per game: {config.game.T_receding_horizon_planning}\n")
        f.write(f"  - Total trajectory steps: 50\n")
        f.write(f"  - Observation steps: {config.game.T_observation}\n")
        f.write(f"  - Compute prediction metrics: {config.testing.receding_horizon.compute_prediction_metrics}\n")
        f.write(f"  - Compute planning metrics: {config.testing.receding_horizon.compute_planning_metrics}\n\n")
        f.write("Results:\n")
        f.write(f"  - Successfully tested: {aggregate_stats['num_samples']} samples\n")
        f.write(f"  - Output directory: {directory}\n\n")
        
        # Add metrics summary if available
        if aggregate_stats['num_samples'] > 0:
            f.write("Metrics Summary (steps 10-50, receding horizon planning phase only):\n")
            f.write("(All statistics include bootstrap confidence intervals)\n\n")
            
            # Basic statistics
            f.write("Basic Statistics:\n")
            f.write(f"  - Mask Sparsity: {aggregate_stats['mask_sparsity']['mean']:.3f} ± {aggregate_stats['mask_sparsity']['std']:.3f}\n")
            f.write(f"    Bootstrap: {aggregate_stats['mask_sparsity']['bootstrap_mean']:.3f} ± {aggregate_stats['mask_sparsity']['bootstrap_std']:.3f}\n")
            f.write(f"    95% CI: [{aggregate_stats['mask_sparsity']['ci_lower']:.3f}, {aggregate_stats['mask_sparsity']['ci_upper']:.3f}]\n")
            
            f.write(f"  - Average Selected Agents: {aggregate_stats['num_selected_agents']['mean']:.2f} ± {aggregate_stats['num_selected_agents']['std']:.2f}\n")
            f.write(f"    Bootstrap: {aggregate_stats['num_selected_agents']['bootstrap_mean']:.2f} ± {aggregate_stats['num_selected_agents']['bootstrap_std']:.2f}\n")
            f.write(f"    95% CI: [{aggregate_stats['num_selected_agents']['ci_lower']:.2f}, {aggregate_stats['num_selected_agents']['ci_upper']:.2f}]\n")
            
            if aggregate_stats['consistency_metric']['count'] > 0:
                f.write(f"  - Consistency Metric: {aggregate_stats['consistency_metric']['mean']:.4f} ± {aggregate_stats['consistency_metric']['std']:.4f}\n")
                f.write(f"    Bootstrap: {aggregate_stats['consistency_metric']['bootstrap_mean']:.4f} ± {aggregate_stats['consistency_metric']['bootstrap_std']:.4f}\n")
                f.write(f"    95% CI: [{aggregate_stats['consistency_metric']['ci_lower']:.4f}, {aggregate_stats['consistency_metric']['ci_upper']:.4f}]\n")
            
            if aggregate_stats['min_distance_ego_others']['count'] > 0:
                f.write(f"  - Minimum Distance (Ego to Others): {aggregate_stats['min_distance_ego_others']['mean']:.4f} ± {aggregate_stats['min_distance_ego_others']['std']:.4f} units\n")
                f.write(f"    Bootstrap: {aggregate_stats['min_distance_ego_others']['bootstrap_mean']:.4f} ± {aggregate_stats['min_distance_ego_others']['bootstrap_std']:.4f} units\n")
                f.write(f"    95% CI: [{aggregate_stats['min_distance_ego_others']['ci_lower']:.4f}, {aggregate_stats['min_distance_ego_others']['ci_upper']:.4f}] units\n")
            
            if aggregate_stats['planning_metrics']['trajectory_smoothness']['count'] > 0:
                f.write(f"  - Trajectory Smoothness (steps 10-50): {aggregate_stats['planning_metrics']['trajectory_smoothness']['mean']:.6f} ± {aggregate_stats['planning_metrics']['trajectory_smoothness']['std']:.6f}\n")
                f.write(f"    Bootstrap: {aggregate_stats['planning_metrics']['trajectory_smoothness']['bootstrap_mean']:.6f} ± {aggregate_stats['planning_metrics']['trajectory_smoothness']['bootstrap_std']:.6f}\n")
                f.write(f"    95% CI: [{aggregate_stats['planning_metrics']['trajectory_smoothness']['ci_lower']:.6f}, {aggregate_stats['planning_metrics']['trajectory_smoothness']['ci_upper']:.6f}]\n")
            
            if aggregate_stats['computation_time']['count'] > 0:
                f.write(f"  - Mean Computation Time per Sample: {aggregate_stats['computation_time']['mean']:.4f}s ± {aggregate_stats['computation_time']['std']:.4f}s\n")
                f.write(f"    Bootstrap: {aggregate_stats['computation_time']['bootstrap_mean']:.4f}s ± {aggregate_stats['computation_time']['bootstrap_std']:.4f}s\n")
                f.write(f"    95% CI: [{aggregate_stats['computation_time']['ci_lower']:.4f}, {aggregate_stats['computation_time']['ci_upper']:.4f}]s\n")
            
            # Prediction metrics
            f.write("\nPrediction Metrics:\n")
            if aggregate_stats['prediction_metrics']['ade']['count'] > 0:
                f.write(f"  - ADE (steps {config.game.T_observation}-50): {aggregate_stats['prediction_metrics']['ade']['mean']:.4f} ± {aggregate_stats['prediction_metrics']['ade']['std']:.4f}\n")
                f.write(f"    Bootstrap: {aggregate_stats['prediction_metrics']['ade']['bootstrap_mean']:.4f} ± {aggregate_stats['prediction_metrics']['ade']['bootstrap_std']:.4f}\n")
                f.write(f"    95% CI: [{aggregate_stats['prediction_metrics']['ade']['ci_lower']:.4f}, {aggregate_stats['prediction_metrics']['ade']['ci_upper']:.4f}]\n")
            if aggregate_stats['prediction_metrics']['fde']['count'] > 0:
                f.write(f"  - FDE (steps {config.game.T_observation}-50): {aggregate_stats['prediction_metrics']['fde']['mean']:.4f} ± {aggregate_stats['prediction_metrics']['fde']['std']:.4f}\n")
                f.write(f"    Bootstrap: {aggregate_stats['prediction_metrics']['fde']['bootstrap_mean']:.4f} ± {aggregate_stats['prediction_metrics']['fde']['bootstrap_std']:.4f}\n")
                f.write(f"    95% CI: [{aggregate_stats['prediction_metrics']['fde']['ci_lower']:.4f}, {aggregate_stats['prediction_metrics']['fde']['ci_upper']:.4f}]\n")
            
            # Planning metrics
            f.write("\nPlanning Metrics:\n")
            if aggregate_stats['planning_metrics']['navigation_cost']['count'] > 0:
                f.write(f"  - Navigation Cost (steps {config.game.T_observation}-50): {aggregate_stats['planning_metrics']['navigation_cost']['mean']:.4f} ± {aggregate_stats['planning_metrics']['navigation_cost']['std']:.4f}\n")
                f.write(f"    Bootstrap: {aggregate_stats['planning_metrics']['navigation_cost']['bootstrap_mean']:.4f} ± {aggregate_stats['planning_metrics']['navigation_cost']['bootstrap_std']:.4f}\n")
                f.write(f"    95% CI: [{aggregate_stats['planning_metrics']['navigation_cost']['ci_lower']:.4f}, {aggregate_stats['planning_metrics']['navigation_cost']['ci_upper']:.4f}]\n")
            if aggregate_stats['planning_metrics']['safety_cost']['count'] > 0:
                f.write(f"  - Safety Cost (steps {config.game.T_observation}-50): {aggregate_stats['planning_metrics']['safety_cost']['mean']:.4f} ± {aggregate_stats['planning_metrics']['safety_cost']['std']:.4f}\n")
                f.write(f"    Bootstrap: {aggregate_stats['planning_metrics']['safety_cost']['bootstrap_mean']:.4f} ± {aggregate_stats['planning_metrics']['safety_cost']['bootstrap_std']:.4f}\n")
                f.write(f"    95% CI: [{aggregate_stats['planning_metrics']['safety_cost']['ci_lower']:.4f}, {aggregate_stats['planning_metrics']['safety_cost']['ci_upper']:.4f}]\n")
            if aggregate_stats['planning_metrics']['control_cost']['count'] > 0:
                f.write(f"  - Control Cost (steps {config.game.T_observation}-50): {aggregate_stats['planning_metrics']['control_cost']['mean']:.4f} ± {aggregate_stats['planning_metrics']['control_cost']['std']:.4f}\n")
                f.write(f"    Bootstrap: {aggregate_stats['planning_metrics']['control_cost']['bootstrap_mean']:.4f} ± {aggregate_stats['planning_metrics']['control_cost']['bootstrap_std']:.4f}\n")
                f.write(f"    95% CI: [{aggregate_stats['planning_metrics']['control_cost']['ci_lower']:.4f}, {aggregate_stats['planning_metrics']['control_cost']['ci_upper']:.4f}]\n")
            if aggregate_stats['planning_metrics']['trajectory_length']['count'] > 0:
                f.write(f"  - Trajectory Length (steps {config.game.T_observation}-50): {aggregate_stats['planning_metrics']['trajectory_length']['mean']:.4f} ± {aggregate_stats['planning_metrics']['trajectory_length']['std']:.4f}\n")
                f.write(f"    Bootstrap: {aggregate_stats['planning_metrics']['trajectory_length']['bootstrap_mean']:.4f} ± {aggregate_stats['planning_metrics']['trajectory_length']['bootstrap_std']:.4f}\n")
                f.write(f"    95% CI: [{aggregate_stats['planning_metrics']['trajectory_length']['ci_lower']:.4f}, {aggregate_stats['planning_metrics']['trajectory_length']['ci_upper']:.4f}]\n")
        
        f.write("\nDirectory Structure:\n")
        if config.testing.receding_horizon.use_baseline:
            f.write(f"  Baseline Results → prediction_test → N_{config.game.N_agents} → {os.path.basename(directory)}\n")
            f.write(f"  Method: {config.testing.receding_horizon.baseline_mode}\n")
            f.write(f"  Parameter: {config.testing.receding_horizon.baseline_parameter}\n")
            f.write(f"  Goal Source: true_goals\n")
        else:
            f.write(f"  PSN Results → {os.path.basename(directory)}\n")
            f.write(f"  Test Type: prediction_test\n")
            f.write(f"  Goal Source: true_goals\n")
    
    print(f"Test summary saved to: {summary_path}")

def compute_statistics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics from test samples."""
    if not samples:
        return {}
    
    # Extract metrics from all samples
    ade_values = []
    fde_values = []
    nav_cost_values = []
    safety_cost_values = []
    control_cost_values = []
    trajectory_lengths = []
    computation_times = []
    
    for sample in samples:
        if 'metrics' in sample:
            metrics = sample['metrics']
            if 'ade' in metrics:
                ade_values.append(metrics['ade'])
            if 'fde' in metrics:
                fde_values.append(metrics['fde'])
            if 'navigation_cost' in metrics:
                nav_cost_values.append(metrics['navigation_cost'])
            if 'safety_cost' in metrics:
                safety_cost_values.append(metrics['safety_cost'])
            if 'control_cost' in metrics:
                control_cost_values.append(metrics['control_cost'])
            if 'trajectory_length' in metrics:
                trajectory_lengths.append(metrics['trajectory_length'])
            if 'computation_time' in metrics:
                computation_times.append(metrics['computation_time'])
    
    # Compute statistics
    stats = {}
    
    if ade_values:
        stats['ade'] = {
            'mean': np.mean(ade_values),
            'std': np.std(ade_values),
            'min': np.min(ade_values),
            'max': np.max(ade_values)
        }
    
    if fde_values:
        stats['fde'] = {
            'mean': np.mean(fde_values),
            'std': np.std(fde_values),
            'min': np.min(fde_values),
            'max': np.max(fde_values)
        }
    
    if nav_cost_values:
        stats['navigation_cost'] = {
            'mean': np.mean(nav_cost_values),
            'std': np.std(nav_cost_values),
            'min': np.min(nav_cost_values),
            'max': np.max(nav_cost_values)
        }
    
    if safety_cost_values:
        stats['safety_cost'] = {
            'mean': np.mean(safety_cost_values),
            'std': np.std(safety_cost_values),
            'min': np.min(safety_cost_values),
            'max': np.max(safety_cost_values)
        }
    
    if control_cost_values:
        stats['control_cost'] = {
            'mean': np.mean(control_cost_values),
            'std': np.std(control_cost_values),
            'min': np.min(control_cost_values),
            'max': np.max(control_cost_values)
        }
    
    if trajectory_lengths:
        stats['trajectory_length'] = {
            'mean': np.mean(trajectory_lengths),
            'std': np.std(trajectory_lengths),
            'min': np.min(trajectory_lengths),
            'max': np.max(trajectory_lengths)
        }
    
    if computation_times:
        stats['computation_time'] = {
            'mean': np.mean(computation_times),
            'std': np.std(computation_times),
            'min': np.min(computation_times),
            'max': np.max(computation_times)
        }
    
    return stats

def analyze_single_configuration(results_dir: str, config_name: str) -> Dict[str, Any]:
    """Analyze a single configuration directory."""
    print(f"Analyzing single configuration: {config_name}")
    print(f"Results directory: {results_dir}")
    
    # Find all test result files
    json_files = glob.glob(os.path.join(results_dir, "psn_ped_test_sample_*.json"))
    
    if not json_files:
        print(f"No test result files found in {results_dir}")
        return {}
    
    print(f"Found {len(json_files)} test result files")
    
    # Load and analyze all samples
    all_samples = []
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                sample_data = json.load(f)
            all_samples.append(sample_data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    if not all_samples:
        print("No valid samples loaded")
        return {}
    
    # Compute aggregate statistics with bootstrap
    aggregate_stats = compute_aggregate_statistics(all_samples)
    
    # Load config for test summary
    from config_loader import load_config
    config = load_config()
    
    # Create test summary file
    create_test_summary(aggregate_stats, config, results_dir)
    
    # Create analysis result
    analysis_result = {
        'config_name': config_name,
        'results_directory': results_dir,
        'sample_count': len(all_samples),
        'statistics': aggregate_stats,
        'samples': all_samples
    }
    
    return {config_name: analysis_result}

def analyze_multiple_configurations(base_dir: str) -> Dict[str, Any]:
    """
    Analyze multiple test configurations and compare them.
    
    Args:
        base_dir: Base directory containing test results
    
    Returns:
        Dictionary containing analysis results for all configurations
    """
    print("=" * 80)
    print("PEDESTRIAN DATA MULTI-CONFIGURATION ANALYSIS")
    print("=" * 80)
    
    # Find all test configurations
    test_configs = find_pedestrian_test_directories(base_dir)
    
    if not test_configs:
        print(f"No pedestrian test configurations found in {base_dir}")
        return {}
    
    print(f"Found {len(test_configs)} test configurations:")
    for i, config in enumerate(test_configs):
        print(f"  {i+1}. {config['name']} ({config['sample_count']} samples)")
        print(f"     Directory: {config['directory']}")
        print(f"     PSN Model: {config['psn_model']}")
    
    # Analyze each configuration
    all_analysis_results = {}
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {config['name'].upper()}")
        print(f"{'='*60}")
        
        try:
            # Load test results for this configuration
            results = load_test_results(config['directory'])
            
            # Perform analysis
            prediction_stats = analyze_prediction_metrics(results)
            planning_stats = analyze_planning_metrics(results)
            computation_stats = analyze_computation_times(results)
            consistency_stats = analyze_consistency_metrics(results)
            min_distance_stats = analyze_minimum_distance_metrics(results)
            trajectory_chars = analyze_trajectory_characteristics(results)
            
            # Store results
            all_analysis_results[config['name']] = {
                'prediction_metrics': prediction_stats,
                'planning_metrics': planning_stats,
                'computation_times': computation_stats,
                'consistency_metrics': consistency_stats,
                'min_distance_metrics': min_distance_stats,
                'trajectory_characteristics': trajectory_chars,
                'sample_count': len(results),
                'directory': config['directory'],
                'psn_model': config['psn_model']
            }
            
            # Print summary for this configuration
            print_analysis_summary(results, prediction_stats, planning_stats, 
                                  computation_stats, trajectory_chars)
            
        except Exception as e:
            print(f"Error analyzing {config['name']}: {e}")
            continue
    
    return all_analysis_results


def compare_configurations(analysis_results: Dict[str, Any]) -> None:
    """
    Compare different test configurations and print comparison table.
    
    Args:
        analysis_results: Dictionary containing analysis results for all configurations
    """
    if len(analysis_results) < 2:
        print("\nNot enough configurations to compare.")
        return
    
    print(f"\n{'='*80}")
    print("CONFIGURATION COMPARISON")
    print(f"{'='*80}")
    
    # Create comparison table
    config_names = list(analysis_results.keys())
    
    print(f"\n{'Configuration':<25} {'Samples':<8} {'ADE':<10} {'FDE':<10} {'Nav Cost':<12} {'Safety Cost':<12}")
    print("-" * 80)
    
    for config_name in config_names:
        config_data = analysis_results[config_name]
        pred_metrics = config_data['prediction_metrics']
        plan_metrics = config_data['planning_metrics']
        
        ade_mean = pred_metrics['ade']['mean']
        fde_mean = pred_metrics['fde']['mean']
        nav_cost_mean = plan_metrics['navigation_cost']['mean']
        safety_cost_mean = plan_metrics['safety_cost']['mean']
        
        print(f"{config_name:<25} {config_data['sample_count']:<8} "
              f"{ade_mean:<10.4f} {fde_mean:<10.4f} {nav_cost_mean:<12.4f} {safety_cost_mean:<12.4f}")


def create_output_directory_from_config():
    """Create output directory based on config.yaml settings."""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_loader import load_config
    config = load_config()
    
    if config.testing.receding_horizon.use_baseline:
        # Baseline results directory
        base_dir = "baseline_results/ped_test"
        goal_source = "true_goals"
        n_agents = config.game.N_agents
        baseline_mode = config.testing.receding_horizon.baseline_mode.replace(" ", "_").lower()
        baseline_param = config.testing.receding_horizon.baseline_parameter
        
        # Directory structure: ped_results_{n_agents}_{method}_param_{param}_{goal_suffix}
        goal_suffix = "goal_inference" if goal_source == "goal_inference" else "goal_true"
        output_dir = os.path.join(base_dir, f"ped_results_{n_agents}_{baseline_mode}_param_{baseline_param}_{goal_suffix}")
    else:
        # PSN results directory - construct path dynamically like test_psn_receding_horizon.py
        test_type = config.testing.receding_horizon.test_type
        goal_source = config.testing.receding_horizon.goal_source
        n_agents = config.game.N_agents
        
        # Determine effective number of agents for model selection
        effective_n_agents = 10 if config.game.N_agents > 10 else config.game.N_agents
        
        # Construct PSN model path like in test_psn_receding_horizon.py
        obs_input_type = config.psn.obs_input_type
        if test_type == "planning_test":
            model_name = f"psn_gru_{obs_input_type}_planning_true_goals"
        else:  # prediction_test
            model_name = f"psn_gru_{obs_input_type}_true_goals"
        
        psn_model_path = f"log/goal_true_N_{effective_n_agents}_T_{config.game.T_total}_obs_{config.game.T_observation}/{model_name}_N_{effective_n_agents}_T_{config.game.T_total}_obs_{config.game.T_observation}_lr_{config.psn.learning_rate}_bs_{config.psn.batch_size}_sigma1_{config.psn.sigma1}_sigma2_{config.psn.sigma2}_epochs_{config.psn.num_epochs}/psn_best_model.pkl"
        
        # Get the PSN model directory
        psn_model_dir = os.path.dirname(psn_model_path)
        
        # Extract method name from PSN model path
        psn_model_name = os.path.basename(psn_model_path).replace('.pkl', '')
        
        # Include selection method in directory name for PSN methods
        selection_method = config.testing.receding_horizon.selection_method
        if selection_method == "threshold":
            method_suffix = f"threshold_{config.testing.receding_horizon.mask_threshold}"
        else:  # rank
            method_suffix = f"rank_{config.testing.receding_horizon.rank}"
        
        # Construct output directory like in test_psn_receding_horizon.py but with ped_results prefix
        if goal_source == "true_goals":
            output_dir = os.path.join(psn_model_dir, f"ped_results_{n_agents}_{test_type}_goal_true_{obs_input_type}_{method_suffix}_{psn_model_name}")
        else:  # goal_inference
            output_dir = os.path.join(psn_model_dir, f"ped_results_{n_agents}_{test_type}_goal_inference_{obs_input_type}_{method_suffix}_{psn_model_name}")
    
    return output_dir

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description='Analyze pedestrian data test results')
    parser.add_argument('--results_dir', type=str, 
                       default=None,
                       help='Base directory containing test result directories (if None, uses config-based directory)')
    parser.add_argument('--config_name', type=str, default=None,
                       help='Specific configuration name to analyze (if None, analyzes all)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for analysis results (auto-generated if None)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple configurations')
    
    args = parser.parse_args()
    
    # Use config-based directory by default
    if args.results_dir is None:
        config_dir = create_output_directory_from_config()
        print(f"Using config-based directory: {config_dir}")
        
        # Check if this is a specific results directory with test files
        if os.path.exists(config_dir) and any(f.startswith('psn_ped_test_sample_') for f in os.listdir(config_dir)):
            # This is a specific results directory, analyze it directly
            analysis_results = analyze_single_configuration(config_dir, "config_based")
            
            # Save results
            if analysis_results:
                combined_results = {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'base_directory': config_dir,
                    'configurations': analysis_results
                }
                
                # Determine output file
                if args.output_file is None:
                    args.output_file = os.path.join(config_dir, "pedestrian_analysis.json")
                
                save_analysis_results(combined_results, args.output_file)
                print(f"Analysis results saved to: {args.output_file}")
                
                # Print summary
                print_configuration_summary(analysis_results)
            else:
                print("No valid test configurations found.")
        else:
            # This is a base directory, search for configurations
            analysis_results = analyze_multiple_configurations(config_dir)
            
            # Save results
            if analysis_results:
                combined_results = {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'base_directory': config_dir,
                    'configurations': analysis_results
                }
                
                # Determine output file
                if args.output_file is None:
                    args.output_file = os.path.join(config_dir, "pedestrian_analysis.json")
                
                save_analysis_results(combined_results, args.output_file)
                print(f"Analysis results saved to: {args.output_file}")
                
                # Print summary
                print_configuration_summary(analysis_results)
            else:
                print("No valid test configurations found.")
    else:
        if args.compare or args.config_name is None:
            # Analyze multiple configurations
            analysis_results = analyze_multiple_configurations(args.results_dir)
            
            if analysis_results:
                # Save combined results
                combined_results = {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'base_directory': args.results_dir,
                    'configurations': analysis_results
                }
                
                save_analysis_results(combined_results, args.output_file)
                
                # Compare configurations
                compare_configurations(analysis_results)
                
                print(f"\nCombined analysis results saved to: {args.output_file}")
            else:
                print("No valid test configurations found.")
        else:
            # Analyze specific configuration
            test_configs = find_pedestrian_test_directories(args.results_dir)
            target_config = None
            
            for config in test_configs:
                if config['name'] == args.config_name:
                    target_config = config
                    break
            
            if target_config is None:
                print(f"Configuration '{args.config_name}' not found.")
                print(f"Available configurations: {[c['name'] for c in test_configs]}")
                return
            
            try:
                results = load_test_results(target_config['directory'])
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return
            
            # Perform analysis
            print(f"Analyzing configuration: {args.config_name}")
            
            prediction_stats = analyze_prediction_metrics(results)
            planning_stats = analyze_planning_metrics(results)
            computation_stats = analyze_computation_times(results)
            consistency_stats = analyze_consistency_metrics(results)
            trajectory_chars = analyze_trajectory_characteristics(results)
            
            # Print summary
            print_analysis_summary(results, prediction_stats, planning_stats, 
                                  computation_stats, trajectory_chars)
            
            # Save results
            analysis_results = {
                'configuration_name': args.config_name,
                'directory': target_config['directory'],
                'psn_model': target_config['psn_model'],
                'prediction_metrics': prediction_stats,
                'planning_metrics': planning_stats,
                'computation_times': computation_stats,
                'consistency_metrics': consistency_stats,
                'trajectory_characteristics': trajectory_chars,
                'sample_count': len(results)
            }
            
            save_analysis_results(analysis_results, args.output_file)
            print(f"\nAnalysis results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
