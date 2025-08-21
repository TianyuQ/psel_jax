#!/usr/bin/env python3
"""
Configuration Loader Utility

This module provides utilities for loading and accessing configuration
parameters from the config.yaml file across all scripts in the project.

Usage:
    from config_loader import load_config
    
    config = load_config()
    dt = config.game.dt
    num_agents = config.game.N_agents
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass
from types import SimpleNamespace


class ConfigLoader:
    """Configuration loader with dot notation access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize with configuration dictionary."""
        self._config = self._dict_to_namespace(config_dict)
    
    def _dict_to_namespace(self, d: Dict[str, Any]) -> SimpleNamespace:
        """Convert dictionary to namespace for dot notation access."""
        if isinstance(d, dict):
            return SimpleNamespace(**{k: self._dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [self._dict_to_namespace(item) for item in d]
        else:
            return d
    
    def __getattr__(self, name: str) -> Any:
        """Allow dot notation access to configuration parameters."""
        return getattr(self._config, name)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        try:
            keys = key.split('.')
            value = self._config
            for k in keys:
                value = getattr(value, k)
            return value
        except AttributeError:
            return default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        return self._namespace_to_dict(self._config)
    
    def _namespace_to_dict(self, namespace: SimpleNamespace) -> Dict[str, Any]:
        """Convert namespace back to dictionary."""
        if isinstance(namespace, SimpleNamespace):
            return {k: self._namespace_to_dict(v) for k, v in namespace.__dict__.items()}
        elif isinstance(namespace, list):
            return [self._namespace_to_dict(item) for item in namespace]
        else:
            return namespace


def load_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Optional path to config file. If None, looks for config.yaml
                    in the project root directory.
    
    Returns:
        ConfigLoader: Configuration object with dot notation access.
    
    Raises:
        FileNotFoundError: If config file is not found.
        yaml.YAMLError: If config file is not valid YAML.
    """
    if config_path is None:
        # Look for config.yaml in the project root
        project_root = Path(__file__).parent
        config_path = project_root / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Process variable substitutions
        config_dict = _process_variable_substitutions(config_dict)
        
        return ConfigLoader(config_dict)
    
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file {config_path}: {e}")


def _process_variable_substitutions(config_dict: dict) -> dict:
    """
    Process variable substitutions in configuration values.
    Currently supports: ${N_agents} -> actual number of agents
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Processed configuration dictionary
    """
    def substitute_variables(value, n_agents):
        if isinstance(value, str):
            return value.replace("${N_agents}", str(n_agents))
        elif isinstance(value, dict):
            return {k: substitute_variables(v, n_agents) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute_variables(item, n_agents) for item in value]
        else:
            return value
    
    # Get N_agents value for substitution
    n_agents = config_dict.get('game', {}).get('N_agents', 4)
    
    # Process substitutions
    return substitute_variables(config_dict, n_agents)


def get_device_config():
    """Get JAX device configuration from config."""
    import jax
    
    config = load_config()
    
    # Determine device based on configuration
    preferred = config.get('device.preferred_device', 'auto')
    
    if preferred == 'gpu' and jax.devices('gpu'):
        device = jax.devices('gpu')[0]
    elif preferred == 'cpu':
        device = jax.devices('cpu')[0]
    else:  # auto
        device = jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0]
    
    return device


def setup_jax_config():
    """Setup JAX configuration based on config file."""
    import jax
    import os
    
    config = load_config()
    
    # Set JAX configuration
    if config.get('device.jax_enable_x64', False):
        jax.config.update('jax_enable_x64', True)
    
    if config.get('debug.jax_disable_jit', False):
        jax.config.update('jax_disable_jit', True)
    
    if config.get('debug.jax_debug_nans', False):
        jax.config.update('jax_debug_nans', True)
    
    # Set platform if specified
    platform = config.get('device.jax_platform_name')
    if platform:
        os.environ['JAX_PLATFORM_NAME'] = platform
    
    # Set GPU memory fraction
    memory_fraction = config.get('device.gpu_memory_fraction', 0.9)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(memory_fraction)


def create_log_dir(base_name: str, config: Optional[ConfigLoader] = None) -> Path:
    """
    Create a log directory with a structured name based on configuration.
    
    Args:
        base_name: Base name for the log directory
        config: Optional configuration object
    
    Returns:
        Path: Created log directory path
    """
    if config is None:
        config = load_config()
    
    # Create timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create structured directory name based on key parameters
    if base_name == "goal_inference":
        dir_name = (f"goal_inference_N_{config.game.N_agents}_"
                   f"T_{config.game.T_total}_"
                   f"obs_{config.game.T_observation}_"
                   f"lr_{config.goal_inference.learning_rate}_"
                   f"bs_{config.goal_inference.batch_size}_"
                   f"goal_loss_weight_{config.goal_inference.goal_loss_weight}_"
                   f"epochs_{config.goal_inference.num_epochs}")
    
    elif base_name == "psn":
        dir_name = (f"psn_N_{config.game.N_agents}_"
                   f"T_{config.game.T_total}_"
                   f"obs_{config.goal_inference.observation_length}_"
                   f"lr_{config.psn.learning_rate}_"
                   f"bs_{config.psn.batch_size}_"
                   f"sigma1_{config.psn.sigma1}_"
                   f"sigma2_{config.psn.sigma2}_"
                   f"epochs_{config.psn.num_epochs}")
    
    elif base_name == "reference_generation":
        dir_name = f"reference_trajectories_{config.game.N_agents}p"
    
    else:
        dir_name = f"{base_name}_{timestamp}"
    
    # Create full path
    log_base = Path(config.get('paths.logs_dir', 'log'))
    log_dir = log_base / dir_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    return log_dir


def get_data_paths(config: Optional[ConfigLoader] = None) -> Dict[str, Path]:
    """
    Get standardized data paths from configuration.
    
    Returns:
        Dict containing commonly used paths
    """
    if config is None:
        config = load_config()
    
    base_path = Path.cwd()
    
    paths = {
        'reference_data': base_path / config.get('paths.reference_data_dir', 'reference_trajectories_10p'),
        'models': base_path / config.get('paths.models_dir', 'models'),
        'checkpoints': base_path / config.get('paths.checkpoints_dir', 'checkpoints'),
        'results': base_path / config.get('paths.results_dir', 'results'),
        'logs': base_path / config.get('paths.logs_dir', 'log'),
        'plots': base_path / config.get('paths.plots_dir', 'plots'),
    }
    
    # Create directories if they don't exist
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths


if __name__ == "__main__":
    # Test the configuration loader
    try:
        config = load_config()
        print("Configuration loaded successfully!")
        print(f"Game parameters: N_agents={config.game.N_agents}, dt={config.game.dt}")
        print(f"PSN learning rate: {config.psn.learning_rate}")
        print(f"Goal inference epochs: {config.goal_inference.num_epochs}")
        
        # Test device configuration
        device = get_device_config()
        print(f"Selected device: {device}")
        
        # Test path creation
        paths = get_data_paths(config)
        print(f"Data paths: {paths}")
        
    except Exception as e:
        print(f"Error testing configuration: {e}")
