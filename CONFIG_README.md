# Configuration System Documentation

## Overview

The PSN project uses a centralized configuration system based on YAML files to manage all parameters and hyperparameters across scripts. This ensures consistency, reproducibility, and easy parameter tuning.

## Files

- **`config.yaml`**: Main configuration file containing all parameters
- **`config_loader.py`**: Utility module for loading and accessing configuration
- **`example_config_usage.py`**: Example script demonstrating configuration usage

## Configuration Structure

The configuration is organized into logical sections:

### 🎮 Game Parameters (`game`)
- Agent counts, time discretization, state/control dimensions
- Environment parameters like radius

### ⚙️ Optimization Parameters (`optimization`) 
- iLQGames solver settings (iterations, step size)
- Cost function weights (navigation, collision, control)

### 🧠 PSN Training (`psn`)
- Network architecture (hidden dimensions, activation)
- Training parameters (learning rate, batch size, epochs)
- Loss function weights (σ1, σ2, σ3)

### 🎯 Goal Inference (`goal_inference`)
- Network architecture and training parameters
- Observation length and loss weights

### 📊 Training Configuration (`training`)
- General training settings (seed, GPU usage)
- Logging and monitoring intervals
- TensorBoard configuration

### 🧪 Testing & Evaluation (`testing`)
- Test data parameters and evaluation metrics
- Visualization and output settings

## Usage

### Basic Usage

```python
from config_loader import load_config, get_device_config, setup_jax_config

# Load configuration
config = load_config()

# Setup JAX according to config
setup_jax_config()
device = get_device_config()

# Access parameters with dot notation
n_agents = config.game.N_agents
learning_rate = config.psn.learning_rate
batch_size = config.psn.batch_size
```

### Advanced Usage

```python
# Access with defaults
debug_mode = config.get('debug.debug_mode', False)

# Create structured log directories
from config_loader import create_log_dir
log_dir = create_log_dir("psn", config)

# Get standardized paths
from config_loader import get_data_paths
paths = get_data_paths(config)
```

## Migrating Scripts

To migrate existing scripts to use the configuration system:

1. **Add imports**:
   ```python
   from config_loader import load_config, get_device_config, setup_jax_config
   ```

2. **Load configuration**:
   ```python
   config = load_config()
   setup_jax_config()
   device = get_device_config()
   ```

3. **Replace hardcoded parameters**:
   ```python
   # Before
   N_agents = 4
   learning_rate = 0.001
   
   # After  
   N_agents = config.game.N_agents
   learning_rate = config.psn.learning_rate
   ```

## Scripts Updated

### ✅ Updated Scripts
- **`goal_inference_pretraining/generate_reference_trajectories.py`**: Fully migrated
- **`examples/player_selection_network_training.py`**: Partially migrated
- **`examples/test_trained_psn.py`**: Partially migrated

### 🔄 Scripts to Update
- `goal_inference_pretraining/psn_training_with_pretrained_goals.py`
- `goal_inference_pretraining/pretrain_goal_inference.py`
- `examples/psn_training_with_reference.py`
- Other test and evaluation scripts

## Configuration Categories

### Required for All Scripts
```yaml
game:
  dt: 0.1
  T_steps: 50
  N_agents: 4
  state_dim: 4
  control_dim: 2
```

### PSN-Specific
```yaml
psn:
  learning_rate: 0.001
  batch_size: 32
  num_epochs: 50
  sigma1: -0.1
  sigma2: 0.0
```

### Goal Inference-Specific
```yaml
goal_inference:
  learning_rate: 0.001
  observation_length: 10
  num_epochs: 1000
```

## Benefits

1. **Consistency**: All scripts use the same parameters
2. **Reproducibility**: Easy to reproduce results with saved configs
3. **Experimentation**: Quick parameter sweeps by modifying config
4. **Documentation**: Parameters are clearly documented and organized
5. **Validation**: Centralized parameter validation and defaults

## Best Practices

1. **Always load config at script start**
2. **Use descriptive parameter names** 
3. **Document parameter meanings** in config.yaml
4. **Validate parameters** before using them
5. **Use structured log directories** for organized results
6. **Version control config files** for reproducibility

## Example Workflow

```bash
# 1. Modify config.yaml for your experiment
vim config.yaml

# 2. Generate reference trajectories
cd goal_inference_pretraining
python generate_reference_trajectories.py

# 3. Train goal inference network
python pretrain_goal_inference.py

# 4. Train PSN with pretrained goals
python psn_training_with_pretrained_goals.py

# 5. Test the trained model
cd ../examples
python test_trained_psn.py --model_path ../log/psn_*/psn_best_model.pkl
```

All scripts will automatically use the same parameters from `config.yaml`!
