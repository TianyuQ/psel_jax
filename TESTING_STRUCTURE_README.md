# Testing Structure for Receding Horizon Planning

This document explains the new hierarchical testing structure for receding horizon planning with goal inference and player selection models.

## Overview

The testing framework now supports 4 main test configurations organized into 2 categories:

### 1. Prediction Test (All agents' goals are not known)
- **1a. prediction_test + true_goals**: Use true goals for all agents
- **1b. prediction_test + goal_inference**: Use inferred goals for all agents

### 2. Planning Test (Ego agent's goal is always known)
- **2a. planning_test + true_goals**: Use true goals for all agents  
- **2b. planning_test + goal_inference**: Use inferred goals for all agents

## Test Configuration

### Configuration File (`config.yaml`)

```yaml
testing:
  receding_horizon:
    # Test type configuration
    test_type: "prediction_test"         # ["prediction_test", "planning_test"]
    goal_source: "true_goals"            # ["true_goals", "goal_inference"]
    
    # Other parameters...
    use_baseline: true
    baseline_mode: "Control Barrier Function"
    baseline_parameter: 2
    num_samples: 2
    use_later_samples: true
```

### Metrics Computed

**Prediction Test (1a, 1b):**
- ADE (Average Displacement Error)
- FDE (Final Displacement Error)  
- Navigation Cost
- Safety Cost
- Control Cost

**Planning Test (2a, 2b):**
- Navigation Cost
- Safety Cost
- Control Cost

## Running Tests

### Method 1: Edit Configuration File

1. Edit `config.yaml` to set desired test type and goal source
2. Run the test script:
   ```bash
   conda activate player_selection
   python3 player_selection_network/test_psn_receding_horizon.py
   ```

### Method 2: Use Example Script

```bash
conda activate player_selection
python3 examples/run_test_examples.py prediction_test true_goals
python3 examples/run_test_examples.py planning_test goal_inference
```

## Output Directory Structure

Results are organized hierarchically:

```
baseline_results/
├── prediction_test/
│   └── N_10/
│       ├── receding_horizon_results_controlbarrierfunction_param_2_goal_true/
│       └── receding_horizon_results_controlbarrierfunction_param_2_goal_inference/
└── planning_test/
    └── N_10/
        ├── receding_horizon_results_controlbarrierfunction_param_2_goal_true/
        └── receding_horizon_results_controlbarrierfunction_param_2_goal_inference/
```

For PSN models:
```
log/goal_true_N_10_T_50_obs_10/
└── psn_gru_true_goals_N_10_T_50_obs_10_lr_0.001_bs_32_sigma1_0.1_sigma2_0.1_epochs_100/
    ├── receding_horizon_results_prediction_test_goal_true_psn_best_model/
    └── receding_horizon_results_planning_test_goal_inference_psn_best_model/
```

## Test Descriptions

### 1a. Prediction Test + True Goals
- **Purpose**: Test player selection with perfect goal knowledge
- **Goal Source**: Ground truth goals for all agents
- **Metrics**: ADE, FDE, Navigation Cost, Safety Cost, Control Cost
- **Use Case**: Baseline for player selection performance

### 1b. Prediction Test + Goal Inference  
- **Purpose**: Test integrated goal inference + player selection
- **Goal Source**: Inferred goals from goal inference model
- **Metrics**: ADE, FDE, Navigation Cost, Safety Cost, Control Cost
- **Use Case**: End-to-end system performance

### 2a. Planning Test + True Goals
- **Purpose**: Test player selection when ego goal is known
- **Goal Source**: Ground truth goals for all agents
- **Metrics**: Navigation Cost, Safety Cost, Control Cost
- **Use Case**: Planning performance with perfect goal knowledge

### 2b. Planning Test + Goal Inference
- **Purpose**: Test player selection with inferred goals when ego goal is known
- **Goal Source**: Inferred goals from goal inference model
- **Metrics**: Navigation Cost, Safety Cost, Control Cost
- **Use Case**: Planning performance with goal inference

## Model Requirements

### For Goal Inference Tests (1b, 2b)
- Goal inference model must be trained and available
- Model path: `log/goal_inference_rh_gru_N_{N_agents}_T_{T_total}_obs_{T_observation}_lr_{lr}_bs_{bs}_goal_loss_weight_{weight}_epochs_{epochs}/goal_inference_rh_best_model.pkl`

### For PSN Tests (when use_baseline=False)
- PSN model must be trained and available
- Model path: `log/goal_true_N_{N_agents}_T_{T_total}_obs_{T_observation}/psn_gru_true_goals_N_{N_agents}_T_{T_total}_obs_{T_observation}_lr_{lr}_bs_{bs}_sigma1_{sigma1}_sigma2_{sigma2}_epochs_{epochs}/psn_best_model.pkl`

## Example Commands

```bash
# Run prediction test with true goals
python3 examples/run_test_examples.py prediction_test true_goals

# Run planning test with goal inference
python3 examples/run_test_examples.py planning_test goal_inference

# Run all 4 test configurations
for test_type in prediction_test planning_test; do
  for goal_source in true_goals goal_inference; do
    echo "Running $test_type with $goal_source..."
    python3 examples/run_test_examples.py $test_type $goal_source
  done
done
```

## Results Analysis

Each test generates:
- Individual sample results (JSON files)
- Trajectory visualizations (GIF files)
- Summary statistics
- Test configuration summary

Results can be compared across different test types to evaluate:
- Goal inference accuracy (1a vs 1b, 2a vs 2b)
- Player selection performance (1a vs 2a, 1b vs 2b)
- Overall system performance (1b vs 2b)
