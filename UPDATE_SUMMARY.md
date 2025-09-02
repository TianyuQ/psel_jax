# Training and Testing Data Directory Update Summary

## Changes Made

This document summarizes the updates made to ensure training and testing scripts use the correct data directories as specified in `config.yaml`.

## Updated Config Structure

The `config.yaml` file now clearly distinguishes between:

### Training Data (Reference Trajectories)
- **Directory**: `reference_trajectories_Np/`
- **Usage**: All model training phases
- **Config Paths**: 
  - `config.training.data_dir`
  - `config.paths.training_data_dir`

### Testing Data (Model-Specific)  
- **Goal Inference Testing**: `reference_trajectories_Np/` 
  - **Purpose**: Tests goal prediction accuracy on clean reference data
  - **Config Path**: `config.testing.goal_inference_data_dir`
  
- **PSN Testing**: `receding_horizon_trajectories_Np/`
  - **Purpose**: Tests player selection in realistic receding horizon scenarios  
  - **Config Path**: `config.testing.psn_data_dir`

## Updated Scripts

### Training Scripts (Now use `config.training.data_dir`)

1. **`goal_inference/pretrain_goal_inference.py`**
   - **Line 92**: `reference_dir = config.training.data_dir`
   - **Purpose**: Goal inference model training uses reference trajectories

2. **`player_selection_network/psn_training_with_pretrained_goals.py`**
   - **Line 129**: `reference_dir = config.training.data_dir`
   - **Purpose**: PSN training uses reference trajectories

3. **`goal_inference/generate_receding_horizon_trajectories.py`**
   - **Line 617**: `reference_dir = config.training.data_dir` (for loading reference data)
   - **Line 597**: `save_dir = config.testing.data_dir` (for saving generated receding horizon data)
   - **Purpose**: Generates receding horizon trajectories from reference trajectories

### Testing Scripts (Now use model-specific testing directories)

4. **`player_selection_network/test_psn_receding_horizon.py`**
   - **Line 1050**: `reference_file = config.testing.psn_data_dir`
   - **Purpose**: PSN testing uses receding horizon trajectories
   - **Added**: Data structure normalization to handle receding horizon file format

5. **`goal_inference/goal_prediction_test.py`**
   - **Line 434**: `default=config.testing.goal_inference_data_dir`
   - **Purpose**: Goal inference testing uses reference trajectories

## Data Flow

```
Reference Trajectory Generation
    ↓ (saves to reference_trajectories_Np/)
    
Training Phase
    ↓ (reads from reference_trajectories_Np/)
    ├─ Goal Inference Training
    └─ PSN Training
    
Receding Horizon Generation
    ↓ (reads reference_trajectories_Np/, saves to receding_horizon_trajectories_Np/)
    
Testing Phase
    ↓ (reads from appropriate data source per model)
    ├─ Goal Inference Testing → reference_trajectories_Np/
    └─ PSN Testing → receding_horizon_trajectories_Np/
```

## Key Benefits

1. **Clear Separation**: Training and testing now use appropriate data sources
2. **Model-Specific Testing**: Each model tests on the data type that makes sense for its purpose
3. **Consistency**: All scripts use the same config-based approach
4. **Flexibility**: Easy to change directories via config without modifying scripts
5. **Documentation**: Config file clearly documents the distinction
6. **Backward Compatibility**: Existing path variables maintained for compatibility
7. **Data Format Handling**: PSN testing script now handles receding horizon data format correctly

## Configuration Headers Added

The `config.yaml` now includes a prominent header explaining the data source distinction:

```yaml
# CRITICAL DATA SOURCE DISTINCTION:
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ TRAINING: Uses reference_trajectories_Np/ directories                  │
# │   - Goal inference model training                                      │
# │   - PSN training with pretrained goals                                 │
# │   - All model training phases use reference trajectory data            │
# │                                                                         │
# │ TESTING:  Uses receding_horizon_trajectories_Np/ directories           │
# │   - PSN receding horizon testing                                       │
# │   - Model evaluation and performance testing                           │
# │   - All testing phases use receding horizon trajectory data            │
# └─────────────────────────────────────────────────────────────────────────┘
```

## Verification

All updated scripts now correctly use:
- Training scripts → `config.training.data_dir` → `reference_trajectories_Np/`
- Goal inference testing → `config.testing.goal_inference_data_dir` → `reference_trajectories_Np/`
- PSN testing → `config.testing.psn_data_dir` → `receding_horizon_trajectories_Np/`

This ensures that each model tests on the appropriate data type for its specific purpose.
