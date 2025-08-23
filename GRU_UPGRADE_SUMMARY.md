# GRU Network Upgrade Summary

## Overview
This document summarizes the changes made to upgrade both the Goal Inference Network and Player Selection Network (PSN) from MLP-based architectures to GRU-based architectures for better temporal sequence processing.

## Key Changes Made

### 1. Goal Inference Network (`goal_inference/pretrain_goal_inference.py`)

**Before (MLP):**
- Simple time averaging: `x = jnp.mean(x, axis=1)` 
- Direct MLP processing: `[128, 64, 32]` hidden dimensions
- Basic dropout: `dropout_rate = 0.1`

**After (GRU):**
- **Temporal Processing**: Shared GRU processes each agent's trajectory sequence
- **Reduced MLP Head**: `[64, 32]` hidden dimensions (smaller, more focused)
- **Enhanced Regularization**: `dropout_rate = 0.3` + AdamW with weight decay
- **Robust Loss**: Huber loss instead of MSE for better outlier handling

**Architecture:**
```
Input: (batch, T_obs, N_agents, state_dim)
  ↓
For each agent:
  Agent trajectory → Shared GRU (hidden_size=64) → Final hidden state
  ↓
Concatenate agent features: (batch, N_agents * 64)
  ↓
MLP Head: [64, 32] → Goal prediction (N_agents * 2)
```

### 2. Player Selection Network (`player_selection_network/psn_training_with_pretrained_goals.py`)

**Before (MLP):**
- Time averaging: `x = jnp.mean(x, axis=1)`
- Simple MLP: `[128, 64, 32]` hidden dimensions
- Basic dropout: `dropout_rate = 0.1`

**After (GRU):**
- **Temporal Processing**: Shared GRU processes each agent's trajectory sequence
- **Reduced MLP Head**: `[64, 32]` hidden dimensions
- **Enhanced Regularization**: `dropout_rate = 0.3` + AdamW with weight decay
- **Proper Dropout**: Added `deterministic` parameter handling

**Architecture:**
```
Input: (batch, T_obs, N_agents, state_dim)
  ↓
For each agent:
  Agent trajectory → Shared GRU (hidden_size=64) → Final hidden state
  ↓
Concatenate agent features: (batch, N_agents * 64)
  ↓
MLP Head: [64, 32] → Mask prediction (N_agents - 1)
```

### 3. Configuration Updates (`config.yaml`)

**Goal Inference:**
```yaml
goal_inference:
  hidden_dims_4p: [64, 32]          # Reduced from [128, 64, 32]
  hidden_dims_10p: [128, 64]        # Reduced from [256, 128, 64]
  gru_hidden_size: 64               # New GRU parameter
  dropout_rate: 0.3                 # Increased from 0.1
```

**PSN:**
```yaml
psn:
  hidden_dims_4p: [64, 32]          # Reduced from [128, 64, 32]
  hidden_dims_10p: [128, 64]        # Reduced from [256, 128, 64]
  gru_hidden_size: 64               # New GRU parameter
  use_batch_norm: false             # Disabled (use dropout instead)
  dropout_rate: 0.3                 # Increased from 0.1
```

### 4. Example Files Updated

- `examples/player_selection_network_training.py`
- `examples/psn_training_with_reference.py`

Both updated with GRU architectures and proper dropout handling.

## Benefits of GRU Upgrade

### 1. **Temporal Awareness**
- **Before**: Only saw averaged states, lost temporal dynamics
- **After**: Processes full trajectory sequence, captures motion patterns

### 2. **Better Generalization**
- **Before**: Large MLP (128→64→32) prone to overfitting on small dataset
- **After**: Smaller MLP head (64→32) + shared GRU weights = more parameter efficient

### 3. **Enhanced Regularization**
- **Before**: Basic dropout (0.1) + Adam
- **After**: Increased dropout (0.3) + AdamW with weight decay (5e-4)

### 4. **Robust Loss Functions**
- **Before**: MSE loss sensitive to outliers
- **After**: Huber loss more robust to trajectory anomalies

## Training Recommendations

### 1. **Hyperparameters**
- **Learning Rate**: Start with `3e-4` (reduced from `1e-3`)
- **Weight Decay**: `5e-4` (L2 regularization)
- **Dropout**: `0.3` (increased regularization)
- **Batch Size**: Keep at `32` (good for small dataset)

### 2. **Early Stopping**
- **Patience**: 15-20 epochs on validation loss
- **Monitor**: Validation loss plateau
- **Restore**: Best weights automatically

### 3. **Data Augmentation**
- **Trajectory Jitter**: Small Gaussian noise (σ = 0.01-0.05)
- **Time Shifts**: Random temporal shifts
- **Step Dropping**: Randomly drop 1-2 steps

## Expected Improvements

### 1. **Reduced Overfitting**
- Smaller train-val gap due to better regularization
- More robust to small dataset (384 training samples)

### 2. **Better Temporal Modeling**
- Captures agent motion patterns (acceleration, turns)
- Distinguishes temporal ordering (left→up vs up→left)

### 3. **Improved Generalization**
- Shared GRU weights across agents
- Parameter-efficient architecture
- Better handling of trajectory variations

## Usage Notes

### 1. **Model Loading**
- Existing trained models are incompatible
- Retrain both networks with new architecture
- Update model paths in configuration

### 2. **Inference**
- Always use `deterministic=True` for evaluation
- Dropout automatically disabled during inference

### 3. **Compatibility**
- All training scripts updated for new architecture
- Configuration files updated with new parameters
- Example files demonstrate proper usage

## Next Steps

1. **Retrain Networks**: Both goal inference and PSN need retraining
2. **Validate Performance**: Check train-val gap improvement
3. **Fine-tune Hyperparameters**: Adjust GRU size, dropout rates if needed
4. **Monitor Training**: Use TensorBoard for loss tracking
5. **Test on New Data**: Validate generalization improvement

## Technical Details

### GRU Implementation
- **Shared Weights**: Same GRU cell for all agents
- **Sequence Length**: Processes all 10 observation steps
- **Hidden Size**: 64 units (balanced capacity vs efficiency)
- **Initialization**: Proper carry state initialization

### Dropout Integration
- **Training Mode**: `deterministic=False` enables dropout
- **Validation Mode**: `deterministic=True` disables dropout
- **Layer Placement**: After each hidden layer (except output)

### Weight Decay
- **Optimizer**: AdamW instead of Adam
- **Decay Rate**: 5e-4 (standard L2 regularization)
- **Benefits**: Prevents weight explosion, improves generalization
