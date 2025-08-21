# Goal Inference Network Pretraining

This directory contains scripts for the two-stage training approach for the Player Selection Network (PSN).

## Overview

The training process is divided into two stages:
1. **Stage 1**: Pretrain a Goal Inference Network to predict agent goals from observation trajectories
2. **Stage 2**: Train the PSN using the pretrained Goal Inference Network

## Stage 1: Goal Inference Network Pretraining

### Purpose
Train a neural network to predict goal positions for all agents based on the first 10 steps of their trajectories.

### Architecture
- **Input**: First 10 steps of all agents' trajectories (T_observation × N_agents × state_dim)
- **Output**: Goal positions for all agents (N_agents × goal_dim)
- **Network**: 3-layer feedforward network with ReLU activations and dropout regularization

### Training Parameters
- **Epochs**: 50
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Goal Loss Weight**: 1.0
- **Data Split**: 75% training (192 samples), 25% validation (64 samples) from 256 total samples

### Data Processing
- **Observation Steps**: 10 (first 10 steps of trajectories)
- **Reference Steps**: 50 (full trajectory length)
- **State Dimension**: 2 (x, y positions)
- **Goal Dimension**: 2 (x, y goal positions)

### TensorBoard Monitoring
The training process logs the following metrics to TensorBoard:
- **Training Loss**: Goal prediction loss over epochs
- **Validation Loss**: Goal prediction loss over epochs

### Output
- **Trained Model**: `goal_inference_best_model.pkl` (best validation performance)
- **Training Config**: `training_config.json`
- **TensorBoard Logs**: Available for monitoring training progress

## Stage 2: PSN Training with Pretrained Goals

### Purpose
Train the Player Selection Network using the pretrained Goal Inference Network to simplify the learning task.

### Architecture
- **Input**: Same observation trajectories as Stage 1
- **Output**: Agent selection mask (N_agents - 1 values)
- **Network**: 2-layer feedforward network with ReLU activations

### Training Parameters
- **Epochs**: 50
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Sigma1**: 0.1 (mask sparsity weight)
- **Sigma2**: 0.0 (similarity loss weight)
- **Sigma3**: 0.0 (goal inference weight - not used as goals are pretrained)

### Loss Functions
- **Binary Loss**: Encourages mask values to be close to 0 or 1
- **Mask Sparsity Loss**: Encourages sparse agent selection
- **Similarity Loss**: Encourages selected agents to have similar goals

### TensorBoard Monitoring
- **Training Loss**: Total loss and component losses over epochs
- **Hyperparameters**: Current sigma values and learning rate

### Output
- **Trained PSN**: `psn_best_model.pkl`
- **Training Config**: `training_config.json`
- **Test Results**: Visualization of agent selection and goal predictions

## Usage

### Stage 1: Goal Inference Pretraining
```bash
cd goal_inference_pretraining
python pretrain_goal_inference.py
```

### Stage 2: PSN Training
```bash
cd goal_inference_pretraining
python psn_training_with_pretrained_goals.py
```

### TensorBoard Monitoring
```bash
# For Stage 1
tensorboard --logdir=log/goal_inference_N_4_T_50_obs_10_lr_0.001_bs_32_goal_loss_weight_1.0_epochs_50

# For Stage 2
tensorboard --logdir=log/psn_N_4_T_50_obs_10_ref_50_lr_0.001_bs_32_sigma1_0.1_sigma2_0.0_sigma3_0.0_epochs_50
```

## Training Workflow

1. **Prepare Data**: Ensure reference trajectories are available in `reference_trajectories_4p/`
2. **Stage 1**: Run goal inference pretraining to get a trained goal prediction model
3. **Stage 2**: Use the pretrained model to train the PSN
4. **Monitor**: Use TensorBoard to track training progress and model performance
5. **Evaluate**: Test the final PSN on new data to assess agent selection quality

## Benefits of Two-Stage Approach

- **Separation of Concerns**: Each network focuses on a specific task
- **Reusability**: Pretrained goal inference network can be used for multiple PSN training runs
- **Better Performance**: Focused training on each specific task
- **Easier Debugging**: Isolated training processes for better error identification
- **Flexibility**: Can experiment with different goal inference approaches independently
