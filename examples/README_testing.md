# Testing Trained PSN Models

This directory contains scripts for testing trained Player Selection Network (PSN) models with goal prediction functionality.

## Overview

The PSN model has been enhanced to output both:
1. **Agent Selection Mask**: Binary mask for selecting important agents
2. **Goal Positions**: Predicted goal positions for all agents

## Test Scripts

### 1. `test_trained_psn.py` - Main Test Script

This is the primary test script that loads a trained model and evaluates its performance.

#### Features
- Loads trained PSN models
- Tests on reference trajectory data
- Evaluates goal prediction accuracy
- Tests agent selection performance
- Solves masked games using predicted goals
- Generates comprehensive performance metrics
- Creates visualizations of results

#### Usage

```bash
# Basic usage
python examples/test_trained_psn.py --model_path path/to/model.pkl

# Test with specific number of samples
python examples/test_trained_psn.py --model_path path/to/model.pkl --num_samples 20

# Save results to directory
python examples/test_trained_psn.py --model_path path/to/model.pkl --save_dir test_results

# Use different reference data
python examples/test_trained_psn.py --model_path path/to/model.pkl --reference_file reference_trajectories_4p/all_reference_trajectories.json

# Get help
python examples/test_trained_psn.py --help
```

#### Command Line Arguments

- `--model_path`: Path to the trained model file (required)
- `--reference_file`: Path to reference trajectory file (default: 10-player data)
- `--num_samples`: Number of samples to test (default: 10)
- `--save_dir`: Directory to save results (optional)

### 2. `run_test_example.py` - Example Runner

This script demonstrates different ways to test models and automatically finds the latest trained model.

#### Usage

```bash
python examples/run_test_example.py
```

#### What it does
- Automatically finds the most recent trained model
- Runs basic tests
- Runs comprehensive tests with result saving
- Shows manual testing commands
- Provides examples for different testing scenarios

## Test Metrics

The test script evaluates the following metrics:

### Goal Prediction Performance
- **RMSE**: Root Mean Square Error between predicted and true goals
- **Per-agent errors**: Individual goal prediction accuracy for each agent
- **Overall accuracy**: Aggregate goal prediction performance

### Agent Selection Performance
- **Mask sparsity**: How selective the model is (higher = more selective)
- **Number of selected agents**: How many agents are chosen for the masked game
- **Selection consistency**: Stability of agent selection across samples

### Game Solving Performance
- **Trajectory similarity**: How well the masked game solution matches reference
- **Success rate**: Percentage of games that can be solved successfully
- **Solution quality**: RMSE between predicted and reference trajectories

## Enhanced Visualization Features

The test script now provides comprehensive 3-panel visualizations that show:

### Panel 1: Goal Predictions vs True Goals
- **Initial positions**: Starting positions of all agents (circles)
- **True goals**: Actual target positions from reference data (squares)
- **Predicted goals**: Goals inferred by the PSN network (triangles)
- **Goal paths**: Dashed lines showing the intended paths from start to predicted goals
- **Goal prediction RMSE**: Overall accuracy metric

### Panel 2: Agent Selection Analysis
- **Selection mask**: Bar chart showing the selection probability for each agent
- **Selection threshold**: Red dashed line at 0.5 indicating selection cutoff
- **Sparsity metrics**: How selective the model is in choosing agents
- **Selection count**: Number of agents selected for the masked game

### Panel 3: Trajectory Comparison
- **Ground truth trajectories**: Reference trajectories for all agents
  - Ego agent: Bold solid line
  - Selected agents: Solid lines
  - Non-selected agents: Dashed lines (low opacity)
- **Computed trajectories**: Solutions from the masked game solver
  - Ego agent: Bold solid line with different style
  - Selected agents: Dotted lines
- **Trajectory RMSE**: Similarity between computed and ground truth trajectories
- **Visual comparison**: Easy to see how well the solver matches the reference

### Visualization Benefits
- **Comprehensive analysis**: All aspects of model performance in one view
- **Easy comparison**: Side-by-side comparison of predictions vs ground truth
- **Performance insight**: Visual understanding of where the model succeeds or fails
- **Debugging aid**: Identify specific issues in goal prediction or trajectory generation

## Output Files

When using `--save_dir`, the script generates:

1. **Individual Sample Plots**: `test_results_sample_XXX.png`
   - **Panel 1 (Left)**: Goal predictions vs true goals visualization
     - Initial positions (circles)
     - True goals (squares)
     - Predicted goals (triangles)
     - Lines from initial to predicted goals
   - **Panel 2 (Center)**: Agent selection mask visualization
     - Bar chart showing mask values for each agent
     - Selection threshold line at 0.5
     - Sparsity and selection count information
   - **Panel 3 (Right)**: Trajectory comparison visualization
     - Ground truth trajectories for all agents
     - Computed trajectories from masked game solver
     - Different line styles for selected vs non-selected agents
     - Trajectory RMSE metric display

2. **Comprehensive Results**: `comprehensive_test_results.json`
   - Aggregate statistics
   - Per-sample detailed results
   - Performance metrics summary
   - Computed trajectories data

## Example Test Workflow

### Step 1: Train a Model
```bash
python examples/psn_training_with_reference.py
```

### Step 2: Find the Trained Model
The model will be saved in:
```
log/psn_N10_T50_obs10_ref50_lr0.001_bs32_sigma10.1_sigma25.0_sigma30.5_epochs5/psn_trained_model.pkl
```

### Step 3: Test the Model
```bash
# Quick test
python examples/test_trained_psn.py --model_path log/psn_N10_T50_obs10_ref50_lr0.001_bs32_sigma10.1_sigma25.0_sigma30.5_epochs5/psn_trained_model.pkl

# Comprehensive test with saving
python examples/test_trained_psn.py --model_path log/psn_N10_T50_obs10_ref50_lr0.001_bs32_sigma10.1_sigma25.0_sigma30.5_epochs5/psn_trained_model.pkl --save_dir my_test_results --num_samples 20
```

### Step 4: Analyze Results
- Check the console output for summary statistics
- View generated plots for visual analysis
- Examine the JSON results file for detailed metrics

## Interpreting Results

### Good Performance Indicators
- **Goal Prediction RMSE < 0.5**: Accurate goal inference
- **Mask Sparsity > 0.7**: Selective agent choice
- **Trajectory Success Rate > 0.8**: Reliable game solving
- **Trajectory RMSE < 1.0**: Good solution quality

### Areas for Improvement
- **High Goal RMSE**: May need more training or data
- **Low Mask Sparsity**: Too many agents selected
- **Low Success Rate**: Game solving issues
- **High Trajectory RMSE**: Poor solution quality

## Troubleshooting

### Common Issues

1. **Model not found**
   - Check the model path is correct
   - Ensure the model file exists and is readable

2. **Reference data mismatch**
   - Verify the reference file has the expected number of agents
   - Check data structure matches expected format

3. **Game solving failures**
   - May indicate numerical instability
   - Check if mask values are reasonable
   - Verify predicted goals are within expected ranges

4. **Memory issues**
   - Reduce `--num_samples` for large models
   - Use CPU instead of GPU if needed

### Getting Help

For issues or questions:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure the model was trained successfully
4. Check file paths and permissions

## Advanced Usage

### Custom Evaluation
You can modify the test script to:
- Add custom metrics
- Change evaluation criteria
- Modify visualization styles
- Add comparison between models

### Batch Testing
Test multiple models:
```bash
for model in log/*/psn_trained_model.pkl; do
    echo "Testing $model"
    python examples/test_trained_psn.py --model_path "$model" --save_dir "test_$(basename $(dirname $model))"
done
```

### Integration with Training
Automatically test after training:
```bash
# Train and test in one script
python examples/psn_training_with_reference.py && \
python examples/test_trained_psn.py --model_path $(find log -name "psn_trained_model.pkl" | head -1)
```
