#!/bin/bash
set -e  # Exit on error

# ==============================================================================
# Standalone Goal Prediction Test Script
# ==============================================================================
# This script tests the pretrained goal inference model on reference trajectory 
# data using receding horizon planning without the complexity of full PSN testing.
#
# Configuration:
# - Uses config.yaml for default parameters (reference_file, num_samples)
# - Only the goal_model path needs to be specified explicitly
# - Automatically detects and validates the goal model file
# ==============================================================================

echo "🚀 Starting Goal Prediction Test..."

# Configuration - Update this path to point to your trained model
GOAL_MODEL="log/goal_inference_N_4_T_50_obs_10_lr_0.001_bs_32_goal_loss_weight_1.0_epochs_1000/goal_inference_best_model.pkl"

# Validate goal model file
echo "📍 Validating goal model: $GOAL_MODEL"
if [ ! -f "$GOAL_MODEL" ]; then
    echo "❌ Error: Goal model file not found: $GOAL_MODEL"
    echo "   Please update the GOAL_MODEL path in this script."
    exit 1
fi
echo "✅ Goal model found and accessible"

# Run the goal prediction test
echo "🔍 Running goal prediction test..."
echo "   • Uses config.yaml for reference_file and num_samples defaults"
echo "   • Goal model: $GOAL_MODEL"
echo ""

python goal_inference/goal_prediction_test.py --goal_model "$GOAL_MODEL"

# Success message
echo ""
echo "🎉 Goal prediction testing completed successfully!"
echo "📁 Check the output directory for detailed results and visualizations."
