python goal_inference_pretraining/test_trained_psn.py \
    --psn_model log/goal_inference_N_4_T_50_obs_10_lr_0.001_bs_32_goal_loss_weight_1.0_epochs_1000/psn_pretrained_goals_N_4_T_50_obs_10_lr_0.001_bs_32_sigma1_-0.1_sigma2_0.0_epochs_50/psn_best_model.pkl \
    --goal_model log/goal_inference_N_4_T_50_obs_10_lr_0.001_bs_32_goal_loss_weight_1.0_epochs_1000/goal_inference_best_model.pkl \
    --reference_file reference_trajectories_4p/all_reference_trajectories.json \
    --num_samples 10