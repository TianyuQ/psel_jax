import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def select_agents_by_mask_np(predicted_mask, selection_method, mask_threshold, rank):
    """
    Selects agents based on a predicted mask using NumPy.
    """
    if selection_method == "threshold":
        selected_mask_indices = np.where(predicted_mask > mask_threshold)[0]
        selected_agents = selected_mask_indices + 1
    elif selection_method == "rank":
        num_other_agents = rank - 1
        if num_other_agents > 0:
            top_indices = np.argsort(predicted_mask)[-num_other_agents:]
            selected_agents = top_indices + 1
        else:
            selected_agents = np.array([])
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    return np.array(selected_agents)

def generate_figure(plot_plan, output_path):
    """
    Creates a static multi-panel figure from a specific plan of data sources.
    """
    num_rows = len(plot_plan)
    num_cols = 4  # t=1s, 2s, 3s, 4s, Full Trajectory

    snapshot_frame_indices = [9, 19, 29, 39]
    full_trajectory_end_frame = 49
    
    ego_color = 'darkblue'
    other_agent_color = 'gray'
    selected_color = 'red'

    fig, axes = plt.subplots(
        num_rows, num_cols, 
        figsize=(num_cols * 3, num_rows * 3), 
        squeeze=False,
        gridspec_kw={'wspace': 0.05, 'hspace': 0.05}
    )
    
    time_labels = ["t = 1s, t = 2s", "t = 3s", "t = 4s", "t = 5s"]

    for row_idx, row_info in enumerate(plot_plan):
        json_file_path = row_info['path']
        selection_config = row_info['selection_config']
        
        with open(json_file_path, 'r') as f:
            results = json.load(f)

        T_observation = results['T_observation']
        T_total = results['T_total']
        n_agents = 5
        normalized_sample_data = results['normalized_sample_data']

        if T_total <= full_trajectory_end_frame:
            print(f"Warning: Data for '{row_info['label']}' is too short. Skipping row.")
            continue
            
        all_steps = snapshot_frame_indices + [full_trajectory_end_frame]
        for col_idx, step in enumerate(all_steps):
            ax = axes[row_idx, col_idx]
            ax.set_aspect('equal')
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-3.5, 3.5)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # --- CHANGE #5: Always plot the full cumulative trajectory from the start ---
            start_idx, end_idx = 0, step

            # Plot non-ego trajectories
            for i in range(n_agents):
                if i == 0: continue
                agent_key = f"agent_{i}"
                if agent_key not in normalized_sample_data["trajectories"]: continue
                traj = np.array(normalized_sample_data["trajectories"][agent_key]["states"])
                if len(traj) == 0: continue
                
                for idx in range(start_idx, end_idx):
                    if idx + 1 >= len(traj): break
                    
                    is_selected = False
                    if selection_config['method'] == 'all':
                        is_selected = True
                    elif idx >= T_observation:
                        iteration_idx = idx - T_observation
                        if iteration_idx < len(results['receding_horizon_results']):
                            mask = np.array(results['receding_horizon_results'][iteration_idx]['predicted_mask'])
                            selected_agents = select_agents_by_mask_np(mask, selection_config['method'], selection_config['value'], selection_config['rank'])
                            if i in selected_agents:
                                is_selected = True

                    segment_color = selected_color if is_selected else other_agent_color
                    ax.plot(traj[idx:idx+2, 0], traj[idx:idx+2, 1], color=segment_color, linewidth=2, alpha=0.8)

            # Plot all agent markers and the ego trajectory
            for i in range(n_agents):
                agent_key = f"agent_{i}"
                if agent_key not in normalized_sample_data["trajectories"]: continue
                traj = np.array(normalized_sample_data["trajectories"][agent_key]["states"])
                if len(traj) == 0: continue
                
                if i == 0: # Plot ego trajectory
                    for idx in range(start_idx, end_idx):
                        if idx + 1 >= len(traj): break
                        ax.plot(traj[idx:idx+2, 0], traj[idx:idx+2, 1], color=ego_color, linewidth=2, alpha=0.8)

                # Determine marker color and shape
                marker_pos = traj[min(step, len(traj)-1), :2]
                is_selected_at_step = False
                if selection_config['method'] == 'all':
                    is_selected_at_step = True
                elif step >= T_observation:
                    iteration_idx = step - T_observation
                    if iteration_idx < len(results['receding_horizon_results']):
                        mask = np.array(results['receding_horizon_results'][iteration_idx]['predicted_mask'])
                        selected_agents = select_agents_by_mask_np(mask, selection_config['method'], selection_config['value'], selection_config['rank'])
                        if i in selected_agents:
                            is_selected_at_step = True
                
                marker_color = ego_color if i == 0 else (selected_color if is_selected_at_step else other_agent_color)
                
                # --- CHANGE #3: Use a circle for all agents' current position ---
                marker = 'o'
                size = 80
                ax.scatter(marker_pos[0], marker_pos[1], color=marker_color, marker=marker, s=size, zorder=5, edgecolors='black', linewidth=0.5)
                
                label_offset = 0.15
                ax.text(marker_pos[0] + label_offset, marker_pos[1] + label_offset, str(i), fontsize=8, ha='left', va='bottom', zorder=6)

            if 'target_positions' in normalized_sample_data:
                for i, pos in enumerate(normalized_sample_data['target_positions']):
                    goal_pos = np.array(pos[:2])
                    goal_color = ego_color if i == 0 else other_agent_color
                    # --- CHANGE #2: Make goal markers smaller ---
                    ax.plot(goal_pos[0], goal_pos[1], marker='*', color=goal_color, markersize=12, zorder=4, linestyle='None')

            if col_idx == 0:
                ax.set_ylabel(row_info['label'], fontsize=12, fontweight='bold')
                
            if row_idx == num_rows - 1:
                ax.annotate(
                    time_labels[col_idx], xy=(0.5, -0.05), xycoords='axes fraction',
                    ha='center', va='top', fontsize=12, fontweight='bold'
                )

    legend_elements = [
        # --- CHANGE #3 (Continued): Update legend to show circle for Ego Agent ---
        Line2D([0], [0], marker='o', color='w', label='Ego Agent', markerfacecolor=ego_color, markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Included Agent(s)', markerfacecolor=selected_color, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Excluded Agent(s)', markerfacecolor=other_agent_color, markersize=10)
    ]
    # --- CHANGE #1: Move the legend closer to the plots ---
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=3, fontsize=12)
    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    
    # --- CHANGE #4: Add bbox_inches='tight' to trim whitespace ---
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Figure saved to: {output_path}")

def main():
    """
    Main function to define a plot plan and generate a single figure for each sample set.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    log_dir_path = os.path.join(project_root, 'log')
    
    experiment_path = os.path.join(log_dir_path, 
                             "goal_true_N_4_T_50_obs_10", 
                             "psn_gru_full_planning_true_goals_N_4_T_50_obs_10_lr_0.002_bs_32_sigma1_0.5_sigma2_0.5_epochs_100")

    sources_config = [
        {
            'key': 'rank',
            'label': 'PSN-Full-Rank[2]', 
            'path': os.path.join(experiment_path, 'receding_horizon_results_planning_test_goal_true_full_rank_2_psn_best_model'),
            'selection_config': {'method': 'rank', 'value': None, 'rank': 2}
        },
        {
            'key': 'threshold',
            'label': 'PSN-Full-th[0.5]', 
            'path': os.path.join(experiment_path, 'receding_horizon_results_planning_test_goal_true_full_threshold_0.5_psn_best_model'),
            'selection_config': {'method': 'threshold', 'value': 0.5, 'rank': None}
        },
        {
            'key': 'all',
            'label': 'All', 
            'path': os.path.join(log_dir_path, 'receding_horizon_results_4_all_param_2_goal_inference'),
            'selection_config': {'method': 'all', 'value': None, 'rank': 10}
        }
    ]

    grouped_samples = {}
    print("Scanning and grouping sample files across all model directories...")
    for source in sources_config:
        run_dir = source['path']
        model_key = source['key']
        if not os.path.isdir(run_dir):
            print(f"  - WARNING: Directory not found for '{source['label']}':\n    {run_dir}")
            continue
        
        json_files = [f for f in os.listdir(run_dir) if f.endswith('.json') and 'receding_horizon_test_sample' in f]

        for fname in json_files:
            try:
                sample_num = int(fname.split('_')[-1].split('.')[0])
                grouped_samples.setdefault(sample_num, {})
                grouped_samples[sample_num][model_key] = os.path.join(run_dir, fname)
            except (ValueError, IndexError):
                print(f"  - Could not parse sample number from filename: {fname}")

    complete_samples = {num: paths for num, paths in grouped_samples.items() if len(paths) == len(sources_config)}

    if not complete_samples:
        print("\nNo complete sets of samples found across all specified directories. Exiting.")
        return

    print(f"\nFound {len(complete_samples)} complete sample sets to process.")
    total_figures_generated = 0

    for sample_num in sorted(complete_samples.keys()):
        print(f"\n--- Processing Sample #{sample_num} ---")
        sample_files = complete_samples[sample_num]
        
        plot_plan = []
        for source in sources_config:
            model_key = source['key']
            plot_plan.append({
                'label': source['label'],
                'path': sample_files[model_key],
                'selection_config': source['selection_config']
            })

        output_path = os.path.join(project_root, f"comparison_figs/comparison_figure_sample_{sample_num}.png")
        
        try:
            generate_figure(plot_plan, output_path)
            total_figures_generated += 1
        except Exception as e:
            print(f"  An unexpected error occurred while generating figure for sample #{sample_num}: {e}")
    
    print(f"\n✓ Script finished. Generated a total of {total_figures_generated} figures.")

if __name__ == "__main__":
    main()