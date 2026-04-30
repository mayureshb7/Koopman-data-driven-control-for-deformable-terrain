# conda environment 
conda activate chrono9
source /opt/ros/humble/setup.bash

#Training files to construct an npz that is extracted by the edmd later
python3 systems/MPC/residual_koopman_train.py   --world_id 1   --log_glob "logs/mpc_baseline_world1/mpc_hmmwv_world1_*.csv"   --out_dir "logs/koopman_models/s1_clean_snow_3"


#Execution code for MPC
python3 setup2.py vehicle=hmmwv system=mpc_run world_id=1 speed=6.0 max_time=60 render=true use_gui=false start_goal_idx=1 mpc_ref_profile=straight_then_sine mpc_ref_split_ratio=0.1 mpc_ref_sine_amplitude=15.0 mpc_ref_sine_cycles=1.0 mpc_track_speed_ref=false mpc_q_v=1e-6 mpc_goal_tolerance_m=5.0 mpc_log=true mpc_save_plots=true mpc_log_dir=logs/mpc

#Execution code for MPC - Koopman
python3 setup2.py vehicle=hmmwv system=mpc_koopman2 world_id=1 speed=6.0 max_time=60 render=true use_gui=false start_goal_idx=1 mpc_ref_profile=straight_then_sine mpc_ref_split_ratio=0.1 mpc_ref_sine_amplitude=15.0 mpc_ref_sine_cycles=1.0 mpc_track_speed_ref=false mpc_q_v=1e-6 mpc_goal_tolerance_m=5.0 rk_enable=true rk_model_path=logs/koopman_models/s1_clean_snow3/residual_koopman_model.npz rk_gain=1.2 rk_pred_horizon=400 rk_terminal_weight=5.0 mpc_log=true mpc_save_plots=true mpc_log_dir=logs/mpc


#requirements
vertibench simulation software
cconda environment chrono9

#setup2.py is used for execution of control as it is the same base for the vertibench examples for different systems of control used
