import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr 

import os
import sys
import glob
import multiprocessing
import random
import numpy as np
import logging
import yaml
import argparse
from PIL import Image
import pandas as pd

class VehicleDataLogger:
    def __init__(self):
        self.rows = []
        self.initial_heights = {}

    def log(self, data):
        self.rows.append(data)

    def save(self, path="vehicle_log.csv"):
        df = pd.DataFrame(self.rows)
        df.to_csv(path, index=False)
        print(f"Saved log to {path}")


from verti_bench.envs.utils.utils import SetChronoDataDirectories
from verti_bench.systems.PID.PID_sim import PIDSim
from verti_bench.systems.EH.EH_sim import EHSim
from verti_bench.systems.MPPI.MPPI_sim import MPPISim
from verti_bench.systems.RL.RL_sim import RLSim
from verti_bench.systems.MCL.MCL_sim import MCLSim
from verti_bench.systems.ACL.ACL_sim import ACLSim
from verti_bench.systems.WMVCT.WMVCT_sim import WMVCTSim
from verti_bench.systems.MPPI6.MPPI6_sim import MPPI6Sim
from verti_bench.systems.TAL.TAL_sim import TALSim
from verti_bench.systems.TNT.TNT_sim import TNTSim
from verti_bench.systems.Manual.Manual_sim import ManualSim
try:
    from verti_bench.systems.MPC.MPC_SIM_Run import MPCSim as MPCSimRun
    from verti_bench.systems.MPC.MPC_Koopman2 import MPCKoopmanSim2
except ImportError:
    from systems.MPC.MPC_SIM_Run import MPCSim as MPCSimRun
    from systems.MPC.MPC_Koopman2 import MPCKoopmanSim2

def single_experiment(config):
    """Run a single simulation experiment"""
    # Create and initialize simulation
    logger = VehicleDataLogger()
    system_name = str(config.get('system', '')).strip().lower()

    if system_name == 'pid':
        sim = PIDSim(config)
    elif system_name == 'eh':
        sim = EHSim(config)
    elif system_name == 'mppi':
        sim = MPPISim(config,logger=logger)
    elif system_name == 'rl':
        sim = RLSim(config)
    elif system_name == 'mcl':
        sim = MCLSim(config)
    elif system_name == 'acl':
        sim = ACLSim(config)
    elif system_name == 'wmvct':
        sim = WMVCTSim(config)
    elif system_name == 'mppi6':
        sim = MPPI6Sim(config)
    elif system_name == 'tal':
        sim = TALSim(config)
    elif system_name == 'tnt':
        sim = TNTSim(config)
    elif system_name == 'manual':
        sim = ManualSim(config)
    elif system_name == 'mpc_run':
        # Alternate compact MPC implementation in systems/MPC/MPC_SIM_Run.py
        sim = MPCSimRun(config)
    elif system_name == 'mpc_koopman2':
        sim = MPCKoopmanSim2(config)
    elif system_name == 'mpc_koopman3':
        # Backward-compatible alias after consolidating compact Koopman variants.
        sim = MPCKoopmanSim2(config)
    else:
        supported_systems = (
            "pid, eh, mppi, rl, mcl, acl, wmvct, mppi6, tal, tnt, manual, "
            "mpc_run, mpc_koopman2, mpc_koopman3(alias)"
        )
        raise ValueError(
            f"Unsupported system '{config.get('system')}'. Supported systems: {supported_systems}"
        )
    sim.initialize()
    
    # Run simulation
    time_to_goal, success, avg_roll, avg_pitch = sim.run()

    sg_idx = getattr(sim, "start_goal_pair_idx", config.get("start_goal_idx", -1))
    log_name = (
        f"vehicle_log_{config['system']}_world{config['world_id']}_sg{sg_idx}.csv"
    )
    logger.save(log_name)
    
    # Return results
    return {
        'time_to_goal': time_to_goal if success else None,
        'success': success,
        'avg_roll': avg_roll,
        'avg_pitch': avg_pitch
    }

def multiple_experiments(config, num_experiments=5):
    """Run multiple simulation experiments and aggregate results"""
    results = []
    
    for i in range(num_experiments):
        print(f"Running experiment {i + 1}/{num_experiments}")
        result = single_experiment(config)
        results.append(result)
        
    # Process results 
    success_count = sum(1 for r in results if r['success'])
    successful_times = [r['time_to_goal'] for r in results if r['time_to_goal'] is not None]
    avg_rolls = [r['avg_roll'] for r in results if r['success']]
    avg_pitches = [r['avg_pitch'] for r in results if r['success']]

    mean_traversal_time = np.mean(successful_times) if successful_times else None
    roll_mean = np.mean(avg_rolls) if avg_rolls else None
    roll_variance = np.var(avg_rolls) if avg_rolls else None
    pitch_mean = np.mean(avg_pitches) if avg_pitches else None
    pitch_variance = np.var(avg_pitches) if avg_pitches else None

    # Print results
    print("--------------------------------------------------------------")
    print(f"Success rate: {success_count}/{num_experiments}")
    if success_count > 0:
        print(f"Mean traversal time (successful trials): {mean_traversal_time:.2f} seconds")
        print(f"Average roll angle: {roll_mean:.2f} degrees, Variance: {roll_variance:.2f}")
        print(f"Average pitch angle: {pitch_mean:.2f} degrees, Variance: {pitch_variance:.2f}")
    else:
        print("No successful trials")
    print("--------------------------------------------------------------")
    
    return results

def parse_arguments():
    processed_args = []
    for arg in sys.argv[1:]: 
        if '=' in arg and not arg.startswith('-'):
            key, value = arg.split('=', 1)
            processed_args.append(f"--{key}")
            processed_args.append(value)
        else:
            processed_args.append(arg)
    
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Run vehicle simulation with various configurations')
    
    # Vehicle and system parameters
    parser.add_argument('--vehicle', type=str, default='hmmwv', help='Vehicle type (default: hmmwv)')
    parser.add_argument('--system', type=str, default='mppi', help='Control system type (default: pid)')
    parser.add_argument('--speed', type=float, default=4.0, help='Target vehicle speed (default: 4.0)')
    
    # World parameters
    parser.add_argument('--world_id', type=int, default=1, help='World ID (1-100, default: 2)')
    parser.add_argument('--scale_factor', type=float, default=1.0, 
                        help='Scale factor for terrain (default: 1.0, options: 1.0, 1/6, 1/10)')
    parser.add_argument(
        '--start_goal_idx',
        type=int,
        default=-1,
        help='Fixed start/goal pair index for world positions (default: -1 random each run)',
    )
    
    # Simulation parameters
    parser.add_argument('--max_time', type=float, default=60.0, help='Maximum simulation time in seconds (default: 60.0)')
    parser.add_argument(
        '--mpc_goal_tolerance_m',
        type=float,
        default=8.0,
        help='Goal distance threshold in meters for success termination (default: 8.0)',
    )
    parser.add_argument('--num_experiments', type=int, default=1, 
                        help='Number of experiments to run (default: 1)')
    
    # Visualization parameters
    parser.add_argument('--render', type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help='Enable rendering (default: True)')
    parser.add_argument('--use_gui', type=lambda x: (str(x).lower() == 'true'), default=False, 
                        help='Enable GUI control (default: False)')
    parser.add_argument(
        '--mpc_live_plot',
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help='Enable MPC live matplotlib monitor (default: True)',
    )
    parser.add_argument(
        '--mpc_save_plots',
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help='Save MPC end-of-run PNG plots (default: True)',
    )
    parser.add_argument(
        '--mpc_log',
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help='Enable MPC CSV logging (default: True)',
    )
    parser.add_argument(
        '--mpc_log_dir',
        type=str,
        default='logs/mpc',
        help='Directory for MPC logs and plots (default: logs/mpc)',
    )
    parser.add_argument(
        '--mpc_live_plot_pause_s',
        type=float,
        default=0.001,
        help='Matplotlib live-plot pause duration in seconds; set 0 to remove pause (default: 0.001)',
    )
    parser.add_argument(
        '--mpc_ref_profile',
        type=str,
        default='straight',
        help='MPC fixed reference profile: straight|sine|curve|sine_then_curve|straight_then_sine (default: straight)',
    )
    parser.add_argument(
        '--mpc_ref_sine_amplitude',
        type=float,
        default=4.0,
        help='Lateral amplitude for sine profile in meters (default: 4.0)',
    )
    parser.add_argument(
        '--mpc_ref_sine_cycles',
        type=float,
        default=1.5,
        help='Number of sine cycles across path for sine profiles (default: 1.5)',
    )
    parser.add_argument(
        '--mpc_ref_curve_amplitude',
        type=float,
        default=6.0,
        help='Lateral amplitude for curve segment in meters (default: 6.0)',
    )
    parser.add_argument(
        '--mpc_ref_split_ratio',
        type=float,
        default=0.6,
        help='Split ratio for sine_then_curve segment transition (default: 0.6)',
    )
    parser.add_argument(
        '--mpc_track_speed_ref',
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help='Enable MPC speed-reference tracking (default: False, XY-only path tracking)',
    )
    parser.add_argument(
        '--mpc_v_ref_target',
        type=float,
        default=None,
        help='MPC speed reference target in m/s (default: use --speed)',
    )
    parser.add_argument(
        '--mpc_curvature_speed_gain',
        type=float,
        default=2.5,
        help='Speed reduction gain based on path curvature (default: 2.5)',
    )
    parser.add_argument(
        '--mpc_v_ref_min_scale',
        type=float,
        default=0.45,
        help='Minimum speed scaling on high-curvature segments (default: 0.45)',
    )
    parser.add_argument(
        '--mpc_q_v',
        type=float,
        default=1e-6,
        help='MPC speed tracking state weight Q_v (default: 1e-6 for XY-only tracking)',
    )
    parser.add_argument(
        '--mpc_ref_preview_speed',
        type=float,
        default=4.0,
        help='Waypoint horizon progression speed for reference indexing [m/s] (default: 4.0)',
    )
    parser.add_argument(
        '--mpc_ref_min_points_per_stage',
        type=float,
        default=0.15,
        help='Minimum waypoint index progression per stage (default: 0.15)',
    )
    parser.add_argument(
        '--rk_enable',
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help='Enable residual Koopman compensation for mpc_koopman mode (default: True)',
    )
    parser.add_argument(
        '--rk_model_path',
        type=str,
        default='logs/koopman_models/s1/residual_koopman_model.npz',
        help='Residual Koopman model .npz path (default: s1 residual model)',
    )
    parser.add_argument(
        '--rk_gain',
        type=float,
        default=1.0,
        help='Scale for residual Koopman compensation delta_u (default: 1.0)',
    )
    parser.add_argument(
        '--rk_pred_horizon',
        type=int,
        default=12,
        help='Residual Koopman multi-step prediction horizon (default: 12)',
    )
    parser.add_argument(
        '--rk_terminal_weight',
        type=float,
        default=5.0,
        help='Terminal state weight multiplier for residual multi-step controller (default: 5.0)',
    )
    parser.add_argument(
        '--rk_du_regularization',
        type=float,
        default=1e-3,
        help='Regularization for mpc_koopman3 one-step residual solve (default: 1e-3)',
    )


    args = parser.parse_args(processed_args)
    return args


if __name__ == '__main__':
    # Load configuration file
    SetChronoDataDirectories()
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create config dictionary from arguments
    config = {
        'vehicle': args.vehicle,
        'speed': args.speed,
        'system': args.system,
        'world_id': args.world_id,
        'start_goal_idx': args.start_goal_idx,
        'max_time': args.max_time,
        'mpc_goal_tolerance_m': args.mpc_goal_tolerance_m,
        'scale_factor': args.scale_factor,
        'render': args.render,
        'use_gui': args.use_gui,
        'mpc_live_plot': args.mpc_live_plot,
        'mpc_live_plot_pause_s': args.mpc_live_plot_pause_s,
        'mpc_save_plots': args.mpc_save_plots,
        'mpc_log': args.mpc_log,
        'mpc_log_dir': args.mpc_log_dir,
        'mpc_ref_profile': args.mpc_ref_profile,
        'mpc_ref_sine_amplitude': args.mpc_ref_sine_amplitude,
        'mpc_ref_sine_cycles': args.mpc_ref_sine_cycles,
        'mpc_ref_curve_amplitude': args.mpc_ref_curve_amplitude,
        'mpc_ref_split_ratio': args.mpc_ref_split_ratio,
        'mpc_track_speed_ref': args.mpc_track_speed_ref,
        'mpc_curvature_speed_gain': args.mpc_curvature_speed_gain,
        'mpc_v_ref_min_scale': args.mpc_v_ref_min_scale,
        'mpc_q_v': args.mpc_q_v,
        'mpc_ref_preview_speed': args.mpc_ref_preview_speed,
        'mpc_ref_min_points_per_stage': args.mpc_ref_min_points_per_stage,
        'rk_enable': args.rk_enable,
        'rk_model_path': args.rk_model_path,
        'rk_gain': args.rk_gain,
        'rk_pred_horizon': args.rk_pred_horizon,
        'rk_terminal_weight': args.rk_terminal_weight,
        'rk_du_regularization': args.rk_du_regularization
    }
    if args.mpc_v_ref_target is not None:
        config['mpc_v_ref_target'] = args.mpc_v_ref_target
    
    print("--------------------------------------------------------------")
    print("Verti-Bench Configs:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("--------------------------------------------------------------")
    
    # Run simulation
    multiple_experiments(config, num_experiments=args.num_experiments)
    
