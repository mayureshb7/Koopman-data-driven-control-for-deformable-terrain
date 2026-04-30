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
from verti_bench.systems.MPC.MPC_sim import MPCSim
from verti_bench.systems.MPC.MPC_Koopman import MPCKoopmanSim
from verti_bench.systems.MPC.MPC_Koopman2 import MPCKoopmanSim2
from verti_bench.systems.MPC.MPC_Koopman3 import MPCKoopmanSim3
from verti_bench.systems.MPC.MPC_Koopman_NN import MPCKoopmanNNSim

def single_experiment(config):
    """Run a single simulation experiment"""
    # Create and initialize simulation
    if config['system'] == 'pid':
        sim = PIDSim(config)
    elif config['system'] == 'eh':
        sim = EHSim(config)
    elif config['system'] == 'mppi':
        sim = MPPISim(config)
    elif config['system'] == 'rl':
        sim = RLSim(config)
    elif config['system'] == 'mcl':
        sim = MCLSim(config)
    elif config['system'] == 'acl':
        sim = ACLSim(config)
    elif config['system'] == 'wmvct':
        sim = WMVCTSim(config)
    elif config['system'] == 'mppi6':
        sim = MPPI6Sim(config)
    elif config['system'] == 'tal':
        sim = TALSim(config)
    elif config['system'] == 'tnt':
        sim = TNTSim(config)
    elif config['system'] == 'manual':
        sim = ManualSim(config)
    elif config['system'] == 'mpc':
        sim = MPCSim(config)
    elif config['system'] == 'mpc_koopman':
        sim = MPCKoopmanSim(config)
    elif config['system'] == 'mpc_koopman2':
        sim = MPCKoopmanSim2(config)
    elif config['system'] == 'mpc_koopman3':
        sim = MPCKoopmanSim3(config)
    elif config['system'] == 'mpc_koopman_nn':
        sim = MPCKoopmanNNSim(config)
    sim.initialize()
    
    # Run simulation
    time_to_goal, success, avg_roll, avg_pitch = sim.run()
    
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
    parser.add_argument('--system', type=str, default='pid', help='Control system type (default: pid)')
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
    parser.add_argument('--num_experiments', type=int, default=1, 
                        help='Number of experiments to run (default: 1)')
    
    # Visualization parameters
    parser.add_argument('--render', type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help='Enable rendering (default: True)')
    parser.add_argument('--use_gui', type=lambda x: (str(x).lower() == 'true'), default=False, 
                        help='Enable GUI control (default: False)')
    parser.add_argument(
        '--rk_nn_control_mode',
        type=str,
        default='one_step',
        choices=['one_step', 'finite_horizon', 'hybrid'],
        help='Control mode for mpc_koopman_nn residual controller (default: one_step)',
    )
    parser.add_argument(
        '--rk_nn_hybrid_error_low',
        type=float,
        default=0.1,
        help='Lower error-norm threshold for hybrid NN mode blend (default: 0.1)',
    )
    parser.add_argument(
        '--rk_nn_hybrid_error_high',
        type=float,
        default=0.5,
        help='Upper error-norm threshold for hybrid NN mode blend (default: 0.5)',
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
        'scale_factor': args.scale_factor,
        'render': args.render,
        'use_gui': args.use_gui,
        'rk_nn_control_mode': args.rk_nn_control_mode,
        'rk_nn_hybrid_error_low': args.rk_nn_hybrid_error_low,
        'rk_nn_hybrid_error_high': args.rk_nn_hybrid_error_high,
    }
    
    print("--------------------------------------------------------------")
    print("Verti-Bench Configs:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("--------------------------------------------------------------")
    
    # Run simulation
    multiple_experiments(config, num_experiments=args.num_experiments)
    
