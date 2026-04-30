import multiprocessing
import queue
import random

import numpy as np
import pychrono as chrono
import pychrono.vehicle as veh
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from verti_bench.MPC.bicycle_model_AxStr import export_bicycle_model
from verti_bench.envs.terrain import TerrainManager
from verti_bench.vehicles.ART import ARTManager
from verti_bench.vehicles.FEDA import FEDAManager
from verti_bench.vehicles.Gator import GatorManager
from verti_bench.vehicles.HMMWV import HMMWVManager
from verti_bench.vehicles.M113 import M113Manager
from verti_bench.vehicles.MAN5t import MAN5tManager
from verti_bench.vehicles.MAN7t import MAN7tManager
from verti_bench.vehicles.MAN10t import MAN10tManager
from verti_bench.vehicles.VW import VWManager
try:
    from verti_bench.systems.MPC.mpc_external_log_plot import MPCExternalLogger
except Exception:
    from systems.MPC.mpc_external_log_plot import MPCExternalLogger


def _wrap_angle(ang):
    return (ang + np.pi) % (2 * np.pi) - np.pi


#queue for latest control command
def _put_latest(q_obj, item):
    for _ in range(3):
        try:
            q_obj.put_nowait(item)
            return True
        except queue.Full:
            try:
                q_obj.get_nowait()
            except queue.Empty:
                pass
        except (BrokenPipeError, EOFError, OSError):
            return False
    return False

# get latest control command from queue as above
def _get_latest(q_obj):
    latest = None
    while True:
        try:
            latest = q_obj.get_nowait()
        except queue.Empty:
            break
    return latest

#Allows to generate the path for the vehicle that is based on either straight, sine or straight and sine
def _build_path(
    start_xy,
    goal_xy,
    n_points,
    path_profile="straight",
    sine_amplitude=0.0,
    sine_cycles=1.0,
    curve_amplitude=0.0,
    split_ratio=0.6,
):
    n_points = max(2, int(n_points))
    dx = float(goal_xy[0] - start_xy[0])
    dy = float(goal_xy[1] - start_xy[1])
    dist = float(np.hypot(dx, dy))
    if dist < 1e-9:
        x_vals = np.full((n_points,), float(start_xy[0]), dtype=float)
        y_vals = np.full((n_points,), float(start_xy[1]), dtype=float)
        phi = np.zeros((n_points,), dtype=float)
        return np.column_stack((x_vals, y_vals, phi))

    ex = dx / dist
    ey = dy / dist
    nx = -ey
    ny = ex
    s = np.linspace(0.0, 1.0, n_points, dtype=float)
    x_line = float(start_xy[0]) + s * dx
    y_line = float(start_xy[1]) + s * dy
    split_ratio = float(np.clip(split_ratio, 0.05, 0.95))
    offset = np.zeros_like(s)
    profile = str(path_profile).strip().lower()

    if profile == "sine":
        offset = float(sine_amplitude) * np.sin(2.0 * np.pi * float(sine_cycles) * s)
    elif profile == "curve":
        offset = float(curve_amplitude) * np.sin(np.pi * s)
    elif profile == "sine_then_curve":
        s1 = s <= split_ratio
        s2 = s > split_ratio
        if np.any(s1):
            s_local = s[s1] / split_ratio
            offset[s1] = float(sine_amplitude) * np.sin(2.0 * np.pi * float(sine_cycles) * s_local)
        if np.any(s2):
            s_local = (s[s2] - split_ratio) / (1.0 - split_ratio)
            offset[s2] = float(curve_amplitude) * np.sin(np.pi * s_local)
    elif profile == "straight_then_sine":
        s2 = s > split_ratio
        if np.any(s2):
            s_local = (s[s2] - split_ratio) / (1.0 - split_ratio)
            ramp = 0.5 * (1.0 - np.cos(np.pi * s_local))
            offset[s2] = float(sine_amplitude) * ramp * np.sin(2.0 * np.pi * float(sine_cycles) * s_local)

    x_vals = x_line + offset * nx
    y_vals = y_line + offset * ny
    dx = np.gradient(x_vals)
    dy = np.gradient(y_vals)
    phi = _wrap_angle(np.arctan2(dy, dx))
    return np.column_stack((x_vals, y_vals, phi))


#based on curren position and thhe path reference it creates asmooth interpolated waypoint used for references further
# This function is used in the GENREFS below that generets yref values corresponding to the ACADOS format
def _interpolate_path(path_xyphi, idx_float):
    if path_xyphi is None or len(path_xyphi) == 0:
        return 0.0, 0.0, 0.0
    idx_float = float(np.clip(idx_float, 0.0, len(path_xyphi) - 1))
    i0 = int(np.floor(idx_float))
    i1 = min(i0 + 1, len(path_xyphi) - 1)
    alpha = float(idx_float - i0)
    p0 = path_xyphi[i0]
    p1 = path_xyphi[i1]
    x = (1.0 - alpha) * float(p0[0]) + alpha * float(p1[0])
    y = (1.0 - alpha) * float(p0[1]) + alpha * float(p1[1])
    dphi = _wrap_angle(float(p1[2]) - float(p0[2]))
    phi = _wrap_angle(float(p0[2]) + alpha * dphi)
    return x, y, phi


def _gen_yrefs(xcurrent, path_xyphi, worker_cfg, progress_idx):
    n_horizon = int(worker_cfg["N_horizon"])
    yrefs = np.zeros((n_horizon + 1, 7), dtype=float)
    if path_xyphi is None or len(path_xyphi) == 0:
        return yrefs, int(progress_idx)

    path_xy = path_xyphi[:, :2]
    search_start = int(np.clip(progress_idx, 0, len(path_xyphi) - 1))
    dxy = path_xy[search_start:] - np.array([xcurrent[0], xcurrent[1]], dtype=float)
    nearest_local = int(np.argmin(np.einsum("ij,ij->i", dxy, dxy)))
    nearest_idx = search_start + nearest_local
    progress_idx = max(float(progress_idx), float(nearest_idx))

    if nearest_idx < len(path_xyphi) - 1:
        p0 = path_xy[nearest_idx]
        p1 = path_xy[nearest_idx + 1]
        seg = p1 - p0
        seg2 = float(np.dot(seg, seg))
        if seg2 > 1e-9:
            rel = np.array([xcurrent[0], xcurrent[1]], dtype=float) - p0
            tau = float(np.clip(np.dot(rel, seg) / seg2, 0.0, 1.0))
            progress_idx = max(progress_idx, float(nearest_idx) + tau)

    dt_stage = float(worker_cfg["T_horizon"]) / max(n_horizon, 1)
    v_preview = max(float(worker_cfg.get("ref_preview_speed", xcurrent[4])), 0.0)
    spacing = max(float(worker_cfg.get("ref_spacing", 1.0)), 1e-3)
    points_per_stage = max((v_preview * dt_stage) / spacing, float(worker_cfg.get("ref_min_points_per_stage", 0.2)))

    last_idx = len(path_xyphi) - 1
    track_speed_ref = bool(worker_cfg.get("track_speed_ref", False))
    if track_speed_ref:
        v_ref = float(np.clip(worker_cfg.get("v_ref_target", xcurrent[4]), worker_cfg["v_min"], worker_cfg["v_max"]))
    else:
        v_ref = float(np.clip(xcurrent[4], worker_cfg["v_min"], worker_cfg["v_max"]))
    for j in range(n_horizon + 1):
        idx_float = min(progress_idx + j * points_per_stage, float(last_idx))
        xr, yr, phir = _interpolate_path(path_xyphi, idx_float)
        yrefs[j, :] = np.array([xr, yr, phir, 0.0, v_ref, 0.0, 0.0], dtype=float)
    return yrefs, float(progress_idx)

# The OCP solver as needed for ACADOS library is created here
def _create_ocp_solver(worker_cfg):
    ocp = AcadosOcp()
    model = export_bicycle_model()
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx
    nz = model.z.size()[0]

    ocp.dims.N = int(worker_cfg["N_horizon"])

    q_mat = np.diag([1e4, 1e4, 2e4, 1e-6, 1e3])
    q_mat[4, 4] = float(worker_cfg.get("q_v", 1e-6))
    r_mat = np.diag([8e4, 2e3])
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W_e = 2 * q_mat
    ocp.cost.W = scipy.linalg.block_diag(q_mat, r_mat)

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    vu = np.zeros((ny, nu))
    vu[nx : nx + nu, :nu] = np.eye(nu)
    ocp.cost.Vu = vu
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.Vz = np.eye(ny, nz)
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    ocp.constraints.lbu = np.array([worker_cfg["ax_min"], worker_cfg["ddelta_min"]])
    ocp.constraints.ubu = np.array([worker_cfg["ax_max"], worker_cfg["ddelta_max"]])
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbx = np.array([worker_cfg["theta_min"], worker_cfg["v_min"]])
    ocp.constraints.ubx = np.array([worker_cfg["theta_max"], worker_cfg["v_max"]])
    ocp.constraints.idxbx = np.array([3, 4])
    ocp.constraints.x0 = np.zeros((nx,))

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 200
    ocp.solver_options.tf = float(worker_cfg["T_horizon"])

    solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json")
    return solver, nx


#MPC is solved here status = solver.solve() with the bounds from the worker and the yref from genrefs
def _mpc_worker_loop(state_queue, control_queue, stop_event, worker_cfg, fixed_ref_xyphi):
    solver, nx = _create_ocp_solver(worker_cfg)
    progress_idx = 0

    while not stop_event.is_set():
        try:
            state = state_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        if state is None:
            break

        xcurrent = np.array(state, dtype=float)
        if xcurrent.shape[0] != nx:
            continue

        solver.set(0, "lbx", xcurrent)
        solver.set(0, "ubx", xcurrent)
        yrefs, progress_idx = _gen_yrefs(xcurrent, fixed_ref_xyphi, worker_cfg, progress_idx)

        for stage in range(int(worker_cfg["N_horizon"])):
            solver.set(stage, "yref", yrefs[stage, :])
        solver.set(int(worker_cfg["N_horizon"]), "yref", yrefs[int(worker_cfg["N_horizon"]), 0:5])

        status = solver.solve()
        if status not in (0, 2):
            _put_latest(
                control_queue,
                {"solver_status": int(status), "ax_cmd": None, "ddelta_cmd": None},
            )
            continue

        u0 = solver.get(0, "u")
        _put_latest(
            control_queue,
            {
                "solver_status": int(status),
                "ax_cmd": float(np.clip(u0[0], worker_cfg["ax_min"], worker_cfg["ax_max"])),
                "ddelta_cmd": float(np.clip(u0[1], worker_cfg["ddelta_min"], worker_cfg["ddelta_max"])),
            },
        )


class MPCSim:
    def __init__(self, config):
        if config["use_gui"] and not config["render"]:
            raise ValueError("If use_gui is True, render must also be True.")

        # Store configuration parameters
        self.config = config
        self.world_id = int(config["world_id"])
        self.scale_factor = float(config["scale_factor"])
        self.render = bool(config["render"])
        self.use_gui = bool(config["use_gui"])
        self.vehicle_type = str(config["vehicle"])
        self.system_type = str(config["system"]).lower()
        self.max_time = float(config["max_time"])
        self.speed = float(config["speed"])
        self.vehicle_type_lower = self.vehicle_type.lower()

        if self.system_type not in (
            "mpc",
            "mpc_run",
            "mpc_koopman",
            "mpc_koopman2",
            "mpc_koopman3",
            "mpc_koopman_nn",
        ):
            raise ValueError(f"Unsupported system type: {self.system_type}.")
        if not (1 <= self.world_id <= 100):
            raise ValueError(f"World ID must be in [1, 100], got {self.world_id}")
        
        # Initialize system
        self.system = chrono.ChSystemNSC()
        self.system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
        self.system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        num_procs = multiprocessing.cpu_count()
        self.system.SetNumThreads(min(8, num_procs), min(8, num_procs), 1)
        
        # Set thread counts based on available CPUs
        self.step_size = self._step_size()
        self.goal_tolerance_m = float(self.config.get("mpc_goal_tolerance_m", 8.0 * self.scale_factor))
        self.stuck_distance = float(self.config.get("mpc_stuck_distance", 0.01))
        self.stuck_time = self._stuck_time()
        self.stuck_counter = 0.0
        self.last_position = None
        self.vis = None
        self.driver = None
        self.driver_inputs = None
        self.vis_freq = 100.0
        self.vis_dur = 1.0 / self.vis_freq
        self.last_vis_time = 0.0

        # Simulation parameters MPC based
        self.N_horizon = int(self.config.get("mpc_N_horizon", 400))
        self.T_horizon = float(self.config.get("mpc_T_horizon", 4.0))
        self.ax_min = float(self.config.get("mpc_ax_min", -5.0))
        self.ax_max = float(self.config.get("mpc_ax_max", 5.0))
        self.ddelta_min = float(self.config.get("mpc_ddelta_min", -0.20))
        self.ddelta_max = float(self.config.get("mpc_ddelta_max", 0.20))
        self.theta_min = float(self.config.get("mpc_theta_min", -0.57))
        self.theta_max = float(self.config.get("mpc_theta_max", 0.57))
        self.v_min = float(self.config.get("mpc_v_min", 0.0))
        self.v_max = float(self.config.get("mpc_v_max", max(7.0, self.speed + 2.0)))
        self.vehicle_accel_max = float(max(self.config.get("mpc_vehicle_accel_max", self.ax_max), 1e-3))
        self.vehicle_brake_max = float(max(self.config.get("mpc_vehicle_brake_max", abs(self.ax_min)), 1e-3))
        self.wait_for_first_mpc_cmd = bool(self.config.get("mpc_wait_for_first_cmd", True))

        self.worker_cfg = {
            "N_horizon": self.N_horizon,
            "T_horizon": self.T_horizon,
            "ax_min": self.ax_min,
            "ax_max": self.ax_max,
            "ddelta_min": self.ddelta_min,
            "ddelta_max": self.ddelta_max,
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "v_ref_target": float(self.config.get("mpc_v_ref_target", self.speed)),
            "track_speed_ref": bool(self.config.get("mpc_track_speed_ref", False)),
            "ref_preview_speed": float(self.config.get("mpc_ref_preview_speed", self.speed)),
            "ref_min_points_per_stage": float(self.config.get("mpc_ref_min_points_per_stage", 0.2)),
            "ref_spacing": float(self.config.get("mpc_ref_spacing", 1.0)),
            "q_v": float(self.config.get("mpc_q_v", 1e-6)),
        }
        self.state_queue = None
        self.control_queue = None
        self.worker_stop_event = None
        self.worker_process = None

        self.delta_estimate = 0.0
        self.has_received_mpc_cmd = False
        self.current_ax_cmd = 0.0
        self.current_ddelta_cmd = 0.0
        self.current_v_target = float(self.speed)
        self.last_mpc_solver_status = -999
        self.start_goal_pair_idx = -1
        self.current_yref0 = np.full((7,), np.nan, dtype=float)
        self.current_ref_xy = None
        self.current_ref_xyphi = None
        self.ref_progress_idx = 0
        self.current_pred_xy = None

        # Minimal live plotting (ported from MPC_sim.py style).
        self.enable_live_plot = bool(self.config.get("mpc_live_plot", True))
        self.live_plot_dt = float(self.config.get("mpc_live_plot_dt", 0.2))
        self.live_plot_pause_s = float(self.config.get("mpc_live_plot_pause_s", 0.001))
        self.last_live_plot_time = -1.0
        self.live_plot_initialized = False
        self.live_fig = None
        self.live_axes = None
        self.mpc_log_rows = []
        self.external_logger = MPCExternalLogger(self)

        self.terrain_manager = TerrainManager(self.world_id, self.scale_factor)
        self._initialize_vehicle()

    def _step_size(self):
        step_sizes = {
            "hmmwv": 5e-3,
            "gator": 2e-3,
            "feda": 1e-3,
            "man5t": 1e-3,
            "man7t": 1e-3,
            "man10t": 1e-3,
            "m113": 8e-4,
            "art": 1e-3,
            "vw": 3e-4,
            "default": 1e-3,
        }
        return step_sizes.get(self.vehicle_type_lower, step_sizes["default"])

    def _stuck_time(self):
        stuck_time = {
            "hmmwv": 10,
            "gator": 40,
            "feda": 40,
            "man5t": 50,
            "man7t": 50,
            "man10t": 60,
            "m113": 60,
            "art": 60,
            "vw": 60,
            "default": 10,
        }
        return stuck_time.get(self.vehicle_type_lower, stuck_time["default"])

    def _initialize_vehicle(self):
        if self.vehicle_type_lower == "hmmwv":
            self.vehicle_manager = HMMWVManager(self.system, self.step_size)
        elif self.vehicle_type_lower == "gator":
            self.vehicle_manager = GatorManager(self.system, self.step_size)
        elif self.vehicle_type_lower == "feda":
            self.vehicle_manager = FEDAManager(self.system, self.step_size)
        elif self.vehicle_type_lower == "man5t":
            self.vehicle_manager = MAN5tManager(self.system, self.step_size)
        elif self.vehicle_type_lower == "man7t":
            self.vehicle_manager = MAN7tManager(self.system, self.step_size)
        elif self.vehicle_type_lower == "man10t":
            self.vehicle_manager = MAN10tManager(self.system, self.step_size)
        elif self.vehicle_type_lower == "m113":
            self.vehicle_manager = M113Manager(self.system, self.step_size)
        elif self.vehicle_type_lower == "art":
            self.vehicle_manager = ARTManager(self.system, self.step_size)
        elif self.vehicle_type_lower == "vw":
            self.vehicle_manager = VWManager(self.system, self.step_size)
        else:
            raise ValueError(f"Unsupported vehicle type: {self.vehicle_type}.")

    def _setup_visualization(self):
        self.vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
        if self.vehicle_type_lower in ["m113"]:
            self.vis = veh.ChTrackedVehicleVisualSystemIrrlicht()
        self.vis.SetWindowTitle("vws in the wild")
        self.vis.SetWindowSize(1920, 1080)
        self.vis.SetChaseCamera(chrono.ChVector3d(-1.0, 0.0, 1.75), 6.0, 0.5)
        self.vis.Initialize()
        self.vis.AddLightDirectional()
        self.vis.AddSkyBox()
        self.vis.AttachVehicle(self.vehicle_manager.vehicle.GetVehicle())
        self.vis.EnableStats(True)

    def _setup_driver(self):
        if self.use_gui:
            self.driver = veh.ChInteractiveDriverIRR(self.vis)
            self.driver.SetSteeringDelta(0.1)
            self.driver.SetThrottleDelta(0.02)
            self.driver.SetBrakingDelta(0.06)
            self.driver.Initialize()
        else:
            self.driver = veh.ChDriver(self.vehicle_manager.vehicle.GetVehicle())
        self.driver_inputs = self.driver.GetInputs()

    # Gets the information of current states frem the simulator or vehicle manager
    def _extract_mpc_state(self):
        pos = self.vehicle_manager.get_position()
        rot = self.vehicle_manager.get_rotation()
        speed = self.vehicle_manager.get_speed()
        return np.array([pos.x, pos.y, _wrap_angle(rot.z), self.delta_estimate, speed], dtype=float)

    # zero order hold solving of the MPC tracking problem.
    def _solve_mpc(self, x_state):
        # Gets the references, the latest command from the mpc worker and sends infromation to the queue
        yrefs, self.ref_progress_idx = _gen_yrefs(x_state, self.fixed_ref_xyphi, self.worker_cfg, self.ref_progress_idx)
        self.current_ref_xy = yrefs[:, 0:2]
        self.current_ref_xyphi = yrefs[:, 0:3]
        self.current_yref0 = yrefs[0, :].copy()

        if self.worker_process is not None and self.worker_process.is_alive() and self.state_queue is not None:
            _put_latest(self.state_queue, x_state)

        latest_u = _get_latest(self.control_queue) if self.control_queue is not None else None
        if not isinstance(latest_u, dict):
            return False
        self.last_mpc_solver_status = int(latest_u.get("solver_status", -999))

        ax_cmd = latest_u.get("ax_cmd", None)
        ddelta_cmd = latest_u.get("ddelta_cmd", None)
        if ax_cmd is None or ddelta_cmd is None:
            return False

        ax_cmd, ddelta_cmd = self._augment_control_with_residual(x_state, float(ax_cmd), float(ddelta_cmd))
        self.current_ax_cmd = float(np.clip(ax_cmd, self.ax_min, self.ax_max))
        self.current_ddelta_cmd = float(np.clip(ddelta_cmd, self.ddelta_min, self.ddelta_max))
        self.has_received_mpc_cmd = True
        return True

    def _start_mpc_worker(self):
        mp_ctx = multiprocessing.get_context("spawn")
        self.state_queue = mp_ctx.Queue(maxsize=1)
        self.control_queue = mp_ctx.Queue(maxsize=1)
        self.worker_stop_event = mp_ctx.Event()
        self.worker_process = mp_ctx.Process(
            target=_mpc_worker_loop,
            args=(
                self.state_queue,
                self.control_queue,
                self.worker_stop_event,
                self.worker_cfg,
                self.fixed_ref_xyphi,
            ),
            daemon=True,
        )
        self.worker_process.start()

    def _stop_mpc_worker(self):
        if self.worker_process is None:
            return

        if self.worker_stop_event is not None:
            self.worker_stop_event.set()
        if self.state_queue is not None:
            _put_latest(self.state_queue, None)

        self.worker_process.join(timeout=1.0)
        if self.worker_process.is_alive():
            self.worker_process.terminate()
            self.worker_process.join(timeout=0.5)

        self.worker_process = None
        self.worker_stop_event = None
        self.state_queue = None
        self.control_queue = None

    # estimate the delta and also cip the acceleration and brake command to match the ax and throttle
    def _apply_driver_inputs(self):
        if self.wait_for_first_mpc_cmd and not self.has_received_mpc_cmd:
            self.driver_inputs.m_steering = 0.0
            self.driver_inputs.m_throttle = 0.0
            self.driver_inputs.m_braking = 1.0
            return

        self.delta_estimate = float(
            np.clip(self.delta_estimate + self.current_ddelta_cmd * self.step_size, self.theta_min, self.theta_max)
        )
        steer_norm = float(np.clip(self.delta_estimate / max(self.theta_max, 1e-6), -1.0, 1.0))
        ax_cmd = float(np.clip(self.current_ax_cmd, self.ax_min, self.ax_max))

        if ax_cmd >= 0.0:
            throttle = float(np.clip(ax_cmd / self.vehicle_accel_max, 0.0, 1.0))
            braking = 0.0
        else:
            throttle = 0.0
            braking = float(np.clip((-ax_cmd) / self.vehicle_brake_max, 0.0, 1.0))

        self.driver_inputs.m_steering = steer_norm
        self.driver_inputs.m_throttle = throttle
        self.driver_inputs.m_braking = braking

    def _augment_control_with_residual(self, x_state, ax_cmd, ddelta_cmd):
        #this is a passthru if the koopman is not used in MPC if it is then the ax_command that is generated is from the residual koopman
        return float(ax_cmd), float(ddelta_cmd)

    def _append_log_row(self, time_now, vehicle_pos, roll, pitch, yaw, speed, goal_dist):
        self.external_logger.append_log_row(
            time_now=time_now,
            vehicle_pos=vehicle_pos,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=speed,
            goal_dist=goal_dist,
        )

    def _init_live_plot(self):
        if not self.enable_live_plot or plt is None or self.live_plot_initialized:
            return

        plt.ion()
        self.live_fig = plt.figure(figsize=(13, 10.5), constrained_layout=True)
        gs = self.live_fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 1.25])
        self.live_axes = {
            "xy": self.live_fig.add_subplot(gs[0, 0]),
            "speed": self.live_fig.add_subplot(gs[0, 1]),
            "mpc": self.live_fig.add_subplot(gs[1, 0]),
            "driver": self.live_fig.add_subplot(gs[1, 1]),
            "poserr": self.live_fig.add_subplot(gs[2, 0]),
            "goal": self.live_fig.add_subplot(gs[2, 1]),
            "pred": self.live_fig.add_subplot(gs[3, :]),
        }
        self.live_fig.suptitle("MPC_SIM_Run Live Monitor")
        self.live_plot_initialized = True
        self.last_live_plot_time = -1.0

    def _update_live_plot(self, time_now):
        if not self.enable_live_plot or plt is None or not self.live_plot_initialized:
            return
        if self.last_live_plot_time >= 0.0 and (time_now - self.last_live_plot_time) < self.live_plot_dt:
            return
        if len(self.mpc_log_rows) < 2:
            return

        t = np.array([r["time"] for r in self.mpc_log_rows], dtype=float)
        x = np.array([r["x"] for r in self.mpc_log_rows], dtype=float)
        y = np.array([r["y"] for r in self.mpc_log_rows], dtype=float)
        v = np.array([r["speed"] for r in self.mpc_log_rows], dtype=float)
        ax_cmd = np.array([r["ax_cmd"] for r in self.mpc_log_rows], dtype=float)
        ddelta_cmd = np.array([r["ddelta_cmd"] for r in self.mpc_log_rows], dtype=float)
        steering = np.array([r["steering_cmd"] for r in self.mpc_log_rows], dtype=float)
        throttle = np.array([r["throttle_cmd"] for r in self.mpc_log_rows], dtype=float)
        braking = np.array([r["braking_cmd"] for r in self.mpc_log_rows], dtype=float)
        goal_distance = np.array([r["goal_distance"] for r in self.mpc_log_rows], dtype=float)
        position_error = np.array([r.get("position_error", np.nan) for r in self.mpc_log_rows], dtype=float)

        ax_xy = self.live_axes["xy"]
        ax_speed = self.live_axes["speed"]
        ax_mpc = self.live_axes["mpc"]
        ax_driver = self.live_axes["driver"]
        ax_poserr = self.live_axes["poserr"]
        ax_goal = self.live_axes["goal"]
        ax_pred = self.live_axes["pred"]

        ax_xy.clear()
        ax_xy.plot(x, y, linewidth=2.0, label="trajectory")
        if self.fixed_ref_xyphi is not None and len(self.fixed_ref_xyphi) > 0:
            ax_xy.plot(
                self.fixed_ref_xyphi[:, 0],
                self.fixed_ref_xyphi[:, 1],
                linestyle="-",
                linewidth=1.8,
                color="tab:purple",
                alpha=0.9,
                label="fixed reference",
            )
        if self.current_ref_xy is not None and len(self.current_ref_xy) > 0:
            ax_xy.plot(
                self.current_ref_xy[:, 0],
                self.current_ref_xy[:, 1],
                linestyle="--",
                linewidth=1.5,
                color="tab:orange",
                label="current refs",
            )
        if self.current_pred_xy is not None and len(self.current_pred_xy) > 0:
            ax_xy.plot(
                self.current_pred_xy[:, 0],
                self.current_pred_xy[:, 1],
                linestyle="-.",
                linewidth=1.8,
                color="tab:red",
                alpha=0.95,
                label="mpc prediction",
            )
        ax_xy.scatter(x[0], y[0], c="g", s=35, label="start")
        ax_xy.scatter(x[-1], y[-1], c="tab:blue", s=30, label="vehicle now")
        if hasattr(self.vehicle_manager, "goal"):
            ax_xy.scatter(
                [self.vehicle_manager.goal.x],
                [self.vehicle_manager.goal.y],
                c="r",
                marker="*",
                s=70,
                label="goal",
            )
        ax_xy.set_title("XY position")
        ax_xy.set_xlabel("x [m]")
        ax_xy.set_ylabel("y [m]")
        ax_xy.grid(True, alpha=0.3)
        ax_xy.axis("equal")
        ax_xy.legend(loc="upper right", fontsize=8)

        ax_speed.clear()
        ax_speed.plot(t, v, label="speed")
        ax_speed.set_title("Speed")
        ax_speed.set_xlabel("time [s]")
        ax_speed.set_ylabel("m/s")
        ax_speed.grid(True, alpha=0.3)
        ax_speed.legend(loc="upper right", fontsize=8)

        ax_mpc.clear()
        ax_mpc.plot(t, ax_cmd, label="ax_cmd")
        ax_mpc.plot(t, ddelta_cmd, label="ddelta_cmd")
        ax_mpc.set_title("MPC controls")
        ax_mpc.set_xlabel("time [s]")
        ax_mpc.grid(True, alpha=0.3)
        ax_mpc.legend(loc="upper right", fontsize=8)

        ax_driver.clear()
        ax_driver.plot(t, steering, label="steering")
        ax_driver.plot(t, throttle, label="throttle")
        ax_driver.plot(t, braking, label="braking")
        ax_driver.set_title("Applied driver inputs")
        ax_driver.set_xlabel("time [s]")
        ax_driver.grid(True, alpha=0.3)
        ax_driver.legend(loc="upper right", fontsize=8)

        ax_poserr.clear()
        ax_poserr.plot(t, position_error, label="position_error", color="tab:red")
        ax_poserr.set_title("Position error")
        ax_poserr.set_xlabel("time [s]")
        ax_poserr.set_ylabel("m")
        ax_poserr.grid(True, alpha=0.3)
        ax_poserr.legend(loc="upper right", fontsize=8)

        ax_goal.clear()
        ax_goal.plot(t, goal_distance, label="goal_distance", color="tab:green")
        ax_goal.set_title("Goal distance")
        ax_goal.set_xlabel("time [s]")
        ax_goal.set_ylabel("m")
        ax_goal.grid(True, alpha=0.3)
        ax_goal.legend(loc="upper right", fontsize=8)

        ax_pred.clear()
        if len(x) > 80:
            ax_pred.plot(x[-80:], y[-80:], linewidth=2.0, color="tab:blue", label="recent executed path")
        else:
            ax_pred.plot(x, y, linewidth=2.0, color="tab:blue", label="executed path")
        if self.current_ref_xy is not None and len(self.current_ref_xy) > 0:
            ax_pred.plot(
                self.current_ref_xy[:, 0],
                self.current_ref_xy[:, 1],
                linestyle="--",
                linewidth=1.4,
                color="tab:orange",
                label="current refs",
            )
        ax_pred.scatter([x[-1]], [y[-1]], c="tab:blue", s=22, label="vehicle now")
        zoom_half = float(self.config.get("mpc_live_plot_zoom_halfspan_m", 8.0 * self.scale_factor))
        ax_pred.set_xlim(x[-1] - zoom_half, x[-1] + zoom_half)
        ax_pred.set_ylim(y[-1] - zoom_half, y[-1] + zoom_half)
        ax_pred.set_aspect("equal", adjustable="box")
        ax_pred.set_title("Prediction vs executed (local view)")
        ax_pred.set_xlabel("x [m]")
        ax_pred.set_ylabel("y [m]")
        ax_pred.grid(True, alpha=0.3)
        ax_pred.legend(loc="upper left", fontsize=8, ncol=2)

        self.live_fig.canvas.draw_idle()
        if self.live_plot_pause_s > 0.0:
            plt.pause(self.live_plot_pause_s)
        self.last_live_plot_time = time_now

    def _close_live_plot(self):
        if plt is None or not self.live_plot_initialized:
            return
        try:
            plt.ioff()
            plt.close(self.live_fig)
        except Exception:
            pass
        self.live_plot_initialized = False
        self.live_fig = None
        self.live_axes = None

    # reaqlized this is needed and it is same as the MPPI with a bit of odifications for mpc
    # DO not CHANGE!!! struggled with the errors here 

    def initialize(self, start_pos=None, goal_pos=None):
        if start_pos is None or goal_pos is None:
            positions = self.terrain_manager.positions
            fixed_idx = int(self.config.get("start_goal_idx", -1)) #matches mpc start goal idx
            if fixed_idx >= 0:
                if fixed_idx >= len(positions):
                    raise ValueError(
                        f"start_goal_idx={fixed_idx} out of range for world {self.world_id}. "
                        f"Valid range: [0, {len(positions) - 1}]"
                    )
                pos_id = fixed_idx
            else:
                pos_id = random.randint(0, len(positions) - 1)
            selected_pair = positions[pos_id]
            start_pos = [i * self.scale_factor for i in selected_pair["start"]]
            goal_pos = [i * self.scale_factor for i in selected_pair["goal"]]
            self.start_goal_pair_idx = int(pos_id)
        else:
            self.start_goal_pair_idx = -1
        # Initialize terrain
        self.terrains = self.terrain_manager.initialize_terrain(self.system) #initializes the terrain   
        # Initialize vehicle
        self.vehicle_manager.initialize_vehicle(start_pos, goal_pos, self.terrain_manager)
        # Set up moving patches if needed
        if self.terrain_manager.terrain_type in ["deformable", "mixed"]:
            deform_terrains = [t for t in self.terrains if isinstance(t, veh.SCMTerrain)]
            self.vehicle_manager.setup_moving_patches(deform_terrains, self.vehicle_type_lower in ["m113"])

        start_actual = self.vehicle_manager.get_position()
        goal_actual = self.vehicle_manager.goal
        start_xy = (float(start_actual.x), float(start_actual.y))
        goal_xy = (float(goal_actual.x), float(goal_actual.y))
        dist = float(np.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1]))
        spacing = max(float(self.config.get("mpc_ref_spacing", 1.0)), 1e-3)
        n_points = max(self.N_horizon + 1, int(np.ceil(dist / spacing)) + 1)
        self.fixed_ref_xyphi = _build_path(
            start_xy=start_xy,
            goal_xy=goal_xy,
            n_points=n_points,
            path_profile=str(self.config.get("mpc_ref_profile", "straight")),
            sine_amplitude=float(self.config.get("mpc_ref_sine_amplitude", 4.0 * self.scale_factor)),
            sine_cycles=float(self.config.get("mpc_ref_sine_cycles", 1.5)),
            curve_amplitude=float(self.config.get("mpc_ref_curve_amplitude", 6.0 * self.scale_factor)),
            split_ratio=float(self.config.get("mpc_ref_split_ratio", 0.6)),
        )
        self.ref_progress_idx = 0
        self.external_logger.set_fixed_reference(self.fixed_ref_xyphi)
        self.external_logger.begin_run(self.start_goal_pair_idx)

        if self.render:
            self._setup_visualization()
        self._setup_driver()
        self._start_mpc_worker()
        self._init_live_plot()

    def run(self):
        if not hasattr(self, "terrains") or not self.terrains:
            raise ValueError("Simulation not initialized. Call initialize() first.")

        start_time = self.system.GetChTime()
        roll_angles = []
        pitch_angles = []
        result = (None, False, 0, 0)

        try:
            while True:
                time_now = self.system.GetChTime()
                # Handle visualization if enabled
                if self.render:
                    if not self.vis.Run():
                        break
                    if self.last_vis_time == 0 or (time_now - self.last_vis_time) > self.vis_dur:
                        self.vis.BeginScene()
                        self.vis.Render()
                        self.vis.EndScene()
                        self.last_vis_time = time_now

                vehicle_pos = self.vehicle_manager.get_position()
                vector_to_goal = self.vehicle_manager.goal - vehicle_pos
                euler = self.vehicle_manager.get_rotation()
                roll = float(euler.x)
                pitch = float(euler.y)
                yaw = float(euler.z)
                roll_angles.append(np.degrees(abs(roll)))
                pitch_angles.append(np.degrees(abs(pitch)))
                # Handle visualization if enabled
                if self.use_gui:
                    self.driver_inputs = self.driver.GetInputs()
                else:
                    x_state = self._extract_mpc_state()
                    self._solve_mpc(x_state)
                    # apply ZOH as last command
                    self._apply_driver_inputs()
                    self._append_log_row(
                        time_now=time_now,
                        vehicle_pos=vehicle_pos,
                        roll=roll,
                        pitch=pitch,
                        yaw=yaw,
                        speed=self.vehicle_manager.get_speed(),
                        goal_dist=vector_to_goal.Length(),
                    )
                    if len(self.mpc_log_rows) % 100 == 0:
                        self.external_logger.flush_runtime_csv()
                    self._update_live_plot(time_now)
                # Check if vehicle is stuck or reached goal
                current_position = (vehicle_pos.x, vehicle_pos.y, vehicle_pos.z)
                if self.last_position is not None:
                    pos_delta = np.sqrt(
                        (current_position[0] - self.last_position[0]) ** 2
                        + (current_position[1] - self.last_position[1]) ** 2
                        + (current_position[2] - self.last_position[2]) ** 2
                    )
                    if pos_delta < self.stuck_distance:
                        self.stuck_counter += self.step_size
                    else:
                        self.stuck_counter = 0.0
                    if self.stuck_counter >= self.stuck_time:
                        avg_roll = np.mean(roll_angles) if roll_angles else 0.0
                        avg_pitch = np.mean(pitch_angles) if pitch_angles else 0.0
                        result = (time_now - start_time, False, avg_roll, avg_pitch)
                        break
                self.last_position = current_position

                if vector_to_goal.Length() < self.goal_tolerance_m:
                    avg_roll = np.mean(roll_angles) if roll_angles else 0.0
                    avg_pitch = np.mean(pitch_angles) if pitch_angles else 0.0
                    result = (time_now - start_time, True, avg_roll, avg_pitch)
                    break

                if time_now > self.max_time:
                    avg_roll = np.mean(roll_angles) if roll_angles else 0.0
                    avg_pitch = np.mean(pitch_angles) if pitch_angles else 0.0
                    result = (time_now - start_time, False, avg_roll, avg_pitch)
                    break

                for terrain in self.terrains:
                    terrain.Synchronize(time_now)
                    if self.vehicle_type_lower in ["m113"]:
                        self.vehicle_manager.synchronize(time_now, self.driver_inputs)
                    else:
                        self.vehicle_manager.synchronize(time_now, self.driver_inputs, terrain)
                    terrain.Advance(self.step_size)

                self.driver.Advance(self.step_size)
                self.vehicle_manager.advance(self.step_size)
                if self.render:
                    self.vis.Synchronize(time_now, self.driver_inputs)
                    self.vis.Advance(self.step_size)
                self.system.DoStepDynamics(self.step_size)
        finally:
            self._stop_mpc_worker()
            self.external_logger.finalize_run()
            self._close_live_plot()

        if self.render and self.vis is not None:
            self.vis.Quit()

        return result
