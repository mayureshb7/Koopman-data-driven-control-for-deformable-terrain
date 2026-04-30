import os
from dataclasses import dataclass

import numpy as np

from verti_bench.systems.MPC.MPC_SIM_Run import MPCSim


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi

# Similar to the MATLAb code sued for edmd where insetad of the x1, x2 states it is building on the error sates
def build_error_state(
    x: float,
    y: float,
    yaw: float,
    speed: float,
    delta_est: float,
    x_ref: float,
    y_ref: float,
    yaw_ref: float,
    speed_ref: float,
    delta_ref: float,
) -> np.ndarray:
    ex = float(x) - float(x_ref)
    ey = float(y) - float(y_ref)
    c = np.cos(float(yaw_ref))
    s = np.sin(float(yaw_ref))
    ex_local = c * ex + s * ey
    ey_local = -s * ex + c * ey
    eyaw = wrap_angle(float(yaw) - float(yaw_ref))
    espeed = float(speed) - float(speed_ref)
    edelta = float(delta_est) - float(delta_ref)
    return np.array([ex_local, ey_local, eyaw, espeed, edelta], dtype=float)

# generates the combinations of the monomials depending on the degree used and the states
def _current_combinations(total: int, n_var: int):
    if n_var == 1:
        return [[total]]
    out = []
    for i in range(total + 1):
        for tail in _current_combinations(total - i, n_var - 1):
            out.append([i] + tail)
    return out

# builds monomials based on the current combinations
def generate_monomial_exponents(n_var: int, degree: int) -> np.ndarray:
    rows = []
    for total in range(2, int(degree) + 1):
        rows.extend(_current_combinations(total, n_var))
    return np.asarray(rows, dtype=int)

#teh main lift function used to lift the error residual states to the higher z/psi space
def lift_edmd(e: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    if exponents.size == 0:
        mono = np.zeros((e.shape[0], 0), dtype=float)
    else:
        mono = np.prod(np.power(e[:, None, :], exponents[None, :, :]), axis=2)
    return np.hstack([e, mono])

# LQR with the recatti recursion and solved using the np.linalg function where the matrices are from the lifted states so that it remains in same space
def finite_horizon_first_gain(
    A: np.ndarray,
    B: np.ndarray,
    Q_lift: np.ndarray,
    R: np.ndarray,
    horizon: int,
    terminal_weight: float,
) -> np.ndarray:
    p = float(terminal_weight) * Q_lift
    gains_rev = []
    for _ in range(max(1, int(horizon))):
        # Defining ricatti recursion
        s = R + B.T @ p @ B
        k = np.linalg.solve(s, B.T @ p @ A) # solving the ricatti recursion with lqr
        gains_rev.append(k) # lookup
        p = Q_lift + A.T @ p @ (A - B @ k) # updating the ricatti recursion
    return k


@dataclass
class ResidualEDMDControllerABC:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    exponents: np.ndarray
    du_limits: np.ndarray
    k0: np.ndarray

    # Loads the model from training file and sets up the Q R and Qlift with ABC matrices to compute the k0 gain
    @classmethod
    def from_npz(
        cls,
        model_path: str,
        q_vec: np.ndarray,
        r_vec: np.ndarray,
        du_limits: np.ndarray,
        pred_horizon: int = 400,
        terminal_weight: float = 5.0,
    ) -> "ResidualEDMDControllerABC":
        with np.load(model_path, allow_pickle=False) as payload:
            A = np.asarray(payload["A"], dtype=float)
            B = np.asarray(payload["B"], dtype=float)
            C = np.asarray(payload["C"], dtype=float)
            basis_exponents = np.asarray(payload["basis_exponents"], dtype=int) if "basis_exponents" in payload.files else None
            degree = int(payload["basis_degree"][0]) if "basis_degree" in payload.files else None

        n_state = C.shape[0]
        if basis_exponents is not None and basis_exponents.ndim == 2 and basis_exponents.shape[1] == n_state:
            exponents = basis_exponents
        else:
            # Default to degree-2 monomial lift when metadata is absent.
            inferred_degree = 2 if degree is None else int(degree)
            exponents = generate_monomial_exponents(n_var=n_state, degree=inferred_degree)

        n_lift_expected = n_state + exponents.shape[0]
        if A.shape[0] == n_lift_expected + 1:
            # Legacy models may include a constant basis term; represent it as x^0.
            exponents = np.vstack([np.zeros((1, n_state), dtype=int), exponents])
            n_lift_expected = n_state + exponents.shape[0]

        if A.shape != (n_lift_expected, n_lift_expected):
            raise ValueError(
                f"Model A shape {A.shape} incompatible with lift size {n_lift_expected}."
            )
        if B.shape[0] != n_lift_expected:
            raise ValueError(f"Model B shape {B.shape} incompatible with lift size {n_lift_expected}.")
        if C.shape[1] != n_lift_expected:
            raise ValueError(f"Model C shape {C.shape} incompatible with lift size {n_lift_expected}.")

        Q = np.diag(np.asarray(q_vec, dtype=float))
        R = np.diag(np.asarray(r_vec, dtype=float))
        Q_lift = C.T @ Q @ C
        k0 = finite_horizon_first_gain(
            A=A,
            B=B,
            Q_lift=Q_lift,
            R=R,
            horizon=pred_horizon,
            terminal_weight=terminal_weight,
        )
        return cls(
            A=A,
            B=B,
            C=C,
            exponents=exponents,
            du_limits=np.asarray(du_limits, dtype=float),
            k0=k0,
        )

    # Computes the control input du form the error states build earlier that are lifted and 
    # the ko gain form the finite hortizon first gain is used to compute the control input
    def compute_delta_u(self, error_state: np.ndarray) -> np.ndarray:
        e = np.asarray(error_state, dtype=float).reshape(1, -1)
        z = lift_edmd(e, self.exponents)
        du = -(z @ self.k0.T)
        du = np.clip(du, -self.du_limits.reshape(1, -1), self.du_limits.reshape(1, -1))
        return du[0]

    def compute_delta_u_from_tracking(
        self,
        x: float,
        y: float,
        yaw: float,
        speed: float,
        delta_est: float,
        x_ref: float,
        y_ref: float,
        yaw_ref: float,
        speed_ref: float,
        delta_ref: float,
    ) -> np.ndarray:
        e = build_error_state(
            x=x,
            y=y,
            yaw=yaw,
            speed=speed,
            delta_est=delta_est,
            x_ref=x_ref,
            y_ref=y_ref,
            yaw_ref=yaw_ref,
            speed_ref=speed_ref,
            delta_ref=delta_ref,
        )
        return self.compute_delta_u(e)


class MPCKoopmanSim2(MPCSim):
    # Computes the u final which is absed on th follwing formula
    # u_final = u_mpc + rk_gain * delta_u_edmd the gain is adusted to reduce the controller error

    def __init__(self, config):
        cfg = dict(config)
        if str(cfg.get("system", "")).lower() == "mpc_koopman2":
            cfg["system"] = "mpc"
        super().__init__(cfg)
        self.system_type = "mpc_koopman2"

        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        model_path_cfg = config.get("rk_model_path", "logs/koopman_models/s1_clean_snow3/residual_koopman_model.npz")
        self.rk_model_path = (
            model_path_cfg if os.path.isabs(model_path_cfg) else os.path.join(self.repo_root, model_path_cfg)
        )

        self.rk_enable = bool(config.get("rk_enable", True))
        self.rk_gain = float(config.get("rk_gain", 1.0))
        self.rk_pred_horizon = int(config.get("rk_pred_horizon", 12))
        self.rk_terminal_weight = float(config.get("rk_terminal_weight", 5.0))
        self.rk_q_vec = np.asarray(config.get("rk_q_weights", [1.0, 1.0, 0.8, 0.3, 0.3]), dtype=float)
        self.rk_r_vec = np.asarray(config.get("rk_r_weights", [0.2, 0.2]), dtype=float)
        self.rk_du_limits = np.asarray(config.get("rk_du_limits", [3.0, 2.0]), dtype=float)

        self.rk_warned = False
        self.rk_loaded = False
        self.rk_controller = None
        self.rk_last_delta_u = np.zeros((2,), dtype=float)
        self.rk_last_error_state = np.zeros((5,), dtype=float)

        if self.rk_enable:
            self._load_residual_koopman_model()

    def _load_residual_koopman_model(self):
        try:
            self.rk_controller = ResidualEDMDControllerABC.from_npz(
                model_path=self.rk_model_path,
                q_vec=self.rk_q_vec,
                r_vec=self.rk_r_vec,
                du_limits=self.rk_du_limits,
                pred_horizon=self.rk_pred_horizon,
                terminal_weight=self.rk_terminal_weight,
            )
            self.rk_loaded = True
            print(
                "[MPC_KOOPMAN2] Loaded A/B/C model: {} | pred_horizon={} | terminal_weight={}".format(
                    self.rk_model_path, self.rk_pred_horizon, self.rk_terminal_weight
                )
            )
        except Exception as exc:
            self.rk_loaded = False
            self.rk_controller = None
            print(f"[MPC_KOOPMAN2] Failed to load model '{self.rk_model_path}': {exc}")
            print("[MPC_KOOPMAN2] Falling back to baseline MPC commands.")

    def _augment_control_with_residual(self, x_state, ax_cmd, ddelta_cmd):
        # Flag if the Koopman is to be used or not and is a passthrough to just give the baseline MPC commands
        if (not self.rk_enable) or (not self.rk_loaded) or (self.rk_controller is None):
            return float(ax_cmd), float(ddelta_cmd)
        if self.current_yref0 is None or len(self.current_yref0) < 7:
            return float(ax_cmd), float(ddelta_cmd)

        try:
            x = float(x_state[0])
            y = float(x_state[1])
            yaw = float(x_state[2])
            delta_est = float(x_state[3])
            speed = float(x_state[4])
            x_ref = float(self.current_yref0[0])
            y_ref = float(self.current_yref0[1])
            yaw_ref = float(self.current_yref0[2])
            delta_ref = float(self.current_yref0[3])
            speed_ref = float(self.current_yref0[4])

            if not np.all(np.isfinite([x_ref, y_ref, yaw_ref, delta_ref, speed_ref])):
                return float(ax_cmd), float(ddelta_cmd)

            delta_u = np.asarray(
                self.rk_controller.compute_delta_u_from_tracking(
                    x=x,
                    y=y,
                    yaw=yaw,
                    speed=speed,
                    delta_est=delta_est,
                    x_ref=x_ref,
                    y_ref=y_ref,
                    yaw_ref=yaw_ref,
                    speed_ref=speed_ref,
                    delta_ref=delta_ref,
                ),
                dtype=float,
            )
            if delta_u.shape[0] != 2 or not np.all(np.isfinite(delta_u)):
                return float(ax_cmd), float(ddelta_cmd)

            self.rk_last_delta_u = delta_u.copy()
            self.rk_last_error_state = np.array(
                [x - x_ref, y - y_ref, yaw - yaw_ref, speed - speed_ref, delta_est - delta_ref],
                dtype=float,
            )
            # add the residual compensation to the baseline MPC commands
            ax_final = float(ax_cmd) + self.rk_gain * float(delta_u[0])
            ddelta_final = float(ddelta_cmd) + self.rk_gain * float(delta_u[1])
            return ax_final, ddelta_final
        except Exception as exc:
            if not self.rk_warned:
                print(f"[MPC_KOOPMAN2] residual compensation failed once: {exc}")
                self.rk_warned = True
            return float(ax_cmd), float(ddelta_cmd)

    def _append_log_row(self, time_now, vehicle_pos, roll, pitch, yaw, speed, goal_dist):
        super()._append_log_row(
            time_now=time_now,
            vehicle_pos=vehicle_pos,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=speed,
            goal_dist=goal_dist,
        )
        if not hasattr(self, "mpc_log_rows") or len(self.mpc_log_rows) == 0:
            return
        row = self.mpc_log_rows[-1]
        row["rk2_enabled"] = bool(self.rk_enable)
        row["rk2_loaded"] = bool(self.rk_loaded)
        row["rk2_gain"] = float(self.rk_gain)
        row["rk2_pred_horizon"] = int(self.rk_pred_horizon)
        row["rk2_terminal_weight"] = float(self.rk_terminal_weight)
        row["rk2_model_path"] = str(self.rk_model_path)
        row["rk2_dax_cmd"] = float(self.rk_last_delta_u[0])
        row["rk2_dddelta_cmd"] = float(self.rk_last_delta_u[1])