import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

@dataclass
class ResidualDataset:
    # e[k] = residual tracking error state (5D), du[k] = residual control (2D)
    e: np.ndarray
    du: np.ndarray
    dt: float
    files: List[str] # extract log files


def parse_csv_vector(text: str, expected_len: int, arg_name: str) -> np.ndarray:
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(vals) != expected_len:
        raise ValueError(f"{arg_name} expects {expected_len} values, got {len(vals)}")
    return np.asarray(vals, dtype=float)


def safe_float(v: str) -> float:
    return 0.0 if v == "" or v.lower() == "nan" else float(v)


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


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
    delta_est_ref: float,
) -> np.ndarray:
    ex = float(x) - float(x_ref)
    ey = float(y) - float(y_ref)
    c = np.cos(float(yaw_ref))
    s = np.sin(float(yaw_ref))
    ex_local = c * ex + s * ey
    ey_local = -s * ex + c * ey
    eyaw = wrap_angle(float(yaw) - float(yaw_ref))
    espeed = float(speed) - float(speed_ref)
    edelta = float(delta_est) - float(delta_est_ref)
    return np.array([ex_local, ey_local, eyaw, espeed, edelta], dtype=float)


def select_world_files(log_glob: str, world_id: int) -> List[str]:
    key = f"world{world_id}_"
    files = [fp for fp in sorted(glob.glob(log_glob)) if key in os.path.basename(fp)]
    if not files:
        raise FileNotFoundError(f"No files for world_id={world_id} with --log_glob='{log_glob}'")
    return files

# build dataset from log files
def build_dataset(files: Sequence[str]) -> ResidualDataset:
    required = {
        "time",
        "x",
        "y",
        "yaw",
        "speed",
        "delta_est",
        "x_ref",
        "y_ref",
        "yaw_ref",
        "speed_ref",
        "delta_est_ref",
        "ax_cmd",
        "ddelta_cmd",
        "ax_cmd_ref",
        "ddelta_cmd_ref",
    }
    e_rows, du_rows, dt_vals = [], [], []

    for fp in files:
        with open(fp, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            missing = [c for c in required if c not in reader.fieldnames]
            if missing:
                raise ValueError(f"Missing columns in {fp}: {missing}")

            prev_t = None
            for row in reader:
                t = safe_float(row["time"])
                if prev_t is not None:
                    dt = t - prev_t
                    if np.isfinite(dt) and dt > 0.0:
                        dt_vals.append(dt)
                prev_t = t

                e_rows.append(
                    build_error_state(
                        row["x"],
                        row["y"],
                        row["yaw"],
                        row["speed"],
                        row["delta_est"],
                        row["x_ref"],
                        row["y_ref"],
                        row["yaw_ref"],
                        row["speed_ref"],
                        row["delta_est_ref"],
                    ).tolist()
                )
                du_rows.append(
                    [
                        safe_float(row["ax_cmd"]) - safe_float(row["ax_cmd_ref"]),
                        safe_float(row["ddelta_cmd"]) - safe_float(row["ddelta_cmd_ref"]),
                    ]
                )

    if len(e_rows) < 3:
        raise ValueError("Not enough rows to fit EDMD residual model.")
    dt = float(np.median(np.asarray(dt_vals, dtype=float))) if dt_vals else 0.005
    return ResidualDataset(
        e=np.asarray(e_rows, dtype=float), #this is where error states are stored
        du=np.asarray(du_rows, dtype=float), #this is where control residuals are stored
        dt=dt,
        files=list(files),
    )

# generates the combinations of the monomials depending on the degree used and the states
def _current_combinations(total: int, n_var: int) -> List[List[int]]:
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

# evaluates monomial features for a batch of states
def evaluate_monomials_batch(x: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    # x shape: [m, n_var], returns [m, n_monomials]
    if exponents.size == 0:
        return np.zeros((x.shape[0], 0), dtype=float)
    return np.prod(np.power(x[:, None, :], exponents[None, :, :]), axis=2)


def lift_edmd(e: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    # z = [e; 2 to deg]
    mono = evaluate_monomials_batch(e, exponents)
    return np.hstack([e, mono])


def get_edmd(X1: np.ndarray, X2: np.ndarray, U: np.ndarray, exponents: np.ndarray, ridge: float):
    # X1, X2: [n_x, m], U: [n_u, m]
    Z1 = lift_edmd(X1.T, exponents).T  # [n_z, m]
    Z2 = lift_edmd(X2.T, exponents).T  # [n_z, m]
    PSI = np.vstack([Z1, U])  # [n_z+n_u, m]

    reg = PSI @ PSI.T + float(ridge) * np.eye(PSI.shape[0], dtype=float)
    G = Z2 @ PSI.T @ np.linalg.inv(reg)
    n_z = Z1.shape[0]
    A = G[:, :n_z]
    B = G[:, n_z:]

    # Decoder C: x_hat = C z.
    n_x = X1.shape[0]
    C = np.hstack([np.eye(n_x, dtype=float), np.zeros((n_x, n_z - n_x), dtype=float)])
    return A, B, C, Z1, Z2

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

#Evaluation after training - one-step MPC
def solve_du_one_step(
    z: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, Q: np.ndarray, R: np.ndarray, du_limits: np.ndarray
) -> np.ndarray:
    # One-step MPC
    F = C @ B
    G = C @ (A @ z.T)
    du = -np.linalg.solve(F.T @ Q @ F + R, F.T @ Q @ G).T
    return np.clip(du, -du_limits.reshape(1, -1), du_limits.reshape(1, -1))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train residual EDMD model from MPC logs and save A/B/C."
    )
    p.add_argument("--log_glob", type=str, default="logs/mpc/*.csv")
    p.add_argument("--world_id", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="logs/koopman_models/world1_residual")
    p.add_argument("--basis_deg", type=int, default=2, help="Monomial max degree for EDMD lift.")
    p.add_argument("--ridge", type=float, default=1e-5, help="EDMD Tikhonov regularization.")
    # additional analysis
    p.add_argument("--q_weights", type=str, default="1.0,1.0,0.8,0.3,0.3")
    p.add_argument("--r_weights", type=str, default="0.2,0.2")
    p.add_argument("--du_limits", type=str, default="3.0,2.0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    q_vec = parse_csv_vector(args.q_weights, 5, "--q_weights")
    r_vec = parse_csv_vector(args.r_weights, 2, "--r_weights")
    du_limits = parse_csv_vector(args.du_limits, 2, "--du_limits")
    files = select_world_files(args.log_glob, args.world_id)
    ds = build_dataset(files)

    # Build time-shifted snapshots similar to the MATLAB code where this is series of the dataset combined and is evaluated for the EDMD model
    # X1 = e_k^T, X2 = e_{k+1}^T, U = du_k^T
    e_k = ds.e[:-1]
    e_next = ds.e[1:]
    du_k = ds.du[:-1]
    X1 = e_k.T
    X2 = e_next.T
    U = du_k.T

    exponents = generate_monomial_exponents(n_var=e_k.shape[1], degree=args.basis_deg)
    A, B, C, Z1, Z2 = get_edmd(X1, X2, U, exponents, ridge=args.ridge) #tthe EDMD is used with the data X1, X2, and U is used similar to the MATLAB code

    z_next_pred = (A @ Z1 + B @ U).T
    e_next_pred = (C @ z_next_pred.T).T
    fit_rmse_z = rmse(z_next_pred, Z2.T)
    fit_rmse_e = rmse(e_next_pred, e_next)

    Q = np.diag(q_vec)
    R = np.diag(r_vec)
    z_k = Z1.T
    du_cmd = solve_du_one_step(z_k, A, B, C, Q, R, du_limits) # one-step MPC
    e_zero = (C @ (A @ z_k.T)).T
    e_ctrl = (C @ (A @ z_k.T + B @ du_cmd.T)).T

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "residual_koopman_model.npz")
    np.savez_compressed(
        model_path,
        A=A,
        B=B,
        C=C,
        basis_type=np.array(["monomials"]),
        basis_degree=np.array([int(args.basis_deg)], dtype=int),
        basis_exponents=exponents.astype(int),
        dt=np.array([ds.dt], dtype=float),
        world_id=np.array([args.world_id], dtype=int),
        error_state_names=np.array(["ex_local", "ey_local", "eyaw", "espeed", "edelta"]),
        control_residual_names=np.array(["dax_cmd", "dddelta_cmd"]),
    )

    metrics = {
        "world_id": int(args.world_id),
        "n_files": int(len(ds.files)),
        "n_samples": int(ds.e.shape[0]),
        "dt_median": float(ds.dt),
        "basis_degree": int(args.basis_deg),
        "n_lifted_states": int(A.shape[0]),
        "fit_rmse_lifted": float(fit_rmse_z),
        "fit_rmse_error_state": float(fit_rmse_e),
        "mean_error_norm_no_comp": float(np.mean(np.linalg.norm(e_zero, axis=1))),
        "mean_error_norm_with_comp": float(np.mean(np.linalg.norm(e_ctrl, axis=1))),
        "source_files": ds.files,
        "model_path": model_path,
    }
    metrics_path = os.path.join(args.out_dir, "residual_koopman_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print("Residual EDMD training complete.")
    print(f"World ID: {args.world_id} | Files: {len(ds.files)} | Samples: {ds.e.shape[0]}")
    print(f"Lifted dim: {A.shape[0]} | basis_deg={args.basis_deg}")
    print(f"Model:   {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Fit RMSE (lifted):      {fit_rmse_z:.6f}")
    print(f"Fit RMSE (error state): {fit_rmse_e:.6f}")


if __name__ == "__main__":
    main()
