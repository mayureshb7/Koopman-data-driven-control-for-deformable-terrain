import argparse

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect residual_koopman_model.npz contents."
    )
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to residual_koopman_model.npz",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    keys = ["A", "B", "C", "q_vec", "r_vec", "du_limits"]

    with np.load(args.model_path, allow_pickle=False) as payload:
        print(f"Model: {args.model_path}")
        print(f"Available keys: {sorted(payload.files)}")
        print("-" * 72)
        for k in keys:
            if k not in payload.files:
                print(f"{k}: <missing>")
                print("-" * 72)
                continue
            v = payload[k]
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")
            print(v)
            print("-" * 72)


if __name__ == "__main__":
    main()
