#!/usr/bin/env python3
"""
Overlay real vs sim motor response curves from replay/tuning outputs.

Supported inputs:
  1) replay_in_sim.py --series_path output JSON
  2) tune_sim_dynamics.py output JSON (uses best_eval.sequences)

Usage:
  python compare_real_sim.py --input calibration/replay_series.json --output_dir calibration/plots
  python compare_real_sim.py --input calibration/tuned_dynamics.json --output_dir calibration/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay real vs sim response curves")
    parser.add_argument("--input", type=str, required=True, help="series json from replay or tuned_dynamics json")
    parser.add_argument("--output_dir", type=str, default="calibration/plots")
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def _load_entries(payload: dict) -> list[dict]:
    entries: list[dict] = []

    # Case A: replay series format
    if isinstance(payload.get("sequences"), list) and payload.get("mode") in ("command", "arm_command"):
        for item in payload["sequences"]:
            if not isinstance(item, dict):
                continue
            if all(k in item for k in ("time_s", "real_delta_rad", "sim_delta_rad")):
                entries.append(
                    {
                        "sequence": str(item.get("sequence", "unknown")),
                        "label": f"{item.get('real_key', '?')} -> {item.get('sim_joint', '?')}",
                        "time_s": np.asarray(item["time_s"], dtype=np.float64),
                        "real": np.asarray(item["real_delta_rad"], dtype=np.float64),
                        "sim": np.asarray(item["sim_delta_rad"], dtype=np.float64),
                        "mae": float(item.get("mae_rad", np.nan)),
                        "rmse": float(item.get("rmse_rad", np.nan)),
                        "maxe": float(item.get("max_err_rad", np.nan)),
                    }
                )

    # Case B: tuned dynamics format
    if not entries and isinstance(payload.get("best_eval"), dict):
        seqs = payload["best_eval"].get("sequences", [])
        if isinstance(seqs, list):
            for seq in seqs:
                if not isinstance(seq, dict):
                    continue
                seq_name = str(seq.get("name", "unknown"))
                for p in seq.get("pairs", []):
                    if not isinstance(p, dict):
                        continue
                    if all(k in p for k in ("time_s", "real_delta_rad", "sim_delta_rad")):
                        entries.append(
                            {
                                "sequence": seq_name,
                                "label": f"{p.get('real_key', '?')} -> {p.get('sim_joint', '?')}",
                                "time_s": np.asarray(p["time_s"], dtype=np.float64),
                                "real": np.asarray(p["real_delta_rad"], dtype=np.float64),
                                "sim": np.asarray(p["sim_delta_rad"], dtype=np.float64),
                                "mae": float(p.get("mae_rad", np.nan)),
                                "rmse": float(p.get("rmse_rad", np.nan)),
                                "maxe": float(p.get("max_err_rad", np.nan)),
                            }
                        )
        arm_tests = payload["best_eval"].get("arm_tests", [])
        if isinstance(arm_tests, list):
            for t in arm_tests:
                if not isinstance(t, dict):
                    continue
                if all(k in t for k in ("time_s", "real_delta_rad", "sim_delta_rad")):
                    entries.append(
                        {
                            "sequence": "arm_sysid",
                            "label": f"{t.get('joint_key', '?')} -> {t.get('sim_joint', '?')}",
                            "time_s": np.asarray(t["time_s"], dtype=np.float64),
                            "real": np.asarray(t["real_delta_rad"], dtype=np.float64),
                            "sim": np.asarray(t["sim_delta_rad"], dtype=np.float64),
                            "mae": float(t.get("mae_rad", np.nan)),
                            "rmse": float(t.get("rmse_rad", np.nan)),
                            "maxe": float(t.get("max_err_rad", np.nan)),
                        }
                    )

    return entries


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    entries = _load_entries(payload)
    if not entries:
        raise RuntimeError(
            "No plot entries found in input JSON. "
            "Use replay_in_sim.py --series_path or tune_sim_dynamics.py output."
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required. pip install matplotlib") from exc

    by_seq: dict[str, list[dict]] = {}
    for e in entries:
        by_seq.setdefault(e["sequence"], []).append(e)

    summary = []

    for seq_name, seq_entries in sorted(by_seq.items()):
        n = len(seq_entries)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), squeeze=False)

        for i, e in enumerate(seq_entries):
            ax = axes[i, 0]
            ax.plot(e["time_s"], e["real"], label="real", linewidth=1.8)
            ax.plot(e["time_s"], e["sim"], label="sim", linewidth=1.4)
            ax.set_title(f"{seq_name} | {e['label']}")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("delta (rad)")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            ax.text(
                0.01,
                0.95,
                f"MAE={e['mae']:.4f} rad | RMSE={e['rmse']:.4f} rad | MAX={e['maxe']:.4f} rad",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )
            summary.append(
                {
                    "sequence": seq_name,
                    "label": e["label"],
                    "mae_rad": e["mae"],
                    "rmse_rad": e["rmse"],
                    "max_err_rad": e["maxe"],
                }
            )

        fig.tight_layout()
        out_png = out_dir / f"overlay_{seq_name}.png"
        fig.savefig(out_png, dpi=args.dpi)
        plt.close(fig)
        print(f"saved: {out_png}")

    # summary CSV
    try:
        import csv

        out_csv = out_dir / "overlay_summary.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sequence", "label", "mae_rad", "rmse_rad", "max_err_rad"],
            )
            writer.writeheader()
            writer.writerows(summary)
        print(f"saved: {out_csv}")
    except Exception as exc:  # noqa: BLE001
        print(f"warning: failed to write summary csv: {exc}")


if __name__ == "__main__":
    main()
