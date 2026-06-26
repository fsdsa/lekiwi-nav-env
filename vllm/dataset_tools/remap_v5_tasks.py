"""
Rewrite v5 dataset's tasks.parquet with arm-state-explicit instructions.

Mapping:
  navigate forward         → navigate forward with arm tucked
  navigate backward        → navigate backward with arm tucked
  navigate turn left       → navigate turn left with arm tucked
  navigate turn right      → navigate turn right with arm tucked
  navigate strafe left     → navigate strafe left with arm tucked
  navigate strafe right    → navigate strafe right with arm tucked
  carry forward            → carry forward holding the object
  carry backward           → carry backward holding the object
  carry left               → carry left holding the object
  carry right              → carry right holding the object
  carry turn left          → carry turn left holding the object
  carry turn right         → carry turn right holding the object
  approach and lift the medicine bottle → (unchanged)
  place/approach and place → (unchanged)

Motivation: text token "tucked"/"holding" becomes learned arm-state signal,
decoupling arm control from image shortcut.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


NAV_SUFFIX = " with arm tucked"
CARRY_SUFFIX = " holding the object"

NAV_TASKS = {
    "navigate forward", "navigate backward",
    "navigate left", "navigate right",
    "navigate turn left", "navigate turn right",
    "navigate strafe left", "navigate strafe right",
}
CARRY_TASKS = {
    "carry forward", "carry backward",
    "carry left", "carry right",
    "carry turn left", "carry turn right",
}


def remap_task(original: str) -> str:
    """Apply arm-state suffix to nav/carry tasks."""
    if original in NAV_TASKS:
        return original + NAV_SUFFIX
    if original in CARRY_TASKS:
        return original + CARRY_SUFFIX
    return original  # approach_and_lift / other → unchanged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="v5 dataset root")
    ap.add_argument("--backup", action="store_true", help="save tasks.parquet.bak before overwrite")
    args = ap.parse_args()

    tasks_path = Path(args.path) / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        raise FileNotFoundError(tasks_path)

    df = pd.read_parquet(tasks_path)
    print(f"Loaded {len(df)} task labels from {tasks_path}")
    print(f"Original:\n{df.to_string()}\n")

    # tasks.parquet has index=task_name, value=task_index (per earlier inspection)
    # → need to reset index, rename values, re-set
    df_reset = df.reset_index()
    if "task" in df_reset.columns:
        name_col = "task"
    elif df_reset.columns[0] != "task_index":
        name_col = df_reset.columns[0]
    else:
        name_col = "task"
    df_reset[name_col] = df_reset[name_col].map(remap_task)

    print(f"Remapped:\n{df_reset.to_string()}\n")

    if args.backup:
        bak = tasks_path.with_suffix(".parquet.bak")
        shutil.copy(tasks_path, bak)
        print(f"Backup: {bak}")

    # Write back with task name as index (matches original structure)
    df_reset.set_index(name_col).to_parquet(tasks_path)

    # Verify readback
    v = pd.read_parquet(tasks_path)
    print(f"\nVerify readback (first 3):\n{v.head(3).to_string()}")
    print(f"\n✓ remapped {tasks_path}")


if __name__ == "__main__":
    main()
