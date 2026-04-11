# Skill-3 Motion24 Demo Notes

This note explains why the `24D motion-prior` Skill-3 demo was created, what
was kept/removed from the old `36D` demo, and which file is the current one to
use.

## Current recommended file

- `combined_skill3_grip_s07_v5_motion24_tuned_phaseA5x.hdf5`

This is the file to use for BC training right now.

## Source file

- Source demo: `combined_skill3_grip_s07_v5_36d.hdf5`

The `24D` demo was not recollected from scratch. It was converted from the
existing `36D` demo by replaying the saved robot states in sim and rebuilding a
new observation.

## Why this was changed

The old `36D/38D` Skill-3 BC kept mixing too many things together:

- carry in Phase A
- lower in Phase B
- release timing
- retract/rest timing
- object-specific state

That made it hard for BC to learn the full motion pattern cleanly.

The goal of `motion24` is different:

- first learn the overall motion prior
- keep Phase A and Phase B in one sequence
- remove object-dependent ambiguity from the observation
- keep only proprioception, destination relation, EE relation, and simple
  sub-phase flags

This is intended as a cleaner BC prior before later switching back to
object-aware training or S2-start training.

## What was removed from the old 36D obs

Removed on purpose:

- object-relative position
- contact force / holding-like object signals
- object bbox
- object category
- init_pose tail that was packed into the old 36D BC obs

These were removed so the policy focuses on the pose/motion sequence itself,
instead of overfitting object-specific state.

## What was kept in the new 24D obs

The `24D` observation is:

1. arm0
2. arm1
3. arm2
4. arm3
5. arm4
6. grip
7. armvel0
8. armvel1
9. armvel2
10. armvel3
11. armvel4
12. grip_vel
13. base_vx
14. base_vy
15. base_wz
16. ee_to_dest_body_x
17. ee_to_dest_body_y
18. dest_rel_body_x
19. dest_rel_body_y
20. dest_rel_body_z
21. ee_z
22. phase_a_flag
23. release_phase_flag
24. retract_started_flag

In short:

- arm/gripper position
- arm/gripper velocity
- base velocity
- EE to destination relation
- destination to base relation
- EE height
- simple sub-phase flags

## Why EE had to be rebuilt

The original HDF5 did not store EE position directly.

So the conversion script replays each saved step in sim and reconstructs:

- robot root pose
- arm joint state
- wrist pose
- EE pose using `Wrist_Roll_08c_v1 + EE_LOCAL_OFFSET`

That is why the converted `24D` file is more trustworthy than a direct
slice/remap from the old `36D`.

## Flag definitions used for the tuned file

The final tuned file uses:

- `release_phase_flag = 1` when:
  - `phase_a_flag == 0`
  - `ee_to_dest_body_xy <= 0.16`
  - `ee_z <= 0.09`

- `retract_started_flag = 1` when:
  - `release_phase_flag == 1`
  - `grip >= 0.55`

These thresholds were chosen after checking the actual old demo distribution.

## Intermediate files

- `combined_skill3_grip_s07_v5_motion24.hdf5`
  - first converted version
  - not recommended
  - default thresholds were too strict, so release/retract flags stayed zero

- `combined_skill3_grip_s07_v5_motion24_tuned.hdf5`
  - tuned version
  - base motion24 file

- `combined_skill3_grip_s07_v5_motion24_tuned_phaseA5x.hdf5`
  - Phase-A-oversampled version
  - current recommended file
  - built by adding `4` extra short Phase-A-only episodes per source episode
  - resulting effective Phase-A exposure is `5x`

## Quick sanity result for the tuned file

For `combined_skill3_grip_s07_v5_motion24_tuned.hdf5`:

- `obs_dim = 24`
- release flag active in `20/20` episodes
- retract flag active in `18/20` episodes

So the flag signals are actually alive in the tuned file.

## Quick sanity result for the Phase-A-oversampled file

For `combined_skill3_grip_s07_v5_motion24_tuned_phaseA5x.hdf5`:

- episodes: `100`
- full episodes: `20`
- short Phase-A-only episodes: `80`
- total steps: `35118`
- Phase A: `10565` steps (`30.1%`)
- Phase B: `24553` steps (`69.9%`)

This was created because the original tuned motion24 file had only about `7.9%`
Phase-A steps, which made the learned policy drift toward long Phase-B arm
patterns too easily.

## Scripts/code related to this

- obs builder: `skill3_bc_obs.py`
- conversion script: `convert_skill3_36d_to_motion24.py`
- Phase-A oversample builder: `make_skill3_motion24_phasea_oversampled.py`
- recollection path with motion24: `record_teleop.py`

## BC training command

```bash
cd /home/yubin11/IsaacLab/scripts/lekiwi_nav_env
source /home/yubin11/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

python train_diffusion_bc.py \
  --demo_path demos_skill3/combined_skill3_grip_s07_v5_motion24_tuned_phaseA5x.hdf5 \
  --obs_dim 24 \
  --down_dims 256 512 1024 \
  --epochs 300 \
  --save_dir checkpoints/dp_bc_skill3_motion24_v2
```

## If recollection is needed later

The teleop recorder now supports direct `motion24` recording with:

```bash
python record_teleop.py \
  --skill combined \
  --s3_obs_mode motion24 \
  --s3_motion_release_xy 0.16 \
  --s3_motion_release_ee_z 0.09 \
  --s3_motion_retract_grip 0.55 \
  ...
```

## Intended training plan

1. Train BC on this `24D motion-prior` demo.
2. Verify that the policy learns the overall motion sequence better.
3. After that, return to object-aware training / S2-start training if needed.

This file is meant to document that this demo is a motion-prior step, not the
final object-aware Skill-3 representation.
