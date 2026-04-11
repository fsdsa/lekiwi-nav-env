## Working Rules For This Repo

- Do not use temporary workaround-style fixes for Skill-3 behavior.
- Specifically, do not propose or implement solutions like:
  - hard-freezing `Phase A` arm/grip just to suppress bad motion
  - masking policy outputs as a shortcut instead of fixing the learned behavior
  - other evaluation-time clamps that hide BC/RL policy defects
- Prefer structural fixes only:
  - demo design / relabeling
  - observation design
  - phase/sub-phase representation
  - reward design
  - policy decomposition only when it is part of the intended training design, not a patch

This note was added because the user explicitly asked to avoid workaround-style fixes and to preserve that constraint for ongoing work in this repo.
