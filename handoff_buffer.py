"""Utility for saving, loading, and sampling Handoff Buffer entries.

The Handoff Buffer stores terminal success states from Skill-2 (ApproachAndGrasp)
and provides uniform random sampling to initialize Skill-3 environments from
those success states.

Each entry contains the full robot and object state at the moment of a
successful grasp:
    root_state     (13,)  — robot root state (pos 3 + quat 4 + lin_vel 3 + ang_vel 3)
    joint_pos      (J,)   — full joint positions
    joint_vel      (J,)   — full joint velocities
    object_pos_w   (3,)   — grasped object world position
    object_quat_w  (4,)   — grasped object world orientation
    home_pos_w     (3,)   — home position for the return task
    active_object_idx (1,) — which object type is active
    object_bbox    (3,)   — object bounding box dimensions
    object_mass    (1,)   — object mass
"""

from __future__ import annotations

import torch


class HandoffBuffer:
    """Ring buffer that stores Skill-2 terminal success states for Skill-3 init.

    Target operating range is 200-500 entries.  Once the buffer reaches
    *capacity* the oldest entries are overwritten (FIFO ring semantics).

    Args:
        capacity: Maximum number of entries the buffer can hold.
        device: Torch device for all stored tensors.
        min_ready: Minimum number of entries before :pyattr:`is_ready` returns True.
    """

    REQUIRED_KEYS: tuple[str, ...] = (
        "root_state",
        "joint_pos",
        "joint_vel",
        "object_pos_w",
        "object_quat_w",
        "home_pos_w",
        "active_object_idx",
        "object_bbox",
        "object_mass",
    )

    def __init__(self, capacity: int = 500, device: str = "cpu", min_ready: int = 50):
        self._capacity = capacity
        self._device = torch.device(device)
        self._min_ready = min_ready

        # Storage is lazily allocated on first `add` call so that tensor
        # shapes (e.g. num_joints) are inferred automatically.
        self._storage: dict[str, torch.Tensor] | None = None
        self._size: int = 0  # current number of valid entries
        self._cursor: int = 0  # next write position (ring pointer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, entries: dict[str, torch.Tensor]) -> None:
        """Add a batch of entries to the buffer.

        Args:
            entries: Mapping from field name to tensor of shape ``(batch, ...)``.
                     All tensors in a single call must share the same batch
                     dimension.  At a minimum the keys listed in
                     :pyattr:`REQUIRED_KEYS` must be present; extra keys are
                     stored as-is.
        """
        missing = set(self.REQUIRED_KEYS) - set(entries.keys())
        if missing:
            raise ValueError(f"Missing required keys: {missing}")

        # Determine batch size from the first tensor.
        batch_size = next(iter(entries.values())).shape[0]

        # Lazily allocate storage on the first call.
        if self._storage is None:
            self._storage = {}
            for key, val in entries.items():
                entry_shape = val.shape[1:]
                self._storage[key] = torch.zeros(
                    (self._capacity, *entry_shape), dtype=val.dtype, device=self._device
                )

        for key, val in entries.items():
            if key not in self._storage:
                # Dynamically add a new key that was not in the first batch.
                entry_shape = val.shape[1:]
                self._storage[key] = torch.zeros(
                    (self._capacity, *entry_shape), dtype=val.dtype, device=self._device
                )
            val = val.to(self._device)

            if batch_size <= self._capacity:
                # Number of slots from cursor to end of the ring.
                space_to_end = self._capacity - self._cursor
                if batch_size <= space_to_end:
                    self._storage[key][self._cursor : self._cursor + batch_size] = val
                else:
                    # Wrap around.
                    self._storage[key][self._cursor :] = val[:space_to_end]
                    overflow = batch_size - space_to_end
                    self._storage[key][:overflow] = val[space_to_end:]
            else:
                # More entries than capacity — keep only the last `capacity`.
                val = val[-self._capacity :]
                self._storage[key][:] = val
                batch_size = self._capacity  # adjust for cursor bookkeeping

        # Update bookkeeping (only once, outside the key loop).
        if batch_size >= self._capacity:
            self._cursor = 0
            self._size = self._capacity
        else:
            self._cursor = (self._cursor + batch_size) % self._capacity
            self._size = min(self._size + batch_size, self._capacity)

    def sample(self, n: int) -> dict[str, torch.Tensor]:
        """Sample *n* entries uniformly at random **with replacement**.

        Args:
            n: Number of entries to sample.

        Returns:
            Dictionary mirroring the stored keys, each with shape ``(n, ...)``.

        Raises:
            RuntimeError: If the buffer is empty.
        """
        if self._size == 0 or self._storage is None:
            raise RuntimeError("Cannot sample from an empty HandoffBuffer.")

        indices = torch.randint(0, self._size, (n,), device=self._device)
        return {key: val[indices] for key, val in self._storage.items()}

    def save(self, path: str) -> None:
        """Persist the buffer to a ``.pt`` file.

        The file contains a single dict with metadata and the tensor data so
        that :pymeth:`load` can reconstruct the buffer exactly.

        Args:
            path: Filesystem path (should end in ``.pt``).
        """
        if self._storage is None:
            raise RuntimeError("Cannot save an empty HandoffBuffer.")

        payload: dict[str, object] = {
            "capacity": self._capacity,
            "min_ready": self._min_ready,
            "size": self._size,
            "cursor": self._cursor,
            "data": {k: v[: self._size].cpu() for k, v in self._storage.items()},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> HandoffBuffer:
        """Reconstruct a :class:`HandoffBuffer` from a ``.pt`` file.

        Args:
            path: Path to the file previously written by :pymeth:`save`.
            device: Torch device to place the loaded tensors on.

        Returns:
            A fully populated :class:`HandoffBuffer`.
        """
        payload = torch.load(path, map_location=device, weights_only=True)

        capacity: int = payload["capacity"]
        min_ready: int = payload["min_ready"]
        size: int = payload["size"]
        cursor: int = payload["cursor"]
        data: dict[str, torch.Tensor] = payload["data"]

        buf = cls(capacity=capacity, device=device, min_ready=min_ready)

        # Allocate storage and copy data in.
        buf._storage = {}
        for key, val in data.items():
            entry_shape = val.shape[1:]
            buf._storage[key] = torch.zeros(
                (capacity, *entry_shape), dtype=val.dtype, device=torch.device(device)
            )
            buf._storage[key][:size] = val.to(device)

        buf._size = size
        buf._cursor = cursor
        return buf

    # ------------------------------------------------------------------
    # Dunder / properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of valid entries currently in the buffer."""
        return self._size

    @property
    def is_ready(self) -> bool:
        """``True`` when the buffer contains at least *min_ready* entries."""
        return self._size >= self._min_ready

    def __repr__(self) -> str:
        return (
            f"HandoffBuffer(size={self._size}, capacity={self._capacity}, "
            f"device={self._device}, is_ready={self.is_ready})"
        )
