"""Coordinate-based rotation of pretrained features ("positional disambiguation")."""

from __future__ import annotations

import math
from enum import Enum

import torch


class DisambiguationMode(str, Enum):
    """Spatial coordinate(s) driving the rotation angle of each feature pair.

    X: column coordinate, Y: row coordinate, XY: pairs alternate Y, X, Y, X, ...
    Y reproduces the rotation used to train the original pretrained-feature models.
    """

    X = "x"
    Y = "y"
    XY = "xy"


def coerce_disambiguation_mode(
    value: bool | str | DisambiguationMode | None,
) -> DisambiguationMode | None:
    """Maps legacy and user-facing values to a mode.

    True -> Y (legacy behaviour), False/None -> None (no rotation),
    "x"/"y"/"xy" (case-insensitive) -> the corresponding mode.
    """
    if value is None or value is False:
        return None
    if value is True:
        return DisambiguationMode.Y
    if isinstance(value, DisambiguationMode):
        return value
    if isinstance(value, str):
        return DisambiguationMode(value.lower())
    raise TypeError(f"Cannot interpret {value!r} as a DisambiguationMode.")


def rotate_features(
    features: torch.Tensor,
    coords: torch.Tensor,
    image_shape: tuple,
    n_rot_dims: int | None = None,
    skip_first: int = 0,
    mode: DisambiguationMode | str = DisambiguationMode.Y,
) -> torch.Tensor:
    """Applies a RoPE-style rotation to each feature vector based on the object's centroid.

    Args:
        features: (n_objects, hidden_state_size) tensor of features.
        coords: (n_objects, 1 + ndim) tensor of coordinates, (t, y, x, ...).
        image_shape: shape of the input image, whose last two entries are (height, width).
        n_rot_dims: Number of feature dimensions to apply rotation to (must be even). If None, rotate all.
        skip_first: Number of dimensions to skip at the beginning of the feature vector. No effect if 0.
        mode: Coordinate(s) driving the angle of each feature pair, see `DisambiguationMode`.

    Returns:
        Rotated features: (n_objects, hidden_state_size)
    """
    mode = coerce_disambiguation_mode(mode)
    if mode is None:
        raise ValueError("mode must be one of DisambiguationMode, got None/False.")

    N, D = features.shape
    assert skip_first < n_rot_dims, "skip_first must be less than n_rot_dims."
    if n_rot_dims is None:
        n_rot_dims = D
    if skip_first != 0:
        n_rot_dims = n_rot_dims - skip_first
    assert n_rot_dims % 2 == 0, "n_rot_dims must be even for RoPE."
    assert n_rot_dims <= D, "n_rot_dims cannot exceed feature dimension."

    # One angle per spatial axis, from coordinates normalized to [0, 1]
    H, W = image_shape[-2], image_shape[-1]
    angle_y = 2 * math.pi * (coords[:, 1] / H)
    angle_x = 2 * math.pi * (coords[:, 2] / W)
    n_pairs = n_rot_dims // 2
    if mode is DisambiguationMode.Y:
        angles = angle_y[:, None].expand(N, n_pairs)
    elif mode is DisambiguationMode.X:
        angles = angle_x[:, None].expand(N, n_pairs)
    else:
        angles = torch.stack([angle_y, angle_x], dim=1).repeat(
            1, math.ceil(n_pairs / 2)
        )[:, :n_pairs]
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    features_rot = features[:, skip_first : n_rot_dims + skip_first].view(N, -1, 2)
    x_feat, y_feat = features_rot[..., 0], features_rot[..., 1]
    x_rot = x_feat * cos - y_feat * sin
    y_rot = x_feat * sin + y_feat * cos
    rotated_part = torch.stack([x_rot, y_rot], dim=-1).reshape(N, n_rot_dims)
    if n_rot_dims < D:
        rotated = torch.cat([rotated_part, features[:, n_rot_dims:]], dim=1)
    else:
        rotated = rotated_part
    return rotated  # (n_objects, d)


def rotate_window_features(
    windows: list[dict],
    image_shape: tuple,
    mode: DisambiguationMode | str,
    key: str = "pretrained_features",
) -> list[dict]:
    """Rotates the pretrained features of inference windows in place.

    Window coords are (y, x) without time, so the timepoints are prepended to match
    the (t, y, x) layout `rotate_features` expects. Unlike training, no random
    coordinate shift is applied.

    Raises if a window has no pretrained features under `key` (the window key differs
    between trackastra versions), so a wrong key cannot silently skip the rotation.
    """
    n_rotated = 0
    for i, w in enumerate(windows):
        if w.get(key) is None:
            raise KeyError(
                f"Window {i} has no pretrained features under '{key}' (keys: {list(w)})."
            )
        pt_feats = w[key]
        if len(pt_feats) == 0:
            continue
        coords = torch.as_tensor(w["coords"]).float()
        timepoints = torch.as_tensor(w["timepoints"]).float()
        coords = torch.cat((timepoints[:, None], coords), dim=1)
        pt_feats = torch.as_tensor(pt_feats).float()
        w[key] = rotate_features(
            pt_feats, coords, image_shape, n_rot_dims=pt_feats.shape[1], mode=mode
        )
        n_rotated += 1
    if windows and n_rotated == 0:
        raise ValueError("Rotation requested but no window had pretrained features.")
    return windows
