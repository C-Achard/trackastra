import pytest
import torch
from trackastra.data.disambiguation import (
    DisambiguationMode,
    coerce_disambiguation_mode,
    rotate_features,
    rotate_window_features,
)

IMAGE_SHAPE = (10, 300, 500)


# Verbatim copy of CTCData.rotate_features at a238b2c (before DisambiguationMode).
def _legacy_rotate_features(
    features: torch.Tensor,
    coords: torch.Tensor,
    image_shape: tuple,
    n_rot_dims: int | None = None,
    skip_first: int = 0,
) -> torch.Tensor:
    """Applies a RoPE-style rotation to each feature vector based on the object's centroid.

    Args:
        features: (n_objects, hidden_state_size) tensor of features.
        coords: (n_objects, 2) tensor of coordinates.
        image_shape: (time, height, width) shape of the input image.
        n_rot_dims: Number of feature dimensions to apply rotation to (must be even). If None, rotate all.
        skip_first: Number of dimensions to skip at the beginning of the feature vector. No effect if 0.

    Returns:
        Rotated features: (n_objects, hidden_state_size)
    """
    import math

    N, D = features.shape
    assert skip_first < n_rot_dims, "skip_first must be less than n_rot_dims."
    if n_rot_dims is None:
        n_rot_dims = D
    if skip_first != 0:
        n_rot_dims = n_rot_dims - skip_first
    assert n_rot_dims % 2 == 0, "n_rot_dims must be even for RoPE."
    assert n_rot_dims <= D, "n_rot_dims cannot exceed feature dimension."

    # Normalize x and y to [0, 1]
    H, W = image_shape[-2], image_shape[-1]
    x_norm = coords[:, 1] / H
    y_norm = coords[:, 2] / W
    # Compute two angles for x and y
    angle_x = 2 * math.pi * x_norm
    angle_y = 2 * math.pi * y_norm
    # Interleave angles for each feature pair
    angles = torch.stack([angle_x, angle_y], dim=1).repeat(1, n_rot_dims // 2)
    angles = angles.view(N, n_rot_dims)
    # Prepare cos/sin
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    # Interleave features for rotation
    # try:
    features_rot = features[:, skip_first : n_rot_dims + skip_first].view(N, -1, 2)
    # except Exception:
    #     breakpoint()
    x_feat, y_feat = features_rot[..., 0], features_rot[..., 1]
    x_rot = x_feat * cos[:, ::2] - y_feat * sin[:, ::2]
    y_rot = x_feat * sin[:, ::2] + y_feat * cos[:, ::2]
    rotated_part = torch.stack([x_rot, y_rot], dim=-1).reshape(N, n_rot_dims)
    if n_rot_dims < D:
        rotated = torch.cat([rotated_part, features[:, n_rot_dims:]], dim=1)
    else:
        rotated = rotated_part
    return rotated  # (n_objects, d)


@pytest.fixture
def inputs():
    g = torch.Generator().manual_seed(0)
    n, d = 17, 64
    features = torch.randn(n, d, generator=g)
    t = torch.randint(0, IMAGE_SHAPE[0], (n, 1), generator=g).float()
    y = torch.rand(n, 1, generator=g) * IMAGE_SHAPE[1]
    x = torch.rand(n, 1, generator=g) * IMAGE_SHAPE[2]
    coords = torch.cat([t, y, x], dim=1)
    return features, coords


def rot(features, coords, mode, image_shape=IMAGE_SHAPE):
    return rotate_features(
        features, coords, image_shape, n_rot_dims=features.shape[1], mode=mode
    )


@pytest.mark.parametrize(
    ("n_rot_dims", "skip_first"), [(64, 0), (32, 0), (64, 2), (32, 4)]
)
def test_y_bit_identical_to_legacy(inputs, n_rot_dims, skip_first):
    features, coords = inputs
    expected = _legacy_rotate_features(
        features, coords, IMAGE_SHAPE, n_rot_dims=n_rot_dims, skip_first=skip_first
    )
    out = rotate_features(
        features,
        coords,
        IMAGE_SHAPE,
        n_rot_dims=n_rot_dims,
        skip_first=skip_first,
        mode=DisambiguationMode.Y,
    )
    assert torch.equal(out, expected)


def test_default_mode_is_y_and_matches_legacy(inputs):
    features, coords = inputs
    out = rotate_features(features, coords, IMAGE_SHAPE, n_rot_dims=64)
    assert torch.equal(out, _legacy_rotate_features(features, coords, IMAGE_SHAPE, 64))


def test_none_n_rot_dims_raises_like_legacy(inputs):
    features, coords = inputs
    with pytest.raises(TypeError):
        _legacy_rotate_features(features, coords, IMAGE_SHAPE)
    with pytest.raises(TypeError):
        rotate_features(features, coords, IMAGE_SHAPE)


def test_ctcdata_wrapper_delegates(inputs):
    data = pytest.importorskip("trackastra.data.data")
    features, coords = inputs
    out = data.CTCData.rotate_features(features, coords, IMAGE_SHAPE, n_rot_dims=64)
    assert torch.equal(out, _legacy_rotate_features(features, coords, IMAGE_SHAPE, 64))


def test_x_is_y_with_swapped_axes(inputs):
    features, coords = inputs
    swapped_coords = coords[:, [0, 2, 1]]
    swapped_shape = (IMAGE_SHAPE[0], IMAGE_SHAPE[2], IMAGE_SHAPE[1])
    assert torch.allclose(
        rot(features, coords, "x"), rot(features, swapped_coords, "y", swapped_shape)
    )


def test_xy_alternates_pairs(inputs):
    features, coords = inputs
    n = len(features)
    by_pair = {m: rot(features, coords, m).view(n, -1, 2) for m in ("x", "y", "xy")}
    assert torch.equal(by_pair["xy"][:, 0::2], by_pair["y"][:, 0::2])
    assert torch.equal(by_pair["xy"][:, 1::2], by_pair["x"][:, 1::2])


def test_xy_odd_number_of_pairs(inputs):
    features, coords = inputs
    features = features[:, :6].contiguous()  # 3 pairs: y, x, y
    out = rot(features, coords, "xy").view(-1, 3, 2)
    y = rot(features, coords, "y").view(-1, 3, 2)
    x = rot(features, coords, "x").view(-1, 3, 2)
    assert torch.equal(out[:, [0, 2]], y[:, [0, 2]])
    assert torch.equal(out[:, 1], x[:, 1])


@pytest.mark.parametrize("mode", list(DisambiguationMode))
def test_modes_preserve_pair_norms_and_change_features(inputs, mode):
    features, coords = inputs
    out = rot(features, coords, mode)
    pair_norm = lambda f: f.view(len(f), -1, 2).norm(dim=-1)  # noqa: E731
    assert torch.allclose(pair_norm(out), pair_norm(features), atol=1e-5)
    assert not torch.allclose(out, features)


def test_modes_differ(inputs):
    features, coords = inputs
    outs = [rot(features, coords, m) for m in ("x", "y", "xy")]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not torch.allclose(outs[i], outs[j])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, DisambiguationMode.Y),
        (False, None),
        (None, None),
        ("x", DisambiguationMode.X),
        ("XY", DisambiguationMode.XY),
        (DisambiguationMode.X, DisambiguationMode.X),
    ],
)
def test_coerce(value, expected):
    assert coerce_disambiguation_mode(value) is expected


def test_coerce_rejects_unknown():
    with pytest.raises(ValueError):
        coerce_disambiguation_mode("z")


def test_rotate_window_features_prepends_time(inputs):
    features, coords = inputs
    window = dict(
        coords=coords[:, 1:].clone(),
        timepoints=coords[:, 0].long(),
        pretrained_features=features.clone(),
    )
    (out,) = rotate_window_features([window], IMAGE_SHAPE[-2:], "xy")
    assert torch.allclose(out["pretrained_features"], rot(features, coords, "xy"))


def test_rotate_window_features_raises_without_features():
    window = dict(
        coords=torch.zeros(3, 2),
        timepoints=torch.zeros(3, dtype=torch.long),
        pretrained_features=None,
    )
    with pytest.raises(KeyError):
        rotate_window_features([window], (100, 100), "y")


def test_rotate_window_features_wrong_key_raises(inputs):
    features, coords = inputs
    window = dict(coords=coords[:, 1:], timepoints=coords[:, 0].long(), pretrained_feats=features)
    with pytest.raises(KeyError):
        rotate_window_features([window], IMAGE_SHAPE[-2:], "y")  # default key


def test_rotate_window_features_custom_key(inputs):
    features, coords = inputs
    window = dict(
        coords=coords[:, 1:].clone(),
        timepoints=coords[:, 0].long(),
        pretrained_feats=features.clone(),
    )
    (out,) = rotate_window_features([window], IMAGE_SHAPE[-2:], "y", key="pretrained_feats")
    assert "pretrained_features" not in out
    assert torch.allclose(out["pretrained_feats"], rot(features, coords, "y"))
